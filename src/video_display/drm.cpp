
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>
#include <xf86drm.h>
#include <xf86drmMode.h>
#include <drm_fourcc.h>

#include <memory>
#include <vector>
#include <condition_variable>
#include <queue>
#include <algorithm>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "pixfmt_conv.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/text.h"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"

#define MOD_NAME "[drm] "

namespace{
        struct frame_deleter{ void operator()(video_frame *f){ vf_free(f); } };
        using frame_uniq = std::unique_ptr<video_frame, frame_deleter>;

        struct drm_connector_deleter{ void operator()(drmModeConnectorPtr c) { drmModeFreeConnector(c); } };
        using Drm_connector_uniq = std::unique_ptr<drmModeConnector, drm_connector_deleter>;

        struct drm_resources_deleter{ void operator()(drmModeResPtr r) { drmModeFreeResources(r); } };
        using Drm_res_uniq = std::unique_ptr<drmModeRes, drm_resources_deleter>;

        struct drm_encoder_deleter{ void operator()(drmModeEncoderPtr e) { drmModeFreeEncoder(e); } };
        using Drm_encoder_uniq = std::unique_ptr<drmModeEncoder, drm_encoder_deleter>;

        struct drm_crtc_deleter{ void operator()(drmModeCrtcPtr c) { drmModeFreeCrtc(c); } };
        using Drm_crtc_uniq = std::unique_ptr<drmModeCrtc, drm_crtc_deleter>;

        class Fd_uniq{
        public:
                Fd_uniq() = default;
                Fd_uniq(int fd) : fd(fd) { }
                Fd_uniq(const Fd_uniq&) = delete;

                ~Fd_uniq(){
                        destruct();
                }

                Fd_uniq& operator=(const Fd_uniq&) = delete;

                int get() { return fd; }
                void reset(int fd){
                        destruct();
                        this->fd = fd;
                }
        private:
                void destruct(){
                        if(fd > 0)
                                close(fd);

                        fd = -1;
                }
                int fd = -1;
        };

        class MemoryMapping{
        public:
                MemoryMapping() = default;
                ~MemoryMapping() {
                        if(!valid())
                                return;

                        munmap(ptr, mapped_size);
                }

                MemoryMapping(const MemoryMapping&) = delete;
                MemoryMapping& operator=(const MemoryMapping&) = delete;

                MemoryMapping(MemoryMapping&& o) noexcept{
                        std::swap(ptr, o.ptr);
                        std::swap(mapped_size, o.mapped_size);
                }

                MemoryMapping& operator=(MemoryMapping&& o) noexcept{
                        std::swap(ptr, o.ptr);
                        std::swap(mapped_size, o.mapped_size);

                        return *this;
                }

                static MemoryMapping create(void *addr, size_t len, int prot, int flags, int fildes, off_t off){
                        MemoryMapping res;
                        res.ptr = mmap(addr, len, prot, flags, fildes, off);
                        if(res.ptr != MAP_FAILED)
                                res.mapped_size = len;

                        return res;
                };

                bool valid() const { return ptr != MAP_FAILED; }
                void *get() const { return ptr; }
                size_t size() const { return mapped_size; }
        private:
                void *ptr = MAP_FAILED;
                size_t mapped_size = 0;
        };

        template<typename T, class Deleter>
        class Uniq_wrapper{
        public:
                Uniq_wrapper() = default;
                Uniq_wrapper(T val): val(val) {  }
                ~Uniq_wrapper(){
                        Deleter()(val);
                }

                Uniq_wrapper(const Uniq_wrapper&) = delete;
                Uniq_wrapper& operator=(const Uniq_wrapper&) = delete;

                Uniq_wrapper(Uniq_wrapper&& o){
                        std::swap(val, o.val);
                }

                Uniq_wrapper& operator=(Uniq_wrapper&& o){
                        std::swap(val, o.val);
                        return *this;
                }

                T& get() { return val; }
        private:
                T val = {};
                
        };

        struct Fb_handle{
                int dri_fd = -1;
                uint32_t handle;
        };

        struct Fb_handle_deleter {
                void operator()(Fb_handle h){
                        if(h.dri_fd < 0)
                                return;

                        drmModeDestroyDumbBuffer(h.dri_fd, h.handle);
                }
        };
        using Fb_handle_uniq = Uniq_wrapper<Fb_handle, Fb_handle_deleter>;

        struct Fb_id{
                int dri_fd = -1;
                uint32_t id;
        };

        struct Fb_id_deleter {
                void operator()(Fb_id i){
                        if(i.dri_fd < 0)
                                return;

                        drmModeRmFB(i.dri_fd, i.id);
                }
        };
        using Fb_id_uniq = Uniq_wrapper<Fb_id, Fb_id_deleter>;

} //anon namespace
  //

struct Drm_state {
        Fd_uniq dri_fd;

        Drm_res_uniq res;
        Drm_connector_uniq connector;
        Drm_encoder_uniq encoder;
        Drm_crtc_uniq crtc;

        drmModeModeInfoPtr mode_info;
};

struct Framebuffer{
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t pitch = 0;
        size_t size = 0;

        Fb_handle_uniq handle;

        uint32_t pix_fmt = 0;

        Fb_id_uniq id;
        MemoryMapping map;
};

struct drm_display_state {
        Drm_state drm;

        Framebuffer splashscreen;

        Framebuffer back_buffer;
        Framebuffer front_buffer;

        video_desc desc;
        frame_uniq frame;

        std::mutex mut;
        std::condition_variable frame_consumed_cv;
        std::queue<frame_uniq> queue;
        std::vector<frame_uniq> free_frames;
};

static bool init_drm_state(drm_display_state *s){
        s->drm.dri_fd.reset(open("/dev/dri/card1", O_RDWR));

        int dri = s->drm.dri_fd.get();

        if(dri < 0){
                log_msg(LOG_LEVEL_ERROR, "Failed to open DRI device\n");
                return false;
        }

        drmVersionPtr version = drmGetVersion(dri);
        if(version){
                log_msg(LOG_LEVEL_INFO, "DRM version: %d.%d.%d (%s), Driver: %s\n",
                                version->version_major,
                                version->version_minor,
                                version->version_patchlevel,
                                version->date,
                                version->name);
                drmFreeVersion(version);
        }


        s->drm.res.reset(drmModeGetResources(dri));
        if(!s->drm.res){
                log_msg(LOG_LEVEL_ERROR, "Failed to get DRI resources\n");
                return false;
        }

        for(int i = 0; i < s->drm.res->count_connectors; i++){
                s->drm.connector.reset(drmModeGetConnectorCurrent(dri, s->drm.res->connectors[i]));

                log_msg(LOG_LEVEL_INFO, "%d-%u\n", s->drm.connector->connector_type, s->drm.connector->connector_type_id);

                for(int i = 0; i < s->drm.connector->count_modes; i++){
                        if(s->drm.connector->modes[i].type & DRM_MODE_TYPE_PREFERRED){
                                s->drm.mode_info = &s->drm.connector->modes[i];
                                break;
                        }
                }

                if(s->drm.mode_info){
                        log_msg(LOG_LEVEL_INFO, "\tPreferred mode: %dx%d (%s)\n", s->drm.mode_info->hdisplay, s->drm.mode_info->vdisplay, s->drm.mode_info->name);
                        break;
                } else {
                        log_msg(LOG_LEVEL_INFO, "\tNo preferred mode\n");
                }
        }

        s->drm.encoder.reset(drmModeGetEncoder(dri, s->drm.connector->encoder_id));
        if(!s->drm.encoder){
                log_msg(LOG_LEVEL_ERROR, "Failed to get encoder\n");
                return false;
        }

        s->drm.crtc.reset(drmModeGetCrtc(dri, s->drm.encoder->crtc_id));
        if(!s->drm.crtc){
                log_msg(LOG_LEVEL_ERROR, "Failed to get crtc\n");
                return false;
        }

        int res = drmSetMaster(dri);
        if(res != 0){
                log_msg(LOG_LEVEL_ERROR, "Unable to get DRM master. Is X11 or Wayland running?\n");
                return false;
        }

        return true;
}

static Framebuffer create_dumb_fb(int dri, int width, int height, uint32_t pix_fmt){
        Framebuffer buf;

        //TODO: Currently pix_fmt is assumed to be single plane and 32 bpp

        int res = 0;

        {
                Fb_handle handle;
                res = drmModeCreateDumbBuffer(dri, width, height, 32, 0, &handle.handle, &buf.pitch, &buf.size);
                if(res != 0){
                        log_msg(LOG_LEVEL_ERROR, "Failed to create a dumb framebuffer (%d)\n", res);
                        return {};
                }
                handle.dri_fd = dri;
                buf.handle = Fb_handle_uniq(handle);
        }
        
        {
                uint32_t handles[] = {buf.handle.get().handle, 0, 0, 0};
                uint32_t pitches[] = {buf.pitch, 0, 0, 0};
                uint32_t offsets[] = {0, 0, 0, 0};
                Fb_id fb_id;
                res = drmModeAddFB2(dri, width, height,
                                pix_fmt, handles, pitches, offsets, &fb_id.id, 0);

                if(res != 0){
                        log_msg(LOG_LEVEL_ERROR, "Failed to add framebuffer (%d)\n", res);
                        return {};
                }

                fb_id.dri_fd = dri;
                buf.id = Fb_id_uniq(fb_id);
        }

        log_msg(LOG_LEVEL_INFO, "Created dumb buffer: pitch %u, handle: %u\n", buf.pitch, buf.handle.get().handle);


        struct drm_mode_map_dumb map_info = {};
        map_info.handle = buf.handle.get().handle;

        res = ioctl(dri, DRM_IOCTL_MODE_MAP_DUMB, &map_info);
        if(res != 0){
                log_msg(LOG_LEVEL_ERROR, "Failed to get map info (%d)\n", res);
                return {};
        }

        buf.map = MemoryMapping::create(0, buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, dri, map_info.offset);

        if(!buf.map.valid()){
                log_msg(LOG_LEVEL_ERROR, "Failed to map buffer\n");
                return {};
        }

        return buf;
}

static void unset_framebuffer(drm_display_state *s){
        int res = 0;
        res = drmModeSetCrtc(s->drm.dri_fd.get(), s->drm.crtc->crtc_id, 0, 0, 0, NULL, 0, NULL);
        if(res < 0){
                log_msg(LOG_LEVEL_ERROR, "Failed to set crtc (%d)\n", res);
        }
}

static void draw_splash(drm_display_state *s){
        int res = 0;
        res = drmModeSetCrtc(s->drm.dri_fd.get(), s->drm.crtc->crtc_id,
                        s->splashscreen.id.get().id, 0, 0, &s->drm.connector->connector_id, 1, s->drm.mode_info);
        if(res < 0){
                log_msg(LOG_LEVEL_ERROR, "Failed to set crtc (%d)\n", res);
        }
}

static void draw_frame(Framebuffer *dst, video_frame *src, int x = 0, int y = 0){
        auto dst_p = static_cast<char *>(dst->map.get());
        auto src_p = static_cast<char *>(src->tiles[0].data);
        auto linesize = vc_get_linesize(src->tiles[0].width, src->color_spec);

        dst_p += dst->pitch * y;
        dst_p += x * 4; //TODO

        for(unsigned y = 0; y < src->tiles[0].height; y++){
                memcpy(dst_p, src_p, linesize);
                dst_p += dst->pitch;
                src_p += linesize;
        }
}

static Framebuffer get_splash_fb(drm_display_state *s, int width, int height){
        auto splash_frame = get_splashscreen();
        int w = splash_frame->tiles[0].width;
        int h = splash_frame->tiles[0].height;

        int x = std::max(0, (width - w) / 2);
        int y = std::max(0, (height - h) / 2);

        auto fb = create_dumb_fb(s->drm.dri_fd.get(), width, height, DRM_FORMAT_XBGR8888);
        draw_frame(&fb, splash_frame, x, y);

        return fb;
}


static void *display_drm_init(struct module *parent, const char *cfg, unsigned int flags)
{
        UNUSED(parent), UNUSED(flags);

        auto s = std::make_unique<drm_display_state>();

        if(!init_drm_state(s.get())){
                return nullptr;
        }

        s->splashscreen = get_splash_fb(s.get(), s->drm.mode_info->hdisplay, s->drm.mode_info->vdisplay);

        draw_splash(s.get());

        return s.release();
}

static void display_drm_done(void *state)
{
        auto s = std::unique_ptr<drm_display_state>(static_cast<drm_display_state *>(state));

        int res = 0;
        res = drmModeSetCrtc(s->drm.dri_fd.get(), s->drm.crtc->crtc_id, s->drm.crtc->buffer_id,
                       s->drm.crtc->x , s->drm.crtc->y, &s->drm.connector->connector_id, 1, &s->drm.crtc->mode);
        if(res < 0){
                log_msg(LOG_LEVEL_ERROR, "Failed to restore original crtc (%d)\n", res);
        }
}

static struct video_frame *display_drm_getf(void *state)
{
        auto s = static_cast<drm_display_state *>(state);

        return s->frame.get();
}

static bool swap_buffers(drm_display_state *s){
        std::swap(s->front_buffer, s->back_buffer);

        int res = 0;
        res = drmModeSetCrtc(s->drm.dri_fd.get(), s->drm.crtc->crtc_id,
                        s->front_buffer.id.get().id, 0, 0, &s->drm.connector->connector_id, 1, s->drm.mode_info);
        if(res < 0){
                log_msg(LOG_LEVEL_ERROR, "Failed to set crtc (%d)\n", res);
                return false;
        }

        return true;
}

static bool display_drm_putf(void *state, struct video_frame *frame, long long flags)
{
        if (flags == PUTF_DISCARD || frame == NULL) {
                return true;
        }

        auto s = static_cast<drm_display_state *>(state);

        draw_frame(&s->back_buffer, frame);
        swap_buffers(s);

        return true;
}

static bool display_drm_get_property(void *state, int property, void *val, size_t *len)
{
        auto s = static_cast<drm_display_state *>(state);

        codec_t codecs[] = {RGBA, UYVY};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(*len < sizeof(codecs)) {
                                return false;
                        }
                        *len = sizeof(codecs);
                        memcpy(val, codecs, *len);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return false;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                default:
                        return false;
        }
        return true;
}

static bool display_drm_reconfigure(void *state, struct video_desc desc)
{
        auto s = static_cast<drm_display_state *>(state);

        s->desc = desc;

        s->frame.reset(vf_alloc_desc_data(desc));

        uint32_t pix_fmt;

        switch(desc.color_spec){
        case RGBA:
                pix_fmt = DRM_FORMAT_XBGR8888;
                break;
        case UYVY:
                pix_fmt = DRM_FORMAT_UYVY;
                break;
        default:
                return false;
        }

        //TODO: check if selected pix_fmt is supported

        s->front_buffer = create_dumb_fb(s->drm.dri_fd.get(), s->drm.mode_info->hdisplay, s->drm.mode_info->vdisplay, pix_fmt);
        s->back_buffer = create_dumb_fb(s->drm.dri_fd.get(), s->drm.mode_info->hdisplay, s->drm.mode_info->vdisplay, pix_fmt);


        return true;
}

static void display_drm_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = NULL;
        *count = 0;
}

static const struct video_display_info display_drm_info = {
        display_drm_probe,
        display_drm_init,
        NULL, // _run
        display_drm_done,
        display_drm_getf,
        display_drm_putf,
        display_drm_reconfigure,
        display_drm_get_property,
        NULL, // _put_audio_frame
        NULL, // _reconfigure_audio
        MOD_NAME,
};

REGISTER_MODULE(drm, &display_drm_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

