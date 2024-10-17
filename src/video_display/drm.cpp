/**
 * @file   video_display/drm.cpp
 * @author Martin Piatka <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2024 CESNET z.s.p.o.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
#include <algorithm>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "pixfmt_conv.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/text.h"
#include "utils/string_view_utils.hpp"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"
#include "hwaccel_drm.h"

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

        struct drm_plane_res_deleter{ void operator()(drmModePlaneResPtr r) { drmModeFreePlaneResources(r); } };
        using Drm_plane_res_uniq = std::unique_ptr<drmModePlaneRes, drm_plane_res_deleter>;

        struct drm_plane_deleter{ void operator()(drmModePlanePtr p) { drmModeFreePlane(p); } };
        using Drm_plane_uniq = std::unique_ptr<drmModePlane, drm_plane_deleter>;

        struct drm_obj_props_deleter{ void operator()(drmModeObjectPropertiesPtr p) { drmModeFreeObjectProperties(p); } };
        using Drm_object_properties_uniq = std::unique_ptr<drmModeObjectProperties, drm_obj_props_deleter>;

        struct drm_prop_deleter{ void operator()(drmModePropertyPtr p) { drmModeFreeProperty(p); } };
        using Drm_property_uniq = std::unique_ptr<drmModePropertyRes, drm_prop_deleter>;

        class Fd_uniq{
        public:
                Fd_uniq() = default;
                Fd_uniq(int fd) : fd(fd) { }
                Fd_uniq(const Fd_uniq&) = delete;
                Fd_uniq(Fd_uniq&& o){
                        std::swap(fd, o.fd);
                }

                ~Fd_uniq(){
                        destruct();
                }

                Fd_uniq& operator=(const Fd_uniq&) = delete;
                Fd_uniq& operator=(Fd_uniq&& o){
                        std::swap(fd, o.fd);
                        return *this;
                }

                operator bool() const { return fd > 0; }

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

                        drm_mode_destroy_dumb destroy_info = {};
                        destroy_info.handle = h.handle;
                        drmIoctl(h.dri_fd, DRM_IOCTL_MODE_DESTROY_DUMB, &destroy_info);
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

        class Gem_handle_manager{
                /* The handles returned from drmPrimeFDToHandle() must not be
                 * double free'd. Two frames can apparently also point to the
                 * same prime buffer, so we really need to reference count it
                 * ourselves.
                 */
        private:
                void add_ref(uint32_t handle){
                        handle_map[handle]++;
                };

                void unref(uint32_t handle){
                        int refs = handle_map[handle]--;
                        assert(refs >= 0 && "Unref called on invalid handle");

                        if(refs == 0){
                                drmIoctl(dri_fd, DRM_IOCTL_GEM_CLOSE, &handle);
                                handle_map.erase(handle);
                        }
                }
                int dri_fd = -1;
                std::map<uint32_t, int> handle_map;

        public:
                Gem_handle_manager(int dri_fd): dri_fd(dri_fd) { }

                class Handle{
                public:
                        Handle() = default;
                        ~Handle(){
                                if(ctx)
                                        ctx->unref(handle);
                        }

                        Handle(const Handle&) = delete;
                        Handle& operator=(const Handle&) = delete;
                        Handle(Handle&& o){
                                std::swap(handle, o.handle);
                                std::swap(ctx, o.ctx);
                        }
                        Handle& operator=(Handle&& o){
                                std::swap(handle, o.handle);
                                std::swap(ctx, o.ctx);
                                return *this;
                        }

                        uint32_t get() const {
                                return handle;
                        }
                private:
                        friend Gem_handle_manager;
                        Handle(uint32_t handle, Gem_handle_manager *ctx): handle(handle), ctx(ctx){
                                ctx->add_ref(handle);
                        }
                        uint32_t handle = 0;
                        Gem_handle_manager *ctx = nullptr;
                };

                Handle get_handle(int fd){
                        uint32_t handle = 0;
                        int res = drmPrimeFDToHandle(dri_fd, fd, &handle);
                        if(res < 0){
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get a GEM handle from prime fd %d\n", fd);
                                return {};
                        }
                        return Handle(handle, this);
                }

        };

} //anon namespace
  //

struct Drm_state {
        Fd_uniq dri_fd;

        std::unique_ptr<Gem_handle_manager> gem_manager;

        Drm_res_uniq res;
        Drm_connector_uniq connector;
        Drm_encoder_uniq encoder;
        Drm_crtc_uniq crtc;
        int crtc_index = -1;

        std::set<uint32_t> supported_drm_formats;
        bool prime_support = false;

        drmModeModeInfoPtr mode_info;
};

struct Drm_framebuffer{
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t pitch = 0;
        size_t size = 0;

        Fb_handle_uniq handle;

        uint32_t pix_fmt = 0;

        Fb_id_uniq id;
        MemoryMapping map;
};

struct Drm_prime_fb{
        frame_uniq frame;
        Gem_handle_manager::Handle gem_objects[4] = {};
        Fb_id_uniq id;
};

struct drm_display_state {
        std::string cfg;
        std::string device_path;
        std::string req_connector;
        int req_width = -1;
        int req_height = -1;
        unsigned req_rate = 0;

        Drm_state drm;

        Drm_framebuffer splashscreen;

        Drm_framebuffer back_buffer;
        Drm_framebuffer front_buffer;

        Drm_prime_fb drm_prime_fb;

        video_desc desc;
        frame_uniq frame;

        std::vector<frame_uniq> free_frames;
};

static std::string get_connector_str(int type, uint32_t id){
        std::string res;
        switch(type){
        case DRM_MODE_CONNECTOR_VGA: res = "VGA"; break;
        case DRM_MODE_CONNECTOR_HDMIA: res = "HDMI-A"; break;
        case DRM_MODE_CONNECTOR_HDMIB: res = "HDMI-B"; break;
        case DRM_MODE_CONNECTOR_DisplayPort: res = "DP"; break;
        case DRM_MODE_CONNECTOR_eDP: res = "eDP"; break;
        case DRM_MODE_CONNECTOR_DVII: res = "DVI-I"; break;
        case DRM_MODE_CONNECTOR_DVID: res = "DVI-D"; break;
        case DRM_MODE_CONNECTOR_DVIA: res = "DVI-A"; break;
#ifdef DRM_MODE_CONNECTOR_USB
        case DRM_MODE_CONNECTOR_USB: res = "USB"; break;
#endif
        default:
                res = std::to_string(type);
        }

        res += "-";
        res += std::to_string(id);

        return res;
}

static void print_drm_driver_info(drm_display_state *s){
        drmVersionPtr version = drmGetVersion(s->drm.dri_fd.get());
        if(!version){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Failed to get DRM driver version\n");
                return;
        }

        log_msg(LOG_LEVEL_INFO, MOD_NAME "DRM version: %d.%d.%d (%s), Driver: %s\n",
                        version->version_major,
                        version->version_minor,
                        version->version_patchlevel,
                        version->date,
                        version->name);
        drmFreeVersion(version);
}

static Fd_uniq open_dri(drm_display_state *s){

        auto do_open = [](const char *path) -> Fd_uniq {
                int fd = open(path, O_RDWR);
                if(fd < 0){
                        if(errno != ENOENT)
                                log_msg(LOG_LEVEL_INFO, MOD_NAME "Failed to open %s (%s)\n", path, strerror(errno));

                        return {};
                }
                Fd_uniq ret(fd);

                Drm_res_uniq resources(drmModeGetResources(fd));
                if(!resources){
                        log_msg(LOG_LEVEL_INFO, MOD_NAME "Failed to get resources on %s (%s)\n", path, strerror(errno));
                        return {};
                }

                uint64_t dumb_support = false;
                int res = 0;
                res = drmGetCap(fd, DRM_CAP_DUMB_BUFFER, &dumb_support);
                if(res < 0 || !dumb_support){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "%s does not support dumb buffers\n", path);
                        return {};
                }

                log_msg(LOG_LEVEL_INFO, MOD_NAME "Opened %s DRI device\n", path);
                return ret;
        };

        if(!s->device_path.empty()){
                auto ret = do_open(s->device_path.c_str());
                if(!ret)
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to open specified device (%s)\n", s->device_path.c_str());

                return ret;
        }

        char buf[256] = {};
        const int max_index = 32;
        for(int i = 0; i < max_index; i++){
                snprintf(buf, sizeof(buf), DRM_DEV_NAME, DRM_DIR_NAME, i);
                auto ret = do_open(buf);
                if(ret)
                        return ret;
        }
        log_msg(LOG_LEVEL_ERROR, MOD_NAME "No suitable DRI device found\n");
        return {};
}

static int64_t get_property(int dri, drmModeObjectPropertiesPtr props, std::string_view name){
        for(unsigned i = 0; i < props->count_props; i++){
                Drm_property_uniq prop(drmModeGetProperty(dri, props->props[i]));
                if(!prop)
                        continue;

                if(prop->name == name)
                        return props->prop_values[i];
        }
        return -1;
}

static bool probe_drm_formats(drm_display_state *s){
        int dri = s->drm.dri_fd.get();
        Drm_plane_res_uniq plane_res(drmModeGetPlaneResources(dri));
        if(!plane_res){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get plane resources (%s)\n", strerror(errno));
                return false;
        }

        Drm_plane_uniq primary_plane;
        for(unsigned i = 0; i < plane_res->count_planes; i++){
                Drm_plane_uniq plane(drmModeGetPlane(dri, plane_res->planes[i]));
                if(!(plane->possible_crtcs & (1 << s->drm.crtc_index)))
                        continue;

                Drm_object_properties_uniq props(drmModeObjectGetProperties(dri, plane_res->planes[i], DRM_MODE_OBJECT_PLANE));
                if(!props){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get plane props (%s)\n", strerror(errno));
                        continue;
                }
                if(get_property(dri, props.get(), "type") == DRM_PLANE_TYPE_PRIMARY){
                        primary_plane = std::move(plane);
                        break;
                }
        }

        if(!primary_plane){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to find the primary plane\n");
                return false;
        }

        for(unsigned i = 0; i < primary_plane->count_formats; i++){
                s->drm.supported_drm_formats.insert(primary_plane->formats[i]);
        }

        return true;
}

static std::vector<Drm_connector_uniq> get_connectors(drm_display_state *s){
        std::vector<Drm_connector_uniq> res;

        for(int i = 0; i < s->drm.res->count_connectors; i++){
                Drm_connector_uniq conn(drmModeGetConnector(s->drm.dri_fd.get(), s->drm.res->connectors[i]));
                if(!conn)
                        continue;

                res.push_back(std::move(conn));
        }

        return res;
}

static bool init_drm_state(drm_display_state *s){
        s->drm.dri_fd = open_dri(s);

        int dri = s->drm.dri_fd.get();
        if(dri < 0){
                return false;
        }

        print_drm_driver_info(s);

        int res = 0;
        res = drmSetClientCap(dri, DRM_CLIENT_CAP_UNIVERSAL_PLANES, 1);
        if(res != 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to set universal planes capab\n");
                return false;
        }

        s->drm.gem_manager = std::make_unique<Gem_handle_manager>(dri);

        s->drm.res.reset(drmModeGetResources(dri));
        if(!s->drm.res){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get DRI resources: %s\n", strerror(errno));
                return false;
        }

        uint64_t prime_support = false;
        res = drmGetCap(dri, DRM_CAP_PRIME, &prime_support);
        if(res < 0 || !prime_support){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "DRM device does not support PRIME buffers\n");
        }
        s->drm.prime_support = prime_support;

        return true;
}

static const char *get_connector_status_str(drmModeConnection c){
        switch(c){
        case DRM_MODE_CONNECTED:
                return " (Connected)";
        case DRM_MODE_DISCONNECTED:
                return " (Not connected)";
        case DRM_MODE_UNKNOWNCONNECTION:
        default:
                return "";
        }
}

static void print_connectors(drm_display_state *s){
        color_printf("Connectors:\n");

        auto connectors = get_connectors(s);
        for(auto& connector : connectors){
                color_printf("\t%s%s:\n", get_connector_str(connector->connector_type, connector->connector_type_id).c_str(),
                                get_connector_status_str(connector->connection));

                for(int i = 0; i < connector->count_modes; i++){
                        auto mode_info = &connector->modes[i];
                        color_printf("\t\t%dx%d@%d (%s)%s\n", mode_info->hdisplay, mode_info->vdisplay, mode_info->vrefresh, mode_info->name,
                                        mode_info->type & DRM_MODE_TYPE_PREFERRED ? " preferred" : "");
                }
        }
}

static bool setup_crtc(drm_display_state *s){
        auto connectors = get_connectors(s);

        for(auto& connector : connectors){
                if(!s->req_connector.empty()){
                        if(s->req_connector == get_connector_str(connector->connector_type, connector->connector_type_id)){
                                s->drm.connector = std::move(connector);
                                break;
                        } else {
                                continue;
                        }
                }

                if(connector->count_modes > 0 && connector->connection == DRM_MODE_CONNECTED){
                        s->drm.connector = std::move(connector);
                        break;
                }
        }

        if(!s->drm.connector){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Connector not found\n");
                return false;
        }

        bool find_preferred_mode = s->req_width <= 0 && s->req_height <= 0 && s->req_rate <= 0;
        for(int i = 0; i < s->drm.connector->count_modes; i++){
                auto mode = &s->drm.connector->modes[i];

                if(s->req_width > 0 && mode->hdisplay != s->req_width)
                        continue;

                if(s->req_height > 0 && mode->vdisplay != s->req_height)
                        continue;

                if(s->req_rate > 0 && mode->vrefresh != s->req_rate)
                        continue;

                if(find_preferred_mode && !(mode->type & DRM_MODE_TYPE_PREFERRED)){
                        continue;
                }

                s->drm.mode_info = mode;
                break;
        }

        if(!s->drm.mode_info){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Mode not found\n");
                return false;
        }

        int dri = s->drm.dri_fd.get();

        std::vector<int> encoder_list;
        if(s->drm.connector->encoder_id){
                encoder_list.push_back(s->drm.connector->encoder_id);
        }
        for(int i = 0; i < s->drm.connector->count_encoders; i++){
                encoder_list.push_back(s->drm.connector->encoders[i]);
        }

        for(int encoder_id : encoder_list){
                s->drm.encoder.reset(drmModeGetEncoder(dri, encoder_id));
                if(!s->drm.encoder)
                        continue;

                std::vector<unsigned> crtc_list;
                if(s->drm.encoder->crtc_id){
                        crtc_list.push_back(s->drm.encoder->crtc_id);
                }
                for(int i = 0; i < s->drm.res->count_crtcs; i++){
                        if((1 << i) & s->drm.encoder->possible_crtcs){
                                crtc_list.push_back(s->drm.res->crtcs[i]);
                        }
                }

                for(unsigned crtc_id : crtc_list){
                        s->drm.crtc.reset(drmModeGetCrtc(dri, crtc_id));
                        if(s->drm.crtc){
                                for(int i = 0; i < s->drm.res->count_crtcs; i++){
                                        if(s->drm.res->crtcs[i] == crtc_id){
                                                s->drm.crtc_index = i;
                                                break;
                                        }
                                }
                                break;
                        }
                }
                if(s->drm.crtc)
                        break;
        }


        if(!s->drm.crtc){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to find encoder crtc pair\n");
                return false;
        }

        if(!probe_drm_formats(s)){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to probe DRM supported formats\n");
                return false;
        }

        int res = 0;
        res = drmSetMaster(dri);
        if(res != 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to get DRM master. Is X11 or Wayland running?\n");
                return false;
        }

        return true;
}

static Drm_framebuffer create_dumb_fb(int dri, int width, int height, uint32_t pix_fmt){
        Drm_framebuffer buf;

        //TODO: Currently pix_fmt is assumed to be single plane and 32 bpp

        int res = 0;

        {
                Fb_handle handle;
                drm_mode_create_dumb create_info = {};
                create_info.width = width;
                create_info.height = height;
                create_info.bpp = 32;
                res = drmIoctl(dri, DRM_IOCTL_MODE_CREATE_DUMB, &create_info);
                if(res != 0){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create a dumb framebuffer (%d)\n", res);
                        return {};
                }
                handle.handle = create_info.handle;
                buf.width = width;
                buf.height = height;
                buf.pitch = create_info.pitch;
                buf.size = create_info.size;
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
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to add framebuffer (%d)\n", res);
                        return {};
                }

                fb_id.dri_fd = dri;
                buf.id = Fb_id_uniq(fb_id);
        }

        log_msg(LOG_LEVEL_INFO, MOD_NAME "Created dumb buffer: pitch %u, handle: %u\n", buf.pitch, buf.handle.get().handle);


        struct drm_mode_map_dumb map_info = {};
        map_info.handle = buf.handle.get().handle;

        res = drmIoctl(dri, DRM_IOCTL_MODE_MAP_DUMB, &map_info);
        if(res != 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get map info (%d)\n", res);
                return {};
        }

        buf.map = MemoryMapping::create(0, buf.size, PROT_READ | PROT_WRITE, MAP_SHARED, dri, map_info.offset);
        if(!buf.map.valid()){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to map buffer\n");
                return {};
        }

        return buf;
}

static bool set_framebuffer(drm_display_state *s, uint32_t fb_id){
        int res = 0;
        res = drmModeSetCrtc(s->drm.dri_fd.get(), s->drm.crtc->crtc_id,
                        fb_id, 0, 0, &s->drm.connector->connector_id, 1, s->drm.mode_info);
        if(res < 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to set crtc (%d)\n", res);
                return false;
        }
        return true;
}

static void draw_splash(drm_display_state *s){
        int res = 0;
        res = drmModeSetCrtc(s->drm.dri_fd.get(), s->drm.crtc->crtc_id,
                        s->splashscreen.id.get().id, 0, 0, &s->drm.connector->connector_id, 1, s->drm.mode_info);
        if(res < 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to set crtc (%d)\n", res);
        }
}

static void draw_frame(Drm_framebuffer *dst, video_frame *src, bool center = true){
        auto dst_p = static_cast<char *>(dst->map.get());
        auto src_p = static_cast<char *>(src->tiles[0].data);

        auto src_w = src->tiles[0].width;
        auto src_h = src->tiles[0].height;

        unsigned x = 0;
        unsigned y = 0;

        if(center){
                if(src_w < dst->width)
                        x = (dst->width - src_w) / 2;
                if(src_h < dst->height)
                        y = (dst->height - src_h) / 2;
        }

        int width = std::min(src_w, dst->width - x);
        int height = std::min(src_h, dst->height - y);

        auto src_pitch = vc_get_linesize(src_w, src->color_spec);
        auto linesize = vc_get_size(width, src->color_spec);

        dst_p += dst->pitch * y;
        dst_p += vc_get_size(x, src->color_spec);

        for(int y = 0; y < height; y++){
                memcpy(dst_p, src_p, linesize);
                dst_p += dst->pitch;
                src_p += src_pitch;
        }
}

static bool drm_format_supported(drm_display_state *s, uint32_t fmt){
        return s->drm.supported_drm_formats.find(fmt) != s->drm.supported_drm_formats.end();
}

static Drm_framebuffer get_splash_fb(drm_display_state *s, int width, int height){
        frame_uniq splash_frame(get_splashscreen());

        uint32_t pix_fmt;
        if(drm_format_supported(s, DRM_FORMAT_XBGR8888)){
                pix_fmt = DRM_FORMAT_XBGR8888;
        } else {
                //XRGB_8888 should be always present
                //red and blue will be swapped, but for splash it doesn't matter much
                pix_fmt = DRM_FORMAT_XRGB8888;
        }
        auto fb = create_dumb_fb(s->drm.dri_fd.get(), width, height, pix_fmt);
        draw_frame(&fb, splash_frame.get(), true);

        return fb;
}

static void *display_drm_init(struct module *parent, const char *cfg, unsigned int flags)
{
        UNUSED(parent), UNUSED(flags);

        auto s = std::make_unique<drm_display_state>();

        if(cfg)
                s->cfg = cfg;

        bool help_requested = false;
        std::string_view sv_cfg(s->cfg);
        while(!sv_cfg.empty()){
                auto token = tokenize(sv_cfg, ':');
                auto key = tokenize(token, '=');
                auto val = tokenize(token, '=');

                if(key == "help"){
                        help_requested = true;
                } else if(key == "dev"){
                        s->device_path = val;
                } else if(key == "connector"){
                        s->req_connector = val;
                } else if(key == "mode"){
                        auto res = tokenize(val, '@');
                        auto rate = tokenize(val, '@');

                        auto width = tokenize(res, 'x');
                        auto height = tokenize(res, 'x');

                        parse_num(width, s->req_width);
                        parse_num(height, s->req_height);
                        parse_num(rate, s->req_rate);
                }
        }

        if(!init_drm_state(s.get())){
                return nullptr;
        }

        if(help_requested){
                color_printf("\n");
                color_printf("DRM display\n");
                color_printf(TBOLD(TRED("\t-t drm"))"[:dev=<path>][:connector=<c>][:mode=<w>x<h>[@<rate>]]\n");
                color_printf("where:\n");
                color_printf(TBOLD("\tpath") "        - Path to the DRI device. If not specified /dev/video[0..32] are tried.\n");
                color_printf(TBOLD("\tconnector") "   - The physical connector the display device is plugged into.\n");
                color_printf(TBOLD("\tmode") "        - Video mode to use. If not specified, the preferred mode is used. \n");
                color_printf("\n");
                print_connectors(s.get());
                return INIT_NOERR;
        }

        if(!setup_crtc(s.get())){
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
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to restore original crtc (%d)\n", res);
        }
}

static struct video_frame *display_drm_getf(void *state)
{
        auto s = static_cast<drm_display_state *>(state);

        while(!s->free_frames.empty()){
                auto frame = std::move(s->free_frames.back());
                s->free_frames.pop_back();

                if(video_desc_eq(video_desc_from_frame(frame.get()), s->desc)){
                        return frame.release();
                }
        }

        return vf_alloc_desc_data(s->desc);
}

static bool swap_buffers(drm_display_state *s){
        std::swap(s->front_buffer, s->back_buffer);
        return set_framebuffer(s, s->front_buffer.id.get().id);
}

static Drm_prime_fb drm_fb_from_frame(drm_display_state *s, frame_uniq frame){
        assert(frame->color_spec == DRM_PRIME);

        Drm_prime_fb fb;
        fb.frame = std::move(frame);
        auto drm_frame = (drm_prime_frame *) fb.frame->tiles[0].data;

        for(int i = 0; i < drm_frame->fd_count; i++){
                fb.gem_objects[i] = s->drm.gem_manager->get_handle(drm_frame->dmabuf_fds[i]);
        }

        uint32_t handles[4] = {};
        for(int i = 0; i < drm_frame->planes; i++){
                handles[i] = fb.gem_objects[drm_frame->fd_indices[i]].get();
        }

        int res = 0;
        Fb_id fb_id;
        res = drmModeAddFB2WithModifiers(s->drm.dri_fd.get(), fb.frame->tiles[0].width, fb.frame->tiles[0].height, drm_frame->drm_format,
                        handles, drm_frame->pitches, drm_frame->offsets, drm_frame->modifiers, &fb_id.id, DRM_MODE_FB_MODIFIERS);
        if(res != 0){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to add FB\n");
        }
        fb_id.dri_fd = s->drm.dri_fd.get();
        fb.id = Fb_id_uniq(fb_id);

        return fb;
}

static void recycle_frame(drm_display_state *s, frame_uniq& frame){
        vf_recycle(frame.get());
        s->free_frames.push_back(std::move(frame));
}

static void recycle_prime_frame(drm_display_state *s, Drm_prime_fb& fb){
        frame_uniq frame = std::move(fb.frame);
        fb = {};
        if(frame){
                vf_recycle(frame.get());
                s->free_frames.push_back(std::move(frame));
        }
}

static bool display_drm_putf(void *state, struct video_frame *f, long long flags)
{
        frame_uniq frame(f);

        if (!frame) {
                return true;
        }

        auto s = static_cast<drm_display_state *>(state);

        if(flags == PUTF_DISCARD){
                recycle_frame(s, frame);
                return true;
        }

        if(frame->color_spec == DRM_PRIME){
                Drm_prime_fb fb = drm_fb_from_frame(s, std::move(frame));
                if(!set_framebuffer(s, fb.id.get().id)){
                        return false;
                }

                recycle_prime_frame(s, s->drm_prime_fb);
                s->drm_prime_fb = std::move(fb);
                return true;
        }

        draw_frame(&s->back_buffer, frame.get());
        recycle_frame(s, frame);
        swap_buffers(s);

        return true;
}

static bool get_codecs(drm_display_state *s, void *val, size_t *len){
        std::vector<codec_t> out;

        if(s->drm.prime_support)
                out.push_back(DRM_PRIME);

        if(s->drm.supported_drm_formats.find(DRM_FORMAT_UYVY) != s->drm.supported_drm_formats.end())
                out.push_back(UYVY);

        out.push_back(RGBA);

        size_t length = sizeof(codec_t) * out.size();

        if(*len < length) {
                return false;
        }
        *len = length;
        memcpy(val, out.data(), *len);

        return true;

}

static bool display_drm_get_property(void *state, int property, void *val, size_t *len)
{
        auto s = static_cast<drm_display_state *>(state);

        int rgb_shift[] = {0, 8, 16};
        if(!drm_format_supported(s, DRM_FORMAT_XBGR8888)){
                std::swap(rgb_shift[0], rgb_shift[2]);
        }

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        return get_codecs(s, val, len);
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
                pix_fmt = drm_format_supported(s, DRM_FORMAT_XBGR8888) ? DRM_FORMAT_XBGR8888 : DRM_FORMAT_XRGB8888;
                break;
        case UYVY:
                pix_fmt = DRM_FORMAT_UYVY;
                break;
        case DRM_PRIME:
                /* We don't create dumb buffers in this case, we import framebuffers from video frames instead
                 */
                pix_fmt = 0; //UNUSED
                if(!s->drm.mode_info || s->drm.mode_info->hdisplay > desc.width || s->drm.mode_info->vdisplay > desc.height){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Video mode resolution must be less or equal to DRM_PRIME frames. Trying to find a suitable mode...\n");
                        s->drm.mode_info = nullptr;
                        for(int i = 0; i < s->drm.connector->count_modes; i++){
                                auto mode = &s->drm.connector->modes[i];

                                if(mode->hdisplay > desc.width)
                                        continue;

                                if(mode->vdisplay > desc.height)
                                        continue;

                                s->drm.mode_info = mode;
                                break;
                        }
                        if(!s->drm.mode_info){
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to find suitable mode\n");
                                return false;
                        }
                }
                return true;
        default:
                return false;
        }

        if(desc.width > s->drm.mode_info->hdisplay || desc.height > s->drm.mode_info->vdisplay){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Video resolution is larger than framebuffer. Only part of video frames will be visible\n");
        }

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

