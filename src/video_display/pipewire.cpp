
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <memory>
#include <atomic>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/text.h"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"

#include "pipewire_common.hpp"
#include "utils/string_view_utils.hpp"
#include <spa/param/video/format-utils.h>

#define MOD_NAME "[pw_disp] "

namespace{
        struct frame_deleter{ void operator()(video_frame *f){ vf_free(f); } };
        using unique_frame = std::unique_ptr<video_frame, frame_deleter>;

        struct memfd_buffer{
                memfd_buffer() = default;
                memfd_buffer(const memfd_buffer&) = delete;
                ~memfd_buffer();

                memfd_buffer& operator=(const memfd_buffer&) = delete;

                /* The buffer is shared by the pipewire buffer and ultragrid.
                 * Since both can be deleted independently at any time from
                 * different threads we need reference counting. 
                 */
                std::atomic<int> ref_count = 0;

                int fd = -1;
                int size = 0;
                void *ptr = nullptr;

                /* Access to these pointers is synchronized by the pipewire
                 * thread loop lock.
                 *
                 * The pw_buffer pointer is also used to signal that the
                 * pipewire buffer got removed and should not be queued by
                 * setting it to null in the on_remove_buffer callback.
                 *
                 * The video_frame pointer is used to signal the current
                 * ownership of the frame. The ownership is transferred in the
                 * getf and putf functions.
                 *
                 * If set - the frame is owned by pipewire and should be
                 * deleted in the on_remove_callback.
                 *
                 * If unset - frame is owned
                 * by ultragrid, which should delete it.
                 */
                pw_buffer *b = nullptr;
                video_frame *f = nullptr;
        };

        memfd_buffer::~memfd_buffer(){
                assert(ref_count == 0);
                if(ptr){
                        munmap(ptr, size);
                }
                if(fd > 0){
                        close(fd);
                }
        }

        memfd_buffer *memfd_buf_create(size_t size){
                auto buf = std::make_unique<memfd_buffer>();

                buf->fd = memfd_create("ultragrid-display", MFD_CLOEXEC);
                if(buf->fd < 0){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot create memfd\n");
                        return nullptr;
                }

                buf->size = size;
                if(ftruncate(buf->fd, size) < 0){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to resize memfd\n");
                        return nullptr;
                }

                buf->ptr = mmap(nullptr, size, PROT_READ|PROT_WRITE,
                                MAP_SHARED, buf->fd, 0);

                return buf.release();
        }

        void memfd_buf_unref(memfd_buffer **buffer){
                auto buf = *buffer;
                *buffer = nullptr;

                int refcount = buf->ref_count.fetch_sub(1) - 1;
                if(refcount < 1)
                        delete buf;
        }

        memfd_buffer *memfd_buf_ref(memfd_buffer *buf){
                buf->ref_count++;
                return buf;
        }

        void memfd_frame_data_deleter(video_frame *f){
                auto buf = static_cast<memfd_buffer *>(f->callbacks.dispose_udata);

                if(buf)
                        memfd_buf_unref(&buf);
        }
}

struct display_pw_state{
        pipewire_state_common pw;

        pw_stream_uniq stream;
        spa_hook_uniq stream_listener;

        std::string target;

        unique_frame dummy_frame;

        video_desc desc = {};
};

static void on_state_changed(void *state, enum pw_stream_state old, enum pw_stream_state new_state, const char *error)
{
        auto s = static_cast<display_pw_state *>(state);
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Stream state change: %s -> %s\n",
                        pw_stream_state_as_string(old),
                        pw_stream_state_as_string(new_state));

        if(error){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Stream error: %s\n", error);
        }

        pw_thread_loop_signal(s->pw.pipewire_loop.get(), false);
}

static void on_param_changed(void *state, uint32_t id, const struct spa_pod *param){
        auto s = static_cast<display_pw_state *>(state);

        if(!param || id != SPA_PARAM_Format)
                return;

        //TODO
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "on param changed\n");

	struct spa_video_info_raw format;
	spa_format_video_raw_parse(param, &format);

	int stride = vc_get_linesize(s->desc.width, s->desc.color_spec);

        log_msg(LOG_LEVEL_NOTICE, "Setting stride to %d\n", stride);

        const int MAX_BUFFERS = 16;

	const struct spa_pod *params[1];

        std::byte buffer[1024];
        auto pod_builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

	params[0] = (spa_pod *) spa_pod_builder_add_object(&pod_builder,
		SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
		SPA_PARAM_BUFFERS_buffers, SPA_POD_CHOICE_RANGE_Int(8, 2, MAX_BUFFERS),
		SPA_PARAM_BUFFERS_blocks,  SPA_POD_Int(1),
		SPA_PARAM_BUFFERS_size,    SPA_POD_Int(stride * s->desc.height),
		SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(1<<SPA_DATA_MemFd),
		SPA_PARAM_BUFFERS_stride,  SPA_POD_Int(stride));

	pw_stream_update_params(s->stream.get(), params, 1);

}

static void on_add_buffer(void *data, struct pw_buffer *buffer){
        auto s = static_cast<display_pw_state *>(data);

        int stride = vc_get_linesize(s->desc.width, s->desc.color_spec);
        int size = stride * s->desc.height;

        struct spa_data *d = buffer->buffer->datas;
        if ((d->type & (1<<SPA_DATA_MemFd)) == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Buffer doesn't support MemFd type\n");
                return;
        }

        auto buf = memfd_buf_create(size);
        buffer->user_data = memfd_buf_ref(buf);

        d->type = SPA_DATA_MemFd;
        d->flags = SPA_DATA_FLAG_READWRITE;
        d->fd = buf->fd;
        d->maxsize = buf->size;
        d->mapoffset = 0;

        auto ug_frame = vf_alloc_desc(s->desc);
        buf->f = ug_frame;
        buf->b = buffer;

        ug_frame->callbacks.data_deleter = memfd_frame_data_deleter;
        ug_frame->callbacks.dispose_udata = memfd_buf_ref(buf);
        ug_frame->tiles[0].data = (char *) buf->ptr;

        log_msg(LOG_LEVEL_NOTICE, "Buffer added\n");
}

static void on_remove_buffer(void *data, struct pw_buffer *buffer){
        auto buf = static_cast<memfd_buffer *>(buffer->user_data);

        if(buf->f)
                vf_free(buf->f);

        buf->b = nullptr;
        memfd_buf_unref(&buf);

        log_msg(LOG_LEVEL_NOTICE, "Buffer removed\n");
}

static void on_process(void *userdata) noexcept{
        auto s = static_cast<display_pw_state *>(userdata);

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "on process\n");
}


const static pw_stream_events stream_events = { 
        .version = PW_VERSION_STREAM_EVENTS,
        .destroy = nullptr,
        .state_changed = on_state_changed,
        .control_info = nullptr,
        .io_changed = nullptr,
        .param_changed = on_param_changed,
        .add_buffer = on_add_buffer,
        .remove_buffer = on_remove_buffer,
        .process = on_process,
        .drained = nullptr,
#if PW_MAJOR > 0 || PW_MINOR > 3 || (PW_MINOR == 3 && PW_MICRO > 39)
        .command = nullptr,
        .trigger_done = nullptr,
#endif
};

static void *display_pw_init(struct module *parent, const char *cfg, unsigned int flags)
{
        auto s = std::make_unique<display_pw_state>();

        std::string_view cfg_sv(cfg);
        while(!cfg_sv.empty()){
                auto tok = tokenize(cfg_sv, ':', '"');

                auto key = tokenize(tok, '=');
                auto val = tokenize(tok, '=');

                if(key == "help"){
                        return INIT_NOERR;
                } else if(key == "target"){
                        s->target = val;
                }
        }

        initialize_pw_common(s->pw);

        log_msg(LOG_LEVEL_INFO, MOD_NAME "Compiled with libpipewire %s\n", pw_get_headers_version());
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Linked with libpipewire %s\n", pw_get_library_version());


        return s.release();
}

static void display_pw_done(void *state)
{
        auto s = std::unique_ptr<display_pw_state>(static_cast<display_pw_state *>(state));

        pw_thread_loop_stop(s->pw.pipewire_loop.get());
}

static struct video_frame *display_pw_getf(void *state)
{
        auto s = static_cast<display_pw_state *>(state);

        auto get_dummy = [](display_pw_state *s){
                if (!s->dummy_frame || video_desc_eq(video_desc_from_frame(s->dummy_frame.get()), s->desc))
                {
                        s->dummy_frame.reset(vf_alloc_desc_data(s->desc));
                }
                return s->dummy_frame.get();
        };

        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        const char *error = nullptr;
        auto stream_state = pw_stream_get_state(s->stream.get(), &error);
        if(stream_state != PW_STREAM_STATE_STREAMING)
                return get_dummy(s);

        struct pw_buffer *b = pw_stream_dequeue_buffer(s->stream.get());
        if (!b) {
                log_msg(LOG_LEVEL_WARNING, "Out of buffers!\n");
                return get_dummy(s);
        }

        auto buf = static_cast<memfd_buffer *>(b->user_data);
        video_frame *f = std::exchange(buf->f, nullptr);

        b->buffer->datas[0].chunk->size = f->tiles[0].data_len;
        b->buffer->datas[0].chunk->offset = 0;
        b->buffer->datas[0].chunk->stride = vc_get_linesize(s->desc.width, s->desc.color_spec);

        return f;
}


static bool display_pw_putf(void *state, struct video_frame *frame, long long flags)
{
        auto s = static_cast<display_pw_state *>(state);

        if (flags == PUTF_DISCARD || frame == NULL || frame == s->dummy_frame.get()) {
                return true;
        }

        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        auto buf = static_cast<memfd_buffer *>(frame->callbacks.dispose_udata);

        if(!buf->b){
                //Frame is invalid - buffer got removed
                vf_free(frame);
                return true;
        }

        buf->f = frame;

        pw_stream_queue_buffer(s->stream.get(), buf->b);

        return true;
}

static bool display_pw_get_property(void *state, int property, void *val, size_t *len)
{
        auto s = static_cast<display_pw_state *>(state);

        codec_t codecs[] = {UYVY, RGB, RGBA};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        {
                                //TODO
                                *len = sizeof(codecs);
                                memcpy(val, codecs, *len);
                        }
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                default:
                        return false;
        }
        return true;
}

static bool display_pw_reconfigure(void *state, struct video_desc desc)
{
        auto s = static_cast<display_pw_state *>(state);

        s->desc = desc;

        auto props = pw_properties_new(
                        PW_KEY_MEDIA_TYPE, "Video",
                        PW_KEY_MEDIA_CATEGORY, "Source",
                        PW_KEY_MEDIA_ROLE, "Communication",
                        PW_KEY_APP_NAME, "UltraGrid",
                        PW_KEY_APP_ICON_NAME, "ultragrid",
                        PW_KEY_NODE_NAME, "ug video out",
                        STREAM_TARGET_PROPERTY_KEY, s->target.c_str(),
                        nullptr);

        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        s->stream.reset(pw_stream_new(
                                s->pw.pipewire_core.get(),
                                "UltraGrid video out",
                                props));

        pw_stream_add_listener(
                        s->stream.get(),
                        &s->stream_listener.get(),
                        &stream_events,
                        s);

	const struct spa_pod *params[1];

        //TODO: Figure out if we can pass the fps (in desc it's not fractional)
        auto framerate = SPA_FRACTION(0, 1);
        auto size = SPA_RECTANGLE(desc.width, desc.height);

        std::byte buffer[1024];
        auto pod_builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

        params[0] = static_cast<spa_pod *> (spa_pod_builder_add_object( &pod_builder,
                                SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
                                SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
                                SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
                                SPA_FORMAT_VIDEO_format, SPA_POD_Id(pw_fmt_from_uv_codec(desc.color_spec)),
                                SPA_FORMAT_VIDEO_size, SPA_POD_Rectangle(&size),
                                SPA_FORMAT_VIDEO_framerate, SPA_POD_Fraction(&framerate)));

        pw_stream_connect(s->stream.get(),
                        PW_DIRECTION_OUTPUT,
                        PW_ID_ANY,
                        static_cast<pw_stream_flags>(
                                PW_STREAM_FLAG_AUTOCONNECT |
                                PW_STREAM_FLAG_DRIVER |
                                PW_STREAM_FLAG_ALLOC_BUFFERS),
                        &params[0], 1);

        return true;
}

static void display_pw_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = NULL;
        *count = 0;
}

static const struct video_display_info display_pw_info = {
        display_pw_probe,
        display_pw_init,
        NULL, // _run
        display_pw_done,
        display_pw_getf,
        display_pw_putf,
        display_pw_reconfigure,
        display_pw_get_property,
        NULL, // _put_audio_frame
        NULL, // _reconfigure_audio
        MOD_NAME,
};

REGISTER_MODULE(pipewire, &display_pw_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
