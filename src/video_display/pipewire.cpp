
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <memory>

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
#include <spa/param/video/format-utils.h>

#define MOD_NAME "[pw_disp] "

struct display_pw_state{
        pipewire_state_common pw;

        pw_stream_uniq stream;
        spa_hook_uniq stream_listener;

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
		SPA_PARAM_BUFFERS_stride,  SPA_POD_Int(stride));

	pw_stream_update_params(s->stream.get(), params, 1);

}

static void on_remove_buffer(void *data, struct pw_buffer *buffer){
        log_msg(LOG_LEVEL_NOTICE, "Buffer removed\n");
}

static void on_process(void *userdata) noexcept{
        auto s = static_cast<display_pw_state *>(userdata);

#if 0
        struct pw_buffer *b = pw_stream_dequeue_buffer(s->stream.get());
        if (!b) {
                pw_log_warn("out of buffers: %m");
                return;
        }
        struct spa_buffer *buf = b->buffer;

        char *dst = static_cast<char *>(buf->datas[0].data);
        if (!dst)
                return;


        buf->datas[0].chunk->offset = 0;
        buf->datas[0].chunk->stride = s->desc.ch_count * s->desc.bps;
        buf->datas[0].chunk->size = to_write;

        pw_stream_queue_buffer(s->stream.get(), b);
#endif
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "on process\n");
}


const static pw_stream_events stream_events = { 
        .version = PW_VERSION_STREAM_EVENTS,
        .destroy = nullptr,
        .state_changed = on_state_changed,
        .control_info = nullptr,
        .io_changed = nullptr,
        .param_changed = on_param_changed,
        .add_buffer = nullptr,
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

        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        /* 
         * TODO: The buffer could be destroyed at any time when the stream
         * disconnects (or for any other reason). Before it is destroyed, the
         * remove_buffer callback is called. If we don't do anything about it
         * the decoder will cause a SIGSEGV when writing to it. The solution
         * (from the pipewire Matrix channel) seems to be to block in the
         * remove_buffer callback until the frame comes back to the display.
         */

        struct pw_buffer *b = pw_stream_dequeue_buffer(s->stream.get());
        if (!b) {
                log_msg(LOG_LEVEL_WARNING, "Out of buffers!\n");
                return vf_alloc_desc_data(s->desc);
        }

        auto f = vf_alloc_desc(s->desc);
        f->callbacks.dispose_udata = b;

        f->tiles[0].data = static_cast<char *>(b->buffer->datas[0].data);

        b->buffer->datas[0].chunk->size = f->tiles[0].data_len;
        b->buffer->datas[0].chunk->offset = 0;
        b->buffer->datas[0].chunk->stride = vc_get_linesize(s->desc.width, s->desc.color_spec);

        return f;
}


static bool display_pw_putf(void *state, struct video_frame *frame, long long flags)
{
        auto s = static_cast<display_pw_state *>(state);

        if (flags == PUTF_DISCARD || frame == NULL) {
                return true;
        }

        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        auto b = static_cast<pw_buffer *>(frame->callbacks.dispose_udata);

        if(b)
                pw_stream_queue_buffer(s->stream.get(), b);
        else
                vf_free(frame);

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
//                        STREAM_TARGET_PROPERTY_KEY, s->target.c_str(),
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
                                PW_STREAM_FLAG_MAP_BUFFERS),
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
