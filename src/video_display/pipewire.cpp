/**
 * @file   video_display/pipewire.cpp
 * @author Martin Piatka <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2023-2026 CESNET, zájmové sdružení právnických osob
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

#include <cassert>
#include <cstdlib>
#include <unistd.h>
#include <cstring>
#include <memory>
#include <atomic>
#include <sys/mman.h>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "video_codec.h"
#include "video_frame.h"
#include "video_display.h"

#include "pipewire_common.hpp"
#include "utils/string_view_utils.hpp"
#include <spa/param/video/format-utils.h>

#define MOD_NAME "[PW disp] "

namespace{
        using unique_frame = std::unique_ptr<video_frame, deleter_from_fcn<vf_free>>;

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
                off_t size = 0;
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
                if(ftruncate(buf->fd, buf->size) < 0){
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
        unique_frame in_flight_frame;

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

static void on_param_changed(void *state, uint32_t id, const spa_pod *param){
        auto s = static_cast<display_pw_state *>(state);

        if(!param || id != SPA_PARAM_Format)
                return;

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "on param changed\n");

        spa_video_info_raw format{};
        spa_format_video_raw_parse(param, &format);

        constexpr int MAX_BUFFERS = 16;
        int stride = vc_get_linesize(s->desc.width, s->desc.color_spec);

        const spa_pod *params[1];
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

static void on_add_buffer(void *data, pw_buffer *buffer){
        auto s = static_cast<display_pw_state *>(data);

        const int stride = vc_get_linesize(s->desc.width, s->desc.color_spec);
        size_t size = stride * s->desc.height;

        spa_data *d = buffer->buffer->datas;
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
        ug_frame->tiles[0].data = static_cast<char *>(buf->ptr);

        log_msg(LOG_LEVEL_VERBOSE, "Buffer added\n");
}

static void on_remove_buffer(void * /*data*/, pw_buffer *buffer){
        auto buf = static_cast<memfd_buffer *>(buffer->user_data);

        if(buf->f)
                vf_free(buf->f);

        buf->b = nullptr;
        memfd_buf_unref(&buf);

        log_msg(LOG_LEVEL_VERBOSE, "Buffer removed\n");
}

constexpr pw_stream_events stream_events = []{
        pw_stream_events events{};
        events.version = PW_VERSION_STREAM_EVENTS;
        events.state_changed = on_state_changed;
        events.param_changed = on_param_changed;
        events.add_buffer = on_add_buffer;
        events.remove_buffer = on_remove_buffer;
        return events;
}();

static void display_pw_help(){
        color_printf("Pipewire video output.\n");
        color_printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-d pipewire" TERM_FG_RESET "[:target=<device>]\n" TERM_RESET);
        color_printf("\n");

        color_printf("Devices:\n");
        print_devices("Stream/Input/Video");
}

static void *display_pw_init(module * /*parent*/, const char *cfg, unsigned int /*flags*/)
{
        auto s = std::make_unique<display_pw_state>();

        std::string_view cfg_sv(cfg);
        while(!cfg_sv.empty()){
                auto tok = tokenize(cfg_sv, ':', '"');

                const auto key = tokenize(tok, '=');
                const auto val = tokenize(tok, '=');

                if(key == "help"){
                        display_pw_help();
                        return INIT_NOERR;
                }

                if(key == "target"){
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

        {
                pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());
                pw_stream_disconnect(s->stream.get());
        }

        pw_thread_loop_stop(s->pw.pipewire_loop.get());
}

static video_frame *display_pw_getf(void *state)
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

        pw_buffer *b = pw_stream_dequeue_buffer(s->stream.get());
        if (!b) {
                log_msg(LOG_LEVEL_WARNING, "Out of buffers!\n");
                return get_dummy(s);
        }

        auto buf = static_cast<memfd_buffer *>(b->user_data);
        s->in_flight_frame.reset(std::exchange(buf->f, nullptr));

        b->buffer->datas[0].chunk->size = s->in_flight_frame->tiles[0].data_len;
        b->buffer->datas[0].chunk->offset = 0;
        b->buffer->datas[0].chunk->stride = vc_get_linesize(s->desc.width, s->desc.color_spec);

        return s->in_flight_frame.get();
}


static bool display_pw_putf(void *state, video_frame *frame, long long flags)
{
        auto s = static_cast<display_pw_state *>(state);

        if (flags == PUTF_DISCARD || !frame || frame == s->dummy_frame.get()) {
                return true;
        }

        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        auto buf = static_cast<memfd_buffer *>(frame->callbacks.dispose_udata);

        if(!buf->b){
                //Frame is invalid - buffer got removed
                return false;
        }

        assert(frame == s->in_flight_frame.get());
        buf->f = s->in_flight_frame.release();

        pw_stream_queue_buffer(s->stream.get(), buf->b);

        return true;
}

static bool display_pw_get_property(void *state, int property, void *val, size_t *len)
{
        [[maybe_unused]] auto s = static_cast<display_pw_state *>(state);

        codec_t codecs[] = {UYVY, RGB, RGBA, BGR};
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

static bool display_pw_reconfigure(void *state, video_desc desc)
{
        auto s = static_cast<display_pw_state *>(state);

        s->desc = desc;

        std::string node_name = "ultragrid_out_" + std::to_string(getpid());

        auto props = pw_properties_new(
                        PW_KEY_MEDIA_TYPE, "Video",
                        PW_KEY_MEDIA_CATEGORY, "Source",
                        PW_KEY_MEDIA_ROLE, "Communication",
                        PW_KEY_APP_NAME, "UltraGrid",
                        PW_KEY_APP_ICON_NAME, "ultragrid",
                        PW_KEY_NODE_NAME, node_name.c_str(),
                        PW_KEY_NODE_DESCRIPTION, "UltraGrid playback",
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

        const spa_pod *params[1];

        auto framerate = SPA_FRACTION(
                        static_cast<unsigned>(get_framerate_n(desc.fps)),
                        static_cast<unsigned>(get_framerate_d(desc.fps)));
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

static void display_pw_probe(device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = nullptr;
        *count = 0;
}

constexpr video_display_info display_pw_info = {
        display_pw_probe,
        display_pw_init,
        nullptr, // _run
        display_pw_done,
        display_pw_getf,
        display_pw_putf,
        display_pw_reconfigure,
        display_pw_get_property,
        nullptr, // _put_audio_frame
        nullptr, // _reconfigure_audio
        MOD_NAME,
};

REGISTER_MODULE(pipewire, &display_pw_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
