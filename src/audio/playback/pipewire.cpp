/**
 * @file   audio/playback/pipewire.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET z.s.p.o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif

#include <memory>

#include <spa/param/audio/format-utils.h>
#include <pipewire/pipewire.h>

#include "audio/audio_playback.h"
#include "audio/types.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/ring_buffer.h"
#include "utils/string_view_utils.hpp"

struct pipewire_thread_loop_deleter { void operator()(pw_thread_loop *l) { pw_thread_loop_destroy(l); } };
using pw_thread_loop_uniq = std::unique_ptr<pw_thread_loop, pipewire_thread_loop_deleter>;

struct pipewire_stream_deleter { void operator()(pw_stream *s) { pw_stream_destroy(s); } };
using pw_stream_uniq = std::unique_ptr<pw_stream, pipewire_stream_deleter>;

class pipewire_thread_loop_lock_guard{
public:
        pipewire_thread_loop_lock_guard(pw_thread_loop *loop) : l(loop) {
                pw_thread_loop_lock(l);
        }
        ~pipewire_thread_loop_lock_guard(){
                pw_thread_loop_unlock(l);
        }
        pipewire_thread_loop_lock_guard(pipewire_thread_loop_lock_guard&) = delete;
        pipewire_thread_loop_lock_guard& operator=(pipewire_thread_loop_lock_guard&) = delete;

private:
        pw_thread_loop *l;
};

struct pipewire_init_guard{
        pipewire_init_guard(){
                pw_init(nullptr, nullptr);
        }
        ~pipewire_init_guard(){
                pw_deinit();
        }
        pipewire_init_guard(pipewire_init_guard&) = delete;
        pipewire_init_guard& operator=(pipewire_init_guard&) = delete;
};

struct state_pipewire_play{
        pipewire_init_guard init_guard;

        audio_desc desc;

        pw_thread_loop_uniq pipewire_loop;

        ring_buffer_uniq ring_buf;
        pw_stream_uniq stream;

        double accumulator = 0;
};

static void audio_play_pw_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = static_cast<device_info *>(calloc(1, sizeof(device_info)));
        strcpy((*available_devices)[0].dev, "");
        strcpy((*available_devices)[0].name, "Default pipewire output");
        *count = 1;
}

static void audio_play_pw_help(){
        color_printf("Pipewire audio output.\n");
}

/* This function can only use realtime-safe calls (no locking, allocating, etc.)
*/
static void on_process(void *userdata) noexcept{
        auto s = static_cast<state_pipewire_play *>(userdata);

        struct pw_buffer *b = pw_stream_dequeue_buffer(s->stream.get());
        if (!b) {
                pw_log_warn("out of buffers: %m");
                return;
        }
        struct spa_buffer *buf = b->buffer;

        char *dst = static_cast<char *>(buf->datas[0].data);
        if (!dst)
                return;

        int to_write = std::min<int>(buf->datas[0].maxsize, ring_get_current_size(s->ring_buf.get()));

        ring_buffer_read(s->ring_buf.get(), dst, to_write);

        buf->datas[0].chunk->offset = 0;
        buf->datas[0].chunk->stride = s->desc.ch_count * s->desc.bps;
        buf->datas[0].chunk->size = to_write;

        pw_stream_queue_buffer(s->stream.get(), b);
}

static void * audio_play_pw_init(const char *cfg){
        std::string_view cfg_sv(cfg);

        if(cfg_sv == "help"){
                audio_play_pw_help();
                return &audio_init_state_ok;
        }

        auto state = std::make_unique<state_pipewire_play>();

        state->pipewire_loop.reset(pw_thread_loop_new("Playback", nullptr));
        pw_thread_loop_start(state->pipewire_loop.get());

        return state.release();
}

static void audio_play_pw_put_frame(void *state, const struct audio_frame *frame){
        auto s = static_cast<state_pipewire_play *>(state);

        ring_buffer_write(s->ring_buf.get(), frame->data, frame->data_len);
}

static bool is_format_supported(void *data, size_t *len){
        struct audio_desc desc;
        if (*len < sizeof(desc)) {
                return false;
        } else {
                memcpy(&desc, data, sizeof(desc));
        }

        return desc.codec == AC_PCM && desc.bps >= 1 && desc.bps <= 4;
}

static bool audio_play_pw_ctl(void *state, int request, void *data, size_t *len){
        UNUSED(state);

        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                return is_format_supported(data, len);
        default:
                return false;

        }
}

const static pw_stream_events stream_events = { 
        .version = PW_VERSION_STREAM_EVENTS,
        .destroy = nullptr,
        .state_changed = nullptr,
        .control_info = nullptr,
        .io_changed = nullptr,
        .param_changed = nullptr,
        .add_buffer = nullptr,
        .remove_buffer = nullptr,
        .process = on_process,
        .drained = nullptr,
        .command = nullptr,
        .trigger_done = nullptr,
};

static int audio_play_pw_reconfigure(void *state, struct audio_desc desc){
        auto s = static_cast<state_pipewire_play *>(state);

        unsigned rate = desc.sample_rate;
        unsigned quant = 128;
        spa_audio_format format = SPA_AUDIO_FORMAT_UNKNOWN;

        switch(desc.bps){
        case 1:
                format = SPA_AUDIO_FORMAT_S8;
                break;
        case 2:
                format = SPA_AUDIO_FORMAT_S16;
                break;
        case 3:
                format = SPA_AUDIO_FORMAT_S24;
                break;
        case 4:
                format = SPA_AUDIO_FORMAT_S32;
                break;
        default:
                break;
        }

        auto props = pw_properties_new(
                        PW_KEY_MEDIA_TYPE, "Audio",
                        PW_KEY_MEDIA_CATEGORY, "Playback",
                        PW_KEY_MEDIA_ROLE, "Communication",
                        PW_KEY_APP_NAME, "UltraGrid",
                        PW_KEY_APP_ICON_NAME, "ultragrid",
                        PW_KEY_NODE_NAME, "ug play",
                        nullptr);

        pw_properties_setf(props, PW_KEY_NODE_RATE, "1/%u", rate);
        pw_properties_setf(props, PW_KEY_NODE_LATENCY, "%u/%u", quant, rate);

        std::byte buffer[1024];
        auto pod_builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

        auto audio_info = SPA_AUDIO_INFO_RAW_INIT(
                        .format = format,
                        .rate = rate,
                        .channels = static_cast<unsigned>(desc.ch_count),
                        );
        const spa_pod *params = spa_format_audio_raw_build(&pod_builder, SPA_PARAM_EnumFormat,
                        &audio_info);

        s->desc = desc;

        /*
         * Pipewire thread loop lock
         */
        pipewire_thread_loop_lock_guard lock(s->pipewire_loop.get());

        s->stream.reset(pw_stream_new_simple(
                                pw_thread_loop_get_loop(s->pipewire_loop.get()),
                                "UltraGrid playback",
                                props,
                                &stream_events,
                                s));

        int buf_len_ms = 50;
        int ring_size = desc.bps * desc.ch_count * (desc.sample_rate * buf_len_ms / 1000);
        s->ring_buf.reset(ring_buffer_init(ring_size));

        pw_stream_connect(s->stream.get(),
                        PW_DIRECTION_OUTPUT,
                        PW_ID_ANY,
                        static_cast<pw_stream_flags>(
                                PW_STREAM_FLAG_AUTOCONNECT |
                                PW_STREAM_FLAG_MAP_BUFFERS |
                                PW_STREAM_FLAG_RT_PROCESS),
                        &params, 1);

        return true;
}

static void audio_play_pw_done(void *state){
        auto s = static_cast<state_pipewire_play *>(state);

        pw_thread_loop_stop(s->pipewire_loop.get());
        delete s;
}

static const struct audio_playback_info aplay_pw_info = {
        audio_play_pw_probe,
        audio_play_pw_init,
        audio_play_pw_put_frame,
        audio_play_pw_ctl,
        audio_play_pw_reconfigure,
        audio_play_pw_done
};

REGISTER_MODULE(pipewire, &aplay_pw_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

