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

#include <algorithm>                    // for min
#include <cassert>                      // for assert
#include <cstddef>                      // for byte, size_t
#include <cstdint>                      // for INT32_MAX, uint32_t
#include <cstdlib>                      // for calloc, free
#include <cstring>                      // for strcpy, memcpy
#include <memory>
#include <string>                       // for char_traits, basic_string
#include <string_view>                  // for operator==, basic_string_view

#include "audio/audio_playback.h"
#include "audio/types.h"
#include "pipewire_common.hpp"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/ring_buffer.h"
#include "utils/string_view_utils.hpp"

#define MOD_NAME "[PW aplay] "

struct state_pipewire_play{
        pipewire_state_common pw;

        pw_stream_uniq stream;
        spa_hook_uniq stream_listener;

        std::string target;

        audio_desc desc;
        ring_buffer_uniq ring_buf;
        unsigned buf_len_ms = 100;
        unsigned quant = 128;
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
        color_printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-r pipewire" TERM_FG_RESET "[:target=<device>][:buffer-len=<millis>][:quant=<samples>]\n" TERM_RESET);
        color_printf("\n");

        color_printf("Devices:\n");
        print_devices("Audio/Sink");
}

/* This function can only use realtime-safe calls (no locking, allocating, etc.)
*/
static void on_process(void *userdata) noexcept{
        auto s = static_cast<state_pipewire_play *>(userdata);

        auto avail = ring_get_current_size(s->ring_buf.get());

        while(avail > 0){
                struct pw_buffer *b = pw_stream_dequeue_buffer(s->stream.get());
                if (!b) {
                        pw_log_warn("out of buffers: %m");
                        return;
                }
                struct spa_buffer *buf = b->buffer;

                char *dst = static_cast<char *>(buf->datas[0].data);
                if (!dst)
                        return;

                int to_write = std::min<int>(buf->datas[0].maxsize, avail);

                ring_buffer_read(s->ring_buf.get(), dst, to_write);
                avail -= to_write;

                buf->datas[0].chunk->offset = 0;
                buf->datas[0].chunk->stride = s->desc.ch_count * s->desc.bps;
                buf->datas[0].chunk->size = to_write;

                pw_stream_queue_buffer(s->stream.get(), b);
        }
}

static void * audio_play_pw_init(const struct audio_playback_opts *opts){
        auto s = std::make_unique<state_pipewire_play>();

        std::string_view cfg_sv(opts->cfg);
        while(!cfg_sv.empty()){
                auto tok = tokenize(cfg_sv, ':', '"');
                auto key = tokenize(tok, '=');
                auto val = tokenize(tok, '=');

                if(key == "help"){
                        audio_play_pw_help();
                        return INIT_NOERR;
                } else if(key == "target"){
                        s->target = val;
                } else if(key == "buffer-len"){
                        parse_num(val, s->buf_len_ms);
                } else if(key == "quant"){
                        parse_num(val, s->quant);
                }
        }

        initialize_pw_common(s->pw);

        log_msg(LOG_LEVEL_INFO, MOD_NAME "Compiled with libpipewire %s\n", pw_get_headers_version());
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Linked with libpipewire %s\n", pw_get_library_version());

        return s.release();
}

static void audio_play_pw_put_frame(void *state, const struct audio_frame *frame){
        auto s = static_cast<state_pipewire_play *>(state);

        auto avail = ring_get_available_write_size(s->ring_buf.get());
        auto to_write = frame->data_len;
        if(to_write > avail){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Got frame of len %d, but ring has only %d free\n", frame->data_len, ring_get_available_write_size(s->ring_buf.get()));
                to_write = avail;
        }

        ring_buffer_write(s->ring_buf.get(), frame->data, to_write);
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

static void on_state_changed(void *state, enum pw_stream_state old, enum pw_stream_state new_state, const char *error)
{
        auto s = static_cast<state_pipewire_play *>(state);
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Stream state change: %s -> %s\n",
                        pw_stream_state_as_string(old),
                        pw_stream_state_as_string(new_state));

        if(error){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Stream error: %s\n", error);
        }

        pw_thread_loop_signal(s->pw.pipewire_loop.get(), false);
}

static void on_param_changed(void *state, uint32_t id, const struct spa_pod *param){
        auto s = static_cast<state_pipewire_play *>(state);

        if(!param || id != SPA_PARAM_Format)
                return;

        spa_audio_info audio_params;
        int res = spa_format_parse(param, &audio_params.media_type, &audio_params.media_subtype);
        if(res < 0
                        || audio_params.media_type != SPA_MEDIA_TYPE_audio
                        || audio_params.media_subtype != SPA_MEDIA_SUBTYPE_raw)
        {
                return;
        }

        spa_format_audio_raw_parse(param, &audio_params.info.raw);

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Pipewire format change: %s, %u ch, %uHz\n",
                        spa_debug_type_find_short_name(spa_type_audio_format, audio_params.info.raw.format),
                        audio_params.info.raw.channels,
                        audio_params.info.raw.rate);

        assert(audio_params.info.raw.rate == (unsigned) s->desc.sample_rate);
        assert(audio_params.info.raw.channels == (unsigned) s->desc.ch_count);
        assert(audio_params.info.raw.format == get_pw_format_from_bps(s->desc.bps));

        std::byte buffer[1024];
        auto pod_builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

        unsigned buffer_size = (s->buf_len_ms * s->desc.sample_rate / 1000) * s->desc.ch_count * s->desc.bps;

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Requesting buffer size %u\n", buffer_size);

        spa_pod *new_params = (spa_pod *) spa_pod_builder_add_object(&pod_builder,
                        SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
                        SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
                        SPA_PARAM_BUFFERS_size, SPA_POD_CHOICE_RANGE_Int(buffer_size, 0, INT32_MAX),
                        SPA_PARAM_BUFFERS_stride, SPA_POD_Int(s->desc.ch_count * s->desc.bps));

        if(!new_params){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to build pw buffer params pod\n");
                return;
        }

        if (pw_stream_update_params(s->stream.get(), const_cast<const spa_pod **>(&new_params), 1) < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to set stream params\n");
        }
}

const static pw_stream_events stream_events = { 
        .version = PW_VERSION_STREAM_EVENTS,
        .destroy = nullptr,
        .state_changed = on_state_changed,
        .control_info = nullptr,
        .io_changed = nullptr,
        .param_changed = on_param_changed,
        .add_buffer = nullptr,
        .remove_buffer = nullptr,
        .process = on_process,
        .drained = nullptr,
#if PW_MAJOR > 0 || PW_MINOR > 3 || (PW_MINOR == 3 && PW_MICRO > 39)
        .command = nullptr,
        .trigger_done = nullptr,
#endif
};

static bool audio_play_pw_reconfigure(void *state, struct audio_desc desc){
        auto s = static_cast<state_pipewire_play *>(state);

        unsigned rate = desc.sample_rate;
        spa_audio_format format = get_pw_format_from_bps(desc.bps);

        auto props = pw_properties_new(
                        PW_KEY_MEDIA_TYPE, "Audio",
                        PW_KEY_MEDIA_CATEGORY, "Playback",
                        PW_KEY_MEDIA_ROLE, "Communication",
                        PW_KEY_APP_NAME, "UltraGrid",
                        PW_KEY_APP_ICON_NAME, "ultragrid",
                        PW_KEY_NODE_NAME, "ug play",
                        STREAM_TARGET_PROPERTY_KEY, s->target.c_str(),
                        nullptr);

        pw_properties_setf(props, PW_KEY_NODE_RATE, "1/%u", rate);
        pw_properties_setf(props, PW_KEY_NODE_LATENCY, "%u/%u", s->quant, rate);

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
        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        s->stream.reset(pw_stream_new(
                                s->pw.pipewire_core.get(),
                                "UltraGrid playback",
                                props));

        pw_stream_add_listener(
                        s->stream.get(),
                        &s->stream_listener.get(),
                        &stream_events,
                        s);

        unsigned ring_size = (s->buf_len_ms * desc.sample_rate / 1000) * desc.ch_count * desc.bps * 2;
        s->ring_buf.reset(ring_buffer_init(ring_size));

        pw_stream_connect(s->stream.get(),
                        PW_DIRECTION_OUTPUT,
                        PW_ID_ANY,
                        static_cast<pw_stream_flags>(
                                PW_STREAM_FLAG_AUTOCONNECT |
                                PW_STREAM_FLAG_MAP_BUFFERS |
                                PW_STREAM_FLAG_RT_PROCESS),
                        &params, 1);

        while(true){
                pw_thread_loop_wait(s->pw.pipewire_loop.get());
                const char *error = nullptr;
                auto stream_state = pw_stream_get_state(s->stream.get(), &error);
                if(stream_state == PW_STREAM_STATE_STREAMING){
                        pw_time time;
                        float delay = 0;
#if PW_MAJOR > 0 || PW_MINOR > 3 || (PW_MINOR == 3 && PW_MICRO >= 50)
                        pw_stream_get_time_n(s->stream.get(), &time, sizeof(time));
                        delay += time.buffered * 1000.f / rate;
#else
                        pw_stream_get_time(s->stream.get(), &time);
#endif

                        delay += time.delay * 1000.f * time.rate.num / time.rate.denom;
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Successfully reconfigured and pipewire reports %.2fms of playback delay\n", delay);
                        return true;
                }
                if(stream_state == PW_STREAM_STATE_ERROR){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize stream: %s\n", error ? error : "(no err msg)");
                        return false;
                }
        }

        return true;
}

static void audio_play_pw_done(void *state){
        auto s = static_cast<state_pipewire_play *>(state);

        pw_thread_loop_stop(s->pw.pipewire_loop.get());
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

