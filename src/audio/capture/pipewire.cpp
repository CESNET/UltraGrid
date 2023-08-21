/**
 * @file   audio/capture/pipewire.cpp
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
#include <vector>
#include <thread>
#include <chrono>

#include "audio/audio_capture.h"
#include "audio/types.h"
#include "pipewire_common.hpp"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/ring_buffer.h"
#include "utils/string_view_utils.hpp"

#define MOD_NAME "[PW cap.] "

struct state_pipewire_cap{
        pipewire_state_common pw;

        pw_stream_uniq stream;
        spa_hook_uniq stream_listener;

        std::string target;

        audio_desc desc;
        ring_buffer_uniq ring_buf;

        struct audio_frame frame;
        std::vector<char> frame_data;

        unsigned ch_count = 1;
        unsigned sample_rate = 48000;
        unsigned quant = 128;
        unsigned bps = 2;
        unsigned buf_len_ms = 100;
};

/*
 * Called from real time thread. Only realtime-safe function calls allowed here.
 */
static void on_process(void *state){
        auto s = static_cast<state_pipewire_cap *>(state);

        struct pw_buffer *b = pw_stream_dequeue_buffer(s->stream.get());
        if (!b) {
                pw_log_warn("out of buffers: %m");
                return;
        }
        struct spa_buffer *buf = b->buffer;

        char *src = static_cast<char *>(buf->datas[0].data);
        if (!src)
                return;

        assert(buf->datas[0].chunk->offset == 0);
        ring_buffer_write(s->ring_buf.get(), src, buf->datas[0].chunk->size);

        pw_stream_queue_buffer(s->stream.get(), b);
}

static void on_param_changed(void *state, uint32_t id, const struct spa_pod *param){
        auto s = static_cast<state_pipewire_cap *>(state);
        spa_audio_info audio_params;

        if(!param || id != SPA_PARAM_Format)
                return;

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

        /* It is not expected that params change while the stream is active,
         * therefore this should run only during during init (we wait for stream to
         * change into the streaming state there).  As a result, we probably don't
         * need any extra synchronization between this and
         * audio_cap_pipewire_read().
         */
        s->frame.ch_count = audio_params.info.raw.channels;
        s->frame.bps = get_bps_from_pw_format(audio_params.info.raw.format);
        s->frame.sample_rate = audio_params.info.raw.rate;
        s->frame.max_size = s->frame.ch_count * s->frame.bps * s->frame.sample_rate;
        s->frame_data.resize(s->frame.max_size);
        s->frame.data = s->frame_data.data();

        std::byte buffer[1024];
        auto pod_builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

        unsigned buffer_size = (s->buf_len_ms * s->frame.sample_rate / 1000) * s->frame.ch_count * s->frame.bps;

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Requesting buffer size %u\n", buffer_size);

        spa_pod *new_params = (spa_pod *) spa_pod_builder_add_object(&pod_builder,
                        SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
                        SPA_PARAM_BUFFERS_blocks, SPA_POD_Int(1),
                        SPA_PARAM_BUFFERS_size, SPA_POD_CHOICE_RANGE_Int(buffer_size, 0, INT32_MAX),
                        SPA_PARAM_BUFFERS_stride, SPA_POD_Int(s->frame.ch_count * s->frame.bps));

        if(!new_params){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to build pw buffer params pod\n");
                return;
        }

        if (pw_stream_update_params(s->stream.get(), const_cast<const spa_pod **>(&new_params), 1) < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to set stream params\n");
        }

}

static void on_state_changed(void *state, enum pw_stream_state old, enum pw_stream_state new_state, const char *error)
{
        auto s = static_cast<state_pipewire_cap *>(state);
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Stream state change: %s -> %s\n",
                        pw_stream_state_as_string(old),
                        pw_stream_state_as_string(new_state));

        if(error){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Stream error: %s\n", error);
        }

        pw_thread_loop_signal(s->pw.pipewire_loop.get(), false);
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

static void audio_cap_pw_help(){
        color_printf("Pipewire audio capture.\n");
        color_printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-r pipewire" TERM_FG_RESET "[:target=<device>][:buffer-len=<millis>][:channels=<ch_count>][:sample-rate=<rate>]\n" TERM_RESET);
        color_printf("\n");

        color_printf("Devices:\n");
        print_devices("Audio/Source");
}

static void *audio_cap_pipewire_init(struct module *parent, const char *cfg){
        UNUSED(parent);
        auto s = std::make_unique<state_pipewire_cap>();

        std::string_view cfg_sv(cfg);
        while(!cfg_sv.empty()){
                auto tok = tokenize(cfg_sv, ':', '"');

                auto key = tokenize(tok, '=');
                auto val = tokenize(tok, '=');

                if(key == "help"){
                        audio_cap_pw_help();
                        return INIT_NOERR;
                } else if(key == "target"){
                        s->target = val;
                } else if(key == "channels"){
                        parse_num(val, s->ch_count);
                } else if(key == "buffer-len"){
                        parse_num(val, s->buf_len_ms);
                } else if(key == "sample-rate"){
                        parse_num(val, s->sample_rate);
                }
        }

        initialize_pw_common(s->pw);

        log_msg(LOG_LEVEL_INFO, MOD_NAME "Compiled with libpipewire %s\n", pw_get_headers_version());
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Linked with libpipewire %s\n", pw_get_library_version());

        auto props = pw_properties_new(
                        PW_KEY_MEDIA_TYPE, "Audio",
                        PW_KEY_MEDIA_CATEGORY, "Capture",
                        PW_KEY_MEDIA_ROLE, "Communication",
                        PW_KEY_APP_NAME, "UltraGrid",
                        PW_KEY_APP_ICON_NAME, "ultragrid",
                        PW_KEY_NODE_NAME, "ug capture",
                        STREAM_TARGET_PROPERTY_KEY, s->target.c_str(),
                        nullptr);

        spa_audio_format format = get_pw_format_from_bps(s->bps);

        pw_properties_setf(props, PW_KEY_NODE_RATE, "1/%u", s->sample_rate);
        pw_properties_setf(props, PW_KEY_NODE_LATENCY, "%u/%u", s->quant, s->sample_rate);

        std::byte buffer[1024];
        auto pod_builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

        auto audio_info = SPA_AUDIO_INFO_RAW_INIT(
                        .format = format,
                        .rate = s->sample_rate,
                        .channels = s->ch_count,
                        );

        const spa_pod *params = spa_format_audio_raw_build(&pod_builder, SPA_PARAM_EnumFormat,
                        &audio_info);

        /*
         * Pipewire thread loop lock
         */
        pipewire_thread_loop_lock_guard lock(s->pw.pipewire_loop.get());

        s->stream.reset(pw_stream_new(
                                s->pw.pipewire_core.get(),
                                "UltraGrid capture",
                                props));

        pw_stream_add_listener(
                        s->stream.get(),
                        &s->stream_listener.get(),
                        &stream_events,
                        s.get());

        int ring_size = s->bps * s->ch_count * (s->sample_rate * s->buf_len_ms * 2 / 1000);
        s->ring_buf.reset(ring_buffer_init(ring_size));

        pw_stream_connect(s->stream.get(),
                        PW_DIRECTION_INPUT,
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
                        break;
                }
                if(stream_state == PW_STREAM_STATE_ERROR){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize stream: %s\n", error ? error : "(no err msg)");
                        return nullptr;
                }
        }

        return s.release();
}

static struct audio_frame *audio_cap_pipewire_read(void *state){
        auto s = static_cast<state_pipewire_cap *>(state);

        s->frame.data_len = ring_buffer_read(s->ring_buf.get(), s->frame.data, s->frame.max_size);

        if(!s->frame.data_len){
                timespec ts{0, 1000000};
                nanosleep(&ts, nullptr);
                return nullptr;
        }

        return &s->frame;
}

static void audio_cap_pipewire_done(void *state){
        auto s = static_cast<state_pipewire_cap *>(state);

        pw_thread_loop_stop(s->pw.pipewire_loop.get());
        delete s;
}

static void audio_cap_pipewire_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = static_cast<device_info *>(calloc(1, sizeof(device_info)));
        strcpy((*available_devices)[0].dev, "");
        strcpy((*available_devices)[0].name, "Default pipewire capture");
        *count = 1;
}

static const struct audio_capture_info acap_pipewire_info = {
        audio_cap_pipewire_probe,
        audio_cap_pipewire_init,
        audio_cap_pipewire_read,
        audio_cap_pipewire_done
};

REGISTER_MODULE(pipewire, &acap_pipewire_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);
