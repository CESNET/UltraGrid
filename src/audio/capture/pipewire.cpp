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

#include "audio/audio_capture.h"
#include "audio/types.h"
#include "audio/pipewire_common.hpp"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/ring_buffer.h"
#include "utils/string_view_utils.hpp"

struct state_pipewire_cap{
        pipewire_state_common pw;

        pw_stream_uniq stream;
        spa_hook_uniq stream_listener;

        std::string target;

        audio_desc desc;
        ring_buffer_uniq ring_buf;

        struct audio_frame frame;
        std::vector<char> frame_data;
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
        spa_audio_info audio_params;

        if(!param || id != SPA_PARAM_Format)
                return;

        //TODO check return
        spa_format_parse(param, &audio_params.media_type, &audio_params.media_subtype);
        if(audio_params.media_type != SPA_MEDIA_TYPE_audio
                        || audio_params.media_subtype != SPA_MEDIA_SUBTYPE_raw)
        {
                return;
        }

        spa_format_audio_raw_parse(param, &audio_params.info.raw);

        log_msg(LOG_LEVEL_NOTICE, "Format change: %u %u %u\n",
                        audio_params.info.raw.format,
                        audio_params.info.raw.rate,
                        audio_params.info.raw.channels);

        //TODO: Actually handle format changes
}

const static pw_stream_events stream_events = { 
        .version = PW_VERSION_STREAM_EVENTS,
        .destroy = nullptr,
        .state_changed = nullptr,
        .control_info = nullptr,
        .io_changed = nullptr,
        .param_changed = on_param_changed,
        .add_buffer = nullptr,
        .remove_buffer = nullptr,
        .process = on_process,
        .drained = nullptr,
        .command = nullptr,
        .trigger_done = nullptr,
};

static void audio_cap_pw_help(){
        color_printf("Pipewire audio capture.\n");
        print_devices("Audio/Source");
}

static void *audio_cap_pipewire_init(struct module *parent, const char *cfg){
        std::string_view cfg_sv(cfg);

        std::string_view key = tokenize(cfg_sv, '=', '\"');
        std::string_view val = tokenize(cfg_sv, '=', '\"');

        std::string_view target_device;

        if(key == "help"){
                audio_cap_pw_help();
                return INIT_NOERR;
        } else if(key == "target"){
                target_device = val;
        }

        auto s = std::make_unique<state_pipewire_cap>();

        fprintf(stdout, "Compiled with libpipewire %s\n"
                        "Linked with libpipewire %s\n",
                        pw_get_headers_version(),
                        pw_get_library_version());

        s->target = std::string(target_device);

        initialize_pw_common(s->pw);

        auto props = pw_properties_new(
                        PW_KEY_MEDIA_TYPE, "Audio",
                        PW_KEY_MEDIA_CATEGORY, "Capture",
                        PW_KEY_MEDIA_ROLE, "Communication",
                        PW_KEY_APP_NAME, "UltraGrid",
                        PW_KEY_APP_ICON_NAME, "ultragrid",
                        PW_KEY_NODE_NAME, "ug capture",
                        PW_KEY_NODE_TARGET, s->target.c_str(), //TODO: deprecated in newer
                        nullptr);

        //TODO: Don't hardcode these
        unsigned rate = 48000;
        unsigned quant = 128;
        spa_audio_format format = SPA_AUDIO_FORMAT_S16;
        unsigned ch_count = 1;

        s->frame.ch_count = ch_count;
        s->frame.bps = 2;
        s->frame.sample_rate = rate;
        s->frame.max_size = s->frame.ch_count * s->frame.bps * s->frame.sample_rate;
        s->frame_data.resize(s->frame.max_size);
        s->frame.data = s->frame_data.data();

        pw_properties_setf(props, PW_KEY_NODE_RATE, "1/%u", rate);
        pw_properties_setf(props, PW_KEY_NODE_LATENCY, "%u/%u", quant, rate);

        std::byte buffer[1024];
        auto pod_builder = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

        auto audio_info = SPA_AUDIO_INFO_RAW_INIT(
                        .format = format,
                        .rate = rate,
                        .channels = ch_count,
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

        int buf_len_ms = 50;
        int ring_size = /*desc.bps TODO*/ 2 * ch_count * (rate * buf_len_ms / 1000);
        s->ring_buf.reset(ring_buffer_init(ring_size));

        pw_stream_connect(s->stream.get(),
                        PW_DIRECTION_INPUT,
                        PW_ID_ANY,
                        static_cast<pw_stream_flags>(
                                PW_STREAM_FLAG_AUTOCONNECT |
                                PW_STREAM_FLAG_MAP_BUFFERS |
                                PW_STREAM_FLAG_RT_PROCESS),
                        &params, 1);

        return s.release();
}

static struct audio_frame *audio_cap_pipewire_read(void *state){
        auto s = static_cast<state_pipewire_cap *>(state);

        s->frame.data_len = ring_buffer_read(s->ring_buf.get(), s->frame.data, s->frame.max_size);

        if(!s->frame.data_len)
                return nullptr;

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
