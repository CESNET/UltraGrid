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
};

static void *audio_cap_pipewire_init(struct module *parent, const char *cfg){
    std::string_view cfg_sv(cfg);

    std::string_view key = tokenize(cfg_sv, '=', '\"');
    std::string_view val = tokenize(cfg_sv, '=', '\"');

    std::string_view target_device;

    if(key == "help"){
        //audio_play_pw_help();
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

    return s.release();
}

static struct audio_frame *audio_cap_pipewire_read(void *state){
    auto s = static_cast<state_pipewire_cap *>(state);


    return nullptr;
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
