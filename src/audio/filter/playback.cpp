/**
 * @file   playback.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET, z. s. p. o.
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
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <memory>
#include <string_view>

#include "debug.h"
#include "module.h"
#include "audio/audio_filter.h"
#include "audio/types.h"
#include "audio/audio_playback.h" 
#include "lib_common.h"
#include "utils/string_view_utils.hpp"

namespace{
struct Playback_dev_deleter{ void operator()(state_audio_playback *p){ audio_playback_done(p); } };
using Playback_dev_uniq = std::unique_ptr<state_audio_playback, Playback_dev_deleter>;
}

struct state_playback{
        state_playback(struct module *mod) : mod(MODULE_CLASS_DATA, mod, this) {  }

        module_raii mod;

        int bps = 0;
        int ch_count = 0;
        int sample_rate = 0;

        Playback_dev_uniq playback_dev;

};

static void usage(){
        printf("Plays audio using an UltraGrid playback device:\n\n");
        printf("playback usage:\n");
        printf("\tplayback:<playback dev>:<dev config>\n\n");
}

static af_result_code parse_cfg(state_playback *s, std::string_view cfg){
        auto tok = tokenize(cfg, ':');
        if(tok.empty() || tok == "help"){
                usage();
                return AF_HELP_SHOWN;
        } 

        auto dev = std::string(tok);
        auto dev_cfg = std::string(tokenize(cfg, ':'));

        state_audio_playback *tmp_dev;
        int ret = audio_playback_init(dev.c_str(), dev_cfg.c_str(), &tmp_dev);
        s->playback_dev.reset(tmp_dev);

        return ret == 0 ? AF_OK : AF_FAILURE;
}

static af_result_code init(struct module *parent, const char *cfg, void **state){
        auto s = std::make_unique<state_playback>(parent);

        auto ret = parse_cfg(s.get(), cfg);
        if(ret == AF_OK)
                *state = s.release();

        return ret;
};

static af_result_code configure(void *state,
                        int in_bps, int in_ch_count, int in_sample_rate)
{
        auto s = static_cast<state_playback *>(state);

        s->bps = in_bps;
        s->ch_count = in_ch_count;
        s->sample_rate = in_sample_rate;

        if (audio_playback_reconfigure(s->playback_dev.get(), s->bps * 8,
                                s->ch_count,
                                s->sample_rate) != TRUE) {
                return AF_FAILURE;
        }

        return AF_OK;
}

static void done(void *state){
        auto s = static_cast<state_playback *>(state);

        delete s;
}

static void get_configured(void *state,
                        int *bps, int *ch_count, int *sample_rate)
{
        auto s = static_cast<state_playback *>(state);

        if(bps) *bps = s->bps;
        if(ch_count) *ch_count = s->ch_count;
        if(sample_rate) *sample_rate = s->sample_rate;
}

static af_result_code filter(void *state, struct audio_frame **frame){
        auto s = static_cast<state_playback *>(state);

        struct message *msg;
        while ((msg = check_message(s->mod.get()))) {
                const char *text = ((msg_universal *) msg)->text;
                if(parse_cfg(s, text) != AF_OK){
                        free_message(msg, new_response(RESPONSE_BAD_REQUEST, nullptr));
                        continue;
                }

                free_message(msg, new_response(RESPONSE_OK, nullptr));
        }

        auto f = *frame;

        if(f->bps != s->bps || f->ch_count != s->ch_count || f->sample_rate != s->sample_rate){
                if(configure(state, f->bps, f->ch_count, f->sample_rate) != AF_OK){
                        return AF_MISCONFIGURED;
                }
        }

        audio_playback_put_frame(s->playback_dev.get(), f);

        return AF_OK;
}

static const struct audio_filter_info audio_filter_playback = {
        .name = "playback",
        .init = init,
        .done = done,
        .configure = configure,
        .get_configured_in = get_configured,
        .get_configured_out = get_configured,
        .filter = filter,
};

REGISTER_MODULE(playback, &audio_filter_playback, LIBRARY_CLASS_AUDIO_FILTER, AUDIO_FILTER_ABI_VERSION);
