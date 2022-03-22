/**
 * @file   delay.cpp
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
#include "lib_common.h"
#include "utils/ring_buffer.h"
#include "utils/misc.h"
#include "utils/sv_parse_num.hpp"

namespace{
        struct Ring_buf_deleter{
                void operator()(ring_buffer_t *ring){ ring_buffer_destroy(ring); }
        };
}

struct state_delay{
        state_delay(struct module *mod) : mod(MODULE_CLASS_DATA, mod, this) {  }

        module_raii mod;

        bool frame_mode = false;
        int req_delay = 0;

        int bps = 0;
        int ch_count = 0;
        int sample_rate = 0;

        int delay_size = 0;
        std::unique_ptr<ring_buffer_t, Ring_buf_deleter> ring;
};

static void usage(){
        printf("Delays audio:\n\n");
        printf("delay usage:\n");
        printf("\tdelay:<delay in milliseconds>\n\n");
}

static af_result_code parse_cfg(state_delay *s, std::string_view cfg){
        auto tok = tokenize(cfg, ':');
        if(tok.empty() || tok == "help"){
                usage();
                return AF_HELP_SHOWN;
        } 

        if(!parse_num(tok, s->req_delay)){
                log_msg(LOG_LEVEL_ERROR, "Failed to parse delay time\n");
                usage();
                return AF_FAILURE;
        }

        s->frame_mode = (tokenize(cfg, ':') == "frames");

        return AF_OK;
}

static af_result_code init(struct module *parent, const char *cfg, void **state){
        auto s = std::make_unique<state_delay>(parent);

        auto ret = parse_cfg(s.get(), cfg);
        if(ret == AF_OK)
                *state = s.release();

        return ret;
};

static void set_delay_size(state_delay *s, int size){
        if(size == s->delay_size)
                return;

        if(size == 0){
                s->ring.reset();
        } else {
                s->ring.reset(ring_buffer_init(size * 2));
                ring_fill(s->ring.get(), 0, size);
        }
        s->delay_size = size;
}

static af_result_code configure(void *state,
                        int in_bps, int in_ch_count, int in_sample_rate)
{
        auto s = static_cast<state_delay *>(state);

        s->bps = in_bps;
        s->ch_count = in_ch_count;
        s->sample_rate = in_sample_rate;

        if(!s->frame_mode){
                int samples = (s->sample_rate / 1000) * s->req_delay;
                set_delay_size(s, samples * s->ch_count * s->bps);
        }

        return AF_OK;
}

static void done(void *state){
        auto s = static_cast<state_delay *>(state);

        delete s;
}

static void get_configured(void *state,
                        int *bps, int *ch_count, int *sample_rate)
{
        auto s = static_cast<state_delay *>(state);

        if(bps) *bps = s->bps;
        if(ch_count) *ch_count = s->ch_count;
        if(sample_rate) *sample_rate = s->sample_rate;
}

static af_result_code filter(void *state, struct audio_frame **frame){
        auto s = static_cast<state_delay *>(state);

        struct message *msg;
        while ((msg = check_message(s->mod.get()))) {
                const char *text = ((msg_universal *) msg)->text;
                if(parse_cfg(s, text) != AF_OK){
                        free_message(msg, new_response(RESPONSE_BAD_REQUEST, nullptr));
                        continue;
                }

                if(!s->frame_mode){
                        int samples = (s->sample_rate / 1000) * s->req_delay;
                        set_delay_size(s, samples * s->ch_count * s->bps);
                }

                free_message(msg, new_response(RESPONSE_OK, nullptr));
        }

        auto f = *frame;
        if(s->frame_mode){
                set_delay_size(s, f->data_len * s->req_delay);
        }

        if(f->bps != s->bps || f->ch_count != s->ch_count){
                if(configure(state, f->bps, f->ch_count, f->sample_rate) != AF_OK){
                        return AF_MISCONFIGURED;
                }
        }

        //If ring is null, delay is 0
        if(!s->ring)
                return AF_OK;

        if(f->data_len <= s->delay_size){
                ring_buffer_write(s->ring.get(), f->data, f->data_len);
                ring_buffer_read(s->ring.get(), f->data, f->data_len);
        } else {
                int excess_size = f->data_len - s->delay_size;

                //Write the last delay_size bytes to buffer
                ring_buffer_write(s->ring.get(), f->data + excess_size, s->delay_size);

                //Move the beggining of frame to the end and prepend delay_size bytes from buffer
                memmove(f->data + s->delay_size, f->data, excess_size);
                ring_buffer_read(s->ring.get(), f->data, s->delay_size);
        }

        return AF_OK;
}

static const struct audio_filter_info audio_filter_delay = {
        .name = "delay",
        .init = init,
        .done = done,
        .configure = configure,
        .get_configured_in = get_configured,
        .get_configured_out = get_configured,
        .filter = filter,
};

REGISTER_MODULE(delay, &audio_filter_delay, LIBRARY_CLASS_AUDIO_FILTER, AUDIO_FILTER_ABI_VERSION);
