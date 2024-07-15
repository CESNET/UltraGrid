/**
 * @file   channel_remap.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2024 CESNET, z. s. p. o.
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
#include "audio/utils.h"
#include "lib_common.h"
#include "utils/misc.h"
#include "utils/string_view_utils.hpp"

#define MOD_NAME "[Remap filter] "

struct state_channel_remap{
        state_channel_remap(struct module *mod) : mod(MODULE_CLASS_DATA, mod, this) {  }

        module_raii mod;

        int bps = 0;
        int ch_count = 0;
        int sample_rate = 0;

        struct channel_map channel_map;

        std::vector<char> out_buffer;
        struct audio_frame out_frame = {};
};

static void usage(){
        color_printf("Audio filter " TBOLD(
            "channel_remap") " allows remapping audio channels.\n\n");
        color_printf("Usage:\n");
        color_printf("\t" TBOLD("--audio-filter channel_remap:[<src_ch>]:<dst_ch>[,[<src_ch>]:<dst_ch>]...\n\n"));
        color_printf("(indexed from zero)\n");
        color_printf("Maps source channels into destination channels.\n");
        color_printf("If multiple source channels are mapped into the same destination channel, they are mixed together.\n");
        color_printf("E.g. 0:0,1:0 means that the first and second channel are mixed to produce a single mono channel.\n");
        color_printf("2:0,3:1 means that the third and fourth channel are mapped into first and second channel respectively.\n");
}

static af_result_code init(struct module *parent, const char *cfg, void **state){
        auto s = std::make_unique<state_channel_remap>(parent);

        if(strcmp(cfg, "help") == 0){
                usage();
                return AF_HELP_SHOWN;
        }

        if(!parse_channel_map_cfg(&s->channel_map, cfg)){
                return AF_FAILURE;
        }

        assert(s->channel_map.size > 0);

        *state = s.release();

        return AF_OK;
};

static af_result_code configure(void *state,
                        int in_bps, int in_ch_count, int in_sample_rate)
{
        auto s = static_cast<state_channel_remap *>(state);

        s->bps = in_bps;
        s->ch_count = in_ch_count;
        s->sample_rate = in_sample_rate;

        s->out_frame.bps = s->bps;
        s->out_frame.sample_rate = s->sample_rate;

        if(s->channel_map.size > in_ch_count){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Audio channel map references channels with idx higher than ch. count!\n");
        }

        return AF_OK;
}

static void done(void *state){
        auto s = static_cast<state_channel_remap *>(state);

        delete s;
}

static void get_configured_in(void *state,
                        int *bps, int *ch_count, int *sample_rate)
{
        auto s = static_cast<state_channel_remap *>(state);

        if(bps) *bps = s->bps;
        if(ch_count) *ch_count = s->ch_count;
        if(sample_rate) *sample_rate = s->sample_rate;
}

static void get_configured_out(void *state,
                        int *bps, int *ch_count, int *sample_rate)
{
        auto s = static_cast<state_channel_remap *>(state);

        if(bps) *bps = s->bps;
        if(ch_count) *ch_count = s->channel_map.max_output + 1;
        if(sample_rate) *sample_rate = s->sample_rate;
}

static af_result_code filter(void *state, struct audio_frame **frame){
        auto s = static_cast<state_channel_remap *>(state);
        auto f = *frame;

        if(f->bps != s->bps || f->ch_count != s->ch_count){
                if(configure(state, f->bps, f->ch_count, f->sample_rate) != AF_OK){
                        return AF_MISCONFIGURED;
                }
        }

        int frame_count = f->data_len / f->ch_count / f->bps;

        s->out_frame.ch_count = s->channel_map.max_output + 1;
        s->out_frame.data_len = frame_count * s->out_frame.bps * s->out_frame.ch_count;
        if(s->out_frame.data_len > s->out_frame.max_size){
                s->out_frame.max_size = s->out_frame.data_len;
                s->out_buffer.resize(s->out_frame.max_size);
                s->out_frame.data = s->out_buffer.data();
        }

        memset(s->out_frame.data, 0, s->out_frame.data_len);

        int max_src_count = std::min(s->channel_map.size, f->ch_count);

        for(int src_ch = 0; src_ch < max_src_count; src_ch++){
                for(int dst_ch = 0; dst_ch < s->channel_map.sizes[src_ch]; dst_ch++){
                        remux_and_mix_channel(s->out_frame.data, f->data,
                                        s->bps, frame_count,
                                        f->ch_count, s->out_frame.ch_count,
                                        src_ch, s->channel_map.map[src_ch][dst_ch], 1.0);
                }
        }

        s->out_frame.timestamp = f->timestamp;

        *frame = &s->out_frame;

        return AF_OK;
}

static const struct audio_filter_info audio_filter_channel_remap = {
        .name = "channel_remap",
        .init = init,
        .done = done,
        .configure = configure,
        .get_configured_in = get_configured_in,
        .get_configured_out = get_configured_out,
        .filter = filter,
};

REGISTER_MODULE(channel_remap, &audio_filter_channel_remap, LIBRARY_CLASS_AUDIO_FILTER, AUDIO_FILTER_ABI_VERSION);
