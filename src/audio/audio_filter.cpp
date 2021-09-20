/**
 * @file   audio_filter.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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

#include "debug.h"

#include <vector>

#include "audio_filter.h"
#include "lib_common.h"
#include "module.h"
#include "utils/color_out.h"

af_result_code audio_filter_init(const char *name, const char *cfg,
                struct audio_filter *filter)
{
        const auto filters = get_libraries_for_class(LIBRARY_CLASS_AUDIO_FILTER,
                        AUDIO_FILTER_ABI_VERSION);

        for(const auto& i : filters){
                auto funcs = static_cast<const audio_filter_info *>(i.second);
                if(strcasecmp(i.first.c_str(), name) == 0){
                        filter->info = funcs;

                        af_result_code ret = funcs->init(cfg, &filter->state);
                        if(ret == AF_FAILURE) {
                                fprintf(stderr, "Unable to initialize filter: %s\n",
                                                name);
                        }
                        return ret;
                }
        }

        fprintf(stderr, "Unable to find capture filter: %s\n", name);
        return AF_FAILURE;
}

void audio_filter_destroy(struct audio_filter *state){
        if(!state->state)
                return;

        state->info->done(state->state);

        state->info = {};
        state->state = {};
}

af_result_code audio_filter(struct audio_filter *state, struct audio_frame **frame){
        return state->info->filter(state->state, frame);
}

af_result_code audio_filter_configure(struct audio_filter *state,
                int bps, int ch_count, int sample_rate)
{
        return state->info->configure(state->state, bps, ch_count, sample_rate);
}

void  audio_filter_get_configured_out(struct audio_filter *state,
                int *bps, int *ch_count, int *sample_rate)
{
        return state->info->get_configured_out(state->state, bps, ch_count, sample_rate);
}

void  audio_filter_get_configured_in(struct audio_filter *state,
                int *bps, int *ch_count, int *sample_rate)
{
        return state->info->get_configured_in(state->state, bps, ch_count, sample_rate);
}

