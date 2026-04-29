/**
 * @file   audio/filter/silence.c
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023-2026 CESNET, zájmové sdružení právnických osob
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

#include <assert.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#include "audio/audio_filter.h"
#include "audio/types.h"
#include "audio/utils.h"         // for remux_channel
#include "compat/c23.h"          // IWYU pragma: keep
#include "debug.h"               // for LOG_LEVEL_WARNING, MSG
#include "lib_common.h"
#include "utils/color_out.h"

struct module;

#define MOD_NAME "[af/silence] "

enum {
        MAX_CHANNELS = 128,
};

struct state_silence {
        struct audio_frame frame;
        bool silence_channels[MAX_CHANNELS];
        bool silence_all;
};

static void
usage()
{
        color_printf("Audio capture " TBOLD(
            "silence") " allows muting individual channels.\n\n");
        color_printf("Usage:\n");
        color_printf("\t" TBOLD("--audio-filter silence[:<idx1>,<idx2>]\n\n"));
        color_printf("(indexed from zero)\n");
        color_printf("If no index is given, all channels will be muted.\n");
}

static enum af_result_code
init(struct module *parent, const char *cfg, void **state)
{
        (void) parent;
        if (strcmp(cfg, "help") == 0) {
                usage();
                return AF_HELP_SHOWN;
        }
        struct state_silence *s = calloc(1, sizeof *s);
        s->silence_all = true;

        const size_t len = strlen(cfg) + 1;
        char         fmt[len];
        strncpy(fmt, cfg, len);
        char *tmp     = fmt;
        char *item    = NULL;
        char *end_ptr = NULL;
        while ((item = strtok_r(tmp, ",", &end_ptr)) != NULL) {
                long val = strtol(item, nullptr, 0);
                assert(val < MAX_CHANNELS);
                s->silence_channels[val] = true;
                s->silence_all = false;
                tmp = NULL;
        }
        *state = s;
        return AF_OK;
}

static enum af_result_code
configure(void *state, int in_bps, int in_ch_count, int in_sample_rate)
{
        struct state_silence *s = state;

        s->frame.bps         = in_bps;
        s->frame.ch_count    = in_ch_count;
        s->frame.sample_rate = in_sample_rate;
        s->frame.max_size    = in_bps * in_ch_count * in_sample_rate;
        s->frame.data        = calloc(1, s->frame.max_size);
        return AF_OK;
}

static void
done(void *state)
{
        free(state);
}

static void
get_configured_desc(void *state, int *bps, int *ch_count, int *sample_rate)
{
        struct state_silence *s = state;

        *bps         = s->frame.bps;
        *ch_count    = s->frame.ch_count;
        *sample_rate = s->frame.sample_rate;
}

static enum af_result_code
filter(void *state, const struct audio_frame **frame)
{
        struct state_silence *s = state;

        s->frame.data_len = (*frame)->data_len;
        if (s->frame.data_len > s->frame.max_size) {
                MSG(WARNING, "Overflow!");
                s->frame.data_len = s->frame.max_size;
        }

        if (s->silence_all) { // already done
                *frame = &s->frame;
                return AF_OK;
        }

        for (int i = 0; i < (*frame)->ch_count; ++i) {
                if (s->silence_channels[i]) {
                        continue;
                }
                remux_channel(s->frame.data, (*frame)->data, s->frame.bps,
                              s->frame.data_len, s->frame.ch_count,
                              s->frame.ch_count, i, i);
        }
        *frame = &s->frame;

        return AF_OK;
}

static const struct audio_filter_info audio_filter_silence = {
        .name               = "silence",
        .init               = init,
        .done               = done,
        .configure          = configure,
        .get_configured_in  = get_configured_desc,
        .get_configured_out = get_configured_desc,
        .filter             = filter,
};

REGISTER_MODULE(silence, &audio_filter_silence, LIBRARY_CLASS_AUDIO_FILTER,
                AUDIO_FILTER_ABI_VERSION);
