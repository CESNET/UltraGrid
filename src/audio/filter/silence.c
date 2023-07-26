/**
 * @file   audio/filter/silence.c
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET, z. s. p. o.
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
#include "lib_common.h"
#include "utils/color_out.h"

enum {
        MAX_CHANNELS = 16,
};

struct state_silence {
        struct audio_desc desc;
        size_t silence_channels[MAX_CHANNELS];
        int silence_channels_count;
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
        if (strlen(cfg) > 0) {
                cfg++;
        }
        if (strcmp(cfg, "help") == 0) {
                usage();
                return AF_HELP_SHOWN;
        }
        struct state_silence *s = calloc(1, sizeof *s);

        const size_t len = strlen(cfg) + 1;
        char         fmt[len];
        strncpy(fmt, cfg, len);
        char *tmp     = fmt;
        char *item    = NULL;
        char *end_ptr = NULL;
        while ((item = strtok_r(tmp, ",", &end_ptr)) != NULL) {
                assert(s->silence_channels_count < MAX_CHANNELS - 1);
                s->silence_channels[s->silence_channels_count++] =
                    strtol(item, NULL, 10);
                tmp = NULL;
        }
        *state = s;
        return AF_OK;
}

static enum af_result_code
configure(void *state, int in_bps, int in_ch_count, int in_sample_rate)
{
        struct state_silence *s = state;

        s->desc.bps         = in_bps;
        s->desc.ch_count    = in_ch_count;
        s->desc.sample_rate = in_sample_rate;
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

        *bps         = s->desc.bps;
        *ch_count    = s->desc.ch_count;
        *sample_rate = s->desc.sample_rate;
}

static enum af_result_code
filter(void *state, struct audio_frame **frame)
{
        struct state_silence *s = state;

        if (s->silence_channels_count == 0) {
                memset((*frame)->data, 0, (*frame)->data_len);
                return AF_OK;
        }

        const int frame_size = s->desc.bps * s->desc.ch_count;
        for (int i = 0; i < s->silence_channels_count; ++i) {
                if (s->silence_channels[i] >= (size_t) (*frame)->ch_count) {
                        continue;
                }
                char *ptr =
                    (*frame)->data + s->silence_channels[i] * (*frame)->bps;
                for (int j = 0; j < (*frame)->data_len; j += frame_size) {
                        memset(ptr, 0, (*frame)->bps);
                        ptr += frame_size;
                }
        }

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
