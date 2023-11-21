/**
 * @file   audio/playback/dummy.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2023 CESNET, z. s. p. o.
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

#include <inttypes.h>
#include <stdlib.h>
#include <string.h>

#include "audio/audio_playback.h"
#include "audio/types.h"
#include "debug.h"
#include "lib_common.h"
#include "types.h"
#include "utils/color_out.h"

#define MOD_NAME "[dummy a.] "

enum {
        NDEF_TS = -1,
};

struct state_dummy_aplay {
        bool debug;
        int64_t first_ts;
        int64_t last_ts;
};

static void audio_play_dummy_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = NULL;
        *count = 0;
}

static void audio_play_dummy_help(void)
{
        color_printf(TBOLD("dummy") " audio playback\n\n");
        color_printf("Usage:\n\t" TBOLD(TRED("-r dummy") "[:debug]") "\n");
        color_printf("\nwhere:\n");
        color_printf("\t" TBOLD("debug") " - audio print frame TS + len\n");
}

static void * audio_play_dummy_init(const char *cfg)
{
        struct state_dummy_aplay *s = calloc(1, sizeof *s);
        if (strcmp(cfg, "debug") == 0) {
                s->debug = true;
        } else if (strcmp(cfg, "help") == 0) {
                audio_play_dummy_help();
                return INIT_NOERR;
        } else if (strlen(cfg) > 0) {
                MSG(ERROR, "Wrong option: %s\n", cfg);
                free(s);
                return NULL;
        }

        s->first_ts = NDEF_TS;
        return s;
}

static void audio_play_dummy_put_frame(void *state, const struct audio_frame *f)
{
        struct state_dummy_aplay *s = state;
        if (!s->debug) {
                return;
        }
        if (s->first_ts == NDEF_TS) {
                s->first_ts = f->timestamp;
        }

        const long long frm_beg_off = f->timestamp - s->first_ts;
        const long long frm_last_off = f->timestamp - s->last_ts;
        printf(MOD_NAME "Received TS: %10" PRId64
                        ", offset: %6lld, sample: %6lld (+%lld), len: %d\n",
               f->timestamp, frm_beg_off, frm_beg_off * f->sample_rate / kHz90,
               frm_last_off * f->sample_rate / kHz90,
               f->data_len / f->bps / f->ch_count);
        s->last_ts = f->timestamp;
}

static void audio_play_dummy_done(void *state)
{
        free(state);
}

static bool audio_play_dummy_ctl(void *state, int request, void *data, size_t *len)
{
        UNUSED(state), UNUSED(data), UNUSED(len);
        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                return true; // we accept any format
        default:
                return false;
        }
}

static bool audio_play_dummy_reconfigure(void *state, struct audio_desc desc)
{
        UNUSED(state), UNUSED(desc);
        return true;
}

static const struct audio_playback_info aplay_dummy_info = {
        audio_play_dummy_probe,
        audio_play_dummy_init,
        audio_play_dummy_put_frame,
        audio_play_dummy_ctl,
        audio_play_dummy_reconfigure,
        audio_play_dummy_done
};

REGISTER_MODULE(dummy, &aplay_dummy_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

