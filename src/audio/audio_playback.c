/**
 * @file   audio/audio_playback.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2025 CESNET
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

#include "audio/audio_playback.h"

#include <stdio.h>          // for printf
#include <stdlib.h>         // for free, calloc
#include <string.h>         // for strncpy
#include <strings.h>        // for strcasecmp

#include "audio/types.h"    // for audio_frame, AC_PCM, audio_desc
#include "debug.h"          // for log_msg, LOG_LEVEL_ERROR, LOG_LEVEL_INFO
#include "host.h"           // for INIT_NOERR
#include "lib_common.h"     // for library_class, list_modules, load_library
#include "tv.h"             // for tv_diff
#include "utils/misc.h"     // for fmt_number_with_delim
#include "video_display.h"  // for DISPLAY_FLAG_AUDIO_AESEBU, DISPLAY_FLAG_A...

struct state_audio_playback {
        char name[128];
        const struct audio_playback_info *funcs;
        void *state;

        struct timeval t0;
        long long int samples_played;
};

void audio_playback_help(bool full)
{
        printf("Available audio playback devices:\n");
        list_modules(LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION, full);
}

int
audio_playback_init(const char *device, const struct audio_playback_opts *opts,
                    struct state_audio_playback **state)
{
        struct state_audio_playback *s= calloc(1, sizeof(*s));
        gettimeofday(&s->t0, NULL);
        s->funcs = load_library(device, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

        if (s->funcs == NULL) {
                log_msg(LOG_LEVEL_ERROR, "Unknown audio playback driver: %s\n", device);
                goto error;
        }

        strncpy(s->name, device, sizeof s->name - 1);
        s->state = s->funcs->init(opts);

        if(!s->state) {
                log_msg(LOG_LEVEL_ERROR, "Error initializing audio playback.\n");
                goto error;
        }

        if (s->state == INIT_NOERR) {
                free(s);
                return 1;
        }

        *state = s;
        return 0;

error:
        free(s);
        return -1;
}

struct state_audio_playback *audio_playback_init_null_device(void)
{
        const struct audio_playback_opts opts = { 0 };
        struct state_audio_playback *device = NULL;
        int ret = audio_playback_init("none", &opts, &device);
        if (ret != 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize null audio playback: %d\n", ret);
        }

        return device;
}

void audio_playback_done(struct state_audio_playback *s)
{
        if (!s) {
                return;
        }

        if (s->samples_played > 0) {
                struct timeval t1;
                gettimeofday(&t1, NULL);

                log_msg(LOG_LEVEL_INFO,
                        "Played %s audio samples in %.2f seconds (%f samples "
                        "per second).\n",
                        fmt_number_with_delim(s->samples_played),
                        tv_diff(t1, s->t0),
                        s->samples_played / tv_diff(t1, s->t0));
        }

        s->funcs->done(s->state);
        free(s);
}

unsigned int audio_playback_get_display_flags(struct state_audio_playback *s)
{
        if(!s)
                return 0u;

        if (strcasecmp(s->name, "embedded") == 0) {
                return DISPLAY_FLAG_AUDIO_EMBEDDED;
        } else if (strcasecmp(s->name, "AESEBU") == 0) {
                return DISPLAY_FLAG_AUDIO_AESEBU;
        } else if (strcasecmp(s->name, "analog") == 0) {
                return DISPLAY_FLAG_AUDIO_ANALOG;
        } else  {
                return 0u;
        }
}

void audio_playback_put_frame(struct state_audio_playback *s, const struct audio_frame *frame)
{
        s->samples_played += frame->data_len / frame->ch_count / frame->bps;
        s->funcs->write(s->state, frame);
}

bool audio_playback_ctl(struct state_audio_playback *s, int request, void *data, size_t *len)
{
        return s->funcs->ctl(s->state, request, data, len);
}

bool audio_playback_reconfigure(struct state_audio_playback *s, int quant_samples, int channels,
                int sample_rate)
{
        return s->funcs->reconfigure(s->state, (struct audio_desc){quant_samples / 8, sample_rate, channels, AC_PCM});
}

void  *audio_playback_get_state_pointer(struct state_audio_playback *s)
{
        return s->state;
}

/* vim: set expandtab sw=8: */

