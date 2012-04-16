/*
 * FILE:    audio/capture/alsa.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 *
 */
#include "config.h"

#ifdef HAVE_ALSA

#include "audio/audio.h"
#include "audio/capture/alsa.h" 
#include "debug.h"
#include <stdlib.h>
#include <string.h>
/* Use the newer ALSA API */
#define ALSA_PCM_NEW_HW_PARAMS_API

#include <alsa/asoundlib.h>

struct state_alsa_capture {
        snd_pcm_t *handle;
        struct audio_frame frame;

        snd_pcm_uframes_t frames;
};

void audio_play_alsa_help(const char *driver_name);
void audio_cap_alsa_help(const char *driver_name)
{
        audio_play_alsa_help(driver_name);
}

void * audio_cap_alsa_init(char *cfg)
{
        struct state_alsa_capture *s;
        int rc;
        snd_pcm_hw_params_t *params;
        unsigned int val;
        int dir;
        char *name = "default";

        s = malloc(sizeof(struct state_alsa_capture));

        s->frame.bps = 2;
        s->frame.sample_rate = 48000;
        s->frame.ch_count = 2;

        if(cfg && strlen(cfg) > 0) {
                name = cfg;
        }

        /* Open PCM device for recording (capture). */
        rc = snd_pcm_open(&s->handle, name,
                SND_PCM_STREAM_CAPTURE, 0);
        if (rc < 0) {
                fprintf(stderr, "unable to open pcm device: %s\n",
                        snd_strerror(rc));
                goto error;
        }

        /* Allocate a hardware parameters object. */
        snd_pcm_hw_params_alloca(&params);

        /* Fill it in with default values. */
        snd_pcm_hw_params_any(s->handle, params);

        /* Set the desired hardware parameters. */

        /* Interleaved mode */
        snd_pcm_hw_params_set_access(s->handle, params,
                SND_PCM_ACCESS_RW_INTERLEAVED);

        /* Signed 16-bit little-endian format */
        snd_pcm_hw_params_set_format(s->handle, params,
                SND_PCM_FORMAT_S16_LE);

        /* Two channels (stereo) */
        snd_pcm_hw_params_set_channels(s->handle, params, s->frame.ch_count);

        /* sampling rate (CD quality) */
        val = s->frame.sample_rate;
        snd_pcm_hw_params_set_rate_near(s->handle, params,
                &val, &dir);

        /* Set period size to 32 frames. */
        s->frames = 128;
        snd_pcm_hw_params_set_period_size_near(s->handle,
                params, &s->frames, &dir);

        /* Write the parameters to the driver */
        rc = snd_pcm_hw_params(s->handle, params);
        if (rc < 0) {
                fprintf(stderr, "unable to set hw parameters: %s\n",
                        snd_strerror(rc));
                goto error;
        }

        /* Use a buffer large enough to hold one period */
        snd_pcm_hw_params_get_period_size(params, &s->frames, &dir);
        s->frame.max_size = s->frames  * s->frame.ch_count * s->frame.bps;
        s->frame.data = (char *) malloc(s->frame.max_size);

        return s;

error:
        free(s);
        return NULL;
}

struct audio_frame *audio_cap_alsa_read(void *state)
{
        struct state_alsa_capture *s = (struct state_alsa_capture *) state;
        int rc;

        rc = snd_pcm_readi(s->handle, s->frame.data, s->frames);
        if (rc == -EPIPE) {
                /* EPIPE means overrun */
                fprintf(stderr, "overrun occurred\n");
                snd_pcm_prepare(s->handle);
        } else if (rc < 0) {
                fprintf(stderr, "error from read: %s\n", snd_strerror(rc));
        } else if (rc != (int)s->frames) {
                fprintf(stderr, "short read, read %d frames\n", rc);
        }

        if(rc > 0) {
                s->frame.data_len = rc * s->frame.bps * s->frame.ch_count;
                return &s->frame;
        } else {
                return NULL;
        }
}

void audio_cap_alsa_finish(void *state)
{
        UNUSED(state);
}

void audio_cap_alsa_done(void *state)
{
        struct state_alsa_capture *s = (struct state_alsa_capture *) state;

        snd_pcm_drain(s->handle);
        snd_pcm_close(s->handle);
        free(s->frame.data);
        free(s);
}

#endif /* HAVE_ALSA */
