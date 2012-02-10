/*
 * FILE:    audio/playback/alsa.c
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


/*
 * Changes should use Safe ALSA API (http://0pointer.de/blog/projects/guide-to-sound-apis).
 *
 * Please, report all differencies from it here:
 * - used format SND_PCM_FORMAT_S24_LE
 * - used "default" device for arbitrary number of channels
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif
#include "debug.h"
#ifdef HAVE_ALSA
#include "audio/audio.h"
#include <alsa/asoundlib.h>
#include "audio/playback/alsa.h" 
#include "debug.h"
#include <stdlib.h>

struct state_alsa_playback {
        snd_pcm_t *handle;
        struct audio_frame frame;
};

int audio_play_alsa_reconfigure(void *state, int quant_samples, int channels,
                                int sample_rate)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        snd_pcm_hw_params_t *params;
        snd_pcm_format_t format;
        unsigned int val;
        int dir;
        int rc;
        snd_pcm_uframes_t frames;

        s->frame.bps = quant_samples / 8;
        s->frame.ch_count = channels;
        s->frame.sample_rate = sample_rate;


        /* Allocate a hardware parameters object. */
        snd_pcm_hw_params_alloca(&params);

        /* Fill it in with default values. */
        snd_pcm_hw_params_any(s->handle, params);

        /* Set the desired hardware parameters. */

        /* Interleaved mode */
        snd_pcm_hw_params_set_access(s->handle, params,
                        SND_PCM_ACCESS_RW_INTERLEAVED);

        switch(quant_samples) {
                case 8:
                        format = SND_PCM_FORMAT_U8;
                        break;
                case 16:
                        format = SND_PCM_FORMAT_S16_LE;
                        break;
                case 24:
                        format = SND_PCM_FORMAT_S24_LE;
                        break;
                case 32:
                        format = SND_PCM_FORMAT_S32_LE;
                        break;
                default:
                        fprintf(stderr, "[ALSA playback] Unsupported BPS for audio (%d).\n", quant_samples);
                        return FALSE;
        }
        /* Signed 16-bit little-endian format */
        snd_pcm_hw_params_set_format(s->handle, params,
                        format);

        /* Two channels (stereo) */
        snd_pcm_hw_params_set_channels(s->handle, params, channels);

        /* 44100 bits/second sampling rate (CD quality) */
        val = sample_rate;
        snd_pcm_hw_params_set_rate_near(s->handle, params,
                        &val, &dir);

        /* Set period size to 1 frame. */
        frames = 1;
        snd_pcm_hw_params_set_period_size_near(s->handle,
                        params, &frames, &dir);

        /* Write the parameters to the driver */
        rc = snd_pcm_hw_params(s->handle, params);
        if (rc < 0) {
                fprintf(stderr,
                        "unable to set hw parameters: %s\n",
                        snd_strerror(rc));
                return FALSE;
        }

        free(s->frame.data);

        s->frame.max_size = s->frame.bps * s->frame.ch_count * s->frame.sample_rate; // can hold up to 1 sec
        s->frame.data = (char*)malloc(s->frame.max_size);
        assert(s->frame.data != NULL);

        return TRUE;
}

void audio_play_alsa_help(void)
{
        void **hints;

        printf("\talsa : default ALSA device (same as \"alsa:default\")\n");
        snd_device_name_hint(-1, "pcm", &hints); 
        while(*hints != NULL) {
                char *tmp = strdup(*(char **) hints);
                char *save_ptr = NULL;
                char *name_part;
                char *name;
                char *desc;
                char *details;


                name_part = strtok_r(tmp + 4, "|", &save_ptr);
                desc = strtok_r(NULL, "|", &save_ptr);
                char *character;
                while((character = strchr(desc, '\n'))) {
                        *character = ' ';
                }
                name = strtok_r(name_part, ":", &save_ptr);
                details = strtok_r(NULL, ":", &save_ptr);
                if(details) {
                        char * index = strstr(details, "DEV");
			if(index) {
                                index += 4;
				printf("\talsa:%s:%s : %s\n", name, index, desc + 4);
			} else {
				printf("\talsa:%s : %s\n", name, desc + 4);
                        }
                } else {
                        printf("\talsa:%s : %s\n", name, desc + 4);
                }
                hints++;
                free(tmp);
        }
}

void * audio_play_alsa_init(char *cfg)
{
        int rc;
        struct state_alsa_playback *s;
        char *name;

        s = calloc(1, sizeof(struct state_alsa_playback));
        if(cfg)
                name = cfg;
        else
                name = "default";
        rc = snd_pcm_open(&s->handle, name,
                                            SND_PCM_STREAM_PLAYBACK, 0);


        if (rc < 0) {
                    fprintf(stderr, "unable to open pcm device: %s\n",
                                    snd_strerror(rc));
                    goto error;
        }
        
        return s;

error:
        free(s);
        return NULL;
}

struct audio_frame *audio_play_alsa_get_frame(void *state)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;

        return &s->frame;
}

void audio_play_alsa_put_frame(void *state, struct audio_frame *frame)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;
        int rc;
        int frames = frame->data_len / (frame->bps * frame->ch_count);

        rc = snd_pcm_writei(s->handle, frame->data, frames);
        if (rc == -EPIPE) {
                /* EPIPE means underrun */
                fprintf(stderr, "underrun occurred\n");
                snd_pcm_prepare(s->handle);
                /* duplicate last data into stream */
                snd_pcm_writei(s->handle, frame->data, frames);
                snd_pcm_writei(s->handle, frame->data, frames);
        } else if (rc < 0) {
                fprintf(stderr, "error from writei: %s\n",
                        snd_strerror(rc));
        }  else if (rc != (int)frames) {
                fprintf(stderr, "short write, write %d frames\n", rc);
        }
}

void audio_play_alsa_done(void *state)
{
        struct state_alsa_playback *s = (struct state_alsa_playback *) state;

        snd_pcm_drain(s->handle);
        snd_pcm_close(s->handle);
        free(s->frame.data);
        free(s);
}

#endif /* HAVE_ALSA */
