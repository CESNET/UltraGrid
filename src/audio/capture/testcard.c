/*
 * FILE:    audio/capture/testcard.h
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

#define MODULE_NAME "[Audio testcard] "

#include "audio/capture/testcard.h" 

#include "audio/audio.h" 

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif
#include "debug.h"
#include "host.h"
#include "tv.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

#define AUDIO_CAPTURE_TESTCARD_MAGIC 0xf4b3c9c9u

#define AUDIO_SAMPLE_RATE 48000
#define AUDIO_BPS 2
#define BUFFER_SEC 1
#define AUDIO_BUFFER_SIZE (AUDIO_SAMPLE_RATE * AUDIO_BPS * \
                (int) audio_capture_channels * BUFFER_SEC)

#define CHUNK 128 * 3 // has to be divisor of AUDIO_SAMLE_RATE

#define FREQUENCY 440
#define VOLUME 1

enum which_sample {
        TONE,
        SILENCE
};

struct state_audio_capture_testcard {
        uint32_t magic;

        struct audio_frame audio;

        double audio_remained,
               seconds_tone_played;
        short int *audio_tone, *audio_silence;

        struct timeval next_audio_time;

        enum which_sample current_sample;
        int samples_played;
};

void audio_cap_testcard_help(const char *driver_name)
{
        UNUSED(driver_name);
        printf("\ttestcard : Testing sound signal (sine at 440 Hz)\n");
}

void * audio_cap_testcard_init(char *cfg)
{
        struct state_audio_capture_testcard *s;
        int i;

        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Available testcard capture:\n");
                audio_cap_testcard_help(NULL);
                return NULL;
        }

        s = (struct state_audio_capture_testcard *) malloc(sizeof(struct state_audio_capture_testcard));
        printf(MODULE_NAME "Generating %d sec tone (%d Hz) / %d sec silence ", BUFFER_SEC, FREQUENCY, BUFFER_SEC);
        printf("(channels: %u; bps: %d; sample rate: %d; frames per packet: %d).\n", audio_capture_channels,
                        AUDIO_BPS, AUDIO_SAMPLE_RATE, CHUNK);
        s->magic = AUDIO_CAPTURE_TESTCARD_MAGIC;
        assert(s != 0);
        UNUSED(cfg);

        s->audio_silence = calloc(1, AUDIO_BUFFER_SIZE /* 1 sec */);
        
        s->audio_tone = calloc(1, AUDIO_BUFFER_SIZE /* 1 sec */);
        short int * data = (short int *) s->audio_tone;
        for( i=0; i < AUDIO_BUFFER_SIZE/2; i+=audio_capture_channels )
        {
                for (int channel = 0; channel < (int) audio_capture_channels; ++channel)
                        data[i + channel] = (float) sin( ((double)(i/audio_capture_channels)/((double)AUDIO_SAMPLE_RATE / FREQUENCY)) * M_PI * 2. ) * SHRT_MAX * VOLUME;
        }

        
        s->audio.bps = AUDIO_BPS;
        s->audio.ch_count = audio_capture_channels;
        s->audio.sample_rate = AUDIO_SAMPLE_RATE;
        s->audio.data_len = CHUNK * AUDIO_BPS * audio_capture_channels;

        s->current_sample = SILENCE;
        s->samples_played = 0;

        gettimeofday(&s->next_audio_time, NULL);

        return s;
}

struct audio_frame *audio_cap_testcard_read(void *state)
{
        struct state_audio_capture_testcard *s;
        s = (struct state_audio_capture_testcard *) state;
        struct timeval curr_time;

        gettimeofday(&curr_time, NULL);

        if(tv_gt(s->next_audio_time, curr_time)) {
                usleep(tv_diff_usec(s->next_audio_time, curr_time));
        } else {
                // we missed more than 2 "frame times", in that case, just drop the packages
                if (tv_diff_usec(curr_time, s->next_audio_time) > 2 * (1000 * 1000 * CHUNK / 48000)) {
                        s->next_audio_time = curr_time;
                        fprintf(stderr, MODULE_NAME "Warning: skipping some samples (late grab call).\n");
                }
        }

        tv_add_usec(&s->next_audio_time, 1000 * 1000 * CHUNK / 48000);

        if(s->current_sample == TONE) {
                s->audio.data = (char* )(s->audio_tone + s->samples_played * audio_capture_channels); // it is short so _not_ (* 2)
        } else {
                s->audio.data = (char *)(s->audio_silence + s->samples_played * audio_capture_channels);
        }

        s->samples_played += CHUNK;
        if(s->samples_played >= AUDIO_SAMPLE_RATE) {
                s->samples_played = 0;
                if(s->current_sample == TONE) {
                        s->current_sample = SILENCE;
                } else {
                        s->current_sample = TONE;
                }
        }

        return &s->audio;
}

void audio_cap_testcard_finish(void *state)
{
        UNUSED(state);
}

void audio_cap_testcard_done(void *state)
{
        struct state_audio_capture_testcard *s = (struct state_audio_capture_testcard *) state;

        assert(s->magic == AUDIO_CAPTURE_TESTCARD_MAGIC);

        free(s->audio_silence);
        free(s->audio_tone);

        free(s);
}

