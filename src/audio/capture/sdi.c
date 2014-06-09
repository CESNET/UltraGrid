/*
 * FILE:    audio/capture/sdi.c
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "audio/audio.h" 
#include "audio/capture/sdi.h" 

#include "compat/platform_semaphore.h"
#include "debug.h"
#include "host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FRAME_NETWORK 0
#define FRAME_CAPTURE 1

struct state_sdi_capture {
        struct audio_frame audio_frame[2];
        pthread_mutex_t lock;
        pthread_cond_t  audio_frame_ready_cv;
};

void * sdi_capture_init(char *cfg)
{
        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Available vidcap audio devices:\n");
                sdi_capture_help("embedded");
                sdi_capture_help("AESEBU");
                sdi_capture_help("analog");
                printf("\t\twhere <index> is index of vidcap device "
                                "to be taken audio from.\n");
                return &audio_init_state_ok;
        }
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) calloc(1, sizeof(struct state_sdi_capture));
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->audio_frame_ready_cv, NULL);
        
        return s;
}

struct audio_frame * sdi_read(void *state)
{
        struct state_sdi_capture *s;
        int             rc = 0;
        struct timespec ts;
        struct timeval  tp;

        
        s = (struct state_sdi_capture *) state;

        gettimeofday(&tp, NULL);
        ts.tv_sec  = tp.tv_sec;
        ts.tv_nsec = tp.tv_usec * 1000;
        ts.tv_nsec += 100 * 1000 * 1000;
        // make it correct
        ts.tv_sec += ts.tv_nsec / 1000000000;
        ts.tv_nsec = ts.tv_nsec % 1000000000;


        pthread_mutex_lock(&s->lock);
        while (rc == 0 && s->audio_frame[FRAME_CAPTURE].data_len == 0) {
                rc = pthread_cond_timedwait(&s->audio_frame_ready_cv, &s->lock, &ts);
        }

        if (rc != 0) {
                pthread_mutex_unlock(&s->lock);
                return NULL;
        }

        // FRAME_NETWORK has been "consumed"
        s->audio_frame[FRAME_NETWORK].data_len = 0;
        // swap
        struct audio_frame tmp;
        memcpy(&tmp, &s->audio_frame[FRAME_CAPTURE], sizeof(struct audio_frame));
        memcpy(&s->audio_frame[FRAME_CAPTURE], &s->audio_frame[FRAME_NETWORK], sizeof(struct audio_frame));
        memcpy(&s->audio_frame[FRAME_NETWORK], &tmp, sizeof(struct audio_frame));
        pthread_mutex_unlock(&s->lock);

        return &s->audio_frame[FRAME_NETWORK];
}

void sdi_capture_done(void *state)
{
        struct state_sdi_capture *s;

        s = (struct state_sdi_capture *) state;
        for(int i = 0; i < 2; ++i) {
                free(s->audio_frame[i].data);
        }
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->audio_frame_ready_cv);
}

void sdi_capture_help(const char *driver_name)
{
        if(strcmp(driver_name, "embedded") == 0) {
                printf("\tembedded[:<index>] : SDI audio (if available)\n");
        } else if(strcmp(driver_name, "AESEBU") == 0) {
                printf("\tAESEBU[:<index>] : separately connected AES/EBU to a grabbing card (if available)\n");
        } else if(strcmp(driver_name, "analog") == 0) {
                printf("\tanalog[:<index>] : analog input of grabbing card (if available)\n");
        }
}

void sdi_capture_new_incoming_frame(void *state, struct audio_frame *frame)
{
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) state;

        /**
         * @todo figure out what if we get too many audio samples buffered -
         * perhaps drop it and report error
         */
        pthread_mutex_lock(&s->lock);

        if(
                        s->audio_frame[FRAME_CAPTURE].bps != frame->bps ||
                        s->audio_frame[FRAME_CAPTURE].ch_count != frame->ch_count ||
                        s->audio_frame[FRAME_CAPTURE].sample_rate != frame->sample_rate
          ) {
                s->audio_frame[FRAME_CAPTURE].bps = frame->bps;
                s->audio_frame[FRAME_CAPTURE].ch_count = frame->ch_count;
                s->audio_frame[FRAME_CAPTURE].sample_rate = frame->sample_rate;
                s->audio_frame[FRAME_CAPTURE].data_len = 0;
        }

        int needed_size = frame->data_len + s->audio_frame[FRAME_CAPTURE].data_len;
        if(needed_size > (int) s->audio_frame[FRAME_CAPTURE].max_size) {
                free(s->audio_frame[FRAME_CAPTURE].data);
                s->audio_frame[FRAME_CAPTURE].max_size = needed_size;
                s->audio_frame[FRAME_CAPTURE].data = malloc(needed_size);
        }
        memcpy(s->audio_frame[FRAME_CAPTURE].data + s->audio_frame[FRAME_CAPTURE].data_len,
                        frame->data, frame->data_len);
        s->audio_frame[FRAME_CAPTURE].data_len += frame->data_len;

        pthread_cond_signal(&s->audio_frame_ready_cv);
        pthread_mutex_unlock(&s->lock);
}

/* vim: set expandtab: sw=8 */
