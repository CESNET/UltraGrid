/*
 * FILE:    audio/utils.c
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

#include "echo.h"

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "debug.h"

#ifdef HAVE_SPEEX
#include "speex/speex_echo.h"
#endif /* HAVE_SPEEX */

#include <stdlib.h>
#include <pthread.h>
#include "utils/ring_buffer.h"

#define SAMPLES_PER_FRAME (48 * 10)
#define FILTER_LENGTH (48 * 500)

struct echo_cancellation {
        SpeexEchoState *echo_state;

        char *near_end_residual;
        int near_end_residual_size;
        ring_buffer_t *far_end;

        struct audio_frame frame;

        pthread_mutex_t lock;
};

static void reconfigure_echo (struct echo_cancellation *s, int sample_rate, int bps);

static void reconfigure_echo (struct echo_cancellation *s, int sample_rate, int bps)
{
        UNUSED(bps);

        s->frame.bps = 2;
        s->frame.ch_count = 1;
        s->frame.sample_rate = sample_rate;
        s->frame.max_size = s->frame.data_len = 0;
        free(s->frame.data);
        s->frame.data = NULL;

        free(s->near_end_residual);
        ring_buffer_destroy(s->far_end);
        s->near_end_residual = NULL;
        s->near_end_residual_size = 0;
        s->far_end = ring_buffer_init(sample_rate * 2 / 2); // 0.5 sec

        speex_echo_ctl(s->echo_state, SPEEX_ECHO_SET_SAMPLING_RATE, &sample_rate); // should the 3rd parameter be int?
}

struct echo_cancellation * echo_cancellation_init(void)
{
        struct echo_cancellation *s = (struct echo_cancellation *) malloc(sizeof(struct echo_cancellation));

        s->echo_state = speex_echo_state_init(SAMPLES_PER_FRAME, FILTER_LENGTH);

        s->frame.data = NULL;
        s->frame.sample_rate = s->frame.bps = 0;
        pthread_mutex_init(&s->lock, NULL);

        printf("Echo cancellation initialized.\n");

        return s;
}

void echo_cancellation_destroy(struct echo_cancellation *s)
{
        if(s->echo_state) {
                speex_echo_state_destroy(s->echo_state);  
        }
        ring_buffer_destroy(s->far_end);
        free(s->near_end_residual);

        pthread_mutex_destroy(&s->lock);

        free(s);
}

void echo_play(struct echo_cancellation *s, struct audio_frame *frame)
{
        pthread_mutex_lock(&s->lock);

        if(!s->far_end) { // near end hasn't initialized yet
                pthread_mutex_unlock(&s->lock);
                return;
        }

        if(frame->ch_count != 1) {
                static int prints = 0;
                if(prints++ % 100 == 0) {
                        fprintf(stderr, "Echo cancellation needs 1 played channel. Disabling echo cancellation.\n"
                                        "Use channel mapping and let only one channel played to enable this feature.\n");
                }
                pthread_mutex_unlock(&s->lock);
                return;
        }

        if(frame->bps != 2) {
                char *tmp = malloc(frame->data_len / frame->bps * 2);
                change_bps(tmp, 2, frame->data, frame->bps, frame->data_len/* bytes */);
                ring_buffer_write(s->far_end, tmp, frame->data_len / frame->bps * 2);
                free(tmp);
        } else {
                ring_buffer_write(s->far_end, frame->data, frame->data_len);
        }


        pthread_mutex_unlock(&s->lock);
}

struct audio_frame * echo_cancel(struct echo_cancellation *s, struct audio_frame *frame)
{
        struct audio_frame *res;

        pthread_mutex_lock(&s->lock);

        if(frame->ch_count != 1) {
                static int prints = 0;
                if(prints++ % 100 == 0)
                        fprintf(stderr, "Echo cancellation needs 1 captured channel. Disabling echo cancellation.\n"
                                        "Use '--audio-capture-channels 1' parameter to capture single channel.\n");
                pthread_mutex_unlock(&s->lock);
                return frame;
        }


        if(frame->sample_rate != s->frame.sample_rate ||
                        frame->bps != s->frame.bps) {
                reconfigure_echo(s, frame->sample_rate, frame->bps);
        }

        char *data;
        char *tmp;
        int data_len;

        if(frame->bps != 2) {
                data_len = frame->data_len / frame->bps * 2;
                data = tmp = malloc(data_len);
                change_bps(tmp, 2, frame->data, frame->bps, frame->data_len/* bytes */);
        } else {
                tmp = NULL;
                data = frame->data;
                data_len = frame->data_len;
        }
        
        const int chunk_size = SAMPLES_PER_FRAME * 2 /* BPS */;
        const int rounded_data_len = (s->near_end_residual_size + data_len) / chunk_size * chunk_size;

        if(rounded_data_len) {
                char *data_to_write = malloc(rounded_data_len);
                char *far_end_tmp = malloc(chunk_size);

                memcpy(data_to_write, s->near_end_residual, s->near_end_residual_size);
                memcpy(data_to_write + s->near_end_residual_size, 
                                data,
                                rounded_data_len - s->near_end_residual_size);

                free(s->near_end_residual);
                s->near_end_residual_size = data_len + s->near_end_residual_size - rounded_data_len;
                s->near_end_residual = malloc(s->near_end_residual_size);
                memcpy(s->near_end_residual, data + data_len - s->near_end_residual_size, s->near_end_residual_size);

                free(s->frame.data);
                s->frame.data = malloc(rounded_data_len);
                s->frame.max_size = rounded_data_len;
                s->frame.data_len = 0;

                const spx_int16_t *near_ptr = (spx_int16_t *)(void *) data_to_write;
                spx_int16_t *out_ptr = (spx_int16_t *)(void *) s->frame.data;

                int read_len_far;
                read_len_far = ring_buffer_read(s->far_end, far_end_tmp, chunk_size);
                while((read_len_far == chunk_size) && s->frame.data_len < rounded_data_len)  {
                        speex_echo_cancellation(s->echo_state, near_ptr,
                                        (spx_int16_t *)(void *) far_end_tmp,
                                        out_ptr);

                        read_len_far = ring_buffer_read(s->far_end, far_end_tmp, chunk_size);
                        near_ptr += chunk_size;
                        out_ptr += chunk_size;
                        s->frame.data_len += chunk_size;
                }
               
                if(s->frame.data_len < rounded_data_len) {
                        memcpy(out_ptr, near_ptr, rounded_data_len - s->frame.data_len);
                }

                free(data_to_write);
                free(far_end_tmp);

                s->frame.data_len = rounded_data_len;

                pthread_mutex_unlock(&s->lock);

                res = &s->frame;
        } else {
                if(!s->near_end_residual) {
                        s->near_end_residual_size = frame->data_len;
                        s->near_end_residual = malloc(frame->data_len);
                        memcpy(s->near_end_residual, frame->data, frame->data_len);
                } else {
                        s->near_end_residual_size += frame->data_len;
                        s->near_end_residual = realloc(s->near_end_residual, s->near_end_residual_size);
                        memcpy(s->near_end_residual + (s->near_end_residual_size - frame->data_len),
                                        frame->data, frame->data_len);
                }

                res = NULL;
        }

        free(tmp);

        pthread_mutex_unlock(&s->lock);

        return res;
}

