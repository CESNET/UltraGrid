/**
 * @file   audio/echo.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2021 CESNET z.s.p.o.
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

#include "audio/utils.h"
#include "debug.h"
#include "echo.h"

#include <speex/speex_echo.h>

#include <stdlib.h>
#include <pthread.h>
#include "utils/ring_buffer.h"

#define SAMPLES_PER_FRAME (48 * 10)
#define FILTER_LENGTH (48 * 500)

struct echo_cancellation {
        SpeexEchoState *echo_state;

        ring_buffer_t *near_end_ringbuf;
        ring_buffer_t *far_end_ringbuf;

        struct audio_frame frame;

        bool drop_near;
        int overfill;

        pthread_mutex_t lock;
};

static void reconfigure_echo (struct echo_cancellation *s, int sample_rate, int bps);

static void reconfigure_echo (struct echo_cancellation *s, int sample_rate, int bps)
{
        UNUSED(bps);

        s->frame.bps = 2;
        s->frame.ch_count = 1;
        s->frame.sample_rate = sample_rate;

        ring_buffer_flush(s->far_end_ringbuf);
        ring_buffer_flush(s->near_end_ringbuf);

        speex_echo_ctl(s->echo_state, SPEEX_ECHO_SET_SAMPLING_RATE, &sample_rate); // should the 3rd parameter be int?
}

struct echo_cancellation * echo_cancellation_init(void)
{
        struct echo_cancellation *s = (struct echo_cancellation *) calloc(1, sizeof(struct echo_cancellation));

        s->echo_state = speex_echo_state_init(SAMPLES_PER_FRAME, FILTER_LENGTH);

        s->frame.data = NULL;
        s->frame.sample_rate = s->frame.bps = 0;
        pthread_mutex_init(&s->lock, NULL);

        const int ringbuf_sample_count = 2 << 15;
        const int bps = 2; //TODO: assuming bps to be 2

        s->far_end_ringbuf = ring_buffer_init(ringbuf_sample_count * bps);
        s->near_end_ringbuf = ring_buffer_init(ringbuf_sample_count * bps);

        s->frame.data = malloc(ringbuf_sample_count * bps);
        s->frame.max_size = ringbuf_sample_count * bps;

        printf("Echo cancellation initialized.\n");

        s->drop_near = true;
        s->overfill = 3000;

        return s;
}

void echo_cancellation_destroy(struct echo_cancellation *s)
{
        if(s->echo_state) {
                speex_echo_state_destroy(s->echo_state);  
        }
        ring_buffer_destroy(s->near_end_ringbuf);
        ring_buffer_destroy(s->far_end_ringbuf);
        free(s->frame.data);

        pthread_mutex_destroy(&s->lock);

        free(s);
}

void echo_play(struct echo_cancellation *s, struct audio_frame *frame)
{
        pthread_mutex_lock(&s->lock);

        if(frame->ch_count != 1) {
                static int prints = 0;
                if(prints++ % 100 == 0) {
                        fprintf(stderr, "Echo cancellation needs 1 played channel. Disabling echo cancellation.\n"
                                        "Use channel mapping and let only one channel played to enable this feature.\n");
                }
                pthread_mutex_unlock(&s->lock);
                return;
        }

        size_t samples = frame->data_len / frame->bps;
        size_t ringbuf_free_samples = ring_get_available_write_size(s->far_end_ringbuf) / 2;

        if(samples > ringbuf_free_samples){
                samples = ringbuf_free_samples;
                log_msg(LOG_LEVEL_WARNING, "Far end ringbuf overflow!\n");
        }


        if(frame->bps != 2) {
                char *tmp = malloc(samples * 2);
                change_bps(tmp, 2, frame->data, frame->bps, frame->data_len/* bytes */);

                ring_buffer_write(s->far_end_ringbuf, tmp, samples * 2);

                free(tmp);
        } else {
                ring_buffer_write(s->far_end_ringbuf, frame->data, samples * 2);
        }

        if(s->drop_near){
                printf("Dropping near end buffer\n");
                ring_buffer_flush(s->near_end_ringbuf);
                s->drop_near = false;
        }

        if(s->overfill){
                int to_fill = s->overfill > samples ? samples : s->overfill;
                ring_buffer_write(s->far_end_ringbuf, frame->data, to_fill * 2);
                s->overfill -= to_fill;
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

        size_t samples = data_len / 2;
        size_t ringbuf_free_samples = ring_get_available_write_size(s->near_end_ringbuf) / 2;
        
        if(samples > ringbuf_free_samples){
                samples = ringbuf_free_samples;
                log_msg(LOG_LEVEL_WARNING, "Near end ringbuf overflow\n");
        }

        ring_buffer_write(s->near_end_ringbuf, data, samples * 2);

        free(tmp);


        size_t near_end_samples = ring_get_current_size(s->near_end_ringbuf) / 2;
        size_t far_end_samples = ring_get_current_size(s->far_end_ringbuf) / 2;
        size_t available_samples = (near_end_samples > far_end_samples) ? far_end_samples : near_end_samples;

        if(true || available_samples < near_end_samples){
                printf("Limited by far end (%lu near, %lu far)\n", near_end_samples, far_end_samples);
        }

        size_t frames_to_process = available_samples / SAMPLES_PER_FRAME;
        if(!frames_to_process){
                pthread_mutex_unlock(&s->lock);
                return NULL;
        }

        size_t out_size = frames_to_process * SAMPLES_PER_FRAME * 2;
        assert(s->frame.max_size >= out_size);
        s->frame.data_len = out_size;

        res = &s->frame;

        spx_int16_t *out_ptr = (spx_int16_t *)(void *) s->frame.data;
        for(size_t i = 0; i < frames_to_process; i++){
                spx_int16_t near_arr[SAMPLES_PER_FRAME];
                spx_int16_t far_arr[SAMPLES_PER_FRAME];

                ring_buffer_read(s->near_end_ringbuf, near_arr, SAMPLES_PER_FRAME * 2);
                ring_buffer_read(s->far_end_ringbuf, far_arr, SAMPLES_PER_FRAME * 2);
                available_samples -= SAMPLES_PER_FRAME;

                speex_echo_cancellation(s->echo_state, near_arr, far_arr, out_ptr); 

                out_ptr += SAMPLES_PER_FRAME;
        }

        pthread_mutex_unlock(&s->lock);

        return res;
}

