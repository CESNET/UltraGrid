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

#define SAMPLES_PER_FRAME (2 << 8) //about 10ms at 48kHz, power of two for easy FFT
#define FILTER_LENGTH (48 * 500)

#define MOD_NAME "[Echo cancel] "

struct echo_cancellation {
        SpeexEchoState *echo_state;

        ring_buffer_t *near_end_ringbuf;
        ring_buffer_t *far_end_ringbuf;

        struct audio_frame frame;

        int prefill;
        bool before_first_near_sample;

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

        const int ringbuf_sample_count = 2 << 15; //should be divisable by SAMPLES_PER_FRAME
        const int bps = 2; //TODO: assuming bps to be 2

        s->far_end_ringbuf = ring_buffer_init(ringbuf_sample_count * bps);
        s->near_end_ringbuf = ring_buffer_init(ringbuf_sample_count * bps);

        s->frame.data = malloc(ringbuf_sample_count * bps);
        s->frame.max_size = ringbuf_sample_count * bps;

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Echo cancellation initialized.\n");

        s->prefill = 0;
        s->before_first_near_sample = true;

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
                        error_msg(MOD_NAME "Echo cancellation needs 1 played channel. Disabling echo cancellation.\n"
                                        "Use channel mapping and let only one channel played to enable this feature.\n");
                }
                pthread_mutex_unlock(&s->lock);
                return;
        }

        if(s->prefill){
                int target = (s->prefill / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME;
                int current = ring_get_current_size(s->far_end_ringbuf);
                //buffer can contain small remainder (<SAMPLES_PER_FRAME)
                int to_fill = target - current;
                s->prefill -= target;
                if(to_fill < 0){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Pre fill requested to %d, but the buffer is already %d!\n", target, current);
                } else {
                        ring_advance_write_idx(s->far_end_ringbuf, to_fill);
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Pre filling far end with %d samples\n", to_fill);
                }
        }

        size_t samples = frame->data_len / frame->bps;
        size_t ringbuf_free_samples = ring_get_available_write_size(s->far_end_ringbuf) / 2;

        if(samples > ringbuf_free_samples){
                samples = ringbuf_free_samples;
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Far end ringbuf overflow!\n");
        }

        if(frame->bps != 2) {
                void *ptr1;
                int size1;
                void *ptr2;
                int size2;
                ring_get_write_regions(s->far_end_ringbuf, samples * 2,
                                &ptr1, &size1, &ptr2, &size2);

                assert(size1 % 2 == 0);
                int in_bytes1 = (size1 / 2) * frame->bps;
                change_bps(ptr1, 2, frame->data, frame->bps, in_bytes1);
                if(ptr2){
                        change_bps(ptr2, 2, frame->data + in_bytes1, frame->bps, frame->data_len - in_bytes1);
                }

                ring_advance_write_idx(s->far_end_ringbuf, samples * 2);
        } else {
                ring_buffer_write(s->far_end_ringbuf, frame->data, samples * 2);
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
                        error_msg(MOD_NAME "Echo cancellation needs 1 captured channel. Disabling echo cancellation.\n"
                                        "Use '--audio-capture-channels 1' parameter to capture single channel.\n");
                pthread_mutex_unlock(&s->lock);
                return frame;
        }


        if(frame->sample_rate != s->frame.sample_rate ||
                        frame->bps != s->frame.bps) {
                reconfigure_echo(s, frame->sample_rate, frame->bps);
        }

        size_t in_frame_samples = frame->data_len / frame->bps;

        size_t ringbuf_free_samples = ring_get_available_write_size(s->near_end_ringbuf) / 2;
        if(in_frame_samples > ringbuf_free_samples){
                in_frame_samples = ringbuf_free_samples;
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Near end ringbuf overflow\n");
        }

        if(s->before_first_near_sample){
                /* It is possible that the capture thread starts late, which
                 * could create an unwanted delay between far and near ends.
                 * To partialy protect against this, drop the contents of far
                 * end buffer, when the very first near end samples arrive.
                 *
                 * This does not however protect against random capture thread
                 * freezes or dropouts.
                 */
                int current = ring_get_current_size(s->far_end_ringbuf);
                //drop only whole frames
                current = (current / SAMPLES_PER_FRAME) * SAMPLES_PER_FRAME;
                ring_advance_read_idx(s->far_end_ringbuf, current);

                s->before_first_near_sample = false;
        }

        if(frame->bps != 2){
                //Need to change bps, put whole incoming frame into ringbuf
                void *ptr1;
                int size1;
                void *ptr2;
                int size2;
                ring_get_write_regions(s->near_end_ringbuf, in_frame_samples * 2,
                                &ptr1, &size1, &ptr2, &size2);

                int in_bytes1 = (size1 / 2) * frame->bps;
                change_bps(ptr1, 2, frame->data, frame->bps, in_bytes1);
                if(ptr2){
                        change_bps(ptr2, 2, frame->data + in_bytes1, frame->bps, frame->data_len - in_bytes1);
                }
                ring_advance_write_idx(s->near_end_ringbuf, in_frame_samples * 2);
        } else {
                ring_buffer_write(s->near_end_ringbuf, frame->data, frame->data_len);
        }

        size_t near_end_samples = ring_get_current_size(s->near_end_ringbuf) / 2;
        size_t far_end_samples = ring_get_current_size(s->far_end_ringbuf) / 2;

        if(far_end_samples < near_end_samples){
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Not enough far end samples (%lu near, %lu far)\n", near_end_samples, far_end_samples);

                //The delay between far end and near end will always be at least
                //recorded frame length
                s->prefill = in_frame_samples;
        }

        size_t frames_to_process = near_end_samples / SAMPLES_PER_FRAME;
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

                if(far_end_samples >= SAMPLES_PER_FRAME){
                        ring_buffer_read(s->far_end_ringbuf, far_arr, SAMPLES_PER_FRAME * 2);
                        ring_buffer_read(s->near_end_ringbuf, near_arr, SAMPLES_PER_FRAME * 2);

                        speex_echo_cancellation(s->echo_state, near_arr, far_arr, out_ptr); 
                        far_end_samples -= SAMPLES_PER_FRAME;
                } else {
                        ring_buffer_read(s->near_end_ringbuf, out_ptr, SAMPLES_PER_FRAME * 2);
                }

                out_ptr += SAMPLES_PER_FRAME;
        }

        pthread_mutex_unlock(&s->lock);

        return res;
}

