/**
 * @file   audio/export.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2021 CESNET, z. s. p. o.
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

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "audio/audio.h"
#include "audio/utils.h"
#include "audio/wav_writer.h"
#include "debug.h"
#include "export.h"
#include "utils/misc.h" // ug_strerror
#include "utils/ring_buffer.h"

#define CACHE_SECONDS                   10

/*
 * we do not need to have possible stalls, so IO is performend in a separate thread
 */
static void *audio_export_thread(void *arg);
static bool configure(struct audio_export *s, struct audio_desc fmt);

struct audio_export {
        char *filename;
        struct wav_writer_file *wav;

        struct audio_desc saved_format;

        ring_buffer_t *ring;

        pthread_t thread_id;
        pthread_mutex_t lock;
        pthread_cond_t worker_cv;
        volatile bool new_work_ready;
        volatile bool worker_waiting;

        volatile bool should_exit_worker;
};

static void *audio_export_thread(void *arg)
{
        struct audio_export *s = arg;

        while(!s->should_exit_worker) {
                pthread_mutex_lock(&s->lock);
                while(!s->new_work_ready) {
                        s->worker_waiting = true;
                        pthread_cond_wait(&s->worker_cv, &s->lock);
                        s->worker_waiting = false;
                }

                int size = ring_get_current_size(s->ring);

                if(s->should_exit_worker && size == 0) {
                        pthread_mutex_unlock(&s->lock);
                        break;
                }

                char *data = malloc(size);
                assert(data);
                size_t res = ring_buffer_read(s->ring, data, size);
                assert(res == (size_t) size);
                assert(ring_get_current_size(s->ring) == 0);
                s->new_work_ready = false;

                pthread_mutex_unlock(&s->lock);

                const int sample_size = s->saved_format.bps * s->saved_format.ch_count;
                int rc = wav_writer_write(s->wav, size / sample_size, data);
                if (rc != 0) {
                        fprintf(stderr, "[Audio export] Problem writing audio samples: %s\n", ug_strerror(-rc));
                }

                free(data);
        }

        return NULL;
}

static bool configure(struct audio_export *s, struct audio_desc fmt) {
        s->saved_format = fmt;

        if ((s->wav = wav_writer_create(s->filename, fmt)) == NULL) {
                fprintf(stderr, "[Audio export] Error writting header!\n");
                return false;
        }

        s->ring = ring_buffer_init(CACHE_SECONDS * fmt.sample_rate * fmt.bps *
                        fmt.ch_count);

        return true;
}

bool audio_export_configure_raw(struct audio_export *s,
                int bps, int sample_rate, int ch_count)
{

        assert(s->saved_format.ch_count == 0 && "Audio export already configured");
        struct audio_desc fmt;

        fmt.codec = AC_PCM;
        fmt.bps = bps;
        fmt.sample_rate = sample_rate;
        fmt.ch_count = ch_count;

        bool res = configure(s, fmt);

        if(!res) {
                fprintf(stderr, "[Audio export] Configuration failed.\n");
                return false;
        }
        pthread_create(&s->thread_id, NULL, audio_export_thread, s);

        return true;
}

struct audio_export * audio_export_init(const char *filename)
{
        struct audio_export *s = calloc(1, sizeof(struct audio_export));
        if(!s) {
                return NULL;
        }

        unlink(filename);
        s->filename = strdup(filename);
        s->thread_id = 0;
        s->ring = NULL;

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->worker_cv, NULL);
        s->new_work_ready = false;
        s->worker_waiting = false;
        s->should_exit_worker = false;

        s->saved_format = (struct audio_desc) { 0, 0, 0, 0 };

        return s;
}

void audio_export_destroy(struct audio_export *s)
{
        if(s) {
                if(s->thread_id) {
                        pthread_mutex_lock(&s->lock);
                        s->should_exit_worker = true;
                        s->new_work_ready = true;
                        if(s->worker_waiting) {
                                pthread_cond_signal(&s->worker_cv);
                        }
                        pthread_mutex_unlock(&s->lock);
                        pthread_join(s->thread_id, NULL);
                }

                if (s->wav != NULL) {
                        wav_writer_close(s->wav);
                }
                if (s->ring != NULL) {
                        ring_buffer_destroy(s->ring);
                }
                pthread_cond_destroy(&s->worker_cv);
                pthread_mutex_destroy(&s->lock);
                free(s->filename);
                free(s);
        }
}

void audio_export_raw(struct audio_export *s, void *data, unsigned len){
        assert(s->saved_format.ch_count != 0 && "Export not configured");
        pthread_mutex_lock(&s->lock);
        ring_buffer_write(s->ring, data, len);
        s->new_work_ready = true;

        if(s->worker_waiting) {
                pthread_cond_signal(&s->worker_cv);
        }
        pthread_mutex_unlock(&s->lock);
}

void audio_export(struct audio_export *s, struct audio_frame *frame)
{
        if(!s) {
                return;
        }

        if(s->saved_format.ch_count == 0) {
                bool res;
                res = configure(s, audio_desc_from_frame(frame));
                if(!res) {
                        fprintf(stderr, "[Audio export] Configuration failed.\n");
                        return;
                }
                pthread_create(&s->thread_id, NULL, audio_export_thread, s);
        } else {
                if(!audio_desc_eq(s->saved_format,
                                        audio_desc_from_frame(frame))) {
                        return;
                }
        }

        audio_export_raw(s, frame->data, frame->data_len);
}

static int write_samples(char *out, const void **channels_data,
                int bps, int ch_count, int size, int start_sample)
{
        int written = 0;
        int in_offset = start_sample*bps;
        for(; written * bps * ch_count < size; written++){
                for(int ch = 0; ch < ch_count; ch++){
                        const char *in_ptr = ((const char *)channels_data[ch]) + in_offset;
                        memcpy(out, in_ptr, bps);
                        out += bps;
                }
                in_offset += bps;
        }

        return written;
}

void audio_export_raw_ch(struct audio_export *s,
                const void **channels_data, unsigned sample_count)
{
        const int ch_count = s->saved_format.ch_count;
        const int bps = s->saved_format.bps;
        assert(ch_count != 0 && "Export not configured");

        int len = s->saved_format.ch_count * s->saved_format.bps * sample_count;

        pthread_mutex_lock(&s->lock);
        void *ptr1;
        int size1;
        void *ptr2;
        int size2;
        int avail = ring_get_write_regions(s->ring, len, &ptr1, &size1, &ptr2, &size2);

        assert((size1 % (bps * ch_count)) == 0);
        assert(!ptr2 || (size2 % (bps * ch_count)) == 0);

        int written = write_samples(ptr1, channels_data, bps, ch_count, size1, 0);
        if(ptr2){
                write_samples(ptr2, channels_data, bps, ch_count, size2, written);
        }

        ring_advance_write_idx(s->ring, avail);

        s->new_work_ready = true;
        if(s->worker_waiting) {
                pthread_cond_signal(&s->worker_cv);
        }
        pthread_mutex_unlock(&s->lock);
}

