/*
 * FILE:    audio/export.c
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
 * File format is taken from:
 * http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
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
#include "debug.h"
#include "export.h"
#include "utils/ring_buffer.h"

/* Chunk size: 4 + 24 + (8 + M * Nc * Ns + (0 or 1)) */
#define CK_MASTER_SIZE_OFFSET            4

#define NCHANNELS_OFFSET                 22 /* Nc */
#define NSAMPLES_PER_SEC_OFFSET          24 /* F */
#define NAVG_BYTES_PER_SEC_OFFSET        28 /* F * M * Nc */
#define NBLOCK_ALIGN_OFFSET              32 /* M * Nc */
#define NBITS_PER_SAMPLE                 34 /* rounds up to 8 * M */


#define CK_DATA_SIZE_OFFSET              40 /*  M * Nc * Ns */

#define DATA_OFFSET                      44


#define CACHE_SECONDS                   10

/*
 * we do not need to have possible stalls, so IO is performend in a separate thread
 */
static void *audio_export_thread(void *arg);
static bool configure(struct audio_export *s, struct audio_desc fmt);
static void finalize(struct audio_export *s);

struct audio_export {
        char *filename;
        FILE *output;

        struct audio_desc saved_format;
        uint32_t total;

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
                res = fwrite(data, sample_size, size / sample_size, s->output);
                s->total += res;
                if(res != (size_t) size / sample_size) {
                        fprintf(stderr, "[Audio export] Problem writing audio samples.\n");
                }

                free(data);
        }

        return NULL;
}

static bool configure(struct audio_export *s, struct audio_desc fmt) {
        size_t res;
        s->saved_format = fmt;

        s->output = fopen(s->filename, "wb+");
        if(!s->output) {
                goto open_err;
        }

        res = fwrite("RIFF", 4, 1, s->output);
        if(res != 1) {
                goto file_err;
        }
        // chunk size - to be added
        res = fwrite("    ", 4, 1, s->output);
        if(res != 1) {
                goto file_err;
        }
        res = fwrite("WAVE", 4, 1, s->output);
        if(res != 1) {
                goto file_err;
        }
        res = fwrite("fmt ", 4, 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        uint32_t chunk_size = 16;
        res = fwrite(&chunk_size, sizeof(chunk_size), 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        uint16_t wave_format_pcm = 0x0001;
        res = fwrite(&wave_format_pcm, sizeof(wave_format_pcm), 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        uint16_t channels = fmt.ch_count;
        res = fwrite(&channels, sizeof(channels), 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        uint32_t sample_rate = fmt.sample_rate;
        res = fwrite(&sample_rate, sizeof(sample_rate), 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        uint32_t avg_bytes_per_sec = fmt.sample_rate * fmt.bps * fmt.ch_count;
        res = fwrite(&avg_bytes_per_sec, sizeof(avg_bytes_per_sec), 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        uint16_t block_align_offset = fmt.bps * fmt.ch_count;
        res = fwrite(&block_align_offset, sizeof(block_align_offset), 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        uint16_t bits_per_sample = fmt.bps * 8;
        res = fwrite(&bits_per_sample, sizeof(bits_per_sample), 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        res = fwrite("data", 4, 1, s->output);
        if(res != 1) {
                goto file_err;
        }

        char blank[4];
        memset((void *) blank, 1, 4);
        res = fwrite(blank, 1, 4, s->output);
        if(res != 4) {
                goto file_err;
        }

        s->ring = ring_buffer_init(CACHE_SECONDS * fmt.sample_rate * fmt.bps *
                        fmt.ch_count);

        return true;

file_err:
        fclose(s->output);
open_err:
        fprintf(stderr, "[Audio export] File opening error. Skipping audio export.\n");

        return false;
}

static void finalize(struct audio_export *s)
{
        int padding_byte_len = 0;
        if((s->saved_format.ch_count * s->saved_format.bps * s->total) % 2 == 1) {
                char padding_byte;
                padding_byte_len = 1;
                fwrite(&padding_byte, sizeof(padding_byte), 1, s->output);
        }


        fseek(s->output, CK_MASTER_SIZE_OFFSET, SEEK_SET);
        uint32_t ck_master_size = 4 + 24 + (8 + s->saved_format.bps *
                        s->saved_format.ch_count * s->total + padding_byte_len);
        size_t res;
        res = fwrite(&ck_master_size, sizeof(ck_master_size), 1, s->output);
        if(res != 1) {
                goto error;
        }

        fseek(s->output, CK_DATA_SIZE_OFFSET, SEEK_SET);
        uint32_t ck_data_size = s->saved_format.bps *
                        s->saved_format.ch_count * s->total;
        res = fwrite(&ck_data_size, sizeof(ck_data_size), 1, s->output);
        if(res != 1) {
                goto error;
        }

        return;
error:
        fprintf(stderr, "[Audio export] Could not finalize file. Audio file may be corrupted.\n");

}

struct audio_export * audio_export_init(char *filename)
{
        struct audio_export *s;

        s = malloc(sizeof(struct audio_export));
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

        s->total = 0;

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

                if(s->total > 0) {
                        finalize(s);
                        fclose(s->output);
                        ring_buffer_destroy(s->ring);
                }
                pthread_cond_destroy(&s->worker_cv);
                pthread_mutex_destroy(&s->lock);
                free(s->filename);
                free(s);
        }
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

        pthread_mutex_lock(&s->lock);
        ring_buffer_write(s->ring, frame->data, frame->data_len);
        s->new_work_ready = true;

        if(s->worker_waiting) {
                pthread_cond_signal(&s->worker_cv);
        }
        pthread_mutex_unlock(&s->lock);
}

