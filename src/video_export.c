/**
 * @file   video_export.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2024 CESNET, z. s. p. o.
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

#include <assert.h>                     // for assert
#include <compat/platform_semaphore.h>
#include <pthread.h>
#include <stdint.h>                     // for uint32_t
#include <stdio.h>
#include <stdlib.h>
#include <string.h>                     // for memcpy, memset, strdup

#include "types.h"                      // for tile, video_frame, video_desc
#include "utils/fs.h"                   // for MAX_PATH_SIZE
#include "video_codec.h"
#include "video_export.h"
#include "video_frame.h"                // for video_desc_from_frame, video_...

#define MAX_QUEUE_SIZE 300

/*
 * we do not need to have possible stalls, so IO is performend in a separate thread
 */
static void *video_export_thread(void *arg);
void output_summary(struct video_export *s);

struct output_entry {
        char *filename;
        char *data;
        int data_len;

        struct output_entry *next;
};

struct video_export {
        char *path;

        uint32_t total;

        pthread_mutex_t lock;
        struct output_entry * volatile head,
                            * volatile tail;
        volatile int queue_len;
        sem_t semaphore;

        struct video_desc saved_desc;

        pthread_t thread_id;
};

static void *video_export_thread(void *arg)
{
        struct video_export *s = (struct video_export *) arg;

        while(1) {
                platform_sem_wait(&s->semaphore);
                struct output_entry *current;

                pthread_mutex_lock(&s->lock);
                {
                        current = s->head;
                        s->head = s->head->next;
                        s->queue_len -= 1;
                }
                pthread_mutex_unlock(&s->lock);

                assert((current->data == NULL && current->data_len == 0) ||
                                (current->data != NULL && current->data_len != 0));

                // poison
                if(current->data == NULL) {
                        return NULL;
                }

                FILE *out = fopen(current->filename, "wb");
                if (out == NULL) {
                        perror("fopen");
                } else {
                        if (fwrite(current->data, current->data_len, 1, out) != 1) {
                                perror("fwrite");
                        }
                        fclose(out);
                }
                free(current->data);
                free(current->filename);
                free(current);
        }

        // never get here
}

struct video_export * video_export_init(const char *path)
{
        struct video_export *s = calloc(1, sizeof *s);
        assert(s != NULL);

        platform_sem_init(&s->semaphore, 0, 0);
        pthread_mutex_init(&s->lock, NULL);
        s->total = s->queue_len = 0;
        assert(path != NULL);
        s->path = strdup(path);
        s->head = s->tail = NULL;

        memset(&s->saved_desc, 0, sizeof(s->saved_desc));

        if(pthread_create(&s->thread_id, NULL, video_export_thread, s) != 0) {
                fprintf(stderr, "[Video exporter] Failed to create thread.\n");
                free(s);
                return NULL;
        }

        return s;
}

void output_summary(struct video_export *s)
{
        char name[MAX_PATH_SIZE];
        snprintf(name, sizeof name, "%s/video.info", s->path);

        FILE *summary = fopen(name, "w");

        if(!summary) {
                perror("Cannot write video export summary file");
                return;
        }

        fprintf(summary, "version %d\n", VIDEO_EXPORT_SUMMARY_VERSION);
        fprintf(summary, "width %d\n", s->saved_desc.width);
        fprintf(summary, "height %d\n", s->saved_desc.height);
        uint32_t fourcc = get_fourcc(s->saved_desc.color_spec);
        fprintf(summary, "fourcc %.4s\n", (char *) &fourcc);
        fprintf(summary, "fps %.2f\n", s->saved_desc.fps);
        fprintf(summary, "interlacing %d\n", (int) s->saved_desc.interlacing);
        fprintf(summary, "count %d\n", s->total);

        fclose(summary);
}

void video_export_destroy(struct video_export *s)
{
        if(s) {
                // poison
                struct output_entry *entry = calloc(sizeof(struct output_entry), 1);

                pthread_mutex_lock(&s->lock);
                {
                        if(s->head) {
                                s->tail->next = entry;
                                s->tail = entry;
                        } else {
                                s->head = s->tail = entry;
                        }
                }
                pthread_mutex_unlock(&s->lock);
                platform_sem_post(&s->semaphore);

                pthread_join(s->thread_id, NULL);
                pthread_mutex_destroy(&s->lock);

                // write summary
                if(s->total > 0) {
                        output_summary(s);
                }

                free(s->path);
                free(s);
        }
}

void video_export(struct video_export *s, struct video_frame *frame)
{
        if(!s) {
                return;
        }

        assert(frame != NULL);
        s->total += 1;

        if(s->saved_desc.width == 0) {
                s->saved_desc = video_desc_from_frame(frame);
        } else {
                if(!video_desc_eq(s->saved_desc, video_desc_from_frame(frame))) {
                        fprintf(stderr, "[Video export] Format change detected, not exporting.\n");
                        return;
                }
        }

        for (unsigned int i = 0; i < frame->tile_count; ++i) {
                assert(frame->tiles[i].data != NULL && frame->tiles[i].data_len != 0);

                struct output_entry *entry = malloc(sizeof(struct output_entry));

                entry->data_len = frame->tiles[i].data_len;
                entry->data = (char *) malloc(entry->data_len);
                entry->filename = malloc(MAX_PATH_SIZE);
                entry->next = NULL;

                if(frame->tile_count == 1) {
                        snprintf(entry->filename, MAX_PATH_SIZE, "%s/%08d.%s", s->path,
                                 s->total,
                                 get_codec_file_extension(frame->color_spec));
                } else {
                        // add also tile index
                        snprintf(entry->filename, MAX_PATH_SIZE, "%s/%08d_%d.%s", s->path,
                                 s->total, i,
                                 get_codec_file_extension(frame->color_spec));
                }
                memcpy(entry->data, frame->tiles[i].data, entry->data_len);

                pthread_mutex_lock(&s->lock);
                {
                        // check if we do not occupy too much memory
                        if(s->queue_len >= MAX_QUEUE_SIZE) {
                                fprintf(stderr, "[Video export] Maximal queue size (%d) exceeded, not saving frame %d.\n",
                                                MAX_QUEUE_SIZE,
                                                s->total); // we increment total size to keep the index
                                pthread_mutex_unlock(&s->lock);
                                free(entry->data);
                                free(entry);
                                return;
                        }

                        if(s->head) {
                                s->tail->next = entry;
                                s->tail = entry;
                        } else {
                                s->head = s->tail = entry;
                        }
                        s->queue_len += 1;
                }
                pthread_mutex_unlock(&s->lock);

                platform_sem_post(&s->semaphore);
        }
}

