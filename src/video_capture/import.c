/*
 * FILE:    import.c
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
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#include "video_export.h"
#include "video_capture/import.h"
//#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <glob.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/poll.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define BUFFER_LEN_MAX 40

#define VIDCAP_IMPORT_ID 0x76FA7F6D

struct processed_entry;

struct processed_entry {
        char *data;
        int data_len;
        struct processed_entry *next;
};

struct vidcap_import_state {
        struct video_frame *frame;
        struct tile *tile;
        char *directory;

        pthread_mutex_t lock;
        pthread_cond_t reader_cv;
        volatile bool reader_waiting;
        pthread_cond_t boss_cv;
        volatile bool boss_waiting;
        struct processed_entry * volatile head, * volatile tail;
        volatile int queue_len;

        volatile bool finish_thread;
        volatile bool reader_finished;

        pthread_t thread_id;

        struct timeval prev_time;
        int count;
};

static void * reading_thread(void *args);

struct vidcap_type *
vidcap_import_probe(void)
{
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_IMPORT_ID;
		vt->name        = "import";
		vt->description = "Video importer (not to be called directly)";
	}
	return vt;
}

void *
vidcap_import_init(char *directory, unsigned int flags)
{
        UNUSED(flags);
	struct vidcap_import_state *s;

	printf("vidcap_import_init\n");

        s = (struct vidcap_import_state *) calloc(1, sizeof(struct vidcap_import_state));
        s->head = s->tail = NULL;
        s->queue_len = 0;

        s->boss_waiting = false;
        s->reader_waiting = false;

        s->finish_thread = false;
        s->reader_finished = false;
        
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->reader_cv, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        
        char *info_filename = malloc(strlen(directory) + sizeof("/video.info") + 1);
        assert(info_filename != NULL);
        strcpy(info_filename, directory);
        strcat(info_filename, "/video.info");

        FILE *info = fopen(info_filename, "r");
        free(info_filename);
        if(info == NULL) {
                perror("[import] Failed to open index file.");
                goto free_state;
        }

        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);
        char line[512];
        uint32_t items_found = 0;
        while(!feof(info)) {
                if(fgets(line, sizeof(line), info) == NULL) {
                        // empty line
                        continue;
                }
                if(strncmp(line, "version ", strlen("version ")) == 0) {
                        long int version = strtol(line + strlen("version "), (char **) NULL, 10);
                        if(version == LONG_MIN || version == LONG_MAX) {
                                fprintf(stderr, "[import] cannot read version line.\n");
                                goto close_file;
                        }
                        if(version != VIDEO_EXPORT_SUMMARY_VERSION) {
                                fprintf(stderr, "[import] Invalid version %ld.\n", version);
                                goto close_file;
                        }
                        items_found |= 1<<0;
                } else if(strncmp(line, "width ", strlen("width ")) == 0) {
                        long int width = strtol(line + strlen("width "), (char **) NULL, 10);
                        if(width == LONG_MIN || width == LONG_MAX) {
                                fprintf(stderr, "[import] cannot read video width.\n");
                                goto close_file;
                        }
                        s->tile->width = width;
                        items_found |= 1<<1;
                } else if(strncmp(line, "height ", strlen("height ")) == 0) {
                        long int height = strtol(line + strlen("height "), (char **) NULL, 10);
                        if(height == LONG_MIN || height == LONG_MAX) {
                                fprintf(stderr, "[import] cannot read video height.\n");
                                goto close_file;
                        }
                        s->tile->height = height;
                        items_found |= 1<<2;
                } else if(strncmp(line, "fourcc ", strlen("fourcc ")) == 0) {
                        char *ptr = line + strlen("fourcc ");
                        if(strlen(ptr) != 5) { // including '\n'
                                fprintf(stderr, "[import] cannot read video FourCC tag.\n");
                                goto close_file;
                        }
                        uint32_t fourcc;
                        memcpy((void *) &fourcc, ptr, sizeof(fourcc));
                        s->frame->color_spec = get_codec_from_fcc(fourcc);
                        if(s->frame->color_spec == (codec_t) -1) {
                                fprintf(stderr, "[import] Requested codec not known.\n");
                                goto close_file;
                        }
                        items_found |= 1<<3;
                } else if(strncmp(line, "fps ", strlen("fps ")) == 0) {
                        char *ptr = line + strlen("fps ");
                        s->frame->fps = strtod(ptr, NULL);
                        if(s->frame->fps == HUGE_VAL || s->frame->fps <= 0) {
                                fprintf(stderr, "[import] Invalid FPS.\n");
                                goto close_file;
                        }
                        items_found |= 1<<4;
                } else if(strncmp(line, "interlacing ", strlen("interlacing ")) == 0) {
                        char *ptr = line + strlen("interlacing ");
                        s->frame->interlacing = atoi(ptr);
                        if(s->frame->interlacing > 4) {
                                fprintf(stderr, "[import] Invalid interlacing.\n");
                                goto close_file;
                        }
                        items_found |= 1<<5;
                } else if(strncmp(line, "count ", strlen("count ")) == 0) {
                        char *ptr = line + strlen("count ");
                        s->count = atoi(ptr);
                        items_found |= 1<<6;
                }
        }

        fclose(info);

        if(items_found != (1 << 7) - 1) {
                fprintf(stderr, "[import] Failed while reading config file - some items missing.\n");
                goto free_frame;
        }

        if(pthread_create(&s->thread_id, NULL, reading_thread, (void *) s) != 0) {
                fprintf(stderr, "Unable to create thread.\n");
                goto free_frame;
        }

        s->directory = strdup(directory);

        gettimeofday(&s->prev_time, NULL);

	return s;
        
close_file:
        fclose(info);
free_frame:
        vf_free(s->frame);
free_state:
        free(s);
        return NULL;
}

void
vidcap_import_finish(void *state)
{
	struct vidcap_import_state *s = (struct vidcap_import_state *) state;

        pthread_mutex_lock(&s->lock);
        {
                s->finish_thread = true;
                if(s->reader_waiting)
                        pthread_cond_signal(&s->reader_cv);
        }
        pthread_mutex_unlock(&s->lock);

	pthread_join(s->thread_id, NULL);
}

void
vidcap_import_done(void *state)
{
	struct vidcap_import_state *s = (struct vidcap_import_state *) state;
	assert(s != NULL);

        struct processed_entry *current = s->head;
        while(current != NULL) {
                free(current->data);
                struct processed_entry *tmp = current;
                current = current->next;
                free(tmp);
        }
        free(s->tile->data);
        vf_free(s->frame);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->reader_cv);
        pthread_cond_destroy(&s->boss_cv);
        free(s->directory);
        free(s);
}

static void * reading_thread(void *args)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) args;
        int index = 0;
        char name[512];

        while(index < s->count && !s->finish_thread) {
                struct processed_entry *new_entry = NULL;
                pthread_mutex_lock(&s->lock);
                {
                        while(s->queue_len >= BUFFER_LEN_MAX - 1 && !s->finish_thread) {
                                s->reader_waiting = true;
                                pthread_cond_wait(&s->reader_cv, &s->lock);
                                s->reader_waiting = false;
                        }
                }
                pthread_mutex_unlock(&s->lock);
                snprintf(name, sizeof(name), "%s/%08d.%s", s->directory, index,
                                get_codec_file_extension(s->frame->color_spec));

                struct stat sb;
                if (stat(name, &sb)) {
                        perror("stat");
                        goto next;
                } 
                FILE *file = fopen(name, "rb");
                if(!file) {
                        perror("fopen");
                        goto next;
                }
                new_entry = malloc(sizeof(struct processed_entry));
                assert(new_entry != NULL);
                new_entry->data_len = sb.st_size;
                new_entry->data = malloc(new_entry->data_len);
                new_entry->next = NULL;
                assert(new_entry->data != NULL);

                size_t res = fread(new_entry->data, new_entry->data_len, 1, file);
                if(res != 1) {
                        perror("fread");
                        goto next;
                }

                pthread_mutex_lock(&s->lock);
                {
                        if(s->head) {
                                s->tail->next = new_entry;
                                s->tail = new_entry;
                        } else {
                                s->head = s->tail = new_entry;
                        }
                        s->queue_len += 1;

                        if(s->boss_waiting)
                                pthread_cond_signal(&s->boss_cv);
                }
                pthread_mutex_unlock(&s->lock);

next:
                index++;
        }

        pthread_mutex_lock(&s->lock);
        {
                s->reader_finished = true;

                if(s->boss_waiting)
                        pthread_cond_signal(&s->boss_cv);
        }
        pthread_mutex_unlock(&s->lock);


        return NULL;
}

struct video_frame *
vidcap_import_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) state;
        struct timeval cur_time;
        
        struct processed_entry *current = NULL;

        // free old data
        free(s->tile->data);
        s->tile->data = NULL;

        pthread_mutex_lock(&s->lock);
        {
                while(s->queue_len == 0 && !s->reader_finished) {
                        s->boss_waiting = true;
                        pthread_cond_wait(&s->boss_cv, &s->lock);
                        s->boss_waiting = false;
                }

                if(s->queue_len == 0 && s->reader_finished) {
                        pthread_mutex_unlock(&s->lock);
                        return NULL;
                }

                current = s->head;
                s->head = s->head->next;
                s->queue_len -= 1;

                if(s->reader_waiting)
                        pthread_cond_signal(&s->reader_cv);
        }
        pthread_mutex_unlock(&s->lock);

        s->tile->data_len = current->data_len;
        s->tile->data = current->data;
        free(current);

        gettimeofday(&cur_time, NULL);
        while(tv_diff_usec(cur_time, s->prev_time) < 1000000.0 / s->frame->fps)
                gettimeofday(&cur_time, NULL);
        tv_add_usec(&s->prev_time, 1000000.0 / s->frame->fps);
        
        *audio = NULL;

	return s->frame;
}

