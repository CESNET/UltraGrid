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
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"
#include "audio/wav_reader.h"
#include "utils/ring_buffer.h"
#include "utils/worker.h"
#include "video_export.h"
#include "video_capture/import.h"
//#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#define BUFFER_LEN_MAX 40
#define MAX_CLIENTS 16

#define CONTROL_PORT 15004
#define VIDCAP_IMPORT_ID 0x76FA7F6D

#define PIPE "/tmp/ultragrid_import.fifo"

#define MAX_NUMBER_WORKERS 100

struct processed_entry;
struct tile_data {
        char *data;
        int data_len;
};

struct processed_entry {
        struct processed_entry *next;
        int count;
        struct tile_data tiles[];
};

typedef enum {
        SEEK,
        FINALIZE,
        PAUSE
} message_t;

struct message;

struct message {
        message_t       type;
        void           *data;
        size_t          data_len;
        struct message *next;
};

typedef enum {
        IMPORT_SEEK_SET,
        IMPORT_SEEK_END,
        IMPORT_SEEK_CUR
} seek_direction_t;

struct seek_data {
        seek_direction_t whence;
        ssize_t offset;
};

struct message_queue {
        struct message *head;
        struct message *tail;
        size_t          len;
};

struct audio_state {
        bool has_audio;
        FILE *file;
        ring_buffer_t *data;
        int total_samples;
        int samples_read;
        pthread_t thread_id;

        pthread_cond_t worker_cv;
        volatile bool worker_waiting;
        pthread_cond_t boss_cv;
        volatile bool boss_waiting;

        pthread_mutex_t lock;
        unsigned long long int played_samples;

        struct message_queue message_queue;
}; 

struct vidcap_import_state {
        struct audio_frame audio_frame;
        struct audio_state audio_state;
        struct video_desc video_desc;
        int frames;
        int frames_prev;
        struct timeval t0;
        char *directory;

        struct message_queue message_queue;

        pthread_mutex_t lock;
        pthread_cond_t worker_cv;
        volatile bool worker_waiting;
        pthread_cond_t boss_cv;
        volatile bool boss_waiting;
        struct processed_entry * volatile head, * volatile tail;
        volatile int queue_len;

        pthread_t thread_id;
        pthread_t control_thread_id;

        struct timeval prev_time;
        int count;

        bool finished;
        bool loop;
        bool o_direct;
        int video_reading_threads_count;
};

#ifdef WIN32
#define WIN32_UNUSED __attribute__((unused))
#else
#define WIN32_UNUSED
#endif

static void * audio_reading_thread(void *args);
static void * reading_thread(void *args);
#ifndef WIN32
static void * control_thread(void *args);
#endif
static bool init_audio(struct vidcap_import_state *state, char *audio_filename);
static void send_message(struct message *msg, struct message_queue *queue);
static struct message* pop_message(struct message_queue *queue);
static int flush_processed(struct processed_entry *list);

static void message_queue_clear(struct message_queue *queue);
static bool parse_msg(char *buffer, char buffer_len, /* out */ char *message, int *new_buffer_len) WIN32_UNUSED;
static void process_msg(struct vidcap_import_state *state, char *message) WIN32_UNUSED;

volatile bool exit_control = false;

static void message_queue_clear(struct message_queue *queue) {
        queue->head = queue->tail = NULL;
        queue->len = 0;
}

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

#define READ_N(buf, len) if (fread(buf, len, 1, audio_file) != 1) goto error_format;

static bool init_audio(struct vidcap_import_state *s, char *audio_filename)
{
        FILE *audio_file = fopen(audio_filename, "r");
        if(!audio_file) {
                perror("Cannot open audio file");
                return false;
        }

        // common commands - will run in any way
        if(!audio_file) {
                goto error_opening;
        }

        struct wav_metadata metadata;

        int ret = read_wav_header(audio_file, &metadata);
        switch(ret) {
                case WAV_HDR_PARSE_READ_ERROR:
                        fprintf(stderr, "Error reading WAV header!\n");
                        goto error_format;
                case WAV_HDR_PARSE_WRONG_FORMAT:
                        fprintf(stderr, "Unsupported WAV format!\n");
                        goto error_format;
                case WAV_HDR_PARSE_NOT_PCM:
                        fprintf(stderr, "Only supported audio format is PCM.\n");
                        goto error_format;
                case WAV_HDR_PARSE_OK:
                        break;
        }

        s->audio_frame.ch_count = metadata.ch_count;
        s->audio_frame.sample_rate = metadata.sample_rate;
        s->audio_frame.bps = metadata.bits_per_sample / 8;
        s->audio_state.total_samples = metadata.data_size / s->audio_frame.bps / s->audio_frame.ch_count;
        s->audio_state.samples_read = 0;

        s->audio_state.data = ring_buffer_init(s->audio_frame.bps * s->audio_frame.sample_rate *
                        s->audio_frame.ch_count * 180);

        s->audio_frame.max_size = s->audio_frame.bps * s->audio_frame.sample_rate * s->audio_frame.ch_count;
        s->audio_frame.data_len = 0;
        s->audio_frame.data = malloc(s->audio_frame.max_size);

        s->audio_state.file = audio_file;

        pthread_cond_init(&s->audio_state.worker_cv, NULL);
        s->audio_state.worker_waiting = false;
        pthread_cond_init(&s->audio_state.boss_cv, NULL);
        s->audio_state.boss_waiting = false;
        pthread_mutex_init(&s->audio_state.lock, NULL);
        s->audio_state.played_samples = 0;


        return true;

error_format:
        fprintf(stderr, "Audio format file error - unknown format\n");
error_opening:
        fclose(audio_file);
        return false;
}

void *
vidcap_import_init(const struct vidcap_params *params)
{
	struct vidcap_import_state *s = NULL;
        FILE *info = NULL; // metadata file

	printf("vidcap_import_init\n");

        s = (struct vidcap_import_state *) calloc(1, sizeof(struct vidcap_import_state));
        s->head = s->tail = NULL;
        s->queue_len = 0;
        s->frames_prev = s->frames = 0;
        gettimeofday(&s->t0, NULL);

        s->boss_waiting = false;
        s->worker_waiting = false;

        s->video_reading_threads_count = 1; // default is single threaded

        char *tmp = strdup(vidcap_params_get_fmt(params));
        char *save_ptr;
        const char *directory = strtok_r(tmp, ":", &save_ptr);
        if (!directory || strcmp(directory, "help") == 0) {
                fprintf(stderr, "Import usage:\n"
                                "\t<directory>{:loop|:mt_reading=<nr_threads>|:o_direct}");
                return NULL;
        }
        char *suffix;
        while ((suffix = strtok_r(NULL, ":", &save_ptr)) != NULL) {
                if (strcmp(suffix, "loop") == 0) {
                        s->loop = true;
                } else if (strncmp(suffix, "mt_reading=",
                                        strlen("mt_reading=")) == 0) {
                        s->video_reading_threads_count = atoi(suffix +
                                        strlen("mt_reading="));
                        assert(s->video_reading_threads_count <=
                                        MAX_NUMBER_WORKERS);
                } else if (strcmp(suffix, "o_direct") == 0) {
                        s->o_direct = true;
                } else {
                        fprintf(stderr, "[Playback] Unrecognized"
                                        " option %s.\n", suffix);
                        return NULL;
                }
        }

        message_queue_clear(&s->message_queue);
        message_queue_clear(&s->audio_state.message_queue);

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->worker_cv, NULL);
        pthread_cond_init(&s->boss_cv, NULL);

        char *audio_filename = malloc(strlen(directory) + sizeof("/soud.wav") + 1);
        assert(audio_filename != NULL);
        strcpy(audio_filename, directory);
        strcat(audio_filename, "/sound.wav");
        if((vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) && init_audio(s, audio_filename)) {
                s->audio_state.has_audio = true;
                if(pthread_create(&s->audio_state.thread_id, NULL, audio_reading_thread, (void *) s) != 0) {
                        fprintf(stderr, "Unable to create thread.\n");
                        goto error;
                }
        } else {
                s->audio_state.has_audio = false;
        }
        free(audio_filename);
        
        char *info_filename = malloc(strlen(directory) + sizeof("/video.info") + 1);
        assert(info_filename != NULL);
        strcpy(info_filename, directory);
        strcat(info_filename, "/video.info");

        info = fopen(info_filename, "r");
        free(info_filename);
        if(info == NULL) {
                perror("[import] Failed to open index file");
                goto error;
        }

        struct video_desc desc;

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
                                goto error;
                        }
                        if(version != VIDEO_EXPORT_SUMMARY_VERSION) {
                                fprintf(stderr, "[import] Invalid version %ld.\n", version);
                                goto error;
                        }
                        items_found |= 1<<0;
                } else if(strncmp(line, "width ", strlen("width ")) == 0) {
                        long int width = strtol(line + strlen("width "), (char **) NULL, 10);
                        if(width == LONG_MIN || width == LONG_MAX) {
                                fprintf(stderr, "[import] cannot read video width.\n");
                                goto error;
                        }
                        desc.width = width;
                        items_found |= 1<<1;
                } else if(strncmp(line, "height ", strlen("height ")) == 0) {
                        long int height = strtol(line + strlen("height "), (char **) NULL, 10);
                        if(height == LONG_MIN || height == LONG_MAX) {
                                fprintf(stderr, "[import] cannot read video height.\n");
                                goto error;
                        }
                        desc.height = height;
                        items_found |= 1<<2;
                } else if(strncmp(line, "fourcc ", strlen("fourcc ")) == 0) {
                        char *ptr = line + strlen("fourcc ");
                        if(strlen(ptr) != 5) { // including '\n'
                                fprintf(stderr, "[import] cannot read video FourCC tag.\n");
                                goto error;
                        }
                        uint32_t fourcc;
                        memcpy((void *) &fourcc, ptr, sizeof(fourcc));
                        desc.color_spec = get_codec_from_fcc(fourcc);
                        if(desc.color_spec == (codec_t) -1) {
                                fprintf(stderr, "[import] Requested codec not known.\n");
                                goto error;
                        }
                        items_found |= 1<<3;
                } else if(strncmp(line, "fps ", strlen("fps ")) == 0) {
                        char *ptr = line + strlen("fps ");
                        desc.fps = strtod(ptr, NULL);
                        if(desc.fps == HUGE_VAL || desc.fps <= 0) {
                                fprintf(stderr, "[import] Invalid FPS.\n");
                                goto error;
                        }
                        items_found |= 1<<4;
                } else if(strncmp(line, "interlacing ", strlen("interlacing ")) == 0) {
                        char *ptr = line + strlen("interlacing ");
                        desc.interlacing = atoi(ptr);
                        if(desc.interlacing > 4) {
                                fprintf(stderr, "[import] Invalid interlacing.\n");
                                goto error;
                        }
                        items_found |= 1<<5;
                } else if(strncmp(line, "count ", strlen("count ")) == 0) {
                        char *ptr = line + strlen("count ");
                        s->count = atoi(ptr);
                        items_found |= 1<<6;
                }
        }

        s->directory = strdup(directory);

        char name[1024];
        snprintf(name, sizeof(name), "%s/%08d.%s", s->directory, 1,
                        get_codec_file_extension(desc.color_spec));

        struct stat sb;
        if (stat(name, &sb) == 0) {
                desc.tile_count = 1;
        } else {
                desc.tile_count = 0;
                for (int i = 0; i < 10; i++) {
                        snprintf(name, sizeof(name), "%s/%08d_%d.%s",
                                        s->directory, 1, i,
                                        get_codec_file_extension(desc.color_spec));
                        if (stat(name, &sb) == 0) {
                                desc.tile_count++;
                        } else {
                                break;
                        }
                }
                if (desc.tile_count == 0) {
                        fprintf(stderr, "Unable to open first file of "
                                        "the video sequence.\n");
                        goto error;
                }
        }

        s->video_desc = desc;

        fclose(info);

        if(items_found != (1 << 7) - 1) {
                fprintf(stderr, "[import] Failed while reading config file - some items missing.\n");
                goto error;
        }

        if(pthread_create(&s->thread_id, NULL, reading_thread, (void *) s) != 0) {
                fprintf(stderr, "Unable to create thread.\n");
                goto error;
        }

#ifndef WIN32
        if(pthread_create(&s->control_thread_id, NULL, control_thread, (void *) s) != 0) {
                fprintf(stderr, "Unable to create control thread.\n");
                goto error;
        }
#endif

        gettimeofday(&s->prev_time, NULL);

	return s;
        
error:
        if (info != NULL)
                fclose(info);
        if (s) {
                free(s->directory);
                free(s);
        }
        return NULL;
}

static void exit_reading_threads(struct vidcap_import_state *s)
{
        struct message *msg = malloc(sizeof(struct message));

        msg->type = FINALIZE;
        msg->data = NULL;
        msg->data_len = 0;
        msg->next = NULL;

        pthread_mutex_lock(&s->lock);
        {
                send_message(msg, &s->message_queue);

                if(s->worker_waiting)
                        pthread_cond_signal(&s->worker_cv);
        }
        pthread_mutex_unlock(&s->lock);

	pthread_join(s->thread_id, NULL);

        // audio
        if(s->audio_state.has_audio) {
                struct message *msg = malloc(sizeof(struct message));

                msg->type = FINALIZE;
                msg->data = NULL;
                msg->data_len = 0;
                msg->next = NULL;

                pthread_mutex_lock(&s->audio_state.lock);
                {
                        send_message(msg, &s->audio_state.message_queue);

                        if(s->audio_state.worker_waiting)
                                pthread_cond_signal(&s->audio_state.worker_cv);
                }
                pthread_mutex_unlock(&s->audio_state.lock);

                pthread_join(s->audio_state.thread_id, NULL);
        }

#ifndef WIN32
        exit_control = true;

        pthread_join(s->control_thread_id, NULL);
#endif
}

static void free_entry(struct processed_entry *entry)
{
        if (entry == NULL) {
                return;
        }
        for (int i = 0; i < entry->count; ++i) {
                aligned_free(entry->tiles[i].data);
        }

        free(entry);
}

static void vidcap_import_finish(void *state)
{
        struct vidcap_import_state *s = (struct vidcap_import_state *) state;

        exit_reading_threads(s);

#ifndef WIN32
        exit_control = true;

        pthread_join(s->control_thread_id, NULL);
#endif
}

static int flush_processed(struct processed_entry *list)
{
        int frames_deleted = 0;
        struct processed_entry *current = list;

        while(current != NULL) {
                struct processed_entry *tmp = current;
                current = current->next;
                free_entry(tmp);
                frames_deleted++;
        }

        return frames_deleted;
}

void vidcap_import_done(void *state)
{
	struct vidcap_import_state *s = (struct vidcap_import_state *) state;
	assert(s != NULL);

        vidcap_import_finish(state);

        flush_processed(s->head);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->worker_cv);
        pthread_cond_destroy(&s->boss_cv);
        free(s->directory);

        // audio
        if(s->audio_state.has_audio) {
                ring_buffer_destroy(s->audio_state.data);

                free(s->audio_frame.data);

                fclose(s->audio_state.file);

                pthread_cond_destroy(&s->audio_state.worker_cv);
                pthread_cond_destroy(&s->audio_state.boss_cv);
                pthread_mutex_destroy(&s->audio_state.lock);
        }

        free(s);
}

/*
 * Message len can be at most buffer_len + 1 (including '\0')
 */
static bool parse_msg(char *buffer, char buffer_len, /* out */ char *message, int *new_buffer_len)
{
        bool ret = false;
        int i = 0;

        while (i < buffer_len) {
                if(buffer[i] == '\0' || buffer[i] == '\n' || buffer[i] == '\r') {
                        ++i;
                } else {
                        break;
                }
        }

        int start = i;

        for( ; i < buffer_len; ++i) {
                if(buffer[i] == '\0' || buffer[i] == '\n' || buffer[i] == '\r') {
                        memcpy(message, buffer + start, i - start);
                        message[i - start] = '\0';
                        ret = true;
                        break;
                }
        }
        
        if(ret) {
                memmove(buffer, buffer + i, buffer_len - i);
                *new_buffer_len = buffer_len - i;
        }

        return ret;
}

static void send_message(struct message *msg, struct message_queue *queue)
{
        if(queue->head) {
                queue->tail->next = msg;
                queue->tail = msg;
        } else {
                queue->head = queue->tail = msg;
        }

        queue->len += 1;
}

static struct message *pop_message(struct message_queue *queue)
{
        assert(queue->len > 0);
        struct message *ret;

        ret = queue->head;
        queue->head = queue->head->next;
        if(queue->head == NULL) {
                queue->tail = NULL;
        }

        queue->len -= 1;

        return ret;
}

static void process_msg(struct vidcap_import_state *s, char *message)
{
        if(strcasecmp(message, "pause") == 0) {
                pthread_mutex_lock(&s->lock);
                {
                        struct message *msg = malloc(sizeof(struct message));
                        msg->type = PAUSE;
                        msg->data = NULL;
                        msg->data_len = 0;
                        msg->next = NULL;

                        send_message(msg, &s->message_queue);

                        if(s->worker_waiting) {
                                pthread_cond_signal(&s->worker_cv);
                        }
                }
                pthread_mutex_unlock(&s->lock);
        } else if(strncasecmp(message, "seek ", strlen("seek ")) == 0) {
                if(s->audio_state.has_audio == true) {
                        fprintf(stderr, "Seeking now allowed if we have audio. (Not yet implemented)\n");
                        return;
                }

                char *time_spec = message + strlen("seek ");

                struct message *msg = malloc(sizeof(struct message));
                struct seek_data *data = malloc(sizeof(struct seek_data));
                msg->type = SEEK;
                msg->data = data;
                msg->data_len = sizeof(struct seek_data);
                msg->next = NULL;

                if(time_spec[0] == '+' || time_spec[0] == '-') {
                        data->whence = IMPORT_SEEK_CUR;
                        if(strchr(time_spec, 's') != NULL) {
                                double val = atof(time_spec);
                                data->offset = val * s->video_desc.fps;
                        } else {
                                data->offset = atoi(time_spec);
                        }
                } else {
                        data->whence = IMPORT_SEEK_SET;
                        if(strchr(time_spec, 's') != NULL) {
                                double val = atof(time_spec);
                                data->offset = val * s->video_desc.fps;
                        } else {
                                data->offset = atoi(time_spec);
                        }
                }

                struct message *audio_msg = NULL;

                if(s->audio_state.has_audio) {
                        audio_msg = malloc(sizeof(struct message));
                        memcpy(audio_msg, msg, sizeof(struct message));

                        if(audio_msg->data) { // deep copy
                                audio_msg->data = malloc(msg->data_len);
                                memcpy(audio_msg->data, msg->data, msg->data_len);
                        }
                }

                pthread_mutex_lock(&s->lock);
                {
                        send_message(msg, &s->message_queue);
                        if(s->worker_waiting) {
                                pthread_cond_signal(&s->worker_cv);
                        }
                }
                pthread_mutex_unlock(&s->lock);

                if(s->audio_state.has_audio) {
                        pthread_mutex_lock(&s->audio_state.lock);
                        {
                                send_message(audio_msg, &s->audio_state.message_queue);
                                if(s->audio_state.worker_waiting) {
                                        pthread_cond_signal(&s->audio_state.worker_cv);
                                }
                        }
                        pthread_mutex_unlock(&s->audio_state.lock);
                }
        } else if(strcasecmp(message, "quit") == 0) {
                exit_uv(0);
        } else {
                fprintf(stderr, "Warning: unknown message: \'%s\'\n", message);
        }
}

struct client;

struct client {
        int fd;
        char buff[1024];
        int buff_len;
        bool pipe;

        struct client *next;
};

#ifndef WIN32
static void * control_thread(void *args)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) args;
        int fd;

        fd = socket(AF_INET6, SOCK_STREAM, 0);
        int val = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
        struct sockaddr_in6 s_in;
        s_in.sin6_family = AF_INET6;
        s_in.sin6_addr = in6addr_any;
        s_in.sin6_port = htons(CONTROL_PORT);

        bind(fd, (const struct sockaddr *) &s_in, sizeof(s_in));
        listen(fd, MAX_CLIENTS);
        struct sockaddr_storage client_addr;
        socklen_t len;

        unlink(PIPE);
        errno = 0;
        int fifo_status = mkfifo(PIPE, 0777);

        struct client *clients = NULL;
        if(!fifo_status) {
                clients = malloc(sizeof(struct client));
                clients->fd = open(PIPE, O_RDONLY | O_NONBLOCK);
                clients->pipe = true;
                clients->buff_len = 0;
                clients->next = NULL;
        } else {
                perror("Video import: unable to create communication pipe");
        }

        while(!exit_control) {
                fd_set set;
                FD_ZERO(&set);
                FD_SET(fd, &set);
                int max_fd = fd + 1;

                struct client *cur = clients;

                while(cur) {
                        FD_SET(cur->fd, &set);
                        if(cur->fd + 1 > max_fd) {
                                max_fd = cur->fd + 1;
                        }
                        cur = cur->next;
                }

                struct timeval timeout = { .tv_sec = 1, .tv_usec = 0 };
                if(select(max_fd, &set, NULL, NULL, &timeout) >= 1) {
                        if(FD_ISSET(fd, &set)) {
                                struct client *new_client = malloc(sizeof(struct client));
                                new_client->fd = accept(fd, (struct sockaddr *) &client_addr, &len);
                                new_client->next = clients;
                                new_client->buff_len = 0;
                                new_client->pipe = false;
                                clients = new_client;
                        }

                        struct client **parent_ptr = &clients;
                        struct client *cur = clients;

                        while(cur) {
                                if(FD_ISSET(cur->fd, &set)) {
                                        ssize_t ret = read(cur->fd, cur->buff + cur->buff_len, 1024 - cur->buff_len);
                                        if(ret == -1) {
                                                fprintf(stderr, "Error reading socket!!!\n");
                                        }
                                        if(ret == 0) {
                                                if(!cur->pipe) {
                                                        close(cur->fd);
                                                        *parent_ptr = cur->next;
                                                        free(cur);
                                                        cur = *parent_ptr; // now next
                                                        continue;
                                                }
                                        }
                                        cur->buff_len += ret;
                                }
                                parent_ptr = &cur->next;
                                cur = cur->next;
                        }
                }

                cur = clients;
                while(cur) {
                        char msg[1024 + 1];
                        int cur_buffer_len;
                        if(parse_msg(cur->buff, cur->buff_len, msg, &cur_buffer_len)) {
                                fprintf(stderr, "msg: %s\n", msg);
                                cur->buff_len = cur_buffer_len;
                                process_msg(s, msg);
                        } else {
                                if(cur->buff_len == 1024) {
                                        fprintf(stderr, "Socket buffer full and no delimited message. Discarding.\n");
                                        cur->buff_len = 0;
                                }
                        }

                        cur = cur->next;
                }
        }

        struct client *cur = clients;
        while(cur) {
                struct client *tmp = cur;
                close(cur->fd);
                cur = cur->next;
                free(tmp);
        }

        close(fd);
        unlink(PIPE);

        return NULL;
}
#endif // WIN32
        
static void * audio_reading_thread(void *args)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) args;

        //while(s->audio_state.samples_read < s->audio_state.total_samples && !s->finish_threads) {
        while(1) {
                int max_read;
                pthread_mutex_lock(&s->audio_state.lock);
                {
                        while((ring_get_current_size(s->audio_state.data) > ring_get_size(s->audio_state.data) * 2 / 3 ||
                                                s->audio_state.samples_read >= s->audio_state.total_samples)
                                        && s->audio_state.message_queue.len == 0) {
                                s->audio_state.worker_waiting = true;
                                pthread_cond_wait(&s->audio_state.worker_cv, &s->audio_state.lock);
                                s->audio_state.worker_waiting = false;
                        }

                        while(s->audio_state.message_queue.len > 0) {
                                struct message *msg = pop_message(&s->audio_state.message_queue);
                                if(msg->type == FINALIZE) {
                                        pthread_mutex_unlock(&s->lock);
                                        free(msg);
                                        goto exited;
                                } else if (msg->type == SEEK) {
                                        fprintf(stderr, "Seeking in audio import not yet implemented.\n");
                                        abort();
                                }
                        }

                        max_read = ring_get_size(s->audio_state.data) -
                                        ring_get_current_size(s->audio_state.data) - 1;

                }
                pthread_mutex_unlock(&s->audio_state.lock);

                char *buffer = (char *) malloc(max_read);

                size_t ret = fread(buffer, s->audio_frame.ch_count * s->audio_frame.bps,
                                max_read / s->audio_frame.ch_count / s->audio_frame.bps, s->audio_state.file);
                s->audio_state.samples_read += ret;

                pthread_mutex_lock(&s->audio_state.lock);
                {
                        ring_buffer_write(s->audio_state.data, buffer, ret * s->audio_frame.ch_count * s->audio_frame.bps);
                        if(s->audio_state.boss_waiting)
                                pthread_cond_signal(&s->audio_state.boss_cv);
                }
                pthread_mutex_unlock(&s->audio_state.lock);

                free(buffer);
        }
exited:

        return NULL;
}

struct video_reader_data {
        char file_name_prefix[512];
        char file_name_suffix[512];
        unsigned int tile_count;
        struct processed_entry *entry;
        bool o_direct;
};

#define ALIGN 512

static void *video_reader_callback(void *arg)
{
        struct video_reader_data *data =
                (struct video_reader_data *) arg;
       
        data->entry = calloc(1, sizeof(struct processed_entry) + data->tile_count * sizeof(struct tile_data));
        assert(data->entry != NULL);
        data->entry->next = NULL;
        data->entry->count = data->tile_count;

        for (unsigned int i = 0; i < data->tile_count; i++) {
                char name[1024];
                char tile_idx[3] = "";
                if (data->tile_count > 1) {
                        sprintf(tile_idx, "_%d", i);
	        }
                snprintf(name, sizeof(name), "%s%s.%s",
                                data->file_name_prefix, tile_idx,
                                data->file_name_suffix);

                struct stat sb;
                if (stat(name, &sb)) {
                        perror("stat");
                        free_entry(data->entry);
                        return NULL;
                }

                int flags = O_RDONLY;
                if (data->o_direct) {
#ifdef HAVE_LINUX
                        flags |= O_DIRECT;
#endif
                }
                int fd = open(name, flags);
                if(fd == -1) {
                        perror("open");
                        return NULL;
                }

                data->entry->tiles[i].data_len = sb.st_size;
                const int aligned_data_len = (data->entry->tiles[i].data_len + ALIGN - 1)
                        / ALIGN * ALIGN;
                // alignment needed when using O_DIRECT flag
                data->entry->tiles[i].data = (char *)
                        aligned_malloc(aligned_data_len, ALIGN);
                assert(data->entry->tiles[i].data != NULL);

                ssize_t bytes = 0;
                do {
                        ssize_t res = read(fd, data->entry->tiles[i].data + bytes,
                                        (data->entry->tiles[i].data_len - bytes + ALIGN - 1)
                                        / ALIGN * ALIGN);
                        if (res <= 0) {
                                perror("read");
                                return NULL;
                        }
                        bytes += res;
                } while (bytes < data->entry->tiles[i].data_len);

                close(fd);
        }

        return data;
}

static void * reading_thread(void *args)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) args;
        int index = 0;

        bool paused = false;

        ///while(index < s->count && !s->finish_threads) {
        while(1) {
                pthread_mutex_lock(&s->lock);
                {
                        while((s->queue_len >= BUFFER_LEN_MAX - 1 || index >= s->count || paused)
                                       && s->message_queue.len == 0) {
                                s->worker_waiting = true;
                                if (index >= s->count) {
                                        s->finished = true;
                                }
                                pthread_cond_wait(&s->worker_cv, &s->lock);
                                s->worker_waiting = false;
                        }

                        while(s->message_queue.len > 0) {
                                struct message *msg = pop_message(&s->message_queue);
                                if(msg->type == FINALIZE) {
                                        pthread_mutex_unlock(&s->lock);
                                        free(msg);
                                        goto exited;
                                } else if(msg->type == PAUSE) {
                                        paused = !paused;
                                        printf("Toggle pause\n");

                                        index -= flush_processed(s->head);
                                        s->queue_len = 0;
                                        s->head = s->tail = NULL;

                                        free(msg);

                                        pthread_mutex_unlock(&s->lock);
                                        goto end_loop;
                                } else if (msg->type == SEEK) {
                                        flush_processed(s->head);
                                        s->queue_len = 0;
                                        s->head = s->tail = NULL;

                                        struct seek_data *data = msg->data;
                                        free(msg);
                                        if(data->whence == IMPORT_SEEK_CUR) {
                                                index += data->offset;
                                        } else if (data->whence == IMPORT_SEEK_SET) {
                                                index = data->offset;
                                        } else if (data->whence == IMPORT_SEEK_END) {
                                                index = s->count + data->offset;
                                        }
                                        printf("Current index: frame %d\n", index);
                                        free(data);
                                } else {
                                        fprintf(stderr, "Unknown message type: %d!\n", msg->type);
                                        abort();
                                }
                        }
                }
                pthread_mutex_unlock(&s->lock);

                /// @todo are these checks necessary?
                if(index < 0) {
                        index = 0;
                }

                if(index >= s->count) {
                        fprintf(stderr, "Warning: Index exceeds available frame count!\n");
                        index = s->count - 1;
                }

                struct video_reader_data data_reader[MAX_NUMBER_WORKERS];
                task_result_handle_t task_handle[MAX_NUMBER_WORKERS];

                int number_workers = s->video_reading_threads_count;
                if (index + number_workers >= s->count) {
                        number_workers = s->count - index;
                }
                // run workers
                for (int i = 0; i < number_workers; ++i) {
                        struct video_reader_data *data =
                                &data_reader[i];
                        data->o_direct = s->o_direct;
                        data->tile_count = s->video_desc.tile_count;
                        snprintf(data->file_name_prefix, sizeof(data->file_name_prefix),
                                        "%s/%08d", s->directory, index + i + 1);
                        strncpy(data->file_name_suffix,
                                        get_codec_file_extension(s->video_desc.color_spec),
                                        sizeof(data->file_name_suffix));
                        data->entry = NULL;
                        task_handle[i] = task_run_async(video_reader_callback, data);
                }

                // wait for workers to finish
                for (int i = 0; i < number_workers; ++i) {
                        struct video_reader_data *data =
                                (struct video_reader_data *)
                                wait_task(task_handle[i]);
                        if (!data || data->entry == NULL)
                                continue;
                        pthread_mutex_lock(&s->lock);
                        {
                                if(s->head) {
                                        s->tail->next = data->entry;
                                        s->tail = data->entry;
                                } else {
                                        s->head = s->tail = data->entry;
                                }
                                s->queue_len += 1;

                                if(s->boss_waiting)
                                        pthread_cond_signal(&s->boss_cv);
                        }
                        pthread_mutex_unlock(&s->lock);
                }
                index += number_workers;
end_loop:
                ;
        }

exited:

        return NULL;
}

static void reset_import(struct vidcap_import_state *s)
{
        exit_reading_threads(s);

        s->finished = false;

        if(pthread_create(&s->thread_id, NULL, reading_thread, (void *) s) != 0) {
                fprintf(stderr, "Unable to create thread.\n");
                /// @todo what to do here
                abort();
        }
        if (s->audio_state.has_audio) {
                if(pthread_create(&s->audio_state.thread_id, NULL, audio_reading_thread, (void *) s) != 0) {
                        fprintf(stderr, "Unable to create thread.\n");
                        abort();
                }
        }
}

static void vidcap_import_dispose_video_frame(struct video_frame *frame) {
        free_entry(frame->dispose_udata);
        vf_free(frame);
}

struct video_frame *
vidcap_import_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) state;
        struct timeval cur_time;
        struct video_frame *ret;
        
        struct processed_entry *current = NULL;

        pthread_mutex_lock(&s->lock);
        {
                if(s->queue_len == 0) {
                        if (s->finished == true && s->loop) {
                                pthread_mutex_unlock(&s->lock);
                                reset_import(s);
                        }

                        s->boss_waiting = true;
                        struct timespec ts;
                        struct timeval  tp;

                        // get time for timeout
                        gettimeofday(&tp, NULL);

                        /* Convert from timeval to timespec */
                        ts.tv_sec  = tp.tv_sec;
                        ts.tv_nsec = tp.tv_usec * 1000;
                        ts.tv_nsec += 2 * 1/s->video_desc.fps * 1000 * 1000 * 1000;
                        // make it correct
                        ts.tv_sec += ts.tv_nsec / 1000000000;
                        ts.tv_nsec = ts.tv_nsec % 1000000000;

                        pthread_cond_timedwait(&s->boss_cv, &s->lock, &ts);
                        s->boss_waiting = false;
                }
 
                if(s->queue_len == 0) {
                        pthread_mutex_unlock(&s->lock);
                        return NULL;
                }

                current = s->head;
                assert(current != NULL);

                s->head = s->head->next;
                s->queue_len -= 1;

                if(s->worker_waiting)
                        pthread_cond_signal(&s->worker_cv);

                ret = vf_alloc_desc(s->video_desc);
                ret->dispose = vidcap_import_dispose_video_frame;
                ret->dispose_udata = current;
                for (unsigned int i = 0; i < s->video_desc.tile_count; ++i) {
                        ret->tiles[i].data_len =
                                current->tiles[i].data_len;
                        ret->tiles[i].data = current->tiles[i].data;
                }
        }
        pthread_mutex_unlock(&s->lock);


        // audio
        if(s->audio_state.has_audio) {
                unsigned long long int requested_samples = (unsigned long long int) (s->frames + 1) *
                        s->audio_frame.sample_rate / s->video_desc.fps - s->audio_state.played_samples;
                if((int) (s->audio_state.played_samples + requested_samples) > s->audio_state.total_samples) {
                        requested_samples = s->audio_state.total_samples - s->audio_state.played_samples;
                }
                unsigned long long int requested_bytes = requested_samples * s->audio_frame.bps * s->audio_frame.ch_count;
                if(requested_bytes) {
                        pthread_mutex_lock(&s->audio_state.lock);
                        {
                                while(ring_get_current_size(s->audio_state.data) < (int) requested_bytes) {
                                        s->audio_state.boss_waiting = true;
                                        pthread_cond_wait(&s->audio_state.boss_cv, &s->audio_state.lock);
                                        s->audio_state.boss_waiting = false;
                                }

                                int ret = ring_buffer_read(s->audio_state.data, s->audio_frame.data, requested_bytes);

                                assert(ret == (int) requested_bytes);
                                s->audio_frame.data_len = requested_bytes;

                                pthread_cond_signal(&s->audio_state.worker_cv);
                        }
                        pthread_mutex_unlock(&s->audio_state.lock);
                }

                s->audio_state.played_samples += requested_samples;
                *audio = &s->audio_frame;
        } else {
                *audio = NULL;
        }


        gettimeofday(&cur_time, NULL);
        while(tv_diff_usec(cur_time, s->prev_time) < 1000000.0 / s->video_desc.fps) {
                gettimeofday(&cur_time, NULL);
        }
        //tv_add_usec(&s->prev_time, 1000000.0 / s->frame->fps);
        s->prev_time = cur_time;

        double seconds = tv_diff(cur_time, s->t0);
        if (seconds >= 5) {
                float fps  = (s->frames - s->frames_prev) / seconds;
                fprintf(stderr, "[import] %d frames in %g seconds = %g FPS\n", s->frames - s->frames_prev, seconds, fps);
                s->t0 = cur_time;
                s->frames_prev = s->frames;
        }

        s->frames += 1;

	return ret;
}

