/**
 * @file   video_capture/import.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2015 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"
#include "audio/wav_reader.h"
#include "utils/ring_buffer.h"
#include "utils/worker.h"
#include "video_export.h"
//#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sstream>
#include <string>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <condition_variable>
#include <chrono>
#include <mutex>

#define BUFFER_LEN_MAX 40
#define MAX_CLIENTS 16

#define CONTROL_PORT 15004
#define VIDCAP_IMPORT_ID 0x76FA7F6D

#define PIPE "/tmp/ultragrid_import.fifo"

#define MAX_NUMBER_WORKERS 100

using std::condition_variable;
using std::chrono::duration;
using std::min;
using std::max;
using std::mutex;
using std::ostringstream;
using std::string;
using std::unique_lock;

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

        condition_variable worker_cv;
        condition_variable boss_cv;

        mutex lock;
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

        mutex lock;
        condition_variable worker_cv;
        condition_variable boss_cv;
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
        bool should_exit_at_end;
        double force_fps;

        volatile bool exit_control = false;
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

static void cleanup_common(struct vidcap_import_state *s);

static void message_queue_clear(struct message_queue *queue) {
        queue->head = queue->tail = NULL;
        queue->len = 0;
}

static struct vidcap_type *
vidcap_import_probe(bool verbose)
{
        UNUSED(verbose);
	struct vidcap_type*		vt;
    
	vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->name        = "import";
		vt->description = "Video importer (not to be called directly)";
	}
	return vt;
}

#define READ_N(buf, len) if (fread(buf, len, 1, audio_file) != 1) goto error_format;

static bool init_audio(struct vidcap_import_state *s, char *audio_filename)
{
        FILE *audio_file = fopen(audio_filename, "rb");
        if(!audio_file) {
                perror("Cannot open audio file");
                return false;
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
        s->audio_frame.data = (char *) malloc(s->audio_frame.max_size);

        s->audio_state.file = audio_file;

        s->audio_state.played_samples = 0;

        return true;

error_format:
        fprintf(stderr, "Audio format file error - unknown format\n");
        fclose(audio_file);
        return false;
}

static int
vidcap_import_init(const struct vidcap_params *params, void **state)
{
	struct vidcap_import_state *s = NULL;
        FILE *info = NULL; // metadata file
        char *tmp = strdup(vidcap_params_get_fmt(params));
        bool disable_audio = false;

try {
	printf("vidcap_import_init\n");

        s = new vidcap_import_state();
        s->head = s->tail = NULL;
        s->queue_len = 0;
        s->frames_prev = s->frames = 0;
        gettimeofday(&s->t0, NULL);

        s->video_reading_threads_count = 1; // default is single threaded

        char *save_ptr = NULL;
        s->directory = strdup(strtok_r(tmp, ":", &save_ptr));
        char *suffix;
        if (!s->directory || strcmp(s->directory, "help") == 0) {
                throw string("Import usage:\n"
                                "\t<directory>{:loop|:mt_reading=<nr_threads>|:o_direct|:exit_at_end:fps=<fps>|:disable_audio}\n"
                                "\t\t<fps> - overrides FPS from sequence metadata\n");
        }
        while ((suffix = strtok_r(NULL, ":", &save_ptr)) != NULL) {
                if (suffix[0] == '\\') { // MSW path
                        assert(strlen(s->directory) == 1); // c:\something -> should be 'c'
                        char *tmp = (char *) malloc(2 + strlen(suffix) + 1);
                        sprintf(tmp, "%c:%s", s->directory[0], suffix);
                        free(s->directory);
                        s->directory = tmp;
                } else if (strcmp(suffix, "loop") == 0) {
                        s->loop = true;
                } else if (strncmp(suffix, "mt_reading=",
                                        strlen("mt_reading=")) == 0) {
                        s->video_reading_threads_count = atoi(suffix +
                                        strlen("mt_reading="));
                        assert(s->video_reading_threads_count <=
                                        MAX_NUMBER_WORKERS);
                } else if (strcmp(suffix, "o_direct") == 0) {
                        s->o_direct = true;
                } else if (strcmp(suffix, "noaudio") == 0) {
                        disable_audio = true;
                } else if (strcmp(suffix, "exit_at_end") == 0) {
                        s->should_exit_at_end = true;
                } else if (strncmp(suffix, "fps=", strlen("fps=")) == 0) {
                        s->force_fps = atof(suffix + strlen("fps="));
                } else {
                        throw string("[Playback] Unrecognized"
                                        " option ") + suffix + ".\n";
                }
        }
        free(tmp);
        tmp = NULL;

        message_queue_clear(&s->message_queue);
        message_queue_clear(&s->audio_state.message_queue);

        char *audio_filename = (char *) malloc(strlen(s->directory) + sizeof("/soud.wav") + 1);
        assert(audio_filename != NULL);
        strcpy(audio_filename, s->directory);
        strcat(audio_filename, "/sound.wav");
        if((vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) && !disable_audio && init_audio(s, audio_filename)) {
                s->audio_state.has_audio = true;
                if(pthread_create(&s->audio_state.thread_id, NULL, audio_reading_thread, (void *) s) != 0) {
                        free(audio_filename);
                        throw string("Unable to create thread.\n");
                }
        } else {
                s->audio_state.has_audio = false;
        }
        free(audio_filename);
        
        char *info_filename = (char *) malloc(strlen(s->directory) + sizeof("/video.info") + 1);
        assert(info_filename != NULL);
        strcpy(info_filename, s->directory);
        strcat(info_filename, "/video.info");

        info = fopen(info_filename, "r");
        free(info_filename);
        if(info == NULL) {
                perror("[import] Failed to open index file");
                throw string();
        }

        struct video_desc desc;
        memset(&desc, 0, sizeof desc);

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
                                throw string("[import] cannot read version line.\n");
                        }
                        if(version != VIDEO_EXPORT_SUMMARY_VERSION) {
                                ostringstream oss;
                                oss << "[import] Invalid version " << version << ".\n";
                                throw oss.str();
                        }
                        items_found |= 1<<0;
                } else if(strncmp(line, "width ", strlen("width ")) == 0) {
                        long int width = strtol(line + strlen("width "), (char **) NULL, 10);
                        if(width == LONG_MIN || width == LONG_MAX) {
                                throw string("[import] cannot read video width.\n");
                        }
                        desc.width = width;
                        items_found |= 1<<1;
                } else if(strncmp(line, "height ", strlen("height ")) == 0) {
                        long int height = strtol(line + strlen("height "), (char **) NULL, 10);
                        if(height == LONG_MIN || height == LONG_MAX) {
                                throw string("[import] cannot read video height.\n");
                        }
                        desc.height = height;
                        items_found |= 1<<2;
                } else if(strncmp(line, "fourcc ", strlen("fourcc ")) == 0) {
                        char *ptr = line + strlen("fourcc ");
                        if(strlen(ptr) != 5) { // including '\n'
                                throw string("[import] cannot read video FourCC tag.\n");
                        }
                        uint32_t fourcc;
                        memcpy((void *) &fourcc, ptr, sizeof(fourcc));
                        desc.color_spec = get_codec_from_fcc(fourcc);
                        if(desc.color_spec == VIDEO_CODEC_NONE) {
                                throw string("[import] Requested codec not known.\n");
                        }
                        items_found |= 1<<3;
                } else if(strncmp(line, "fps ", strlen("fps ")) == 0) {
                        char *ptr = line + strlen("fps ");
                        desc.fps = strtod(ptr, NULL);
                        if(desc.fps == HUGE_VAL || desc.fps <= 0) {
                                throw string("[import] Invalid FPS.\n");
                        }
                        items_found |= 1<<4;
                } else if(strncmp(line, "interlacing ", strlen("interlacing ")) == 0) {
                        char *ptr = line + strlen("interlacing ");
                        desc.interlacing = (interlacing_t) atoi(ptr);
                        if(desc.interlacing > 4) {
                                throw string("[import] Invalid interlacing.\n");
                        }
                        items_found |= 1<<5;
                } else if(strncmp(line, "count ", strlen("count ")) == 0) {
                        char *ptr = line + strlen("count ");
                        s->count = atoi(ptr);
                        items_found |= 1<<6;
                }
        }

        // override metadata fps setting
        if (s->force_fps > 0.0) {
                desc.fps = s->force_fps;
        }

        assert(desc.color_spec != VIDEO_CODEC_NONE && desc.width != 0 && desc.height != 0 && desc.fps != 0.0 &&
                        s->count != 0);

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
                        throw string("Unable to open first file of "
                                        "the video sequence.\n");
                }
        }

        s->video_desc = desc;

        fclose(info);
        info = NULL;

        if(items_found != (1 << 7) - 1) {
                throw string("[import] Failed while reading config file - some items missing.\n");
        }

        if(pthread_create(&s->thread_id, NULL, reading_thread, (void *) s) != 0) {
                throw string("Unable to create thread.\n");
        }

#ifndef WIN32
        if(pthread_create(&s->control_thread_id, NULL, control_thread, (void *) s) != 0) {
                throw string("Unable to create control thread.\n");
        }
#endif

        gettimeofday(&s->prev_time, NULL);

        *state = s;
	return VIDCAP_INIT_OK;
} catch (string const & str) {
        fprintf(stderr, "%s", str.c_str());
        free(tmp);
        if (info != NULL)
                fclose(info);
        cleanup_common(s);
        delete s;
        return VIDCAP_INIT_FAIL;
}
}

static void exit_reading_threads(struct vidcap_import_state *s)
{
        struct message *msg = (struct message *) malloc(sizeof(struct message));

        msg->type = FINALIZE;
        msg->data = NULL;
        msg->data_len = 0;
        msg->next = NULL;

        {
                unique_lock<mutex> lk(s->lock);
                send_message(msg, &s->message_queue);
                lk.unlock();
                s->worker_cv.notify_one();
        }

	pthread_join(s->thread_id, NULL);

        // audio
        if(s->audio_state.has_audio) {
                struct message *msg = (struct message *) malloc(sizeof(struct message));

                msg->type = FINALIZE;
                msg->data = NULL;
                msg->data_len = 0;
                msg->next = NULL;

                {
                        unique_lock<mutex> lk(s->audio_state.lock);
                        send_message(msg, &s->audio_state.message_queue);
                        lk.unlock();
                        s->audio_state.worker_cv.notify_one();
                }

                pthread_join(s->audio_state.thread_id, NULL);
        }
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
        s->exit_control = true;

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

static void cleanup_common(struct vidcap_import_state *s) {
        flush_processed(s->head);

        free(s->directory);

        // audio
        if(s->audio_state.has_audio) {
                ring_buffer_destroy(s->audio_state.data);

                free(s->audio_frame.data);

                fclose(s->audio_state.file);
        }
}

static void vidcap_import_done(void *state)
{
	struct vidcap_import_state *s = (struct vidcap_import_state *) state;
	assert(s != NULL);

        vidcap_import_finish(state);

        cleanup_common(s);
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
                struct message *msg = (struct message *) malloc(sizeof(struct message));
                msg->type = PAUSE;
                msg->data = NULL;
                msg->data_len = 0;
                msg->next = NULL;

                unique_lock<mutex> lk(s->lock);
                send_message(msg, &s->message_queue);
                lk.unlock();
                s->worker_cv.notify_one();
        } else if(strncasecmp(message, "seek ", strlen("seek ")) == 0) {
                if(s->audio_state.has_audio == true) {
                        fprintf(stderr, "Seeking now allowed if we have audio. (Not yet implemented)\n");
                        return;
                }

                char *time_spec = message + strlen("seek ");

                struct message *msg = (struct message *) malloc(sizeof(struct message));
                struct seek_data *data = (struct seek_data *) malloc(sizeof(struct seek_data));
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
                        audio_msg = (struct message *) malloc(sizeof(struct message));
                        memcpy(audio_msg, msg, sizeof(struct message));

                        if(audio_msg->data) { // deep copy
                                audio_msg->data = malloc(msg->data_len);
                                memcpy(audio_msg->data, msg->data, msg->data_len);
                        }
                }

                {
                        unique_lock<mutex> lk(s->lock);
                        send_message(msg, &s->message_queue);
                        lk.unlock();
                        s->worker_cv.notify_one();
                }

                if (s->audio_state.has_audio) {
                        unique_lock<mutex> lk(s->audio_state.lock);
                        send_message(audio_msg, &s->audio_state.message_queue);
                        lk.unlock();
                        s->audio_state.worker_cv.notify_one();
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
        fd_t fd;
        int rc;

        fd = socket(AF_INET6, SOCK_STREAM, 0);
        assert(fd != INVALID_SOCKET);
        int val = 1;
        rc = setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &val, sizeof(val));
        if (rc != 0) {
                perror("Video import - setsockopt");
        }
        struct sockaddr_in6 s_in;
        s_in.sin6_family = AF_INET6;
        s_in.sin6_addr = in6addr_any;
        s_in.sin6_port = htons(CONTROL_PORT);

        rc = bind(fd, (const struct sockaddr *) &s_in, sizeof(s_in));
        if (rc != 0) {
                perror("Video import: unable to bind communication pipe");
                CLOSESOCKET(fd);
                return NULL;
        }
        listen(fd, MAX_CLIENTS);
        struct sockaddr_storage client_addr;
        socklen_t len;

        unlink(PIPE);
        errno = 0;
        rc = mkfifo(PIPE, 0777);

        struct client *clients = NULL;
        if (rc == 0) {
                clients = (struct client *) malloc(sizeof(struct client));
                clients->fd = open(PIPE, O_RDONLY | O_NONBLOCK);
                assert(clients->fd != -1);
                clients->pipe = true;
                clients->buff_len = 0;
                clients->next = NULL;
        } else {
                perror("Video import: unable to create communication pipe");
                CLOSESOCKET(fd);
                return NULL;
        }

        while (!s->exit_control) {
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
                                int new_fd = accept(fd, (struct sockaddr *) &client_addr, &len);
                                if (new_fd != -1) {
                                        struct client *new_client = (struct client *) malloc(sizeof(struct client));
                                        new_client->fd = new_fd;
                                        new_client->next = clients;
                                        new_client->buff_len = 0;
                                        new_client->pipe = false;
                                        clients = new_client;
                                } else {
                                        perror("Control socket: cannot accept new connection");
                                }
                        }

                        struct client **parent_ptr = &clients;
                        struct client *cur = clients;

                        while(cur) {
                                if(FD_ISSET(cur->fd, &set)) {
                                        ssize_t ret = read(cur->fd, cur->buff + cur->buff_len, 1024 - cur->buff_len);
                                        if(ret == -1) {
                                                perror("Error reading socket");
                                        }
                                        if(ret == 0) {
                                                if(!cur->pipe) {
                                                        CLOSESOCKET(cur->fd);
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
                CLOSESOCKET(cur->fd);
                cur = cur->next;
                free(tmp);
        }

        CLOSESOCKET(fd);
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
                {
                        unique_lock<mutex> lk(s->audio_state.lock);
                        while((ring_get_current_size(s->audio_state.data) > ring_get_size(s->audio_state.data) * 2 / 3 ||
                                                s->audio_state.samples_read >= s->audio_state.total_samples)
                                        && s->audio_state.message_queue.len == 0) {
                                s->audio_state.worker_cv.wait(lk);
                        }

                        while(s->audio_state.message_queue.len > 0) {
                                struct message *msg = pop_message(&s->audio_state.message_queue);
                                if(msg->type == FINALIZE) {
                                        free(msg);
                                        return NULL;
                                } else if (msg->type == SEEK) {
                                        fprintf(stderr, "Seeking in audio import not yet implemented.\n");
                                        abort();
                                }
                        }

                        max_read = ring_get_size(s->audio_state.data) -
                                        ring_get_current_size(s->audio_state.data) - 1;

                }

                char *buffer = (char *) malloc(max_read);

                size_t ret = fread(buffer, s->audio_frame.ch_count * s->audio_frame.bps,
                                max_read / s->audio_frame.ch_count / s->audio_frame.bps, s->audio_state.file);
                s->audio_state.samples_read += ret;

                {
                        unique_lock<mutex> lk(s->audio_state.lock);
                        ring_buffer_write(s->audio_state.data, buffer, ret * s->audio_frame.ch_count * s->audio_frame.bps);
                        lk.unlock();
                        s->audio_state.boss_cv.notify_one();
                }

                free(buffer);
        }

        return NULL;
}

struct video_reader_data {
        char file_name_prefix[512];
        char file_name_suffix[512];
        unsigned int tile_count;
        struct processed_entry *entry;
        bool o_direct;
};

#define ALLOC_ALIGN 512

static void *video_reader_callback(void *arg)
{
        struct video_reader_data *data =
                (struct video_reader_data *) arg;
       
        data->entry = (struct processed_entry *) calloc(1, sizeof(struct processed_entry) + data->tile_count * sizeof(struct tile_data));
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

                int flags = O_RDONLY;
#ifdef WIN32
                flags |= O_BINARY;
#endif
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
                if (fstat(fd, &sb)) {
                        perror("fstat");
                        close(fd);
                        free_entry(data->entry);
                        return NULL;
                }

                data->entry->tiles[i].data_len = sb.st_size;
                const int aligned_data_len = (data->entry->tiles[i].data_len + ALLOC_ALIGN - 1)
                        / ALLOC_ALIGN * ALLOC_ALIGN;
                // alignment needed when using O_DIRECT flag
                data->entry->tiles[i].data = (char *)
                        aligned_malloc(aligned_data_len, ALLOC_ALIGN);
                assert(data->entry->tiles[i].data != NULL);

                ssize_t bytes = 0;
                do {
                        ssize_t res = read(fd, data->entry->tiles[i].data + bytes,
                                        (data->entry->tiles[i].data_len - bytes + ALLOC_ALIGN - 1)
                                        / ALLOC_ALIGN * ALLOC_ALIGN);
                        if (res <= 0) {
                                perror("read");
                                for (unsigned int i = 0; i < data->tile_count; i++) {
                                        aligned_free(data->entry->tiles[i].data);
                                }
                                free(data->entry);
                                close(fd);
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
                {
                        unique_lock<mutex> lk(s->lock);
                        while((s->queue_len >= BUFFER_LEN_MAX - 1 || index >= s->count || paused)
                                       && s->message_queue.len == 0) {
                                if (index >= s->count) {
                                        s->finished = true;
                                }
                                s->worker_cv.wait(lk);
                        }

                        while(s->message_queue.len > 0) {
                                struct message *msg = pop_message(&s->message_queue);
                                if(msg->type == FINALIZE) {
                                        free(msg);
                                        return NULL;
                                } else if(msg->type == PAUSE) {
                                        paused = !paused;
                                        printf("Toggle pause\n");

                                        index -= flush_processed(s->head);
                                        s->queue_len = 0;
                                        s->head = s->tail = NULL;

                                        free(msg);
                                } else if (msg->type == SEEK) {
                                        flush_processed(s->head);
                                        s->queue_len = 0;
                                        s->head = s->tail = NULL;

                                        struct seek_data *data = (struct seek_data *) msg->data;
                                        free(msg);
                                        if(data->whence == IMPORT_SEEK_CUR) {
                                                index += data->offset;
                                        } else if (data->whence == IMPORT_SEEK_SET) {
                                                index = data->offset;
                                        } else if (data->whence == IMPORT_SEEK_END) {
                                                index = s->count + data->offset;
                                        }
                                        index = min(max(0, index), s->count - 1);
                                        printf("Current index: frame %d\n", index);
                                        free(data);
                                } else {
                                        fprintf(stderr, "Unknown message type: %d!\n", msg->type);
                                        abort();
                                }
                        }
                }

                /// @todo are these checks necessary?
                index = min(max(0, index), s->count - 1);

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
                        {
                                unique_lock<mutex> lk(s->lock);
                                if(s->head) {
                                        s->tail->next = data->entry;
                                        s->tail = data->entry;
                                } else {
                                        s->head = s->tail = data->entry;
                                }
                                s->queue_len += 1;

                                lk.unlock();
                                s->boss_cv.notify_one();
                        }
                }
                index += number_workers;
        }

        return NULL;
}

static void reset_import(struct vidcap_import_state *s)
{
        exit_reading_threads(s);

        s->finished = false;

        // clear audio state
        /// @todo
        /// This stuff is very ugly, rewrite it
        if (s->audio_state.has_audio) {
                s->audio_state.played_samples = 0;
                s->audio_state.samples_read = 0;
                ring_buffer_flush(s->audio_state.data);
                fseek(s->audio_state.file, 0L, SEEK_SET);
                struct wav_metadata metadata;
                read_wav_header(s->audio_state.file, &metadata); // skip metadata
        }
        s->frames_prev = s->frames = 0;

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
        free_entry((struct processed_entry *) frame->dispose_udata);
        vf_free(frame);
}

static struct video_frame *
vidcap_import_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) state;
        struct timeval cur_time;
        struct video_frame *ret;
        
        struct processed_entry *current = NULL;

        {
                unique_lock<mutex> lk(s->lock);
                if(s->queue_len == 0) {
                        if (s->finished == true && s->loop) {
                                lk.unlock();
                                reset_import(s);
                                lk.lock();
                        }
                        if (s->finished == true && s->should_exit_at_end == true) {
                                exit_uv(0);
                        }

                        s->boss_cv.wait_for(lk, duration<double>(2 * 1/s->video_desc.fps));
                }
 
                if(s->queue_len == 0) {
                        return NULL;
                }

                current = s->head;
                assert(current != NULL);

                s->head = s->head->next;
                s->queue_len -= 1;

                lk.unlock();
                s->worker_cv.notify_one();

                ret = vf_alloc_desc(s->video_desc);
                ret->dispose = vidcap_import_dispose_video_frame;
                ret->dispose_udata = current;
                for (unsigned int i = 0; i < s->video_desc.tile_count; ++i) {
                        ret->tiles[i].data_len =
                                current->tiles[i].data_len;
                        ret->tiles[i].data = current->tiles[i].data;
                }
        }


        // audio
        if(s->audio_state.has_audio) {
                unsigned long long int requested_samples = (unsigned long long int) (s->frames + 1) *
                        s->audio_frame.sample_rate / s->video_desc.fps - s->audio_state.played_samples;
                if((int) (s->audio_state.played_samples + requested_samples) > s->audio_state.total_samples) {
                        requested_samples = s->audio_state.total_samples - s->audio_state.played_samples;
                }
                unsigned long long int requested_bytes = requested_samples * s->audio_frame.bps * s->audio_frame.ch_count;
                if(requested_bytes) {
                        {
                                requested_bytes = min<unsigned long long>(requested_bytes,s->audio_frame.max_size);
                                unique_lock<mutex> lk(s->audio_state.lock);
                                while(ring_get_current_size(s->audio_state.data) < (int) requested_bytes) {
                                        s->audio_state.boss_cv.wait(lk);;
                                }

                                int ret = ring_buffer_read(s->audio_state.data, s->audio_frame.data, requested_bytes);

                                assert(ret == (int) requested_bytes);

                                lk.unlock();
                                s->audio_state.worker_cv.notify_one();
                        }
                }

                s->audio_frame.data_len = requested_bytes;
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
                log_msg(LOG_LEVEL_INFO, "[import] %d frames in %g seconds = %g FPS\n", s->frames - s->frames_prev, seconds, fps);
                s->t0 = cur_time;
                s->frames_prev = s->frames;
        }

        s->frames += 1;

	return ret;
}

static const struct video_capture_info vidcap_import_info = {
        vidcap_import_probe,
        vidcap_import_init,
        vidcap_import_done,
        vidcap_import_grab,
};

REGISTER_MODULE(import, &vidcap_import_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

