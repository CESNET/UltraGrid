/**
 * @file   video_capture/import.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2023 CESNET, z. s. p. o.
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
/**
 * @file
 * @todo
 * There is a race condition between audio and video when seeking the stream.
 * Audio may be seeked and video not yet (or vice versa). Since those parts are
 * independent, ther will perhaps be needed to have _getf() and seek completion
 * mutually exclusive. Perhaps not much harmfull, seems that it could cause at
 * most 1 frame AV-desync.
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

#include "audio/types.h"
#include "audio/wav_reader.h"
#include "keyboard_control.h"
#include "messaging.h"
#include "module.h"
#include "playback.h"
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/ring_buffer.h"
#include "utils/worker.h"
#include "video_export.h"

#include <pthread.h>
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

#define BUFFER_LEN_MAX 40
#define MAX_CLIENTS 16

#define VIDCAP_IMPORT_ID 0x76FA7F6D

#define PIPE "/tmp/ultragrid_import.fifo"

#define MAX_NUMBER_WORKERS 100
#define MOD_NAME "[import] "

using std::min;
using std::max;
using std::string;
using std::to_string;

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
} import_message_t;

struct import_message;

struct import_message {
        import_message_t       type;
        void           *data;
        size_t          data_len;
        struct import_message *next;
};

typedef enum {
        IMPORT_SEEK_SET,
        IMPORT_SEEK_CUR,
} seek_direction_t;

struct seek_data {
        seek_direction_t whence;
        ssize_t offset;
};

struct message_queue {
        struct import_message *head;
        struct import_message *tail;
        size_t          len;
};

struct audio_state {
        bool has_audio = false;
        FILE *file;
        struct wav_metadata metadata;
        ring_buffer_t *data;
        int total_samples;
        int samples_read;
        pthread_t thread_id;

        pthread_cond_t worker_cv;
        pthread_cond_t boss_cv;

        pthread_mutex_t lock;
        long long int played_samples = 0;
        int video_frames_played = 0;

        struct message_queue message_queue;
}; 

struct vidcap_import_state {
        struct module mod;
        struct module *parent;
        struct audio_frame audio_frame;
        struct audio_state audio_state;
        struct video_desc video_desc;
        char *directory;
        char tile_delim; // eg. '_' for format "00000001_0.yuv"

        struct message_queue message_queue;

        pthread_mutex_t lock;
        pthread_cond_t worker_cv;
        pthread_cond_t boss_cv;
        struct processed_entry * volatile head, * volatile tail;
        volatile int queue_len;

        pthread_t video_thread_id;

        struct timeval prev_time;
        long video_frame_count = 0L;

        bool has_video = true;
        bool finished;
        bool loop;
        bool o_direct;
        int video_reading_threads_count;
        bool should_exit_at_end;
        double force_fps;
};

static void * audio_reading_thread(void *args);
static void * video_reading_thread(void *args);
static void send_message(struct import_message *msg, struct message_queue *queue);
static struct import_message* pop_message(struct message_queue *queue);
static int flush_processed(struct processed_entry *list);
static void message_queue_clear(struct message_queue *queue);
static void vidcap_import_new_message(struct module *);
static void process_msg(struct vidcap_import_state *state, const char *message);

static void cleanup_common(struct vidcap_import_state *s);

static void message_queue_clear(struct message_queue *queue) {
        queue->head = queue->tail = NULL;
        queue->len = 0;
}

static void vidcap_import_probe(device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_cards = nullptr;
        *count = 0;
}

#define READ_N(buf, len) if (fread(buf, len, 1, audio_file) != 1) goto error_format;

static bool init_audio(struct vidcap_import_state *s, const char *audio_filename)
{
        FILE *audio_file = fopen(audio_filename, "rb");
        if(!audio_file) {
                perror("Cannot open audio file");
                return false;
        }

        int ret = read_wav_header(audio_file, &s->audio_state.metadata);
        if (ret != WAV_HDR_PARSE_OK) {
                        log_msg(LOG_LEVEL_ERROR, "%s!\n", get_wav_error(ret));
                        goto error_format;
        }

        s->audio_frame.ch_count = s->audio_state.metadata.ch_count;
        s->audio_frame.sample_rate = s->audio_state.metadata.sample_rate;
        s->audio_frame.bps = s->audio_state.metadata.bits_per_sample / 8;
        s->audio_state.total_samples = s->audio_state.metadata.data_size / s->audio_frame.bps / s->audio_frame.ch_count;
        s->audio_state.samples_read = 0;

        s->audio_state.data = ring_buffer_init(s->audio_frame.bps * s->audio_frame.sample_rate *
                        s->audio_frame.ch_count * 180);

        s->audio_frame.max_size = s->audio_frame.bps * s->audio_frame.sample_rate * s->audio_frame.ch_count;
        s->audio_frame.data_len = 0;
        s->audio_frame.data = (char *) malloc(s->audio_frame.max_size);

        s->audio_state.file = audio_file;

        pthread_cond_init(&s->audio_state.worker_cv, NULL);
        pthread_cond_init(&s->audio_state.boss_cv, NULL);
        pthread_mutex_init(&s->audio_state.lock, NULL);

        return true;

error_format:
        fprintf(stderr, "Audio format file error - unknown format\n");
        fclose(audio_file);
        return false;
}

/// @param prefix option name including " " (eg. "width ")
static long strtol_checked(const char *line, const char *prefix, long min_val, long max_val) {
        using namespace std::string_literals;
        long int val = strtol(line + strlen(prefix), static_cast<char **>(nullptr), 10);
        if (val == LONG_MIN || val == LONG_MAX) {
                throw ug_runtime_error("cannot read "s + prefix + "line.");
        }
        if (val < min_val || val > max_val) {
                throw ug_runtime_error(string(prefix) + "out of range [" + to_string(min_val) + ".." + to_string(max_val) + "]");
        }
        return val;
}

static video_desc parse_video_desc_info(FILE *info, long *video_frame_count) {
        struct video_desc desc{};

        char line[512];
        uint32_t items_found = 0;
        while (!feof(info)) {
                if (fgets(line, sizeof(line), info) == nullptr) {
                        // empty line
                        continue;
                }
                if(strncmp(line, "version ", strlen("version ")) == 0) {
                        strtol_checked(line, "version ", VIDEO_EXPORT_SUMMARY_VERSION, VIDEO_EXPORT_SUMMARY_VERSION);
                        items_found |= 1U<<0U;
                } else if(strncmp(line, "width ", strlen("width ")) == 0) {
                        desc.width = strtol_checked(line, "width ", 0, INT_MAX);
                        items_found |= 1U<<1U;
                } else if(strncmp(line, "height ", strlen("height ")) == 0) {
                        desc.height = strtol_checked(line, "height ", 0, INT_MAX);
                        items_found |= 1U<<2U;
                } else if(strncmp(line, "fourcc ", strlen("fourcc ")) == 0) {
                        char *ptr = line + strlen("fourcc ");
                        if(strlen(ptr) != 5) { // including '\n'
                                throw ug_runtime_error("cannot read video FourCC tag.");
                        }
                        uint32_t fourcc = 0U;
                        memcpy((void *) &fourcc, ptr, sizeof(fourcc));
                        desc.color_spec = get_codec_from_fcc(fourcc);
                        if(desc.color_spec == VIDEO_CODEC_NONE) {
                                throw ug_runtime_error("Requested codec not known.");
                        }
                        items_found |= 1U<<3U;
                } else if(strncmp(line, "fps ", strlen("fps ")) == 0) {
                        char *ptr = line + strlen("fps ");
                        desc.fps = strtod(ptr, nullptr);
                        if(desc.fps == HUGE_VAL || desc.fps <= 0) {
                                throw ug_runtime_error("Invalid FPS.");
                        }
                        items_found |= 1U<<4U;
                } else if(strncmp(line, "interlacing ", strlen("interlacing ")) == 0) {
                        desc.interlacing = (interlacing_t) strtol_checked(line, "interlacing ", 0, INTERLACING_MAX);
                        items_found |= 1U<<5U;
                } else if(strncmp(line, "count ", strlen("count ")) == 0) {
                        *video_frame_count = strtol_checked(line, "count ", 0, LONG_MAX);
                        items_found |= 1U<<6U;
                }
        }

        if(items_found != (1U << 7U) - 1U) {
                throw ug_runtime_error("Failed while reading config file - some items missing.");
        }

        assert((desc.color_spec != VIDEO_CODEC_NONE && desc.width != 0 && desc.height != 0 && desc.fps != 0.0 &&
                                *video_frame_count != 0));
        return desc;
}

static int get_tile_count(const char *directory, codec_t color_spec, char *tile_delim) {
        char name[1024];
        snprintf(name, sizeof(name), "%s/%08d.%s", directory, 1,
                       get_codec_file_extension(color_spec));

        struct stat sb;
        if (stat(name, &sb) == 0) {
                return 1;
        }
        int tile_count = 0;
        char possible_tile_delim[] = { '_', '-' };
        for (unsigned int d = 0; d < sizeof possible_tile_delim; d++) {
                for (int i = 0; i < 10; i++) {
                        snprintf(name, sizeof(name), "%s/%08d%c%d.%s",
                                        directory, 1,
                                        possible_tile_delim[d], i,
                                        get_codec_file_extension(color_spec));
                        if (stat(name, &sb) == 0) {
                                tile_count++;
                        } else {
                                break;
                        }
                }
                if (tile_count > 0) {
                        *tile_delim = possible_tile_delim[d];
                        break;
                }
        }
        if (tile_count == 0) {
                throw ug_runtime_error("Unable to open first file of "
                                "the video sequence.");
        }
        return tile_count;
}

static int
vidcap_import_init(struct vidcap_params *params, void **state)
{
	struct vidcap_import_state *s = NULL;
        FILE *info = NULL; // metadata file
        char *tmp = strdup(vidcap_params_get_fmt(params));
        bool disable_audio = false;

        using namespace std::string_literals;
try {
	printf("vidcap_import_init\n");

        if (strlen(tmp) == 0 || strcmp(tmp, "help") == 0) {
                color_printf("Import usage:\n"
                                TERM_BOLD TERM_FG_RED "\t<directory>" TERM_FG_RESET "{:loop|:mt_reading=<nr_threads>|:o_direct|:exit_at_end|:fps=<fps>|frames=<n>|:disable_audio}\n" TERM_RESET
                                "where\n"
                                TERM_BOLD "\t<fps>" TERM_RESET " - overrides FPS from sequence metadata\n"
                                TERM_BOLD "\t<n>  " TERM_RESET " - use only N first frames fron sequence (if less than available frames)\n");
                delete s;
                free(tmp);
                return VIDCAP_INIT_NOERR;
        }

        s = new vidcap_import_state();
        pthread_cond_init(&s->worker_cv, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        pthread_mutex_init(&s->lock, NULL);
        s->head = s->tail = NULL;
        s->queue_len = 0;

        s->parent = vidcap_params_get_parent(params);
        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_DATA;
        s->mod.priv_data = s;
        s->mod.new_message = vidcap_import_new_message;
        module_register(&s->mod, s->parent);

        s->video_reading_threads_count = 1; // default is single threaded

        char *save_ptr = NULL;
        char *suffix;
        s->directory = strtok_r(tmp, ":", &save_ptr);
        if (s->directory == nullptr) {
                throw ug_runtime_error("Wrong directory name!");
        }
        s->directory = strdup(s->directory); // make a copy

        while ((suffix = strtok_r(NULL, ":", &save_ptr)) != NULL) {
                if (suffix[0] == '\\') { // MSW path
                        assert(strlen(s->directory) == 1); // c:\something -> should be 'c'
                        const size_t len = 2 + strlen(suffix) + 1;
                        char *tmp = (char *) malloc(len);
                        snprintf(tmp, len, "%c:%s", s->directory[0], suffix);
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
                } else if (strcmp(suffix, "opportunistic_audio") == 0) { // skip
                } else if (strcmp(suffix, "exit_at_end") == 0) {
                        s->should_exit_at_end = true;
                } else if (strncmp(suffix, "fps=", strlen("fps=")) == 0) {
                        s->force_fps = atof(suffix + strlen("fps="));
                } else if (strstr(suffix, "frames=") == suffix) {
                        s->video_frame_count = strtol(strchr(suffix, '=') + 1, nullptr, 10);
                } else {
                        throw ug_runtime_error("Unrecognized option"s
                                        + suffix + ".\n");
                }
        }
        free(tmp);
        tmp = NULL;

        // strip video.info if user included in path
        if (strstr(s->directory, "video.info") == s->directory) {
                strcpy(s->directory, ".");
        }
        if (strstr(s->directory, "/video.info") != nullptr) {
                *strrchr(s->directory, '/') = '\0';
        }

        message_queue_clear(&s->message_queue);
        message_queue_clear(&s->audio_state.message_queue);

        std::string audio_filename = std::string(s->directory) + "/sound.wav";
        if((vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_EMBEDDED) && !disable_audio && init_audio(s, audio_filename.c_str())) {
                s->audio_state.has_audio = true;
        }
        
        std::string info_filename = std::string(s->directory) + "/video.info";
        info = fopen(info_filename.c_str(), "r");
        if (info == nullptr) {
                perror(MOD_NAME "Failed to open video index file");
                if (!s->audio_state.has_audio) {
                        if (errno == ENOENT) {
                                throw ug_runtime_error("Invalid directory?");
                        }
                        throw ug_runtime_error("");
                }
                s->has_video = false;
                s->video_desc.fps = 30; // used to sample audio
        }

        if (s->has_video) {
                long frame_count = 0;
                s->video_desc = parse_video_desc_info(info, &frame_count);
                s->video_frame_count = s->video_frame_count == 0 ? frame_count : MIN(s->video_frame_count, frame_count);

                fclose(info);
                info = NULL;

                s->video_desc.tile_count = get_tile_count(s->directory, s->video_desc.color_spec, &s->tile_delim);
        }

        // override metadata fps setting
        if (s->force_fps > 0.0) {
                s->video_desc.fps = s->force_fps;
        }

        if (s->audio_state.has_audio) {
                if(pthread_create(&s->audio_state.thread_id, NULL, audio_reading_thread, (void *) s) != 0) {
                        throw ug_runtime_error("Unable to create thread.");
                }
        }
        if (s->has_video) {
                if (pthread_create(&s->video_thread_id, NULL, video_reading_thread, (void *) s) != 0) {
                        throw ug_runtime_error("Unable to create thread.");
                }
        }

        gettimeofday(&s->prev_time, NULL);

        playback_register_keyboard_ctl(&s->mod);

        *state = s;
	return VIDCAP_INIT_OK;
} catch (ug_runtime_error const & e) {
        LOG(LOG_LEVEL_ERROR) << MOD_NAME << e.what() << "\n";
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
        if (s->has_video) {
                struct import_message *msg = (struct import_message *) malloc(sizeof(struct import_message));

                msg->type = FINALIZE;
                msg->data = NULL;
                msg->data_len = 0;
                msg->next = NULL;

                pthread_mutex_lock(&s->lock);
                send_message(msg, &s->message_queue);
                pthread_mutex_unlock(&s->lock);
                pthread_cond_signal(&s->worker_cv);

                pthread_join(s->video_thread_id, NULL);
        }

        // audio
        if(s->audio_state.has_audio) {
                struct import_message *msg = (struct import_message *) malloc(sizeof(struct import_message));

                msg->type = FINALIZE;
                msg->data = NULL;
                msg->data_len = 0;
                msg->next = NULL;

                {
                        pthread_mutex_lock(&s->audio_state.lock);
                        send_message(msg, &s->audio_state.message_queue);
                        pthread_mutex_unlock(&s->audio_state.lock);
                        pthread_cond_signal(&s->audio_state.worker_cv);
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
                pthread_cond_destroy(&s->audio_state.worker_cv);
                pthread_cond_destroy(&s->audio_state.boss_cv);
                pthread_mutex_destroy(&s->audio_state.lock);
        }

        module_done(&s->mod);
        pthread_cond_destroy(&s->worker_cv);
        pthread_cond_destroy(&s->boss_cv);
        pthread_mutex_destroy(&s->lock);
}

static void vidcap_import_done(void *state)
{
	struct vidcap_import_state *s = (struct vidcap_import_state *) state;
	assert(s != NULL);

        vidcap_import_finish(state);

        cleanup_common(s);
        delete s;
}

static void send_message(struct import_message *msg, struct message_queue *queue)
{
        if(queue->head) {
                queue->tail->next = msg;
                queue->tail = msg;
        } else {
                queue->head = queue->tail = msg;
        }

        queue->len += 1;
}

static struct import_message *pop_message(struct message_queue *queue)
{
        assert(queue->len > 0);
        struct import_message *ret;

        ret = queue->head;
        queue->head = queue->head->next;
        if(queue->head == NULL) {
                queue->tail = NULL;
        }

        queue->len -= 1;

        return ret;
}

static void vidcap_import_new_message(struct module *mod) {
        struct msg_universal *m;
        while ((m = (struct msg_universal *) check_message(mod))) {
                process_msg((vidcap_import_state *) mod->priv_data, m->text);
                free_message((struct message *) m, new_response(RESPONSE_ACCEPTED, "import is processing the request"));
        }
}

static void process_msg(struct vidcap_import_state *s, const char *message)
{
        if(strcasecmp(message, "pause") == 0) {
                struct import_message *msg = (struct import_message *) malloc(sizeof(struct import_message));
                msg->type = PAUSE;
                msg->data = NULL;
                msg->data_len = 0;
                msg->next = NULL;

                pthread_mutex_lock(&s->lock);
                send_message(msg, &s->message_queue);
                pthread_mutex_unlock(&s->lock);
                pthread_cond_signal(&s->worker_cv);
        } else if(strncasecmp(message, "seek ", strlen("seek ")) == 0) {
                const char *time_spec = message + strlen("seek ");

                struct import_message *msg = (struct import_message *) malloc(sizeof(struct import_message));
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

                struct import_message *audio_msg = NULL;

                if(s->audio_state.has_audio) {
                        audio_msg = (struct import_message *) malloc(sizeof(struct import_message));
                        memcpy(audio_msg, msg, sizeof(struct import_message));

                        if(audio_msg->data) { // deep copy
                                audio_msg->data = malloc(msg->data_len);
                                memcpy(audio_msg->data, msg->data, msg->data_len);
                        }
                }

                {
                        pthread_mutex_lock(&s->lock);
                        send_message(msg, &s->message_queue);
                        pthread_mutex_unlock(&s->lock);
                        pthread_cond_signal(&s->worker_cv);
                }

                if (s->audio_state.has_audio) {
                        pthread_mutex_lock(&s->audio_state.lock);
                        send_message(audio_msg, &s->audio_state.message_queue);
                        pthread_mutex_unlock(&s->audio_state.lock);
                        pthread_cond_signal(&s->audio_state.worker_cv);
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
        
static void * audio_reading_thread(void *args)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) args;

        //while(s->audio_state.samples_read < s->audio_state.total_samples && !s->finish_threads) {
        while(1) {
                int max_read;
                {
                        pthread_mutex_lock(&s->audio_state.lock);
                        while((ring_get_current_size(s->audio_state.data) > ring_get_size(s->audio_state.data) * 2 / 3 ||
                                                s->audio_state.samples_read >= s->audio_state.total_samples)
                                        && s->audio_state.message_queue.len == 0) {
                                pthread_cond_wait(&s->audio_state.worker_cv, &s->audio_state.lock);
                        }

                        while(s->audio_state.message_queue.len > 0) {
                                struct import_message *msg = pop_message(&s->audio_state.message_queue);
                                if(msg->type == FINALIZE) {
                                        free(msg);
                                        pthread_mutex_unlock(&s->audio_state.lock);
                                        return NULL;
                                } else if (msg->type == SEEK) {
                                        struct seek_data *data = (struct seek_data *) msg->data;
                                        free(msg);
                                        long long bytes = (s->audio_frame.sample_rate * data->offset / s->video_desc.fps) * s->audio_frame.bps * s->audio_frame.ch_count;
                                        if (data->whence == IMPORT_SEEK_CUR) {
                                                bytes += s->audio_state.played_samples * s->audio_frame.bps * s->audio_frame.ch_count;
                                                bytes = max<long long>(0, bytes);
                                        }
                                        int ret = wav_seek(s->audio_state.file, bytes, SEEK_SET, &s->audio_state.metadata);
                                        log_msg(LOG_LEVEL_NOTICE, "Audio seek %lld bytes\n", bytes);
                                        if (ret != 0) {
                                                perror("wav_seek");
                                        }
                                        ring_buffer_flush(s->audio_state.data);
                                        s->audio_state.video_frames_played = max<long long>(0, s->audio_state.video_frames_played + data->offset);
                                        s->audio_state.samples_read = bytes / (s->audio_frame.bps * s->audio_frame.ch_count);
                                        s->audio_state.samples_read = min(s->audio_state.samples_read, s->audio_state.total_samples);
                                        s->audio_state.played_samples = s->audio_state.samples_read;
                                        free(data);
                                }
                        }

                        max_read = ring_get_size(s->audio_state.data) -
                                        ring_get_current_size(s->audio_state.data) - 1;

                        pthread_mutex_unlock(&s->audio_state.lock);
                }

                char *buffer = (char *) malloc(max_read);

                size_t samples = wav_read(buffer, max_read / s->audio_frame.ch_count / s->audio_frame.bps, s->audio_state.file, &s->audio_state.metadata);
                s->audio_state.samples_read += samples;

                {
                        pthread_mutex_lock(&s->audio_state.lock);
                        ring_buffer_write(s->audio_state.data, buffer, samples * s->audio_frame.ch_count * s->audio_frame.bps);
                        pthread_mutex_unlock(&s->audio_state.lock);
                        pthread_cond_signal(&s->audio_state.boss_cv);
                }

                free(buffer);
        }

        return NULL;
}

struct video_reader_data {
        char file_name_prefix[512];
        char file_name_suffix[512];
        char tile_delim;
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
                char name[1048];
                char tile_idx[3] = "";
                if (data->tile_count > 1) {
                        snprintf(tile_idx, sizeof tile_idx, "%c%d", data->tile_delim, i);
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

static void * video_reading_thread(void *args)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) args;
        long index = 0;

        bool paused = false;

        ///while(index < s->video_frame_count && !s->finish_threads) {
        while(1) {
                {
                        pthread_mutex_lock(&s->lock);
                        while((s->queue_len >= BUFFER_LEN_MAX - 1 || index >= s->video_frame_count || paused)
                                       && s->message_queue.len == 0) {
                                if (index >= s->video_frame_count) {
                                        s->finished = true;
                                }
                                pthread_cond_wait(&s->worker_cv, &s->lock);
                        }

                        while(s->message_queue.len > 0) {
                                struct import_message *msg = pop_message(&s->message_queue);
                                if(msg->type == FINALIZE) {
                                        free(msg);
                                        pthread_mutex_unlock(&s->lock);
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
                                        }
                                        index = min(max(0L, index), s->video_frame_count - 1);
                                        printf("Current index: frame %ld\n", index);
                                        free(data);
                                } else {
                                        fprintf(stderr, "Unknown message type: %d!\n", msg->type);
                                        abort();
                                }
                        }
                        pthread_mutex_unlock(&s->lock);
                }

                /// @todo are these checks necessary?
                index = min(max(0L, index), s->video_frame_count - 1);

                struct video_reader_data data_reader[MAX_NUMBER_WORKERS];
                task_result_handle_t task_handle[MAX_NUMBER_WORKERS];

                int number_workers = s->video_reading_threads_count;
                if (index + number_workers >= s->video_frame_count) {
                        number_workers = s->video_frame_count - index;
                }
                // run workers
                for (int i = 0; i < number_workers; ++i) {
                        struct video_reader_data *data =
                                &data_reader[i];
                        data->o_direct = s->o_direct;
                        data->tile_count = s->video_desc.tile_count;
                        data->tile_delim = s->tile_delim;
                        snprintf(data->file_name_prefix, sizeof(data->file_name_prefix),
                                        "%s/%08ld", s->directory, index + i + 1);
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
                                pthread_mutex_lock(&s->lock);
                                if(s->head) {
                                        s->tail->next = data->entry;
                                        s->tail = data->entry;
                                } else {
                                        s->head = s->tail = data->entry;
                                }
                                s->queue_len += 1;

                                pthread_mutex_unlock(&s->lock);
                                pthread_cond_signal(&s->boss_cv);
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
                if (wav_seek(s->audio_state.file, 0L, SEEK_SET, &s->audio_state.metadata) != 0) {
                        perror("wav_seek");
                }
                s->audio_state.video_frames_played = 0;
        }

        if (s->has_video) {
                if (pthread_create(&s->video_thread_id, NULL, video_reading_thread, (void *) s) != 0) {
                        fprintf(stderr, "Unable to create thread.\n");
                        /// @todo what to do here
                        abort();
                }
        }
        if (s->audio_state.has_audio) {
                if(pthread_create(&s->audio_state.thread_id, NULL, audio_reading_thread, (void *) s) != 0) {
                        fprintf(stderr, "Unable to create thread.\n");
                        abort();
                }
        }
}

static void vidcap_import_dispose_video_frame(struct video_frame *frame) {
        free_entry((struct processed_entry *) frame->callbacks.dispose_udata);
        vf_free(frame);
}

static struct video_frame *
vidcap_import_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_import_state 	*s = (struct vidcap_import_state *) state;
        struct timeval cur_time;
        struct video_frame *ret = nullptr;
        
        struct processed_entry *current = NULL;

        if (s->has_video) {
                pthread_mutex_lock(&s->lock);
                if(s->queue_len == 0) {
                        if (s->finished == true && s->loop) {
                                pthread_mutex_unlock(&s->lock);
                                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Rewinding the sequence.\n");
                                reset_import(s);
                                pthread_mutex_lock(&s->lock);
                        }
                        if (s->finished == true && s->should_exit_at_end == true) {
                                exit_uv(0);
                        }
                        struct timespec ts;
                        timespec_get(&ts, TIME_UTC);
                        ts_add_nsec(&ts, 2 * NS_IN_SEC / s->video_desc.fps);
                        pthread_cond_timedwait(&s->boss_cv, &s->lock, &ts);
                }
 
                if(s->queue_len == 0) {
                        return NULL;
                }

                current = s->head;
                assert(current != NULL);

                s->head = s->head->next;
                s->queue_len -= 1;

                pthread_mutex_unlock(&s->lock);
                pthread_cond_signal(&s->worker_cv);

                ret = vf_alloc_desc(s->video_desc);
                ret->callbacks.dispose = vidcap_import_dispose_video_frame;
                ret->callbacks.dispose_udata = current;
                for (unsigned int i = 0; i < s->video_desc.tile_count; ++i) {
                        ret->tiles[i].data_len =
                                current->tiles[i].data_len;
                        ret->tiles[i].data = current->tiles[i].data;
                }
        }

        // audio
        if(s->audio_state.has_audio) {
                auto get_req_bytes = [&]() {
                        long long int requested_samples = (long long int) (s->audio_state.video_frames_played + 0) *
                                s->audio_frame.sample_rate / s->video_desc.fps - s->audio_state.played_samples;
                        if (requested_samples <= 0) {
                                return 0LL;
                        }
                        if ((s->audio_state.played_samples + requested_samples) > s->audio_state.total_samples) {
                                requested_samples = s->audio_state.total_samples - s->audio_state.played_samples;
                        }
                        return requested_samples * s->audio_frame.bps * s->audio_frame.ch_count;
                };
                pthread_mutex_lock(&s->audio_state.lock);
                unsigned long long int requested_bytes = min<unsigned long long>(get_req_bytes(), s->audio_frame.max_size);
                while(ring_get_current_size(s->audio_state.data) < (int) requested_bytes) {
                        pthread_cond_wait(&s->audio_state.boss_cv, &s->audio_state.lock);
                        requested_bytes = min<unsigned long long>(get_req_bytes(), s->audio_frame.max_size);
                }

                int ret = ring_buffer_read(s->audio_state.data, s->audio_frame.data, requested_bytes);
                s->audio_frame.data_len = ret;
                s->audio_state.played_samples += ret / (s->audio_frame.bps * s->audio_frame.ch_count);
                *audio = &s->audio_frame;
                s->audio_state.video_frames_played += 1;
                pthread_mutex_unlock(&s->audio_state.lock);
                pthread_cond_signal(&s->audio_state.worker_cv);
        } else {
                *audio = NULL;
        }

        gettimeofday(&cur_time, NULL);
        while(tv_diff_usec(cur_time, s->prev_time) < 1000000.0 / s->video_desc.fps) {
                gettimeofday(&cur_time, NULL);
        }
        //tv_add_usec(&s->prev_time, 1000000.0 / s->frame->fps);
        s->prev_time = cur_time;

	return ret;
}

static const struct video_capture_info vidcap_import_info = {
        vidcap_import_probe,
        vidcap_import_init,
        vidcap_import_done,
        vidcap_import_grab,
        MOD_NAME,
};

REGISTER_MODULE(import, &vidcap_import_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

