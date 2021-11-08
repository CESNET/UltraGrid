/**
 * @file   export.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2017-2021 CESNET z.s.p.o.
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

#include <sys/types.h>
#include <dirent.h>
#include <limits.h>

#include "export.h"

#include "audio/export.h"
#include "debug.h"
#include "messaging.h"
#include "module.h"
#include "utils/color_out.h"
#include "utils/fs.h" // MAX_PATH_SIZE
#include "video_export.h"

#define MOD_NAME "[export] "

struct exporter {
        struct module mod;
        char *dir;
        bool override;
        bool dir_auto;
        struct video_export *video_export;
        struct audio_export *audio_export;
        bool exporting;
        bool noaudio;
        bool novideo;
        pthread_mutex_t lock;

        long long int limit; ///< number of video frames to record, -1 == unlimited (default)
};

static bool create_dir(struct exporter *s);
static bool enable_export(struct exporter *s);
static void disable_export(struct exporter *s);
static void process_messages(struct exporter *s);

static void new_msg(struct module *mod) {
        process_messages(mod->priv_data);
}

#define HANDLE_ERROR export_destroy(s); return NULL;
#define OPTLEN_MAX 40 ///< max length of cmdline options

struct exporter *export_init(struct module *parent, const char *cfg, bool should_export)
{
        struct exporter *s = calloc(1, sizeof(struct exporter));
        pthread_mutex_init(&s->lock, NULL);
        s->limit = -1;

        if (cfg) {
                if (strcmp(cfg, "help") == 0) {
                        color_out(0, "Usage:\n");
                        color_out(COLOR_OUT_RED | COLOR_OUT_BOLD, "\t--record");
                        color_out(COLOR_OUT_BOLD, "[=<dir>[:limit=<n>][:noaudio][:novideo][:override][:paused]]\n");
                        color_out(0, "where\n");
                        color_out(COLOR_OUT_BOLD, "\tlimit=<n>");
                        color_out(0, " - write at most <n> video frames\n");
                        color_out(COLOR_OUT_BOLD, "\toverride");
                        color_out(0, " - export even if it would override existing files in the given directory\n");
                        color_out(COLOR_OUT_BOLD, "\tnoaudio | novideo");
                        color_out(0, " - do not export audio/video\n");
                        color_out(COLOR_OUT_BOLD, "\tpaused");
                        color_out(0, " - use specified directory but do not export immediately (can be started with a key or through control socket)\n");
                        export_destroy(s);
                        return NULL;
                }
                char cfg_copy[PATH_MAX + OPTLEN_MAX] = "";
                char *save_ptr = NULL;
                strncpy(cfg_copy, cfg, sizeof cfg_copy - 1);
                s->dir = strtok_r(cfg_copy, ":", &save_ptr);
                if (s->dir == NULL) {
                        HANDLE_ERROR
                }
                s->dir = strdup(s->dir);
                char *item = NULL;
                while ((item = strtok_r(NULL, ":", &save_ptr)) != NULL) {
                        if (strstr(item, "noaudio") == item) {
                                s->noaudio = true;
                        } else if (strstr(item, "novideo") == item) {
                                s->novideo = true;
                        } else if (strstr(item, "override") == item) {
                                s->override = true;
                        } else if (strstr(item, "paused") == item) {
                                should_export = false; // start paused
                        } else if (strstr(item, "limit=") == item) {
                                s->limit = strtoll(item + strlen("limit="), NULL, 0);
                                if (s->limit < 0) {
                                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong limit: %s!\n", item + strlen("limit="));
                                        HANDLE_ERROR
                                }
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong option: %s!\n", item);
                                HANDLE_ERROR
                        }
                }
        } else {
                s->dir_auto = true;
        }

        if (should_export) {
                if (!enable_export(s)) {
                        HANDLE_ERROR
                }
        }

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_EXPORTER;
        s->mod.new_message = new_msg;
        s->mod.priv_data = s;
        module_register(&s->mod, parent);

        return s;
}

static bool enable_export(struct exporter *s)
{
        if (!create_dir(s)) {
                goto error;
        }

        if (!s->novideo) {
                s->video_export = video_export_init(s->dir);
                if (!s->video_export) {
                        goto error;
                }
        }

        if (!s->noaudio) {
                char name[MAX_PATH_SIZE];
                snprintf(name, sizeof name, "%s/sound.wav", s->dir);
                s->audio_export = audio_export_init(name);
                if (!s->audio_export) {
                        goto error;
                }
        }

        s->exporting = true;

        return true;

error:
        video_export_destroy(s->video_export);
        s->video_export = NULL;
        return false;
}

/**
 * Tries to create directories export.<date>[-????]
 * inside directory prefix. If succesful, returns its
 * name.
 */
static char *create_anonymous_dir(const char *prefix)
{
        for (int i = 1; i <= 9999; i++) {
                size_t max_len = strlen(prefix) + 1 + 21;
                char *name = malloc(max_len);
                time_t t = time(NULL);
                struct tm *tmp = localtime(&t);
                strcpy(name, prefix);
                strcat(name, "/");
                strftime(name + strlen(name), max_len, "export.%Y%m%d", tmp);
                if (i > 1) {
                        char num[6];
                        snprintf(num, sizeof num, "-%d", i);
                        strncat(name, num, sizeof name - strlen(name) - 1);
                }
                int ret = platform_mkdir(name);
                if(ret == -1) {
                        if(errno == EEXIST) { // record exists, try next directory
                                free(name);
                                continue;
                        }
                        fprintf(stderr, "[Export] Directory creation failed: %s\n",
                                        strerror(errno));
                        free(name);
                        return false;
                } else {
                        return name;
                }
        }
        return NULL;
}

static bool dir_is_empty(const char *dir) {
        DIR *d = opendir(dir);
        if (!d) {
                return false;
        }
        readdir(d); // skip . and ..
        readdir(d);
        bool ret = readdir(d) == NULL;
        closedir(d);

        return ret;
}

static bool create_dir(struct exporter *s)
{
        if (!s->dir) {
                s->dir = create_anonymous_dir(".");
        } else {
                int ret = platform_mkdir(s->dir);
                if(ret == -1) {
                        if(errno != EEXIST) {
                                perror("[Export] Directory creation failed");
                                return false;
                        }
                        if (dir_is_empty(s->dir)) {
                                log_msg(LOG_LEVEL_NOTICE, "[Export] Warning: directory %s exists but is an empty directory - using for export.\n", s->dir);
                        } else if (s->override) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: directory %s exists and is not an empty directory but using as requested.\n", s->dir);
                        } else {
                                log_msg(LOG_LEVEL_WARNING, "[Export] Warning: directory %s exists and is not an empty directory! Trying to create subdir.\n", s->dir);
                                char *prefix = s->dir;
                                s->dir = create_anonymous_dir(prefix);
                                free(prefix);
                        }
                }
        }

        if (s->dir) {
                color_out(COLOR_OUT_BOLD | COLOR_OUT_YELLOW, "Using export directory: %s\n", s->dir);
                return true;
        } else {
                return false;
        }
}

static void disable_export(struct exporter *s) {
        audio_export_destroy(s->audio_export);
        video_export_destroy(s->video_export);
        s->audio_export = NULL;
        s->video_export = NULL;
        if (s->dir_auto) {
                free(s->dir);
                s->dir = NULL;
        }
        s->exporting = false;
}

void export_destroy(struct exporter *s) {
        disable_export(s);

        pthread_mutex_destroy(&s->lock);
        module_done(&s->mod);
        free(s->dir);
        free(s);
}

static void process_messages(struct exporter *s) {
        struct message *m;
        while ((m = check_message(&s->mod))) {
                struct response *r;
                pthread_mutex_lock(&s->lock);
                struct msg_universal *msg = (struct msg_universal *) m;
                if (strcmp(msg->text, "toggle") == 0) {
                        if (s->exporting) {
                                disable_export(s);
                        } else {
                                enable_export(s);
                        }
                        log_msg(LOG_LEVEL_NOTICE, "Exporting: %s\n", s->exporting ? "ON" : "OFF");
                        r = new_response(RESPONSE_OK, NULL);
                } else if (strcmp(msg->text, "status") == 0) {
                        r = new_response(RESPONSE_OK, s->exporting ? "true" : "false");
                } else {
                        r = new_response(RESPONSE_NOT_FOUND, NULL);
                }
                pthread_mutex_unlock(&s->lock);
                free_message(m, r);
        }
}

void export_audio(struct exporter *s, struct audio_frame *frame)
{
        if(!s){
                return;
        }

        process_messages(s);

        pthread_mutex_lock(&s->lock);
        if (s->exporting) {
                audio_export(s->audio_export, frame);
        }
        pthread_mutex_unlock(&s->lock);
}

void export_video(struct exporter *s, struct video_frame *frame)
{
        if(!s){
                return;
        }

        process_messages(s);

        pthread_mutex_lock(&s->lock);
        if (s->exporting) {
                video_export(s->video_export, frame);
        }
        if (s->limit > 0) {
                if (--s->limit == 0) {
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Stopping export - limit reached.\n");
                        disable_export(s);
                }
        }
        pthread_mutex_unlock(&s->lock);
}

