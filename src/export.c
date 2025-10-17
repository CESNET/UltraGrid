/**
 * @file   export.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2017-2024 CESNET
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

#include <dirent.h>
#include <errno.h>            // for errno, EEXIST
#include <limits.h>
#include <stdbool.h>
#include <stdlib.h>           // for NULL, free, calloc, strtol
#include <string.h>           // for strdup
#include <sys/types.h>
#include <time.h>

#include "export.h"

#include "audio/export.h"
#define WANT_MKDIR
#include "compat/misc.h"      // for mkdir
#include "compat/time.h"      // for localtime_s
#include "debug.h"
#include "host.h"
#include "messaging.h"
#include "module.h"
#include "utils/color_out.h"
#include "utils/fs.h" // MAX_PATH_SIZE
#include "utils/misc.h"
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
        bool exit_on_limit;
};

static bool create_dir(struct exporter *s);
static bool enable_export(struct exporter *s);
static void disable_export(struct exporter *s);
static void process_messages(struct exporter *s);

static void new_msg(struct module *mod) {
        process_messages(mod->priv_data);
}

static void usage() {
        color_printf("Usage:\n");
        color_printf("\t" TBOLD(
            TRED("--record") "[=<dir>[:limit=<n>[:exit_on_limit]][:noaudio]"
            "[:novideo][:override][:paused]] ") "\n" "\t" TBOLD(TRED("-E")
            "[<dir>[:<opts>]]") "\n\t" TBOLD("--record=help | -Ehelp") "\n");
        color_printf("where\n");
        color_printf(TERM_BOLD "\tlimit=<n>" TERM_RESET "         - write at "
                     " most <n> video frames (with optional exit)\n");
        color_printf(TERM_BOLD "\toverride" TERM_RESET
                               "          - export even if it would override "
                               "existing files in the given directory\n");
        color_printf(TERM_BOLD "\tnoaudio | novideo" TERM_RESET " - do not export audio/video\n");
        color_printf(TERM_BOLD "\tpaused" TERM_RESET "            - use specified directory but do not export immediately (can be started with a key or through control socket)\n");
}

static bool
parse_options(struct exporter *s, const char *ccfg, bool *should_export)
{
        if (ccfg == NULL) {
                return true;
        }
        char buf[STR_LEN];
        snprintf(buf, sizeof buf, "%s", ccfg);
        char *cfg = buf;
        char *save_ptr = NULL;
        char *item = NULL;
        while ((item = strtok_r(cfg, ":", &save_ptr)) != NULL) {
                if (strstr(item, "help") == item) {
                        usage();
                        return false;
                } else if (strstr(item, "noaudio") == item) {
                        s->noaudio = true;
                } else if (strstr(item, "novideo") == item) {
                        s->novideo = true;
                } else if (strstr(item, "override") == item) {
                        s->override = true;
                } else if (strstr(item, "paused") == item) {
                        *should_export = false; // start paused
                } else if (strstr(item, "limit=") == item) {
                        s->limit = strtoll(item + strlen("limit="), NULL, 0);
                        if (s->limit < 0) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong limit: %s!\n", item + strlen("limit="));
                                return false;
                        }
                } else if (strcmp(item, "exit_on_limit") == 0) {
                        s->exit_on_limit = true;
                } else if (s->dir == NULL && cfg != NULL) {
                        s->dir = strdup(item);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong option: %s!\n", item);
                        return false;
                }
                cfg = NULL;
        }
        return true;
}

#define HANDLE_ERROR export_destroy(s); return NULL;

struct exporter *export_init(struct module *parent, const char *cfg, bool should_export)
{
        struct exporter *s = calloc(1, sizeof(struct exporter));
        pthread_mutex_init(&s->lock, NULL);
        s->limit = -1;

        if (!parse_options(s, cfg, &should_export)) {
                HANDLE_ERROR
        }
        if (s->dir == NULL) {
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
 * inside directory prefix. If successful, returns its
 * name.
 */
static char *
create_implicit_dir(const char *prefix)
{
        enum {
                MAX_EXPORTS = 9999,
        };
        for (int i = 1; i <= MAX_EXPORTS; i++) {
                char       name[MAX_PATH_SIZE];
                time_t     t      = time(NULL);
                struct tm  tm_buf = { 0 };
                localtime_s(&t, &tm_buf);
                snprintf(name, sizeof name, "%s/", prefix);
                strftime(name + strlen(name), sizeof name - strlen(name),
                         "export.%Y%m%d", &tm_buf);
                if (i > 1) {
                        snprintf(name + strlen(name),
                                 sizeof name - strlen(name), "-%d", i);
                }
                int ret = mkdir(name, S_IRWXU | S_IRWXG | S_IRWXO);
                if(ret == -1) {
                        if(errno == EEXIST) { // record exists, try next directory
                                continue;
                        }
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "Directory creation failed: %s\n",
                                ug_strerror(errno));
                        return NULL;
                }
                return strdup(name);
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
                s->dir = create_implicit_dir(".");
        } else {
                int ret = mkdir(s->dir, S_IRWXU | S_IRWXG | S_IRWXO);
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
                                MSG(ERROR,
                                    "Warning: directory %s exists and is not "
                                    "an empty directory! Trying to create "
                                    "subdir.\n",
                                    s->dir);
                                char *prefix = s->dir;
                                s->dir = create_implicit_dir(prefix);
                                free(prefix);
                        }
                }
        }

        if (s->dir) {
                color_printf(TERM_BOLD TERM_FG_YELLOW "Using export directory: %s\n" TERM_RESET, s->dir);
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
                        if (s->exit_on_limit) {
                                exit_uv(0);
                        }
                }
        }
        pthread_mutex_unlock(&s->lock);
}

