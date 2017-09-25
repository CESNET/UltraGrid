/**
 * @file   export.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2017 CESNET z.s.p.o.
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

#include "export.h"

#include "audio/export.h"
#include "debug.h"
#include "messaging.h"
#include "module.h"
#include "video_export.h"

struct exporter {
        struct module mod;
        char *dir;
        bool dir_auto;
        struct video_export *video_export;
        struct audio_export *audio_export;
        bool exporting;
        pthread_mutex_t lock;
};

static bool create_dir(struct exporter *s);
static bool enable_export(struct exporter *s);
static void disable_export(struct exporter *s);
static void process_messages(struct exporter *s);

static void new_msg(struct module *mod) {
        process_messages(mod->priv_data);
}

struct exporter *export_init(struct module *parent, const char *path, bool should_export)
{
        struct exporter *s = calloc(1, sizeof(struct exporter));
        pthread_mutex_init(&s->lock, NULL);

        if (path) {
                s->dir = strdup(path);
        } else {
                s->dir_auto = true;
        }

        if (should_export) {
                if (!enable_export(s)) {
                        goto error;
                }
        }

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_EXPORTER;
        s->mod.new_message = new_msg;
        s->mod.priv_data = s;
        module_register(&s->mod, parent);

        return s;

error:
        pthread_mutex_destroy(&s->lock);
        free(s->dir);
        free(s);
        return NULL;
}

static bool enable_export(struct exporter *s)
{
        if (!create_dir(s)) {
                goto error;
        }

        s->video_export = video_export_init(s->dir);
        if (!s->video_export) {
                goto error;
        }

        char name[512];
        snprintf(name, 512, "%s/sound.wav", s->dir);
        s->audio_export = audio_export_init(name);
        if (!s->audio_export) {
                goto error;
        }

        s->exporting = true;

        pthread_mutex_unlock(&s->lock);
        return true;

error:
        video_export_destroy(s->video_export);
        s->video_export = NULL;
        return false;
}

static bool create_dir(struct exporter *s)
{
        if (!s->dir) {
                for (int i = 1; i <= 9999; i++) {
                        char name[21];
                        time_t t = time(NULL);
                        struct tm *tmp = localtime(&t);
                        strftime(name, sizeof name, "export.%Y%m%d", tmp);
                        if (i > 1) {
                                char num[6];
                                snprintf(num, sizeof num, "-%d", i);
                                strncat(name, num, sizeof name - strlen(name) - 1);
                        }
                        int ret = platform_mkdir(name);
                        if(ret == -1) {
                                if(errno == EEXIST) {
                                        continue;
                                } else {
                                        fprintf(stderr, "[Export] Directory creation failed: %s\n",
                                                        strerror(errno));
                                        return false;
                                }
                        } else {
                                s->dir = strdup(name);
                                break;
                        }
                }
        } else {
                int ret = platform_mkdir(s->dir);
                if(ret == -1) {
                        if(errno == EEXIST) {
                                fprintf(stderr, "[Export] Warning: directory %s exists!\n", s->dir);
                                return false;
                        } else {
                                perror("[Export] Directory creation failed");
                                return false;
                        }
                }
        }

        if (s->dir) {
                printf("Using export directory: %s\n", s->dir);
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
                        log_msg(LOG_LEVEL_NOTICE, "Exporing: %s\n", s->exporting ? "ON" : "OFF");
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
        pthread_mutex_unlock(&s->lock);
}

