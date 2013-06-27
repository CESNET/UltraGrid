/*
 * FILE:    video_decompress.c
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

#include <stdio.h>
#include <string.h>
#include "messaging.h"
#include "module.h"
#include "video_codec.h"
#include "video_compress.h"
#include "video_compress/dxt_glsl.h"
#include "video_compress/fastdxt.h"
#include "video_compress/libavcodec.h"
#include "video_compress/jpeg.h"
#include "video_compress/none.h"
#include "video_compress/uyvy.h"
#include "lib_common.h"
#include "compat/platform_spin.h"
#include "utils/worker.h"

/* *_str are symbol names inside library */
struct compress_t {
        const char        * name;
        const char        * library_name;

        compress_init_t     init;
        const char        * init_str;
        compress_frame_t    compress_frame;
        const char         *compress_frame_str;
        compress_tile_t     compress_tile;
        const char         *compress_tile_str;

        void *handle;
};

struct compress_state_real {
        struct compress_t  *handle;
        struct module     **state;
        unsigned int        state_count;
        char                compress_options[1024];

        struct video_frame *out_frame[2];

        unsigned int uncompressed:1;
};

struct compress_state {
        struct module mod;
        struct compress_state_real *ptr;
        platform_spin_t spin;
};

typedef struct compress_state compress_state_proxy;

struct module compress_init_noerr;

static void init_compressions(void);
static struct video_frame *compress_frame_tiles(struct compress_state_real *s, struct video_frame *frame,
                int buffer_index, struct module *parent);
static void *compress_tile(void *arg);
static struct response *compress_change_callback(struct module *receiver, struct message *msg);
static int compress_init_real(struct module *parent, char *config_string,
                struct compress_state_real **state);
static void compress_done_real(struct compress_state_real *s);
static void compress_done(struct module *mod);

struct compress_t compress_modules[] = {
#if defined HAVE_FASTDXT || defined BUILD_LIBRARIES
        {
                "FastDXT",
                "fastdxt",
                MK_NAME(fastdxt_init),
                MK_NAME(NULL),
                MK_NAME(fastdxt_compress_tile),
                NULL
        },
#endif
#if defined HAVE_DXT_GLSL || defined BUILD_LIBRARIES
        {
                "RTDXT",
                "rtdxt",
                MK_NAME(dxt_glsl_compress_init),
                MK_NAME(dxt_glsl_compress),
                MK_NAME(NULL),
                NULL
        },
#endif
#if defined HAVE_JPEG || defined  BUILD_LIBRARIES
        {
                "JPEG",
                "jpeg",
                MK_NAME(jpeg_compress_init),
                MK_NAME(jpeg_compress),
                MK_NAME(NULL),
                NULL
        },
#endif
#if defined HAVE_COMPRESS_UYVY || defined  BUILD_LIBRARIES
        {
                "UYVY",
                "uyvy",
                MK_NAME(uyvy_compress_init),
                MK_NAME(uyvy_compress),
                MK_NAME(NULL),
                NULL
        },
#endif
#if defined HAVE_LAVC || defined  BUILD_LIBRARIES
        {
                "libavcodec",
                "libavcodec",
                MK_NAME(libavcodec_compress_init),
                MK_NAME(NULL),
                MK_NAME(libavcodec_compress_tile),
                NULL
        },
#endif
        {
                "none",
                NULL,
                MK_STATIC(none_compress_init),
                MK_STATIC(none_compress),
                MK_STATIC(NULL),
                NULL
        },
};

#define MAX_COMPRESS_MODULES (sizeof(compress_modules)/sizeof(struct compress_t))

static struct compress_t *available_compress_modules[MAX_COMPRESS_MODULES];
static int compress_modules_count = 0;

#ifdef BUILD_LIBRARIES
/* definded in video_display.c */
void *open_library(const char *name);

static void *compress_open_library(const char *compress_name)
{
        char name[128];
        snprintf(name, sizeof(name), "vcompress_%s.so.%d", compress_name, VIDEO_COMPRESS_ABI_VERSION);

        return open_library(name);
}

static int compress_fill_symbols(struct compress_t *compression)
{
        void *handle = compression->handle;

        compression->init = (compress_init_t)
                dlsym(handle, compression->init_str);
        compression->compress_frame = (struct video_frame * (*)(void *, struct video_frame *, int))
                dlsym(handle, compression->compress_frame_str);
        compression->compress_tile = (struct tile * (*)(void *, struct tile*, struct video_desc *, int))
                dlsym(handle, compression->compress_tile_str);

        if(!compression->init || (compression->compress_frame == 0 && compression->compress_tile == 0)
                        ) {
                fprintf(stderr, "Library %s opening error: %s \n", compression->library_name, dlerror());
                return FALSE;
        }
        return TRUE;
}
#endif

static pthread_once_t compression_list_initialized = PTHREAD_ONCE_INIT;

static void init_compressions(void)
{
        unsigned int i;
        for(i = 0; i < sizeof(compress_modules)/sizeof(struct compress_t); ++i) {
#ifdef BUILD_LIBRARIES
                if(compress_modules[i].library_name) {
                        int ret;
                        compress_modules[i].handle = compress_open_library(compress_modules[i].library_name);
                        if(!compress_modules[i].handle) continue;
                        ret = compress_fill_symbols(&compress_modules[i]);
                        if(!ret) {
                                fprintf(stderr, "Opening symbols from library %s failed.\n", compress_modules[i].library_name);
                                continue;
                        }
                }
#endif
                available_compress_modules[compress_modules_count] = &compress_modules[i];
                compress_modules_count++;
        }
}

void show_compress_help()
{
        int i;
        pthread_once(&compression_list_initialized, init_compressions);
        printf("Possible compression modules (see '-c <module>:help' for options):\n");
        for(i = 0; i < compress_modules_count; ++i) {
                printf("\t%s\n", available_compress_modules[i]->name);
        }
}

static struct response *compress_change_callback(struct module *receiver, struct message *msg)
{
        struct msg_change_compress_data *data =
                (struct msg_change_compress_data *) msg;
        compress_state_proxy *proxy = receiver->priv_data;

        if(data->what == CHANGE_PARAMS) {
                platform_spin_lock(&proxy->spin);
                struct response *resp = NULL;
                for(unsigned int i = 0; i < proxy->ptr->state_count; ++i) {
                        if(resp) {
                                resp->deleter(resp);
                        }
                        struct msg_change_compress_data *tmp_data =
                                (struct msg_change_compress_data *)
                                new_message(sizeof(struct msg_change_compress_data));
                        tmp_data->what = data->what;
                        strncpy(tmp_data->config_string, data->config_string,
                                        sizeof(tmp_data->config_string) - 1);
                        resp = send_message_to_receiver(proxy->ptr->state[i],
                                        (struct message *) tmp_data);
                }
                platform_spin_unlock(&proxy->spin);

                free_message(msg);

                return resp;
        }

#if 0
        // this change relates to current compress, resend request to it
        if(strcasecmp(s->handle->name, data->module) == 0) {
                data->what = CHANGE_PARAMS;
                return send_message(messaging_instance(), MSG_CHANGE_COMPRESS, msg->data);
        }
#endif

        struct compress_state_real *new_state;
        char config[1024];
        strncpy(config, data->config_string, sizeof(config));

        int ret = compress_init_real(&proxy->mod, config, &new_state);
        if(ret == 0) {
                struct compress_state_real *old = proxy->ptr;
                platform_spin_lock(&proxy->spin);
                proxy->ptr = new_state;
                platform_spin_unlock(&proxy->spin);
                compress_done_real(old);

                return new_response(RESPONSE_OK, NULL);
        }

        return new_response(RESPONSE_INT_SERV_ERR, NULL);
}

int compress_init(struct module *parent, char *config_string, struct compress_state **state) {
        struct compress_state_real *s;

        compress_state_proxy *proxy;
        proxy = malloc(sizeof(compress_state_proxy));

        module_init_default(&proxy->mod);
        proxy->mod.cls = MODULE_CLASS_COMPRESS;
        proxy->mod.priv_data = proxy;
        proxy->mod.deleter = compress_done;
        proxy->mod.msg_callback = compress_change_callback;

        int ret = compress_init_real(&proxy->mod, config_string, &s);
        if(ret == 0) {
                proxy->ptr = s;

                platform_spin_init(&proxy->spin);
                *state = proxy;
                //subscribe_messages(messaging_instance(), MSG_CHANGE_COMPRESS, compress_change_callback,
                //                proxy);
        } else {
                free(proxy);
        }

        module_register(&proxy->mod, parent);

        return ret;
}

static int compress_init_real(struct module *parent, char *config_string, struct compress_state_real **state)
{
        struct compress_state_real *s;
        char *compress_options = NULL;
        
        if(!config_string) 
                return -1;
        
        if(strcmp(config_string, "help") == 0)
        {
                show_compress_help();
                return 1;
        }

        pthread_once(&compression_list_initialized, init_compressions);
        
        s = (struct compress_state_real *) calloc(1, sizeof(struct compress_state_real));
        s->state_count = 1;
        if(strcmp(config_string, "none") == 0) {
                s->uncompressed = TRUE;
        } else {
                s->uncompressed = FALSE;
        }
        int i;
        for(i = 0; i < compress_modules_count; ++i) {
                if(strncasecmp(config_string, available_compress_modules[i]->name,
                                strlen(available_compress_modules[i]->name)) == 0) {
                        s->handle = available_compress_modules[i];
                        if(config_string[strlen(available_compress_modules[i]->name)] == ':') 
                                compress_options = config_string +
                                        strlen(available_compress_modules[i]->name) + 1;
                        else
                                compress_options = "";
                }
        }
        if(!s->handle) {
                fprintf(stderr, "Unknown compression: %s\n", config_string);
                free(s);
                return -1;
        }
        strncpy(s->compress_options, compress_options, sizeof(s->compress_options) - 1);
        s->compress_options[sizeof(s->compress_options) - 1] = '\0';
        if(s->handle->init) {
                s->state = calloc(1, sizeof(struct module *));
                char compress_options[1024];
                strncpy(compress_options, s->compress_options, sizeof(compress_options) - 1);
                compress_options[sizeof(compress_options) - 1] = '\0';
                s->state[0] = s->handle->init(parent, compress_options);
                if(!s->state[0]) {
                        fprintf(stderr, "Compression initialization failed: %s\n", config_string);
                        free(s->state);
                        free(s);
                        return -1;
                }
                if(s->state[0] == &compress_init_noerr) {
                        free(s->state);
                        free(s);
                        return 1;
                }
                for(int i = 0; i < 2; ++i) {
                        s->out_frame[i] = vf_alloc(1);
                }
        } else {
                return -1;
        }

        *state = s;
        return 0;
}

const char *get_compress_name(compress_state_proxy *proxy)
{
        if(proxy)
                return proxy->ptr->handle->name;
        else
                return NULL;
}

struct video_frame *compress_frame(compress_state_proxy *proxy, struct video_frame *frame, int buffer_index)
{
        struct video_frame *ret;
        if(!proxy)
                return NULL;

        platform_spin_lock(&proxy->spin);

        struct compress_state_real *s = proxy->ptr;

        if(s->handle->compress_frame) {
                ret = s->handle->compress_frame(s->state[0], frame, buffer_index);
        } else if(s->handle->compress_tile) {
                ret = compress_frame_tiles(s, frame, buffer_index, &proxy->mod);
        } else {
                ret = NULL;
        }

        platform_spin_unlock(&proxy->spin);

        return ret;
}

struct compress_data {
        void *state;
        struct tile *tile;
        struct video_desc desc;
        int buffer_index;

        compress_tile_t callback;
        void *ret;
};

static void *compress_tile(void *arg) {
        struct compress_data *s = (struct compress_data *) arg;

        s->ret = s->callback(s->state, s->tile, &s->desc, s->buffer_index);

        return s;
}

static struct video_frame *compress_frame_tiles(struct compress_state_real *s, struct video_frame *frame,
                int buffer_index, struct module *parent)
{
        if(frame->tile_count != s->state_count) {
                s->state = realloc(s->state, frame->tile_count * sizeof(struct module *));
                for(unsigned int i = s->state_count; i < frame->tile_count; ++i) {
                        char compress_options[1024];
                        strncpy(compress_options, s->compress_options, sizeof(compress_options));
                        compress_options[sizeof(compress_options) - 1] = '\0';
                        s->state[i] = s->handle->init(parent, compress_options);
                        if(!s->state[i]) {
                                fprintf(stderr, "Compression initialization failed\n");
                                return NULL;
                        }
                }
                for(int i = 0; i < 2; ++i) {
                        vf_free(s->out_frame[i]);
                        s->out_frame[i] = vf_alloc(frame->tile_count);
                }
                s->state_count = frame->tile_count;
        }

        task_result_handle_t task_handle[frame->tile_count];

        struct compress_data data_tile[frame->tile_count];
        for(unsigned int i = 0; i < frame->tile_count; ++i) {
                struct compress_data *data = &data_tile[i];
                data->state = s->state[i];
                data->tile = &frame->tiles[i];
                data->desc = video_desc_from_frame(frame);
                data->desc.tile_count = 1;
                data->buffer_index = buffer_index;;
                data->callback = s->handle->compress_tile;

                task_handle[i] = task_run_async(compress_tile, data);
        }

        for(unsigned int i = 0; i < frame->tile_count; ++i) {
                struct compress_data *data = wait_task(task_handle[i]);

                if(i == 0) { // update metadata from first tile
                        data->desc.tile_count = frame->tile_count;
                        vf_write_desc(s->out_frame[buffer_index], data->desc);
                }

                if(data->ret) {
                        memcpy(&s->out_frame[buffer_index]->tiles[i], data->ret, sizeof(struct tile));
                } else {
                        return NULL;
                }
        }

        return s->out_frame[buffer_index];
}

static void compress_done(struct module *mod)
{
        if(!mod)
                return;

        compress_state_proxy *proxy = mod->priv_data;
        struct compress_state_real *s = proxy->ptr;
        compress_done_real(s);

        platform_spin_destroy(&proxy->spin);
        free(proxy);
}

static void compress_done_real(struct compress_state_real *s)
{
        if(!s)
                return;

        for(unsigned int i = 0; i < s->state_count; ++i) {
                module_done(s->state[i]);
        }
        free(s->state);
        free(s);
}

int is_compress_none(compress_state_proxy *proxy)
{
        assert(proxy != NULL);
        struct compress_state_real *s = proxy->ptr;

        return s->uncompressed;
}

