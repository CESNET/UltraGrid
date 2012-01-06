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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "video_codec.h"
#include "video_compress.h"
#include "video_compress/dxt_glsl.h"
#include "video_compress/fastdxt.h"
#include "video_compress/jpeg.h"
#include "lib_common.h"

/* *_str are symbol names inside library */
struct compress_t {
        const char        * name;
        const char        * library_name;

        compress_init_t     init;
        const char        * init_str;
        compress_compress_t compress;
        const char         *compress_str;
        compress_done_t     done;
        const char         *done_str;

        void *handle;
};

struct compress_state {
        struct compress_t *handle;
        void *state;
};

void init_compressions(void);


struct compress_t compress_modules[] = {
#if defined HAVE_FASTDXT || defined BUILD_LIBRARIES
        {"FastDXT", "fastdxt", MK_NAME(fastdxt_init), MK_NAME(fastdxt_compress), MK_NAME(fastdxt_done), NULL },
#endif
#if defined HAVE_DXT_GLSL || defined BUILD_LIBRARIES
        {"RTDXT", "rtdxt", MK_NAME(dxt_glsl_compress_init), MK_NAME(dxt_glsl_compress), MK_NAME(dxt_glsl_compress_done), NULL},
#endif
#if defined HAVE_JPEG || defined  BUILD_LIBRARIES
        {"JPEG", "jpeg", MK_NAME(jpeg_compress_init), MK_NAME(jpeg_compress), MK_NAME(jpeg_compress_done), NULL},
#endif
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

        compression->init = (void *(*) (char *))
                dlsym(handle, compression->init_str);
        compression->compress = (struct video_frame * (*)(void *, struct video_frame *))
                dlsym(handle, compression->compress_str);
        compression->done = (void (*)(void *))
                dlsym(handle, compression->done_str);

        
        if(!compression->init || !compression->compress || !compression->done) {
                fprintf(stderr, "Library %s opening error: %s \n", compression->library_name, dlerror());
                return FALSE;
        }
        return TRUE;
}
#endif

void init_compressions(void)
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
        init_compressions();
        printf("Possible compression modules (see '-c <module>:help' for options):\n");
        for(i = 0; i < compress_modules_count; ++i) {
                printf("\t%s\n", available_compress_modules[i]->name);
        }
}

struct compress_state *compress_init(char *config_string)
{
        struct compress_state *s;
        char *compress_options = NULL;
        
        if(!config_string) 
                return NULL;
        
        if(strcmp(config_string, "help") == 0)
        {
                show_compress_help();
                return NULL;
        }

        init_compressions();
        
        s = (struct compress_state *) malloc(sizeof(struct compress_state));
        s->handle = NULL;
        int i;
        for(i = 0; i < compress_modules_count; ++i) {
                if(strncasecmp(config_string, available_compress_modules[i]->name,
                                strlen(available_compress_modules[i]->name)) == 0) {
                        s->handle = available_compress_modules[i];
                        if(config_string[strlen(available_compress_modules[i]->name)] == ':') 
                                        compress_options = config_string +
                                                strlen(available_compress_modules[i]->name) + 1;
                }
        }
        if(!s->handle) {
                fprintf(stderr, "Unknown compression: %s\n", config_string);
                free(s);
                return NULL;
        }
        if(s->handle->init) {
                s->state = s->handle->init(compress_options);
                if(!s->state) {
                        fprintf(stderr, "Compression initialization failed: %s\n", config_string);
                        free(s);
                        return NULL;
                }
        } else {
                return NULL;
        }
        return s;
}

const char *get_compress_name(struct compress_state *s)
{
        if(s)
                return s->handle->name;
        else
                return NULL;
}

struct video_frame *compress_frame(struct compress_state *s, struct video_frame *frame)
{
        if(s)
                return s->handle->compress(s->state, frame);
        else
                return NULL;
}

void compress_done(struct compress_state *s)
{
        if(s) s->handle->done(s->state);
}

