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

struct compress_t {
        const char * name;
        compress_init_t init;
        compress_compress_t compress;
        compress_done_t done;
};

struct compress_state {
        struct compress_t *handle;
        void *state;
};

const struct compress_t compress_modules[] = {
#ifdef HAVE_FASTDXT
        {"FastDXT", fastdxt_init, fastdxt_compress, fastdxt_done },
#endif
#ifdef HAVE_DXT_GLSL
        {"RTDXT", dxt_glsl_compress_init, dxt_glsl_compress, dxt_glsl_compress_done},
#endif
#ifdef HAVE_JPEG
        {"JPEG", jpeg_compress_init, jpeg_compress, jpeg_compress_done},
#endif
        {NULL, NULL, NULL, NULL}
};

void show_compress_help()
{
        int i;
        printf("Possible compression modules (see '-c <module>:help' for options):\n");
        for(i = 0; compress_modules[i].name != NULL; ++i)
                printf("\t%s\n", compress_modules[i].name);
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
        
        s = (struct compress_state *) malloc(sizeof(struct compress_state));
        s->handle = NULL;
        int i;
        for(i = 0; compress_modules[i].name != NULL; ++i) {
                if(strncasecmp(config_string, compress_modules[i].name,
                                strlen(compress_modules[i].name)) == 0) {
                        s->handle = &compress_modules[i];
                        if(config_string[strlen(compress_modules[i].name)] == ':') 
                                        compress_options = config_string +
                                                strlen(compress_modules[i].name) + 1;
                }
        }
        if(!s->handle) {
                fprintf(stderr, "Unknown compression: %s\n", config_string);
                free(s);
                return NULL;
        }
        s->state = s->handle->init(compress_options);
        if(!s->state) {
                fprintf(stderr, "Compression initialization failed: %s\n", config_string);
                free(s);
                return NULL;
        }
        return s;
}

const char *get_compress_name(struct compress_state *s)
{
        if(s) return s->handle->name;
}

struct video_frame *compress_frame(struct compress_state *s, struct video_frame *frame)
{
        if(s)
                return s->handle->compress(s->state, frame);
}

void compress_done(struct compress_state *s)
{
        if(s) s->handle->done(s->state);
}

