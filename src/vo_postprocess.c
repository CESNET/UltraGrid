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
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <string.h>
#include "lib_common.h"
#include "vo_postprocess.h"

extern char **uv_argv;

struct vo_postprocess_state {
        const struct vo_postprocess_info *funcs;
        void *state;
};

void show_vo_postprocess_help()
{
        printf("Possible postprocess modules:\n");
        list_modules(LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
}

struct vo_postprocess_state *vo_postprocess_init(const char *config_string)
{
        struct vo_postprocess_state *s;
        char *vo_postprocess_options = NULL;

        if(!config_string) 
                return NULL;

        if(strcmp(config_string, "help") == 0)
        {
                show_vo_postprocess_help();
                return NULL;
        }

        s = (struct vo_postprocess_state *) malloc(sizeof(struct vo_postprocess_state));

        char *lib_name = strdup(config_string);
        if (strchr(lib_name, ':')) {
                *strchr(lib_name, ':') = '\0';
        }

        const struct vo_postprocess_info *funcs = load_library(lib_name, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
        if (!funcs) {
                fprintf(stderr, "Unknown postprocess module: %s\n", lib_name);
                free(s);
                free(lib_name);
                return NULL;
        }
        free(lib_name);

        s->funcs = funcs;
        if (strchr(config_string, ':'))
                vo_postprocess_options = strchr(config_string, ':') + 1;
        s->state = s->funcs->init(vo_postprocess_options);
        if(!s->state) {
                fprintf(stderr, "Postprocessing initialization failed: %s\n", config_string);
                free(s);
                return NULL;
        }
        return s;
}

int vo_postprocess_reconfigure(struct vo_postprocess_state *s,
                struct video_desc desc)
{
        if(s) {
                return s->funcs->reconfigure(s->state, desc);
        } else {
                return FALSE;
        }
}

struct video_frame * vo_postprocess_getf(struct vo_postprocess_state *s)
{
        if(s) {
                return s->funcs->getf(s->state);
        } else {
                return NULL;
        }
}

bool vo_postprocess(struct vo_postprocess_state *s, struct video_frame *in,
                struct video_frame *out, int req_pitch)
{
        if(s)
                return s->funcs->vo_postprocess(s->state, in, out, req_pitch);
        else
                return false;
}

void vo_postprocess_done(struct vo_postprocess_state *s)
{
        if(s) s->funcs->done(s->state);
}

void vo_postprocess_get_out_desc(struct vo_postprocess_state *s, struct video_desc *out, int *display_mode, int *out_frames_count)
{
        if(s) s->funcs->get_out_desc(s->state, out, display_mode, out_frames_count);
}

bool vo_postprocess_get_property(struct vo_postprocess_state *s, int property, void *val, size_t *len)
{
        if(s) return s->funcs->get_property(s, property, val, len);
        else return false;
}

