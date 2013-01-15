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
#include "vo_postprocess.h"
#include "vo_postprocess/interlace.h"
#include "vo_postprocess/double-framerate.h"
#include "vo_postprocess/scale.h"
#include "vo_postprocess/split.h"
#include "vo_postprocess/3d-interlaced.h"
#include "lib_common.h"

extern char **uv_argv;

struct vo_postprocess_t {
        const char * name; /* this is the user given name */
        const char * library_name; /* must be NULL if library is allways statically lined */
        vo_postprocess_init_t init;
        const char *init_str;
        vo_postprocess_reconfigure_t reconfigure;
        const char *reconfigure_str;
        vo_postprocess_getf_t getf;
        const char *getf_str;
        vo_postprocess_get_out_desc_t get_out_desc;
        const char *get_out_desc_str;
        vo_postprocess_get_property_t get_property;
        const char *get_property_str;
        vo_postprocess_t vo_postprocess;
        const char *vo_postprocess_str;
        vo_postprocess_done_t done;
        const char *done_str;

        void *handle; /* for dynamically loaded libraries */
};

struct vo_postprocess_state {
        struct vo_postprocess_t *handle;
        void *state;
};

struct vo_postprocess_t vo_postprocess_modules[] = {
        {"interlace",
                NULL,
                MK_STATIC(interlace_init),
                MK_STATIC(interlace_reconfigure),
                MK_STATIC(interlace_getf),
                MK_STATIC(interlace_get_out_desc),
                MK_STATIC(interlace_get_property),
                MK_STATIC(interlace_postprocess),
                MK_STATIC(interlace_done),
                NULL
        },
        {"double-framerate",
                NULL,
                MK_STATIC(df_init),
                MK_STATIC(df_reconfigure),
                MK_STATIC(df_getf),
                MK_STATIC(df_get_out_desc),
                MK_STATIC(df_get_property),
                MK_STATIC(df_postprocess),
                MK_STATIC(df_done),
                NULL
        },
#if defined HAVE_SCALE || defined BUILD_LIBRARIES
        {"scale",
                "scale",
                MK_NAME(scale_init),
                MK_NAME(scale_reconfigure), 
                MK_NAME(scale_getf),
                MK_NAME(scale_get_out_desc),
                MK_NAME(scale_get_property),
                MK_NAME(scale_postprocess), 
                MK_NAME(scale_done),
                NULL
        },
#endif /* HAVE_SCREEN_CAP */
        {"split",
                NULL,
                MK_STATIC(split_init),
                MK_STATIC(split_postprocess_reconfigure),
                MK_STATIC(split_getf),
                MK_STATIC(split_get_out_desc),
                MK_STATIC(split_get_property),
                MK_STATIC(split_postprocess),
                MK_STATIC(split_done),
                NULL
        },
        {"3d-interlaced",
                NULL,
                MK_STATIC(interlaced_3d_init),
                MK_STATIC(interlaced_3d_postprocess_reconfigure),
                MK_STATIC(interlaced_3d_getf),
                MK_STATIC(interlaced_3d_get_out_desc),
                MK_STATIC(interlaced_3d_get_property),
                MK_STATIC(interlaced_3d_postprocess),
                MK_STATIC(interlaced_3d_done),
                NULL
        },
        { NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL }
};



#ifdef BUILD_LIBRARIES
/* definded in video_display.c */
void *open_library(const char *name);
static void *vo_pp_open_library(const char *vidcap_name);
static int vo_pp_fill_symbols(struct vo_postprocess_t *device);

static void *vo_pp_open_library(const char *vidcap_name)
{
        char name[128];
        snprintf(name, sizeof(name), "vo_pp_%s.so.%d", vidcap_name, VO_PP_ABI_VERSION);

        return open_library(name);
}


static int vo_pp_fill_symbols(struct vo_postprocess_t *device)
{
        void *handle = device->handle;
        
        device->init = (vo_postprocess_init_t)
                dlsym(handle, device->init_str);
        device->reconfigure =
                (vo_postprocess_reconfigure_t )
                dlsym(handle, device->reconfigure_str);
        device->getf =
                (vo_postprocess_getf_t)
                dlsym(handle, device->getf_str);
        device->get_out_desc =
                (vo_postprocess_get_out_desc_t)
                dlsym(handle, device->get_out_desc_str);
        device->get_property =
                (vo_postprocess_get_property_t)
                dlsym(handle, device->get_property_str);
        device->vo_postprocess=
                (vo_postprocess_t)
                dlsym(handle, device->vo_postprocess_str);
        device->done =
                (vo_postprocess_done_t)
                dlsym(handle, device->done_str);

        if(!device->init || !device->reconfigure ||
                        !device->getf ||
                        !device->get_out_desc ||
                        !device->get_property ||
                        !device->vo_postprocess ||
                        !device->done) {
                fprintf(stderr, "Library %s opening error: %s \n", device->library_name, dlerror());
                return FALSE;
        }

        return TRUE;
}
#endif /* BUILD_LIBRARIES */



void show_vo_postprocess_help()
{
        int i;
        printf("Possible postprocess modules:\n");
        for(i = 0; vo_postprocess_modules[i].name != NULL; ++i)
                printf("\t%s\n", vo_postprocess_modules[i].name);
}

struct vo_postprocess_state *vo_postprocess_init(char *config_string)
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
        s->handle = NULL;
        int i;
        for(i = 0; vo_postprocess_modules[i].name != NULL; ++i) {
                if(strncasecmp(config_string, vo_postprocess_modules[i].name,
                                        strlen(vo_postprocess_modules[i].name)) == 0) {
                        /* found it */
#ifdef BUILD_LIBRARIES
                        if(vo_postprocess_modules[i].library_name) {
                                vo_postprocess_modules[i].handle =
                                        vo_pp_open_library(vo_postprocess_modules[i].library_name);
                                if(!vo_postprocess_modules[i].handle) {
                                        free(s);
                                        fprintf(stderr, "Unable to load postprocess library.\n");
                                        return NULL;
                                }
                                int ret = vo_pp_fill_symbols(&vo_postprocess_modules[i]);
                                if(!ret) {
                                        free(s);
                                        fprintf(stderr, "Unable to load postprocess library.\n");
                                        return NULL;
                                }
                        }
#endif /* BUILD_LIBRARIES */
                        s->handle = &vo_postprocess_modules[i];
                        if(config_string[strlen(vo_postprocess_modules[i].name)] == ':') 
                                vo_postprocess_options = config_string +
                                        strlen(vo_postprocess_modules[i].name) + 1;
                }
        }
        if(!s->handle) {
                fprintf(stderr, "Unknown postprocess module: %s\n", config_string);
                free(s);
                return NULL;
        }
        s->state = s->handle->init(vo_postprocess_options);
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
                return s->handle->reconfigure(s->state, desc);
        } else {
                return FALSE;
        }
}

struct video_frame * vo_postprocess_getf(struct vo_postprocess_state *s)
{
        if(s) {
                return s->handle->getf(s->state);
        } else {
                return NULL;
        }
}

bool vo_postprocess(struct vo_postprocess_state *s, struct video_frame *in,
                struct video_frame *out, int req_pitch)
{
        if(s)
                return s->handle->vo_postprocess(s->state, in, out, req_pitch);
        else
                return false;
}

void vo_postprocess_done(struct vo_postprocess_state *s)
{
        if(s) s->handle->done(s->state);
}

void vo_postprocess_get_out_desc(struct vo_postprocess_state *s, struct video_desc *out, int *display_mode, int *out_frames_count)
{
        if(s) s->handle->get_out_desc(s->state, out, display_mode, out_frames_count);
}

bool vo_postprocess_get_property(struct vo_postprocess_state *s, int property, void *val, size_t *len)
{
        if(s) return s->handle->get_property(s, property, val, len);
        else return false;
}

