/*
 * FILE:    video_decompress/dxt_glsl.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2011 CESNET z.s.p.o.
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
#include "debug.h"

#include "video_codec.h"
#include "video_display.h" /* DISPLAY_PROPERTY_VIDEO_SEPARATE_FILES */
#include "vo_postprocess.h" /* VO_PP_DOES_CHANGE_TILING_MODE */
#include <pthread.h>
#include <stdlib.h>
#include "vo_postprocess/split.h"

struct state_split {
        struct video_frame *in;
        int grid_width, grid_height;
};


bool split_get_property(void *state, int property, void *val, size_t *len)
{
        bool ret;

        UNUSED(state);

        switch(property) {
                case VO_PP_DOES_CHANGE_TILING_MODE:
                        if(*len >= sizeof(bool)) {
                                *(bool *) val = true;
                                *len = sizeof(bool);
                        } else {
                                *len = 0;
                        }
                        ret = true;
                        break;
                default:
                        ret = false;
        }

        return ret;
}

static void usage()
{
        printf("-p split:<X>:<Y>\n");
        printf("\tsplit to XxY tiles\n");
}

void * split_init(char *config) {
        struct state_split *s;
        char *save_ptr = NULL;
        char *item;

        if(!config || strcmp(config, "help") == 0) {
                usage();
                return NULL;
        }
        s = (struct state_split *) 
                        malloc(sizeof(struct state_split));
        
        item = strtok_r(config, ":", &save_ptr);
        s->grid_width = atoi(item);
        item = strtok_r(config, ":", &save_ptr);
        if(!item) {
                usage();
                free(s);
                return NULL;
        }

        s->grid_height = atoi(item);

        s->in = vf_alloc(1);
        
        return s;
}

int split_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_split *s = (struct state_split *) state;
        struct tile *in_tile = vf_get_tile(s->in, 0);
        
        s->in->color_spec = desc.color_spec;
        s->in->fps = desc.fps;
        s->in->interlacing = desc.interlacing;

        in_tile->width = desc.width;
        in_tile->height = desc.height;

        in_tile->linesize = vc_get_linesize(desc.width, desc.color_spec);
        in_tile->data_len = in_tile->linesize * desc.height;
        in_tile->data = malloc(in_tile->data_len);
        
        return TRUE;
}

struct video_frame * split_getf(void *state)
{
        struct state_split *s = (struct state_split *) state;

        return s->in;
}

bool split_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_split *s = (struct state_split *) state;
        UNUSED(req_pitch);

        vf_split(out, in, s->grid_width, s->grid_height, FALSE);

        return true;
}

void split_done(void *state)
{
        struct state_split *s = (struct state_split *) state;
        
        free(vf_get_tile(s->in, 0)->data);
        vf_free(s->in);
        free(state);
}

void split_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_split *s = (struct state_split *) state;

        out->width = vf_get_tile(s->in, 0)->width / s->grid_width;
        out->height = vf_get_tile(s->in, 0)->height / s->grid_height;
        out->color_spec = s->in->color_spec;
        out->interlacing = s->in->interlacing;
        out->fps = s->in->fps;
        out->tile_count = s->grid_width * s->grid_height;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}

