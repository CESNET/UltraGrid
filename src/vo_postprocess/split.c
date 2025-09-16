/**
 * @file   vo_postprocess/split.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2025 CESNET, zájmové sdružení právnických osob
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

#include <assert.h>          // for assert
#include <stdbool.h>         // for bool, true, false
#include <stdio.h>           // for printf
#include <stdlib.h>          // for free, NULL, atoi, malloc, size_t
#include <string.h>          // for strtok_r, strcmp, strdup

#include "debug.h"
#include "lib_common.h"
#include "utils/vf_split.h"
#include "video.h"
#include "video_display.h" /* DISPLAY_PROPERTY_VIDEO_SEPARATE_FILES */
#include "vo_postprocess.h" /* VO_PP_DOES_CHANGE_TILING_MODE */
#include <pthread.h>
#include <stdlib.h>

#define MOD_NAME "[split] "

struct state_split {
        struct video_frame *in;
        int grid_width, grid_height;
};


static bool split_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);

        switch(property) {
                case VO_PP_DOES_CHANGE_TILING_MODE:
                        if(*len >= sizeof(bool)) {
                                *(bool *) val = true;
                                *len = sizeof(bool);
                        } else {
                                *len = 0;
                        }
                        return true;
        }
        return false;
}

static void usage()
{
        printf("usage:\n");
        printf("-p split:<X>:<Y>\n");
        printf("\tsplit to XxY tiles\n");
}

static void * split_init(const char *config) {
        char *save_ptr = NULL;
        char *item;

        if (strcmp(config, "help") == 0) {
                usage();
                return NULL;
        }
        struct state_split *s = (struct state_split *)
                        malloc(sizeof(struct state_split));

        char *tmp = strdup(config);
        assert(tmp != NULL);
        
        item = strtok_r(tmp, ":", &save_ptr);
        if (item != NULL) {
                s->grid_width = atoi(item);
                item          = strtok_r(NULL, ":", &save_ptr);
        }
        if(!item) {
                MSG(ERROR, "Wrong usage!\n");
                usage();
                free(s);
                free(tmp);
                return NULL;
        }

        s->grid_height = atoi(item);
        free(tmp);

        s->in = vf_alloc(1);
        
        return s;
}

static bool
split_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_split *s = (struct state_split *) state;
        struct tile *in_tile = vf_get_tile(s->in, 0);
        
        s->in->color_spec = desc.color_spec;
        s->in->fps = desc.fps;
        s->in->interlacing = desc.interlacing;

        in_tile->width = desc.width;
        in_tile->height = desc.height;

        in_tile->data_len = vc_get_linesize(desc.width, desc.color_spec) *
                desc.height;
        in_tile->data = malloc(in_tile->data_len);
        
        return true;
}

static struct video_frame * split_getf(void *state)
{
        struct state_split *s = (struct state_split *) state;

        return s->in;
}

static bool split_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_split *s = (struct state_split *) state;
        UNUSED(req_pitch);

        vf_split(out, in, s->grid_width, s->grid_height, false);

        return true;
}

static void split_done(void *state)
{
        struct state_split *s = (struct state_split *) state;
        
        free(vf_get_tile(s->in, 0)->data);
        vf_free(s->in);
        free(state);
}

static void split_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
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

static const struct vo_postprocess_info vo_pp_split_info = {
        split_init,
        split_postprocess_reconfigure,
        split_getf,
        split_get_out_desc,
        split_get_property,
        split_postprocess,
        split_done,
};

REGISTER_MODULE(split, &vo_pp_split_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);

