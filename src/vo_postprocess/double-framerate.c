/*
 * FILE:    vo_postprocess/double-framerate.c
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
#endif
#include "debug.h"

#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include "vo_postprocess/double-framerate.h"
#include "video_display.h"

struct state_df {
        struct video_frame *in;
        char *buffers[2];
        int buffer_current;
};

static void usage()
{
        printf("-p double_framerate\n");
}

void * df_init(char *config) {
        struct state_df *s;

        if(config && strcmp(config, "help") == 0) {
                usage();
                return NULL;
        }

        s = (struct state_df *) 
                        malloc(sizeof(struct state_df));

        assert(s != NULL);
        s->in = vf_alloc(1);
        s->buffers[0] = s->buffers[1] = NULL;
        s->buffer_current = 0;
        
        return s;
}

bool df_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return false;
}

int df_reconfigure(void *state, struct video_desc desc)
{
        struct state_df *s = (struct state_df *) state;
        struct tile *in_tile = vf_get_tile(s->in, 0);

        free(s->buffers[0]);
        free(s->buffers[1]);
        
        s->in->color_spec = desc.color_spec;
        s->in->fps = desc.fps;
        s->in->interlacing = desc.interlacing;
        if(desc.interlacing != INTERLACED_MERGED) {
                fprintf(stderr, "[Double Framerate] Warning: %s video detected. This filter is intended "
                               "mainly for interlaced merged video. The result might be incorrect.\n",
                               get_interlacing_description(desc.interlacing)); 
        }

        in_tile->width = desc.width;
        in_tile->height = desc.height;

        in_tile->linesize = vc_get_linesize(desc.width, desc.color_spec);
        in_tile->data_len = in_tile->linesize * desc.height;

        s->buffers[0] = malloc(in_tile->data_len);
        s->buffers[1] = malloc(in_tile->data_len);
        in_tile->data = s->buffers[s->buffer_current];
        
        return TRUE;
}

struct video_frame * df_getf(void *state)
{
        struct state_df *s = (struct state_df *) state;

        s->buffer_current = (s->buffer_current + 1) % 2;
        s->in->tiles[0].data = s->buffers[s->buffer_current];

        return s->in;
}

bool df_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_df *s = (struct state_df *) state;
        unsigned int y;

        if(in != NULL) {
                char *src = s->buffers[(s->buffer_current + 1) % 2] + vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
                char *dst = out->tiles[0].data + req_pitch;
                for (y = 0; y < out->tiles[0].height; y += 2) {
                        memcpy(dst, src, vc_get_linesize(s->in->tiles[0].width, s->in->color_spec));
                        dst += 2 * req_pitch;
                        src += 2 * vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
                }
                src = s->buffers[s->buffer_current];
                dst = out->tiles[0].data;
                for (y = 1; y < out->tiles[0].height; y += 2) {
                        memcpy(dst, src, vc_get_linesize(s->in->tiles[0].width, s->in->color_spec));
                        dst += 2 * req_pitch;
                        src += 2 * vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
                }
        } else {
                char *src = s->buffers[s->buffer_current];
                char *dst = out->tiles[0].data;
                for (y = 0; y < out->tiles[0].height; ++y) {
                        memcpy(dst, src, vc_get_linesize(s->in->tiles[0].width, s->in->color_spec));
                        dst += req_pitch;
                        src += vc_get_linesize(s->in->tiles[0].width, s->in->color_spec);
                }
        }

        return true;
}

void df_done(void *state)
{
        struct state_df *s = (struct state_df *) state;
        
        free(s->buffers[0]);
        free(s->buffers[1]);
        vf_free(s->in);
        free(state);
}

void df_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_df *s = (struct state_df *) state;

        out->width = vf_get_tile(s->in, 0)->width;
        out->height = vf_get_tile(s->in, 0)->height;
        out->color_spec = s->in->color_spec;
        out->interlacing = PROGRESSIVE;
        out->fps = s->in->fps * 2.0;
        out->tile_count = 1;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 2;
}
