/*
 * FILE:    vo_postprocess/interlace.c
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
#include "vo_postprocess/interlace.h"
#include "video_display.h"

enum last_frame {
        ODD = 0,
        EVEN = 1
};

struct state_interlace {
        struct video_frame *odd,
                           *even;

        enum last_frame last;
};

static void usage()
{
        printf("-p interlace\n");
}

void * interlace_init(char *config) {
        struct state_interlace *s;

        if(config && strcmp(config, "help") == 0) {
                usage();
                return NULL;
        }

        s = (struct state_interlace *) 
                        malloc(sizeof(struct state_interlace));

        assert(s != NULL);
        s->odd = NULL;
        s->even = NULL;
        s->last = EVEN; // next one will be od
        
        return s;
}

bool interlace_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return false;
}

int interlace_reconfigure(void *state, struct video_desc desc)
{
        struct state_interlace *s = (struct state_interlace *) state;

        vf_free_data(s->odd);
        vf_free_data(s->even);
        s->odd = vf_alloc_desc_data(desc);
        s->even = vf_alloc_desc_data(desc);

        s->last = EVEN;

        return TRUE;
}

struct video_frame * interlace_getf(void *state)
{
        struct state_interlace *s = (struct state_interlace *) state;
        struct video_frame *ret;

        if((s->last + 1) % 2 == 0) {
                ret = s->odd;
        } else {
                ret = s->even;
        }

        s->last = (s->last + 1) % 2;

        return ret;
}

bool interlace_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_interlace *s = (struct state_interlace *) state;

        UNUSED(in); // we know which frame it is

        if(s->last == ODD) {
                return false;
        }

        for(unsigned int tile = 0; tile < out->tile_count; ++tile) {
                for (unsigned int i = 0; i < out->tiles[0].height; ++i) {
                        memcpy(out->tiles[tile].data + i * req_pitch,
                                        (i % 2 == 0 ?
                                         s->odd->tiles[tile].data :
                                         s->even->tiles[tile].data) 
                                        + i * s->odd->tiles[tile].linesize,
                                        s->odd->tiles[tile].linesize);
                }
        }

        return true;
}

void interlace_done(void *state)
{
        struct state_interlace *s = (struct state_interlace *) state;
        
        vf_free_data(s->odd);
        vf_free_data(s->even);

        free(state);
}

void interlace_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_interlace *s = (struct state_interlace *) state;

        out->width = vf_get_tile(s->odd, 0)->width;
        out->height = vf_get_tile(s->odd, 0)->height;
        out->color_spec = s->odd->color_spec;
        out->interlacing = INTERLACED_MERGED;
        out->fps = s->odd->fps / 2.0;
        out->tile_count = s->odd->tile_count;

        UNUSED(in_display_mode);
        //*in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}
