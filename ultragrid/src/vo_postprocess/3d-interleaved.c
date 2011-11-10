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

#include "config.h"
#include "config_unix.h"
#include "debug.h"

#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include "vo_postprocess/3d-interleaved.h"

struct state_interleaved_3d {
        struct video_frame *in;
};

void * interleaved_3d_init(char *config) {
        struct state_interleaved_3d *s;
        UNUSED(config);
        
        s = (struct state_interleaved_3d *) 
                        malloc(sizeof(struct state_interleaved_3d));
        s->in = vf_alloc(2, 1);
        
        return s;
}

struct video_frame * interleaved_3d_postprocess_reconfigure(void *state, struct video_desc desc, struct tile_info ti)
{
        struct state_interleaved_3d *s = (struct state_interleaved_3d *) state;
        
        UNUSED(ti);
        assert(ti.x_count == 2 && ti.y_count == 1);
        
        s->in->color_spec = desc.color_spec;
        s->in->fps = desc.fps;
        s->in->aux = desc.aux;
        tile_get(s->in, 0, 0)->width = 
                tile_get(s->in, 1, 0)->width = desc.width / 2;
        tile_get(s->in, 0, 0)->height = 
                tile_get(s->in, 1, 0)->height = desc.height;
                
        tile_get(s->in, 0, 0)->data_len =
                tile_get(s->in, 1, 0)->data_len = vc_get_linesize(desc.width / 2, desc.color_spec)
                                * desc.height;
        tile_get(s->in, 0, 0)->data = malloc(tile_get(s->in, 0, 0)->data_len);
        tile_get(s->in, 1, 0)->data = malloc(tile_get(s->in, 1, 0)->data_len);
        
        return s->in;
}

void interleaved_3d_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        int x;
        
        char *out_data = tile_get(out, 0, 0)->data;
        int linesize = vc_get_linesize(tile_get(out, 0, 0)->width, out->color_spec);
        
        /* we compute avg from line k/2*2 and k/2*2+1 for left eye and put
         * to (k/2*2)th line. Than we compute avg of same lines number
         * and put it to the following line, which creates interleaved stereo */
        for (x = 0; x < tile_get(out, 0, 0)->height; ++x) {
                int linepos;
                char *line1 = tile_get(in, x % 2, 0)->data +  (x / 2) * 2 * linesize;
                char *line2 = tile_get(in, x % 2, 0)->data +  ((x / 2) * 2 + 1) * linesize;
                
                for(linepos = 0; linepos < linesize; linepos += 16) {
                        asm volatile ("movdqu (%0), %%xmm0\n"
                                      "pavgb (%1), %%xmm0\n"
                                      "movdqu %%xmm0, (%2)\n"
                                      ::"r" ((unsigned long *)
                                                            line1),
                                      "r"((unsigned long *) line2),
                                      "r"((unsigned long *) out_data));
                        out_data += 16;
                        line1 += 16;
                        line2 += 16;
                }
        }
}

void interleaved_3d_done(void *state)
{
        struct state_interleaved_3d *s = (struct state_interleaved_3d *) state;
        
        free(tile_get(s->in, 0, 0)->data);
        free(tile_get(s->in, 1, 0)->data);
        vf_free(s->in);
        free(state);
}

void interleaved_3d_get_out_desc(void *state, struct video_desc_ti *out)
{
        struct state_interleaved_3d *s = (struct state_interleaved_3d *) state;

        out->desc.width = tile_get(s->in, 0, 0)->width; /* not *2 !!!!!!*/
        out->desc.height = tile_get(s->in, 0, 0)->height;
        out->desc.color_spec = s->in->color_spec;
        out->desc.aux = s->in->aux;
        out->desc.fps = s->in->fps;
        
        out->ti.x_count = s->in->grid_width;
        out->ti.y_count = s->in->grid_height;
}
