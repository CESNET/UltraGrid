/**
 * @file   vo_postprocess/3d-interlaced.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2015 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <pthread.h>
#include <stdlib.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h" /* DISPLAY_PROPERTY_VIDEO_SEPARATE_FILES */
#include "vo_postprocess.h"

#define MOD_NAME "[interlaced_3d] "

struct state_interlaced_3d {
        struct video_frame *in;
};

static bool interlaced_3d_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return false;
}


static void * interlaced_3d_init(const char *config) {
        if (strcmp(config, "help") == 0) {
                printf("3d-interlaced takes no parameters.\n");
                return NULL;
        }
#ifdef __SSE2__
        struct state_interlaced_3d *s = (struct state_interlaced_3d *)
                malloc(sizeof(struct state_interlaced_3d));
        s->in = vf_alloc(2);
        return s;
#else
        log_msg(LOG_LEVEL_ERROR, MOD_NAME "SSE2 not supported on this platform!\n");
        return NULL;
#endif
}

static int interlaced_3d_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_interlaced_3d *s = (struct state_interlaced_3d *) state;
        
        assert(desc.tile_count == 2);
        
        s->in->color_spec = desc.color_spec;
        s->in->fps = desc.fps;
        s->in->interlacing = desc.interlacing;
        vf_get_tile(s->in, 0)->width = 
                vf_get_tile(s->in, 1)->width = desc.width;
        vf_get_tile(s->in, 0)->height = 
                vf_get_tile(s->in, 1)->height = desc.height;
                
        vf_get_tile(s->in, 0)->data_len =
                vf_get_tile(s->in, 1)->data_len = vc_get_linesize(desc.width, desc.color_spec)
                                * desc.height;
        vf_get_tile(s->in, 0)->data = malloc(vf_get_tile(s->in, 0)->data_len);
        vf_get_tile(s->in, 1)->data = malloc(vf_get_tile(s->in, 1)->data_len);
        
        return TRUE;
}

static struct video_frame * interlaced_3d_getf(void *state)
{
        struct state_interlaced_3d *s = (struct state_interlaced_3d *) state;

        return s->in;
}

/**
 * Creates from 2 tiles (left and right eye) one in interlaced format.
 *
 * @param[in]  state     postprocessor state
 * @param[in]  in        input frame. Must contain exactly 2 tiles
 * @param[out] out       output frame to be written to. Should have only ony tile
 * @param[in]  req_pitch requested pitch in buffer
 */
static bool interlaced_3d_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        UNUSED (state);
        UNUSED (req_pitch);
        
#ifdef __SSE2__
        char *out_data = vf_get_tile(out, 0)->data;
        int linesize = vc_get_linesize(vf_get_tile(out, 0)->width, out->color_spec);
        
        /* we compute avg from line k/2*2 and k/2*2+1 for left eye and put
         * to (k/2*2)th line. Than we compute avg of same lines number
         * and put it to the following line, which creates interlaced stereo */
        for (unsigned int x = 0; x < vf_get_tile(out, 0)->height; ++x) {
                int linepos;
                char *line1 = vf_get_tile(in, x % 2)->data +  (x / 2) * 2 * linesize;
                char *line2 = vf_get_tile(in, x % 2)->data +  ((x / 2) * 2 + 1) * linesize;
                
                for(linepos = 0; linepos < linesize; linepos += 16) {
                        asm volatile ("movdqu (%0), %%xmm0\n"
                                      "pavgb (%1), %%xmm0\n"
                                      "movdqu %%xmm0, (%2)\n"
                                      ::"r" ((unsigned long *)(void *)
                                                            line1),
                                      "r"((unsigned long *)(void *) line2),
                                      "r"((unsigned long *)(void *) out_data));
                        out_data += 16;
                        line1 += 16;
                        line2 += 16;
                }
        }

        return true;
#else
        UNUSED(in);
        UNUSED(out);
        return false;
#endif
}

static void interlaced_3d_done(void *state)
{
        struct state_interlaced_3d *s = (struct state_interlaced_3d *) state;
        
        free(vf_get_tile(s->in, 0)->data);
        free(vf_get_tile(s->in, 1)->data);
        vf_free(s->in);
        free(state);
}

static void interlaced_3d_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_interlaced_3d *s = (struct state_interlaced_3d *) state;

        out->width = vf_get_tile(s->in, 0)->width; /* not *2 !!!!!!*/
        out->height = vf_get_tile(s->in, 0)->height;
        out->color_spec = s->in->color_spec;
        out->interlacing = s->in->interlacing;
        out->fps = s->in->fps;
        out->tile_count = 1;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
        *out_frames = 1;
}

static const struct vo_postprocess_info vo_pp_interlaced_3d_info = {
        interlaced_3d_init,
        interlaced_3d_postprocess_reconfigure,
        interlaced_3d_getf,
        interlaced_3d_get_out_desc,
        interlaced_3d_get_property,
        interlaced_3d_postprocess,
        interlaced_3d_done,
};

REGISTER_MODULE(interlaced_3d, &vo_pp_interlaced_3d_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);

