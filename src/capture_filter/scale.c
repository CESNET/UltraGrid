/*
 * FILE:    capture_filter/scale.c
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_codec.h"
#include "utils/video_frame_pool.h"

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_scale {
        struct video_frame *frame;
};

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        UNUSED(cfg);
        struct video_desc desc;

        struct state_scale *s = (struct state_scale *) calloc(1, sizeof(struct state_scale));
        desc.width = 3840;
        desc.height = 2160;
        desc.color_spec = UYVY;
        desc.fps = 25;
        desc.interlacing = PROGRESSIVE;
        desc.tile_count = 1;
        s->frame = vf_alloc_desc(desc);
        s->frame->tiles[0].data = (char *) malloc(s->frame->tiles[0].data_len);

        *state = s;
        return 0;
}

static void done(void *state)
{
        struct state_scale *s = (struct state_scale *) state;

        vf_free(s->frame);
        free(state);
}

static struct video_frame *filter(void *state, struct video_frame *in_frame)
{
        struct state_scale *s = (struct state_scale *) state;

        uint32_t *in = (uint32_t *) in_frame->tiles[0].data;
        uint32_t *out1 = (uint32_t *) s->frame->tiles[0].data;
        uint32_t *out2 = (uint32_t *) (s->frame->tiles[0].data +
                vc_get_linesize(3840, in_frame->color_spec));
        for (int y = 0; y < 2160; y += 2) {
                for (int x = 0; x < 3840; x += sizeof(uint32_t)) {
                        *out1++ = *out2++ = *in;
                        *out1++ = *out2++ = *in++;
                }
                out1 += vc_get_linesize(3840, in_frame->color_spec) / sizeof(uint32_t);
                out2 += vc_get_linesize(3840, in_frame->color_spec) / sizeof(uint32_t);
        }
        
        VIDEO_FRAME_DISPOSE(in_frame);

        return s->frame;
}

static const struct capture_filter_info capture_filter_scale = {
        .name = "scale",
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(scale, &capture_filter_scale, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

