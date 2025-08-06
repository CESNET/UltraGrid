/**
 * @file   capture_filter/temporal_3d.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2025 CESNET
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

#include <assert.h>           // for assert
#include <stdint.h>           // for uint32_t
#include <stdio.h>            // for printf, NULL
#include <stdlib.h>           // for free, calloc, malloc
#include <string.h>           // for memcpy, strlen

#include "capture_filter.h"   // for CAPTURE_FILTER_ABI_VERSION, capture_fil...
#include "compat/strings.h"   // for strcasecmp
#include "lib_common.h"       // for REGISTER_MODULE, library_class
#include "types.h"            // for tile, video_frame, video_frame_callbacks
#include "utils/color_out.h"  // for color_printf, TBOLD
#include "utils/macros.h"     // for to_fourcc
#include "video_frame.h"      // for VIDEO_FRAME_DISPOSE, vf_alloc_desc, vf_...

struct module;

#define MAGIC to_fourcc('c', 'f', 't', '3')

struct state_temporal_3d {
        uint32_t magic;
        struct video_frame *f;
};

static void usage() {
        color_printf("Combines temporarily-interlaced 3D.\n\n");
        color_printf(TBOLD("temporal_3d") " usage:\n");
        printf("\ttemporal_3d\n\n");
        printf("(takes no arguments)\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
        (void) parent;

        if(strcasecmp(cfg, "help") == 0) {
                usage();
                return 1;
        }

        if(strlen(cfg) > 0) {
                usage();
                return -1;
        }

        struct state_temporal_3d *s = calloc(1 ,sizeof *s);
        s->magic = MAGIC;
        *state = s;
        return 0;
}

static void dispose_frame(struct video_frame *f) {
        VIDEO_FRAME_DISPOSE((struct video_frame *) f->callbacks.dispose_udata);
        free(f->tiles[0].data); // f->data_delter is not set so vf_free doesn't
                                // delete...
        vf_free(f);
}

static void done(void *state)
{
        struct state_temporal_3d *s = state;
        assert(s->magic == MAGIC);
        if (s->f) {
                dispose_frame(s->f);
        }
        free(s);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_temporal_3d *s = state;
        assert(s->magic == MAGIC);

        assert(in->tile_count == 1);

        if (s->f == NULL) {
                struct video_desc desc = video_desc_from_frame(in);
                desc.tile_count = 2;
                desc.fps /= 2;
                s->f = vf_alloc_desc(desc);
                s->f->tiles[0].data = malloc(s->f->tiles[0].data_len);
                memcpy(s->f->tiles[0].data, in->tiles[0].data, in->tiles[0].data_len);
                VIDEO_FRAME_DISPOSE(in);
                return NULL;
        }

        struct video_frame *f      = s->f;
        s->f->tiles[1].data = in->tiles[0].data;
        f->callbacks.dispose_udata = in;
        f->callbacks.dispose       = dispose_frame;
        s->f                       = NULL;

        return f;
}

static const struct capture_filter_info capture_filter_temporal_3d = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(temporal_3d, &capture_filter_temporal_3d, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

