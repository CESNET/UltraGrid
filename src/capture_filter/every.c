/**
 * @file   capture_filter/every.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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

#include <stdio.h>            // for printf
#include <stdlib.h>           // for atoi, calloc, free
#include <string.h>           // for strchr, memcpy, strlen
#include <strings.h>          // for strcasecmp

#include "capture_filter.h"

#include "debug.h"
#include "compat/c23.h"       // IWYU pragma: keep
#include "lib_common.h"
#include "types.h"            // for video_frame, video_frame_callbacks, tile
#include "utils/color_out.h"
#include "video_frame.h"      // for VIDEO_FRAME_DISPOSE, vf_alloc_desc, vf_...
#include "vo_postprocess/capture_filter_wrapper.h"

#define MOD_NAME "[cf/every] "

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_every {
        int num;
        int denom;
        int current;
        char *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper
                                ///< (otherwise unused)
};

static void usage() {
        color_printf("Passes only every n-th frame.\n\n");

        color_printf("See also captures filter:\n");
        color_printf(
            "\t" TBOLD("ratelimit")
            " - limits to given FPS (timing) but without dropping frames\n");
        color_printf("\t" TBOLD("add_frame")
                     " - frame stuffing (eg. for 50p->60p conversion)\n\n");

        color_printf(TBOLD("every") " usage:\n");
        printf("\tevery:numerator[/denominator]\n\n");

        color_printf("Examples:\n");
        color_printf("\t" TBOLD("every:2")
                     " - every second frame will be dropped\n");
        color_printf("\t" TBOLD("every:0")
                     "- The special case  can be used to discard all frames\n");
        color_printf("\t" TBOLD("every:6/5")
                     " - drops every 6th frame - effectively can be used to "
                     "convert 60p->50p\n\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        int n;
        int denom = 1;;
        if (strlen(cfg) == 0) {
                usage();
                return -1;
        }
        if(strcasecmp(cfg, "help") == 0) {
                usage();
                return 1;
        }
        n = atoi(cfg);
        if(strchr(cfg, '/')) {
                denom = atoi(strchr(cfg, '/') + 1);
        }
        if (denom > n && n != 0) {
                MSG(ERROR,
                    "Numerator has to be greater or equal to denominator.\n");
                return -1;
        }

        struct state_every *s = calloc(1, sizeof(struct state_every));
        s->num = n;
        s->denom = denom;

        s->current = -1;

        *state = s;
        return 0;
}

static void done(void *state)
{
        free(state);
}

static void dispose_frame(struct video_frame *f) {
        VIDEO_FRAME_DISPOSE((struct video_frame *) f->callbacks.dispose_udata);
        vf_free(f);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        if (in == nullptr) {
                return nullptr;
        }
        struct state_every *s = state;

        if (s->num == 0) {
                VIDEO_FRAME_DISPOSE(in);
                return NULL;
        }

        s->current = (s->current + 1) % s->num;

        if (s->current >= s->denom) {
                VIDEO_FRAME_DISPOSE(in);
                return NULL;
        }

        struct video_frame *frame = vf_alloc_desc(video_desc_from_frame(in));
        if (s->vo_pp_out_buffer) {
                frame->tiles[0].data = s->vo_pp_out_buffer;
                // copy data
                memcpy(frame->tiles[0].data, in->tiles[0].data,
                       in->tiles[0].data_len);
        } else { // do not copy data
                memcpy(frame->tiles, in->tiles, in->tile_count * sizeof(struct tile));
                frame->fps /= (double) s->num / s->denom;

                frame->callbacks.dispose = dispose_frame;
                frame->callbacks.dispose_udata = in;
        }

        return frame;
}

// for ADD_VO_PP_CAPTURE_FILTER_WRAPP
static void
vo_pp_set_out_buffer(void *state, char *buffer)
{
        struct state_every *s = state;
        s->vo_pp_out_buffer   = buffer;
}

static const struct capture_filter_info capture_filter_every = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(every, &capture_filter_every, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(every, init, filter, done,
                                 vo_pp_set_out_buffer, nullptr);
