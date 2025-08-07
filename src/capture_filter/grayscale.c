/**
 * @file   capture_filter/grayscale.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2025 CESNET
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

#include <string.h>                                 // for strcmp, strlen, NULL
#include <stdlib.h>                                 // for calloc, free, malloc

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "types.h"                                  // for tile, video_frame
#include "utils/color_out.h"
#include "video_frame.h"                            // for vf_alloc_desc
#include "vo_postprocess/capture_filter_wrapper.h"
struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_grayscale {
        char *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper (otherwise unused)
};

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        if (strlen(cfg) > 0) {
                color_printf(TRED(TBOLD("grayscale")) " converts image to grayscale, takes no arguments\n");
                return strcmp(cfg, "help") == 0 ? 1 : -1;
        }
        *state = calloc(1, sizeof(struct state_grayscale));
        return 0;
}

static void done(void *state)
{
        free(state);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_grayscale *s = state;

        if (in->color_spec != UYVY) {
                log_msg(LOG_LEVEL_WARNING, "Cannot create grayscale from other codec than UYVY!\n");
                return in;
        }
        struct video_frame *out = vf_alloc_desc(video_desc_from_frame(in));
        if (s->vo_pp_out_buffer) {
                out->tiles[0].data = s->vo_pp_out_buffer;
        } else {
                out->tiles[0].data = (char *) malloc(out->tiles[0].data_len);
                out->callbacks.data_deleter = vf_data_deleter;
        }
        out->callbacks.dispose = vf_free;

        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

        for (unsigned int i = 0; i < in->tiles[0].width * in->tiles[0].height; ++i) {
                *out_data++ = 127;
                in_data++;
                *out_data++ = *in_data++;
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static void vo_pp_set_out_buffer(void *state, char *buffer)
{
        struct state_grayscale *s = state;
        s->vo_pp_out_buffer = buffer;
}


static const struct capture_filter_info capture_filter_grayscale = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(grayscale, &capture_filter_grayscale, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
// coverity[leaked_storage:SUPPRESS]
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(grayscale, init, filter, done, vo_pp_set_out_buffer, NULL)

