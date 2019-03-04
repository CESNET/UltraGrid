/**
 * @file   capture_filter/split.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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

#include "capture_filter.h"

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/vf_split.h"
#include "video.h"
#include "video_codec.h"

#define MAX_TILES 16

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_split {
	int x, y;
};

static void usage() {
        printf("Splits frame to XxY tiles:\n\n");
        printf("split usage:\n");
        color_out(COLOR_OUT_BOLD, "\tsplit:x:y\n\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        int x, y;

        if (!cfg || strchr(cfg, ':') == NULL) {
                usage();
                return -1;
        }
        if (strcasecmp(cfg, "help") == 0) {
                usage();
                return 1;
        }

        x = atoi(cfg);
        y = atoi(strchr(cfg, ':') + 1);
        assert(x > 0 && y > 0);


        struct state_split *s = calloc(1, sizeof(struct state_split));
        s->x = x;
        s->y = y;

        *state = s;
        return 0;
}

static void done(void *state)
{
        free(state);
}

static void dispose_frame(struct video_frame *f) {
        VIDEO_FRAME_DISPOSE((struct video_frame *) f->callbacks.dispose_udata);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_split *s = state;

        struct video_desc desc = video_desc_from_frame(in);
        desc.tile_count = s->x * s->y;
        desc.width /= s->x;
        desc.height /= s->y;
        struct video_frame *out = vf_alloc_desc_data(desc);
        vf_split(out, in, s->x, s->y, 0);
        out->callbacks.dispose = vf_free;

        VIDEO_FRAME_DISPOSE(in);
        return out;
}

static const struct capture_filter_info capture_filter_split = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(split, &capture_filter_split, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

