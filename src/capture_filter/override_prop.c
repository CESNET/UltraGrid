/**
 * @file   capture_filter/override_prop.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET, z. s. p. o.
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
#include "video.h"
#include "video_codec.h"

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_override_prop {
        struct video_desc new_desc;
};

static void usage() {
        color_printf(
            TBOLD("override_prop") " allows overriding video properties.\n\n");
        printf("usage:\n\t" TBOLD("-F override_prop:fps=<n>") "\n");
        printf("where:\n\t<fps> - new (metadata) FPS value ("
               "suffixed 'i' for interlaced)\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (strcmp(cfg, "help") == 0 || strstr(cfg, "fps=") != cfg) {
                usage();
                return strcmp(cfg, "help") == 0;
        }

        struct state_override_prop *s = calloc(1, sizeof *s);
        if (!parse_fps(strchr(cfg, '=') + 1, &s->new_desc)) {
                free(s);
                return -1;
        }

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
        struct state_override_prop *s = state;

        struct video_frame *out = vf_alloc_desc(video_desc_from_frame(in));
        memcpy(out->tiles, in->tiles, in->tile_count * sizeof(struct tile));
        out->fps = s->new_desc.fps;
        out->interlacing = s->new_desc.interlacing;

        out->callbacks.dispose = dispose_frame;
        out->callbacks.dispose_udata = in;

        return out;
}

static const struct capture_filter_info capture_filter_override_prop = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(override_prop, &capture_filter_override_prop, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

