/**
 * @file   capture_filter/ratelimit.c
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
#include "tv.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_codec.h"

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_ratelimit {
        time_ns_t next_frame_time;
        double fps;
};

static void usage() {
        color_printf("Filter " TBOLD("ratelimite") " limits frame rate of video stream.\n"
                     "This filter is related to a capture filter " TBOLD("every")".\n\n");
        printf("ratelimit usage:\n\n");
        color_printf(TBOLD("\t--capture-filter ratelimit:<fps>") " -t <capture>\n\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (strlen(cfg) == 0 || strcasecmp(cfg, "help") == 0) {
                usage();
                return strlen(cfg) == 0 ? -1 : 1;
        }

        struct state_ratelimit *s = calloc(1, sizeof *s);
        s->fps = strtod(cfg, NULL);
        s->next_frame_time = get_time_in_ns();
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
        struct state_ratelimit *s = state;

        time_ns_t t = get_time_in_ns();

        if (t < s->next_frame_time) {
                VIDEO_FRAME_DISPOSE(in);
                return NULL;
        }

        struct video_frame *frame = vf_alloc_desc(video_desc_from_frame(in));
        memcpy(frame->tiles, in->tiles, in->tile_count * sizeof(struct tile));
        frame->fps = s->fps;

        frame->callbacks.dispose = dispose_frame;
        frame->callbacks.dispose_udata = in;

        s->next_frame_time += NS_IN_SEC_DBL / s->fps;

        return frame;
}

static const struct capture_filter_info capture_filter_ratelimit = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(ratelimit, &capture_filter_ratelimit, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

