/*
 * FILE:    capture_filter/every.c
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

#define MAX_TILES 16

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_every {
        int num;
        int denom;
        int current;
        struct video_frame *frame;
};

static void usage() {
        printf("Passes only every n-th frame:\n\n");
        printf("every usage:\n");
        printf("\tevery:numerator[/denominator]\n\n");
        printf("Example: every:2 - every second frame will be dropped\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        int n;
        int denom = 1;;
        if(cfg) {
                if(strcasecmp(cfg, "help") == 0) {
                        usage();
                        return 1;
                }
                n = atoi(cfg);
                if(strchr(cfg, '/')) {
                        denom = atoi(strchr(cfg, '/') + 1);
                }
                if (denom > n) {
                        log_msg(LOG_LEVEL_ERROR, "Currently, numerator has to be greater "
                               "(or equal, which, however, has a little use) then denominator.\n");
                        return -1;
                }
        } else {
                usage();
                return -1;
        }

        struct state_every *s = calloc(1, sizeof(struct state_every));
        s->num = n;
        s->denom = denom;
        s->frame = vf_alloc(MAX_TILES);

        s->current = -1;

        *state = s;
        return 0;
}

static void done(void *state)
{
        struct state_every *s = state;

        s->frame->data_deleter = NULL;
        vf_free(s->frame);
        free(state);
}

static void dispose_frame(struct video_frame *f) {
        VIDEO_FRAME_DISPOSE((struct video_frame *) f->dispose_udata);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_every *s = state;

        assert(in->tile_count <= MAX_TILES);
        struct tile *tiles = s->frame->tiles;
        memcpy(s->frame, in, sizeof(struct video_frame));
        s->frame->tiles = tiles;
        memcpy(s->frame->tiles, in->tiles, in->tile_count * sizeof(struct tile));
        s->frame->fps /= (double) s->num / s->denom;

        s->current = (s->current + 1) % s->num;

        s->frame->dispose = dispose_frame;
        s->frame->dispose_udata = in;

        if (s->current < s->denom) {
                return s->frame;
        } else {
                VIDEO_FRAME_DISPOSE(in);
                return NULL;
        }
}

static const struct capture_filter_info capture_filter_every = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(every, &capture_filter_every, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

