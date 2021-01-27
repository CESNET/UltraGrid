/**
 * @file   capture_filter/stripe.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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

#include "video.h"
#include "video_codec.h"

#define FACTOR 8

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

static int state_stripe;

static void usage() {
        printf("Strip frame by 8\n\n");
        printf("Usage:\n");
        printf("\tstripe\n\n");
        printf("Example: stripe - divide width by 8, multiply height by 8\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (cfg && strlen(cfg) > 0) {
                usage();
                return 1;
        }

        *state = &state_stripe;
        return 0;
}

static void done(void *state)
{
        assert(state == &state_stripe);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        assert(state == &state_stripe);
        struct video_desc desc = video_desc_from_frame(in);
        desc.width /= FACTOR;
        desc.height *= FACTOR;
        struct video_frame *out = vf_alloc_desc_data(desc);

        out->callbacks.dispose = vf_free;

        if (in->color_spec == I420) {
                int in_linesize = in->tiles[0].width;
                int out_linesize = out->tiles[0].width;
                for (unsigned int y = 0; y < in->tiles[0].height; ++y) {
                        for (unsigned int i = 0; i < FACTOR; ++i) {
                                memcpy(out->tiles[0].data + y * out_linesize + i * out_linesize * in->tiles[0].height, in->tiles[0].data + y * in_linesize + i * out_linesize, out_linesize);
                        }
                }
                in_linesize /= 2;
                out_linesize /= 2;
                for (unsigned int y = 0; y < in->tiles[0].height / 2; ++y) {
                        // u
                        for (unsigned int i = 0; i < FACTOR; ++i) {
                                const int in_off = in->tiles[0].width * in->tiles[0].height;
                                const int out_off = out->tiles[0].width * out->tiles[0].height;
                                memcpy(out->tiles[0].data + out_off + y * out_linesize + i * out_linesize * in->tiles[0].height / 2, in->tiles[0].data + in_off + y * in_linesize + i * out_linesize, out_linesize);
                        }
                        // v
                        for (unsigned int i = 0; i < FACTOR; ++i) {
                                const int in_off = in->tiles[0].width * in->tiles[0].height / 4 * 5;
                                const int out_off = out->tiles[0].width * out->tiles[0].height / 4 * 5;
                                memcpy(out->tiles[0].data + out_off + y * out_linesize + i * out_linesize * in->tiles[0].height / 2, in->tiles[0].data + in_off + y * in_linesize + i * out_linesize, out_linesize);
                        }
                }
        } else {
                int in_linesize = vc_get_linesize(in->tiles[0].width, out->color_spec);
                int out_linesize = vc_get_linesize(out->tiles[0].width, out->color_spec);
                for (unsigned int y = 0; y < in->tiles[0].height; ++y) {
                        for (unsigned int i = 0; i < FACTOR; ++i) {
                                memcpy(out->tiles[0].data + y * out_linesize + i * out_linesize * in->tiles[0].height, in->tiles[0].data + y * in_linesize + i * out_linesize, out_linesize);
                        }
                }
        }

        vf_copy_metadata(out, in);

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static const struct capture_filter_info capture_filter_stripe = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(stripe, &capture_filter_stripe, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

