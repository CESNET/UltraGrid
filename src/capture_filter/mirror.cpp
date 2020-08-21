/**
 * @file   capture_filter/mirror.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2020 CESNET, z. s. p. o.
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
#include "vo_postprocess/capture_filter_wrapper.h"

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_mirror {
        char *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper (otherwise unused)
};

static int init(struct module *, const char *, void **state)
{
        *state = static_cast<state_mirror *>(calloc(1, sizeof(state_mirror)));
        return 0;
}

static void done(void *state)
{
        free(state);
}

static void mirror_line_UYVY(unsigned char *dst, const unsigned char *src, int linesize)
{
        dst = dst + linesize;
        while (linesize > 3) {
                unsigned char u, y1, v, y2;
                u = *src++;
                y1 = *src++;
                v = *src++;
                y2 = *src++;

                *--dst = y1;
                *--dst = v;
                *--dst = y2;
                *--dst = u;

                linesize -= 4;
        }
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        auto *s = static_cast<state_mirror *>(state);

        if (in->color_spec != UYVY) {
                log_msg(LOG_LEVEL_WARNING, "Only supported colorspace for mirror is currently UYVY!\n");
                return in;
        }

        struct video_frame *out = vf_alloc_desc(video_desc_from_frame(in));
        if (s->vo_pp_out_buffer) {
                out->tiles[0].data = s->vo_pp_out_buffer;
        } else {
                out->tiles[0].data = static_cast<char *>(malloc(out->tiles[0].data_len));
                out->callbacks.data_deleter = vf_data_deleter;
        }
        out->callbacks.dispose = vf_free;

        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

        int linesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
        for (unsigned int y = 0; y < in->tiles[0].height; ++y) {
                mirror_line_UYVY(out_data + y * linesize, in_data + y * linesize, linesize);
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static void vo_pp_set_out_buffer(void *state, char *buffer)
{
        auto *s = static_cast<struct state_mirror *>(state);
        s->vo_pp_out_buffer = buffer;
}

static const struct capture_filter_info capture_filter_mirror = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(mirror, &capture_filter_mirror, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(mirror, init, filter, done, vo_pp_set_out_buffer)

