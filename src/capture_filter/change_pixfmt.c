/**
 * @file   capture_filter/change_pixfmt.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020 CESNET, z. s. p. o.
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

#define MOD_NAME "[change pixfmt cap. f.] "

struct state_capture_filter_change_pixfmt {
        codec_t to_codec;
};

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (!cfg || strcmp(cfg, "help") == 0) {
                printf("Performs pixel format change transformation.\n\n"
                       "usage:\n");
                color_out(COLOR_OUT_BOLD, "\t--capture-filter change_pixfmt:<name>\n");
                return 1;
        }

        struct state_capture_filter_change_pixfmt *s = calloc(1, sizeof(struct state_capture_filter_change_pixfmt));
        s->to_codec = get_codec_from_name(cfg);
        if (!s->to_codec) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong codec: %s\n", cfg);
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

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_capture_filter_change_pixfmt *s = state;
        struct video_desc desc = video_desc_from_frame(in);
        desc.color_spec = s->to_codec;
        decoder_t decoder = get_decoder_from_to(in->color_spec, s->to_codec, true);

        if (!decoder) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to find decoder!\n");
                return NULL;
        }
        struct video_frame *out = vf_alloc_desc_data(desc);
        out->callbacks.dispose = vf_free;

        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
        unsigned char *out_data = (unsigned char *) out->tiles[0].data;
        int src_linesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
        int dst_linesize = vc_get_linesize(in->tiles[0].width, s->to_codec);

        for (unsigned int i = 0; i < in->tiles[0].height; i += 1) {
                decoder(out_data, in_data, dst_linesize, 0, 8, 16);
                in_data += src_linesize;
                out_data += dst_linesize;
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static const struct capture_filter_info capture_filter_change_pixfmt = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(change_pixfmt, &capture_filter_change_pixfmt, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

/* vim: set expandtab sw=8: */
