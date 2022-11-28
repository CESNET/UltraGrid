/**
 * @file   capture_filter/color.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET, z. s. p. o.
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
#include "vo_postprocess/capture_filter_wrapper.h"

#define MOD_NAME "[color] "

struct state_capture_filter_color {
        char *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper (otherwise unused)
};

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        cfg = cfg ? cfg : ""; // pp passes NULL for empty config
        if (strcmp(cfg, "help") == 0) {
                color_printf("\nFilter " TERM_FG_RED TERM_BOLD "color" TERM_RESET " computees average color of the picture.\n");
                color_printf(TERM_FG_YELLOW "Note: currenty only center pixel is printed!\n" TERM_FG_RESET);
                return 1;
        }

        *state = calloc(1, sizeof(struct state_capture_filter_color));
        return 0;
}

static void done(void *state)
{
        free(state);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_capture_filter_color *s = state;
        struct video_desc desc = video_desc_from_frame(in);
        struct video_frame *out = NULL;
        if (s->vo_pp_out_buffer) {
                out = vf_alloc_desc(desc);
                out->tiles[0].data = s->vo_pp_out_buffer;
                out->callbacks.dispose = vf_free;
                memcpy(out->tiles[0].data, in->tiles[0].data, in->tiles[0].data_len);
        } else {
                out = in;
        }

        int pix_blk_size = get_pf_block_bytes(in->color_spec);
        int linesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
        unsigned char *block_in = (unsigned char *) in->tiles[0].data + in->tiles[0].height / 2 * linesize + ((linesize / 2) / pix_blk_size * pix_blk_size);
        unsigned char uyvy[4 + MAX_PADDING];

        if (s->vo_pp_out_buffer) {
                VIDEO_FRAME_DISPOSE(in);
        }

        decoder_t dec = get_decoder_from_to(in->color_spec, UYVY);
        if (!dec) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot get decoder from %s to UYVY!\n", get_codec_name(in->color_spec));
                return out;
        }
        dec(uyvy, block_in, 4, DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
        log_msg(LOG_LEVEL_INFO, "[color %s] Center color is Y=%hhu U=%hhu V=%hhu\n", s->vo_pp_out_buffer ? "pp" : "cap. f.", uyvy[1], uyvy[0], uyvy[2]);

        return out;
}

static void vo_pp_set_out_buffer(void *state, char *buffer)
{
        struct state_capture_filter_color *s = state;
        s->vo_pp_out_buffer = buffer;
}

static const struct capture_filter_info capture_filter_color = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(color, &capture_filter_color, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(color, init, filter, done, vo_pp_set_out_buffer)

/* vim: set expandtab sw=8: */
