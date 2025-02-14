/**
 * @file   testcard_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020-2024 CESNET
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

#include "video_capture/testcard_common.h"

#include <stdio.h>            // for printf
#include <stdlib.h>           // for NULL, free, abort, malloc

#include "debug.h"            // for LOG_LEVEL_FATAL, MSG
#include "pixfmt_conv.h"
#include "utils/color_out.h"
#include "video_codec.h"      // for get_codec_name, vc_get_size, get_bits_p...

#define MOD_NAME "[testcard_common] "

const uint32_t rect_colors[] = {
        0xff0000ffLU,
        0xff00ff00LU,
        0xffff0000LU,
        0xff00ffffLU,
        0xffffff00LU,
        0xffff00ffLU
};

void testcard_fillRect(struct testcard_pixmap *s, struct testcard_rect *r, uint32_t color)
{
        uint32_t *data = s->data;

        for (int cur_x = r->x; cur_x < r->x + r->w; ++cur_x) {
                for (int cur_y = r->y; cur_y < r->y + r->h; ++cur_y) {
                        if (cur_x < s->w) {
                                *(data + (long) s->w * cur_y + cur_x) = color;
                        }
                }
        }
}

/**
 * @param[in] in buffer in UYVY
 * @retval       buffer in I420
 */
static void
toI420(unsigned char *out, const unsigned char *input, int width, int height)
{
        const size_t y_h = height;
        const size_t chr_h = (y_h + 1) / 2;
        int          out_linesize[3] = { width,
                                         (width + 1) / 2,
                                         (width + 1) / 2 };
        unsigned char *out_data[3] = { out,
                                       out + (y_h * out_linesize[0]),
                                       out + (y_h * out_linesize[0]) +
                                           (chr_h * out_linesize[1]) };
        uyvy_to_i420(out_data, out_linesize, input, width, height);
}

void testcard_convert_buffer(codec_t in_c, codec_t out_c, unsigned char *out, unsigned const char *in, int width, int height)
{
        unsigned char *tmp_buffer = NULL;
        if (out_c == I420 || out_c == YUYV || (in_c == RGBA && out_c == v210)) {
                decoder_t decoder = get_decoder_from_to(in_c, UYVY);
                tmp_buffer =  malloc(2L * ((width + 1U) ^ 1U) * height);
                long in_linesize = vc_get_size(width, in_c);
                long out_linesize = vc_get_size(width, UYVY);
                for (int i = 0; i < height; ++i) {
                        decoder(tmp_buffer + i * out_linesize, in + i * in_linesize, vc_get_size(width, UYVY), DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
                }
                in = tmp_buffer;
                in_c = UYVY;
        }
        if (out_c == I420) {
                toI420(out, in, width, height);
                free(tmp_buffer);
                return;
        }
        decoder_t decoder = get_decoder_from_to(in_c, out_c);
        if (decoder == NULL) {
                MSG(FATAL, "No decoder from %s to %s!\n", get_codec_name(in_c),
                    get_codec_name(out_c));
                abort();
        }
        long out_linesize = vc_get_linesize(width, out_c);
        long in_linesize = vc_get_linesize(width, in_c);
        for (int i = 0; i < height; ++i) {
                decoder(out + i * out_linesize, in + i * in_linesize, vc_get_size(width, out_c), DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
        }
        free(tmp_buffer);
}

static bool testcard_conv_handled_internally(codec_t c)
{
        return c == I420 || c == YUYV || c == v210;
}

void testcard_show_codec_help(const char *name, bool src_8b_only)
{
        bool print_i420 = !src_8b_only; // testcard2 cannot handle planar format, anyway
        printf("Supported codecs (%s):\n", name);

        color_printf(TERM_FG_RED "\t8 bits\n" TERM_RESET);
        for (codec_t c = VIDEO_CODEC_FIRST; c != VIDEO_CODEC_COUNT; c = (int) c + 1) {
                if (is_codec_opaque(c) || get_bits_per_component(c) != 8 || (get_decoder_from_to(RGBA, c) == VIDEO_CODEC_NONE
                                        && !testcard_conv_handled_internally(c))) {
                        continue;
                }
                if (c == I420 && !print_i420) {
                        continue;
                }
                color_printf(TERM_BOLD "\t\t%-4s" TERM_RESET " - %s\n", get_codec_name(c), get_codec_name_long(c));
        }

        color_printf(TERM_FG_RED "\t10+ bits\n" TERM_RESET);
        for (codec_t c = VIDEO_CODEC_FIRST; c != VIDEO_CODEC_COUNT; c = (int) c + 1) {
                if (is_codec_opaque(c) || get_bits_per_component(c) == 8) {
                        continue;
                }
                if (get_decoder_from_to(RGBA, c) == VIDEO_CODEC_NONE &&
                                ((src_8b_only && c != v210) || get_decoder_from_to(RG48, c) == VIDEO_CODEC_NONE)) {
                        continue;
                }
                color_printf(TERM_BOLD "\t\t%-4s" TERM_RESET " - %s\n", get_codec_name(c), get_codec_name_long(c));
        }
}

bool testcard_has_conversion(codec_t c)
{
        return get_decoder_from_to(RG48, c) != NULL ||
                get_decoder_from_to(RGBA, c) != NULL ||
                testcard_conv_handled_internally(c);
}

