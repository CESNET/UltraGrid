/**
 * @file   testcard_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020-2022 CESNET
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
#endif // HAVE_CONFIG_H

#include "video.h"
#include "video_capture/testcard_common.h"

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
 * @retval       buffer in I420 (must be deallocated by the caller)
 * @note
 * Caller must deallocate returned buffer
 */
static void toI420(unsigned char *out, const unsigned char *input, int width, int height)
{
        const unsigned char *in = (const unsigned char *) input;
        int w_ch = (width + 1) / 2;
        int h_ch = (height + 1) / 2;
        unsigned char *y = out;
        unsigned char *u0 = out + width * height;
        unsigned char *v0 = out + width * height + w_ch * h_ch;
        unsigned char *u1 = u0, *v1 = v0;

        for (int i = 0; i < height; i += 1) {
                for (int j = 0; j < ((width + 1) & ~1); j += 2) {
                        // U
                        if (i % 2 == 0) {
                                *u0++ = *in++;
                        } else { // average with every 2nd row
                                *u1 = (*u1 + *in++) / 2;
                                u1++;
                        }
                        // Y
                        *y++ = *in++;
                        // V
                        if (i % 2 == 0) {
                                *v0++ = *in++;
                        } else { // average with every 2nd row
                                *v1 = (*v1 + *in++) / 2;
                                v1++;
                        }
                        // Y
                        if (j + 1 == width) {
                                in++;
                        } else {
                                *y++ = *in++;
                        }
                }
        }
}

void testcard_convert_buffer(codec_t in_c, codec_t out_c, unsigned char *out, unsigned const char *in, int width, int height)
{
        unsigned char *tmp_buffer = NULL;
        if (out_c == I420 || out_c == YUYV || (in_c == RGBA || out_c == v210)) {
                decoder_t decoder = get_decoder_from_to(in_c, UYVY);
                tmp_buffer =  malloc(2L * ((width + 1U) ^ 1U) * height);
                long in_linesize = vc_get_linesize(width, in_c);
                long out_linesize = vc_get_linesize(width, UYVY);
                for (int i = 0; i < height; ++i) {
                        decoder(tmp_buffer + i * out_linesize, in + i * in_linesize, out_linesize, DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
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
        assert(decoder != NULL);
        long out_linesize = vc_get_linesize(width, out_c);
        long in_linesize = vc_get_linesize(width, in_c);
        for (int i = 0; i < height; ++i) {
                decoder(out + i * out_linesize, in + i * in_linesize, vc_get_linesize(width, out_c), DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
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

        printf("\t8 bits\n");
        for (codec_t c = VIDEO_CODEC_FIRST; c != VIDEO_CODEC_COUNT; c = (int) c + 1) {
                if (is_codec_opaque(c) || get_bits_per_component(c) != 8 || (get_decoder_from_to(RGBA, c) == VIDEO_CODEC_NONE
                                        && !testcard_conv_handled_internally(c))) {
                        continue;
                }
                if (c == I420 && !print_i420) {
                        continue;
                }
                printf("\t\t'%s' - %s\n", get_codec_name(c), get_codec_name_long(c));
        }

        printf("\t10+ bits\n");
        for (codec_t c = VIDEO_CODEC_FIRST; c != VIDEO_CODEC_COUNT; c = (int) c + 1) {
                if (is_codec_opaque(c) || get_bits_per_component(c) == 8) {
                        continue;
                }
                if (get_decoder_from_to(RGBA, c) == VIDEO_CODEC_NONE &&
                                ((src_8b_only && c != v210) || get_decoder_from_to(RG48, c) == VIDEO_CODEC_NONE)) {
                        continue;
                }
                printf("\t\t'%s' - %s\n", get_codec_name(c), get_codec_name_long(c));
        }
}

bool testcard_has_conversion(codec_t c)
{
        return get_decoder_from_to(RG48, c) != NULL ||
                get_decoder_from_to(RGBA, c) != NULL ||
                testcard_conv_handled_internally(c);
}

