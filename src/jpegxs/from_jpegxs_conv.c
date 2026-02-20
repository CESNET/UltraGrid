/**
 * @file   jpegxs/from_jpegxs_conv.c
 * @author Jan Frejlach     <536577@mail.muni.cz>
 */
/*
 * Copyright (c) 2026 CESNET
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

#include "jpegxs_conv.h"

#include <stdint.h>                // for uint16_t, uint8_t, uint32_t
#include <svt-jpegxs/SvtJpegxs.h>
#include <string.h>

#include "pixfmt_conv.h"           // for rgbp12le_to_r12l
#include "types.h"
#include "video_codec.h"

static void yuv422p_to_uyvy(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; y++) {
                uint8_t *src_y = (uint8_t *) src->data_yuv[0] + y * src->stride[0];
                uint8_t *src_u = (uint8_t *) src->data_yuv[1] + y * src->stride[1];
                uint8_t *src_v = (uint8_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; x += 2) {
                        *dst++ = *src_u++;
                        *dst++ = *src_y++;
                        *dst++ = *src_v++;
                        *dst++ = *src_y++;
                }
        }
}

static void yuv422p_to_yuyv(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; y++) {
                uint8_t *src_y = (uint8_t *) src->data_yuv[0] + y * src->stride[0];
                uint8_t *src_u = (uint8_t *) src->data_yuv[1] + y * src->stride[1];
                uint8_t *src_v = (uint8_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; x += 2) {
                        *dst++ = *src_y++;
                        *dst++ = *src_u++;
                        *dst++ = *src_y++;
                        *dst++ = *src_v++;
                }
        }
}

static void yuv420p_to_i420(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        const int y_size = width * height;
        const int uv_size = (width / 2) * (height / 2);

        uint8_t *dst_y = dst;
        uint8_t *dst_u = dst_y + y_size;
        uint8_t *dst_v = dst_u + uv_size;

        memcpy(dst_y, src->data_yuv[0], y_size);
        memcpy(dst_u, src->data_yuv[1], uv_size);
        memcpy(dst_v, src->data_yuv[2], uv_size);
}

static void rgbp_to_rgb(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint8_t *src_r = (uint8_t *) src->data_yuv[0] + y * src->stride[0];
                uint8_t *src_g = (uint8_t *) src->data_yuv[1] + y * src->stride[1];
                uint8_t *src_b = (uint8_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; ++x) {
                        *dst++ = *src_r++;
                        *dst++ = *src_g++;
                        *dst++ = *src_b++;
                }
        }
}

static void yuv422p10le_to_v210(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint32_t *dst_row = (uint32_t *)(dst + y * vc_get_linesize(width, v210));
                uint16_t *src_y = (uint16_t *) src->data_yuv[0] + y * src->stride[0];
                uint16_t *src_u = (uint16_t *) src->data_yuv[1] + y * src->stride[1];
                uint16_t *src_v = (uint16_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; x += 6) {
                        uint16_t y0 = *src_y++;
                        uint16_t y1 = *src_y++;
                        uint16_t y2 = *src_y++;
                        uint16_t y3 = *src_y++;
                        uint16_t y4 = *src_y++;
                        uint16_t y5 = *src_y++;

                        uint16_t u0 = *src_u++;
                        uint16_t u2 = *src_u++;
                        uint16_t u4 = *src_u++;

                        uint16_t v0 = *src_v++;
                        uint16_t v2 = *src_v++;
                        uint16_t v4 = *src_v++;

                        uint32_t w0 = ((v0 & 0x3FF) << 20) | ((y0 & 0x3FF) << 10) | (u0 & 0x3FF);
                        uint32_t w1 = ((y2 & 0x3FF) << 20) | ((u2 & 0x3FF) << 10) | (y1 & 0x3FF);
                        uint32_t w2 = ((u4 & 0x3FF) << 20) | ((y3 & 0x3FF) << 10) | (v2 & 0x3FF);
                        uint32_t w3 = ((y5 & 0x3FF) << 20) | ((v4 & 0x3FF) << 10) | (y4 & 0x3FF);
                        
                        *dst_row++ = w0;
                        *dst_row++ = w1;
                        *dst_row++ = w2;
                        *dst_row++ = w3;
                }
        }
}

static void rgbp10le_to_r10k(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint32_t *dst_row = (uint32_t *)(dst + y * vc_get_linesize(width, R10k));
                uint16_t *src_r = (uint16_t *) src->data_yuv[0] + y * src->stride[0];
                uint16_t *src_g = (uint16_t *) src->data_yuv[1] + y * src->stride[1];
                uint16_t *src_b = (uint16_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; ++x) {
                        uint16_t r = *src_r++;
                        uint16_t g = *src_g++;
                        uint16_t b = *src_b++;
                        uint32_t w = ((r & 0x3FF) << 22) | ((g & 0x3FF) << 12) | ((b & 0x3FF) << 2);
                        *dst_row++ = __builtin_bswap32(w);
                }
        }
}

static void jxs_rgbp12le_to_r12l(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        unsigned char const *in_data[3] = { src->data_yuv[0], src->data_yuv[1],
                                            src->data_yuv[2] };
        const int in_linesize[3] = { (int) (src->stride[0] * sizeof(uint16_t)),
                                     (int) (src->stride[1] * sizeof(uint16_t)),
                                     (int) (src->stride[2] *
                                            sizeof(uint16_t)) };
        rgbp12le_to_r12l(dst, vc_get_linesize(width, R12L), in_data,
                         in_linesize, width, height);
}

static void
rgbp12le_to_rg48(const svt_jpeg_xs_image_buffer_t *src, int width, int height,
                 uint8_t *dst)
{
        for (int y = 0; y < height; ++y) {
                uint16_t *dst_row = (uint16_t *)(dst + y * vc_get_linesize(width, RG48));
                uint16_t *src_r = (uint16_t *) src->data_yuv[0] + y * src->stride[0];
                uint16_t *src_g = (uint16_t *) src->data_yuv[1] + y * src->stride[1];
                uint16_t *src_b = (uint16_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; ++x) {
                        *dst_row++ = *src_r++ << 4;
                        *dst_row++ = *src_g++ << 4;
                        *dst_row++ = *src_b++ << 4;
                }
        }
}

static const struct jpegxs_to_uv_conversion jpegxs_to_uv_conversions[] = {
        { COLOUR_FORMAT_PLANAR_YUV422, UYVY, yuv422p_to_uyvy },
        { COLOUR_FORMAT_PLANAR_YUV422, YUYV, yuv422p_to_yuyv },
        { COLOUR_FORMAT_PLANAR_YUV420, I420, yuv420p_to_i420 },
        { COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, RGB, rgbp_to_rgb },
        { COLOUR_FORMAT_PLANAR_YUV422, v210, yuv422p10le_to_v210 },
        { COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, R10k, rgbp10le_to_r10k },
        { COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, R12L, jxs_rgbp12le_to_r12l},
        { COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, RG48, rgbp12le_to_rg48},
        { COLOUR_FORMAT_INVALID, VIDEO_CODEC_NONE, NULL }
};

const struct jpegxs_to_uv_conversion *get_jpegxs_to_uv_conversion(codec_t codec) {

        const struct jpegxs_to_uv_conversion *conv = jpegxs_to_uv_conversions;
        while (conv->dst != VIDEO_CODEC_NONE) {
                if (conv->dst == codec)
                        return conv;
                conv++;
        }

        return NULL;
}
