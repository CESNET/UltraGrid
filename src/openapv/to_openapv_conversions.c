/**
 * @file   openapv/to_openapv_conversions.c
 * @author Juraj Zemančík    <550535@mail.muni.cz>
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

#include "to_openapv_conversions.h"

#include "../color_space.h"
#include "../video_codec.h"

#include <stdint.h>

static void uyvy_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const int y_stride_px = dst->s[0] / (int) sizeof(uint16_t);
        const int u_stride_px = dst->s[1] / (int) sizeof(uint16_t);
        const int v_stride_px = dst->s[2] / (int) sizeof(uint16_t);

        for (int y = 0; y < height; ++y) {
                uint16_t *dst_y = (uint16_t *) dst->a[0] + y * y_stride_px;
                uint16_t *dst_u = (uint16_t *) dst->a[1] + y * u_stride_px;
                uint16_t *dst_v = (uint16_t *) dst->a[2] + y * v_stride_px;

                for (int x = 0; x < width; x += 2) {
                        *dst_u++ = *src++ << 2;
                        *dst_y++ = *src++ << 2;
                        *dst_v++ = *src++ << 2;
                        *dst_y++ = *src++ << 2;
                }
        }
}

static void yuyv_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const int y_stride_px = dst->s[0] / (int) sizeof(uint16_t);
        const int u_stride_px = dst->s[1] / (int) sizeof(uint16_t);
        const int v_stride_px = dst->s[2] / (int) sizeof(uint16_t);
        
        for (int y = 0; y < height; ++y) {
                uint16_t *dst_y = (uint16_t *) dst->a[0] + y * y_stride_px;
                uint16_t *dst_u = (uint16_t *) dst->a[1] + y * u_stride_px;
                uint16_t *dst_v = (uint16_t *) dst->a[2] + y * v_stride_px;

                for (int x = 0; x < width; x += 2) {
                        *dst_y++ = *src++ << 2;
                        *dst_u++ = *src++ << 2;
                        *dst_y++ = *src++ << 2;
                        *dst_v++ = *src++ << 2;
                }
        }
}

static void v210_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const int y_stride_px = dst->s[0] / (int) sizeof(uint16_t);
        const int u_stride_px = dst->s[1] / (int) sizeof(uint16_t);
        const int v_stride_px = dst->s[2] / (int) sizeof(uint16_t);

        for (int y = 0; y < height; ++y) {
                const uint32_t *src_row = (const uint32_t *)(src + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) dst->a[0] + y * y_stride_px;
                uint16_t *dst_u = (uint16_t *) dst->a[1] + y * u_stride_px;
                uint16_t *dst_v = (uint16_t *) dst->a[2] + y * v_stride_px;

                for (int x = 0; x < width; x += 6) {
                        uint32_t w0 = *src_row++;
                        uint32_t w1 = *src_row++;
                        uint32_t w2 = *src_row++;
                        uint32_t w3 = *src_row++;

                        *dst_y++ = (w0 >> 10) & 0x3ff;
                        *dst_y++ = w1 & 0x3ff;
                        *dst_y++ = (w1 >> 20) & 0x3ff;
                        *dst_y++ = (w2 >> 10) & 0x3ff;
                        *dst_y++ = w3 & 0x3ff;
                        *dst_y++ = (w3 >> 20) & 0x3ff;

                        *dst_u++ = w0 & 0x3ff;
                        *dst_u++ = (w1 >> 10) & 0x3ff;
                        *dst_u++ = (w2 >> 20) & 0x3ff;

                        *dst_v++ = (w0 >> 20) & 0x3ff;
                        *dst_v++ = w2 & 0x3ff;
                        *dst_v++ = (w3 >> 10) & 0x3ff;
                }
        } 
}

static void rgb_to_yuv444p10(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH10);
        const int y_sp = dst->s[0] / (int) sizeof(uint16_t);
        const int u_sp = dst->s[1] / (int) sizeof(uint16_t);
        const int v_sp = dst->s[2] / (int) sizeof(uint16_t);
        const int src_pitch = vc_get_linesize(width, RGB);

        for (int y = 0; y < height; ++y) {
                const uint8_t *s = src + y * src_pitch;
                uint16_t *dy = (uint16_t *) dst->a[0] + y * y_sp;
                uint16_t *du = (uint16_t *) dst->a[1] + y * u_sp;
                uint16_t *dv = (uint16_t *) dst->a[2] + y * v_sp;

                for (int x = 0; x < width; ++x) {
                        comp_type_t r = (comp_type_t) s[0] << 2;
                        comp_type_t g = (comp_type_t) s[1] << 2;
                        comp_type_t b = (comp_type_t) s[2] << 2;
                        s += 3;
                        *dy++ = (uint16_t) ((RGB_TO_Y (cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 4)));
                        *du++ = (uint16_t) ((RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 1)));
                        *dv++ = (uint16_t) ((RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 1)));
                }
        }
}

static void rgba_to_yuv4444p10(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH10);
        const int y_sp = dst->s[0] / (int) sizeof(uint16_t);
        const int u_sp = dst->s[1] / (int) sizeof(uint16_t);
        const int v_sp = dst->s[2] / (int) sizeof(uint16_t);
        const int a_sp = dst->s[3] / (int) sizeof(uint16_t);
        const int src_pitch = vc_get_linesize(width, RGBA);

        for (int y = 0; y < height; ++y) {
                const uint8_t *s = src + y * src_pitch;
                uint16_t *dy = (uint16_t *) dst->a[0] + y * y_sp;
                uint16_t *du = (uint16_t *) dst->a[1] + y * u_sp;
                uint16_t *dv = (uint16_t *) dst->a[2] + y * v_sp;
                uint16_t *da = (uint16_t *) dst->a[3] + y * a_sp;

                for (int x = 0; x < width; ++x) {
                        comp_type_t r = (comp_type_t) s[0] << 2;
                        comp_type_t g = (comp_type_t) s[1] << 2;
                        comp_type_t b = (comp_type_t) s[2] << 2;
                        uint16_t    a = (uint16_t)    s[3] << 2;
                        s += 4;
                        *dy++ = (uint16_t) ((RGB_TO_Y (cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 4)));
                        *du++ = (uint16_t) ((RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 1)));
                        *dv++ = (uint16_t) ((RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 1)));
                        *da++ = a;
                }
        }
}

static void r10k_to_yuv444p10(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH10);
        const int y_sp = dst->s[0] / (int) sizeof(uint16_t);
        const int u_sp = dst->s[1] / (int) sizeof(uint16_t);
        const int v_sp = dst->s[2] / (int) sizeof(uint16_t);
        const int src_pitch = vc_get_linesize(width, R10k);

        for (int y = 0; y < height; ++y) {
                const uint8_t *s = src + y * src_pitch;
                uint16_t *dy = (uint16_t *) dst->a[0] + y * y_sp;
                uint16_t *du = (uint16_t *) dst->a[1] + y * u_sp;
                uint16_t *dv = (uint16_t *) dst->a[2] + y * v_sp;

                for (int x = 0; x < width; ++x) {
                        comp_type_t r = ((comp_type_t) s[0] << 2) | (s[1] >> 6);
                        comp_type_t g = (((comp_type_t) s[1] & 0x3F) << 4) | (s[2] >> 4);
                        comp_type_t b = (((comp_type_t) s[2] & 0x0F) << 6) | (s[3] >> 2);
                        s += 4;
                        *dy++ = (uint16_t) ((RGB_TO_Y (cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 4)));
                        *du++ = (uint16_t) ((RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 1)));
                        *dv++ = (uint16_t) ((RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH10 - 1)));
                }
        }
}

static void r12l_to_yuv444p12(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH12);
        const int y_sp = dst->s[0] / (int) sizeof(uint16_t);
        const int u_sp = dst->s[1] / (int) sizeof(uint16_t);
        const int v_sp = dst->s[2] / (int) sizeof(uint16_t);
        const int src_pitch = vc_get_linesize(width, R12L);

        for (int y = 0; y < height; ++y) {
                const uint8_t *s = src + y * src_pitch;
                uint16_t *dy = (uint16_t *) dst->a[0] + y * y_sp;
                uint16_t *du = (uint16_t *) dst->a[1] + y * u_sp;
                uint16_t *dv = (uint16_t *) dst->a[2] + y * v_sp;

                int x = 0;
                while (x < width) {
                        comp_type_t rgb[8][3];
                        rgb[0][0] = ((comp_type_t) (s[1]  & 0x0F) << 8) |  s[0];
                        rgb[0][1] = ((comp_type_t)  s[2]         << 4) | (s[1]  >> 4);
                        rgb[0][2] = ((comp_type_t) (s[4]  & 0x0F) << 8) |  s[3];
                        rgb[1][0] = ((comp_type_t)  s[5]         << 4) | (s[4]  >> 4);
                        rgb[1][1] = ((comp_type_t) (s[7]  & 0x0F) << 8) |  s[6];
                        rgb[1][2] = ((comp_type_t)  s[8]         << 4) | (s[7]  >> 4);
                        rgb[2][0] = ((comp_type_t) (s[10] & 0x0F) << 8) |  s[9];
                        rgb[2][1] = ((comp_type_t)  s[11]        << 4) | (s[10] >> 4);
                        rgb[2][2] = ((comp_type_t) (s[13] & 0x0F) << 8) |  s[12];
                        rgb[3][0] = ((comp_type_t)  s[14]        << 4) | (s[13] >> 4);
                        rgb[3][1] = ((comp_type_t) (s[16] & 0x0F) << 8) |  s[15];
                        rgb[3][2] = ((comp_type_t)  s[17]        << 4) | (s[16] >> 4);
                        rgb[4][0] = ((comp_type_t) (s[19] & 0x0F) << 8) |  s[18];
                        rgb[4][1] = ((comp_type_t)  s[20]        << 4) | (s[19] >> 4);
                        rgb[4][2] = ((comp_type_t) (s[22] & 0x0F) << 8) |  s[21];
                        rgb[5][0] = ((comp_type_t)  s[23]        << 4) | (s[22] >> 4);
                        rgb[5][1] = ((comp_type_t) (s[25] & 0x0F) << 8) |  s[24];
                        rgb[5][2] = ((comp_type_t)  s[26]        << 4) | (s[25] >> 4);
                        rgb[6][0] = ((comp_type_t) (s[28] & 0x0F) << 8) |  s[27];
                        rgb[6][1] = ((comp_type_t)  s[29]        << 4) | (s[28] >> 4);
                        rgb[6][2] = ((comp_type_t) (s[31] & 0x0F) << 8) |  s[30];
                        rgb[7][0] = ((comp_type_t)  s[32]        << 4) | (s[31] >> 4);
                        rgb[7][1] = ((comp_type_t) (s[34] & 0x0F) << 8) |  s[33];
                        rgb[7][2] = ((comp_type_t)  s[35]        << 4) | (s[34] >> 4);
                        s += 36;

                        const int n = (width - x) < 8 ? (width - x) : 8;
                        for (int i = 0; i < n; ++i) {
                                comp_type_t r = rgb[i][0], g = rgb[i][1], b = rgb[i][2];
                                *dy++ = (uint16_t) ((RGB_TO_Y (cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH12 - 4)));
                                *du++ = (uint16_t) ((RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH12 - 1)));
                                *dv++ = (uint16_t) ((RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH12 - 1)));
                        }
                        x += 8;
                }
        }
}

static void vuya_to_yuv4444p10(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const int y_sp = dst->s[0] / (int) sizeof(uint16_t);
        const int u_sp = dst->s[1] / (int) sizeof(uint16_t);
        const int v_sp = dst->s[2] / (int) sizeof(uint16_t);
        const int a_sp = dst->s[3] / (int) sizeof(uint16_t);
        const int src_pitch = vc_get_linesize(width, VUYA);

        for (int y = 0; y < height; ++y) {
                const uint8_t *s = src + y * src_pitch;
                uint16_t *dy = (uint16_t *) dst->a[0] + y * y_sp;
                uint16_t *du = (uint16_t *) dst->a[1] + y * u_sp;
                uint16_t *dv = (uint16_t *) dst->a[2] + y * v_sp;
                uint16_t *da = (uint16_t *) dst->a[3] + y * a_sp;

                for (int x = 0; x < width; ++x) {
                        *dv++ = (uint16_t) (s[0] << 2);
                        *du++ = (uint16_t) (s[1] << 2);
                        *dy++ = (uint16_t) (s[2] << 2);
                        *da++ = (uint16_t) (s[3] << 2);
                        s += 4;
                }
        }
}

static const struct uv_to_openapv_conversion uv_to_openapv_conversions[] = {
        { UYVY, OAPV_CS_YCBCR422_10LE,  uyvy_to_yuv422p     },
        { YUYV, OAPV_CS_YCBCR422_10LE,  yuyv_to_yuv422p     },
        { v210, OAPV_CS_YCBCR422_10LE,  v210_to_yuv422p     },
        { RGB,  OAPV_CS_YCBCR444_10LE,  rgb_to_yuv444p10    },
        { RGBA, OAPV_CS_YCBCR4444_10LE, rgba_to_yuv4444p10  },
        { VUYA, OAPV_CS_YCBCR4444_10LE, vuya_to_yuv4444p10  },
        { R10k, OAPV_CS_YCBCR444_10LE,  r10k_to_yuv444p10   },
        { R12L, OAPV_CS_YCBCR444_12LE,  r12l_to_yuv444p12   },
        { VIDEO_CODEC_NONE, OAPV_CS_UNKNOWN, NULL }
};

const struct uv_to_openapv_conversion *get_uv_to_openapv_conversion(codec_t codec) {

        const struct uv_to_openapv_conversion *conv = uv_to_openapv_conversions;
        while (conv->src_color_format != VIDEO_CODEC_NONE) {
                if (conv->src_color_format == codec)
                        return conv;
                conv++;
        }

        return NULL;
}