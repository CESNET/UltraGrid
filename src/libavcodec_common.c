/**
 * @file   libavcodec_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2013-2019 CESNET, z. s. p. o.
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

#include "hwaccel_vdpau.h"
#include "libavcodec_common.h"
#include "video.h"

#ifdef __SSE3__
#include "pmmintrin.h"
// compat with older Clang compiler
#ifndef _mm_bslli_si128
#define _mm_bslli_si128 _mm_slli_si128
#endif
#ifndef _mm_bsrli_si128
#define _mm_bsrli_si128 _mm_srli_si128
#endif
#endif

#undef max
#undef min
#define max(a, b)      (((a) > (b))? (a): (b))
#define min(a, b)      (((a) < (b))? (a): (b))

void uyvy_to_yuv420p(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                unsigned char *src = in_data + y * (width * 2);
                /*  every odd row */
                unsigned char *src2 = in_data + (y + 1) * (width * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_y2 = out_frame->data[0] + out_frame->linesize[0] * (y + 1);
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y / 2;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y / 2;
                for(int x = 0; x < width / 2; ++x) {
                        *dst_cb++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                }
        }
}

void uyvy_to_yuv422p(AVFrame *out_frame, unsigned char *src, int width, int height)
{
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;
                for(int x = 0; x < width; x += 2) {
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

void uyvy_to_yuv444p(AVFrame *out_frame, unsigned char *src, int width, int height)
{
        for(int y = 0; y < height; ++y) {
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;
                for(int x = 0; x < width; x += 2) {
                        *dst_cb++ = *src;
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

void uyvy_to_nv12(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                unsigned char *src = in_data + y * (width * 2);
                /*  every odd row */
                unsigned char *src2 = in_data + (y + 1) * (width * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_y2 = out_frame->data[0] + out_frame->linesize[0] * (y + 1);
                unsigned char *dst_cbcr = out_frame->data[1] + out_frame->linesize[1] * y / 2;

                int x = 0;
#ifdef __SSE3__
                __m128i yuv;
                __m128i yuv2;
                __m128i y1;
                __m128i y2;
                __m128i y3;
                __m128i y4;
                __m128i uv;
                __m128i uv2;
                __m128i uv3;
                __m128i uv4;
                __m128i ymask = _mm_set1_epi32(0xFF00FF00);
                __m128i dsty;
                __m128i dsty2;
                __m128i dstuv;

                for (; x < (width - 15); x += 16){
                        yuv = _mm_lddqu_si128((__m128i const*) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*) src2);
                        src += 16;
                        src2 += 16;

                        y1 = _mm_and_si128(ymask, yuv);
                        y1 = _mm_bsrli_si128(y1, 1);
                        y2 = _mm_and_si128(ymask, yuv2);
                        y2 = _mm_bsrli_si128(y2, 1);

                        uv = _mm_andnot_si128(ymask, yuv);
                        uv2 = _mm_andnot_si128(ymask, yuv2);

                        uv = _mm_avg_epu8(uv, uv2);

                        yuv = _mm_lddqu_si128((__m128i const*) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*) src2);
                        src += 16;
                        src2 += 16;

                        y3 = _mm_and_si128(ymask, yuv);
                        y3 = _mm_bsrli_si128(y3, 1);
                        y4 = _mm_and_si128(ymask, yuv2);
                        y4 = _mm_bsrli_si128(y4, 1);

                        uv3 = _mm_andnot_si128(ymask, yuv);
                        uv4 = _mm_andnot_si128(ymask, yuv2);

                        uv3 = _mm_avg_epu8(uv3, uv4);

                        dsty = _mm_packus_epi16(y1, y3);
                        dsty2 = _mm_packus_epi16(y2, y4);
                        dstuv = _mm_packus_epi16(uv, uv3);
                        _mm_storeu_si128((__m128i *) dst_y, dsty);
                        _mm_storeu_si128((__m128i *) dst_y2, dsty2);
                        _mm_storeu_si128((__m128i *) dst_cbcr, dstuv);
                        dst_y += 16;
                        dst_y2 += 16;
                        dst_cbcr += 16;
                }
#endif
                for(; x < width - 1; x += 2) {
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                }
        }
}

void v210_to_yuv420p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y / 2);
                for(int x = 0; x < width / 6; ++x) {
			//block 1, bits  0 -  9: U0+0
			//block 1, bits 10 - 19: Y0
			//block 1, bits 20 - 29: V0+1
			//block 2, bits  0 -  9: Y1
			//block 2, bits 10 - 19: U2+3
			//block 2, bits 20 - 29: Y2
			//block 3, bits  0 -  9: V2+3
			//block 3, bits 10 - 19: Y3
			//block 3, bits 20 - 29: U4+5
			//block 4, bits  0 -  9: Y4
			//block 4, bits 10 - 19: V4+5
			//block 4, bits 20 - 29: Y5
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;
                        w1_0 = *src2++;
                        w1_1 = *src2++;
                        w1_2 = *src2++;
                        w1_3 = *src2++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_y2++ = (w1_0 >> 10) & 0x3ff;
                        *dst_y2++ = w1_1 & 0x3ff;
                        *dst_y2++ = (w1_1 >> 20) & 0x3ff;
                        *dst_y2++ = (w1_2 >> 10) & 0x3ff;
                        *dst_y2++ = w1_3 & 0x3ff;
                        *dst_y2++ = (w1_3 >> 20) & 0x3ff;

                        *dst_cb++ = ((w0_0 & 0x3ff) + (w1_0 & 0x3ff)) / 2;
                        *dst_cb++ = (((w0_1 >> 10) & 0x3ff) + ((w1_1 >> 10) & 0x3ff)) / 2;
                        *dst_cb++ = (((w0_2 >> 20) & 0x3ff) + ((w1_2 >> 20) & 0x3ff)) / 2;

                        *dst_cr++ = (((w0_0 >> 20) & 0x3ff) + ((w1_0 >> 20) & 0x3ff)) / 2;
                        *dst_cr++ = ((w0_2 & 0x3ff) + (w1_2 & 0x3ff)) / 2;
                        *dst_cr++ = (((w0_3 >> 10) & 0x3ff) + ((w1_3 >> 10) & 0x3ff)) / 2;
                }
        }
}

void v210_to_yuv422p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);
                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;

                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                }
        }
}

void v210_to_yuv444p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);
                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;

                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                }
        }
}

void v210_to_p010le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cbcr = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);
                for(int x = 0; x < width / 6; ++x) {
			//block 1, bits  0 -  9: U0+0
			//block 1, bits 10 - 19: Y0
			//block 1, bits 20 - 29: V0+1
			//block 2, bits  0 -  9: Y1
			//block 2, bits 10 - 19: U2+3
			//block 2, bits 20 - 29: Y2
			//block 3, bits  0 -  9: V2+3
			//block 3, bits 10 - 19: Y3
			//block 3, bits 20 - 29: U4+5
			//block 4, bits  0 -  9: Y4
			//block 4, bits 10 - 19: V4+5
			//block 4, bits 20 - 29: Y5
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;
                        w1_0 = *src2++;
                        w1_1 = *src2++;
                        w1_2 = *src2++;
                        w1_3 = *src2++;

                        *dst_y++ = ((w0_0 >> 10) & 0x3ff) << 6;
                        *dst_y++ = (w0_1 & 0x3ff) << 6;
                        *dst_y++ = ((w0_1 >> 20) & 0x3ff) << 6;
                        *dst_y++ = ((w0_2 >> 10) & 0x3ff) << 6;
                        *dst_y++ = (w0_3 & 0x3ff) << 6;
                        *dst_y++ = ((w0_3 >> 20) & 0x3ff) << 6;

                        *dst_y2++ = ((w1_0 >> 10) & 0x3ff) << 6;
                        *dst_y2++ = (w1_1 & 0x3ff) << 6;
                        *dst_y2++ = ((w1_1 >> 20) & 0x3ff) << 6;
                        *dst_y2++ = ((w1_2 >> 10) & 0x3ff) << 6;
                        *dst_y2++ = (w1_3 & 0x3ff) << 6;
                        *dst_y2++ = ((w1_3 >> 20) & 0x3ff) << 6;

                        *dst_cbcr++ = (((w0_0 & 0x3ff) + (w1_0 & 0x3ff)) / 2) << 6; // Cb
                        *dst_cbcr++ = ((((w0_0 >> 20) & 0x3ff) + ((w1_0 >> 20) & 0x3ff)) / 2) << 6; // Cr
                        *dst_cbcr++ = ((((w0_1 >> 10) & 0x3ff) + ((w1_1 >> 10) & 0x3ff)) / 2) << 6; // Cb
                        *dst_cbcr++ = (((w0_2 & 0x3ff) + (w1_2 & 0x3ff)) / 2) << 6; // Cr
                        *dst_cbcr++ = ((((w0_2 >> 20) & 0x3ff) + ((w1_2 >> 20) & 0x3ff)) / 2) << 6; // Cb
                        *dst_cbcr++ = ((((w0_3 >> 10) & 0x3ff) + ((w1_3 >> 10) & 0x3ff)) / 2) << 6; // Cr
                }
        }
}

void rgb_to_bgr0(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, RGB);
        int dst_linesize = vc_get_linesize(width, RGBA);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst = out_frame->data[0] + out_frame->linesize[0] * y;
                vc_copylineRGBtoRGBA(dst, src, dst_linesize, 16, 8, 0);
        }
}

static void rgb_rgba_to_gbrp(AVFrame *out_frame, unsigned char *in_data, int width, int height, bool rgba)
{
        int src_linesize = vc_get_linesize(width, RGB);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst_b = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_g = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_r = out_frame->data[2] + out_frame->linesize[2] * y;
                for (int x = 0; x < width; ++x) {
                        *dst_r++ = *src++;
                        *dst_g++ = *src++;
                        *dst_b++ = *src++;
                        if (rgba) {
                                src++;
                        }
                }
        }
}

void rgb_to_gbrp(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        rgb_rgba_to_gbrp(out_frame, in_data, width, height, false);
}

void rgba_to_gbrp(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        rgb_rgba_to_gbrp(out_frame, in_data, width, height, true);
}

void r10k_to_gbrp10le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R10k);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_b = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_g = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);
                for (int x = 0; x < width; ++x) {
                        unsigned char w0 = *src++;
                        unsigned char w1 = *src++;
                        unsigned char w2 = *src++;
                        unsigned char w3 = *src++;
                        *dst_r++ = w0 << 2 | w1 >> 6;
                        *dst_g++ = (w1 & 0x3f) << 4 | w2 >> 4;
                        *dst_b++ = (w2 & 0xf) << 6 | w3 >> 2;
                }
        }
}

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

void r12l_to_gbrp12le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_b = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_g = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);
		for (int x = 0; x < width; x += 8) {
			uint16_t tmp;
			tmp = src[BYTE_SWAP(0)];
			tmp |= (src[BYTE_SWAP(1)] & 0xf) << 8;
			*dst_r++ = tmp; // r0
			*dst_g++ = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // g0
			tmp = src[BYTE_SWAP(3)];
			src += 4;
			tmp |= (src[BYTE_SWAP(0)] & 0xf) << 8;
			*dst_b++ = tmp; // b0
			*dst_r++ = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // r1
			tmp = src[BYTE_SWAP(2)];
			tmp |= (src[BYTE_SWAP(3)] & 0xf) << 8;
			*dst_g++ = tmp; // g1
			tmp = src[BYTE_SWAP(3)] >> 4;
			src += 4;
			*dst_b++ = src[BYTE_SWAP(0)] << 4 | tmp; // b1
			tmp = src[BYTE_SWAP(1)];
			tmp |= (src[BYTE_SWAP(2)] & 0xf) << 8;
			*dst_r++ = tmp; // r2
			*dst_g++ = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // g2
			src += 4;
			tmp = src[BYTE_SWAP(0)];
			tmp |= (src[BYTE_SWAP(1)] & 0xf) << 8;
			*dst_b++ = tmp; // b2
			*dst_r++ = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // r3
			tmp = src[BYTE_SWAP(3)];
			src += 4;
			tmp |= (src[BYTE_SWAP(0)] & 0xf) << 8;
			*dst_g++ = tmp; // g3
			*dst_b++ = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // b3
			tmp = src[BYTE_SWAP(2)];
			tmp |= (src[BYTE_SWAP(3)] & 0xf) << 8;
			*dst_r++ = tmp; // r4
			tmp = src[BYTE_SWAP(3)] >> 4;
			src += 4;
			*dst_g++ = src[BYTE_SWAP(0)] << 4 | tmp; // g4
			tmp = src[BYTE_SWAP(1)];
			tmp |= (src[BYTE_SWAP(2)] & 0xf) << 8;
			*dst_b++ = tmp; // b4
			*dst_r++ = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // r5
			src += 4;
			tmp = src[BYTE_SWAP(0)];
			tmp |= (src[BYTE_SWAP(1)] & 0xf) << 8;
			*dst_g++ = tmp; // g5
			*dst_b++ = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // b5
			tmp = src[BYTE_SWAP(3)];
			src += 4;
			tmp |= (src[BYTE_SWAP(0)] & 0xf) << 8;
			*dst_r++ = tmp; // r6
			*dst_g++ = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // g6
			tmp = src[BYTE_SWAP(2)];
			tmp |= (src[BYTE_SWAP(3)] & 0xf) << 8;
			*dst_b++ = tmp; // b6
			tmp = src[BYTE_SWAP(3)] >> 4;
			src += 4;
			*dst_r++ = src[BYTE_SWAP(0)] << 4 | tmp; // r7
			tmp = src[BYTE_SWAP(1)];
			tmp |= (src[BYTE_SWAP(2)] & 0xf) << 8;
			*dst_g++ = tmp; // g7
			*dst_b++ = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // b7
			src += 4;
                }
        }
}

void nv12_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cbcr = (char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                char *dst = dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                }
        }
}

void rgb24_to_uyvy(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoUYVY((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0], vc_get_linesize(width, UYVY), 0, 0, 0);
        }
}

void memcpy_data(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        UNUSED(width);
        for (int y = 0; y < height; ++y) {
                memcpy(dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                frame->linesize[0]);
        }
}

void gbrp_to_rgb(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        uint8_t *buf = (uint8_t *) dst_buffer + y * pitch + x * 3;
                        int src_idx = y * frame->linesize[0] + x;
                        buf[2] = frame->data[0][src_idx];
                        buf[1] = frame->data[1][src_idx];
                        buf[0] = frame->data[2][src_idx];
                }
        }
}

void gbrp_to_rgba(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        for (int y = 0; y < height; ++y) {
                uint32_t *line = (uint32_t *) ((uint8_t *) dst_buffer + y * pitch);
                int src_idx = y * frame->linesize[0];
                for (int x = 0; x < width; ++x) {
                        *line++ = frame->data[0][src_idx] << rgb_shift[B] |
                                frame->data[1][src_idx] << rgb_shift[G] |
                                frame->data[2][src_idx] << rgb_shift[R];
                        src_idx += 1;
                }
        }
}

void gbrp10le_to_r10k(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_b = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_g = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
		unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;
                for (int x = 0; x < width; ++x) {
			*dst++ = *src_r >> 2;
			*dst++ = (*src_r++ & 0x3) << 6 | *src_g >> 4;
			*dst++ = (*src_g++ & 0xf) << 4 | *src_b >> 6;
			*dst++ = (*src_b++ & 0x3f) << 2;
                }
        }
}

void gbrp10le_to_rgb(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_b = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_g = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
		unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;
                for (int x = 0; x < width; ++x) {
			*dst++ = *src_r++ >> 2;
			*dst++ = *src_g++ >> 2;
			*dst++ = *src_b++ >> 2;
                }
        }
}
void gbrp10le_to_rgba(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        for (int y = 0; y < height; ++y) {
                uint16_t *src_b = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_g = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
		uint32_t *dst = (uint32_t *) (dst_buffer + y * pitch);
                for (int x = 0; x < width; ++x) {
			*dst++ = (*src_r++ >> 2) << rgb_shift[0] | (*src_g++ >> 2) << rgb_shift[1] |
                                (*src_b++ >> 2) << rgb_shift[2];
                }
        }
}

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

void gbrp12le_to_r12l(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_b = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_g = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;
                for (int x = 0; x < width; x += 8) {
                        dst[BYTE_SWAP(0)] = *src_r & 0xff;
                        dst[BYTE_SWAP(1)] = (*src_g & 0xf) << 4 | *src_r++ >> 8;
                        dst[BYTE_SWAP(2)] = *src_g++ >> 4;
                        dst[BYTE_SWAP(3)] = *src_b & 0xff;
                        dst[4 + BYTE_SWAP(0)] = (*src_r & 0xf) << 4 | *src_b++ >> 8;
                        dst[4 + BYTE_SWAP(1)] = *src_r++ >> 4;
                        dst[4 + BYTE_SWAP(2)] = *src_g & 0xff;
                        dst[4 + BYTE_SWAP(3)] = (*src_b & 0xf) << 4 | *src_g++ >> 8;
                        dst[8 + BYTE_SWAP(0)] = *src_b++ >> 4;
                        dst[8 + BYTE_SWAP(1)] = *src_r & 0xff;
                        dst[8 + BYTE_SWAP(2)] = (*src_g & 0xf) << 4 | *src_r++ >> 8;
                        dst[8 + BYTE_SWAP(3)] = *src_g++ >> 4;
                        dst[12 + BYTE_SWAP(0)] = *src_b & 0xff;
                        dst[12 + BYTE_SWAP(1)] = (*src_r & 0xf) << 4 | *src_b++ >> 8;
                        dst[12 + BYTE_SWAP(2)] = *src_r++ >> 4;
                        dst[12 + BYTE_SWAP(3)] = *src_g & 0xff;
                        dst[16 + BYTE_SWAP(0)] = (*src_b & 0xf) << 4 | *src_g++ >> 8;
                        dst[16 + BYTE_SWAP(1)] = *src_b++ >> 4;
                        dst[16 + BYTE_SWAP(2)] = *src_r & 0xff;
                        dst[16 + BYTE_SWAP(3)] = (*src_g & 0xf) << 4 | *src_r++ >> 8;
                        dst[20 + BYTE_SWAP(0)] = *src_g++ >> 4;
                        dst[20 + BYTE_SWAP(1)] = *src_b & 0xff;
                        dst[20 + BYTE_SWAP(2)] = (*src_r & 0xf) << 4 | *src_b++ >> 8;
                        dst[20 + BYTE_SWAP(3)] = *src_r++ >> 4;;
                        dst[24 + BYTE_SWAP(0)] = *src_g & 0xff;
                        dst[24 + BYTE_SWAP(1)] = (*src_b & 0xf) << 4 | *src_g++ >> 8;
                        dst[24 + BYTE_SWAP(2)] = *src_b++ >> 4;
                        dst[24 + BYTE_SWAP(3)] = *src_r & 0xff;
                        dst[28 + BYTE_SWAP(0)] = (*src_g & 0xf) << 4 | *src_r++ >> 8;
                        dst[28 + BYTE_SWAP(1)] = *src_g++ >> 4;
                        dst[28 + BYTE_SWAP(2)] = *src_b & 0xff;
                        dst[28 + BYTE_SWAP(3)] = (*src_r & 0xf) << 4 | *src_b++ >> 8;
                        dst[32 + BYTE_SWAP(0)] = *src_r++ >> 4;
                        dst[32 + BYTE_SWAP(1)] = *src_g & 0xff;
                        dst[32 + BYTE_SWAP(2)] = (*src_b & 0xf) << 4 | *src_g++ >> 8;
                        dst[32 + BYTE_SWAP(3)] = *src_b++ >> 4;
                        dst += 36;
                }
        }
}

void gbrp12le_to_rgb(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_b = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_g = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;
                for (int x = 0; x < width; ++x) {
                        *dst++ = *src_r++ >> 4;
                        *dst++ = *src_g++ >> 4;
                        *dst++ = *src_b++ >> 4;
                }
        }
}

void gbrp12le_to_rgba(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        for (int y = 0; y < height; ++y) {
                uint16_t *src_b = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_g = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *) (dst_buffer + y * pitch);
                for (int x = 0; x < width; ++x) {
			*dst++ = (*src_r++ >> 4) << rgb_shift[0] | (*src_g++ >> 4) << rgb_shift[1] |
                                (*src_b++ >> 4) << rgb_shift[2];
                }
        }
}

void rgb48le_to_rgba(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toRGBA((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, RGBA), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

void rgb48le_to_r12l(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch, int rgb_shift[])
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toR12L((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, R12L), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

void yuv420p_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height / 2; ++y) {
                char *src_y1 = (char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                char *src_y2 = (char *) in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1);
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst1 = dst_buffer + (y * 2) * pitch;
                char *dst2 = dst_buffer + (y * 2 + 1) * pitch;

                int x = 0;

#ifdef __SSE3__
                __m128i y1;
                __m128i y2;
                __m128i u1;
                __m128i u2;
                __m128i v1;
                __m128i v2;
                __m128i out1l;
                __m128i out1h;
                __m128i out2l;
                __m128i out2h;
                __m128i zero = _mm_set1_epi32(0);

                for (; x < width - 15; x += 16){
                        y1 = _mm_lddqu_si128((__m128i const*) src_y1);
                        y2 = _mm_lddqu_si128((__m128i const*) src_y2);
                        src_y1 += 16;
                        src_y2 += 16;

                        out1l = _mm_unpacklo_epi8(zero, y1);
                        out1h = _mm_unpackhi_epi8(zero, y1);
                        out2l = _mm_unpacklo_epi8(zero, y2);
                        out2h = _mm_unpackhi_epi8(zero, y2);

                        u1 = _mm_lddqu_si128((__m128i const*) src_cb);
                        v1 = _mm_lddqu_si128((__m128i const*) src_cr);
                        src_cb += 8;
                        src_cr += 8;

                        u1 = _mm_unpacklo_epi8(u1, zero);
                        v1 = _mm_unpacklo_epi8(v1, zero);
                        u2 = _mm_unpackhi_epi8(u1, zero);
                        v2 = _mm_unpackhi_epi8(v1, zero);
                        u1 = _mm_unpacklo_epi8(u1, zero);
                        v1 = _mm_unpacklo_epi8(v1, zero);

                        v1 = _mm_bslli_si128(v1, 2);
                        v2 = _mm_bslli_si128(v2, 2);

                        u1 = _mm_or_si128(u1, v1);
                        u2 = _mm_or_si128(u2, v2);

                        out1l = _mm_or_si128(out1l, u1);
                        out1h = _mm_or_si128(out1h, u2);
                        out2l = _mm_or_si128(out2l, u1);
                        out2h = _mm_or_si128(out2h, u2);

                        _mm_storeu_si128((__m128i *) dst1, out1l);
                        dst1 += 16;
                        _mm_storeu_si128((__m128i *) dst1, out1h);
                        dst1 += 16;
                        _mm_storeu_si128((__m128i *) dst2, out2l);
                        dst2 += 16;
                        _mm_storeu_si128((__m128i *) dst2, out2h);
                        dst2 += 16;
                }
#endif

                for(; x < width - 1; x += 2) {
                        *dst1++ = *src_cb;
                        *dst1++ = *src_y1++;
                        *dst1++ = *src_cr;
                        *dst1++ = *src_y1++;

                        *dst2++ = *src_cb++;
                        *dst2++ = *src_y2++;
                        *dst2++ = *src_cr++;
                        *dst2++ = *src_y2++;
                }
        }
}

void yuv420p_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height / 2; ++y) {
                uint8_t *src_y1 = (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint8_t *src_y2 = (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src_cb << 2;
                        w1_0 = *src_cb << 2;
                        src_cb++;
                        w0_0 = w0_0 | (*src_y1++ << 2) << 10;
                        w1_0 = w1_0 | (*src_y2++ << 2) << 10;
                        w0_0 = w0_0 | (*src_cr << 2) << 20;
                        w1_0 = w1_0 | (*src_cr << 2) << 20;
                        src_cr++;

                        w0_1 = *src_y1++ << 2;
                        w1_1 = *src_y2++ << 2;
                        w0_1 = w0_1 | (*src_cb << 2) << 10;
                        w1_1 = w1_1 | (*src_cb << 2) << 10;
                        src_cb++;
                        w0_1 = w0_1 | (*src_y1++ << 2) << 20;
                        w1_1 = w1_1 | (*src_y2++ << 2) << 20;

                        w0_2 = *src_cr << 2;
                        w1_2 = *src_cr << 2;
                        src_cr++;
                        w0_2 = w0_2 | (*src_y1++ << 2) << 10;
                        w1_2 = w1_2 | (*src_y2++ << 2) << 10;
                        w0_2 = w0_2 | (*src_cb << 2) << 20;
                        w1_2 = w1_2 | (*src_cb << 2) << 20;
                        src_cb++;

                        w0_3 = *src_y1++;
                        w1_3 = *src_y2++;
                        w0_3 = w0_3 | (*src_cr << 2) << 10;
                        w1_3 = w1_3 | (*src_cr << 2) << 10;
                        src_cr++;
                        w0_3 = w0_3 | (*src_y1++ << 2) << 20;
                        w1_3 = w1_3 | (*src_y2++ << 2) << 20;

                        *dst1++ = w0_0;
                        *dst1++ = w0_1;
                        *dst1++ = w0_2;
                        *dst1++ = w0_3;

                        *dst2++ = w1_0;
                        *dst2++ = w1_1;
                        *dst2++ = w1_2;
                        *dst2++ = w1_3;
                }
        }
}

void yuv422p_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cr++;
                        *dst++ = *src_y++;
                }
        }
}

void yuv422p_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src_cb++ << 2;
                        w0_0 = w0_0 | (*src_y++ << 2) << 10;
                        w0_0 = w0_0 | (*src_cr++ << 2) << 20;

                        w0_1 = *src_y++ << 2;
                        w0_1 = w0_1 | (*src_cb++ << 2) << 10;
                        w0_1 = w0_1 | (*src_y++ << 2) << 20;

                        w0_2 = *src_cr++ << 2;
                        w0_2 = w0_2 | (*src_y++ << 2) << 10;
                        w0_2 = w0_2 | (*src_cb++ << 2) << 20;

                        w0_3 = *src_y++ << 2;
                        w0_3 = w0_3 | (*src_cr++ << 2) << 10;
                        w0_3 = w0_3 | (*src_y++ << 2) << 20;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}


void yuv444p_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = (*src_cb + *(src_cb + 1)) / 2;
                        src_cb += 2;
                        *dst++ = *src_y++;
                        *dst++ = (*src_cr + *(src_cr + 1)) / 2;
                        src_cr += 2;
                        *dst++ = *src_y++;
                }
        }
}

void yuv444p_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = ((src_cb[0] << 2) + (src_cb[1] << 2)) / 2;
                        w0_0 = w0_0 | (*src_y++ << 2) << 10;
                        w0_0 = w0_0 | ((src_cr[0] << 2) + (src_cr[1] << 2)) / 2 << 20;
                        src_cb += 2;
                        src_cr += 2;

                        w0_1 = *src_y++ << 2;
                        w0_1 = w0_1 | ((src_cb[0] << 2) + (src_cb[1] << 2)) / 2 << 10;
                        w0_1 = w0_1 | (*src_y++ << 2) << 20;
                        src_cb += 2;

                        w0_2 = ((src_cr[0] << 2) + (src_cr[1] << 2)) / 2;
                        w0_2 = w0_2 | (*src_y++ << 2) << 10;
                        w0_2 = w0_2 | ((src_cb[0] << 2) + (src_cb[1] << 2)) / 2 << 20;
                        src_cr += 2;
                        src_cb += 2;

                        w0_3 = *src_y++ << 2;
                        w0_3 = w0_3 | ((src_cr[0] << 2) + (src_cr[1] << 2)) / 2 << 10;
                        w0_3 = w0_3 | (*src_y++ << 2) << 20;
                        src_cr += 2;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}


/**
 * Changes pixel format from planar YUV 422 to packed RGB.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
void nv12_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cbcr = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        int cb = *src_cbcr++ - 128;
                        int cr = *src_cbcr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y++ << 16;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

/**
 * Changes pixel format from planar YUV 422 to packed RGB.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
void yuv422p_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y++ << 16;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

/**
 * Changes pixel format from planar YUV 422 to packed RGB.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
void yuv420p_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height / 2; ++y) {
                unsigned char *src_y1 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                unsigned char *src_y2 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1);
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst1 = (unsigned char *) dst_buffer + pitch * (y * 2);
                unsigned char *dst2 = (unsigned char *) dst_buffer + pitch * (y * 2 + 1);
                for(int x = 0; x < width / 2; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y1++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst1++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y1++ << 16;
                        *dst1++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y2++ << 16;
                        *dst2++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y2++ << 16;
                        *dst2++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

/**
 * Changes pixel format from planar YUV 444 to packed RGB.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
void yuv444p_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;
                for(int x = 0; x < width; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

void yuv420p10le_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src_cb;
                        w1_0 = *src_cb;
                        src_cb++;
                        w0_0 = w0_0 | (*src_y1++) << 10;
                        w1_0 = w1_0 | (*src_y2++) << 10;
                        w0_0 = w0_0 | (*src_cr) << 20;
                        w1_0 = w1_0 | (*src_cr) << 20;
                        src_cr++;

                        w0_1 = *src_y1++;
                        w1_1 = *src_y2++;
                        w0_1 = w0_1 | (*src_cb) << 10;
                        w1_1 = w1_1 | (*src_cb) << 10;
                        src_cb++;
                        w0_1 = w0_1 | (*src_y1++) << 20;
                        w1_1 = w1_1 | (*src_y2++) << 20;

                        w0_2 = *src_cr;
                        w1_2 = *src_cr;
                        src_cr++;
                        w0_2 = w0_2 | (*src_y1++) << 10;
                        w1_2 = w1_2 | (*src_y2++) << 10;
                        w0_2 = w0_2 | (*src_cb) << 20;
                        w1_2 = w1_2 | (*src_cb) << 20;
                        src_cb++;

                        w0_3 = *src_y1++;
                        w1_3 = *src_y2++;
                        w0_3 = w0_3 | (*src_cr) << 10;
                        w1_3 = w1_3 | (*src_cr) << 10;
                        src_cr++;
                        w0_3 = w0_3 | (*src_y1++) << 20;
                        w1_3 = w1_3 | (*src_y2++) << 20;

                        *dst1++ = w0_0;
                        *dst1++ = w0_1;
                        *dst1++ = w0_2;
                        *dst1++ = w0_3;

                        *dst2++ = w1_0;
                        *dst2++ = w1_1;
                        *dst2++ = w1_2;
                        *dst2++ = w1_3;
                }
        }
}

void yuv422p10le_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src_cb++;
                        w0_0 = w0_0 | (*src_y++) << 10;
                        w0_0 = w0_0 | (*src_cr++) << 20;

                        w0_1 = *src_y++;
                        w0_1 = w0_1 | (*src_cb++) << 10;
                        w0_1 = w0_1 | (*src_y++) << 20;

                        w0_2 = *src_cr++;
                        w0_2 = w0_2 | (*src_y++) << 10;
                        w0_2 = w0_2 | (*src_cb++) << 20;

                        w0_3 = *src_y++;
                        w0_3 = w0_3 | (*src_cr++) << 10;
                        w0_3 = w0_3 | (*src_y++) << 20;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}

void yuv444p10le_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = (src_cb[0] + src_cb[1]) / 2;
                        w0_0 = w0_0 | (*src_y++) << 10;
                        w0_0 = w0_0 | (src_cr[0] + src_cr[1]) / 2 << 20;
                        src_cb += 2;
                        src_cr += 2;

                        w0_1 = *src_y++;
                        w0_1 = w0_1 | (src_cb[0] + src_cb[1]) / 2 << 10;
                        w0_1 = w0_1 | (*src_y++) << 20;
                        src_cb += 2;

                        w0_2 = (src_cr[0] + src_cr[1]) / 2;
                        w0_2 = w0_2 | (*src_y++) << 10;
                        w0_2 = w0_2 | (src_cb[0] + src_cb[1]) / 2 << 20;
                        src_cr += 2;
                        src_cb += 2;

                        w0_3 = *src_y++;
                        w0_3 = w0_3 | (src_cr[0] + src_cr[1]) / 2 << 10;
                        w0_3 = w0_3 | (*src_y++) << 20;
                        src_cr += 2;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}

void yuv420p10le_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst1 = (uint8_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint8_t *dst2 = (uint8_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                for(int x = 0; x < width / 2; ++x) {
                        uint8_t tmp;
                        // U
                        tmp = *src_cb++ >> 2;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 2;
                        *dst2++ = *src_y2++ >> 2;
                        // V
                        tmp = *src_cr++ >> 2;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 2;
                        *dst2++ = *src_y2++ >> 2;
                }
        }
}

void yuv422p10le_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++ >> 2;
                        *dst++ = *src_y++ >> 2;
                        *dst++ = *src_cr++ >> 2;
                        *dst++ = *src_y++ >> 2;
                }
        }
}

void yuv444p10le_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = (src_cb[0] + src_cb[0]) / 2 >> 2;
                        *dst++ = *src_y++ >> 2;
                        *dst++ = (src_cr[0] + src_cr[1]) / 2 >> 2;
                        *dst++ = *src_y++ >> 2;
                        src_cb += 2;
                        src_cr += 2;
                }
        }
}

void yuv420p10le_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        char *tmp = malloc(vc_get_linesize(UYVY, width) * height);
        char *uyvy = tmp;
        yuv420p10le_to_uyvy(uyvy, in_frame, width, height, vc_get_linesize(UYVY, width), rgb_shift);
        for (int i = 0; i < height; i++) {
                vc_copylineUYVYtoRGB((unsigned char *) dst_buffer, (unsigned char *) uyvy, vc_get_linesize(RGB, width), 0, 0, 0);
                uyvy += vc_get_linesize(UYVY, width);
                dst_buffer += pitch;
        }
        free(tmp);
}

void yuv422p10le_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        char *tmp = malloc(vc_get_linesize(UYVY, width) * height);
        char *uyvy = tmp;
        yuv422p10le_to_uyvy(uyvy, in_frame, width, height, vc_get_linesize(UYVY, width), rgb_shift);
        for (int i = 0; i < height; i++) {
                vc_copylineUYVYtoRGB((unsigned char *) dst_buffer, (unsigned char *) uyvy, vc_get_linesize(RGB, width), 0, 0, 0);
                uyvy += vc_get_linesize(UYVY, width);
                dst_buffer += pitch;
        }
        free(tmp);
}

void yuv444p10le_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        char *tmp = malloc(vc_get_linesize(UYVY, width) * height);
        char *uyvy = tmp;
        yuv444p10le_to_uyvy(uyvy, in_frame, width, height, vc_get_linesize(UYVY, width), rgb_shift);
        for (int i = 0; i < height; i++) {
                vc_copylineUYVYtoRGB((unsigned char *) dst_buffer, (unsigned char *) uyvy, vc_get_linesize(RGB, width), 0, 0, 0);
                uyvy += vc_get_linesize(UYVY, width);
                dst_buffer += pitch;
        }
        free(tmp);
}

void p010le_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint8_t *src_y1 = (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint8_t *src_y2 = (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint8_t *src_cbcr = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src_cbcr << 2; // Cb0
                        w1_0 = *src_cbcr << 2;
                        src_cbcr++; // Cr0
                        w0_0 = w0_0 | (*src_y1++ << 2) << 10;
                        w1_0 = w1_0 | (*src_y2++ << 2) << 10;
                        w0_0 = w0_0 | (*src_cbcr << 2) << 20;
                        w1_0 = w1_0 | (*src_cbcr << 2) << 20;
                        src_cbcr++; // Cb1

                        w0_1 = *src_y1++ << 2;
                        w1_1 = *src_y2++ << 2;
                        w0_1 = w0_1 | (*src_cbcr << 2) << 10;
                        w1_1 = w1_1 | (*src_cbcr << 2) << 10;
                        src_cbcr++; // Cr1
                        w0_1 = w0_1 | (*src_y1++ << 2) << 20;
                        w1_1 = w1_1 | (*src_y2++ << 2) << 20;

                        w0_2 = *src_cbcr << 2;
                        w1_2 = *src_cbcr << 2;
                        src_cbcr++;
                        w0_2 = w0_2 | (*src_y1++ << 2) << 10;
                        w1_2 = w1_2 | (*src_y2++ << 2) << 10;
                        w0_2 = w0_2 | (*src_cbcr << 2) << 20;
                        w1_2 = w1_2 | (*src_cbcr << 2) << 20;
                        src_cbcr++;

                        w0_3 = *src_y1++;
                        w1_3 = *src_y2++;
                        w0_3 = w0_3 | (*src_cbcr << 2) << 10;
                        w1_3 = w1_3 | (*src_cbcr << 2) << 10;
                        src_cbcr++;
                        w0_3 = w0_3 | (*src_y1++ << 2) << 20;
                        w1_3 = w1_3 | (*src_y2++ << 2) << 20;

                        *dst1++ = w0_0;
                        *dst1++ = w0_1;
                        *dst1++ = w0_2;
                        *dst1++ = w0_3;

                        *dst2++ = w1_0;
                        *dst2++ = w1_1;
                        *dst2++ = w1_2;
                        *dst2++ = w1_3;
                }
        }
}

void p010le_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cbcr = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *dst1 = (uint8_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint8_t *dst2 = (uint8_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                for(int x = 0; x < width / 2; ++x) {
                        uint8_t tmp;
                        // U
                        tmp = *src_cbcr++ >> 2;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 2;
                        *dst2++ = *src_y2++ >> 2;
                        // V
                        tmp = *src_cbcr++ >> 2;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 2;
                        *dst2++ = *src_y2++ >> 2;
                }
        }
}

#ifdef HWACC_VDPAU
void av_vdpau_to_ug_vdpau(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch, int rgb_shift[])
{
        UNUSED(width);
        UNUSED(height);
        UNUSED(pitch);
        UNUSED(rgb_shift);

        struct video_frame_callbacks *callbacks = in_frame->opaque;

        hw_vdpau_frame *out = (hw_vdpau_frame *) dst_buffer;

        hw_vdpau_frame_init(out);

        hw_vdpau_frame_from_avframe(out, in_frame);

        callbacks->recycle = hw_vdpau_recycle_callback; 
        callbacks->copy = hw_vdpau_copy_callback; 
}
#endif

