/**
 * @file   libavcodec/lavc_video_conversions.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2013-2022 CESNET, z. s. p. o.
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
/**
 * @file
 * References:
 * 1. [v210](https://wiki.multimedia.cx/index.php/V210)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "color.h"
#include "config_common.h"
#include "host.h"
#include "libavcodec/to_lavc_vid_conv.h"
#include "utils/macros.h" // OPTIMIZED_FOR
#include "video.h"

#ifdef __SSE3__
#include "pmmintrin.h"
#endif

#if LIBAVUTIL_VERSION_INT > AV_VERSION_INT(51, 63, 100) // FFMPEG commit e9757066e11
#define HAVE_12_AND_14_PLANAR_COLORSPACES 1
#endif

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic warning "-Wpass-failed"

static void uyvy_to_yuv420p(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int y;
        for (y = 0; y < height - 1; y += 2) {
                /*  every even row */
                unsigned char *src = in_data + y * (((width + 1) & ~1) * 2);
                /*  every odd row */
                unsigned char *src2 = in_data + (y + 1) * (((width + 1) & ~1) * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_y2 = out_frame->data[0] + out_frame->linesize[0] * (y + 1);
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * (y / 2);
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * (y / 2);

                int x;
                OPTIMIZED_FOR (x = 0; x < width - 1; x += 2) {
                        *dst_cb++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                }
                if (x < width) {
                        *dst_cb++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cr++ = (*src++ + *src2++) / 2;
                }
        }
        if (y < height) {
                unsigned char *src = in_data + y * (((width + 1) & ~1) * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * (y / 2);
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * (y / 2);
                int x;
                OPTIMIZED_FOR (x = 0; x < width - 1; x += 2) {
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
                if (x < width) {
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src++;
                }
        }
}

static void uyvy_to_yuv422p(AVFrame * __restrict out_frame, unsigned char * __restrict src, int width, int height)
{
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;

                OPTIMIZED_FOR (int x = 0; x < width; x += 2) {
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

static void uyvy_to_yuv444p(AVFrame * __restrict out_frame, unsigned char * __restrict src, int width, int height)
{
        for(int y = 0; y < height; ++y) {
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;

                OPTIMIZED_FOR (int x = 0; x < width; x += 2) {
                        *dst_cb++ = *src;
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

static void uyvy_to_nv12(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
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
                        yuv = _mm_lddqu_si128((__m128i const*)(const void *) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*)(const void *) src2);
                        src += 16;
                        src2 += 16;

                        y1 = _mm_and_si128(ymask, yuv);
                        y1 = _mm_bsrli_si128(y1, 1);
                        y2 = _mm_and_si128(ymask, yuv2);
                        y2 = _mm_bsrli_si128(y2, 1);

                        uv = _mm_andnot_si128(ymask, yuv);
                        uv2 = _mm_andnot_si128(ymask, yuv2);

                        uv = _mm_avg_epu8(uv, uv2);

                        yuv = _mm_lddqu_si128((__m128i const*)(const void *) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*)(const void *) src2);
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
                        _mm_storeu_si128((__m128i *)(void *) dst_y, dsty);
                        _mm_storeu_si128((__m128i *)(void *) dst_y2, dsty2);
                        _mm_storeu_si128((__m128i *)(void *) dst_cbcr, dstuv);
                        dst_y += 16;
                        dst_y2 += 16;
                        dst_cbcr += 16;
                }
#endif

                OPTIMIZED_FOR (; x < width - 1; x += 2) {
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                }
        }
}

static void v210_to_yuv420p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *)(void *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y / 2);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

static void v210_to_yuv422p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

static void v210_to_yuv444p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

static void v210_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0 = *src++;
                        uint32_t w0_1 = *src++;
                        uint32_t w0_2 = *src++;
                        uint32_t w0_3 = *src++;

                        *dst_y++ = ((w0_0 >> 10U) & 0x3FFU) << 6U;
                        *dst_y++ = (w0_1 & 0x3FFU) << 6U;
                        *dst_y++ = ((w0_1 >> 20U) & 0x3FFU) << 6U;
                        *dst_y++ = ((w0_2 >> 10U) & 0x3FFU) << 6U;
                        *dst_y++ = (w0_3 & 0x3FFU) << 6U;
                        *dst_y++ = ((w0_3 >> 20U) & 0x3FFU) << 6U;

                        *dst_cb++ = (w0_0 & 0x3FFU) << 6U;
                        *dst_cb++ = (w0_0 & 0x3FFU) << 6U;
                        *dst_cb++ = ((w0_1 >> 10U) & 0x3FFU) << 6U;
                        *dst_cb++ = ((w0_1 >> 10U) & 0x3FFU) << 6U;
                        *dst_cb++ = ((w0_2 >> 20U) & 0x3FFU) << 6U;
                        *dst_cb++ = ((w0_2 >> 20U) & 0x3FFU) << 6U;

                        *dst_cr++ = ((w0_0 >> 20U) & 0x3FFU) << 6U;
                        *dst_cr++ = ((w0_0 >> 20U) & 0x3FFU) << 6U;
                        *dst_cr++ = (w0_2 & 0x3FFU) << 6U;
                        *dst_cr++ = (w0_2 & 0x3FFU) << 6U;
                        *dst_cr++ = ((w0_3 >> 10U) & 0x3FFU) << 6U;
                        *dst_cr++ = ((w0_3 >> 10U) & 0x3FFU) << 6U;
                }
        }
}

static void v210_to_p010le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);

        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *)(void *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cbcr = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

#ifdef HAVE_P210
static void v210_to_p210le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);

        for(int y = 0; y < height; y++) {
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cbcr = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0 = *src++;
                        uint32_t w0_1 = *src++;
                        uint32_t w0_2 = *src++;
                        uint32_t w0_3 = *src++;

                        *dst_y++ = ((w0_0 >> 10) & 0x3ff) << 6;
                        *dst_y++ = (w0_1 & 0x3ff) << 6;
                        *dst_y++ = ((w0_1 >> 20) & 0x3ff) << 6;
                        *dst_y++ = ((w0_2 >> 10) & 0x3ff) << 6;
                        *dst_y++ = (w0_3 & 0x3ff) << 6;
                        *dst_y++ = ((w0_3 >> 20) & 0x3ff) << 6;

                        *dst_cbcr++ = (w0_0 & 0x3ff) << 6; // Cb
                        *dst_cbcr++ = ((w0_0 >> 20) & 0x3ff) << 6; // Cr
                        *dst_cbcr++ = ((w0_1 >> 10) & 0x3ff) << 6; // Cb
                        *dst_cbcr++ = (w0_2 & 0x3ff) << 6; // Cr
                        *dst_cbcr++ = ((w0_2 >> 20) & 0x3ff) << 6; // Cb
                        *dst_cbcr++ = ((w0_3 >> 10) & 0x3ff) << 6; // Cr
                }
        }
}
#endif

#if defined __GNUC__
static inline void r10k_to_yuv42Xp10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, int v_subsampl_rate)
        __attribute__((always_inline));
#endif
static inline void r10k_to_yuv42Xp10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, int v_subsampl_rate)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, R10k);
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * (y / v_subsampl_rate));
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * (y / v_subsampl_rate));
                unsigned char *src = in_data + y * src_linesize;
                int iterations = width / 2;
                OPTIMIZED_FOR(int x = 0; x < iterations; x++){
                        comp_type_t r = src[0] << 2 | src[1] >> 6;
                        comp_type_t g = (src[1] & 0x3f ) << 4 | src[2] >> 4;
                        comp_type_t b = (src[2] & 0x0f) << 6 | src[3] >> 2;

                        comp_type_t res_y = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE)) + (1<<(10-4));
                        comp_type_t res_cb = (RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE)) + (1<<(10-1));
                        comp_type_t res_cr = (RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE)) + (1<<(10-1));

                        dst_y[x * 2] = CLAMP_LIMITED_Y(res_y, 10);
                        src += 4;

                        r = src[0] << 2 | src[1] >> 6;
                        g = (src[1] & 0x3f ) << 4 | src[2] >> 4;
                        b = (src[2] & 0x0f) << 6 | src[3] >> 2;

                        res_y = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE)) + (1<<(10-4));
                        res_cb += (RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE)) + (1<<(10-1));
                        res_cr += (RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE)) + (1<<(10-1));

                        res_cb /= 2;
                        res_cr /= 2;
                        res_y = CLAMP_LIMITED_Y(res_y, 10);
                        res_cb = CLAMP_LIMITED_CBCR(res_cb, 10);
                        res_cr = CLAMP_LIMITED_CBCR(res_cr, 10);

                        dst_y[x * 2 + 1] = res_y;
                        if (v_subsampl_rate == 1) {
                                dst_cb[x] = res_cb;
                                dst_cr[x] = res_cr;
                        } else {
                                if (x % 2 == 0) {
                                        dst_cb[x] = res_cb;
                                        dst_cr[x] = res_cr;
                                } else {
                                        dst_cb[x] = (dst_cb[x] + res_cb) / 2;
                                        dst_cr[x] = (dst_cr[x] + res_cr) / 2;
                                }
                        }

                        src += 4;
                }

        }
}

static void r10k_to_yuv420p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv42Xp10le(out_frame, in_data, width, height, 2);
}

static void r10k_to_yuv422p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv42Xp10le(out_frame, in_data, width, height, 1);
}

/**
 * Converts to yuv444p 10/12/14 le
 */
#if defined __GNUC__
static inline void r10k_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
        __attribute__((always_inline));
#endif
static inline void r10k_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, R10k);
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                unsigned char *src = in_data + y * src_linesize;
                OPTIMIZED_FOR(int x = 0; x < width; x++){
                        comp_type_t r = src[0] << 2 | src[1] >> 6;
                        comp_type_t g = (src[1] & 0x3F ) << 4 | src[2] >> 4;
                        comp_type_t b = (src[2] & 0x0F) << 6 | src[3] >> 2;

                        comp_type_t res_y = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE+10-depth)) + (1<<(depth-4));
                        comp_type_t res_cb = (RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE+10-depth)) + (1<<(depth-1));
                        comp_type_t res_cr = (RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE+10-depth)) + (1<<(depth-1));

                        *dst_y++ = CLAMP(res_y, 1<<(depth-4), 235 * (1<<(depth-8)));
                        *dst_cb++ = CLAMP(res_cb, 1<<(depth-4), 240 * (1<<(depth-8)));
                        *dst_cr++ = CLAMP(res_cr, 1<<(depth-4), 240 * (1<<(depth-8)));
                        src += 4;
                }
        }
}

static void r10k_to_yuv444p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void r10k_to_yuv444p12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void r10k_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

// RGB full range to YCbCr bt. 709 limited range
#if defined __GNUC__
static inline void r12l_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
        __attribute__((always_inline));
#endif
static inline void r12l_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

#define WRITE_RES \
        res_y = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE+12-depth)) + (1<<(depth-4));\
        res_cb = (RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE+12-depth)) + (1<<(depth-1));\
        res_cr = (RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE+12-depth)) + (1<<(depth-1));\
        *dst_y++ = CLAMP(res_y, 1<<(depth-4), 235 * (1<<(depth-8)));\
        *dst_cb++ = CLAMP(res_cb, 1<<(depth-4), 240 * (1<<(depth-8)));\
        *dst_cr++ = CLAMP(res_cr, 1<<(depth-4), 240 * (1<<(depth-8)));

        const int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
			comp_type_t r = 0;
			comp_type_t g = 0;
			comp_type_t b = 0;
                        comp_type_t res_y = 0;
                        comp_type_t res_cb = 0;
                        comp_type_t res_cr = 0;

			r = src[BYTE_SWAP(0)];
			r |= (src[BYTE_SWAP(1)] & 0xF) << 8;
			g = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // g0
			b = src[BYTE_SWAP(3)];
			src += 4;

			b |= (src[BYTE_SWAP(0)] & 0xF) << 8;
                        WRITE_RES // 0
			r = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // r1
			g = src[BYTE_SWAP(2)];
			g |= (src[BYTE_SWAP(3)] & 0xF) << 8;
			b = src[BYTE_SWAP(3)] >> 4;
			src += 4;

			b |= src[BYTE_SWAP(0)] << 4; // b1
                        WRITE_RES // 1
			r = src[BYTE_SWAP(1)];
			r |= (src[BYTE_SWAP(2)] & 0xF) << 8;
			g = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // g2
			src += 4;

			b = src[BYTE_SWAP(0)];
			b |= (src[BYTE_SWAP(1)] & 0xF) << 8;
                        WRITE_RES // 2
			r = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // r3
			g = src[BYTE_SWAP(3)];
			src += 4;

			g |= (src[BYTE_SWAP(0)] & 0xF) << 8;
			b = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // b3
                        WRITE_RES // 3
			r = src[BYTE_SWAP(2)];
			r |= (src[BYTE_SWAP(3)] & 0xF) << 8;
			g = src[BYTE_SWAP(3)] >> 4;
			src += 4;

			g |= src[BYTE_SWAP(0)] << 4; // g4
			b = src[BYTE_SWAP(1)];
			b |= (src[BYTE_SWAP(2)] & 0xF) << 8;
			WRITE_RES // 4
			r = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // r5
			src += 4;

			g = src[BYTE_SWAP(0)];
			g |= (src[BYTE_SWAP(1)] & 0xF) << 8;
			b = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // b5
                        WRITE_RES // 5
			r = src[BYTE_SWAP(3)];
			src += 4;

			r |= (src[BYTE_SWAP(0)] & 0xF) << 8;
			g = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // g6
			b = src[BYTE_SWAP(2)];
			b |= (src[BYTE_SWAP(3)] & 0xF) << 8;
                        WRITE_RES // 6
			r = src[BYTE_SWAP(3)] >> 4;
			src += 4;

			r |= src[BYTE_SWAP(0)] << 4; // r7
			g = src[BYTE_SWAP(1)];
			g |= (src[BYTE_SWAP(2)] & 0xF) << 8;
			b = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // b7
                        WRITE_RES // 7
			src += 4;
                }
        }
}

static void r12l_to_yuv444p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void r12l_to_yuv444p12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void r12l_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

/// @brief Converts RG48 to yuv444p 10/12/14 le
#if defined __GNUC__
static inline void rg48_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
        __attribute__((always_inline));
#endif
static inline void rg48_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, RG48);
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                uint16_t *src = (uint16_t *)(void *) (in_data + y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < width; x++){
                        comp_type_t r = *src++;
                        comp_type_t g = *src++;
                        comp_type_t b = *src++;

                        comp_type_t res_y = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE+16-depth)) + (1<<(depth-4));
                        comp_type_t res_cb = (RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE+16-depth)) + (1<<(depth-1));
                        comp_type_t res_cr = (RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE+16-depth)) + (1<<(depth-1));

                        *dst_y++ = CLAMP(res_y, 1<<(depth-4), 235 * (1<<(depth-8)));
                        *dst_cb++ = CLAMP(res_cb, 1<<(depth-4), 240 * (1<<(depth-8)));
                        *dst_cr++ = CLAMP(res_cr, 1<<(depth-4), 240 * (1<<(depth-8)));
                }
        }
}

static void rg48_to_yuv444p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void rg48_to_yuv444p12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void rg48_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

#if defined __GNUC__
static inline void y216_to_yuv422pXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int depth)
        __attribute__((always_inline));
#endif
static inline void y216_to_yuv422pXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int depth)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, Y216);

        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                uint16_t *src = (uint16_t *)(void *) (in_data + y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < (width + 1) / 2; x++){
                        *dst_y++ = *src++ >> (16U - depth);
                        *dst_cb++ = *src++ >> (16U - depth);
                        *dst_y++ = *src++ >> (16U - depth);
                        *dst_cr++ = *src++ >> (16U - depth);
                }
        }
}

static void y216_to_yuv422p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        y216_to_yuv422pXXle(out_frame, in_data, width, height, 10);
}

static void y216_to_yuv422p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        y216_to_yuv422pXXle(out_frame, in_data, width, height, 16);
}

static void y216_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, Y216);

        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                uint16_t *src = (uint16_t *)(void *) (in_data + y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < (width + 1) / 2; x++){
                        *dst_y++ = *src++;
                        dst_cb[0] = dst_cb[1] = *src++;
                        dst_cb += 2;
                        *dst_y++ = *src++;
                        dst_cr[0] = dst_cr[1] = *src++;
                        dst_cr += 2;
                }
        }
}

static void rgb_to_bgr0(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, RGB);
        int dst_linesize = vc_get_linesize(width, RGBA);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst = out_frame->data[0] + out_frame->linesize[0] * y;
                vc_copylineRGBtoRGBA(dst, src, dst_linesize, 16, 8, 0);
        }
}

static void r10k_to_bgr0(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R10k);
        int dst_linesize = vc_get_linesize(width, RGBA);
        decoder_t vc_copyliner10k = get_decoder_from_to(R10k, RGBA);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst = out_frame->data[0] + out_frame->linesize[0] * y;
                vc_copyliner10k(dst, src, dst_linesize, 16, 8, 0);
        }
}

#if defined __GNUC__
static inline void rgb_rgba_to_gbrp(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, int bpp)
        __attribute__((always_inline));
#endif
static inline void rgb_rgba_to_gbrp(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, int bpp)
{
        int src_linesize = bpp * width;
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst_g = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_b = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_r = out_frame->data[2] + out_frame->linesize[2] * y;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst_r++ = src[0];
                        *dst_g++ = src[1];
                        *dst_b++ = src[2];
                        src += bpp;
                }
        }
}

/**
 * @todo
 * Aspiring conversion from R10K to AV_PIX_FMT_X2RGB10LE. As is, it worked with
 * current (130d19bf2) FFmpeg, but the resulting stream was weird 8-bit 4:2:0
 * RGB because the module doesn't set 444 and 10-bit properties (IS_* macros).
 * However, if those macros were toggled on, stream with artifacts is produded.
 */
static void r10k_to_x2rgb10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height) ATTRIBUTE(unused);

static void r10k_to_x2rgb10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R10k);
        for (int y = 0; y < height; ++y) {
                uint32_t *src = (uint32_t *)(void *) (in_data + y * src_linesize);
                uint32_t *dst = (uint32_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                for (int x = 0; x < width; ++x) {
                        *dst++ = htonl(*src++); /// @todo worked, but should be AFAIK htonl(*src++)>>2
                }
        }
}

static void rgb_to_gbrp(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rgb_rgba_to_gbrp(out_frame, in_data, width, height, 3);
}

static void rgba_to_gbrp(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rgb_rgba_to_gbrp(out_frame, in_data, width, height, 4);
}

#if defined __GNUC__
static inline void r10k_to_gbrpXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int depth)
        __attribute__((always_inline));
#endif
static inline void r10k_to_gbrpXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int depth)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        int src_linesize = vc_get_linesize(width, R10k);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_g = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_b = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        unsigned char w0 = *src++;
                        unsigned char w1 = *src++;
                        unsigned char w2 = *src++;
                        unsigned char w3 = *src++;
                        *dst_r++ = (w0 << 2U | w1 >> 6U) << (depth - 10U);
                        *dst_g++ = ((w1 & 0x3FU) << 4U | w2 >> 4U) << (depth - 10U);
                        *dst_b++ = ((w2 & 0xFU) << 6U | w3 >> 2U) << (depth - 10U);
                }
        }
}

static void r10k_to_gbrp10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_gbrpXXle(out_frame, in_data, width, height, 10U);
}

static void r10k_to_gbrp16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_gbrpXXle(out_frame, in_data, width, height, 16U);
}

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

/// @note out_depth needs to be at least 12
#if defined __GNUC__
static inline void r12l_to_gbrpXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int out_depth)
        __attribute__((always_inline));
#endif
static inline void r12l_to_gbrpXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int out_depth)
{
        assert(out_depth >= 12);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

#undef S
#define S(x) ((x) << (out_depth - 12U))

        int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_g = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_b = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
                        uint16_t tmp = src[BYTE_SWAP(0)];
                        tmp |= (src[BYTE_SWAP(1)] & 0xFU) << 8U;
                        *dst_r++ = S(tmp); // r0
                        *dst_g++ = S(src[BYTE_SWAP(2)] << 4U | src[BYTE_SWAP(1)] >> 4U); // g0
                        tmp = src[BYTE_SWAP(3)];
                        src += 4;
                        tmp |= (src[BYTE_SWAP(0)] & 0xFU) << 8U;
                        *dst_b++ = S(tmp); // b0
                        *dst_r++ = S(src[BYTE_SWAP(1)] << 4U | src[BYTE_SWAP(0)] >> 4U); // r1
                        tmp = src[BYTE_SWAP(2)];
                        tmp |= (src[BYTE_SWAP(3)] & 0xFU) << 8U;
                        *dst_g++ = S(tmp); // g1
                        tmp = src[BYTE_SWAP(3)] >> 4U;
                        src += 4;
                        *dst_b++ = S(src[BYTE_SWAP(0)] << 4U | tmp); // b1
                        tmp = src[BYTE_SWAP(1)];
                        tmp |= (src[BYTE_SWAP(2)] & 0xFU) << 8U;
                        *dst_r++ = S(tmp); // r2
                        *dst_g++ = S(src[BYTE_SWAP(3)] << 4U | src[BYTE_SWAP(2)] >> 4U); // g2
                        src += 4;
                        tmp = src[BYTE_SWAP(0)];
                        tmp |= (src[BYTE_SWAP(1)] & 0xFU) << 8U;
                        *dst_b++ = S(tmp); // b2
                        *dst_r++ = S(src[BYTE_SWAP(2)] << 4U | src[BYTE_SWAP(1)] >> 4U); // r3
                        tmp = src[BYTE_SWAP(3)];
                        src += 4;
                        tmp |= (src[BYTE_SWAP(0)] & 0xFU) << 8U;
                        *dst_g++ = S(tmp); // g3
                        *dst_b++ = S(src[BYTE_SWAP(1)] << 4U | src[BYTE_SWAP(0)] >> 4U); // b3
                        tmp = src[BYTE_SWAP(2)];
                        tmp |= (src[BYTE_SWAP(3)] & 0xFU) << 8U;
                        *dst_r++ = S(tmp); // r4
                        tmp = src[BYTE_SWAP(3)] >> 4U;
                        src += 4;
                        *dst_g++ = S(src[BYTE_SWAP(0)] << 4U | tmp); // g4
                        tmp = src[BYTE_SWAP(1)];
                        tmp |= (src[BYTE_SWAP(2)] & 0xFU) << 8U;
                        *dst_b++ = S(tmp); // b4
                        *dst_r++ = S(src[BYTE_SWAP(3)] << 4U | src[BYTE_SWAP(2)] >> 4U); // r5
                        src += 4;
                        tmp = src[BYTE_SWAP(0)];
                        tmp |= (src[BYTE_SWAP(1)] & 0xFU) << 8U;
                        *dst_g++ = S(tmp); // g5
                        *dst_b++ = S(src[BYTE_SWAP(2)] << 4U | src[BYTE_SWAP(1)] >> 4U); // b5
                        tmp = src[BYTE_SWAP(3)];
                        src += 4;
                        tmp |= (src[BYTE_SWAP(0)] & 0xFU) << 8U;
                        *dst_r++ = S(tmp); // r6
                        *dst_g++ = S(src[BYTE_SWAP(1)] << 4U | src[BYTE_SWAP(0)] >> 4U); // g6
                        tmp = src[BYTE_SWAP(2)];
                        tmp |= (src[BYTE_SWAP(3)] & 0xFU) << 8U;
                        *dst_b++ = S(tmp); // b6
                        tmp = src[BYTE_SWAP(3)] >> 4U;
                        src += 4;
                        *dst_r++ = S(src[BYTE_SWAP(0)] << 4U | tmp); // r7
                        tmp = src[BYTE_SWAP(1)];
                        tmp |= (src[BYTE_SWAP(2)] & 0xFU) << 8U;
                        *dst_g++ = S(tmp); // g7
                        *dst_b++ = S(src[BYTE_SWAP(3)] << 4U | src[BYTE_SWAP(2)] >> 4U); // b7
                        src += 4;
                }
        }
}

static void r12l_to_gbrp16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_frame, in_data, width, height, 16U);
}

#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
static void r12l_to_gbrp12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_frame, in_data, width, height, 12U);
}

static void rg48_to_gbrp12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        int src_linesize = vc_get_linesize(width, RG48);
        for (int y = 0; y < height; ++y) {
                uint16_t *src = (uint16_t *)(void *) (in_data + y * src_linesize);
                uint16_t *dst_g = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_b = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst_r++ = *src++ >> 4U;
                        *dst_g++ = *src++ >> 4U;
                        *dst_b++ = *src++ >> 4U;
                }
        }
}
#endif

//
// conversion dispatchers
//
/**
 * @brief returns list of available conversion. Terminated by uv_to_av_conversion::src == VIDEO_CODEC_NONE
 */
const struct uv_to_av_conversion *get_uv_to_av_conversions() {
        /**
         * Conversions from UltraGrid to FFMPEG formats.
         *
         * Currently do not add an "upgrade" conversion (UYVY->10b) because also
         * UltraGrid decoder can be used first and thus conversion v210->UYVY->10b
         * may be used resulting in a precision loss. If needed, put the upgrade
         * conversions below the others.
         */
        static const struct uv_to_av_conversion uv_to_av_conversions[] = {
                { v210, AV_PIX_FMT_YUV420P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_yuv420p10le },
                { v210, AV_PIX_FMT_YUV422P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_yuv422p10le },
                { v210, AV_PIX_FMT_YUV444P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_yuv444p10le },
                { v210, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_yuv444p16le },
                { R10k, AV_PIX_FMT_YUV444P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv444p10le },
                { R10k, AV_PIX_FMT_YUV444P12LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv444p12le },
                { R10k, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv444p16le },
                { R12L, AV_PIX_FMT_YUV444P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r12l_to_yuv444p10le },
                { R12L, AV_PIX_FMT_YUV444P12LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r12l_to_yuv444p12le },
                { R12L, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r12l_to_yuv444p16le },
                { RG48, AV_PIX_FMT_YUV444P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, rg48_to_yuv444p10le },
                { RG48, AV_PIX_FMT_YUV444P12LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, rg48_to_yuv444p12le },
                { RG48, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, rg48_to_yuv444p16le },
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(55, 15, 100) // FFMPEG commit c2869b4640f
                { v210, AV_PIX_FMT_P010LE,      AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_p010le },
#endif
#ifdef HAVE_P210
                { v210, AV_PIX_FMT_P210LE,      AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_p210le },
#endif
                { UYVY, AV_PIX_FMT_YUV422P,     AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv422p },
                { UYVY, AV_PIX_FMT_YUVJ422P,    AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv422p },
                { UYVY, AV_PIX_FMT_YUV420P,     AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv420p },
                { UYVY, AV_PIX_FMT_YUVJ420P,    AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv420p },
                { UYVY, AV_PIX_FMT_NV12,        AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_nv12 },
                { UYVY, AV_PIX_FMT_YUV444P,     AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv444p },
                { UYVY, AV_PIX_FMT_YUVJ444P,    AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv444p },
                { Y216, AV_PIX_FMT_YUV422P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, y216_to_yuv422p10le },
                { Y216, AV_PIX_FMT_YUV422P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, y216_to_yuv422p16le },
                { Y216, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, y216_to_yuv444p16le },
                { RGB, AV_PIX_FMT_BGR0,         AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, rgb_to_bgr0 },
                { RGB, AV_PIX_FMT_GBRP,         AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, rgb_to_gbrp },
                { RGBA, AV_PIX_FMT_GBRP,        AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, rgba_to_gbrp },
                { R10k, AV_PIX_FMT_BGR0,        AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r10k_to_bgr0 },
                { R10k, AV_PIX_FMT_GBRP10LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r10k_to_gbrp10le },
                { R10k, AV_PIX_FMT_GBRP16LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r10k_to_gbrp16le },
                //{ R10k, AV_PIX_FMT_X2RGB10LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r10k_to_x2rgb10le },
                { R10k, AV_PIX_FMT_YUV422P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv422p10le },
                { R10k, AV_PIX_FMT_YUV420P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv420p10le },
#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
                { R12L, AV_PIX_FMT_GBRP12LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r12l_to_gbrp12le },
                { R12L, AV_PIX_FMT_GBRP16LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r12l_to_gbrp16le },
                { RG48, AV_PIX_FMT_GBRP12LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, rg48_to_gbrp12le },
#endif
                { 0, 0, 0, 0, 0 }
        };
        return uv_to_av_conversions;
}

pixfmt_callback_t get_uv_to_av_conversion(codec_t uv_codec, int av_codec) {
        for (const struct uv_to_av_conversion *conversions = get_uv_to_av_conversions();
                        conversions->func != 0; conversions++) {
                if (conversions->dst == av_codec &&
                                conversions->src == uv_codec) {
                        return conversions->func;
                }
        }

        return NULL;
}

void get_av_pixfmt_details(codec_t uv_codec, int av_codec, enum AVColorSpace *colorspace, enum AVColorRange *color_range)
{
        for (const struct uv_to_av_conversion *conversions = get_uv_to_av_conversions();
                        conversions->func != 0; conversions++) {
                if (conversions->dst == av_codec &&
                                conversions->src == uv_codec) {
                        *colorspace = conversions->colorspace;
                        *color_range = conversions->color_range;
                        return;
                }
        }
}


#pragma GCC diagnostic pop

/* vi: set expandtab sw=8: */
