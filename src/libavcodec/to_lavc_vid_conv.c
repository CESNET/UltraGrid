/**
 * @file   libavcodec/lavc_video_conversions.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2013-2025 CESNET
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
 * To measure performance of conversions, use tools/benchmark_ff_convs
 *
 * References:
 * 1. [v210](https://wiki.multimedia.cx/index.php/V210)
 */

#define __STDC_WANT_LIB_EXT1__ 1 // qsort_s

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "color_space.h"
#include "compat/net.h"                        // for htonl
#include "compat/qsort_s.h"
#include "debug.h"
#include "host.h"
#include "libavcodec/to_lavc_vid_conv.h"
#include "libavcodec/to_lavc_vid_conv_cuda.h"
#include "utils/macros.h" // OPTIMIZED_FOR
#include "utils/parallel_conv.h"
#include "utils/worker.h"
#include "video.h"

#ifdef __SSE3__
#include "pmmintrin.h"
#endif

#define MOD_NAME "[to_lavc_vid_conv] "

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic warning "-Wpass-failed"

static void uyvy_to_yuv420p(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        uyvy_to_i420(out_frame->data, out_frame->linesize, in_data, width,
                     height);
}

static void uyvy_to_yuv422p(AVFrame * __restrict out_frame, const unsigned char * __restrict src, int width, int height)
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

static void uyvy_to_vuya(AVFrame * __restrict out_frame, const unsigned char * __restrict src, int width, int height)
        __attribute__((unused));
static void uyvy_to_vuya(AVFrame * __restrict out_frame, const unsigned char * __restrict src, int width, int height)
{
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *dst = out_frame->data[0] + out_frame->linesize[0] * y;

                OPTIMIZED_FOR (int x = 0; x < width; x += 2) {
                        *dst++ = src[2];
                        *dst++ = src[0];
                        *dst++ = src[1];
                        *dst++ = 0xff;
                        *dst++ = src[2];
                        *dst++ = src[0];
                        *dst++ = src[3];
                        *dst++ = 0xff;
                        src += 4;
                }
        }
}

static void uyvy_to_yuv444p(AVFrame * __restrict out_frame, const unsigned char * __restrict src, int width, int height)
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

static void to_lavc_uyvy_to_nv12(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        uyvy_to_nv12(out_frame->data, out_frame->linesize, in_data, width,
                     height);
}

static void v210_to_yuv420p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                const uint32_t *src2 = (const uint32_t *)(const void *) (in_data + (y + 1) * vc_get_linesize(width, v210));
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

static void v210_to_yuv422p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * vc_get_linesize(width, v210));
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

static void v210_to_yuv444p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * vc_get_linesize(width, v210));
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

static void v210_to_yuv444p16le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * vc_get_linesize(width, v210));
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

static void v210_to_xv30(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
        __attribute__((unused));
static void v210_to_xv30(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 4 == 0);

        for(int y = 0; y < height; y += 1) {
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * vc_get_linesize(width, v210));
                uint32_t *dst = (uint32_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);

                OPTIMIZED_FOR (int x = 0; x < (width + 5) / 6; ++x) {
                        uint32_t w0 = *src++;
                        uint32_t w1 = *src++;
                        uint32_t w2 = *src++;
                        uint32_t w3 = *src++;
                        *dst++ = w0;
                        *dst++ = (w0 & 0xFFF003FFU) | (w1 & 0x3FFU) << 10U;

                        *dst++ = (w2 & 0x3FFU) << 20U | (w1 & 0x3FF00000U) >> 10U | (w1 & 0xFFC00U) >> 10U;
                        *dst++ = (w2 & 0x3FFU) << 20U | (w2 & 0xFFC00U) | (w1 & 0xFFC00U) >> 10U;

                        *dst++ = (w3 & 0xFFC00U) << 10U | (w3 & 0x3FFU) << 10U | (w2 & 0x3FF00000U) >> 20;
                        *dst++ = (w3 & 0xFFC00U) << 10U | (w3 & 0x3FF00000U) >> 10 | (w2 & 0x3FF00000U) >> 20;
                }
        }
}

static void v210_to_y210(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
        __attribute__((unused));
static void v210_to_y210(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);

                OPTIMIZED_FOR (int x = 0; x < (width + 5) / 6; ++x) {
                        uint32_t w = *src++;
                        dst[1] = ((w >>  0U) & 0x3FFU) << 6U; // U
                        dst[0] = ((w >> 10U) & 0x3FFU) << 6U; // Y
                        dst[3] = ((w >> 20U) & 0x3FFU) << 6U; // V
                        w = *src++;
                        dst[2] = ((w >>  0U) & 0x3FFU) << 6U; // Y
                        dst[5] = ((w >> 10U) & 0x3FFU) << 6U; // U
                        dst[4] = ((w >> 20U) & 0x3FFU) << 6U; // Y
                        w = *src++;
                        dst[7] = ((w >>  0U) & 0x3FFU) << 6U; // V
                        dst[6] = ((w >> 10U) & 0x3FFU) << 6U; // Y
                        dst[9] = ((w >> 20U) & 0x3FFU) << 6U; // U
                        w = *src++;
                        dst[8] = ((w >>  0U) & 0x3FFU) << 6U; // Y
                        dst[11] = ((w >> 10U) & 0x3FFU) << 6U; // V
                        dst[10] = ((w >> 20U) & 0x3FFU) << 6U; // Y
                        dst += 12;
                }
        }
}

static void y416_to_xv30(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
        __attribute__((unused));
static void y416_to_xv30(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 4 == 0);

        for(ptrdiff_t y = 0; y < height; y += 1) {
                const uint16_t *src = (const uint16_t *)(const void *) (in_data + y * vc_get_linesize(width, Y416));
                uint32_t *dst = (uint32_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        unsigned u = *src++;
                        unsigned y = *src++;
                        unsigned v = *src++;
                        unsigned a = *src++;
                        *dst++ = (a >> 14U) << 30U |
                                (v >> 6U) << 20U |
                                (y >> 6U) << 10U |
                                (u >> 6U);
                }
        }
}

static void
to_lavc_v210_to_p010le(AVFrame *__restrict out_frame,
                       const unsigned char *__restrict in_data, int width,
                       int height)
{
        v210_to_p010le(out_frame->data, out_frame->linesize, in_data, width,
                       height);
}

static void
to_lavc_y216_to_p010le(AVFrame *__restrict out_frame,
                       const unsigned char *__restrict in_data, int width,
                       int height)
{
        y216_to_p010le(out_frame->data, out_frame->linesize, in_data, width,
                       height);
}

#if P210_PRESENT
static void v210_to_p210le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);

        for(int y = 0; y < height; y++) {
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * vc_get_linesize(width, v210));
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
static inline void r10k_to_yuv42Xp10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, int v_subsampl_rate)
        __attribute__((always_inline));
#endif
static inline void r10k_to_yuv42Xp10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, int v_subsampl_rate)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        enum {
                D_DEPTH = 10,
        };

        const int src_linesize = vc_get_linesize(width, R10k);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, D_DEPTH);
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * (y / v_subsampl_rate));
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * (y / v_subsampl_rate));
                const unsigned char *src = in_data + y * src_linesize;
                int iterations = width / 2;
                OPTIMIZED_FOR(int x = 0; x < iterations; x++){
                        comp_type_t r = src[0] << 2 | src[1] >> 6;
                        comp_type_t g = (src[1] & 0x3f ) << 4 | src[2] >> 4;
                        comp_type_t b = (src[2] & 0x0f) << 6 | src[3] >> 2;

                        comp_type_t res_y =
                            (RGB_TO_Y(cfs, r, g, b) >> (COMP_BASE)) +
                            (1 << (D_DEPTH - 4));
                        comp_type_t res_cb =
                            (RGB_TO_CB(cfs, r, g, b) >> (COMP_BASE)) +
                            (1 << (D_DEPTH - 1));
                        comp_type_t res_cr =
                            (RGB_TO_CR(cfs, r, g, b) >> (COMP_BASE)) +
                            (1 << (D_DEPTH - 1));

                        dst_y[x * 2] = CLAMP_LIMITED_Y(res_y, 10);
                        src += 4;

                        r = src[0] << 2 | src[1] >> 6;
                        g = (src[1] & 0x3f ) << 4 | src[2] >> 4;
                        b = (src[2] & 0x0f) << 6 | src[3] >> 2;

                        res_y = (RGB_TO_Y(cfs, r, g, b) >> (COMP_BASE)) +
                                (1 << (D_DEPTH - 4));
                        res_cb += (RGB_TO_CB(cfs, r, g, b) >> (COMP_BASE)) +
                                  (1 << (D_DEPTH - 1));
                        res_cr += (RGB_TO_CR(cfs, r, g, b) >> (COMP_BASE)) +
                                  (1 << (D_DEPTH - 1));

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

static void r10k_to_yuv420p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv42Xp10le(out_frame, in_data, width, height, 2);
}

static void r10k_to_yuv422p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv42Xp10le(out_frame, in_data, width, height, 1);
}

/**
 * Converts to yuv444p 10/12/14 le
 */
#if defined __GNUC__
static inline void r10k_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
        __attribute__((always_inline));
#endif
static inline void r10k_to_yuv444pXXle(int out_depth, AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, R10k);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, out_depth);
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                const unsigned char *src = in_data + y * src_linesize;
                OPTIMIZED_FOR(int x = 0; x < width; x++){
                        comp_type_t r = src[0] << 2 | src[1] >> 6;
                        comp_type_t g = (src[1] & 0x3F ) << 4 | src[2] >> 4;
                        comp_type_t b = (src[2] & 0x0F) << 6 | src[3] >> 2;

                        comp_type_t res_y = (RGB_TO_Y(cfs, r, g, b) >>
                                             (COMP_BASE + 10 - out_depth)) +
                                            (1 << (out_depth - 4));
                        comp_type_t res_cb = (RGB_TO_CB(cfs, r, g, b) >>
                                              (COMP_BASE + 10 - out_depth)) +
                                             (1 << (out_depth - 1));
                        comp_type_t res_cr = (RGB_TO_CR(cfs, r, g, b) >>
                                              (COMP_BASE + 10 - out_depth)) +
                                             (1 << (out_depth - 1));

                        *dst_y++ = CLAMP_LIMITED_Y(res_y, out_depth);
                        *dst_cb++ = CLAMP_LIMITED_CBCR(res_cb, out_depth);
                        *dst_cr++ = CLAMP_LIMITED_CBCR(res_cr, out_depth);
                        src += 4;
                }
        }
}

static void r10k_to_yuv444p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void r10k_to_yuv444p12le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void r10k_to_yuv444p16le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

#if defined __GNUC__
static inline void r12l_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
        __attribute__((always_inline));
#endif
static inline void r12l_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, depth);
#define WRITE_RES \
        res_y = (RGB_TO_Y(cfs, r, g, b) >> (COMP_BASE + 12 - depth)) + \
                (1 << (depth - 4)); \
        res_cb = (RGB_TO_CB(cfs, r, g, b) >> (COMP_BASE + 12 - depth)) + \
                 (1 << (depth - 1)); \
        res_cr = (RGB_TO_CR(cfs, r, g, b) >> (COMP_BASE + 12 - depth)) + \
                 (1 << (depth - 1)); \
        *dst_y++  = CLAMP_LIMITED_Y(res_y, depth); \
        *dst_cb++ = CLAMP_LIMITED_CBCR(res_cb, depth); \
        *dst_cr++ = CLAMP_LIMITED_CBCR(res_cr, depth);

        const int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                const unsigned char *src = in_data + y * src_linesize;
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

static void r12l_to_yuv444p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void r12l_to_yuv444p12le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void r12l_to_yuv444p16le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

/// @brief Converts RG48 to yuv444p 10/12/14 le
#if defined __GNUC__
static inline void rg48_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
        __attribute__((always_inline));
#endif
static inline void rg48_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, RG48);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, depth);
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                const uint16_t *src = (const uint16_t *)(const void *) (in_data + y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < width; x++){
                        comp_type_t r = *src++;
                        comp_type_t g = *src++;
                        comp_type_t b = *src++;

                        comp_type_t res_y = (RGB_TO_Y(cfs, r, g, b) >>
                                             (COMP_BASE + 16 - depth)) +
                                            (1 << (depth - 4));
                        comp_type_t res_cb = (RGB_TO_CB(cfs, r, g, b) >>
                                              (COMP_BASE + 16 - depth)) +
                                             (1 << (depth - 1));
                        comp_type_t res_cr = (RGB_TO_CR(cfs, r, g, b) >>
                                              (COMP_BASE + 16 - depth)) +
                                             (1 << (depth - 1));

                        *dst_y++  = CLAMP_LIMITED_Y(res_y, depth);
                        *dst_cb++ = CLAMP_LIMITED_CBCR(res_cb, depth);
                        *dst_cr++ = CLAMP_LIMITED_CBCR(res_cr, depth);
                }
        }
}

static void rg48_to_yuv444p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void rg48_to_yuv444p12le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void rg48_to_yuv444p16le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

static void
rgb_to_yuv444p(AVFrame *__restrict out_frame,
               const unsigned char *__restrict in_data, int width, int height)
{
        typedef uint8_t t; // in/out type
        enum {
                DEPTH = sizeof(t) * CHAR_BIT,
        };
        const ptrdiff_t src_linesize = vc_get_linesize(width, RGB);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH);
        for (ptrdiff_t y = 0; y < height; y++) {
                const t *src =
                    (const t *) (const void *) (in_data + y * src_linesize);
                t *dst_y =
                    (t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                t *dst_cb =
                    (t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                t *dst_cr =
                    (t *) (out_frame->data[2] + out_frame->linesize[2] * y);
                OPTIMIZED_FOR(int x = 0; x < width; x++)
                {
                        const comp_type_t r = *src++;
                        const comp_type_t g = *src++;
                        const comp_type_t b = *src++;

                        const comp_type_t res_y =
                            (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) +
                            (1 << (DEPTH - 4));
                        const comp_type_t res_cb =
                            (RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) +
                            (1 << (DEPTH - 1));
                        const comp_type_t res_cr =
                            (RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) +
                            (1 << (DEPTH - 1));

                        *dst_y++  = CLAMP_LIMITED_Y(res_y, DEPTH);
                        *dst_cb++ = CLAMP_LIMITED_Y(res_cb, DEPTH);
                        *dst_cr++ = CLAMP_LIMITED_Y(res_cr, DEPTH);
                }
        }
}

#if defined __GNUC__
static inline void y216_to_yuv422pXXle(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, unsigned int depth)
        __attribute__((always_inline));
#endif
static inline void y216_to_yuv422pXXle(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, unsigned int depth)
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
                const uint16_t *src = (const uint16_t *)(const void *) (in_data + y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < (width + 1) / 2; x++){
                        *dst_y++ = *src++ >> (16U - depth);
                        *dst_cb++ = *src++ >> (16U - depth);
                        *dst_y++ = *src++ >> (16U - depth);
                        *dst_cr++ = *src++ >> (16U - depth);
                }
        }
}

static void y216_to_yuv422p10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        y216_to_yuv422pXXle(out_frame, in_data, width, height, 10);
}

static void y216_to_yuv422p16le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        y216_to_yuv422pXXle(out_frame, in_data, width, height, 16);
}

static void y216_to_yuv444p16le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
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
                const uint16_t *src = (const uint16_t *)(const void *) (in_data + y * src_linesize);
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

static void rgb_to_bgr0(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, RGB);
        int dst_linesize = vc_get_linesize(width, RGBA);
        for (int y = 0; y < height; ++y) {
                const unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst = out_frame->data[0] + out_frame->linesize[0] * y;
                vc_copylineRGBtoRGBA(dst, src, dst_linesize, 16, 8, 0);
        }
}

static void r10k_to_bgr0(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R10k);
        int dst_linesize = vc_get_linesize(width, RGBA);
        decoder_t vc_copyliner10k = get_decoder_from_to(R10k, RGBA);
        for (int y = 0; y < height; ++y) {
                const unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst = out_frame->data[0] + out_frame->linesize[0] * y;
                vc_copyliner10k(dst, src, dst_linesize, 16, 8, 0);
        }
}

#if defined __GNUC__
static inline void rgb_rgba_to_gbrp(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, int bpp)
        __attribute__((always_inline));
#endif
static inline void rgb_rgba_to_gbrp(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, int bpp)
{
        int src_linesize = bpp * width;
        for (int y = 0; y < height; ++y) {
                const unsigned char *src = in_data + y * src_linesize;
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

static void r10k_to_x2rgb10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height) __attribute__((unused));

static void r10k_to_x2rgb10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R10k);
        for (int y = 0; y < height; ++y) {
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * src_linesize);
                uint32_t *dst = (uint32_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                for (int x = 0; x < width; ++x) {
                        *dst++ = htonl(*src++) >> 2;
                }
        }
}

static void rgb_to_gbrp(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        rgb_rgba_to_gbrp(out_frame, in_data, width, height, 3);
}

static void rgba_to_gbrp(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        rgb_rgba_to_gbrp(out_frame, in_data, width, height, 4);
}

static void
to_lavc_rgba_to_bgra(AVFrame *__restrict out_frame,
                     const unsigned char *__restrict in_data, int width,
                     int height)
{
        rgba_to_bgra(out_frame->data, out_frame->linesize, in_data, width,
                     height);
}

#if defined __GNUC__
static inline void r10k_to_gbrpXXle(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, unsigned int depth)
        __attribute__((always_inline));
#endif
static inline void r10k_to_gbrpXXle(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, unsigned int depth)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        int src_linesize = vc_get_linesize(width, R10k);
        for (int y = 0; y < height; ++y) {
                const unsigned char *src = in_data + y * src_linesize;
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

static void r10k_to_gbrp10le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_gbrpXXle(out_frame, in_data, width, height, 10U);
}

static void r10k_to_gbrp16le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_gbrpXXle(out_frame, in_data, width, height, 16U);
}

/// @note out_depth needs to be at least 12
#if defined __GNUC__
static inline void r12l_to_gbrpXXle(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, unsigned int out_depth)
        __attribute__((always_inline));
#endif
static inline void r12l_to_gbrpXXle(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height, unsigned int out_depth)
{
        assert(out_depth >= 12);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

#undef S
#define S(x) ((x) << (out_depth - 12U))

        int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                const unsigned char *src = in_data + y * src_linesize;
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

static void r12l_to_gbrp16le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_frame, in_data, width, height, 16U);
}

static void r12l_to_gbrp12le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_frame, in_data, width, height, 12U);
        memset(out_frame->data[1], 255, 5000);
}

static void rg48_to_gbrp12le(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        int src_linesize = vc_get_linesize(width, RG48);
        for (int y = 0; y < height; ++y) {
                const uint16_t *src = (const uint16_t *)(const void *) (in_data + y * src_linesize);
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

static void to_lavc_memcpy_data(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height) __attribute__((unused)); // defined below

//
// conversion dispatchers
//
typedef void uv_to_av_convert(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height);
typedef uv_to_av_convert *pixfmt_callback_t;
struct uv_to_av_conversion {
        codec_t src;
        enum AVPixelFormat dst;
        pixfmt_callback_t func;        ///< conversion function
};
// @brief returns list of available conversion. Terminated by uv_to_av_conversion::src == VIDEO_CODEC_NONE
static const struct uv_to_av_conversion *get_uv_to_av_conversions() {
        static const struct uv_to_av_conversion uv_to_av_conversions[] = {
                { v210, AV_PIX_FMT_YUV420P10LE, v210_to_yuv420p10le },
                { v210, AV_PIX_FMT_YUV422P10LE, v210_to_yuv422p10le },
                { v210, AV_PIX_FMT_YUV444P10LE, v210_to_yuv444p10le },
                { v210, AV_PIX_FMT_YUV444P16LE, v210_to_yuv444p16le },
#if XV3X_PRESENT
                { v210, AV_PIX_FMT_XV30,         v210_to_xv30 },
                { Y216, AV_PIX_FMT_Y212,         to_lavc_memcpy_data },
                { Y416, AV_PIX_FMT_XV30,         y416_to_xv30 },
                { v210, AV_PIX_FMT_Y212,         v210_to_y210 },
#endif
#if Y210_PRESENT
                { v210, AV_PIX_FMT_Y210,         v210_to_y210 },
                { Y216, AV_PIX_FMT_Y210,         to_lavc_memcpy_data },
#endif
                { R10k, AV_PIX_FMT_YUV444P10LE, r10k_to_yuv444p10le },
                { R10k, AV_PIX_FMT_YUV444P12LE, r10k_to_yuv444p12le },
                { R10k, AV_PIX_FMT_YUV444P16LE, r10k_to_yuv444p16le },
                { R12L, AV_PIX_FMT_YUV444P10LE, r12l_to_yuv444p10le },
                { R12L, AV_PIX_FMT_YUV444P12LE, r12l_to_yuv444p12le },
                { R12L, AV_PIX_FMT_YUV444P16LE, r12l_to_yuv444p16le },
                { RG48, AV_PIX_FMT_YUV444P10LE, rg48_to_yuv444p10le },
                { RG48, AV_PIX_FMT_YUV444P12LE, rg48_to_yuv444p12le },
                { RG48, AV_PIX_FMT_YUV444P16LE, rg48_to_yuv444p16le },
                { v210, AV_PIX_FMT_P010LE,      to_lavc_v210_to_p010le },
                { Y216, AV_PIX_FMT_P010LE,      to_lavc_y216_to_p010le },
#if P210_PRESENT
                { v210, AV_PIX_FMT_P210LE,      v210_to_p210le },
#endif
                { UYVY, AV_PIX_FMT_YUV422P,     uyvy_to_yuv422p },
                { UYVY, AV_PIX_FMT_YUVJ422P,    uyvy_to_yuv422p },
#if VUYX_PRESENT
                { UYVY, AV_PIX_FMT_VUYA,        uyvy_to_vuya },
                { UYVY, AV_PIX_FMT_VUYX,        uyvy_to_vuya },
#endif
                { UYVY, AV_PIX_FMT_YUV420P,     uyvy_to_yuv420p },
                { UYVY, AV_PIX_FMT_YUVJ420P,    uyvy_to_yuv420p },
                { UYVY, AV_PIX_FMT_NV12,        to_lavc_uyvy_to_nv12 },
                { UYVY, AV_PIX_FMT_YUV444P,     uyvy_to_yuv444p },
                { UYVY, AV_PIX_FMT_YUVJ444P,    uyvy_to_yuv444p },
                { Y216, AV_PIX_FMT_YUV422P10LE, y216_to_yuv422p10le },
                { Y216, AV_PIX_FMT_YUV422P16LE, y216_to_yuv422p16le },
                { Y216, AV_PIX_FMT_YUV444P16LE, y216_to_yuv444p16le },
                { RGB, AV_PIX_FMT_BGR0,         rgb_to_bgr0 },
                { RGB, AV_PIX_FMT_GBRP,         rgb_to_gbrp },
                { RGB, AV_PIX_FMT_YUV444P,      rgb_to_yuv444p },
                { RGBA, AV_PIX_FMT_GBRP,        rgba_to_gbrp },
                { RGBA, AV_PIX_FMT_BGRA,        to_lavc_rgba_to_bgra },
                { R10k, AV_PIX_FMT_BGR0,        r10k_to_bgr0 },
                { R10k, AV_PIX_FMT_GBRP10LE,    r10k_to_gbrp10le },
                { R10k, AV_PIX_FMT_GBRP16LE,    r10k_to_gbrp16le },
#if X2RGB10LE_PRESENT
                { R10k, AV_PIX_FMT_X2RGB10LE,   r10k_to_x2rgb10le },
#endif
                { R10k, AV_PIX_FMT_YUV422P10LE, r10k_to_yuv422p10le },
                { R10k, AV_PIX_FMT_YUV420P10LE, r10k_to_yuv420p10le },
                { R12L, AV_PIX_FMT_GBRP12LE,    r12l_to_gbrp12le },
                { R12L, AV_PIX_FMT_GBRP16LE,    r12l_to_gbrp16le },
                { RG48, AV_PIX_FMT_GBRP12LE,    rg48_to_gbrp12le },
                { VIDEO_CODEC_NONE, AV_PIX_FMT_NONE, 0 }
        };
        return uv_to_av_conversions;
}

void get_av_pixfmt_details(enum AVPixelFormat av_codec, enum AVColorSpace *colorspace, enum AVColorRange *color_range)
{
        const struct AVPixFmtDescriptor *avd = av_pix_fmt_desc_get(av_codec);
        if (!avd) {
                return;
        }
        if ((avd->flags & AV_PIX_FMT_FLAG_RGB) != 0) {
                *colorspace = AVCOL_SPC_RGB;
                *color_range = AVCOL_RANGE_JPEG;
        } else {
                *colorspace = AVCOL_SPC_BT709;
                *color_range = AVCOL_RANGE_MPEG;
        }
}

static int get_intermediate_codecs_from_uv_to_av(codec_t in, enum AVPixelFormat av, codec_t fmts[VIDEO_CODEC_COUNT]) {
        int fmt_set[VIDEO_CODEC_COUNT] = { VIDEO_CODEC_NONE }; // to avoid multiple occurrences
        for (const struct uv_to_av_pixfmt *i = get_av_to_ug_pixfmts(); i->uv_codec != VIDEO_CODEC_NONE; ++i) { // no AV conversion needed - direct mapping
                decoder_t decoder = get_decoder_from_to(in, i->uv_codec);
                if (decoder && i->av_pixfmt == av) {
                        fmt_set[i->uv_codec] = 1;
                }
        }
        for (const struct uv_to_av_conversion *c = get_uv_to_av_conversions(); c->src != VIDEO_CODEC_NONE; ++c) { // AV conversion needed
                decoder_t decoder = get_decoder_from_to(in, c->src);
                if (decoder && c->dst == av) {
                        fmt_set[c->src] = 1;
                }
        }

        int nb_fmts = 0;
        for (int i = 0; i < VIDEO_CODEC_COUNT; ++i) {
                if (fmt_set[i]) {
                        fmts[nb_fmts++] = (codec_t) i;
                }
        }
        return nb_fmts;
}

static QSORT_S_COMP_DEFINE(compare_uv_pixfmts, a, b, orig_c) {
        const codec_t *pix_a = (const codec_t *) a;
        const codec_t *pix_b = (const codec_t *) b;
        const struct pixfmt_desc *src_desc = (struct pixfmt_desc *) orig_c;
        struct pixfmt_desc desc_a = get_pixfmt_desc(*pix_a);
        struct pixfmt_desc desc_b = get_pixfmt_desc(*pix_b);

        int ret = compare_pixdesc(&desc_a, &desc_b, src_desc);
        return ret != 0 ? ret : (int) *pix_a - (int) *pix_b;
}

/**
 * Returns a UltraGrid decoder needed to decode from the UltraGrid codec in
 * to out with respect to conversions in @ref conversions. Therefore it should
 * be feasible to convert in to out and then convert out to av (last step may
 * be omitted if the format is native for both indicated in
 * ug_to_av_pixfmt_map).
 */
static decoder_t get_decoder_from_uv_to_uv(codec_t in, enum AVPixelFormat av, codec_t *out) {
        codec_t intermediate_codecs[VIDEO_CODEC_COUNT];
        int ic_count = get_intermediate_codecs_from_uv_to_av(in, av, intermediate_codecs);
        if (ic_count == 0) {
                return NULL;
        }

        struct pixfmt_desc src_desc = get_pixfmt_desc(in);
        qsort_s(intermediate_codecs, ic_count, sizeof intermediate_codecs[0], compare_uv_pixfmts, &src_desc);
        *out = intermediate_codecs[0];
        return get_decoder_from_to(in, *out);
}

decoder_t (*testable_get_decoder_from_uv_to_uv)(codec_t in, enum AVPixelFormat av, codec_t *out) = get_decoder_from_uv_to_uv; // external linkage for test

static pixfmt_callback_t select_pixfmt_callback(enum AVPixelFormat fmt, codec_t src) {
        // no conversion needed
        if (get_ug_to_av_pixfmt(src) != AV_PIX_FMT_NONE
                        && get_ug_to_av_pixfmt(src) == fmt) {
                return NULL;
        }

        for (const struct uv_to_av_conversion *c = get_uv_to_av_conversions(); c->src != VIDEO_CODEC_NONE; c++) { // FFMPEG conversion needed
                if (c->src == src && c->dst == fmt) {
                        return c->func;
                }
        }

        log_msg(LOG_LEVEL_FATAL, "[lavc] Cannot find conversion to any of encoder supported pixel format.\n");
        abort();
}

struct lavc_compare_convs_data {
        struct pixfmt_desc src_desc;
        struct pixfmt_desc descs[AV_PIX_FMT_NB];
        int steps[AV_PIX_FMT_NB]; ///< conversion steps - 1 if uv->av or uv->uv; 2 for uv->uv->av conversion
};
static inline int lavc_compare_convs_inner(enum AVPixelFormat pix_a, enum AVPixelFormat pix_b, const struct lavc_compare_convs_data *comp_data) {
        struct pixfmt_desc desc_a = comp_data->descs[pix_a];
        struct pixfmt_desc desc_b = comp_data->descs[pix_b];

        int ret = 0;
        if ((ret = compare_pixdesc(&desc_a, &desc_b, &comp_data->src_desc)) != 0) {
                return ret;
        }
        // if undistinguishable, it's possible that some resulting pixfmt is closer than another
        struct pixfmt_desc reduced_src_desc = { .depth = MIN(desc_a.depth, comp_data->src_desc.depth), // take minimum
                .subsampling = MIN(desc_a.subsampling, comp_data->src_desc.subsampling),  //   desc from src & dest to
                .rgb = comp_data->src_desc.rgb }; //                         reflect eventual intermediate degradation
        desc_a = av_pixfmt_get_desc(pix_a);
        desc_b = av_pixfmt_get_desc(pix_b);
        if ((ret = compare_pixdesc(&desc_a, &desc_b, &reduced_src_desc)) != 0) {
                return ret;
        }
        int steps_a = comp_data->steps[pix_a];
        int steps_b = comp_data->steps[pix_b];
        if (steps_a != steps_b) {
                return steps_a - steps_b;
        }
        return (int) pix_a - (int) pix_b;
}
static QSORT_S_COMP_DEFINE(lavc_compare_convs, a, b, comp_data_v) {
        enum AVPixelFormat pix_a = *(const enum AVPixelFormat *) a;
        enum AVPixelFormat pix_b = *(const enum AVPixelFormat *) b;
        int ret = lavc_compare_convs_inner(pix_a, pix_b, comp_data_v);
        log_msg(LOG_LEVEL_DEBUG2, MOD_NAME "%s %c %s\n", av_get_pix_fmt_name(pix_a), ret == 0 ? '=' : ret < 0 ? '<' : '>', av_get_pix_fmt_name(pix_b));
        return ret;
}

static inline _Bool filter(const struct to_lavc_req_prop *req_prop, codec_t uv_format, enum AVPixelFormat avpixfmt) {
        if (req_prop->force_conv_to != VIDEO_CODEC_NONE && req_prop->force_conv_to != uv_format) {
                return 0;
        }
        if (req_prop->subsampling != 0 && req_prop->subsampling != av_pixfmt_get_subsampling(avpixfmt)) {
                return 0;
        }
        const struct AVPixFmtDescriptor *pd = av_pix_fmt_desc_get(avpixfmt);
        if (req_prop->depth != 0 && req_prop->depth != pd->comp[0].depth) {
                return 0;
        }
        if (req_prop->rgb != -1 && req_prop->rgb != ((pd->flags & AV_PIX_FMT_FLAG_RGB) != 0U ? 1 : 0)) {
                return 0;
        }
        return 1;
}

/**
 * sets fmt_set[avformat]=1 for every AVPixelFormat that can be converted to
 * @param[out] comp_data conversion properties (basically of AVPixelFormat, but
 * if there is a conversion over UG codec, the lowest spec is used)
 */
static void
set_convertible_formats(codec_t in_codec, struct to_lavc_req_prop req_prop,
                        int fmt_set[static AV_PIX_FMT_NB],
                        struct lavc_compare_convs_data *comp_data)
{
        for (const struct uv_to_av_pixfmt *i = get_av_to_ug_pixfmts(); i->uv_codec != VIDEO_CODEC_NONE; ++i) { // no AV conversion needed, only UV pixfmt change
                if (!get_decoder_from_to(in_codec, i->uv_codec)) {
                        continue;
                }
                if (!filter(&req_prop, i->uv_codec, i->av_pixfmt)) {
                        continue;
                }
                fmt_set[i->av_pixfmt]   = 1;
                struct pixfmt_desc desc = av_pixfmt_get_desc(i->av_pixfmt);
                log_msg(
                    LOG_LEVEL_DEBUG2,
                    MOD_NAME
                    "conversion ->%s prop:\t%2d b, subsampling %d, RGB: %d\n",
                    get_codec_name(i->uv_codec), desc.depth, desc.subsampling,
                    desc.rgb);
                comp_data->descs[i->av_pixfmt] = desc;
                comp_data->steps[i->av_pixfmt] = 1;
        }
        for (const struct uv_to_av_conversion *c = get_uv_to_av_conversions();
             c->src != VIDEO_CODEC_NONE;
             c++) { // AV conv needed (with possible UV pixfmt change)
                if (c->src != in_codec &&
                    !get_decoder_from_to(in_codec, c->src)) {
                        continue;
                }
                if (!filter(&req_prop, c->src, c->dst)) {
                        continue;
                }
                fmt_set[c->dst]             = 1;
                struct pixfmt_desc desc_src = get_pixfmt_desc(in_codec);
                struct pixfmt_desc desc_uv  = get_pixfmt_desc(c->src);
                struct pixfmt_desc desc_av  = av_pixfmt_get_desc(c->dst);
                struct pixfmt_desc desc     = {
                            .depth = MIN(desc_av.depth, desc_uv.depth),
                            .subsampling =
                            MIN(desc_av.subsampling, desc_uv.subsampling),
                            .rgb = desc_av.rgb == desc_uv.rgb ? desc_av.rgb
                                                              : !desc_src.rgb
                };
                if (compare_pixdesc(&desc, &comp_data->descs[c->dst],
                                    &desc_src) <
                    0) { // override only with better
                        log_msg(LOG_LEVEL_DEBUG2,
                                MOD_NAME "conversion %s->%s prop:\t%2d b, "
                                         "subsampling %d, RGB: %d\n",
                                get_codec_name(c->src),
                                av_get_pix_fmt_name(c->dst), desc.depth,
                                desc.subsampling, desc.rgb);
                        comp_data->descs[c->dst] = desc;
                        comp_data->steps[c->dst] =
                            get_decoder_from_to(in_codec, c->src) == vc_memcpy
                                ? 1
                                : 2;
                }
        }
}

static void
set_convertible_formats_cuda(codec_t in_codec, struct to_lavc_req_prop req_prop,
                             int fmt_set[static AV_PIX_FMT_NB],
                             struct lavc_compare_convs_data *comp_data)
{
        for (unsigned i = 0; i < sizeof to_lavc_cuda_supp_formats /
                                     sizeof to_lavc_cuda_supp_formats[0];
             i++) {
                const enum AVPixelFormat f = to_lavc_cuda_supp_formats[i];
                if (!filter(&req_prop, in_codec, f)) {
                        continue;
                }
                fmt_set[f]                    = 1;
                const struct pixfmt_desc desc = av_pixfmt_get_desc(f);
                MSG(DEBUG2, "Adding CUDA-convertible format %s\n",
                    av_get_pix_fmt_name(f));
                comp_data->descs[f] = desc;
                comp_data->steps[f] = 1;
        }
}

/// @todo TOREMOVE after cuda conversions implemented
static bool
to_lavc_cuda_conv_enabled()
{
        if (!cuda_devices_explicit) {
                return false;
        }
        struct to_lavc_vid_conv_cuda *s =
            to_lavc_vid_conv_cuda_init(UYVY, 1, 1, AV_PIX_FMT_YUV444P);
        if (s == NULL) {
                return false;
        }
        to_lavc_vid_conv_cuda_destroy(&s);
        return true;
}

/**
 * Returns list of pix_fmts that UltraGrid can supply to the encoder.
 * The list is ordered according to input description and requested subsampling.
 *
 * If uv->uv->av conversion is performed, worst parameter in chain is taken
 * (eg. for v210->UYVY->10b the bit depth is 8 even though on both ends are 10).
 */
int get_available_pix_fmts(codec_t in_codec, struct to_lavc_req_prop req_prop,
                enum AVPixelFormat fmts[AV_PIX_FMT_NB])
{
        int nb_fmts = 0;
        // add the format itself if it matches the ultragrid one
        enum AVPixelFormat mapped_av_fmt = get_ug_to_av_pixfmt(in_codec);
        if (mapped_av_fmt != AV_PIX_FMT_NONE) {
                if (filter(&req_prop, in_codec, mapped_av_fmt)) {
                        fmts[nb_fmts++] = mapped_av_fmt;
                }
        }

        int sort_start_idx = nb_fmts;
        int fmt_set[AV_PIX_FMT_NB] = { 0 }; // to avoid multiple occurrences; for every added element, comp_data must be also set
        struct lavc_compare_convs_data comp_data = { 0 };
        if (to_lavc_cuda_conv_enabled()) {
                set_convertible_formats_cuda(in_codec, req_prop, fmt_set,
                                        &comp_data);
        } else {
                set_convertible_formats(in_codec, req_prop, fmt_set,
                                        &comp_data);
        }
        for (int i = 0; i < AV_PIX_FMT_NB; ++i) {
                if (fmt_set[i]) {
                        fmts[nb_fmts++] = (enum AVPixelFormat) i;
                }
        }

        comp_data.src_desc = get_pixfmt_desc(in_codec);
        qsort_s(fmts + sort_start_idx, nb_fmts - sort_start_idx, sizeof fmts[0], lavc_compare_convs, &comp_data);

#ifdef HWACC_VAAPI
        fmts[nb_fmts++] = AV_PIX_FMT_VAAPI;
#endif
#ifdef HWACC_VULKAN
        fmts[nb_fmts++] = AV_PIX_FMT_VULKAN;
#endif

        return nb_fmts;
}

struct to_lavc_vid_conv {
        struct AVFrame     *out_frame;
        struct AVFrame    **out_frame_parts; ///< out_frame slices for parallel conversion
        int                 thread_count;
        struct AVFrame     *tmp_frame; ///< dummy input buffer pointers' wrapper
        codec_t             in_pixfmt;
        unsigned char      *decoded; ///< intermediate buffer if uv pixfmt conversions needed
        codec_t             decoded_codec;
        decoder_t           decoder;
        pixfmt_callback_t   pixfmt_conv_callback;

        struct to_lavc_vid_conv_cuda *cuda_conv_state;
};

static void to_lavc_memcpy_data(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height)
{
        struct to_lavc_vid_conv *s = out_frame->opaque;
        size_t linesize = vc_get_linesize(width, s->decoded_codec);
        size_t linelength = vc_get_size(width, s->decoded_codec);
        for (int comp = 0; comp < AV_NUM_DATA_POINTERS; ++comp) {
                if (out_frame->data[comp] == NULL) {
                        break;
                }
                for (ptrdiff_t y = 0; y < height; ++y) {
                        memcpy(out_frame->data[comp] + y * out_frame->linesize[comp], in_data + y * linesize,
                                        linelength);
                }
        }
}

struct to_lavc_vid_conv *to_lavc_vid_conv_init(codec_t in_pixfmt, int width, int height, enum AVPixelFormat out_pixfmt, int thread_count) {
        assert(thread_count > 0);
        int ret = 0;
        struct to_lavc_vid_conv *s = (struct to_lavc_vid_conv *) calloc(1, sizeof *s);
        s->in_pixfmt = in_pixfmt;
        s->thread_count = thread_count;
        s->out_frame_parts = (struct AVFrame **) calloc(thread_count, sizeof *s->out_frame_parts);
        for (int i = 0; i < thread_count; i++) {
                s->out_frame_parts[i] = av_frame_alloc();
        }
        s->out_frame = av_frame_alloc();
        s->tmp_frame = av_frame_alloc();
        if (!s->out_frame || !s->tmp_frame) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate video frame\n");
                to_lavc_vid_conv_destroy(&s);
                return NULL;
        }
        s->out_frame->pts = -1;
        s->tmp_frame->format = s->out_frame->format = out_pixfmt;
        s->tmp_frame->width = s->out_frame->width = width;
        s->tmp_frame->height = s->out_frame->height = height;
        get_av_pixfmt_details(out_pixfmt, &s->out_frame->colorspace,
                              &s->out_frame->color_range);
        av_frame_copy_props(s->tmp_frame, s->out_frame);
        s->out_frame->opaque = s;

        ret = av_frame_get_buffer(s->out_frame, 0);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate raw picture buffer\n");
                to_lavc_vid_conv_destroy(&s);
                return NULL;
        }

        if (get_ug_to_av_pixfmt(in_pixfmt) != AV_PIX_FMT_NONE
                        && out_pixfmt == get_ug_to_av_pixfmt(in_pixfmt)) {
                s->decoded_codec = in_pixfmt;
                s->decoder = vc_memcpy;
        } else {
                if (to_lavc_cuda_conv_enabled()) {
                        s->cuda_conv_state = to_lavc_vid_conv_cuda_init(
                            in_pixfmt, width, height, out_pixfmt);
                        if (s->cuda_conv_state != NULL) {
                                MSG(NOTICE, "Using CUDA FFmpeg conversions.\n");
                                return s;
                        }
                        MSG(ERROR, "Unable to initialize CUDA conv state!\n");
                }
                s->decoder = get_decoder_from_uv_to_uv(in_pixfmt, out_pixfmt, &s->decoded_codec);
                if (s->decoder == NULL) {
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Failed to find a way to convert %s to %s\n",
                                        get_codec_name(in_pixfmt), av_get_pix_fmt_name(out_pixfmt));
                        to_lavc_vid_conv_destroy(&s);
                        return NULL;
                }
        }
        verbose_msg(MOD_NAME "converting %s to %s over %s\n", get_codec_name(in_pixfmt),
                        av_get_pix_fmt_name(out_pixfmt), get_codec_name(s->decoded_codec));
        watch_pixfmt_degrade(MOD_NAME, get_pixfmt_desc(in_pixfmt), get_pixfmt_desc(s->decoded_codec));
        watch_pixfmt_degrade(MOD_NAME, get_pixfmt_desc(s->decoded_codec), av_pixfmt_get_desc(out_pixfmt));
        s->decoded = (unsigned char *) malloc(
            (long) vc_get_linesize(width, s->decoded_codec) * height +
            MAX_PADDING);

        s->pixfmt_conv_callback = select_pixfmt_callback(out_pixfmt, s->decoded_codec);

        return s;
};

// frame has some linesizes as mapped ultragrid equivalent pixfmt
static bool same_linesizes(codec_t codec, AVFrame *frame)
{
        if (!codec_is_planar(codec)) {
                return vc_get_linesize(frame->width, codec) == frame->linesize[0];
        }
        assert(get_bits_per_component(codec) == 8);
        int sub[8];
        codec_get_planes_subsampling(codec, sub);
        for (ptrdiff_t i = 0; i < 4; ++i) {
                if (sub[2 * i] == 0) {
                        return true;
                }
                if (frame->linesize[i] != (frame->width + sub[2 * i] - 1) / sub[2 * i]) {
                        return false;
                }
        }
        return true;
}

struct pixfmt_conv_task_data {
        pixfmt_callback_t callback;
        AVFrame *out_frame;
        const unsigned char *in_data;
        int width;
        int height;
};

static void *pixfmt_conv_task(void *arg) {
        struct pixfmt_conv_task_data *data = (struct pixfmt_conv_task_data *) arg;
        data->callback(data->out_frame, data->in_data, data->width, data->height);
        return NULL;
}

static void
set_worker_subframes(struct to_lavc_vid_conv *s)
{
        for (ptrdiff_t i = 0; i < s->thread_count; ++i) {
                int chunk_size = (s->out_frame->height / s->thread_count) & ~1;
                s->out_frame_parts[i]->data[0] =
                    s->out_frame->data[0] +
                    i * s->out_frame->linesize[0] * chunk_size;

                if (av_pix_fmt_desc_get(s->out_frame->format)->log2_chroma_h ==
                    1) { // eg. 4:2:0
                        chunk_size /= 2;
                }
                s->out_frame_parts[i]->data[1] =
                    s->out_frame->data[1] +
                    i * s->out_frame->linesize[1] * chunk_size;
                s->out_frame_parts[i]->data[2] =
                    s->out_frame->data[2] +
                    i * s->out_frame->linesize[2] * chunk_size;
                s->out_frame_parts[i]->linesize[0] = s->out_frame->linesize[0];
                s->out_frame_parts[i]->linesize[1] = s->out_frame->linesize[1];
                s->out_frame_parts[i]->linesize[2] = s->out_frame->linesize[2];
                s->out_frame_parts[i]->opaque      = s->out_frame->opaque;
        }
}

/// @return AVFrame with converted data (if needed); valid until next to_lavc_vid_conv()
///         call or to_lavc_vid_conv_destroy()
struct AVFrame *to_lavc_vid_conv(struct to_lavc_vid_conv *s, char *in_data) {
        if (s->cuda_conv_state != NULL) {
                return to_lavc_vid_conv_cuda(s->cuda_conv_state, in_data);
        }
        int ret = 0;
        unsigned char *decoded = NULL;
        if ((ret = av_frame_make_writable(s->out_frame)) != 0) {
                print_libav_error(LOG_LEVEL_ERROR, MOD_NAME "Cannot make frame writable", ret);
                return NULL;
        }

        time_ns_t t0 = get_time_in_ns();
        if (s->decoder != vc_memcpy) {
                int src_linesize = vc_get_linesize(s->out_frame->width, s->in_pixfmt);
                int dst_linesize = vc_get_linesize(s->out_frame->width, s->decoded_codec);
                parallel_pix_conv(s->out_frame->height, (char *) s->decoded, dst_linesize, in_data, src_linesize, s->decoder, s->thread_count);
                decoded = s->decoded;
        } else {
                decoded = (unsigned char *) in_data;
        }

        time_ns_t t1 = get_time_in_ns();
        AVFrame *frame = s->out_frame;
        if (s->pixfmt_conv_callback != NULL) {
                set_worker_subframes(s);
                struct pixfmt_conv_task_data data[s->thread_count];
                for(int i = 0; i < s->thread_count; ++i) {
                        data[i].callback = s->pixfmt_conv_callback;
                        data[i].out_frame = s->out_frame_parts[i];

                        size_t height = s->out_frame->height / s->thread_count & ~1; // height needs to be even
                        if (i < s->thread_count - 1) {
                                data[i].height = height;
                        } else { // we are last so we need to do the rest
                                data[i].height = s->out_frame->height -
                                        height * (s->thread_count - 1);
                        }
                        data[i].width = s->out_frame->width;
                        data[i].in_data = decoded + i * height *
                                vc_get_linesize(s->out_frame->width, s->decoded_codec);
                }
                task_run_parallel(pixfmt_conv_task, s->thread_count, data, sizeof data[0], NULL);
        } else { // no pixel format conversion needed
                if (codec_is_planar(s->decoded_codec) && !same_linesizes(s->decoded_codec, s->out_frame)) {
                        assert(get_bits_per_component(s->decoded_codec) == 8);
                        int sub[8];
                        codec_get_planes_subsampling(s->decoded_codec, sub);
                        const unsigned char *in = decoded;
                        for (ptrdiff_t i = 0; i < 4; ++i) {
                                if (sub[2 * i] == 0) {
                                        break;
                                }
                                int linesize = (s->out_frame->width + sub[2 * i] - 1) / sub[2 * i];
                                int lines = (s->out_frame->height + sub[2 * i + 1] - 1) / sub[2 * i + 1];
                                for (ptrdiff_t y = 0; y < lines; ++y) {
                                        memcpy(s->out_frame->data[i] + y * s->out_frame->linesize[i], in, linesize);
                                        in += linesize;
                                }
                        }
                } else { // just set pointers to input buffer
                        frame = s->tmp_frame;
                        memcpy(frame->linesize, s->out_frame->linesize, sizeof frame->linesize);
                        if (codec_is_planar(s->decoded_codec)) {
                                buf_get_planes(s->out_frame->width, s->out_frame->height, s->decoded_codec, (char *) decoded, (char **) frame->data);
                        } else {
                                frame->data[0] = (uint8_t *) decoded;
                        }
                }
        }
        time_ns_t t2 = get_time_in_ns();
        MSG(DEBUG2,
            "duration uv pixfmt change: %f ms, av format change: %f ms\n",
            NS_TO_MS((double) (t1 - t0)), NS_TO_MS((double) (t2 - t1)));
        return frame;
};

void to_lavc_vid_conv_destroy(struct to_lavc_vid_conv **s_p) {
        struct to_lavc_vid_conv *s = *s_p;
        if (s == NULL) {
                return;
        }
        for (int i = 0; i < s->thread_count; i++) {
                av_frame_free(&s->out_frame_parts[i]);
        }
        free(s->out_frame_parts);
        av_frame_free(&s->out_frame);
        av_frame_free(&s->tmp_frame);
        free(s->decoded);
        free(s);
        *s_p = NULL;
}

#pragma GCC diagnostic pop

/* vi: set expandtab sw=8: */
