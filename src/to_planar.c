/**
 * @file   to_planar.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2026 CESNET, zájmové sdružení právnických osob
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

#include "to_planar.h"

#include <assert.h>        // for assert
#include <endian.h>        // for BIG_ENDIAN, BYTE_ORDER
#include <stdint.h>        // for uint16_t, uint32_t, uintptr_t, uint8_t
#include <stdlib.h>        // for size_t, free, malloc, NULL
#include <string.h>        // for memcpy

#ifdef __SSE3__
#include <emmintrin.h>     // for __m128i, _mm_and_si128, _mm_andnot_si128
#include <pmmintrin.h>     // for _mm_lddqu_si128
#endif

#include "types.h"         // for depth, v210, R12L, RGBA, UYVY, Y216
#include "utils/macros.h"  // for OPTIMIZED_FOR, ALWAYS_INLINE
#include "video_codec.h"   // for vc_get_linesize

#if BYTE_ORDER == BIG_ENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

/**
 * converts v210 to P010 - 2-plane 10-bit YCbCr 4:2;0 with U/V combined (samples are
 * stored in MSB of 16b word)
 *
 * neither input nor output need to be padded
 */
void
v210_to_p010le(unsigned char *__restrict *__restrict out_data,
               const int *__restrict out_linesize,
               const unsigned char *__restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert(out_linesize[0] % 2 == 0);
        assert(out_linesize[1] % 2 == 0);

        void *garbage = NULL;

        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                const uint32_t *src = (const uint32_t *)(const void *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                const uint32_t *src2 = (const uint32_t *)(const void *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_data[0] + out_linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *)(void *) (out_data[0] + out_linesize[0] * (y + 1));
                uint16_t *dst_cbcr = (uint16_t *)(void *) (out_data[1] + out_linesize[1] * y / 2);

                // handle height % 2 == 1 (last line)
                if (height - y == 1) {
                        dst_y2 = garbage = malloc(out_linesize[0]);
                        src2 = src;
                }
                // handle margin of width % 6 != 0
                int w = (width + 5) / 6 * 6;
                if (height - y == 1 || height - y == 2) {
                        w = width;
                }

                OPTIMIZED_FOR (int x = 0; x < w / 6; ++x) {
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

                // handle margin of width % 6 != 0 - just copy last at most 5
                // pix from above for simplicity (1 or 2 lines)
                if ((height - y == 1 || height - y == 2) && width % 6 != 0) {
                        const size_t pix_cnt = width % 6;
                        memcpy(dst_y, dst_y - out_linesize[0], pix_cnt * 2);
                        if (height - y == 2) {
                                memcpy(dst_y2, dst_y - out_linesize[0],
                                       pix_cnt * 2);
                        }
                        memcpy(dst_cbcr, dst_cbcr - out_linesize[1], pix_cnt * 2);
                }
        }

        free(garbage);
}

/**
 * converts Y216 to P010 - 2-plane 10-bit YCbCr 4:2;0 with U/V combined like nv12
 * (samples are stored in MSB of 16b word)
 *
 * @todo
 * currently the choma from every second line is taken (not averaged)
 */
void
y216_to_p010le(unsigned char *__restrict *__restrict out_data,
               const int *__restrict out_linesize,
               const unsigned char *__restrict in_data, int width, int height)
{
        const size_t src_linesize = vc_get_linesize(width, Y216);
        for (int i = 0; i < (height + 1) / 2; ++i) {
                const uint16_t *in =
                    (const void *) (in_data + ((size_t) 2 * i * src_linesize));
                uint16_t *out_y =
                    (void *) (out_data[0] + ((size_t) 2 * i * out_linesize[0]));
                uint16_t *out_chr =
                    (void *) (out_data[1] + ((size_t) i * out_linesize[1]));
                int j = 0;
                for ( ; j < width / 2; ++j) {
                        *out_y++   = *in++; // Y1
                        *out_chr++ = *in++; // Cb
                        *out_y++   = *in++; // Y2
                        *out_chr++ = *in++; // Cr
                }
                if (width % 2 == 1) {
                        *out_y++   = *in++; // Y1
                        *out_chr++ = *in++; // Cb
                        in++; // drop Y2
                        *out_chr++ = *in++; // Cr
                }
                if (2 * i + 1 < height) {
                        for (j = 0; j < width / 2; ++j) {
                                *out_y++ = *in++; // Y1
                                in++;             // Cb
                                *out_y++ = *in++; // Y2
                                in++;             // Cr
                        }
                        if (width % 2 == 1) {
                                *out_y++   = *in++; // Y1
                        }
                }
        }
}

/**
 * @todo
 * write a unit test but may be ok (irregular size handling code copied from
 * uyvy_to_i420 that is testec)
 */
void
uyvy_to_nv12(unsigned char *__restrict *__restrict out_data,
             const int *__restrict out_linesize,
             const unsigned char *__restrict in_data, int width, int height)
{
        for (size_t y = 0; y < (size_t) height; y += 2) {
                /*  every even row */
                const unsigned char *src = in_data + (y * ((size_t) width * 2));
                /*  every odd row */
                const unsigned char *src2 = src + ((size_t) width * 2);
                unsigned char *dst_y      = out_data[0] + (out_linesize[0] * y);
                unsigned char *dst_y2     = dst_y + out_linesize[0];
                unsigned char *dst_cbcr =
                    out_data[1] + (out_linesize[1] * (y / 2));

                if ((int) y == height - 1) { // last line when height % 2 != 0
                        src2 = src;
                        dst_y2 = dst_y;
                }

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

                if (width % 2 == 1) { // last row - do not process 2nd Y
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                }
        }
}

void
rgba_to_bgra(unsigned char *__restrict *__restrict out_data,
             const int *__restrict out_linesize,
             const unsigned char *__restrict in_data, int width, int height)
{
        const size_t src_linesize = vc_get_linesize(width, RGBA);
        for (size_t i = 0; i < (size_t) height; ++i) {
                const uint8_t *in  = in_data + (i * src_linesize);
                uint8_t       *out = out_data[0] + (i * out_linesize[0]);
                for (int i = 0; i < width; ++i) {
                        *out++ = in[2]; // B
                        *out++ = in[1]; // G
                        *out++ = in[0]; // R
                        *out++ = in[3]; // A
                        in += 4;
                }
        }
}

/**
 * converts UYVY to planar YUV 4:2:0
 *
 * @sa uyvy_to_i422
 */
void
uyvy_to_i420(unsigned char *__restrict *__restrict out_data,
             const int *__restrict out_linesize, const unsigned char *__restrict in_data,
             int width, int height)
{
        size_t                     src_linesize = vc_get_linesize(width, UYVY);
        for (size_t i = 0; i < (size_t) (height + 1) / 2; ++i) {
                const unsigned char *in1 = in_data + (2 * i * src_linesize);
                const unsigned char *in2 = in1 + src_linesize;
                unsigned char       *y1 =
                    out_data[0] + ((2ULL * i) * out_linesize[0]);
                unsigned char *y2 = y1 + out_linesize[0];
                unsigned char *u  = out_data[1] + (i * out_linesize[1]);
                unsigned char *v  = out_data[2] + (i * out_linesize[2]);

                // handle height % 2 == 1
                if (2 * i + 1 == (size_t) height) {
                        y2  = y1;
                        in2 = in1;
                }

                int j = 0;
                for (; j < width / 2; ++j) {
                        *u++  = (*in1++ + *in2++ + 1) / 2;
                        *y1++ = *in1++;
                        *y2++ = *in2++;
                        *v++  = (*in1++ + *in2++ + 1) / 2;
                        *y1++ = *in1++;
                        *y2++ = *in2++;
                }
                if (width % 2 == 1) { // do not overwrite EOL
                        *u++  = (*in1++ + *in2++ + 1) / 2;
                        *y1++ = *in1++;
                        *y2++ = *in2++;
                        *v++  = (*in1++ + *in2++ + 1) / 2;
                }
        }
}

/// @note out_depth needs to be at least 12
ALWAYS_INLINE static inline void
r12l_to_gbrpXXle(unsigned char *__restrict *__restrict out_data,
                 const int *__restrict out_linesize,
                 const unsigned char *__restrict in_data, int width, int height,
                 unsigned int out_depth, int rind, int gind, int bind)
{
        assert(out_depth >= 12);
        assert((uintptr_t) out_linesize[0] % 2 == 0);
        assert((uintptr_t) out_linesize[1] % 2 == 0);
        assert((uintptr_t) out_linesize[2] % 2 == 0);

#define S(x) ((x) << (out_depth - 12U))

        int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                const unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_r = (uint16_t *)(void *) (out_data[rind] + out_linesize[rind] * y);
                uint16_t *dst_g = (uint16_t *)(void *) (out_data[gind] + out_linesize[gind] * y);
                uint16_t *dst_b = (uint16_t *)(void *) (out_data[bind] + out_linesize[bind] * y);

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
#undef S
}

void
r12l_to_gbrp12le(unsigned char *__restrict *__restrict out_data,
                 const int *__restrict out_linesize,
                 const unsigned char *__restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_data, out_linesize, in_data, width, height,
                         DEPTH12, 2, 0, 1);
}

void
r12l_to_gbrp16le(unsigned char *__restrict *__restrict out_data,
                 const int *__restrict out_linesize,
                 const unsigned char *__restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_data, out_linesize, in_data, width, height,
                         DEPTH16, 2, 0, 1);
}

void
r12l_to_rgbp12le(unsigned char *__restrict *__restrict out_data,
                 const int *__restrict out_linesize,
                 const unsigned char *__restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_data, out_linesize, in_data, width, height,
                         DEPTH12, 0, 1, 2);
}
