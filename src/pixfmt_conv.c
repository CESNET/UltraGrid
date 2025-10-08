/**
 * @file   pixfmt_conv.c
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @brief the file contains conversions between UG pixel formats
 *
 * To measure performance of conversions, use `tools/convert benchmark`.
 */
/* Copyright (c) 2005-2025 CESNET
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#define __STDC_WANT_LIB_EXT1__ 1

#include "pixfmt_conv.h"

#include <assert.h>          // for assert
#include <stddef.h>          // for NULL, size_t
#include <stdint.h>          // for uint32_t, uintptr_t, uint16_t, uint64_t
#include <string.h>          // for memcpy

#include "color_space.h"
#include "compat/qsort_s.h"
#include "debug.h"
#include "utils/macros.h" // to_fourcc, OPTIMEZED_FOR, CLAMP
#include "video_codec.h"

#ifdef __SSE3__
#include "pmmintrin.h"
#endif
#ifdef __SSSE3__
#include "tmmintrin.h"
#endif

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

#define MOD_NAME "[pixfmt_conv] "

/**
 * @brief Converts v210 to UYVY
 * @param[out] dst     4-byte aligned output buffer where UYVY will be stored
 * @param[in]  src     4-byte aligned input buffer containing v210 (by definition of v210
 *                     should be even aligned to 16B boundary)
 * @param[in]  dst_len length of data that should be written to dst buffer (in bytes)
 */
static void vc_copylinev210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        struct {
                unsigned a:10;
                unsigned b:10;
                unsigned c:10;
                unsigned p1:2;
        } const *s;
        register uint32_t *d;
        register uint32_t tmp;

        d = (uint32_t *)(void *) dst;
        s = (const void *)src;

        while (dst_len >= 12) {
                tmp = (s->a >> 2) | (s->b >> 2) << 8 | (((s)->c >> 2) << 16);
                s++;
                *(d++) = tmp | ((s->a >> 2) << 24);
                tmp = (s->b >> 2) | (((s)->c >> 2) << 8);
                s++;
                *(d++) = tmp | ((s->a >> 2) << 16) | ((s->b >> 2) << 24);
                tmp = (s->c >> 2);
                s++;
                *(d++) =
                    tmp | ((s->a >> 2) << 8) | ((s->b >> 2) << 16) |
                    ((s->c >> 2) << 24);
                s++;

                dst_len -= 12;
        }
        if (dst_len >= 4) {
                tmp = (s->a >> 2) | (s->b >> 2) << 8 | (((s)->c >> 2) << 16);
                s++;
                *(d++) = tmp | ((s->a >> 2) << 24);
        }
        if (dst_len >= 8) {
                tmp = (s->b >> 2) | (((s)->c >> 2) << 8);
                s++;
                *(d++) = tmp | ((s->a >> 2) << 16) | ((s->b >> 2) << 24);
        }
}

/**
 * @brief Converts from YUYV to UYVY.
 * @copydetails vc_copylinev210
 */
static void vc_copylineYUYV(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
#if defined __SSE2__
        register uint32_t *d;
        register const uint32_t *s;
        const uint32_t * const end = (uint32_t *)(void *) dst + dst_len / 4;

        uint32_t mask[4] = {
                0xff00ff00ul,
                0xff00ff00ul,
                0xff00ff00ul,
                0xff00ff00ul};

        d = (uint32_t *)(void *) dst;
        s = (const uint32_t *)(const void *) src;

        assert(dst_len % 4 == 0);

        if((dst_len % 16 == 0)) {
                asm("movdqu (%0), %%xmm4\n"
                                "movdqa %%xmm4, %%xmm5\n"
                                "psrldq $1, %%xmm5\n"
                                : :"r"(mask));
                while(d < end) {
                        asm volatile ("movdqu (%0), %%xmm0\n"
                                        "movdqu %%xmm0, %%xmm1\n"
                                        "pand %%xmm4, %%xmm0\n"
                                        "psrldq $1, %%xmm0\n"
                                        "pand %%xmm5, %%xmm1\n"
                                        "pslldq $1, %%xmm1\n"
                                        "por %%xmm0, %%xmm1\n"
                                        "movdqu %%xmm1, (%1)\n"::"r" (s), "r"(d));
                        s += 4;
                        d += 4;
                }
        } else {
                while(d < end) {
                        register uint32_t tmp = *s;
                        *d = ((tmp & 0x00ff0000) << 8) | ((tmp & 0xff000000) >> 8) |
                                ((tmp & 0x000000ff) << 8) | ((tmp & 0x0000ff00) >> 8);
                        s++;
                        d++;

                }
        }
#else
	char u, y1, v, y2;
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
		y1 = *src++;
		u = *src++;
		y2 = *src++;
		v = *src++;
		*dst++ = u;
		*dst++ = y1;
		*dst++ = v;
		*dst++ = y2;
	}
#endif
}

/**
 * @brief Converts from R10k to RGBA
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     4B-aligned buffer containing pixels in R10k
 * @param[in]  dst_len length of data that should be written to dst buffer (in bytes)
 * @param[in]  rshift  destination red shift
 * @param[in]  gshift  destination green shift
 * @param[in]  bshift  destination blue shift
 */
static void
vc_copyliner10k(unsigned char * __restrict dst, const unsigned char * __restrict src, int len, int rshift,
                int gshift, int bshift)
{
        struct {
                unsigned r:8;

                unsigned gh:6;
                unsigned p1:2;

                unsigned bh:4;
                unsigned p2:2;
                unsigned gl:2;

                unsigned p3:2;
                unsigned p4:2;
                unsigned bl:4;
        } const *s;
        register uint32_t *d;
        register uint32_t tmp;
        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);

        d = (uint32_t *)(void *) dst;
        s = (const void *) src;

        while (len >= 16) {
                tmp =
                    (alpha_mask | s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                tmp =
                    (alpha_mask | s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                tmp =
                    (alpha_mask | s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                tmp =
                    (alpha_mask | s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                len -= 16;
        }
        while (len >= 4) {
                tmp =
                    (alpha_mask | s->
                     r << rshift) | (((s->gh << 2) | s->
                                      gl) << gshift) | (((s->bh << 4) | s->
                                                         bl) << bshift);
                s++;
                *(d++) = tmp;
                len -= 4;
        }
}

static void
vc_copyliner10ktoRG48(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        while  (dstlen > 0) {
                dst[1] = *src++; // Rhi
                unsigned int byte2 = *src++;
                unsigned int byte3 = *src++;
                unsigned int byte4 = *src++;
                dst[0] = byte2 & 0xC0U; // Rlo
                dst[3] = byte2 << 2U | byte3 >> 6U; // Ghi
                dst[2] = (byte3 & 0x30U) << 2U; // Glo
                dst[5] = (byte3 & 0xFU) << 4U | byte4 >> 4U; // Bhi
                dst[4] = (byte4 & 0xCU) << 4U; // Blo
                dst += 6;
                dstlen -= 6;
        }
}

static void vc_copyliner10ktoY416(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        enum {
                D_DEPTH = 16,
        };
        assert((uintptr_t) dst % 2 == 0);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, D_DEPTH);
        uint16_t *d = (void *) dst;
        OPTIMIZED_FOR (int x = 0; x < dstlen; x += 8) {
                unsigned int byte1 = *src++;
                unsigned int byte2 = *src++;
                unsigned int byte3 = *src++;
                unsigned int byte4 = *src++;
                comp_type_t r, g, b;
                r = byte1 << 8U | (byte2 & 0xC0U);
                g = (byte2 & 0x3FU) << 10U | (byte3 & 0xF0U) << 2U;
                b = (byte3 & 0xFU) << 12U | (byte4 & 0xFCU) << 4U;
                comp_type_t u =
                    (RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) +
                    (1 << (D_DEPTH - 1));
                *d++ = CLAMP_LIMITED_CBCR(u, D_DEPTH);
                comp_type_t y =
                    (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) +
                    (1 << (D_DEPTH - 4));
                *d++ = CLAMP_LIMITED_Y(y, D_DEPTH);
                comp_type_t v =
                    (RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) +
                    (1 << (D_DEPTH - 1));
                *d++ = CLAMP_LIMITED_CBCR(v, D_DEPTH);
                *d++ = 0xFFFFU;
        }
}

static void vc_copyliner10ktoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        OPTIMIZED_FOR (int x = 0; x < dstlen; x += 3) {
                *(dst++) = src[0];
                *(dst++) = src[1] << 2 | src[2] >> 6;
                *(dst++) = src[2] << 4 | src[3] >> 4;
                src += 4;
        }
}

/**
 * @brief Converts from R12L to RGB
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     buffer containing pixels in R12L
 * @param[in]  dst_len length of data that should be written to dst buffer (in bytes)
 * @param[in]  rshift  ignored
 * @param[in]  gshift  ignored
 * @param[in]  bshift  ignored
 */
static void
vc_copylineR12LtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dstlen - 24; x += 24) {
                uint8_t tmp;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                *dst++ = tmp; // r0
                *dst++ = src[BYTE_SWAP(2)]; // g0
                tmp = src[BYTE_SWAP(3)]>> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                *dst++ = tmp; // b0
                *dst++ = src[BYTE_SWAP(1)]; // r1
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                *dst++ = tmp; // g1
                *dst++ = src[BYTE_SWAP(0)]; // b1
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                *dst++ = tmp; // r2
                *dst++ = src[BYTE_SWAP(3)]; // g2
                src += 4;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                *dst++ = tmp; // b2
                *dst++ = src[BYTE_SWAP(2)]; // r3
                tmp = src[BYTE_SWAP(3)] >> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                *dst++ = tmp; // g3
                *dst++ = src[BYTE_SWAP(1)]; // b3
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                *dst++ = tmp; // r4
                *dst++ = src[BYTE_SWAP(0)]; // g4
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                *dst++ = tmp; // b4
                *dst++ = src[BYTE_SWAP(3)]; // r5
                src += 4;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                *dst++ = tmp; // g5
                *dst++ = src[BYTE_SWAP(2)]; // b5
                tmp = src[BYTE_SWAP(3)] >> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                *dst++ = tmp; // r6
                *dst++ = src[BYTE_SWAP(1)]; // g6
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                *dst++ = tmp; // b6
                *dst++ = src[BYTE_SWAP(0)]; // r7
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                *dst++ = tmp; // g7
                *dst++ = src[BYTE_SWAP(3)]; // b7
                src += 4;
        }
}

/**
 * @brief Converts from R12L to RGBA
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     buffer containing pixels in R12L
 * @param[in]  dstlen  length of data that should be written to dst buffer (in bytes)
 * @param[in]  rshift  destination red shift
 * @param[in]  gshift  destination green shift
 * @param[in]  bshift  destination blue shift
 */
static void
vc_copylineR12L(unsigned char *dst, const unsigned char *src, int dstlen, int rshift,
                int gshift, int bshift)
{
        assert((uintptr_t) dst % sizeof(uint32_t) == 0);
        uint32_t *d = (uint32_t *)(void *) dst;
        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);

        OPTIMIZED_FOR (int x = 0; x <= dstlen - 32; x += 32) {
                uint8_t tmp;
                uint8_t r, g, b;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                r = tmp; // r0
                g = src[BYTE_SWAP(2)]; // g0
                tmp = src[BYTE_SWAP(3)]>> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                b = tmp; // b0
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                r = src[BYTE_SWAP(1)]; // r1
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                g = tmp; // g1
                b = src[BYTE_SWAP(0)]; // b1
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                r = tmp; // r2
                g = src[BYTE_SWAP(3)]; // g2
                src += 4;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                b = tmp; // b2
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                r = src[BYTE_SWAP(2)]; // r3
                tmp = src[BYTE_SWAP(3)] >> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                g = tmp; // g3
                b = src[BYTE_SWAP(1)]; // b3
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                r = tmp; // r4
                g = src[BYTE_SWAP(0)]; // g4
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                b = tmp; // b4
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                r = src[BYTE_SWAP(3)]; // r5
                src += 4;
                tmp = src[BYTE_SWAP(0)] >> 4;
                tmp |= src[BYTE_SWAP(1)] << 4;
                g = tmp; // g5
                b = src[BYTE_SWAP(2)]; // b5
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                tmp = src[BYTE_SWAP(3)] >> 4;
                src += 4;
                tmp |= src[BYTE_SWAP(0)] << 4;
                r = tmp; // r6
                g = src[BYTE_SWAP(1)]; // g6
                tmp = src[BYTE_SWAP(2)] >> 4;
                tmp |= src[BYTE_SWAP(3)] << 4;
                src += 4;
                b = tmp; // b6
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                r = src[BYTE_SWAP(0)]; // r7
                tmp = src[BYTE_SWAP(1)] >> 4;
                tmp |= src[BYTE_SWAP(2)] << 4;
                g = tmp; // g7
                b = src[BYTE_SWAP(3)]; // b7
                src += 4;
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
        }
}

/**
 * @brief Changes color channels' order in RGBA
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     4B-aligned buffer containing pixels in RGBA
 * @param[in]  dst_len length of data that should be written to dst buffer (in bytes)
 * @param[in]  rshift  destination red shift
 * @param[in]  gshift  destination green shift
 * @param[in]  bshift  destination blue shift
 *
 * @note
 * alpha is set always to 0xFF (not preserved if different in source)
 */
void
vc_copylineRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int len, int rshift,
                int gshift, int bshift)
{
        register uint32_t *d = (uint32_t *)(void *) dst;
        register const uint32_t *s = (const uint32_t *)(const void *) src;
        register uint32_t tmp;

        if (rshift == 0 && gshift == 8 && bshift == 16) {
                memcpy(dst, src, len);
        } else {
                uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);
                while (len >= 16) {
                        register unsigned int r, g, b;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        len -= 16;
                }
                while (len >= 4) {
                        register unsigned int r, g, b;
                        tmp = *(s++);
                        r = tmp & 0xff;
                        g = (tmp >> 8) & 0xff;
                        b = (tmp >> 16) & 0xff;
                        tmp = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
                        *(d++) = tmp;
                        len -= 4;
                }
        }
}

/**
 * @brief Converts from DVS10 to v210
 * @copydetails vc_copylinev210
 */
static void vc_copylineDVS10toV210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        unsigned int *d;
        const unsigned int *s1;
        register unsigned int a,b;
        d = (unsigned int *)(void *) dst;
        s1 = (const unsigned int *)(const void *) src;

        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
                a = b = *s1++;
                b = ((b >> 24) * 0x00010101) & 0x00300c03;
                a <<= 2;
                b |= a & (0xff<<2);
                a <<= 2;
                b |= a & (0xff00<<4);
                a <<= 2;
                b |= a & (0xff0000<<6);
                *d++ = b;
        }
}

/* convert 10bits Cb Y Cr A Y Cb Y A to 8bits Cb Y Cr Y Cb Y */

/* TODO:
 * - undo it - currently this decoder is broken - but rather remove DVS10 entirely
 * - if not, replace !(__APPLE__ || HAVE_32B_LINUX) with something like __SSE2__
 */
#if 0 /* !(__APPLE__ || HAVE_32B_LINUX) */

void vc_copylineDVS10(unsigned char *dst, unsigned char *src, int src_len)
{
        register unsigned char *_d = dst, *_s = src;

        while (src_len > 0) {

 asm("movd %0, %%xmm4\n": :"r"(0xffffff));

                asm volatile ("movdqa (%0), %%xmm0\n"
                              "movdqa 16(%0), %%xmm5\n"
                              "movdqa %%xmm0, %%xmm1\n"
                              "movdqa %%xmm0, %%xmm2\n"
                              "movdqa %%xmm0, %%xmm3\n"
                              "pand  %%xmm4, %%xmm0\n"
                              "movdqa %%xmm5, %%xmm6\n"
                              "movdqa %%xmm5, %%xmm7\n"
                              "movdqa %%xmm5, %%xmm8\n"
                              "pand  %%xmm4, %%xmm5\n"
                              "pslldq $4, %%xmm4\n"
                              "pand  %%xmm4, %%xmm1\n"
                              "pand  %%xmm4, %%xmm6\n"
                              "pslldq $4, %%xmm4\n"
                              "psrldq $1, %%xmm1\n"
                              "psrldq $1, %%xmm6\n"
                              "pand  %%xmm4, %%xmm2\n"
                              "pand  %%xmm4, %%xmm7\n"
                              "pslldq $4, %%xmm4\n"
                              "psrldq $2, %%xmm2\n"
                              "psrldq $2, %%xmm7\n"
                              "pand  %%xmm4, %%xmm3\n"
                              "pand  %%xmm4, %%xmm8\n"
                              "por %%xmm1, %%xmm0\n"
                              "psrldq $3, %%xmm3\n"
                              "psrldq $3, %%xmm8\n"
                              "por %%xmm2, %%xmm0\n"
                              "por %%xmm6, %%xmm5\n"
                              "por %%xmm3, %%xmm0\n"
                              "por %%xmm7, %%xmm5\n"
                              "movdq2q %%xmm0, %%mm0\n"
                              "por %%xmm8, %%xmm5\n"
                              "movdqa %%xmm5, %%xmm1\n"
                              "pslldq $12, %%xmm5\n"
                              "psrldq $4, %%xmm1\n"
                              "por %%xmm5, %%xmm0\n"
                              "psrldq $8, %%xmm0\n"
                              "movq %%mm0, (%1)\n"
                              "movdq2q %%xmm0, %%mm1\n"
                              "movdq2q %%xmm1, %%mm2\n"
                              "movq %%mm1, 8(%1)\n"
                              "movq %%mm2, 16(%1)\n"::"r" (_s), "r"(_d));
                _s += 32;
                _d += 24;
                src_len -= 32;
        }
}

#else

/**
 * @brief Converts from DVS10 to UYVY
 * @copydetails vc_copylinev210
 */
static void vc_copylineDVS10(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        int src_len = dst_len / 1.5; /* right units */
        register const uint64_t *s;
        register uint64_t *d;

        register uint64_t a1, a2, a3, a4;

        d = (uint64_t *)(void *) dst;
        s = (const uint64_t *)(const void *) src;

        OPTIMIZED_FOR (int x = 0; x <= src_len - 16; x += 16) {
                a1 = *(s++);
                a2 = *(s++);
                a3 = *(s++);
                a4 = *(s++);

                a1 = (a1 & 0xffffff) | ((a1 >> 8) & 0xffffff000000LL);
                a2 = (a2 & 0xffffff) | ((a2 >> 8) & 0xffffff000000LL);
                a3 = (a3 & 0xffffff) | ((a3 >> 8) & 0xffffff000000LL);
                a4 = (a4 & 0xffffff) | ((a4 >> 8) & 0xffffff000000LL);

                *(d++) = a1 | (a2 << 48);       /* 0xa2|a2|a1|a1|a1|a1|a1|a1 */
                *(d++) = (a2 >> 16) | (a3 << 32);       /* 0xa3|a3|a3|a3|a2|a2|a2|a2 */
                *(d++) = (a3 >> 32) | (a4 << 16);       /* 0xa4|a4|a4|a4|a4|a4|a3|a3 */
        }
}

#endif                          /* !(__APPLE__ || HAVE_32B_LINUX) */

/**
 * @brief Changes color order of an RGB
 *
 * @note
 * Unlike most of the non-RGBA conversions, RGB shifts are respected.
 *
 * @copydetails vc_copyliner10k
 */
static void vc_copylineRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        register unsigned int r, g, b;
        union {
                unsigned int out;
                unsigned char c[4];
        } u;

        if (rshift == 0 && gshift == 8 && bshift == 16) {
                memcpy(dst, src, dst_len);
        } else {
                OPTIMIZED_FOR (int x = 0; x <= dst_len - 3; x += 3) {
                        r = *src++;
                        g = *src++;
                        b = *src++;
                        u.out = (r << rshift) | (g << gshift) | (b << bshift);
                        *dst++ = u.c[0];
                        *dst++ = u.c[1];
                        *dst++ = u.c[2];
                }
        }
}

/**
 * @brief Converts from RGBA to RGB. Channels in RGBA can be differently ordered.
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     4B-aligned buffer containing pixels in RGBA
 * @param[in]  dst_len length of data that should be written to dst buffer (in bytes)
 * @param[in]  rshift  source red shift
 * @param[in]  gshift  source green shift
 * @param[in]  bshift  source blue shift
 *
 * @note
 * In opposite to the defined semantic of {r,g,b}shift, here instead of destination
 * shifts the shifts define the source codec properties.
 */
static void vc_copylineRGBAtoRGBwithShift(unsigned char * __restrict dst2, const unsigned char * __restrict src2, int dst_len, int rshift, int gshift, int bshift)
{
	register const uint32_t * src = (const uint32_t *)(const void *) src2;
	register uint32_t * dst = (uint32_t *)(void *) dst2;
        int x;
        OPTIMIZED_FOR (x = 0; x <= dst_len - 12; x += 12) {
		register uint32_t in1 = *src++;
		register uint32_t in2 = *src++;
		register uint32_t in3 = *src++;
		register uint32_t in4 = *src++;

                *dst++ = ((in2 >> rshift)) << 24 |
                        ((in1 >> bshift) & 0xff) << 16 |
                        ((in1 >> gshift) & 0xff) << 8 |
                        ((in1 >> rshift) & 0xff);
                *dst++ = ((in3 >> gshift)) << 24 |
                        ((in3 >> rshift) & 0xff) << 16 |
                        ((in2 >> bshift) & 0xff) << 8 |
                        ((in2 >> gshift) & 0xff);
                *dst++  = ((in4 >> bshift)) << 24 |
                        ((in4 >> gshift) & 0xff) << 16 |
                        ((in4 >> rshift) & 0xff) << 8 |
                        ((in3 >> bshift) & 0xff);
        }

        uint8_t *dst_c = (uint8_t *) dst;
        for (; x <= dst_len - 3; x += 3) {
		register uint32_t in = *src++;
                *dst_c++ = (in >> rshift) & 0xff;
                *dst_c++ = (in >> gshift) & 0xff;
                *dst_c++ = (in >> bshift) & 0xff;
        }
}

/**
 * @brief Converts from AGBR to RGB
 * @copydetails vc_copylinev210
 * @see vc_copylineRGBAtoRGBwithShift
 * @see vc_copylineRGBAtoRGB
 */
void vc_copylineABGRtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        enum {
                SRC_RSHIFT = 24,
                SRC_GSHIFT = 16,
                SRC_BSHIFT = 8,
        };
#ifdef __SSSE3__
        __m128i shuf = _mm_setr_epi8(3, 2, 1, 7, 6, 5, 11, 10, 9, 15, 14, 13, 0xff, 0xff, 0xff, 0xff);
        __m128i loaded;
        int x;
        for (x = 0; x <= dst_len - 24; x += 12) {
                loaded = _mm_lddqu_si128((const __m128i_u *) src);
                loaded = _mm_shuffle_epi8(loaded, shuf);
                _mm_storeu_si128((__m128i_u *)dst, loaded);

                src += 16;
                dst += 12;
        }

        uint8_t *dst_c = (uint8_t *) dst;
        for (; x <= dst_len - 3; x += 3) {
                register uint32_t in = *(const uint32_t *) (const void *) src;
                *dst_c++ = (in >> SRC_RSHIFT) & 0xff;
                *dst_c++ = (in >> SRC_GSHIFT) & 0xff;
                *dst_c++ = (in >> SRC_BSHIFT) & 0xff;
        }
#else
        vc_copylineRGBAtoRGBwithShift(dst, src, dst_len, SRC_RSHIFT, SRC_GSHIFT,
                                      SRC_BSHIFT);
#endif
}

void
vc_copylineBGRAtoRGB(unsigned char *__restrict dst,
                     const unsigned char *__restrict src2, int dst_len,
                     int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        enum {
                SRC_RSHIFT = 16,
                SRC_GSHIFT = 8,
                SRC_BSHIFT = 0,
        };
        vc_copylineRGBAtoRGBwithShift(dst, src2, dst_len, SRC_RSHIFT, SRC_GSHIFT,
                                      SRC_BSHIFT);
}

/**
 * @brief Converts from RGBA to RGB
 * @copydetails vc_copylineRGBAtoRGBwithShift
 */
static void vc_copylineRGBAtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        enum {
                SRC_RSHIFT = 0,
                SRC_GSHIFT = 8,
                SRC_BSHIFT = 16,
        };
#ifdef __SSSE3__
        __m128i shuf = _mm_setr_epi8(0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 0xff, 0xff, 0xff, 0xff);
        __m128i loaded;
        int x;
        for (x = 0; x <= dst_len - 24; x += 12) {
                loaded = _mm_lddqu_si128((const __m128i_u *) src);
                loaded = _mm_shuffle_epi8(loaded, shuf);
                _mm_storeu_si128((__m128i_u *)dst, loaded);

                src += 16;
                dst += 12;
        }

        uint8_t *dst_c = (uint8_t *) dst;
        for (; x <= dst_len - 3; x += 3) {
                register uint32_t in = *(const uint32_t *) (const void *) src;
                *dst_c++ = (in >> SRC_RSHIFT) & 0xff;
                *dst_c++ = (in >> SRC_GSHIFT) & 0xff;
                *dst_c++ = (in >> SRC_BSHIFT) & 0xff;
        }
#else
        vc_copylineRGBAtoRGBwithShift(dst, src, dst_len, SRC_RSHIFT, SRC_GSHIFT,
                                      SRC_BSHIFT);
#endif
}

/**
 * @brief Converts RGBA with different shifts to RGBA
 *
 * dst and src may overlap
 */
void vc_copylineToRGBA_inplace(unsigned char *dst, const unsigned char *src, int dst_len,
                int src_rshift, int src_gshift, int src_bshift)
{
	register const uint32_t * in = (const uint32_t *)(const void *) src;
	register uint32_t * out = (uint32_t *)(void *) dst;
        while (dst_len >= 4) {
		register uint32_t in_val = *in++;

                *out++ = ((in_val >> src_rshift) & 0xff) |
                        ((in_val >> src_gshift) & 0xff) << 8 |
                        ((in_val >> src_bshift) & 0xff) << 16;

                dst_len -= 4;
        }
}

/**
 * @brief Converts UYVY to grayscale.
 * @todo is this correct??
 */
void vc_copylineUYVYtoGrayscale(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 2; x += 2) {
                src++; // U
                *dst++ = *src++; // Y
                src++; // V
                *dst++ = *src++; // Y
        }
}

/**
 * @brief Converts RGB to RGBA
 * @copydetails vc_copyliner10k
 */
void vc_copylineRGBtoRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{

        int x = 0;


#ifdef __SSSE3__
        __m128i alpha_mask_sse = _mm_set1_epi32(0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift));
        __m128i shuf;

        // common shifts are 0 8 16 (RGB) or 16 8 0 (BGR)
        // generalized solution would be too complex and not faster
        __m128i loaded;
        bool is_simd_compat = false;

        if (rshift == 0 && gshift == 8 && bshift == 16) {
                is_simd_compat = true;
                shuf = _mm_setr_epi8(0, 1, 2, 0xff, 3, 4, 5, 0xff, 6, 7, 8, 0xff, 9, 10, 11, 0xff);
        } else if (rshift == 16 && gshift == 8 && bshift == 0) {
                is_simd_compat = true;
                shuf = _mm_setr_epi8(2, 1, 0, 0xff, 5, 4, 3, 0xff, 8, 7, 6, 0xff, 11, 10, 9, 0xff);
        }

        if (is_simd_compat) {
                for (x = 0; x <= dst_len - 32; x += 16) {
                        loaded = _mm_lddqu_si128((const __m128i_u *) src);
                        loaded = _mm_shuffle_epi8(loaded, shuf);
                        loaded = _mm_or_si128(loaded, alpha_mask_sse);
                        _mm_storeu_si128((__m128i_u *)dst, loaded);
                        src += 12;
                        dst += 16;
                }
        }
#endif

        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);
        register unsigned int r, g, b;
        register uint32_t *d = (uint32_t *)(void *) dst;

        OPTIMIZED_FOR (; x <= dst_len - 4; x += 4) {
                r = *src++;
                g = *src++;
                b = *src++;
                
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
        }
}

/**
 * @brief Converts RGB(A) into UYVY
 *
 * @copydetails vc_copyliner10k
 * @param[in] roff     red offset in bytes (0 for RGB)
 * @param[in] goff     green offset in bytes (1 for RGB)
 * @param[in] boff     blue offset in bytes (2 for RGB)
 * @param[in] pix_size source pixel size (3 for RGB, 4 for RGBA)
 */
#if defined __GNUC__
static inline void vc_copylineToUYVY(unsigned char *__restrict dst,
                                     const unsigned char *__restrict src,
                                     int dst_len, int roff, int goff, int boff,
                                     int pix_size)
    __attribute__((always_inline));
#endif
static inline void
vc_copylineToUYVY(unsigned char *__restrict dst,
                  const unsigned char *__restrict src, int dst_len, int roff,
                  int goff, int boff, int pix_size)
{
        enum {
                UYVY_BPP   = 4,
                LOOP_ITEMS = 2,
        };
        const struct color_coeffs cfs  = *get_color_coeffs(CS_DFL, DEPTH8);
        register uint32_t         *d   = (uint32_t *) (void *) dst;
        int r, g, b;
        int y1, y2, u ,v;
#define loop_body \
                r = src[roff];\
                g = src[goff];\
                b = src[boff];\
                src += pix_size;\
                y1 = (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) + 16;\
                u = RGB_TO_CB(cfs, r, g, b);\
                v = RGB_TO_CR(cfs, r, g, b);\
                r = src[roff];\
                g = src[goff];\
                b = src[boff];\
                src += pix_size;\
                y2 = (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) + 16;\
                u += RGB_TO_CB(cfs, r, g, b);\
                v += RGB_TO_CR(cfs, r, g, b);\
                u = ((u / 2) >> COMP_BASE) + 128;\
                v = ((v / 2) >> COMP_BASE) + 128;\
\
                *d++ = ((y2 & 0xFF) << 24) | ((v & 0xFF) << 16) | \
                       ((y1 & 0xFF) << 8) | (u & 0xFF);

        const int count = (dst_len + 3) / UYVY_BPP;
        int       x     = 0;
        for (; x < count / LOOP_ITEMS; x++) {
                loop_body
                loop_body
        }
        x *= LOOP_ITEMS;
        for (; x < count; x++) {
                loop_body
        }
#undef loop_body
}

/**
 * @brief Converts 8-bit YCbCr (packed 4:2:2 in 32-bit) word to RGB.
 *
 * Converts 8-bit YCbCr (packed 4:2:2 in 32-bit word to RGB. Offset of YCbCr
 * components can be given by parameters (in bytes). This macro is used by
 * vc_copylineUYVYtoRGB() and vc_copylineYUYVtoRGB().
 *
 * @todo make it faster if needed
 * @param rgb16   true if output is 16-bit RGB, otherwise false
 */
#define copylineYUVtoRGB(dst, src, dst_len, y1_off, y2_off, u_off, v_off, rgb16) {\
        enum { DEPTH = DEPTH8 };\
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH);\
        OPTIMIZED_FOR (int x = 0; x <= (dst_len) - 6 * (1 + (rgb16)); x += 6 * (1 + (rgb16))) {\
                register int y1 = cfs.y_scale * ((src)[y1_off] - 16);\
                register int y2 = cfs.y_scale * ((src)[y2_off] - 16);\
                register int u = (src)[u_off] - 128;\
                register int v = (src)[v_off] - 128;\
                int val;\
                src += 4;\
                if (rgb16) *(dst)++ = 0;\
                val = YCBCR_TO_R(cfs, y1, u, v) >> COMP_BASE;\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = YCBCR_TO_G(cfs, y1, u, v) >> COMP_BASE;\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = YCBCR_TO_B(cfs, y1, u, v) >> COMP_BASE;\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = YCBCR_TO_R(cfs, y2, u, v) >> COMP_BASE;\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = YCBCR_TO_G(cfs, y2, u, v) >> COMP_BASE;\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = YCBCR_TO_B(cfs, y2, u, v) >> COMP_BASE;\
                *(dst)++ = CLAMP(val, 0, 255);\
        }\
}

/**
 * @brief Converts UYVY to RGB.
 * @see copylineYUVtoRGB
 * @param[out] dst     output buffer for RGB
 * @param[in]  src     input buffer with UYVY
 */
static void vc_copylineUYVYtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        copylineYUVtoRGB(dst, src, dst_len, 1, 3, 0, 2, 0);
}

/**
 * @brief Converts YUYV to RGB.
 * @see copylineYUVtoRGB
 * @param[out] dst     output buffer for RGB
 * @param[in]  src     input buffer with YUYV
 */
static void vc_copylineYUYVtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        copylineYUVtoRGB(dst, src, dst_len, 0, 2, 1, 3, 0);
}

static void vc_copylineUYVYtoRG48(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        copylineYUVtoRGB(dst, src, dst_len, 1, 3, 0, 2, 1);
}

/**
 * @brief Converts UYVY to RGBA.
 * @param[out] dst     output buffer for RGBA
 * @param[in]  src     input buffer with UYVY
 */
static void vc_copylineUYVYtoRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        assert((uintptr_t) dst % sizeof(uint32_t) == 0);
        uint32_t *dst32 = (uint32_t *)(void *) dst;
        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 8; x += 8) {
                register int y1, y2, u ,v;
                u = *src++;
                y1 = *src++;
                v = *src++;
                y2 = *src++;
                int r = 1.164 * (y1 - 16) + 1.793 * (v - 128);
                int g = 1.164 * (y1 - 16) - 0.534 * (v - 128) - 0.213 * (u - 128);
                int b = 1.164 * (y1 - 16) + 2.115 * (u - 128);
                r = CLAMP(r, 0, 255);
                g = CLAMP(g, 0, 255);
                b = CLAMP(b, 0, 255);
                *dst32++ = alpha_mask | r << rshift | g << gshift | b << bshift;
                r = 1.164 * (y2 - 16) + 1.793 * (v - 128);
                g = 1.164 * (y2 - 16) - 0.534 * (v - 128) - 0.213 * (u - 128);
                b = 1.164 * (y2 - 16) + 2.115 * (u - 128);
                r = CLAMP(r, 0, 255);
                g = CLAMP(g, 0, 255);
                b = CLAMP(b, 0, 255);
                *dst32++ = alpha_mask | r << rshift | g << gshift | b << bshift;
        }
}

/**
 * @brief Converts UYVY to RGB using SSE.
 * Uses Rec. 709 with standard SDI ceiling and floor
 * There can be some inaccuracies due to the use of integer arithmetic
 * @copydetails vc_copylinev210
 * @todo make it faster if needed
 */
void vc_copylineUYVYtoRGB_SSE(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
#ifdef __SSSE3__
        __m128i yfactor = _mm_set1_epi16(74); //1.164 << 6
        __m128i rvfactor = _mm_set1_epi16(115); //1.793 << 6
        __m128i gvfactor = _mm_set1_epi16(34); //0.534 << 6
        __m128i gufactor = _mm_set1_epi16(14); //0.213 << 6
        __m128i bufactor = _mm_set1_epi16(135); //2.115 << 6

        __m128i ysub = _mm_set1_epi16(16);
        __m128i uvsub = _mm_set1_epi16(128);

        __m128i zero128 = _mm_set1_epi32(0);
        __m128i max = _mm_set1_epi16(255);
        //YYVVYYUU
        __m128i ymask = _mm_set1_epi32(0x00FF00FF);
        __m128i umask = _mm_set1_epi32(0x000000FF);

        __m128i rgbshuffle = _mm_setr_epi8(0, 2, 1, 4, 6, 5, 8, 10, 9, 12, 14, 13, 15, 11, 7, 3);

        __m128i yuv;
        __m128i rgb;

        __m128i y;
        __m128i u;
        __m128i v;
        __m128i r;
        __m128i g;
        __m128i b;

        while(dst_len >= 28){
                yuv = _mm_lddqu_si128((__m128i const*)(const void *) src);
                src += 16;

                u = _mm_and_si128(yuv, umask);
                u = _mm_or_si128(u, _mm_bslli_si128(u, 2));

                yuv = _mm_bsrli_si128(yuv, 1);
                y = _mm_and_si128(yuv, ymask);

                yuv = _mm_bsrli_si128(yuv, 1);
                v = _mm_and_si128(yuv, umask);
                v = _mm_or_si128(v, _mm_bslli_si128(v, 2));

                y = _mm_subs_epi16(y, ysub);
                y = _mm_mullo_epi16(y, yfactor);

                u = _mm_subs_epi16(u, uvsub);
                v = _mm_subs_epi16(v, uvsub);

                r = _mm_adds_epi16(y, _mm_mullo_epi16(v, rvfactor));
                g = _mm_subs_epi16(y, _mm_mullo_epi16(v, gvfactor));
                g = _mm_subs_epi16(g, _mm_mullo_epi16(u, gufactor));
                b = _mm_adds_epi16(y, _mm_mullo_epi16(u, bufactor));

                //Make sure that the result is in the interval 0..255
                r = _mm_max_epi16(zero128, r);
                g = _mm_max_epi16(zero128, g);
                b = _mm_max_epi16(zero128, b);

                r = _mm_srli_epi16(r, 6);
                g = _mm_srli_epi16(g, 6);
                b = _mm_srli_epi16(b, 6);

                r = _mm_min_epi16(max, r);
                g = _mm_min_epi16(max, g);
                b = _mm_min_epi16(max, b);

                rgb = _mm_or_si128(_mm_bslli_si128(g, 1), r);
                rgb = _mm_unpacklo_epi8(rgb, b);
                rgb = _mm_shuffle_epi8(rgb, rgbshuffle);
                _mm_storeu_si128((__m128i *)(void *) dst, rgb);
                dst += 12;

                rgb = _mm_or_si128(_mm_bslli_si128(g, 1), r);
                rgb = _mm_unpackhi_epi8(rgb, b);
                rgb = _mm_shuffle_epi8(rgb, rgbshuffle);
                _mm_storeu_si128((__m128i *)(void *) dst, rgb);
                dst += 12;

                dst_len -= 24;
        }
#endif
        //copy last few pixels
        vc_copylineUYVYtoRGB(dst, src, dst_len, rshift, gshift, bshift);
}

#if defined __GNUC__
static inline void vc_copylineRGB_AtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int pix_pad)
        __attribute__((always_inline));
#endif
static inline void vc_copylineRGB_AtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int pix_pad) {
#define FETCH_DATA \
                r = *src++; \
                g = *src++; \
                b = *src++; \
                src += pix_pad;
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 36; x += 36) {
                unsigned r, g, b;
                FETCH_DATA
                dst[BYTE_SWAP(0)] = r << 4;
                dst[BYTE_SWAP(1)] = r >> 4;
                dst[BYTE_SWAP(2)] = g;
                dst[BYTE_SWAP(3)] = b << 4;
                dst[4 + BYTE_SWAP(0)] = b >> 4;
                FETCH_DATA
                dst[4 + BYTE_SWAP(1)] = r;
                dst[4 + BYTE_SWAP(2)] = g << 4;
                dst[4 + BYTE_SWAP(3)] = g >> 4;
                dst[8 + BYTE_SWAP(0)] = b;
                FETCH_DATA
                dst[8 + BYTE_SWAP(1)] = r << 4;
                dst[8 + BYTE_SWAP(2)] = r >> 4;
                dst[8 + BYTE_SWAP(3)] = g;
                dst[12 + BYTE_SWAP(0)] = b << 4;
                dst[12 + BYTE_SWAP(1)] = b >> 4;
                FETCH_DATA
                dst[12 + BYTE_SWAP(2)] = r;
                dst[12 + BYTE_SWAP(3)] = g << 4;
                dst[16 + BYTE_SWAP(0)] = g >> 4;
                dst[16 + BYTE_SWAP(1)] = b;
                FETCH_DATA
                dst[16 + BYTE_SWAP(2)] = r << 4;
                dst[16 + BYTE_SWAP(3)] = r >> 4;
                dst[20 + BYTE_SWAP(0)] = g;
                dst[20 + BYTE_SWAP(1)] = b << 4;
                dst[20 + BYTE_SWAP(2)] = b >> 4;
                FETCH_DATA
                dst[20 + BYTE_SWAP(3)] = r;
                dst[24 + BYTE_SWAP(0)] = g << 4;
                dst[24 + BYTE_SWAP(1)] = g >> 4;
                dst[24 + BYTE_SWAP(2)] = b;
                FETCH_DATA
                dst[24 + BYTE_SWAP(3)] = r << 4;
                dst[28 + BYTE_SWAP(0)] = r >> 4;
                dst[28 + BYTE_SWAP(1)] = g;
                dst[28 + BYTE_SWAP(2)] = b << 4;
                dst[28 + BYTE_SWAP(3)] = b >> 4;
                FETCH_DATA
                dst[32 + BYTE_SWAP(0)] = r;
                dst[32 + BYTE_SWAP(1)] = g << 4;
                dst[32 + BYTE_SWAP(2)] = g >> 4;
                dst[32 + BYTE_SWAP(3)] = b;
                dst += 36;
        }
#undef FETCH_DATA
}

/**
 * Converts 8-bit RGB to 12-bit packed RGB in full range (compatible with
 * SMPTE 268M DPX version 1, Annex C, Method C4 packing).
 */
static void vc_copylineRGBtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len,
                int rshift, int gshift, int bshift) {
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        vc_copylineRGB_AtoR12L(dst, src, dst_len, 0);
}

static void vc_copylineRGBAtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len,
                int rshift, int gshift, int bshift) {
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        vc_copylineRGB_AtoR12L(dst, src, dst_len, 1);
}

static void vc_copylineRGBAtoRG48(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len,
                int rshift, int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dst_len - 6; x += 6) {
                *dst++ = 0;
                *dst++ = *src++;
                *dst++ = 0;
                *dst++ = *src++;
                *dst++ = 0;
                *dst++ = *src++;
                src++;
        }
}

static void vc_copylineRGBtoRG48(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len,
                int rshift, int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dst_len - 2; x += 2) {
                *dst++ = 0;
                *dst++ = *src++;
        }
}

/**
 * @brief Converts R12L to RG48.
 * Converts 12-bit packed RGB in full range (compatible with
 * SMPTE 268M DPX version 1, Annex C, Method C4 packing) to 16-bit RGB
 * @copydetails vc_copylinev210
 */
static void vc_copylineR12LtoRG48(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        enum {
                PIX_COUNT = 8,
                RG48_BPP  = 6,
                OUT_BL_SZ = PIX_COUNT * RG48_BPP,
        };
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        int x = 0;
        OPTIMIZED_FOR(; x < dst_len - OUT_BL_SZ + 1; x += OUT_BL_SZ)
        {
                //0
                //R
                *dst++ = src[BYTE_SWAP(0)] << 4;
                *dst++ = (src[BYTE_SWAP(1)] << 4) | (src[BYTE_SWAP(0)] >> 4);
                //G
                *dst++ = src[BYTE_SWAP(1)] & 0xF0;
                *dst++ = src[BYTE_SWAP(2)];
                //B
                *dst++ = src[BYTE_SWAP(3)] << 4;
                *dst++ = (src[4 + BYTE_SWAP(0)] << 4) | (src[BYTE_SWAP(3)] >> 4);

                //1
                *dst++ = src[4 + BYTE_SWAP(0)] & 0xF0;
                *dst++ = src[4 + BYTE_SWAP(1)];

                *dst++ = src[4 + BYTE_SWAP(2)] << 4;
                *dst++ = (src[4 + BYTE_SWAP(3)] << 4) | (src[4 + BYTE_SWAP(2)] >> 4);

                *dst++ = src[4 + BYTE_SWAP(3)] & 0xF0;
                *dst++ = src[8 + BYTE_SWAP(0)];

                //2
                *dst++ = src[8 + BYTE_SWAP(1)] << 4;
                *dst++ = (src[8 + BYTE_SWAP(2)] << 4) | (src[8 + BYTE_SWAP(1)] >> 4);

                *dst++ = src[8 + BYTE_SWAP(2)] & 0xF0;
                *dst++ = src[8 + BYTE_SWAP(3)];

                *dst++ = src[12 + BYTE_SWAP(0)] << 4;
                *dst++ = (src[12 + BYTE_SWAP(1)] << 4) | (src[12 + BYTE_SWAP(0)] >> 4);

                //3
                *dst++ = src[12 + BYTE_SWAP(1)] & 0xF0;
                *dst++ = src[12 + BYTE_SWAP(2)];

                *dst++ = src[12 + BYTE_SWAP(3)] << 4;
                *dst++ = (src[16 + BYTE_SWAP(0)] << 4) | (src[12 + BYTE_SWAP(3)] >> 4);

                *dst++ = src[16 + BYTE_SWAP(0)] & 0xF0;
                *dst++ = src[16 + BYTE_SWAP(1)];

                //4
                *dst++ = src[16 + BYTE_SWAP(2)] << 4;
                *dst++ = (src[16 + BYTE_SWAP(3)] << 4) | (src[16 + BYTE_SWAP(2)] >> 4);

                *dst++ = src[16 + BYTE_SWAP(3)] & 0xF0;
                *dst++ = src[20 + BYTE_SWAP(0)];

                *dst++ = src[20 + BYTE_SWAP(1)] << 4;
                *dst++ = (src[20 + BYTE_SWAP(2)] << 4) | (src[20 + BYTE_SWAP(1)] >> 4);

                //5
                *dst++ = src[20 + BYTE_SWAP(2)] & 0xF0;
                *dst++ = src[20 + BYTE_SWAP(3)];

                *dst++ = src[24 + BYTE_SWAP(0)] << 4;
                *dst++ = (src[24 + BYTE_SWAP(1)] << 4) | (src[24 + BYTE_SWAP(0)] >> 4);

                *dst++ = src[24 + BYTE_SWAP(1)] & 0xF0;
                *dst++ = src[24 + BYTE_SWAP(2)];

                //6
                *dst++ = src[24 + BYTE_SWAP(3)] << 4;
                *dst++ = (src[28 + BYTE_SWAP(0)] << 4) | (src[24 + BYTE_SWAP(3)] >> 4);

                *dst++ = src[28 + BYTE_SWAP(0)] & 0xF0;
                *dst++ = src[28 + BYTE_SWAP(1)];

                *dst++ = src[28 + BYTE_SWAP(2)] << 4;
                *dst++ = (src[28 + BYTE_SWAP(3)] << 4) | (src[28 + BYTE_SWAP(2)] >> 4);

                //7
                *dst++ = src[28 + BYTE_SWAP(3)] & 0xF0;
                *dst++ = src[32 + BYTE_SWAP(0)];

                *dst++ = src[32 + BYTE_SWAP(1)] << 4;
                *dst++ = (src[32 + BYTE_SWAP(2)] << 4) | (src[32 + BYTE_SWAP(1)] >> 4);

                *dst++ = src[32 + BYTE_SWAP(2)] & 0xF0;
                *dst++ = src[32 + BYTE_SWAP(3)];

                src += 36;
        }
        if (x == dst_len) {
                return;
        }
        // compute last incomplete block if dst_len % OUT_BL_SZ != 0, not
        // writing past the requested dst_len
        unsigned char tmp_buf[OUT_BL_SZ];
        vc_copylineR12LtoRG48(tmp_buf, src, OUT_BL_SZ, rshift, gshift, bshift);
        memcpy(dst, tmp_buf, dst_len - x);
}

static void vc_copylineR12LtoY416(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        enum {
                D_DEPTH = 16,
        };
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        assert((uintptr_t) dst % sizeof(uint16_t) == 0);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, D_DEPTH);
        uint16_t *d = (void *) dst;

#define WRITE_RES \
        u = (RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) + \
            (1 << (D_DEPTH - 1)); \
        *d++ = CLAMP_LIMITED_CBCR(u, D_DEPTH); \
        y    = (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) + \
            (1 << (D_DEPTH - 4)); \
        *d++ = CLAMP_LIMITED_Y(y, D_DEPTH); \
        v    = (RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) + \
            (1 << (D_DEPTH - 1)); \
        *d++ = CLAMP_LIMITED_CBCR(v, D_DEPTH); \
        *d++ = 0xFFFFU;

        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 64) {
                comp_type_t r, g, b;
                comp_type_t y, u, v;

                r = (src[BYTE_SWAP(1)] & 0xFU) << 12U | src[BYTE_SWAP(0)] << 4U;                        //0
                g = src[BYTE_SWAP(2)] << 8U | (src[BYTE_SWAP(1)] & 0xF0U);
                b = (src[4 + BYTE_SWAP(0)] & 0xFU) << 12U | src[BYTE_SWAP(3)] << 4U;
                WRITE_RES
                r = src[4 + BYTE_SWAP(1)] << 8U | (src[4 + BYTE_SWAP(0)] & 0xF0U);                      //1
                g = (src[4 + BYTE_SWAP(3)] & 0xFU) << 12U | (src[4 + BYTE_SWAP(2)]) << 4U;
                b = src[8 + BYTE_SWAP(0)] << 8U | (src[4 + BYTE_SWAP(3)] & 0xF0U);
                WRITE_RES
                r = (src[8 + BYTE_SWAP(2)] & 0xFU) << 12U |src[8 + BYTE_SWAP(1)] << 4U;                 //2
                g = src[8 + BYTE_SWAP(3)] << 8U | (src[8 + BYTE_SWAP(2)] & 0xF0U);
                b = (src[12 + BYTE_SWAP(1)] & 0xFU) << 12U | src[12 + BYTE_SWAP(0)] << 4U;
                WRITE_RES
                r = src[12 + BYTE_SWAP(2)] << 8U | (src[12 + BYTE_SWAP(1)] & 0xF0U);                    //3
                g = (src[16 + BYTE_SWAP(0)] & 0xFU) << 12U | src[12 + BYTE_SWAP(3)] << 4U;
                b = src[16 + BYTE_SWAP(1)] << 8U | (src[16 + BYTE_SWAP(0)] & 0xF0U);
                WRITE_RES
                r = (src[16 + BYTE_SWAP(3)] & 0xFU) << 12U | src[16 + BYTE_SWAP(2)] << 4U;              //4
                g = src[20 + BYTE_SWAP(0)] << 8U | (src[16 + BYTE_SWAP(3)] & 0xF0U);
                b = (src[20 + BYTE_SWAP(2)] & 0xFU) << 12U | src[20 + BYTE_SWAP(1)] << 4U;
                WRITE_RES
                r = src[20 + BYTE_SWAP(3)] << 8U | (src[20 + BYTE_SWAP(2)] & 0xF0U);                    //5
                g = (src[24 + BYTE_SWAP(1)] & 0xFU) << 12U | src[24 + BYTE_SWAP(0)] << 4U;
                b = src[24 + BYTE_SWAP(2)] << 8U | (src[24 + BYTE_SWAP(1)] & 0xF0U);
                WRITE_RES
                r = (src[28 + BYTE_SWAP(0)] & 0xFU) << 12U | src[24 + BYTE_SWAP(3)] << 4U;              //6
                g = src[28 + BYTE_SWAP(1)] << 8U | (src[28 + BYTE_SWAP(0)] & 0xF0U);
                b = (src[28 + BYTE_SWAP(3)] & 0xFU) << 12U | src[28 + BYTE_SWAP(2)] << 4U;
                WRITE_RES
                r = src[32 + BYTE_SWAP(0)] << 8U | (src[28 + BYTE_SWAP(3)] & 0xF0U);                    //7
                g = (src[32 + BYTE_SWAP(2)] & 0xFU) << 12U | src[32 + BYTE_SWAP(1)] << 4U;
                b = src[32 + BYTE_SWAP(3)] << 8U | (src[32 + BYTE_SWAP(2)] & 0xF0U);
                WRITE_RES

                src += 36;
        }
#undef WRITE_RES
}

static void
vc_copylineR12LtoUYVY(unsigned char *__restrict dst,
                      const unsigned char *__restrict src, int dst_len,
                      int rshift, int gshift, int bshift)
{
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        enum {
                D_DPTH        = 8,
                YOFF          = 1 << (D_DPTH - 4),
                COFF          = 1 << (D_DPTH - 1),
                PX_PER_BLK_IN = 8,
                UYVY_BPP      = 2,
                DST_BYTE_SZ   = PX_PER_BLK_IN * UYVY_BPP,
        };
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, D_DPTH);
        uint8_t *d = (void *) dst;
#define WRITE_RES \
        { \
                comp_type_t u = ((RGB_TO_CB(cfs, r1, g1, b1) + \
                                  RGB_TO_CB(cfs, r2, g2, b2)) >> \
                                 (COMP_BASE + D_DPTH + 1)) + \
                                COFF; \
                *d++          = CLAMP_LIMITED_CBCR(u, D_DPTH); \
                comp_type_t y = (RGB_TO_Y(cfs, r1, g1, b1) >> \
                                 (COMP_BASE + D_DPTH)) + \
                                YOFF; \
                *d++          = CLAMP_LIMITED_Y(y, D_DPTH); \
                comp_type_t v = ((RGB_TO_CR(cfs, r1, g1, b1) + \
                                  RGB_TO_CR(cfs, r2, g2, b2)) >> \
                                 (COMP_BASE + D_DPTH + 1)) + \
                                COFF; \
                *d++ = CLAMP_LIMITED_CBCR(v, D_DPTH); \
                y    = (RGB_TO_Y(cfs, r2, g2, b2) >> \
                     (COMP_BASE + D_DPTH)) + \
                    YOFF; \
                *d++ = CLAMP_LIMITED_Y(y, D_DPTH); \
        }
        OPTIMIZED_FOR(int x = 0; x < dst_len; x += DST_BYTE_SZ)
        {
                comp_type_t r1 = (src[BYTE_SWAP(1)] & 0xFU) << 12U |
                                 src[BYTE_SWAP(0)] << 4U; // 0
                comp_type_t g1 =
                    src[BYTE_SWAP(2)] << 8U | (src[BYTE_SWAP(1)] & 0xF0U);
                comp_type_t b1 = (src[4 + BYTE_SWAP(0)] & 0xFU) << 12U |
                                 src[BYTE_SWAP(3)] << 4U;
                comp_type_t r2 = src[4 + BYTE_SWAP(1)] << 8U |
                                 (src[4 + BYTE_SWAP(0)] & 0xF0U); // 1
                comp_type_t g2 = (src[4 + BYTE_SWAP(3)] & 0xFU) << 12U |
                                 (src[4 + BYTE_SWAP(2)]) << 4U;
                comp_type_t b2 = src[8 + BYTE_SWAP(0)] << 8U |
                                 (src[4 + BYTE_SWAP(3)] & 0xF0U);
                WRITE_RES
                r1 = (src[8 + BYTE_SWAP(2)] & 0xFU) << 12U |
                     src[8 + BYTE_SWAP(1)] << 4U; // 2
                g1 = src[8 + BYTE_SWAP(3)] << 8U |
                     (src[8 + BYTE_SWAP(2)] & 0xF0U);
                b1 = (src[12 + BYTE_SWAP(1)] & 0xFU) << 12U |
                     src[12 + BYTE_SWAP(0)] << 4U;
                r2 = src[12 + BYTE_SWAP(2)] << 8U |
                     (src[12 + BYTE_SWAP(1)] & 0xF0U); // 3
                g2 = (src[16 + BYTE_SWAP(0)] & 0xFU) << 12U |
                     src[12 + BYTE_SWAP(3)] << 4U;
                b2 = src[16 + BYTE_SWAP(1)] << 8U |
                     (src[16 + BYTE_SWAP(0)] & 0xF0U);
                WRITE_RES
                r1 = (src[16 + BYTE_SWAP(3)] & 0xFU) << 12U |
                     src[16 + BYTE_SWAP(2)] << 4U; // 4
                g1 = src[20 + BYTE_SWAP(0)] << 8U |
                     (src[16 + BYTE_SWAP(3)] & 0xF0U);
                b1 = (src[20 + BYTE_SWAP(2)] & 0xFU) << 12U |
                     src[20 + BYTE_SWAP(1)] << 4U;
                r2 = src[20 + BYTE_SWAP(3)] << 8U |
                     (src[20 + BYTE_SWAP(2)] & 0xF0U); // 5
                g2 = (src[24 + BYTE_SWAP(1)] & 0xFU) << 12U |
                     src[24 + BYTE_SWAP(0)] << 4U;
                b2 = src[24 + BYTE_SWAP(2)] << 8U |
                     (src[24 + BYTE_SWAP(1)] & 0xF0U);
                WRITE_RES
                r1 = (src[28 + BYTE_SWAP(0)] & 0xFU) << 12U |
                     src[24 + BYTE_SWAP(3)] << 4U; // 6
                g1 = src[28 + BYTE_SWAP(1)] << 8U |
                     (src[28 + BYTE_SWAP(0)] & 0xF0U);
                b1 = (src[28 + BYTE_SWAP(3)] & 0xFU) << 12U |
                     src[28 + BYTE_SWAP(2)] << 4U;
                r2 = src[32 + BYTE_SWAP(0)] << 8U |
                     (src[28 + BYTE_SWAP(3)] & 0xF0U); // 7
                g2 = (src[32 + BYTE_SWAP(2)] & 0xFU) << 12U |
                     src[32 + BYTE_SWAP(1)] << 4U;
                b2 = src[32 + BYTE_SWAP(3)] << 8U |
                     (src[32 + BYTE_SWAP(2)] & 0xF0U);
                WRITE_RES

                src += 36;
        }
#undef WRITE_RES
}

static void vc_copylineR12LtoR10k(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 32; x += 32) {
                //0
                // Rh8 (pattern repeating in each group)
                *dst++ = (src[BYTE_SWAP(1)] << 4) | (src[BYTE_SWAP(0)] >> 4);
                // Rl2 + Gh6
                *dst++ = (src[BYTE_SWAP(0)] & 0xC) << 4 | src[BYTE_SWAP(2)] >> 2;
                // Gl4 + Bh4
                *dst++ = src[BYTE_SWAP(2)] << 6 | (src[BYTE_SWAP(1)] & 0xC0) >> 2 | // G
                        (src[4 + BYTE_SWAP(0)] & 0xF);
                // Blo6
                *dst++ = src[BYTE_SWAP(3)];
                // 1
                *dst++ = src[4 + BYTE_SWAP(1)];
                *dst++ = (src[4 + BYTE_SWAP(0)] & 0xC0) | (src[4 + BYTE_SWAP(3)] & 0xF) << 2 | src[4 + BYTE_SWAP(2)] >> 6;
                *dst++ = (src[4 + BYTE_SWAP(2)] & 0x3C) << 2 | src[8 + BYTE_SWAP(0)] >> 4;
                *dst++ = src[8 + BYTE_SWAP(0)] << 4 | (src[4 + BYTE_SWAP(0)] & 0xF0) >> 4;
                // 2
                *dst++ = (src[8 + BYTE_SWAP(2)] << 4) | (src[8 + BYTE_SWAP(1)] >> 4);
                *dst++ = (src[8 + BYTE_SWAP(1)] & 0xC) << 4 | src[8 + BYTE_SWAP(3)] >> 2; // Rl2 + Gh6
                *dst++ = src[8 + BYTE_SWAP(3)] << 6 | (src[8 + BYTE_SWAP(2)] & 0xC0) >> 2 | (src[12 + BYTE_SWAP(1)] & 0xF); // Gl4+Bh4
                *dst++ = src[12 + BYTE_SWAP(0)];
                // 3
                *dst++ = src[12 + BYTE_SWAP(2)]; // Rh8
                *dst++ = (src[12 + BYTE_SWAP(1)] & 0xC0) | ((src[16 + BYTE_SWAP(0)] & 0xF) << 2) | src[12 + BYTE_SWAP(3)] >> 6; // Rl2 + Gh6
                *dst++ = (src[12 + BYTE_SWAP(3)] & 0x3C) << 2 | src[16 + BYTE_SWAP(1)] >> 4; // Gl4 + Bh4
                *dst++ = src[16 + BYTE_SWAP(1)] << 4 | (src[16 + BYTE_SWAP(0)] & 0xF0) >> 4;
                // 4
                *dst++ = (src[16 + BYTE_SWAP(3)] << 4) | (src[16 + BYTE_SWAP(2)] >> 4);
                *dst++ = (src[16 + BYTE_SWAP(2)] & 0xC) << 4 | src[20 + BYTE_SWAP(0)] >> 2;
                *dst++ = src[20 + BYTE_SWAP(0)] << 6 | (src[16 + BYTE_SWAP(3)] & 0xC0) >> 2 | (src[20 + BYTE_SWAP(2)] & 0xF);
                *dst++ = src[20 + BYTE_SWAP(1)];
                // 5
                *dst++ = src[20 + BYTE_SWAP(3)];
                *dst++ = (src[20 + BYTE_SWAP(2)] & 0xC0) | (src[24 + BYTE_SWAP(1)] & 0xF) << 2 | src[24 + BYTE_SWAP(0)] >> 6;
                *dst++ = (src[24 + BYTE_SWAP(0)] & 0x3C) << 2 | src[24 + BYTE_SWAP(2)] >> 4;
                *dst++ = src[24 + BYTE_SWAP(2)] << 4 | src[24 + BYTE_SWAP(1)] >> 4;
                // 6
                *dst++ = (src[28 + BYTE_SWAP(0)] << 4) | (src[24 + BYTE_SWAP(3)] >> 4);
                *dst++ = (src[24 + BYTE_SWAP(3)] & 0xC) << 4 | src[28 + BYTE_SWAP(1)] >> 2;
                *dst++ = src[28 + BYTE_SWAP(1)] << 6 | (src[28 + BYTE_SWAP(0)] & 0xC0) >> 2 | (src[28 + BYTE_SWAP(3)] & 0xF);
                *dst++ = src[28 + BYTE_SWAP(2)];
                // 7
                *dst++ = src[32 + BYTE_SWAP(0)];
                *dst++ = (src[28 + BYTE_SWAP(3)] & 0xC0) | (src[32 + BYTE_SWAP(2)] & 0xF) << 2 | (src[32 + BYTE_SWAP(1)] >> 6);
                *dst++ = (src[32 + BYTE_SWAP(1)] & 0x3C) << 2 | src[32 + BYTE_SWAP(3)] >> 4;
                *dst++ = src[32 + BYTE_SWAP(3)] << 4 | src[32 + BYTE_SWAP(2)] >> 4;

                src += 36;
        }
}

/**
 * @brief Converts RG48 to R12L.
 * Converts 16-bit RGB to 12-bit packed RGB in full range (compatible with
 * SMPTE 268M DPX version 1, Annex C, Method C4 packing)
 * @copydetails vc_copylinev210
 */
static void vc_copylineRG48toR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 36; x += 36) {
                //0
                dst[BYTE_SWAP(0)] = src[0] >> 4;
                dst[BYTE_SWAP(0)] |= src[1] << 4;
                dst[BYTE_SWAP(1)] = src[1] >> 4;
                src += 2;

                dst[BYTE_SWAP(1)] |= src[0] & 0xF0;
                dst[BYTE_SWAP(2)] = src[1];
                src += 2;

                dst[BYTE_SWAP(3)] = src[0] >> 4;
                dst[BYTE_SWAP(3)] |= src[1] << 4;
                dst[4 + BYTE_SWAP(0)] = src[1] >> 4;
                src += 2;

                //1
                dst[4 + BYTE_SWAP(0)] |= src[0] & 0xF0;
                dst[4 + BYTE_SWAP(1)] = src[1];
                src += 2;

                dst[4 + BYTE_SWAP(2)] = src[0] >> 4;
                dst[4 + BYTE_SWAP(2)] |= src[1] << 4;
                dst[4 + BYTE_SWAP(3)] = src[1] >> 4;
                src += 2;

                dst[4 + BYTE_SWAP(3)] |= src[0] & 0xF0;
                dst[8 + BYTE_SWAP(0)] = src[1];
                src += 2;

                //2
                dst[8 + BYTE_SWAP(1)] = src[0] >> 4;
                dst[8 + BYTE_SWAP(1)] |= src[1] << 4;
                dst[8 + BYTE_SWAP(2)] = src[1] >> 4;
                src += 2;

                dst[8 + BYTE_SWAP(2)] |= src[0] & 0xF0;
                dst[8 + BYTE_SWAP(3)] = src[1];
                src += 2;

                dst[12 + BYTE_SWAP(0)] = src[0] >> 4;
                dst[12 + BYTE_SWAP(0)] |= src[1] << 4;
                dst[12 + BYTE_SWAP(1)] = src[1] >> 4;
                src += 2;

                //3
                dst[12 + BYTE_SWAP(1)] |= src[0] & 0xF0;
                dst[12 + BYTE_SWAP(2)] = src[1];
                src += 2;

                dst[12 + BYTE_SWAP(3)] = src[0] >> 4;
                dst[12 + BYTE_SWAP(3)] |= src[1] << 4;
                dst[16 + BYTE_SWAP(0)] = src[1] >> 4;
                src += 2;

                dst[16 + BYTE_SWAP(0)] |= src[0] & 0xF0;
                dst[16 + BYTE_SWAP(1)] = src[1];
                src += 2;

                //4
                dst[16 + BYTE_SWAP(2)] = src[0] >> 4;
                dst[16 + BYTE_SWAP(2)] |= src[1] << 4;
                dst[16 + BYTE_SWAP(3)] = src[1] >> 4;
                src += 2;

                dst[16 + BYTE_SWAP(3)] |= src[0] & 0xF0;
                dst[20 + BYTE_SWAP(0)] = src[1];
                src += 2;

                dst[20 + BYTE_SWAP(1)] = src[0] >> 4;
                dst[20 + BYTE_SWAP(1)] |= src[1] << 4;
                dst[20 + BYTE_SWAP(2)] = src[1] >> 4;
                src += 2;

                //5
                dst[20 + BYTE_SWAP(2)] |= src[0] & 0xF0;
                dst[20 + BYTE_SWAP(3)] = src[1];
                src += 2;

                dst[24 + BYTE_SWAP(0)] = src[0] >> 4;
                dst[24 + BYTE_SWAP(0)] |= src[1] << 4;
                dst[24 + BYTE_SWAP(1)] = src[1] >> 4;
                src += 2;

                dst[24 + BYTE_SWAP(1)] |= src[0] & 0xF0;
                dst[24 + BYTE_SWAP(2)] = src[1];
                src += 2;

                //6
                dst[24 + BYTE_SWAP(3)] = src[0] >> 4;
                dst[24 + BYTE_SWAP(3)] |= src[1] << 4;
                dst[28 + BYTE_SWAP(0)] = src[1] >> 4;
                src += 2;

                dst[28 + BYTE_SWAP(0)] |= src[0] & 0xF0;
                dst[28 + BYTE_SWAP(1)] = src[1];
                src += 2;

                dst[28 + BYTE_SWAP(2)] = src[0] >> 4;
                dst[28 + BYTE_SWAP(2)] |= src[1] << 4;
                dst[28 + BYTE_SWAP(3)] = src[1] >> 4;
                src += 2;

                //7
                dst[28 + BYTE_SWAP(3)] |= src[0] & 0xF0;
                dst[32 + BYTE_SWAP(0)] = src[1];
                src += 2;

                dst[32 + BYTE_SWAP(1)] = src[0] >> 4;
                dst[32 + BYTE_SWAP(1)] |= src[1] << 4;
                dst[32 + BYTE_SWAP(2)] = src[1] >> 4;
                src += 2;

                dst[32 + BYTE_SWAP(2)] |= src[0] & 0xF0;
                dst[32 + BYTE_SWAP(3)] = src[1];
                src += 2;

                dst += 36;
        }
}

static void vc_copylineY416toR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        enum {
                S_DEPTH = 16,
                D_DEPTH = 12,
        };
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, S_DEPTH);
#define GET_NEXT \
        u = *in++ - (1 << (S_DEPTH - 1)); \
        y = cfs.y_scale * (*in++ - (1 << (S_DEPTH - 4))); \
        v = *in++ - (1 << (S_DEPTH - 1)); \
        in++; \
        r = YCBCR_TO_R(cfs, y, u, v) >> (COMP_BASE + 4U); \
        g = YCBCR_TO_G(cfs, y, u, v) >> (COMP_BASE + 4U); \
        b = YCBCR_TO_B(cfs, y, u, v) >> (COMP_BASE + 4U); \
        r = CLAMP_FULL(r, D_DEPTH); \
        g = CLAMP_FULL(g, D_DEPTH); \
        b = CLAMP_FULL(b, D_DEPTH);

        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        const uint16_t *in = (const void *) src;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 36) {
                comp_type_t y, u, v, r, g, b;
                comp_type_t tmp;

                GET_NEXT // 0
                dst[BYTE_SWAP(0)] = r & 0xFFU;
                dst[BYTE_SWAP(1)] = (g & 0xFU) << 4U | r >> 8U;
                dst[BYTE_SWAP(2)] = g >> 4U;
                dst[BYTE_SWAP(3)] = b & 0xFFU;
                tmp = b >> 8U;

                GET_NEXT // 1
                dst[4 + BYTE_SWAP(0)] = (r & 0xFU) << 4U | tmp;
                dst[4 + BYTE_SWAP(1)] = r >> 4U;
                dst[4 + BYTE_SWAP(2)] = g & 0xFFU;
                dst[4 + BYTE_SWAP(3)] = (b & 0xFU) << 4U | g >> 8U;

                dst[8 + BYTE_SWAP(0)] = b >> 4U;
                GET_NEXT // 2
                dst[8 + BYTE_SWAP(1)] = r & 0xFFu;
                dst[8 + BYTE_SWAP(2)] = (g & 0xFU) << 4U | r >> 8U;
                dst[8 + BYTE_SWAP(3)] = g >> 4U;

                dst[12 + BYTE_SWAP(0)] = b & 0xFFU;
                tmp = b >> 8U;
                GET_NEXT // 3
                dst[12 + BYTE_SWAP(1)] = (r & 0xFU) << 4U | tmp;
                dst[12 + BYTE_SWAP(2)] = r >> 4U;
                dst[12 + BYTE_SWAP(3)] = g & 0xFFU;

                dst[16 + BYTE_SWAP(0)] = (b & 0xFU) << 4U | g >> 8U;
                dst[16 + BYTE_SWAP(1)] = b >> 4U;
                GET_NEXT // 4
                dst[16 + BYTE_SWAP(2)] = r & 0xFFU;
                dst[16 + BYTE_SWAP(3)] = (g & 0xFU) << 4U | r >> 8U;

                dst[20 + BYTE_SWAP(0)] = g >> 4U;
                dst[20 + BYTE_SWAP(1)] = b & 0xFFU;
                tmp = b >> 8U;
                GET_NEXT // 5
                dst[20 + BYTE_SWAP(2)] = (r & 0xFU) << 4U | tmp;
                dst[20 + BYTE_SWAP(3)] = r >> 4U;

                dst[24 + BYTE_SWAP(0)] = g & 0xFFU;
                dst[24 + BYTE_SWAP(1)] = (b & 0xFU) << 4U | g >> 8U;
                dst[24 + BYTE_SWAP(2)] = b >> 4U;
                GET_NEXT // 6
                dst[24 + BYTE_SWAP(3)] = r & 0xFFU;

                dst[28 + BYTE_SWAP(0)] = (g & 0xFU) << 4U | r >> 8U;
                dst[28 + BYTE_SWAP(1)] = g >> 4U;
                dst[28 + BYTE_SWAP(2)] = b & 0xFFU;
                tmp = b >> 8U;
                GET_NEXT // 7
                dst[28 + BYTE_SWAP(3)] = (r & 0xFU) << 4U | tmp;

                dst[32 + BYTE_SWAP(0)] = r >> 4U;
                dst[32 + BYTE_SWAP(1)] = g & 0xFFU;
                dst[32 + BYTE_SWAP(2)] = (b & 0xFU) << 4U | g >> 8U;
                dst[32 + BYTE_SWAP(3)] = b >> 4U;

                dst += 36;
        }
#undef GET_NEXT
}

static void vc_copylineY416toR10k(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        enum {
                S_DEPTH = 16,
        };
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, S_DEPTH);
        const uint16_t *in = (const void *) src;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 4) {
                comp_type_t y, u, v, r, g, b;

                u = *in++ - (1 << (S_DEPTH - 1));
                y = cfs.y_scale * (*in++ - (1 << (S_DEPTH - 4)));
                v = *in++ - (1 << (S_DEPTH - 1));
                in++;
                r = YCBCR_TO_R(cfs, y, u, v) >> (COMP_BASE + 6U);
                g = YCBCR_TO_G(cfs, y, u, v) >> (COMP_BASE + 6U);
                b = YCBCR_TO_B(cfs, y, u, v) >> (COMP_BASE + 6U);
                r = CLAMP_FULL(r, 10);
                g = CLAMP_FULL(g, 10);
                b = CLAMP_FULL(b, 10);

                *dst++ = r >> 2U;
                *dst++ = (r & 0x3U) << 6U | g >> 4U;
                *dst++ = (g & 0xFU) << 4U | b >> 6U;
                *dst++ = (b & 0x3FU) << 2U;
        }
}

static void vc_copylineY416toRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        enum {
                S_DEPTH = 16,
        };
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, S_DEPTH);
        const uint16_t *in = (const void *) src;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 3) {
                comp_type_t y, u, v, r, g, b;

                u = *in++ - (1 << (S_DEPTH - 1));
                y = cfs.y_scale * (*in++ - (1 << (S_DEPTH - 4)));
                v = *in++ - (1 << (S_DEPTH - 1));
                in++;
                r = YCBCR_TO_R(cfs, y, u, v) >> (COMP_BASE + 8U);
                g = YCBCR_TO_G(cfs, y, u, v) >> (COMP_BASE + 8U);
                b = YCBCR_TO_B(cfs, y, u, v) >> (COMP_BASE + 8U);
                r = CLAMP_FULL(r, 8);
                g = CLAMP_FULL(g, 8);
                b = CLAMP_FULL(b, 8);

                *dst++ = r;
                *dst++ = g;
                *dst++ = b;
        }
}

static void vc_copylineY416toRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        enum {
                S_DEPTH = 16,
        };
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 4 == 0);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, S_DEPTH);
        const uint16_t *in = (const void *) src;
        uint32_t *out = (void *) dst;
        const uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 4) {
                comp_type_t y, u, v, r, g, b;

                u = *in++ - (1 << (S_DEPTH - 1));
                y = cfs.y_scale * (*in++ - (1 << (S_DEPTH - 4)));
                v = *in++ - (1 << (S_DEPTH - 1));
                in++;
                r = YCBCR_TO_R(cfs, y, u, v) >> (COMP_BASE + 8U);
                g = YCBCR_TO_G(cfs, y, u, v) >> (COMP_BASE + 8U);
                b = YCBCR_TO_B(cfs, y, u, v) >> (COMP_BASE + 8U);
                r = CLAMP_FULL(r, 8);
                g = CLAMP_FULL(g, 8);
                b = CLAMP_FULL(b, 8);

                *out++ = alpha_mask | r << rshift | g << gshift | b << bshift;
        }
}

static void vc_copylineRG48toR10k(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % sizeof(uint16_t) == 0);
        assert((uintptr_t) dst % sizeof(uint32_t) == 0);
        const uint16_t *in = (const uint16_t *)(const void *) src;
        uint32_t *out = (uint32_t *)(void *) dst;
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
                *out++ = r << 22U | g << 12U | b << 2U | 0x3FU; /// @todo just a stub
#else
                unsigned r = *in++ >> 6;
                unsigned g = *in++ >> 6;
                unsigned b = *in++ >> 6;
                // B5-B0 XX | G3-G0 B9-B6 | R1-R0 G9-G4 | R9-R2
                *out++ = (b & 0x3FU) << 26U | 0x3000000U | (g & 0xFU) << 20U | (b >> 6U) << 16U | (r & 0x3U) << 14U | (g >> 4U) << 8U | r >> 2U;
#endif
        }
}

static void vc_copylineRG48toRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 3; x += 3) {
                *dst++ = src[1];
                *dst++ = src[3];
                *dst++ = src[5];
                src += 6;
        }
}

static void vc_copylineRG48toRGBA(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        assert((uintptr_t) dst % sizeof(uint32_t) == 0);
        uint32_t *dst32 = (uint32_t *)(void *) dst;
        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
                *dst32++ = alpha_mask | src[1] << rshift | src[3] << gshift | src[5] << bshift;
                src += 6;
        }
}

/**
 * @brief Converts RGB to UYVY.
 * @copydetails vc_copylinev210
 */
static void vc_copylineRGBtoUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        vc_copylineToUYVY(dst, src, dst_len, 0, 1, 2, 3);
}

/**
 * @brief Converts RGB to UYVY using SSE.
 * Uses full scale Rec. 601 YUV (aka JPEG)
 * There can be some inaccuracies due to the use of integer arithmetic
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBtoUYVY_SSE(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
#ifdef __SSSE3__
        __m128i rgb;
        __m128i yuv;

        __m128i r;
        __m128i g;
        __m128i b;

        __m128i y;
        __m128i u;
        __m128i v;

        //BB BB BB BB GR GR GR GR
        __m128i shuffle = _mm_setr_epi8(0, 1, 3, 4, 6, 7, 9, 10, 2, 2, 5, 5, 8, 8, 11, 11);
        __m128i mask = _mm_setr_epi16(0x00FF, 0x00FF, 0x00FF, 0x00FF, 0, 0, 0, 0);
        __m128i lowmask = _mm_setr_epi16(0, 0, 0, 0, 0x00FF, 0x00FF, 0x00FF, 0x00FF);
        __m128i zero = _mm_set1_epi16(0);

        __m128i yrf = _mm_set1_epi16(23);
        __m128i ygf = _mm_set1_epi16(79);
        __m128i ybf = _mm_set1_epi16(8);

        __m128i urf = _mm_set1_epi16(13);
        __m128i ugf = _mm_set1_epi16(43);
        __m128i ubf = _mm_set1_epi16(56);

        __m128i vrf = _mm_set1_epi16(56);
        __m128i vgf = _mm_set1_epi16(51);
        __m128i vbf = _mm_set1_epi16(5);

        __m128i yadd = _mm_set1_epi16(2048);
        __m128i uvadd = _mm_set1_epi16(16384);

        while(dst_len >= 20){
                //Load first 4 pixels
                rgb = _mm_lddqu_si128((__m128i const*)(const void *) src);

                src += 12;
                rgb = _mm_shuffle_epi8(rgb, shuffle);

                r = _mm_and_si128(rgb, mask);
                rgb = _mm_bsrli_si128(rgb, 1);
                //0B BB BB BB BG RG RG RG
                g = _mm_and_si128(rgb, mask);
                rgb = _mm_bsrli_si128(rgb, 7);
                //00 00 00 00 BB BB BB BB
                b = _mm_and_si128(rgb, mask);

                //Load next 4 pixels
                rgb = _mm_lddqu_si128((__m128i const*)(const void *) src);
                src += 12;
                rgb = _mm_shuffle_epi8(rgb, shuffle);
                b = _mm_or_si128(b, _mm_and_si128(rgb, lowmask));
                rgb = _mm_bslli_si128(rgb, 7);
                g = _mm_or_si128(g, _mm_and_si128(rgb, lowmask));
                //0B BB BB BB BG RG RG RG
                rgb = _mm_bslli_si128(rgb, 1);
                r = _mm_or_si128(r, _mm_and_si128(rgb, lowmask));
                //00 00 00 00 BB BB BB BB

                //Compute YUV values
                y = _mm_adds_epi16(yadd, _mm_mullo_epi16(r, yrf));
                y = _mm_adds_epi16(y, _mm_mullo_epi16(g, ygf));
                y = _mm_adds_epi16(y, _mm_mullo_epi16(b, ybf));

                u = _mm_subs_epi16(uvadd, _mm_mullo_epi16(r, urf));
                u = _mm_subs_epi16(u, _mm_mullo_epi16(g, ugf));
                u = _mm_adds_epi16(u, _mm_mullo_epi16(b, ubf));

                v = _mm_adds_epi16(uvadd, _mm_mullo_epi16(r, vrf));
                v = _mm_subs_epi16(v, _mm_mullo_epi16(g, vgf));
                v = _mm_subs_epi16(v, _mm_mullo_epi16(b, vbf));

                y = _mm_srli_epi16(y, 7);
                u = _mm_srli_epi16(u, 7);
                v = _mm_srli_epi16(v, 7);

                u = _mm_hadd_epi16(u, v);
                u = _mm_srli_epi16(u, 1);

                v = _mm_unpackhi_epi16(zero, u);
                u = _mm_unpacklo_epi16(u, zero);

                y = _mm_bslli_si128(y, 1);
                yuv = _mm_or_si128(y, u);
                yuv = _mm_or_si128(yuv, v);

                _mm_storeu_si128((__m128i *)(void *) dst, yuv);
                dst += 16;
                dst_len -= 16;
        }
#endif
        //copy last few pixels
        vc_copylineRGBtoUYVY(dst, src, dst_len, rshift, gshift, bshift);
}

/**
 * @brief Converts RGB to Grayscale using SSE.
 * There can be some inaccuracies due to the use of integer arithmetic
 * @copydetails vc_copylinev210
 */
void vc_copylineRGBtoGrayscale_SSE(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
#ifdef __SSSE3__
        __m128i rgb;

        __m128i r;
        __m128i g;
        __m128i b;

        __m128i y;

        //BB BB BB BB GR GR GR GR
        __m128i inshuffle = _mm_setr_epi8(0, 1, 3, 4, 6, 7, 9, 10, 2, 2, 5, 5, 8, 8, 11, 11);
        __m128i outshuffle = _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 1, 1, 1, 1, 1, 1, 1);
        __m128i mask = _mm_setr_epi16(0x00FF, 0x00FF, 0x00FF, 0x00FF, 0, 0, 0, 0);
        __m128i lowmask = _mm_setr_epi16(0, 0, 0, 0, 0x00FF, 0x00FF, 0x00FF, 0x00FF);

        __m128i yrf = _mm_set1_epi16(23);
        __m128i ygf = _mm_set1_epi16(79);
        __m128i ybf = _mm_set1_epi16(8);

        __m128i yadd = _mm_set1_epi16(2048);

        while(dst_len >= 16){
                //Load first 4 pixels
                rgb = _mm_lddqu_si128((__m128i const*)(const void *) src);

                src += 12;
                rgb = _mm_shuffle_epi8(rgb, inshuffle);

                //BB BB BB BB GR GR GR GR
                r = _mm_and_si128(rgb, mask);
                rgb = _mm_bsrli_si128(rgb, 1);
                //0B BB BB BB BG RG RG RG
                g = _mm_and_si128(rgb, mask);
                rgb = _mm_bsrli_si128(rgb, 7);
                //00 00 00 00 BB BB BB BB
                b = _mm_and_si128(rgb, mask);

                //Load next 4 pixels
                rgb = _mm_lddqu_si128((__m128i const*)(const void *) src);
                src += 12;
                rgb = _mm_shuffle_epi8(rgb, inshuffle);
				//BB BB BB BB GR GR GR GR
                b = _mm_or_si128(b, _mm_and_si128(rgb, lowmask));
                rgb = _mm_bslli_si128(rgb, 7);
				//BG RG RG RG R0 00 00 00
                g = _mm_or_si128(g, _mm_and_si128(rgb, lowmask));
                rgb = _mm_bslli_si128(rgb, 1);
                //GR GR GR GR 00 00 00 00
                r = _mm_or_si128(r, _mm_and_si128(rgb, lowmask));

                //Compute Y values
                y = _mm_adds_epi16(yadd, _mm_mullo_epi16(r, yrf));
                y = _mm_adds_epi16(y, _mm_mullo_epi16(g, ygf));
                y = _mm_adds_epi16(y, _mm_mullo_epi16(b, ybf));

                y = _mm_srli_epi16(y, 7);

                y = _mm_shuffle_epi8(y, outshuffle);

                _mm_storeu_si128((__m128i *)(void *) dst, y);
                dst += 8;
                dst_len -= 8;
        }
#endif
        //copy last few pixels
        register int ri, gi, bi;
        register int y1, y2;
        register uint16_t *d = (uint16_t *)(void *) dst;
        OPTIMIZED_FOR (int x = 0; x <= dst_len - 2; x += 2) {
                ri = *(src++);
                gi = *(src++);
                bi = *(src++);
                y1 = 11993 * ri + 40239 * gi + 4063 * bi + (1<<20);
                ri = *(src++);
                gi = *(src++);
                bi = *(src++);
                y2 = 11993 * ri + 40239 * gi + 4063 * bi + (1<<20);

                *d++ = (CLAMP(y2, 0, (1<<24)-1) >> 16) << 8 |
                       (CLAMP(y1, 0, (1<<24)-1) >> 16);
        }
}

/**
 * @brief Converts BGR to UYVY.
 * @copydetails vc_copylinev210
 */
static void vc_copylineBGRtoUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        vc_copylineToUYVY(dst, src, dst_len, 2, 1, 0, 3);
}

static void
vc_copylineRGBAtoVUYA(unsigned char *__restrict dst,
                      const unsigned char *__restrict src, int dst_len,
                      int rshift, int gshift, int bshift)
{
        (void) rshift, (void) gshift, (void) bshift;
        const struct color_coeffs cfs  = *get_color_coeffs(CS_DFL, DEPTH8);
        while (dst_len > 3) {
                comp_type_t r = *src++;
                comp_type_t g = *src++;
                comp_type_t b = *src++;
                comp_type_t a = *src++;

                *dst++ = (RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) +
                         (1 << (DEPTH8 - 1));
                *dst++ = (RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) +
                         (1 << (DEPTH8 - 1));
                *dst++ =
                    (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) + (1 << (DEPTH8 - 4));
                *dst++ = a;
                dst_len -= 4;
        }
}

/**
 * @brief Converts RGBA to UYVY.
 * @copydetails vc_copylinev210
 *
 * @note
 * not using restricted pointers - vc_copylineR10ktoUYVY uses it in place.
 */
static void vc_copylineRGBAtoUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        vc_copylineToUYVY(dst, src, dst_len, 0, 1, 2, 4);
}

static void vc_copylineR10ktoUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        const unsigned char *const end = dst + dst_len;
        while (dst < end) {
                unsigned char rgb[6];
                rgb[0] = src[0]; // R
                rgb[1] = src[1] << 2 | src[2] >> 6; // G
                rgb[2] = src[2] << 4 | src[3] >> 4; // B
                src += 4;
                rgb[3] = src[0]; // R
                rgb[4] = src[1] << 2 | src[2] >> 6; // G
                rgb[5] = src[2] << 4 | src[3] >> 4; // B
                src += 4;
                vc_copylineRGBtoUYVY(dst, rgb, 4, DEFAULT_R_SHIFT, DEFAULT_G_SHIFT, DEFAULT_B_SHIFT);
                dst += 4;
        }
}

static void vc_copylineRG48toUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        vc_copylineToUYVY(dst, src, dst_len, 1, 3, 5, 6);
}

/**
 * offset of coefficients is 16 bits, 14 bits from RGB is used
 */
static void vc_copylineRG48toV210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        enum {
                D_DEPTH = 10,
        };
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, D_DEPTH);
#define COMP_OFF (COMP_BASE+(16-10))
#define FETCH_BLOCK \
                r = *in++; \
                g = *in++; \
                b = *in++; \
                y1 = (RGB_TO_Y(cfs, r, g, b) >> COMP_OFF) + (1<<6); \
                u = RGB_TO_CB(cfs, r, g, b) >> COMP_OFF; \
                v = RGB_TO_CR(cfs, r, g, b) >> COMP_OFF; \
                r = *in++; \
                g = *in++; \
                b = *in++; \
                y2 = (RGB_TO_Y(cfs, r, g, b) >> COMP_OFF) + (1<<6); \
                u += RGB_TO_CB(cfs, r, g, b) >> COMP_OFF; \
                v += RGB_TO_CR(cfs, r, g, b) >> COMP_OFF; \
                y1 = CLAMP_LIMITED_Y(y1, 10); \
                y2 = CLAMP_LIMITED_Y(y2, 10); \
                u = u / 2 + (1<<9); \
                v = v / 2 + (1<<9); \
                u = CLAMP_LIMITED_CBCR(u, 10); \
                v = CLAMP_LIMITED_CBCR(v, 10);

        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 4 == 0);
        const uint16_t *in = (const uint16_t *)(const void *) src;
        uint32_t *d = (uint32_t *)(void *) dst;
        OPTIMIZED_FOR (int x = 0; x <= (dst_len) - 16; x += 16) {
                comp_type_t y1, y2, u ,v;
                comp_type_t r, g, b;

                FETCH_BLOCK
                *d++ = u | y1 << 10 | v << 20;
                *d = y2;

                FETCH_BLOCK
                *d |= u << 10 | y1 << 20;
                *++d = v | y2 << 10;

                FETCH_BLOCK
                *d |= u << 20;
                *++d = y1 | v << 10 | y2 << 20;
                d++;
        }
#undef COMP_OFF
#undef FETCH_BLOCK
}


static void vc_copylineRG48toY216(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        enum {
                D_DEPTH = 16,
        };
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 2 == 0);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, D_DEPTH);
        const uint16_t *in = (const void *) src;
        uint16_t *d = (void *) dst;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 8) {
                comp_type_t r, g, b;
                comp_type_t y, u, v;
                r = *in++;
                g = *in++;
                b = *in++;
                y = (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) +
                    (1 << (D_DEPTH - 4));
                *d++ = CLAMP_LIMITED_Y(y, 16);
                u = (RGB_TO_CB(cfs, r, g, b) >> COMP_BASE);
                v = (RGB_TO_CR(cfs, r, g, b) >> COMP_BASE);
                r = *in++;
                g = *in++;
                b = *in++;
                u = ((u + (RGB_TO_CB(cfs, r, g, b) >> COMP_BASE)) /
                             2) +
                    (1 << 15);
                *d++ = CLAMP_LIMITED_CBCR(u, 16);
                y    = (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) +
                    (1 << 12);
                *d++ = CLAMP_LIMITED_Y(y, 16);
                v = ((v + (RGB_TO_CR(cfs, r, g, b) >> COMP_BASE)) /
                             2) +
                    (1 << 15);
                *d++ = CLAMP_LIMITED_CBCR(v, 16);
        }
}

static void vc_copylineRG48toY416(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        enum {
                D_DEPTH = 16,
        };
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 2 == 0);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, D_DEPTH);
        const uint16_t *in = (const void *) src;
        uint16_t *d = (void *) dst;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 8) {
                comp_type_t r, g, b;
                r = *in++;
                g = *in++;
                b = *in++;
                comp_type_t u =
                    (RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) +
                    (1 << (D_DEPTH - 1));
                *d++          = CLAMP_LIMITED_CBCR(u, D_DEPTH);
                comp_type_t y =
                    (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) +
                    (1 << (D_DEPTH - 4));
                *d++ = CLAMP_LIMITED_Y(y, D_DEPTH);
                comp_type_t v =
                    (RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) +
                    (1 << (D_DEPTH - 1));
                *d++ = CLAMP_LIMITED_CBCR(v, D_DEPTH);
                *d++ = 0xFFFFU;
        }
}

static void vc_copylineY416toRG48(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        enum {
                S_DEPTH = 16,
        };
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 2 == 0);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, S_DEPTH);
        const uint16_t *in = (const void *) src;
        uint16_t *d = (void *) dst;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 6) {
                comp_type_t u = *in++ - (1 << (S_DEPTH - 1));
                comp_type_t y =
                    cfs.y_scale * (*in++ - (1 << (S_DEPTH - 4)));
                comp_type_t v = *in++ - (1 << (S_DEPTH - 1));
                in++;
                comp_type_t r =
                    YCBCR_TO_R(cfs, y, u, v) >> COMP_BASE;
                comp_type_t g =
                    YCBCR_TO_G(cfs, y, u, v) >> COMP_BASE;
                comp_type_t b =
                    YCBCR_TO_B(cfs, y, u, v) >> COMP_BASE;
                *d++ = CLAMP_FULL(r, 16);
                *d++ = CLAMP_FULL(g, 16);
                *d++ = CLAMP_FULL(b, 16);
        }
}

/**
 * Converts BGR to RGB.
 * @copydetails vc_copylinev210
 */
static void vc_copylineBGRtoRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        vc_copylineRGB(dst, src, dst_len, 16, 8, 0);
}

void vc_memcpy(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        memcpy(dst, src, dst_len);
}

static void vc_copylineRGBAtoR10k(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        struct {
                unsigned r:8;

                unsigned gh:6;
                unsigned p1:2;

                unsigned bh:4;
                unsigned p2:2;
                unsigned gl:2;

                unsigned p3:2;
                unsigned p4:2;
                unsigned bl:4;
        } *d = (void *) dst;

        while (dst_len >= 4) {
                unsigned int r = *(src++);
                unsigned int g = *(src++);
                unsigned int b = *(src++);
                src++;

                d->r = r;
                d->gh = g >> 2U;
                d->gl = g & 0x3U;
                d->bh = b >> 4U;
                d->bl = b & 0xFU;

                d->p1 = 0;
                d->p2 = 0;
                d->p3 = 0x3U;
                d->p4 = 0;

                d++;
                dst_len -= 4;
        }
}

static void vc_copylineUYVYtoV210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        struct {
                unsigned a:10;
                unsigned b:10;
                unsigned c:10;
                unsigned p1:2;
        } *p = (void *)dst;
        while (dst_len >= 4) {
                unsigned int u = *(src++);
                unsigned int y = *(src++);
                unsigned int v = *(src++);

                p->a = u << 2U;
                p->b = y << 2U;
                p->c = v << 2U;
                p->p1 = 0;

                p++;

                dst_len -= 4;
        }
}

static void vc_copylineUYVYtoY216(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        while (dst_len >= 8) {
                *dst++ = 0;
                *dst++ = src[1]; // Y0
                *dst++ = 0;
                *dst++ = src[0]; // U
                *dst++ = 0;
                *dst++ = src[3]; // Y1
                *dst++ = 0;
                *dst++ = src[2]; // V
                src += 4;
                dst_len -= 8;
        }
}

static void vc_copylineUYVYtoY416(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        while (dst_len >= 12) {
                *dst++ = 0;
                *dst++ = src[0]; // U
                *dst++ = 0;
                *dst++ = src[1]; // Y0
                *dst++ = 0;
                *dst++ = src[2]; // V
                *dst++ = 0xFFU;
                *dst++ = 0xFFU;  // A
                *dst++ = 0;
                *dst++ = src[0]; // U
                *dst++ = 0;
                *dst++ = src[3]; // Y1
                *dst++ = 0;
                *dst++ = src[2]; // V
                *dst++ = 0xFFU;
                *dst++ = 0xFFU;  // A
                src += 4;
                dst_len -= 16;
        }
        if (dst_len >= 8) {
                *dst++ = 0;
                *dst++ = src[0]; // U
                *dst++ = 0;
                *dst++ = src[1]; // Y0
                *dst++ = 0;
                *dst++ = src[2]; // V
                *dst++ = 0xFFU;
                *dst++ = 0xFFU;  // A
        }
}

static void
vc_copylineVUYAtoY416(unsigned char *__restrict dst,
                      const unsigned char *__restrict src, int dst_len,
                      int rshift, int gshift, int bshift)
{
        (void) rshift, (void) gshift, (void) bshift;
        const int dst_bs = get_pf_block_bytes(Y416); // 8
        while (dst_len > dst_bs - 1) {
                *dst++ = 0;
                *dst++ = src[1]; // U
                *dst++ = 0;
                *dst++ = src[2]; // Y
                *dst++ = 0;
                *dst++ = src[0]; // V
                *dst++ = 0;
                *dst++ = src[3]; // A
                src += 4;
                dst_len -= dst_bs;
        }
}

static void
vc_copylineVUYAtoUYVY(unsigned char *__restrict dst,
                      const unsigned char *__restrict src, int dst_len,
                      int rshift, int gshift, int bshift)
{
        (void) rshift, (void) gshift, (void) bshift;
        const int dst_bs = get_pf_block_bytes(UYVY); // 4
        while (dst_len > dst_bs - 1) {
                *dst++ = (src[1] + src[5]) / 2; // U
                *dst++ = src[2];                // Y0
                *dst++ = (src[0] + src[4]) / 2; // V
                *dst++ = src[7];                // Y1
                src += 8;
                dst_len -= dst_bs;
        }
}

static void
vc_copylineVUYAtoRGB(unsigned char *__restrict dst,
                     const unsigned char *__restrict src, int dst_len,
                     int rshift, int gshift, int bshift)
{
        (void) rshift, (void) gshift, (void) bshift;
        enum {
                S_DEPTH = 8,
        };
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, S_DEPTH);
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 3) {
                comp_type_t v = *src++ - (1 << (S_DEPTH - 1));
                comp_type_t u = *src++ - (1 << (S_DEPTH - 1));
                comp_type_t y = cfs.y_scale * (*src++ - (1 << (S_DEPTH - 4)));
                src++; // alpha
                comp_type_t r = YCBCR_TO_R(cfs, y, u, v) >> COMP_BASE;
                comp_type_t g = YCBCR_TO_G(cfs, y, u, v) >> COMP_BASE;
                comp_type_t b = YCBCR_TO_B(cfs, y, u, v) >> COMP_BASE;
                *dst++ = CLAMP_FULL(r, 8);
                *dst++ = CLAMP_FULL(g, 8);
                *dst++ = CLAMP_FULL(b, 8);
        }
}

static void vc_copylineY216toUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        while (dst_len >= 4) {
                *dst++ = src[3];
                *dst++ = src[1];
                *dst++ = src[7];
                *dst++ = src[5];
                src += 8;
                dst_len -= 4;
        }
}

static void vc_copylineY416toUYVY(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        while (dst_len >= 4) {
                *dst++ = (src[1] + src[9]) / 2; // U
                *dst++ = src[3]; // Y0
                *dst++ = (src[5] + src[13]) / 2; // V
                *dst++ = src[11]; // Y1
                src += 16;
                dst_len -= 4;
        }
}

static void vc_copylineY216toV210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 4 == 0);
        OPTIMIZED_FOR (int x = 0; x < (dst_len + 15) / 16; ++x) {
                const uint16_t *s = (const uint16_t *)(const void *) (src + x * 24);
                uint32_t *d = (uint32_t *)(void *) (dst + x * 16);
                uint16_t y1, u, y2, v;
                y1 = s[0];
                u = s[1];
                y2 = s[2];
                v = s[3];
                d[0] = u >> 6U | y1 >> 6U << 10U | v >> 6U << 20U;
                y1 = s[4];
                u = s[5];
                d[1] = y2 >> 6U | u >> 6U << 10U | y1 >> 6U << 20U;
                y2 = s[6];
                v = s[7];
                y1 = s[8];
                u = s[9];
                d[2] = v >> 6U | y2 >> 6U << 10U | u >> 6U << 20U;
                y2 = s[10];
                v = s[11];
                d[3] = y1 >> 6U | v >> 6U << 10U | y2 >> 6U << 20U;
        }
}

static void vc_copylineV210toY216(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) dst % 2 == 0);
        assert((uintptr_t) src % 4 == 0);
        OPTIMIZED_FOR (int x = 0; x < dst_len / 24; ++x) {
                const uint32_t *s = (const void *) (src + x * 16);
                uint16_t *d = (void *) (dst + x * 24);
                uint32_t tmp = *s++;
                unsigned u = (tmp & 0x3FFU) << 6U;
                unsigned y0 = ((tmp >> 10U) & 0x3FFU) << 6U;
                unsigned v = ((tmp >> 20U) & 0x3FFU) << 6U;
                tmp = *s++;
                unsigned y1 = (tmp & 0x3FFU) << 6U;
                *d++ = y0;
                *d++ = u;
                *d++ = y1;
                *d++ = v;
                u = ((tmp >> 10U) & 0x3FFU) << 6U;
                y0 = ((tmp >> 20U) & 0x3FFU) << 6U;
                tmp = *s++;
                v = (tmp & 0x3FFU) << 6U;
                y1 = ((tmp >> 10U) & 0x3FFU) << 6U;
                *d++ = y0;
                *d++ = u;
                *d++ = y1;
                *d++ = v;
                u = ((tmp >> 20U) & 0x3FFU) << 6U;
                tmp = *s++;
                y0 = (tmp & 0x3FFU) << 6U;
                v = ((tmp >> 10U) & 0x3FFU) << 6U;
                y1 = ((tmp >> 20U) & 0x3FFU) << 6U;
                *d++ = y0;
                *d++ = u;
                *d++ = y1;
                *d++ = v;
        }
}

static void vc_copylineV210toY416(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) dst % 2 == 0);
        assert((uintptr_t) src % 4 == 0);
        OPTIMIZED_FOR (int x = 0; x < dst_len / 48; ++x) {
                const uint32_t *s = (const void *) (src + x * 16);
                uint16_t *d = (void *) (dst + x * 48);
                uint16_t u, v;
                uint32_t tmp;
                tmp = *s++;
                u = (tmp & 0x3FFU) << 6U;
                *d++ = u;                             // 1 U
                *d++ = ((tmp >> 10U) & 0x3FFU) << 6U; //   Y
                v = ((tmp >> 20U) & 0x3FFU) << 6U;
                *d++ = v;                             //   V
                *d++ = 0xFFFFU;                       //   A
                *d++ = u;                             // 2 U
                tmp = *s++;
                *d++ = (tmp & 0x3FFU) << 6U;          //   Y
                *d++ = v;                             //   V
                *d++ = 0xFFFFU;                       //   A
                u = ((tmp >> 10U) & 0x3FFU) << 6U;
                *d++ = u;                             // 3 U
                *d++ = ((tmp >> 20U) & 0x3FFU) << 6U; //   Y
                tmp = *s++;
                v = (tmp & 0x3FFU) << 6U;
                *d++ = v;                             //   V
                *d++ = 0xFFFFU;                       //   A
                *d++ = u;                             // 4 U
                *d++ = ((tmp >> 10U) & 0x3FFU) << 6U; //   Y
                *d++ = v;                             //   V
                *d++ = 0xFFFFU;                       //   A
                u = ((tmp >> 20U) & 0x3FFU) << 6U;
                *d++ = u;                             // 5 U
                tmp = *s++;
                *d++ = (tmp & 0x3FFU) << 6U;          //   Y
                v = ((tmp >> 10U) & 0x3FFU) << 6U;
                *d++ = v;                             //   V
                *d++ = 0xFFFFU;                       //   A
                *d++ = u;                             // 6 U
                *d++ = ((tmp >> 20U) & 0x3FFU) << 6U; //   Y
                *d++ = v;                             //   V
                *d++ = 0xFFFFU;                       //   A
        }
}

static void vc_copylineV210toRGB(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        enum {
                IDEPTH    = 8, // 10, but cherry-picking 8 bits
                Y_SHIFT   = 1 << (IDEPTH - 4),
                C_SHIFT   = 1 << (IDEPTH - 1),
                ODEPTH    = 8,
                PIX_COUNT = 6,
                RGB_BPP   = 3,
                OUT_BL_SZ = PIX_COUNT * RGB_BPP,
        };
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, IDEPTH);
#define WRITE_YUV_AS_RGB(y, u, v) \
        (y) = cfs.y_scale * ((y) - Y_SHIFT); \
        val = (YCBCR_TO_R(cfs, (y), (u), (v)) >> (COMP_BASE)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH); \
        val = (YCBCR_TO_G(cfs, (y), (u), (v)) >> (COMP_BASE)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH); \
        val = (YCBCR_TO_B(cfs, (y), (u), (v)) >> (COMP_BASE)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH);

        // read 8 bits from v210 directly
#define DECLARE_LOAD_V210_COMPONENTS(a, b, c) \
        comp_type_t a = (src[1] & 0x3) << 6 | src[0] >> 2;\
        comp_type_t b = (src[2] & 0xF) << 4 | src[1] >> 4;\
        comp_type_t c = (src[3] & 0x3F) << 2 | src[2] >> 6;\

        OPTIMIZED_FOR (int x = 0; x < dst_len; x += OUT_BL_SZ){
                DECLARE_LOAD_V210_COMPONENTS(u01, y0, v01);
                src += 4;
                DECLARE_LOAD_V210_COMPONENTS(y1, u23, y2);
                src += 4;
                DECLARE_LOAD_V210_COMPONENTS(v23, y3, u45);
                src += 4;
                DECLARE_LOAD_V210_COMPONENTS(y4, v45, y5);
                src += 4;

                comp_type_t val = 0;

                u01 -= C_SHIFT;
                v01 -= C_SHIFT;
                u23 -= C_SHIFT;
                v23 -= C_SHIFT;
                u45 -= C_SHIFT;
                v45 -= C_SHIFT;
                WRITE_YUV_AS_RGB(y0, u01, v01);
                WRITE_YUV_AS_RGB(y1, u01, v01);
                WRITE_YUV_AS_RGB(y2, u23, v23);
                WRITE_YUV_AS_RGB(y3, u23, v23);
                WRITE_YUV_AS_RGB(y4, u45, v45);
                WRITE_YUV_AS_RGB(y5, u45, v45);
        }
#undef WRITE_YUV_AS_RGB
#undef DECLARE_LOAD_V210_COMPONENTS
}

static void
vc_copylineV210toRG48(unsigned char *__restrict d,
                      const unsigned char *__restrict src, int dst_len,
                      int rshift, int gshift, int bshift)
{
        uint16_t *dst = (void *) d;
        enum {
                IDEPTH    = 10,
                Y_SHIFT   = 1 << (IDEPTH - 4),
                C_SHIFT   = 1 << (IDEPTH - 1),
                ODEPTH    = 16,
                DIFF_BPP  = ODEPTH - IDEPTH,
                PIX_COUNT = 6,
                RG48_BPP  = 6,
                OUT_BL_SZ = PIX_COUNT * RG48_BPP,
        };
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, IDEPTH);
#define WRITE_YUV_AS_RGB(y, u, v) \
        (y) = cfs.y_scale * ((y) - Y_SHIFT); \
        val = (YCBCR_TO_R(cfs, (y), (u), (v)) >> (COMP_BASE - DIFF_BPP)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH); \
        val = (YCBCR_TO_G(cfs, (y), (u), (v)) >> (COMP_BASE - DIFF_BPP)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH); \
        val = (YCBCR_TO_B(cfs, (y), (u), (v)) >> (COMP_BASE - DIFF_BPP)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH);

        // read 8 bits from v210 directly
#define DECLARE_LOAD_V210_COMPONENTS(a, b, c) \
        comp_type_t a = (src[1] & 0x3) << 8 | src[0];\
        comp_type_t b = (src[2] & 0xF) << 6 | src[1] >> 2;\
        comp_type_t c = (src[3] & 0x3F) << 4 | src[2] >> 4;\

        OPTIMIZED_FOR (int x = 0; x < dst_len; x += OUT_BL_SZ){
                DECLARE_LOAD_V210_COMPONENTS(u01, y0, v01);
                src += 4;
                DECLARE_LOAD_V210_COMPONENTS(y1, u23, y2);
                src += 4;
                DECLARE_LOAD_V210_COMPONENTS(v23, y3, u45);
                src += 4;
                DECLARE_LOAD_V210_COMPONENTS(y4, v45, y5);
                src += 4;

                comp_type_t val = 0;

                u01 -= C_SHIFT;
                v01 -= C_SHIFT;
                u23 -= C_SHIFT;
                v23 -= C_SHIFT;
                u45 -= C_SHIFT;
                v45 -= C_SHIFT;
                WRITE_YUV_AS_RGB(y0, u01, v01);
                WRITE_YUV_AS_RGB(y1, u01, v01);
                WRITE_YUV_AS_RGB(y2, u23, v23);
                WRITE_YUV_AS_RGB(y3, u23, v23);
                WRITE_YUV_AS_RGB(y4, u45, v45);
                WRITE_YUV_AS_RGB(y5, u45, v45);
        }
#undef WRITE_YUV_AS_RGB
#undef DECLARE_LOAD_V210_COMPONENTS
}

static void vc_copylineY416toV210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 4 == 0);
        OPTIMIZED_FOR (int x = 0; x < dst_len / 16; ++x) {
                const uint16_t *s = (const uint16_t *)(const void *) (src + x * 48);
                uint32_t *d = (uint32_t *)(void *) (dst + x * 16);
                uint16_t y1, u, y2, v;
                u = (s[0] + s[4]) / 2;
                y1 = s[1];
                v = (s[2] + s[6]) / 2;
                y2 = s[5];
                d[0] = u >> 6U | y1 >> 6U << 10U | v >> 6U << 20U;
                y1 = s[9];
                u = (s[8] + s[12]) / 2;
                d[1] = y2 >> 6U | u >> 6U << 10U | y1 >> 6U << 20U;
                y2 = s[13];
                v = (s[10] + s[14]) / 2;
                y1 = s[17];
                u = (s[16] + s[20]) / 2;
                d[2] = v >> 6U | y2 >> 6U << 10U | u >> 6U << 20U;
                y2 = s[21];
                v = (s[18] + s[22]) / 2;
                d[3] = y1 >> 6U | v >> 6U << 10U | y2 >> 6U << 20U;
        }
}

struct decoder_item {
        decoder_t decoder;
        codec_t in;
        codec_t out;
};

static const struct decoder_item decoders[] = {
        { vc_copylineDVS10,       DVS10, UYVY },
        { vc_copylinev210,        v210,  UYVY },
        { vc_copylineYUYV,        YUYV,  UYVY },
        { vc_copylineYUYV,        UYVY,  YUYV },
        { vc_copyliner10k,        R10k,  RGBA },
        { vc_copyliner10ktoRG48,  R10k,  RG48 },
        { vc_copyliner10ktoY416,  R10k,  Y416 },
        { vc_copyliner10ktoRGB,   R10k,  RGB },
        { vc_copylineR12L,        R12L,  RGBA },
        { vc_copylineR12LtoRGB,   R12L,  RGB },
        { vc_copylineR12LtoRG48,  R12L,  RG48 },
        { vc_copylineR12LtoR10k,  R12L,  R10k },
        { vc_copylineR12LtoY416,  R12L,  Y416 },
        { vc_copylineR12LtoUYVY,  R12L,  UYVY },
        { vc_copylineRGBAtoR12L,  RGBA,  R12L },
        { vc_copylineRGBtoR12L,   RGB,   R12L },
        { vc_copylineRGBAtoRG48,  RGBA,  RG48 },
        { vc_copylineRGBtoRG48,   RGB,   RG48 },
        { vc_copylineUYVYtoRG48,  UYVY,  RG48 },
        { vc_copylineRG48toR12L,  RG48,  R12L },
        { vc_copylineRG48toR10k,  RG48,  R10k },
        { vc_copylineRG48toRGB,   RG48,  RGB },
        { vc_copylineRG48toRGBA,  RG48,  RGBA },
        { vc_copylineRG48toUYVY,  RG48,  UYVY },
        { vc_copylineRG48toV210,  RG48,  v210 },
        { vc_copylineRG48toY216,  RG48,  Y216 },
        { vc_copylineRG48toY416,  RG48,  Y416 },
        { vc_copylineY416toRG48,  Y416,  RG48 },
        { vc_copylineRGBA,        RGBA,  RGBA },
        { vc_copylineDVS10toV210, DVS10, v210 },
        { vc_copylineRGBAtoRGB,   RGBA,  RGB },
        { vc_copylineRGBtoRGBA,   RGB,   RGBA },
        { vc_copylineRGBtoUYVY,   RGB,   UYVY },
        { vc_copylineRGBAtoVUYA,  RGBA,  VUYA},
        { vc_copylineUYVYtoRGB,   UYVY,  RGB },
        { vc_copylineUYVYtoRGBA,  UYVY,  RGBA },
        { vc_copylineYUYVtoRGB,   YUYV,  RGB },
        { vc_copylineBGRtoUYVY,   BGR,   UYVY },
        { vc_copylineR10ktoUYVY,  R10k,  UYVY },
        { vc_copylineRGBAtoUYVY,  RGBA,  UYVY },
        { vc_copylineBGRtoRGB,    BGR,   RGB },
        { vc_copylineRGB,         RGB,   RGB },
        { vc_copylineRGBAtoR10k,  RGBA,  R10k },
        { vc_copylineUYVYtoV210,  UYVY,  v210 },
        { vc_copylineUYVYtoY216,  UYVY,  Y216 },
        { vc_copylineUYVYtoY416,  UYVY,  Y416 },
        { vc_copylineVUYAtoY416,  VUYA,  Y416 },
        { vc_copylineVUYAtoUYVY,  VUYA,  UYVY },
        { vc_copylineVUYAtoRGB,   VUYA,  RGB  },
        { vc_copylineY216toUYVY,  Y216,  UYVY },
        { vc_copylineY216toV210,  Y216,  v210 },
        { vc_copylineY416toUYVY,  Y416,  UYVY },
        { vc_copylineY416toV210,  Y416,  v210 },
        { vc_copylineY416toR12L,  Y416,  R12L },
        { vc_copylineY416toR10k,  Y416,  R10k },
        { vc_copylineY416toRGB,   Y416,  RGB },
        { vc_copylineY416toRGBA,  Y416,  RGBA },
        { vc_copylineV210toY216,  v210,  Y216 },
        { vc_copylineV210toY416,  v210,  Y416 },
        { vc_copylineV210toRGB,   v210,  RGB  },
        { vc_copylineV210toRG48,  v210,  RG48 },
};

/**
 * Returns line decoder for specifiedn input and output codec.
 *
 * If in == out, vc_memcpy is returned.
 */
decoder_t get_decoder_from_to(codec_t in, codec_t out) {
        if (in == out &&
                        (out != RGBA && out != RGB)) { // vc_copylineRGB[A] may change shift
                return vc_memcpy;
        }

        for (unsigned int i = 0; i < sizeof(decoders)/sizeof(struct decoder_item); ++i) {
                if (decoders[i].in == in && decoders[i].out == out) {
                        return decoders[i].decoder;
                }
        }

        MSG(DEBUG, "No decoder from %s to %s!\n", get_codec_name(in),
            get_codec_name(out));
        return NULL;
}

// less is better
static QSORT_S_COMP_DEFINE(best_decoder_cmp, a, b, desc_src) {
        codec_t codec_a = *(const codec_t *) a;
        codec_t codec_b = *(const codec_t *) b;
        struct pixfmt_desc *src_desc = desc_src;
        struct pixfmt_desc desc_a = get_pixfmt_desc(codec_a);
        struct pixfmt_desc desc_b = get_pixfmt_desc(codec_b);

        int ret = compare_pixdesc(&desc_a, &desc_b, src_desc);
        if (ret != 0) {
                return ret;
        }

        return (int) codec_a - (int) codec_b;
}

/**
 * Returns best decoder for input codec.
 *
 * If in == out, vc_memcpy is returned.
 */
decoder_t get_best_decoder_from(codec_t in, const codec_t *out_candidates, codec_t *out)
{
        if (codec_is_in_set(in, out_candidates) && (in != RGBA && in != RGB)) { // vc_copylineRGB[A] may change shift
                *out = in;
                return vc_memcpy;
        }

        codec_t candidates[VIDEO_CODEC_END];
        const codec_t *it = out_candidates;
        size_t count = 0;
        while (*it != VIDEO_CODEC_NONE) {
                if (get_decoder_from_to(in, *it)) {
                        assert(count < VIDEO_CODEC_END && "Too much codecs, some used multiple times!");
                        candidates[count++] = *it;
                }
                it++;
        }
        if (count == 0) {
                return NULL;
        }
        struct pixfmt_desc src_desc = get_pixfmt_desc(in);
        qsort_s(candidates, count, sizeof(codec_t), best_decoder_cmp, &src_desc);
        *out = candidates[0];
        return get_decoder_from_to(in, *out);
}

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

/* vim: set expandtab sw=8: */
