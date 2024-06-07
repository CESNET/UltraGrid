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
 * @brief This file contains video codec-related functions.
 *
 * This file contains video codecs' metadata and helper
 * functions as well as pixelformat converting functions.
 */
/* Copyright (c) 2005-2023 CESNET z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H
#include "config_unix.h"
#include "config_win32.h"

#include <assert.h>

#include "color.h"
#include "compat/qsort_s.h"
#include "debug.h"
#include "pixfmt_conv.h"
#include "utils/macros.h" // to_fourcc, OPTIMEZED_FOR, CLAMP
#include "video_codec.h"

#ifdef __SSSE3__
#include "tmmintrin.h"
#endif

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

/**
 * @brief Converts v210 to UYVY
 * @param[out] dst     4-byte aligned output buffer where UYVY will be stored
 * @param[in]  src     4-byte aligned input buffer containing v210 (by definition of v210
 *                     should be even aligned to 16B boundary)
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
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
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
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
        assert((uintptr_t) dst % 2 == 0);
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
                comp_type_t u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE) + (1<<15);
                *d++ = CLAMP_LIMITED_CBCR(u, 16);
                comp_type_t y = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE) + (1<<12);
                *d++ = CLAMP_LIMITED_Y(y, 16);
                comp_type_t v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE) + (1<<15);
                *d++ = CLAMP_LIMITED_CBCR(v, 16);
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
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
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

static void
vc_copylineR12L(unsigned char *dst, const unsigned char *src, int dstlen, int rshift, int gshift, int bshift);
static void 
vc_copylineR12LtoRGB_SSE(unsigned char *dst, const unsigned char *src, int dstlen, int rshift, int gshift, int bshift);

// #ifdef __SSSE3__
#if 1
#include <immintrin.h>
#include <assert.h>
static void
vc_copylineR12LtoRGB_SSE(unsigned char *dst, const unsigned char *src, int dstlen, int rshift, int gshift, int bshift)
{
        // assert(false); // verify that the function is executed

        // Map 3 bytes of input to 2 bytes of output
        // byte whose high nibble I need -> low nibble of 1st byte
        // byte whose low nibble I need  -> high nibble of 1st byte
        // whole 3rd byte                -> whole 2nd byte
        
        #define Z 0x80 // clear dst byte in shuffle
        #define F 0xff

        __m128i leftmask              = _mm_setr_epi8(F, 0, 0, F, 0, 0, F, 0, 0, F, 0, 0, F, 0, 0, F);
        __m128i centermask            = _mm_setr_epi8(0, F, 0, 0, F, 0, 0, F, 0, 0, F, 0, 0, F, 0, 0);
        __m128i rightmask             = _mm_setr_epi8(0, 0, F, 0, 0, F, 0, 0, F, 0, 0, F, 0, 0, F, 0);

        __m128i unpack_hilo_to_odd0   = _mm_setr_epi8(0,  Z,  3,  Z,  6,  Z,  9,  Z, 12,  Z, 15,  Z,  2,  Z,  5,  Z);
        __m128i unpack_hilo_to_odd1   = _mm_setr_epi8(8,  Z, 11,  Z, 14,  Z,  1,  Z,  4,  Z,  7,  Z, 10,  Z, 13,  Z);
        __m128i unpack_lohi_to_odd0   = _mm_setr_epi8(1,  Z,  4,  Z,  7,  Z, 10,  Z, 13,  Z,  0,  Z,  3,  Z,  6,  Z);
        __m128i unpack_lohi_to_odd1   = _mm_setr_epi8(9,  Z, 12,  Z, 15,  Z,  2,  Z,  5,  Z,  8,  Z, 11,  Z, 14,  Z);
        __m128i unpack_sameb_to_even0 = _mm_setr_epi8(Z,  2,  Z,  5,  Z,  8,  Z, 11,  Z, 14,  Z,  1,  Z,  4,  Z,  7);
        __m128i unpack_sameb_to_even1 = _mm_setr_epi8(Z, 10,  Z, 13,  Z,  0,  Z,  3,  Z,  6,  Z,  9,  Z, 12,  Z, 15);

        #undef Z
        #undef F

        int x;
        OPTIMIZED_FOR (x = 0; x <= dstlen - 32; x += 32) {
                __m128i chunk0, chunk1, chunk2;
                chunk0 = _mm_lddqu_si128((__m128i const*)(const void *)  src);
                chunk1 = _mm_lddqu_si128((__m128i const*)(const void *) (src + 16));
                chunk2 = _mm_lddqu_si128((__m128i const*)(const void *) (src + 32));

        #ifdef WORDS_BIGENDIAN
                __m128i shuffle_BEtoLE = _mm_setr_epi8(3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12);
                chunk0 = _mm_shuffle_epi8(chunk0, shuffle_BEtoLE);
                chunk1 = _mm_shuffle_epi8(chunk1, shuffle_BEtoLE);
                chunk2 = _mm_shuffle_epi8(chunk2, shuffle_BEtoLE);
        #endif

                __m128i hitolo; // positions ≡ 0 (mod 3)
                {
                        __m128i hitolo0, hitolo1, hitolo2;
                        hitolo0 = _mm_and_si128(chunk0, leftmask);   //  0 = 0*3  =  0 + 0
                        hitolo1 = _mm_and_si128(chunk1, rightmask);  // 18 = 6*3  = 16 + 2
                        hitolo2 = _mm_and_si128(chunk2, centermask); // 33 = 11*3 = 32 + 1
                        hitolo  = _mm_or_si128(hitolo0, hitolo1);
                        hitolo  = _mm_or_si128(hitolo, hitolo2);
                }

                __m128i lotohi; // positions ≡ 1 (mod 3)
                {
                        __m128i lotohi0, lotohi1, lotohi2;
                        lotohi0 = _mm_and_si128(chunk0, centermask); //  1 = 0*3 + 1  =  0 + 1
                        lotohi1 = _mm_and_si128(chunk1, leftmask);   // 16 = 5*3 + 1  = 16 + 0
                        lotohi2 = _mm_and_si128(chunk2, rightmask);  // 34 = 11*3 + 1 = 32 + 2
                        lotohi  = _mm_or_si128(lotohi0, lotohi1);
                        lotohi  = _mm_or_si128(lotohi, lotohi2);
                }

                __m128i copybyte; // positions ≡ 2 (mod 3)
                {
                        __m128i copybyte0, copybyte1, copybyte2;
                        copybyte0 = _mm_and_si128(chunk0, rightmask);   //  2 = 0 * 3 + 2  = 0 + 2
                        copybyte1 = _mm_and_si128(chunk1, centermask);  // 17 = 5 * 3 + 2  = 16 + 1
                        copybyte2 = _mm_and_si128(chunk2, leftmask);    // 32 = 10 * 3 + 2 = 32 + 0
                        copybyte  = _mm_or_si128(copybyte0, copybyte1);
                        copybyte  = _mm_or_si128(copybyte, copybyte2);
                }

                // uninterleave, moving to correct locations
                __m128i lotohi_unp0, lotohi_unp1, hitolo_unp0, hitolo_unp1,
                        copybyte_unp0, copybyte_unp1;
                hitolo_unp0   = _mm_shuffle_epi8(hitolo,   unpack_hilo_to_odd0);
                hitolo_unp1   = _mm_shuffle_epi8(hitolo,   unpack_hilo_to_odd1);
                lotohi_unp0   = _mm_shuffle_epi8(lotohi,   unpack_lohi_to_odd0);
                lotohi_unp1   = _mm_shuffle_epi8(lotohi,   unpack_lohi_to_odd1);
                copybyte_unp0 = _mm_shuffle_epi8(copybyte, unpack_sameb_to_even0);
                copybyte_unp1 = _mm_shuffle_epi8(copybyte, unpack_sameb_to_even1);

                // actually bitshift low -> high  and high -> low
                lotohi_unp0 = _mm_slli_epi16(lotohi_unp0, 4);
                lotohi_unp1 = _mm_slli_epi16(lotohi_unp1, 4);
                hitolo_unp0 = _mm_srli_epi16(hitolo_unp0, 4);
                hitolo_unp1 = _mm_srli_epi16(hitolo_unp1, 4);

                // assemble
                __m128i res0, res1;
                res0 = _mm_or_si128(copybyte_unp0, lotohi_unp0);
                res0 = _mm_or_si128(res0,          hitolo_unp0);
                res1 = _mm_or_si128(copybyte_unp1, lotohi_unp1);
                res1 = _mm_or_si128(res1,          hitolo_unp1);


                // store
                _mm_storeu_si128((__m128i_u *)  dst,       res0);
                _mm_storeu_si128((__m128i_u *) (dst + 16), res1);

                src += 48;
                dst += 32;
        }
        // copy leftover bytes
        dstlen -= x;

        vc_copylineR12LtoRGB(dst, src, dstlen, rshift, gshift, bshift);
}

#endif // __SSSE3__


/**
 * @brief Converts from R12L to RGBA
 *
 * @param[out] dst     4B-aligned buffer that will contain result
 * @param[in]  src     buffer containing pixels in R12L
 * @param[in]  dstlen  length of data that should be writen to dst buffer (in bytes)
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
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
 * @param[in]  rshift  destination red shift
 * @param[in]  gshift  destination green shift
 * @param[in]  bshift  destination blue shift
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

/* TODO: undo it - currently this decoder is broken */
#if 0 /* !(HAVE_MACOSX || HAVE_32B_LINUX) */

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

#endif                          /* !(HAVE_MACOSX || HAVE_32B_LINUX) */

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
 * @param[in]  dst_len length of data that should be writen to dst buffer (in bytes)
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
void vc_copylineABGRtoRGB(unsigned char * __restrict dst2, const unsigned char * __restrict src2, int dst_len, int rshift, int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        vc_copylineRGBAtoRGBwithShift(dst2, src2, dst_len, 16, 8, 0);
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
        vc_copylineRGBAtoRGBwithShift(dst, src, dst_len, 0, 8, 16);
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
        register unsigned int r, g, b;
        register uint32_t *d = (uint32_t *)(void *) dst;
        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);

        OPTIMIZED_FOR (int x = 0; x <= dst_len - 4; x += 4) {
                r = *src++;
                g = *src++;
                b = *src++;
                
                *d++ = alpha_mask | (r << rshift) | (g << gshift) | (b << bshift);
        }
}

/**
 * @brief Converts RGB(A) into UYVY
 *
 * Uses Rec. 709 with standard SDI ceiling and floor
 * @copydetails vc_copyliner10k
 * @param[in] roff     red offset in bytes (0 for RGB)
 * @param[in] goff     green offset in bytes (1 for RGB)
 * @param[in] boff     blue offset in bytes (2 for RGB)
 * @param[in] pix_size source pixel size (3 for RGB, 4 for RGBA)
 */
#define vc_copylineToUYVY709(dst, src, dst_len, roff, goff, boff, pix_size) {\
        register uint32_t *d = (uint32_t *)(void *) dst;\
        OPTIMIZED_FOR (int x = 0; x <= (dst_len) - 4; x += 4) {\
                int r, g, b;\
                int y1, y2, u ,v;\
                r = src[roff];\
                g = src[goff];\
                b = src[boff];\
                src += pix_size;\
                y1 = 11993 * r + 40239 * g + 4063 * b + (1<<20);\
                u  = -6619 * r -22151 * g + 28770 * b;\
                v  = 28770 * r - 26149 * g - 2621 * b;\
                r = src[roff];\
                g = src[goff];\
                b = src[boff];\
                src += pix_size;\
                y2 = 11993 * r + 40239 * g + 4063 * b + (1<<20);\
                u += -6619 * r -22151 * g + 28770 * b;\
                v += 28770 * r - 26149 * g - 2621 * b;\
                u = u / 2 + (1<<23);\
                v = v / 2 + (1<<23);\
\
                *d++ = (CLAMP(y2, 0, (1<<24)-1) >> 16) << 24 |\
                        (CLAMP(v, 0, (1<<24)-1) >> 16) << 16 |\
                        (CLAMP(y1, 0, (1<<24)-1) >> 16) << 8 |\
                        (CLAMP(u, 0, (1<<24)-1) >> 16);\
        }\
}

/**
 * @brief Converts 8-bit YCbCr (packed 4:2:2 in 32-bit) word to RGB.
 *
 * Converts 8-bit YCbCr (packed 4:2:2 in 32-bit word to RGB. Offset of YCbCr
 * components can be given by parameters (in bytes). This macro is used by
 * vc_copylineUYVYtoRGB() and vc_copylineYUYVtoRGB().
 *
 * Uses Rec. 709 with standard SDI ceiling and floor
 *
 * @todo make it faster if needed
 * @param rgb16   true if output is 16-bit RGB, otherwise false
 */
#define copylineYUVtoRGB(dst, src, dst_len, y1_off, y2_off, u_off, v_off, rgb16) {\
        OPTIMIZED_FOR (int x = 0; x <= (dst_len) - 6 * (1 + (rgb16)); x += 6 * (1 + (rgb16))) {\
                register int y1 = (src)[y1_off];\
                register int y2 = (src)[y2_off];\
                register int u = (src)[u_off];\
                register int v = (src)[v_off];\
                int val;\
                src += 4;\
                if (rgb16) *(dst)++ = 0;\
                val = 1.164 * (y1 - 16) + 1.793 * (v - 128);\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = 1.164 * (y1 - 16) - 0.534 * (v - 128) - 0.213 * (u - 128);\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = 1.164 * (y1 - 16) + 2.115 * (u - 128);\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = 1.164 * (y2 - 16) + 1.793 * (v - 128);\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = 1.164 * (y2 - 16) - 0.534 * (v - 128) - 0.213 * (u - 128);\
                *(dst)++ = CLAMP(val, 0, 255);\
                if (rgb16) *(dst)++ = 0;\
                val = 1.164 * (y2 - 16) + 2.115 * (u - 128);\
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
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += OUT_BL_SZ) {
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
}

static void vc_copylineR12LtoY416(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        assert((uintptr_t) dst % sizeof(uint16_t) == 0);
        uint16_t *d = (void *) dst;
#define WRITE_RES \
                u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE) + (1<<15); \
                *d++ = CLAMP_LIMITED_CBCR(u, 16); \
                y = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE) + (1<<12); \
                *d++ = CLAMP_LIMITED_Y(y, 16); \
                v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE) + (1<<15); \
                *d++ = CLAMP_LIMITED_CBCR(v, 16); \
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
        uint8_t *d = (void *) dst;
#define WRITE_RES \
        { \
                comp_type_t u = ((RGB_TO_CB_709_SCALED(r1, g1, b1) + \
                                  RGB_TO_CB_709_SCALED(r2, g2, b2)) >> \
                                 (COMP_BASE + D_DPTH + 1)) + \
                                COFF; \
                *d++          = CLAMP_LIMITED_CBCR(u, D_DPTH); \
                comp_type_t y = (RGB_TO_Y_709_SCALED(r1, g1, b1) >> \
                                 (COMP_BASE + D_DPTH)) + \
                                YOFF; \
                *d++          = CLAMP_LIMITED_Y(y, D_DPTH); \
                comp_type_t v = ((RGB_TO_CR_709_SCALED(r1, g1, b1) + \
                                  RGB_TO_CR_709_SCALED(r2, g2, b2)) >> \
                                 (COMP_BASE + D_DPTH + 1)) + \
                                COFF; \
                *d++ = CLAMP_LIMITED_CBCR(v, D_DPTH); \
                y    = (RGB_TO_Y_709_SCALED(r2, g2, b2) >> \
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
#define GET_NEXT \
        u = *in++ - (1<<15); \
        y = Y_SCALE * (*in++ - (1<<12)); \
        v = *in++ - (1<<15); \
        in++; \
        r = (YCBCR_TO_R_709_SCALED(y, u, v) >> (COMP_BASE + 4U)); \
        g = (YCBCR_TO_G_709_SCALED(y, u, v) >> (COMP_BASE + 4U)); \
        b = (YCBCR_TO_B_709_SCALED(y, u, v) >> (COMP_BASE + 4U)); \
        r = CLAMP_FULL(r, 12); \
        g = CLAMP_FULL(g, 12); \
        b = CLAMP_FULL(b, 12);

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
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        const uint16_t *in = (const void *) src;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 4) {
                comp_type_t y, u, v, r, g, b;

                u = *in++ - (1<<15);
                y = Y_SCALE * (*in++ - (1<<12));
                v = *in++ - (1<<15);
                in++;
                r = (YCBCR_TO_R_709_SCALED(y, u, v) >> (COMP_BASE + 6U));
                g = (YCBCR_TO_G_709_SCALED(y, u, v) >> (COMP_BASE + 6U));
                b = (YCBCR_TO_B_709_SCALED(y, u, v) >> (COMP_BASE + 6U));
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
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        const uint16_t *in = (const void *) src;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 3) {
                comp_type_t y, u, v, r, g, b;

                u = *in++ - (1<<15);
                y = Y_SCALE * (*in++ - (1<<12));
                v = *in++ - (1<<15);
                in++;
                r = (YCBCR_TO_R_709_SCALED(y, u, v) >> (COMP_BASE + 8U));
                g = (YCBCR_TO_G_709_SCALED(y, u, v) >> (COMP_BASE + 8U));
                b = (YCBCR_TO_B_709_SCALED(y, u, v) >> (COMP_BASE + 8U));
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
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 4 == 0);
        const uint16_t *in = (const void *) src;
        uint32_t *out = (void *) dst;
        const uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << rshift) ^ (0xFFU << gshift) ^ (0xFFU << bshift);
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 4) {
                comp_type_t y, u, v, r, g, b;

                u = *in++ - (1<<15);
                y = Y_SCALE * (*in++ - (1<<12));
                v = *in++ - (1<<15);
                in++;
                r = (YCBCR_TO_R_709_SCALED(y, u, v) >> (COMP_BASE + 8U));
                g = (YCBCR_TO_G_709_SCALED(y, u, v) >> (COMP_BASE + 8U));
                b = (YCBCR_TO_B_709_SCALED(y, u, v) >> (COMP_BASE + 8U));
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
#ifdef WORDS_BIGENDIAN
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
        vc_copylineToUYVY709(dst, src, dst_len, 0, 1, 2, 3);
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
        vc_copylineToUYVY709(dst, src, dst_len, 2, 1, 0, 3);
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
        vc_copylineToUYVY709(dst, src, dst_len, 0, 1, 2, 4);
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
        vc_copylineToUYVY709(dst, src, dst_len, 1, 3, 5, 6);
}

/**
 * offset of coefficients is 16 bits, 14 bits from RGB is used
 */
static void vc_copylineRG48toV210(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
#define COMP_OFF (COMP_BASE+(16-10))
#define FETCH_BLOCK \
                r = *in++; \
                g = *in++; \
                b = *in++; \
                y1 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_OFF) + (1<<6); \
                u = RGB_TO_CB_709_SCALED(r, g, b) >> COMP_OFF; \
                v = RGB_TO_CR_709_SCALED(r, g, b) >> COMP_OFF; \
                r = *in++; \
                g = *in++; \
                b = *in++; \
                y2 = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_OFF) + (1<<6); \
                u += RGB_TO_CB_709_SCALED(r, g, b) >> COMP_OFF; \
                v += RGB_TO_CR_709_SCALED(r, g, b) >> COMP_OFF; \
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
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 2 == 0);
        const uint16_t *in = (const void *) src;
        uint16_t *d = (void *) dst;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 8) {
                comp_type_t r, g, b;
                comp_type_t y, u, v;
                r = *in++;
                g = *in++;
                b = *in++;
                y = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE) + (1<<12);
                *d++ = CLAMP_LIMITED_Y(y, 16);
                u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE);
                v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE);
                r = *in++;
                g = *in++;
                b = *in++;
                u = (u + (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE) / 2) + (1<<15);
                *d++ = CLAMP_LIMITED_CBCR(u, 16);
                y = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE) + (1<<12);
                *d++ = CLAMP_LIMITED_Y(y, 16);
                v = (v + (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE) / 2) + (1<<15);
                *d++ = CLAMP_LIMITED_CBCR(v, 16);
        }
}

static void vc_copylineRG48toY416(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 2 == 0);
        const uint16_t *in = (const void *) src;
        uint16_t *d = (void *) dst;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 8) {
                comp_type_t r, g, b;
                r = *in++;
                g = *in++;
                b = *in++;
                comp_type_t u = (RGB_TO_CB_709_SCALED(r, g, b) >> COMP_BASE) + (1<<15);
                *d++ = CLAMP_LIMITED_CBCR(u, 16);
                comp_type_t y = (RGB_TO_Y_709_SCALED(r, g, b) >> COMP_BASE) + (1<<12);
                *d++ = CLAMP_LIMITED_Y(y, 16);
                comp_type_t v = (RGB_TO_CR_709_SCALED(r, g, b) >> COMP_BASE) + (1<<15);
                *d++ = CLAMP_LIMITED_CBCR(v, 16);
                *d++ = 0xFFFFU;
        }
}

static void vc_copylineY416toRG48(unsigned char * __restrict dst, const unsigned char * __restrict src, int dst_len, int rshift,
                int gshift, int bshift) {
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        assert((uintptr_t) src % 2 == 0);
        assert((uintptr_t) dst % 2 == 0);
        const uint16_t *in = (const void *) src;
        uint16_t *d = (void *) dst;
        OPTIMIZED_FOR (int x = 0; x < dst_len; x += 6) {
                comp_type_t u = *in++ - (1<<15);
                comp_type_t y = Y_SCALE * (*in++ - (1<<12));
                comp_type_t v = *in++ - (1<<15);
                in++;
                comp_type_t r = (YCBCR_TO_R_709_SCALED(y, u, v) >> COMP_BASE);
                comp_type_t g = (YCBCR_TO_G_709_SCALED(y, u, v) >> COMP_BASE);
                comp_type_t b = (YCBCR_TO_B_709_SCALED(y, u, v) >> COMP_BASE);
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
                IDEPTH    = 8,
                Y_SHIFT   = 1 << (IDEPTH - 4),
                C_SHIFT   = 1 << (IDEPTH - 1),
                ODEPTH    = 8,
                PIX_COUNT = 6,
                RGB_BPP   = 3,
                OUT_BL_SZ = PIX_COUNT * RGB_BPP,
        };
        UNUSED(rshift), UNUSED(gshift), UNUSED(bshift);
#define WRITE_YUV_AS_RGB(y, u, v) \
        (y) = Y_SCALE * ((y) - Y_SHIFT); \
        val = (YCBCR_TO_R_709_SCALED((y), (u), (v)) >> (COMP_BASE)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH); \
        val = (YCBCR_TO_G_709_SCALED((y), (u), (v)) >> (COMP_BASE)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH); \
        val = (YCBCR_TO_B_709_SCALED((y), (u), (v)) >> (COMP_BASE)); \
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
#define WRITE_YUV_AS_RGB(y, u, v) \
        (y) = Y_SCALE * ((y) - Y_SHIFT); \
        val = (YCBCR_TO_R_709_SCALED((y), (u), (v)) >> (COMP_BASE - DIFF_BPP)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH); \
        val = (YCBCR_TO_G_709_SCALED((y), (u), (v)) >> (COMP_BASE - DIFF_BPP)); \
        *(dst++) = CLAMP_FULL(val, ODEPTH); \
        val = (YCBCR_TO_B_709_SCALED((y), (u), (v)) >> (COMP_BASE - DIFF_BPP)); \
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
#if 1
        { vc_copylineR12LtoRGB_SSE,   R12L,  RGB },
#endif
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

/* vim: set expandtab sw=8: */
