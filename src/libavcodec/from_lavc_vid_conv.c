/**
 * @file   libavcodec/from_lavc_vid_conv.c
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
 *
 * @todo
 * Some conversions to RGBA ignore RGB-shifts - either fix that or deprecate RGB-shifts
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
#include "host.h"
#include "hwaccel_vdpau.h"
#include "hwaccel_rpi4.h"
#include "libavcodec/from_lavc_vid_conv.h"
#include "video.h"

#ifdef __SSE3__
#include "pmmintrin.h"
#endif

#undef MAX
#undef MIN
#define MAX(a, b)      (((a) > (b))? (a): (b))
#define MIN(a, b)      (((a) < (b))? (a): (b))

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

static void nv12_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cbcr = (char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                char *dst = dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                }
        }
}

static void rgb24_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoUYVY((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0], vc_get_linesize(width, UYVY), 0, 0, 0);
        }
}

static void memcpy_data(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        UNUSED(width);
        for (int comp = 0; comp < AV_NUM_DATA_POINTERS; ++comp) {
                if (frame->data[comp] == NULL) {
                        break;
                }
                for (int y = 0; y < height; ++y) {
                        memcpy(dst_buffer + y * pitch, frame->data[comp] + y * frame->linesize[comp],
                                        frame->linesize[comp]);
                }
        }
}

static void rgb24_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoRGBA((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, RGBA), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void gbrp_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        uint8_t *buf = (uint8_t *) dst_buffer + y * pitch + x * 3;
                        int src_idx = y * frame->linesize[0] + x;
                        buf[0] = frame->data[2][src_idx]; // R
                        buf[1] = frame->data[0][src_idx]; // G
                        buf[2] = frame->data[1][src_idx]; // B
                }
        }
}

static void gbrp_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        assert((uintptr_t) dst_buffer % 4 == 0);

        for (int y = 0; y < height; ++y) {
                uint32_t *line = (uint32_t *)(void *) (dst_buffer + y * pitch);
                int src_idx = y * frame->linesize[0];

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *line++ = frame->data[2][src_idx] << rgb_shift[R] |
                                frame->data[0][src_idx] << rgb_shift[G] |
                                frame->data[1][src_idx] << rgb_shift[B];
                        src_idx += 1;
                }
        }
}

#if defined __GNUC__
static inline void gbrap_to_rgb_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, int comp_count)
        __attribute__((always_inline));
#endif
static inline void gbrap_to_rgb_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, int comp_count)
{
        assert(rgb_shift[R] == DEFAULT_R_SHIFT && rgb_shift[G] == DEFAULT_G_SHIFT && rgb_shift[B] == DEFAULT_B_SHIFT);
        for (int y = 0; y < height; ++y) {
                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        uint8_t *buf = (uint8_t *) dst_buffer + y * pitch + x * comp_count;
                        int src_idx = y * frame->linesize[0] + x;
                        buf[0] = frame->data[2][src_idx]; // R
                        buf[1] = frame->data[0][src_idx]; // G
                        buf[2] = frame->data[1][src_idx]; // B
                        if (comp_count == 4) {
                                buf[3] = frame->data[3][src_idx]; // B
                        }
                }
        }
}

static void gbrap_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrap_to_rgb_rgba(dst_buffer, frame, width, height, pitch, rgb_shift, 4);
}

static void gbrap_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrap_to_rgb_rgba(dst_buffer, frame, width, height, pitch, rgb_shift, 3);
}

#if defined __GNUC__
static inline void gbrpXXle_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void gbrpXXle_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_r >> (in_depth - 8U);
                        *dst++ = ((*src_r++ >> (in_depth - 10U)) & 0x3U) << 6U | *src_g >> (in_depth - 6U);
                        *dst++ = ((*src_g++ >> (in_depth - 10U)) & 0xFU) << 4U | *src_b >> (in_depth - 4U);
                        *dst++ = ((*src_b++ >> (in_depth - 10U)) & 0x3FU) << 2U;
                }
        }
}

static void gbrp10le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r10k(dst_buffer, frame, width, height, pitch, rgb_shift, 10U);
}

static void gbrp16le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r10k(dst_buffer, frame, width, height, pitch, rgb_shift, 16U);
}

#if defined __GNUC__
static inline void yuv444pXXle_to_r10k(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
        __attribute__((always_inline));
#endif
static inline void yuv444pXXle_to_r10k(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
		unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t y = (Y_SCALE * (*src_y++ - (1<<(depth-4))));
                        comp_type_t cr = *src_cr++ - (1<<(depth-1));
                        comp_type_t cb = *src_cb++ - (1<<(depth-1));

                        comp_type_t r = YCBCR_TO_R_709_SCALED(y, cb, cr) >> (COMP_BASE-10+depth);
                        comp_type_t g = YCBCR_TO_G_709_SCALED(y, cb, cr) >> (COMP_BASE-10+depth);
                        comp_type_t b = YCBCR_TO_B_709_SCALED(y, cb, cr) >> (COMP_BASE-10+depth);
                        // r g b is now on 10 bit scale

                        r = CLAMP_FULL(r, 10);
                        g = CLAMP_FULL(g, 10);
                        b = CLAMP_FULL(b, 10);

			*dst++ = r >> 2;
			*dst++ = (r & 0x3) << 6 | g >> 4;
			*dst++ = (g & 0xF) << 4 | b >> 6;
			*dst++ = (b & 0x3F) << 2;
                }
        }
}

static void yuv444p10le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r10k(10, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p12le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r10k(12, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p16le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r10k(16, dst_buffer, frame, width, height, pitch, rgb_shift);
}

#if defined __GNUC__
static inline void yuv444pXXle_to_r12l(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
        __attribute__((always_inline));
#endif
static inline void yuv444pXXle_to_r12l(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                for (int x = 0; x < width; x += 8) {
                        comp_type_t r[8];
                        comp_type_t g[8];
                        comp_type_t b[8];
                        OPTIMIZED_FOR (int j = 0; j < 8; ++j) {
                                comp_type_t y = (Y_SCALE * (*src_y++ - (1<<(depth-4))));
                                comp_type_t cr = *src_cr++ - (1<<(depth-1));
                                comp_type_t cb = *src_cb++ - (1<<(depth-1));
                                comp_type_t rr = YCBCR_TO_R_709_SCALED(y, cb, cr) >> (COMP_BASE-12+depth);
                                comp_type_t gg = YCBCR_TO_G_709_SCALED(y, cb, cr) >> (COMP_BASE-12+depth);
                                comp_type_t bb = YCBCR_TO_B_709_SCALED(y, cb, cr) >> (COMP_BASE-12+depth);
                                r[j] = CLAMP_FULL(rr, 12);
                                g[j] = CLAMP_FULL(gg, 12);
                                b[j] = CLAMP_FULL(bb, 12);
                        }

                        dst[BYTE_SWAP(0)] = r[0] & 0xff;
                        dst[BYTE_SWAP(1)] = (g[0] & 0xf) << 4 | r[0] >> 8;
                        dst[BYTE_SWAP(2)] = g[0] >> 4;
                        dst[BYTE_SWAP(3)] = b[0] & 0xff;
                        dst[4 + BYTE_SWAP(0)] = (r[1] & 0xf) << 4 | b[0] >> 8;
                        dst[4 + BYTE_SWAP(1)] = r[1] >> 4;
                        dst[4 + BYTE_SWAP(2)] = g[1] & 0xff;
                        dst[4 + BYTE_SWAP(3)] = (b[1] & 0xf) << 4 | g[1] >> 8;
                        dst[8 + BYTE_SWAP(0)] = b[1] >> 4;
                        dst[8 + BYTE_SWAP(1)] = r[2] & 0xff;
                        dst[8 + BYTE_SWAP(2)] = (g[2] & 0xf) << 4 | r[2] >> 8;
                        dst[8 + BYTE_SWAP(3)] = g[2] >> 4;
                        dst[12 + BYTE_SWAP(0)] = b[2] & 0xff;
                        dst[12 + BYTE_SWAP(1)] = (r[3] & 0xf) << 4 | b[2] >> 8;
                        dst[12 + BYTE_SWAP(2)] = r[3] >> 4;
                        dst[12 + BYTE_SWAP(3)] = g[3] & 0xff;
                        dst[16 + BYTE_SWAP(0)] = (b[3] & 0xf) << 4 | g[3] >> 8;
                        dst[16 + BYTE_SWAP(1)] = b[3] >> 4;
                        dst[16 + BYTE_SWAP(2)] = r[4] & 0xff;
                        dst[16 + BYTE_SWAP(3)] = (g[4] & 0xf) << 4 | r[4] >> 8;
                        dst[20 + BYTE_SWAP(0)] = g[4] >> 4;
                        dst[20 + BYTE_SWAP(1)] = b[4] & 0xff;
                        dst[20 + BYTE_SWAP(2)] = (r[5] & 0xf) << 4 | b[4] >> 8;
                        dst[20 + BYTE_SWAP(3)] = r[5] >> 4;;
                        dst[24 + BYTE_SWAP(0)] = g[5] & 0xff;
                        dst[24 + BYTE_SWAP(1)] = (b[5] & 0xf) << 4 | g[5] >> 8;
                        dst[24 + BYTE_SWAP(2)] = b[5] >> 4;
                        dst[24 + BYTE_SWAP(3)] = r[6] & 0xff;
                        dst[28 + BYTE_SWAP(0)] = (g[6] & 0xf) << 4 | r[6] >> 8;
                        dst[28 + BYTE_SWAP(1)] = g[6] >> 4;
                        dst[28 + BYTE_SWAP(2)] = b[6] & 0xff;
                        dst[28 + BYTE_SWAP(3)] = (r[7] & 0xf) << 4 | b[6] >> 8;
                        dst[32 + BYTE_SWAP(0)] = r[7] >> 4;
                        dst[32 + BYTE_SWAP(1)] = g[7] & 0xff;
                        dst[32 + BYTE_SWAP(2)] = (b[7] & 0xf) << 4 | g[7] >> 8;
                        dst[32 + BYTE_SWAP(3)] = b[7] >> 4;
                        dst += 36;
                }
        }
}

static void yuv444p10le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r12l(10, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p12le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r12l(12, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p16le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r12l(16, dst_buffer, frame, width, height, pitch, rgb_shift);
}

#if defined __GNUC__
static inline void yuv444pXXle_to_rg48(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
        __attribute__((always_inline));
#endif
static inline void yuv444pXXle_to_rg48(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        assert((uintptr_t) dst_buffer % 2 == 0);
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                uint16_t *dst = (uint16_t *)(void *) (dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t y = (Y_SCALE * (*src_y++ - (1<<(depth-4))));
                        comp_type_t cr = *src_cr++ - (1<<(depth-1));
                        comp_type_t cb = *src_cb++ - (1<<(depth-1));

                        comp_type_t r = YCBCR_TO_R_709_SCALED(y, cb, cr) >> (COMP_BASE-16+depth);
                        comp_type_t g = YCBCR_TO_G_709_SCALED(y, cb, cr) >> (COMP_BASE-16+depth);
                        comp_type_t b = YCBCR_TO_B_709_SCALED(y, cb, cr) >> (COMP_BASE-16+depth);
                        // r g b is now on 16 bit scale

                        *dst++ = CLAMP_FULL(r, 16);
                        *dst++ = CLAMP_FULL(g, 16);
                        *dst++ = CLAMP_FULL(b, 16);
                }
        }
}

static void yuv444p10le_to_rg48(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_rg48(10, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p12le_to_rg48(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_rg48(12, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p16le_to_rg48(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_rg48(16, dst_buffer, frame, width, height, pitch, rgb_shift);
}

#if defined __GNUC__
static inline void gbrpXXle_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void gbrpXXle_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

#undef S
#define S(x) ((x) >> (in_depth - 12))

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
                        dst[BYTE_SWAP(0)] = S(*src_r) & 0xff;
                        dst[BYTE_SWAP(1)] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[BYTE_SWAP(2)] = S(*src_g++) >> 4;
                        dst[BYTE_SWAP(3)] = S(*src_b) & 0xff;
                        dst[4 + BYTE_SWAP(0)] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[4 + BYTE_SWAP(1)] = S(*src_r++) >> 4;
                        dst[4 + BYTE_SWAP(2)] = S(*src_g) & 0xff;
                        dst[4 + BYTE_SWAP(3)] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[8 + BYTE_SWAP(0)] = S(*src_b++) >> 4;
                        dst[8 + BYTE_SWAP(1)] = S(*src_r) & 0xff;
                        dst[8 + BYTE_SWAP(2)] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[8 + BYTE_SWAP(3)] = S(*src_g++) >> 4;
                        dst[12 + BYTE_SWAP(0)] = S(*src_b) & 0xff;
                        dst[12 + BYTE_SWAP(1)] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[12 + BYTE_SWAP(2)] = S(*src_r++) >> 4;
                        dst[12 + BYTE_SWAP(3)] = S(*src_g) & 0xff;
                        dst[16 + BYTE_SWAP(0)] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[16 + BYTE_SWAP(1)] = S(*src_b++) >> 4;
                        dst[16 + BYTE_SWAP(2)] = S(*src_r) & 0xff;
                        dst[16 + BYTE_SWAP(3)] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[20 + BYTE_SWAP(0)] = S(*src_g++) >> 4;
                        dst[20 + BYTE_SWAP(1)] = S(*src_b) & 0xff;
                        dst[20 + BYTE_SWAP(2)] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[20 + BYTE_SWAP(3)] = S(*src_r++) >> 4;;
                        dst[24 + BYTE_SWAP(0)] = S(*src_g) & 0xff;
                        dst[24 + BYTE_SWAP(1)] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[24 + BYTE_SWAP(2)] = S(*src_b++) >> 4;
                        dst[24 + BYTE_SWAP(3)] = S(*src_r) & 0xff;
                        dst[28 + BYTE_SWAP(0)] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[28 + BYTE_SWAP(1)] = S(*src_g++) >> 4;
                        dst[28 + BYTE_SWAP(2)] = S(*src_b) & 0xff;
                        dst[28 + BYTE_SWAP(3)] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[32 + BYTE_SWAP(0)] = S(*src_r++) >> 4;
                        dst[32 + BYTE_SWAP(1)] = S(*src_g) & 0xff;
                        dst[32 + BYTE_SWAP(2)] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[32 + BYTE_SWAP(3)] = S(*src_b++) >> 4;
                        dst += 36;
                }
        }
}

#if defined __GNUC__
static inline void gbrpXXle_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void gbrpXXle_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_r++ >> (in_depth - 8U);
                        *dst++ = *src_g++ >> (in_depth - 8U);
                        *dst++ = *src_b++ >> (in_depth - 8U);
                }
        }
}

#if defined __GNUC__
static inline void gbrpXXle_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void gbrpXXle_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
{
        assert((uintptr_t) dst_buffer % 4 == 0);
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *) (dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = (*src_r++ >> (in_depth - 8U)) << rgb_shift[0] | (*src_g++ >> (in_depth - 8U)) << rgb_shift[1] |
                                (*src_b++ >> (in_depth - 8U)) << rgb_shift[2];
                }
        }
}

static void gbrp10le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgb(dst_buffer, frame, width, height, pitch, rgb_shift, 10);
}

static void gbrp10le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgba(dst_buffer, frame, width, height, pitch, rgb_shift, 10);
}

#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
static void gbrp12le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r12l(dst_buffer, frame, width, height, pitch, rgb_shift, 12U);
}

static void gbrp12le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r10k(dst_buffer, frame, width, height, pitch, rgb_shift, 12U);
}

static void gbrp12le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgb(dst_buffer, frame, width, height, pitch, rgb_shift, 12U);
}

static void gbrp12le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgba(dst_buffer, frame, width, height, pitch, rgb_shift, 12U);
}
#endif

static void gbrp16le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r12l(dst_buffer, frame, width, height, pitch, rgb_shift, 16U);
}

static void gbrp16le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgb(dst_buffer, frame, width, height, pitch, rgb_shift, 16U);
}

static void gbrp16le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgba(dst_buffer, frame, width, height, pitch, rgb_shift, 16U);
}

static void rgb48le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toRGBA((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, RGBA), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void rgb48le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toR12L((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, R12L), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void yuv420p_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (height + 1) / 2; ++y) {
                int scnd_row = y * 2 + 1;
                if (scnd_row == height) {
                        scnd_row = height - 1;
                }
                char *src_y1 = (char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                char *src_y2 = (char *) in_frame->data[0] + in_frame->linesize[0] * scnd_row;
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst1 = dst_buffer + (y * 2) * pitch;
                char *dst2 = dst_buffer + scnd_row * pitch;

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
                        y1 = _mm_lddqu_si128((__m128i const*)(const void *) src_y1);
                        y2 = _mm_lddqu_si128((__m128i const*)(const void *) src_y2);
                        src_y1 += 16;
                        src_y2 += 16;

                        out1l = _mm_unpacklo_epi8(zero, y1);
                        out1h = _mm_unpackhi_epi8(zero, y1);
                        out2l = _mm_unpacklo_epi8(zero, y2);
                        out2h = _mm_unpackhi_epi8(zero, y2);

                        u1 = _mm_lddqu_si128((__m128i const*)(const void *) src_cb);
                        v1 = _mm_lddqu_si128((__m128i const*)(const void *) src_cr);
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

                        _mm_storeu_si128((__m128i *)(void *) dst1, out1l);
                        dst1 += 16;
                        _mm_storeu_si128((__m128i *)(void *) dst1, out1h);
                        dst1 += 16;
                        _mm_storeu_si128((__m128i *)(void *) dst2, out2l);
                        dst2 += 16;
                        _mm_storeu_si128((__m128i *)(void *) dst2, out2h);
                        dst2 += 16;
                }
#endif


                OPTIMIZED_FOR (; x < width - 1; x += 2) {
                        *dst1++ = *src_cb;
                        *dst1++ = *src_y1++;
                        *dst1++ = *src_cr;
                        *dst1++ = *src_y1++;

                        *dst2++ = *src_cb++;
                        *dst2++ = *src_y2++;
                        *dst2++ = *src_cr++;
                        *dst2++ = *src_y2++;
                }
                if (x < width) {
                        *dst1++ = *src_cb;
                        *dst1++ = *src_y1++;
                        *dst1++ = *src_cr;
                        *dst1++ = 0;

                        *dst2++ = *src_cb++;
                        *dst2++ = *src_y2++;
                        *dst2++ = *src_cr++;
                        *dst2++ = 0;
                }
        }
}

static void yuv420p_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint8_t *src_y1 = (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint8_t *src_y2 = (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

static void yuv422p_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cr++;
                        *dst++ = *src_y++;
                }
        }
}

static void yuv422p_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

static void yuv444p_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = (*src_cb + *(src_cb + 1)) / 2;
                        src_cb += 2;
                        *dst++ = *src_y++;
                        *dst++ = (*src_cr + *(src_cr + 1)) / 2;
                        src_cr += 2;
                        *dst++ = *src_y++;
                }
        }
}

static void yuv444p16le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y + 1;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y + 1;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y + 1;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = (*src_cb + *(src_cb + 2)) / 2;
                        src_cb += 4;
                        *dst++ = *src_y;
                        src_y += 2;
                        *dst++ = (*src_cr + *(src_cr + 2)) / 2;
                        src_cr += 4;
                        *dst++ = *src_y;
                        src_y += 2;
                }
        }
}

static void yuv444p_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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
 * Changes pixel format from planar YUV 422 to packed RGB/A.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
#if defined __GNUC__
static inline void nv12_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
        __attribute__((always_inline));
#endif
static inline void nv12_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
{
        assert((uintptr_t) dst_buffer % 4 == 0);

        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cbcr = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cb = *src_cbcr++ - 128;
                        comp_type_t cr = *src_cbcr++ - 128;
                        comp_type_t y = *src_y++ * Y_SCALE;
                        comp_type_t r = YCBCR_TO_R_709_SCALED(y, cb, cr);
                        comp_type_t g = YCBCR_TO_G_709_SCALED(y, cb, cr);
                        comp_type_t b = YCBCR_TO_B_709_SCALED(y, cb, cr);
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = FORMAT_RGBA(r >> COMP_BASE, g >> COMP_BASE, b >> COMP_BASE, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL(r >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL(g >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL(b >> COMP_BASE, 8);
                        }

                        y = *src_y++ * Y_SCALE;
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = FORMAT_RGBA(r >> COMP_BASE, g >> COMP_BASE, b >> COMP_BASE, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL(r >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL(g >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL(b >> COMP_BASE, 8);
                        }
                }
        }
}

static void nv12_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        nv12_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void nv12_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        nv12_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

#if defined __GNUC__
static inline void yuv8p_to_rgb(int subsampling, char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba) __attribute__((always_inline));
#endif
/**
 * Changes pixel format from planar 8-bit YUV to packed RGB/A.
 * Color space is assumed ITU-T Rec. 709 limited range.
 */
static inline void yuv8p_to_rgb(int subsampling, char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
{
        for(int y = 0; y < height / 2; ++y) {
                unsigned char *src_y1 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                unsigned char *src_y2 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1);
                unsigned char *dst1 = (unsigned char *) dst_buffer + pitch * (y * 2);
                unsigned char *dst2 = (unsigned char *) dst_buffer + pitch * (y * 2 + 1);

                unsigned char *src_cb1 = NULL;
                unsigned char *src_cr1 = NULL;
                unsigned char *src_cb2 = NULL;
                unsigned char *src_cr2 = NULL;
                if (subsampling == 420) {
                        src_cb1 = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                        src_cr1 = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                } else {
                        src_cb1 = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y * 2);
                        src_cr1 = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * (y * 2);
                        src_cb2 = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y * 2 + 1);
                        src_cr2 = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * (y * 2 + 1);
                }

#define WRITE_RES_YUV8P_TO_RGB(DST) if (rgba) {\
                                *((uint32_t *)(void *) DST) = FORMAT_RGBA((r) >> COMP_BASE, (g) >> COMP_BASE, (b) >> COMP_BASE, 8);\
                                DST += 4;\
                        } else {\
                                *DST++ = CLAMP_FULL((r) >> COMP_BASE, 8);\
                                *DST++ = CLAMP_FULL((g) >> COMP_BASE, 8);\
                                *DST++ = CLAMP_FULL((b) >> COMP_BASE, 8);\
                        }\

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cb = *src_cb1++ - 128;
                        comp_type_t cr = *src_cr1++ - 128;
                        comp_type_t y = *src_y1++ * Y_SCALE;
                        comp_type_t r = YCBCR_TO_R_709_SCALED(y, cb, cr);
                        comp_type_t g = YCBCR_TO_G_709_SCALED(y, cb, cr);
                        comp_type_t b = YCBCR_TO_B_709_SCALED(y, cb, cr);
                        WRITE_RES_YUV8P_TO_RGB(dst1)

                        y = *src_y1++ * Y_SCALE;
                        WRITE_RES_YUV8P_TO_RGB(dst1)

                        if (subsampling == 422) {
                                cb = *src_cb2++ - 128;
                                cr = *src_cr2++ - 128;
                                r = YCBCR_TO_R_709_SCALED(y, cb, cr);
                                g = YCBCR_TO_G_709_SCALED(y, cb, cr);
                                b = YCBCR_TO_B_709_SCALED(y, cb, cr);
                        }
                        y = *src_y2++ * Y_SCALE;
                        WRITE_RES_YUV8P_TO_RGB(dst2)

                        y = *src_y2++ * Y_SCALE;
                        WRITE_RES_YUV8P_TO_RGB(dst2)
                }
        }
}

static void yuv420p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv8p_to_rgb(420, dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv420p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv8p_to_rgb(420, dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

static void yuv422p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv8p_to_rgb(422, dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv422p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv8p_to_rgb(422, dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}


/**
 * Changes pixel format from planar YUV 444 to packed RGB/A.
 */
#if defined __GNUC__
static inline void yuv444p_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
        __attribute__((always_inline));
#endif
static inline void yuv444p_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
{
        assert((uintptr_t) dst_buffer % 4 == 0);

        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ * Y_SCALE;
                        comp_type_t r = YCBCR_TO_R_709_SCALED(y, cb, cr);
                        comp_type_t g = YCBCR_TO_G_709_SCALED(y, cb, cr);
                        comp_type_t b = YCBCR_TO_B_709_SCALED(y, cb, cr);
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = FORMAT_RGBA((r) >> COMP_BASE, (g) >> COMP_BASE, (b) >> COMP_BASE, 8);
                                dst += 4;
                        } else {
                                *dst++ = MIN(MAX((r) >> COMP_BASE, 1), 254);
                                *dst++ = MIN(MAX((g) >> COMP_BASE, 1), 254);
                                *dst++ = MIN(MAX((b) >> COMP_BASE, 1), 254);
                        }
                }
        }
}

static void yuv444p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv444p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

static void yuv420p10le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

static void yuv422p10le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

static void yuv444p10le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
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

static void yuv444p16le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = ((src_cb[0] >> 6U) + (src_cb[1] >> 6U)) / 2;
                        w0_0 = w0_0 | (*src_y++ >> 6U) << 10U;
                        w0_0 = w0_0 | ((src_cr[0] >> 6U) + (src_cr[1] >> 6U)) / 2 << 20U;
                        src_cb += 2;
                        src_cr += 2;

                        w0_1 = *src_y++;
                        w0_1 = w0_1 | ((src_cb[0] >> 6U) + (src_cb[1] >> 6U)) / 2 << 10U;
                        w0_1 = w0_1 | (*src_y++ >> 6U) << 20U;
                        src_cb += 2;

                        w0_2 = ((src_cr[0] >> 6U) + (src_cr[1] >> 6U)) / 2;
                        w0_2 = w0_2 | (*src_y++ >> 6U) << 10U;
                        w0_2 = w0_2 | ((src_cb[0] >> 6U) + (src_cb[1] >> 6U)) / 2 << 20U;
                        src_cr += 2;
                        src_cb += 2;

                        w0_3 = *src_y++;
                        w0_3 = w0_3 | ((src_cr[0] >> 6U) + (src_cr[1] >> 6U)) / 2 << 10U;
                        w0_3 = w0_3 | ((*src_y++ >> 6U)) << 20U;
                        src_cr += 2;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}

static void yuv420p10le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst1 = (uint8_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint8_t *dst2 = (uint8_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
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

static void yuv422p10le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
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

static void yuv444p10le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = (src_cb[0] + src_cb[1]) / 2 >> 2;
                        *dst++ = *src_y++ >> 2;
                        *dst++ = (src_cr[0] + src_cr[1]) / 2 >> 2;
                        *dst++ = *src_y++ >> 2;
                        src_cb += 2;
                        src_cr += 2;
                }
        }
}

#if defined __GNUC__
static inline void yuvp10le_to_rgb(int subsampling, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, int out_bit_depth) __attribute__((always_inline));
#endif
static inline void yuvp10le_to_rgb(int subsampling, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, int out_bit_depth)
{
        assert((uintptr_t) dst_buffer % 4 == 0);
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        assert(subsampling == 422 || subsampling == 420);
        assert(out_bit_depth == 24 || out_bit_depth == 30 || out_bit_depth == 32);
        const int bpp = out_bit_depth == 30 ? 10 : 8;

        for (int y = 0; y < height / 2; ++y) {
                uint16_t * __restrict src_y1 = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * 2 * y);
                uint16_t * __restrict src_y2 = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * (2 * y + 1));
                uint16_t * __restrict src_cb1;
                uint16_t * __restrict src_cr1;
                uint16_t * __restrict src_cb2;
                uint16_t * __restrict src_cr2;
                if (subsampling == 420) {
                        src_cb1 = src_cb2 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                        src_cr1 = src_cr2 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                } else {
                        src_cb1 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * (2 * y));
                        src_cb2 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * (2 * y + 1));
                        src_cr1 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * (2 * y));
                        src_cr2 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * (2 * y + 1));
                }
                unsigned char *dst1 = (unsigned char *) dst_buffer + (2 * y) * pitch;
                unsigned char *dst2 = (unsigned char *) dst_buffer + (2 * y + 1) * pitch;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cr = *src_cr1++ - (1<<9);
                        comp_type_t cb = *src_cb1++ - (1<<9);
                        comp_type_t rr = YCBCR_TO_R_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        comp_type_t gg = YCBCR_TO_G_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        comp_type_t bb = YCBCR_TO_B_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));

#                       define WRITE_RES_YUV10P_TO_RGB(Y, DST) {\
                                comp_type_t r = Y + rr;\
                                comp_type_t g = Y + gg;\
                                comp_type_t b = Y + bb;\
                                r = CLAMP_FULL(r, bpp);\
                                g = CLAMP_FULL(g, bpp);\
                                b = CLAMP_FULL(b, bpp);\
                                if (out_bit_depth == 32) {\
                                        *((uint32_t *)(void *) DST) = (r << rgb_shift[R] | g << rgb_shift[G] | b << rgb_shift[B]);\
                                        DST += 4;\
                                } else if (out_bit_depth == 24) {\
                                        *DST++ = r;\
                                        *DST++ = g;\
                                        *DST++ = b;\
                                } else {\
                                        *((uint32_t *)(void *) (DST)) = r >> 2U | (r & 0x3U) << 14 | g >> 4U << 8U | (g & 0xFU) << 20U | b >> 6U << 16U | (b & 0x3FU) << 26U;\
                                        /*      == htonl(r << 22U | g << 12U | b << 2U) */ \
                                        DST += 4;\
                                }\
                        }

                        comp_type_t y1 = (Y_SCALE * (*src_y1++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y1, dst1)

                        comp_type_t y11 = (Y_SCALE * (*src_y1++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y11, dst1)

                        if (subsampling == 422) {
                                cr = *src_cr2++ - (1<<9);
                                cb = *src_cb2++ - (1<<9);
                                rr = YCBCR_TO_R_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                                gg = YCBCR_TO_G_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                                bb = YCBCR_TO_B_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        }

                        comp_type_t y2 = (Y_SCALE * (*src_y2++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y2, dst2)

                        comp_type_t y22 = (Y_SCALE * (*src_y2++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y22, dst2)
                }
        }
}

#define MAKE_YUV_TO_RGB_FUNCTION_NAME(subs, out_bit_depth) yuv ## subs ## p10le_to_rgb ## out_bit_depth

#define MAKE_YUV_TO_RGB_FUNCTION(subs, out_bit_depth) static void MAKE_YUV_TO_RGB_FUNCTION_NAME(subs, out_bit_depth)(char * __restrict dst_buffer, AVFrame * __restrict in_frame,\
                int width, int height, int pitch, const int * __restrict rgb_shift) {\
        yuvp10le_to_rgb(subs, dst_buffer, in_frame, width, height, pitch, rgb_shift, out_bit_depth);\
}

MAKE_YUV_TO_RGB_FUNCTION(420, 24)
MAKE_YUV_TO_RGB_FUNCTION(420, 30)
MAKE_YUV_TO_RGB_FUNCTION(420, 32)
MAKE_YUV_TO_RGB_FUNCTION(422, 24)
MAKE_YUV_TO_RGB_FUNCTION(422, 30)
MAKE_YUV_TO_RGB_FUNCTION(422, 32)

#if defined __GNUC__
static inline void yuv444p10le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
        __attribute__((always_inline));
#endif
static inline void yuv444p10le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
{
        for (int y = 0; y < height; y++) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t cb = (*src_cb++ >> 2) - 128;
                        comp_type_t cr = (*src_cr++ >> 2) - 128;
                        comp_type_t y = (*src_y++ >> 2) * Y_SCALE;
                        comp_type_t r = YCBCR_TO_R_709_SCALED(y, cb, cr);
                        comp_type_t g = YCBCR_TO_G_709_SCALED(y, cb, cr);
                        comp_type_t b = YCBCR_TO_B_709_SCALED(y, cb, cr);
                        if (rgba) {
                                *(uint32_t *)(void *) dst = FORMAT_RGBA((r) >> COMP_BASE, (g) >> COMP_BASE, (b) >> COMP_BASE, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL((r) >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL((g) >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL((b) >> COMP_BASE, 8);
                        }
                }
        }
}

static void yuv444p10le_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444p10le_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv444p10le_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444p10le_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

#ifdef HAVE_P210
static void p210le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) in_frame->data[1] % 2 == 0);
        assert((uintptr_t) dst_buffer % 4 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cbcr = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0 = *src_cbcr >> 6; // Cb0
                        src_cbcr++; // Cr0
                        w0_0 = w0_0 | (*src_y++ >> 6) << 10;
                        w0_0 = w0_0 | (*src_cbcr >> 6) << 20;
                        src_cbcr++; // Cb1

                        uint32_t w0_1 = *src_y++ >> 6;
                        w0_1 = w0_1 | (*src_cbcr >> 6) << 10;
                        src_cbcr++; // Cr1
                        w0_1 = w0_1 | (*src_y++ >> 6) << 20;

                        uint32_t w0_2 = *src_cbcr >> 6;
                        src_cbcr++;
                        w0_2 = w0_2 | (*src_y++ >> 6) << 10;
                        w0_2 = w0_2 | (*src_cbcr >> 6) << 20;
                        src_cbcr++;

                        uint32_t w0_3 = *src_y++ >> 6;
                        w0_3 = w0_3 | (*src_cbcr >> 6) << 10;
                        src_cbcr++;
                        w0_3 = w0_3 | (*src_y++ >> 6) << 20;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}
#endif

static void p010le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) in_frame->data[1] % 2 == 0);
        assert((uintptr_t) dst_buffer % 4 == 0 && pitch % 4 == 0);
        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *) (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *) (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cbcr = (uint16_t *)(void *) (in_frame->data[1] + in_frame->linesize[1] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src_cbcr >> 6; // Cb0
                        w1_0 = *src_cbcr >> 6;
                        src_cbcr++; // Cr0
                        w0_0 = w0_0 | (*src_y1++ >> 6) << 10;
                        w1_0 = w1_0 | (*src_y2++ >> 6) << 10;
                        w0_0 = w0_0 | (*src_cbcr >> 6) << 20;
                        w1_0 = w1_0 | (*src_cbcr >> 6) << 20;
                        src_cbcr++; // Cb1

                        w0_1 = *src_y1++ >> 6;
                        w1_1 = *src_y2++ >> 6;
                        w0_1 = w0_1 | (*src_cbcr >> 6) << 10;
                        w1_1 = w1_1 | (*src_cbcr >> 6) << 10;
                        src_cbcr++; // Cr1
                        w0_1 = w0_1 | (*src_y1++ >> 6) << 20;
                        w1_1 = w1_1 | (*src_y2++ >> 6) << 20;

                        w0_2 = *src_cbcr >> 6;
                        w1_2 = *src_cbcr >> 6;
                        src_cbcr++;
                        w0_2 = w0_2 | (*src_y1++ >> 6) << 10;
                        w1_2 = w1_2 | (*src_y2++ >> 6) << 10;
                        w0_2 = w0_2 | (*src_cbcr >> 6) << 20;
                        w1_2 = w1_2 | (*src_cbcr >> 6) << 20;
                        src_cbcr++;

                        w0_3 = *src_y1++ >> 6;
                        w1_3 = *src_y2++ >> 6;
                        w0_3 = w0_3 | (*src_cbcr >> 6) << 10;
                        w1_3 = w1_3 | (*src_cbcr >> 6) << 10;
                        src_cbcr++;
                        w0_3 = w0_3 | (*src_y1++ >> 6) << 20;
                        w1_3 = w1_3 | (*src_y2++ >> 6) << 20;

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

static void p010le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cbcr = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *dst1 = (uint8_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint8_t *dst2 = (uint8_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        uint8_t tmp;
                        // U
                        tmp = *src_cbcr++ >> 8;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 8;
                        *dst2++ = *src_y2++ >> 8;
                        // V
                        tmp = *src_cbcr++ >> 8;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 8;
                        *dst2++ = *src_y2++ >> 8;
                }
        }
}

#ifdef HAVE_P210
static void p210le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) in_frame->data[1] % 2 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cbcr = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst = *src_cbcr++ >> 8;
                        *dst++ = *src_y++ >> 8;
                        *dst++ = *src_cbcr++ >> 8;
                        *dst++ = *src_y++ >> 8;
                }
        }
}
#endif

#ifdef HWACC_VDPAU
static void av_vdpau_to_ug_vdpau(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(width);
        UNUSED(height);
        UNUSED(pitch);
        UNUSED(rgb_shift);

        struct video_frame_callbacks *callbacks = in_frame->opaque;

        hw_vdpau_frame *out = (hw_vdpau_frame *)(void *) dst_buffer;

        hw_vdpau_frame_init(out);

        hw_vdpau_frame_from_avframe(out, in_frame);

        callbacks->recycle = hw_vdpau_recycle_callback; 
        callbacks->copy = hw_vdpau_copy_callback; 
}
#endif

#ifdef HWACC_RPI4
static void av_rpi4_8_to_ug(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(width);
        UNUSED(height);
        UNUSED(pitch);
        UNUSED(rgb_shift);

        av_frame_wrapper *out = (av_frame_wrapper *)(void *) dst_buffer;
        av_frame_ref(out->av_frame, in_frame);
}
#endif

static void ayuv64_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint8_t *src = (uint8_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < ((width + 1) & ~1); ++x) {
                        *dst++ = (src[1] + src[9] / 2);  // U
                        *dst++ = src[3];                 // Y
                        *dst++ = (src[5] + src[13] / 2); // V
                        *dst++ = src[11];                // Y
                }
        }
}

static void ayuv64_to_y416(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) dst_buffer % 2 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *dst = (uint16_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = src[2]; // U
                        *dst++ = src[1]; // Y
                        *dst++ = src[3]; // V
                        *dst++ = src[0]; // A
                        src += 4;
                }
        }
}

/**
 * @brief returns list of available conversion. Terminated by uv_to_av_conversion::uv_codec == VIDEO_CODEC_NONE
 */
const struct av_to_uv_conversion *get_av_to_uv_conversions() {
        static const struct av_to_uv_conversion av_to_uv_conversions[] = {
                // 10-bit YUV
                {AV_PIX_FMT_YUV420P10LE, v210, yuv420p10le_to_v210, true},
                {AV_PIX_FMT_YUV420P10LE, UYVY, yuv420p10le_to_uyvy, false},
                {AV_PIX_FMT_YUV420P10LE, RGB, yuv420p10le_to_rgb24, false},
                {AV_PIX_FMT_YUV420P10LE, RGBA, yuv420p10le_to_rgb32, false},
                {AV_PIX_FMT_YUV420P10LE, R10k, yuv420p10le_to_rgb30, false},
                {AV_PIX_FMT_YUV422P10LE, v210, yuv422p10le_to_v210, true},
                {AV_PIX_FMT_YUV422P10LE, UYVY, yuv422p10le_to_uyvy, false},
                {AV_PIX_FMT_YUV422P10LE, RGB, yuv422p10le_to_rgb24, false},
                {AV_PIX_FMT_YUV422P10LE, RGBA, yuv422p10le_to_rgb32, false},
                {AV_PIX_FMT_YUV422P10LE, R10k, yuv422p10le_to_rgb30, false},
                {AV_PIX_FMT_YUV444P10LE, v210, yuv444p10le_to_v210, true},
                {AV_PIX_FMT_YUV444P10LE, UYVY, yuv444p10le_to_uyvy, false},
                {AV_PIX_FMT_YUV444P10LE, R10k, yuv444p10le_to_r10k, false},
                {AV_PIX_FMT_YUV444P10LE, RGB, yuv444p10le_to_rgb24, false},
                {AV_PIX_FMT_YUV444P10LE, RGBA, yuv444p10le_to_rgb32, false},
                {AV_PIX_FMT_YUV444P10LE, R12L, yuv444p10le_to_r12l, false},
                {AV_PIX_FMT_YUV444P10LE, RG48, yuv444p10le_to_rg48, false},
#ifdef HAVE_P210
                {AV_PIX_FMT_P210LE, v210, p210le_to_v210, true},
                {AV_PIX_FMT_P210LE, UYVY, p210le_to_uyvy, false},
#endif
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(55, 15, 100) // FFMPEG commit c2869b4640f
                {AV_PIX_FMT_P010LE, v210, p010le_to_v210, true},
                {AV_PIX_FMT_P010LE, UYVY, p010le_to_uyvy, true},
#endif
                // 8-bit YUV
                {AV_PIX_FMT_YUV420P, v210, yuv420p_to_v210, false},
                {AV_PIX_FMT_YUV420P, UYVY, yuv420p_to_uyvy, true},
                {AV_PIX_FMT_YUV420P, RGB, yuv420p_to_rgb24, false},
                {AV_PIX_FMT_YUV420P, RGBA, yuv420p_to_rgb32, false},
                {AV_PIX_FMT_YUV422P, v210, yuv422p_to_v210, false},
                {AV_PIX_FMT_YUV422P, UYVY, yuv422p_to_uyvy, true},
                {AV_PIX_FMT_YUV422P, RGB, yuv422p_to_rgb24, false},
                {AV_PIX_FMT_YUV422P, RGBA, yuv422p_to_rgb32, false},
                {AV_PIX_FMT_YUV444P, v210, yuv444p_to_v210, false},
                {AV_PIX_FMT_YUV444P, UYVY, yuv444p_to_uyvy, true},
                {AV_PIX_FMT_YUV444P, RGB, yuv444p_to_rgb24, false},
                {AV_PIX_FMT_YUV444P, RGBA, yuv444p_to_rgb32, false},
                // 8-bit YUV - this should be supposedly full range JPEG but lavd decoder doesn't honor
                // GPUJPEG's SPIFF header indicating YUV BT.709 limited range. The YUVJ pixel formats
                // are detected only for GPUJPEG generated JPEGs.
                {AV_PIX_FMT_YUVJ420P, v210, yuv420p_to_v210, false},
                {AV_PIX_FMT_YUVJ420P, UYVY, yuv420p_to_uyvy, true},
                {AV_PIX_FMT_YUVJ420P, RGB, yuv420p_to_rgb24, false},
                {AV_PIX_FMT_YUVJ420P, RGBA, yuv420p_to_rgb32, false},
                {AV_PIX_FMT_YUVJ422P, v210, yuv422p_to_v210, false},
                {AV_PIX_FMT_YUVJ422P, UYVY, yuv422p_to_uyvy, true},
                {AV_PIX_FMT_YUVJ422P, RGB, yuv422p_to_rgb24, false},
                {AV_PIX_FMT_YUVJ422P, RGBA, yuv422p_to_rgb32, false},
                {AV_PIX_FMT_YUVJ444P, v210, yuv444p_to_v210, false},
                {AV_PIX_FMT_YUVJ444P, UYVY, yuv444p_to_uyvy, true},
                {AV_PIX_FMT_YUVJ444P, RGB, yuv444p_to_rgb24, false},
                {AV_PIX_FMT_YUVJ444P, RGBA, yuv444p_to_rgb32, false},
                // 8-bit YUV (NV12)
                {AV_PIX_FMT_NV12, UYVY, nv12_to_uyvy, true},
                {AV_PIX_FMT_NV12, RGB, nv12_to_rgb24, false},
                {AV_PIX_FMT_NV12, RGBA, nv12_to_rgb32, false},
                // 12-bit YUV
                {AV_PIX_FMT_YUV444P12LE, R10k, yuv444p12le_to_r10k, false},
                {AV_PIX_FMT_YUV444P12LE, R12L, yuv444p12le_to_r12l, false},
                {AV_PIX_FMT_YUV444P12LE, RG48, yuv444p12le_to_rg48, false},
                // 16-bit YUV
                {AV_PIX_FMT_YUV444P16LE, R10k, yuv444p16le_to_r10k, false},
                {AV_PIX_FMT_YUV444P16LE, R12L, yuv444p16le_to_r12l, false},
                {AV_PIX_FMT_YUV444P16LE, RG48, yuv444p16le_to_rg48, false},
                {AV_PIX_FMT_YUV444P16LE, UYVY, yuv444p16le_to_uyvy, false},
                {AV_PIX_FMT_YUV444P16LE, v210, yuv444p16le_to_v210, false},
                {AV_PIX_FMT_AYUV64, UYVY, ayuv64_to_uyvy, false },
                {AV_PIX_FMT_AYUV64, Y416, ayuv64_to_y416, true },
                // RGB
                {AV_PIX_FMT_GBRAP, RGB, gbrap_to_rgb, false},
                {AV_PIX_FMT_GBRAP, RGBA, gbrap_to_rgba, true},
                {AV_PIX_FMT_GBRP, RGB, gbrp_to_rgb, true},
                {AV_PIX_FMT_GBRP, RGBA, gbrp_to_rgba, true},
                {AV_PIX_FMT_RGB24, UYVY, rgb24_to_uyvy, false},
                {AV_PIX_FMT_RGB24, RGB, memcpy_data, true},
                {AV_PIX_FMT_RGB24, RGBA, rgb24_to_rgb32, false},
                {AV_PIX_FMT_GBRP10LE, R10k, gbrp10le_to_r10k, true},
                {AV_PIX_FMT_GBRP10LE, RGB, gbrp10le_to_rgb, false},
                {AV_PIX_FMT_GBRP10LE, RGBA, gbrp10le_to_rgba, false},
#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
                {AV_PIX_FMT_GBRP12LE, R12L, gbrp12le_to_r12l, true},
                {AV_PIX_FMT_GBRP12LE, R10k, gbrp12le_to_r10k, true},
                {AV_PIX_FMT_GBRP12LE, RGB, gbrp12le_to_rgb, false},
                {AV_PIX_FMT_GBRP12LE, RGBA, gbrp12le_to_rgba, false},
#endif
                {AV_PIX_FMT_GBRP16LE, R12L, gbrp16le_to_r12l, true},
                {AV_PIX_FMT_GBRP16LE, R10k, gbrp16le_to_r10k, true},
                {AV_PIX_FMT_GBRP12LE, RGB, gbrp16le_to_rgb, false},
                {AV_PIX_FMT_GBRP12LE, RGBA, gbrp16le_to_rgba, false},
                {AV_PIX_FMT_RGB48LE, RG48, memcpy_data, true},
                {AV_PIX_FMT_RGB48LE, R12L, rgb48le_to_r12l, false},
                {AV_PIX_FMT_RGB48LE, RGBA, rgb48le_to_rgba, false},
#ifdef HWACC_VDPAU
                // HW acceleration
                {AV_PIX_FMT_VDPAU, HW_VDPAU, av_vdpau_to_ug_vdpau, false},
#endif
#ifdef HWACC_RPI4
                {AV_PIX_FMT_RPI4_8, RPI4_8, av_rpi4_8_to_ug, false},
#endif
                {0, 0, 0, 0}
        };
        return av_to_uv_conversions;
}

av_to_uv_convert_p get_av_to_uv_conversion(int av_codec, codec_t uv_codec) {
        for (const struct av_to_uv_conversion *conversions = get_av_to_uv_conversions();
                        conversions->convert != 0; conversions++) {
                if (conversions->av_codec == av_codec &&
                                conversions->uv_codec == uv_codec) {
                        return conversions->convert;
                }
        }

        return NULL;
}

#pragma GCC diagnostic pop

/* vi: set expandtab sw=8: */
