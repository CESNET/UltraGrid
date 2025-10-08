/**
 * @file   libavcodec/from_lavc_vid_conv.c
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
 *
 * @todo
 * Some conversions to RGBA ignore RGB-shifts - either fix that or deprecate RGB-shifts
 */

#include <assert.h>
#include <libavutil/pixdesc.h>
#include <libavutil/pixfmt.h>
#include <libavutil/hwcontext_drm.h>
#if __STDC_VERSION__ < 202311L
#include <stdalign.h>     // for alignof
#endif
#include <stdbool.h>
#include <stddef.h>                              // for NULL, ptrdiff_t, size_t
#include <stdint.h>

#include "color_space.h"
#include "compat/qsort_s.h"
#ifdef HAVE_CONFIG_H
#include "config.h"       // for HWACC_VDPAU
#endif
#include "debug.h"
#include "host.h"
#include "hwaccel_vdpau.h"
#include "hwaccel_drm.h"
#include "libavcodec/from_lavc_vid_conv.h"
#include "libavcodec/from_lavc_vid_conv_cuda.h"
#include "libavcodec/lavc_common.h"
#include "pixfmt_conv.h"
#include "types.h"
#include "utils/debug.h"  // for DEBUG_TIMER_*
#include "utils/macros.h" // OPTIMIZED_FOR
#include "utils/misc.h"   // get_cpu_core_count
#include "utils/worker.h" // task_run_parallel
#include "video.h"
#include "video_codec.h"
#include "video_decompress.h" // for VDEC_PRIO_*

#ifdef __SSE3__
#include "pmmintrin.h"
#endif

#define MOD_NAME "[from_lavc_vid_conv] "

#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

// shortcuts
#define R R_SHIFT_IDX
#define G G_SHIFT_IDX
#define B B_SHIFT_IDX

#define MK_RGBA(r, g, b, alpha_mask, depth) \
        FORMAT_RGBA((r), (g), (b), (d.rgb_shift)[R], (d.rgb_shift)[G], \
                    (d.rgb_shift)[B], (alpha_mask), (depth))

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic warning "-Wpass-failed"

// prototypes
static enum colorspace get_cs_for_conv(AVFrame *f, codec_t av_to_uv_pf,
                                       int *lmt_rng);

/// @brief data for av_to_uv conversions
struct av_conv_data {
        char *__restrict dst_buffer;
        AVFrame *__restrict in_frame;
        size_t pitch;
        int    rgb_shift[3];
        // currently following 2 parameters are tweaked only for input
        // YCbCr->RGB conversion for RGB->YCbCr conversion, these should be
        // CS_DFL and 1 always
        enum colorspace cs_coeffs;
        int             lmt_rng; // 0 for full-range src YCbCr, 1 otherwise
};

static void
nv12_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cbcr = (char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                char *dst = d.dst_buffer + d.pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                }
        }
}

static void
rgb24_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        decoder_t vc_copylineRGBtoUYVY = get_decoder_from_to(RGB, UYVY);
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoUYVY((unsigned char *) d.dst_buffer + y * d.pitch,
                                     frame->data[0] + y * frame->linesize[0],
                                     vc_get_linesize(width, UYVY), 0, 0, 0);
        }
}

static void memcpy_data(struct av_conv_data d) __attribute__((unused));
static void
memcpy_data(struct av_conv_data d)
{
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        for (int comp = 0; comp < AV_NUM_DATA_POINTERS; ++comp) {
                if (frame->data[comp] == NULL) {
                        break;
                }
                for (int y = 0; y < height; ++y) {
                        memcpy(d.dst_buffer + y * d.pitch,
                               frame->data[comp] + y * frame->linesize[comp],
                               frame->linesize[comp]);
                }
        }
}

static void
rgb24_to_rgb32(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoRGBA(
                    (unsigned char *) d.dst_buffer + y * d.pitch,
                    frame->data[0] + y * frame->linesize[0],
                    vc_get_linesize(width, RGBA), d.rgb_shift[0],
                    d.rgb_shift[1], d.rgb_shift[2]);
        }
}

static void
gbrp_to_rgb(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        for (int y = 0; y < height; ++y) {
                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        uint8_t *buf = (uint8_t *) d.dst_buffer + y * d.pitch + x * 3;
                        int src_idx = y * frame->linesize[0] + x;
                        buf[0] = frame->data[2][src_idx]; // R
                        buf[1] = frame->data[0][src_idx]; // G
                        buf[2] = frame->data[1][src_idx]; // B
                }
        }
}

static void
gbrp_to_rgba(struct av_conv_data d)
{
        assert((uintptr_t) d.dst_buffer % 4 == 0);
        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << d.rgb_shift[R]) ^
                              (0xFFU << d.rgb_shift[G]) ^
                              (0xFFU << d.rgb_shift[B]);
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        for (int y = 0; y < height; ++y) {
                uint32_t *line = (void *) (d.dst_buffer + y * d.pitch);
                int src_idx = y * frame->linesize[0];

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *line++ = alpha_mask |
                                frame->data[2][src_idx] << d.rgb_shift[R] |
                                frame->data[0][src_idx] << d.rgb_shift[G] |
                                frame->data[1][src_idx] << d.rgb_shift[B];
                        src_idx += 1;
                }
        }
}

#if defined __GNUC__
static inline void gbrap_to_rgb_rgba(struct av_conv_data d, int comp_count)
        __attribute__((always_inline));
#endif
static inline void
gbrap_to_rgb_rgba(struct av_conv_data d, int comp_count)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert(d.rgb_shift[R] == DEFAULT_R_SHIFT &&
               d.rgb_shift[G] == DEFAULT_G_SHIFT &&
               d.rgb_shift[B] == DEFAULT_B_SHIFT);

        for (int y = 0; y < height; ++y) {
                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        uint8_t *buf = (uint8_t *) d.dst_buffer + y * d.pitch + x * comp_count;
                        int src_idx = y * frame->linesize[0] + x;
                        buf[0] = frame->data[2][src_idx]; // R
                        buf[1] = frame->data[0][src_idx]; // G
                        buf[2] = frame->data[1][src_idx]; // B
                        if (comp_count == 4) {
                                buf[3] = frame->data[3][src_idx]; // A
                        }
                }
        }
}

static void
gbrap_to_rgba(struct av_conv_data d)
{
        gbrap_to_rgb_rgba(d, 4);
}

static void
gbrap_to_rgb(struct av_conv_data d)
{
        gbrap_to_rgb_rgba(d, 3);
}

#if defined __GNUC__
static inline void gbrpXXle_to_r10k(struct av_conv_data d, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void
gbrpXXle_to_r10k(struct av_conv_data d, unsigned int in_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) d.dst_buffer + y * d.pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_r >> (in_depth - 8U);
                        *dst++ = ((*src_r++ >> (in_depth - 10U)) & 0x3U) << 6U | *src_g >> (in_depth - 6U);
                        *dst++ = ((*src_g++ >> (in_depth - 10U)) & 0xFU) << 4U | *src_b >> (in_depth - 4U);
                        *dst++ = ((*src_b++ >> (in_depth - 10U)) & 0x3FU) << 2U | 0x3U;
                }
        }
}

static void
gbrp10le_to_r10k(struct av_conv_data d)
{
        gbrpXXle_to_r10k(d, DEPTH10);
}

static void
gbrp16le_to_r10k(struct av_conv_data d)
{
        gbrpXXle_to_r10k(d, DEPTH16);
}

#if defined __GNUC__
static inline void yuv444pXXle_to_r10k(struct av_conv_data d, int depth)
        __attribute__((always_inline));
#endif
static inline void
yuv444pXXle_to_r10k(struct av_conv_data d, int depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        const struct color_coeffs cfs = *get_color_coeffs(d.cs_coeffs, depth * d.lmt_rng);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst =
                    (unsigned char *) d.dst_buffer + y * d.pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t y = (cfs.y_scale * (*src_y++ - (1<<(depth-4))));
                        comp_type_t cr = *src_cr++ - (1<<(depth-1));
                        comp_type_t cb = *src_cb++ - (1<<(depth-1));

                        comp_type_t r = YCBCR_TO_R(cfs, y, cb, cr) >> (COMP_BASE-10+depth);
                        comp_type_t g = YCBCR_TO_G(cfs, y, cb, cr) >> (COMP_BASE-10+depth);
                        comp_type_t b = YCBCR_TO_B(cfs, y, cb, cr) >> (COMP_BASE-10+depth);
                        // r g b is now on 10 bit scale

                        r = CLAMP_FULL(r, 10);
                        g = CLAMP_FULL(g, 10);
                        b = CLAMP_FULL(b, 10);

			*dst++ = r >> 2;
			*dst++ = (r & 0x3) << 6 | g >> 4;
			*dst++ = (g & 0xF) << 4 | b >> 6;
			*dst++ = (b & 0x3F) << 2 | 0x3U;
                }
        }
}

static void
yuv444p10le_to_r10k(struct av_conv_data d)
{
        yuv444pXXle_to_r10k(d, DEPTH10);
}

static void
yuv444p12le_to_r10k(struct av_conv_data d)
{
        yuv444pXXle_to_r10k(d, DEPTH12);
}

static void
yuv444p16le_to_r10k(struct av_conv_data d)
{
        yuv444pXXle_to_r10k(d, DEPTH16);
}

#if defined __GNUC__
static inline void yuv444pXXle_to_r12l(struct av_conv_data d, int depth)
        __attribute__((always_inline));
#endif
static inline void
yuv444pXXle_to_r12l(struct av_conv_data d, int depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        const struct color_coeffs cfs = *get_color_coeffs(d.cs_coeffs, depth * d.lmt_rng);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst =
                    (unsigned char *) d.dst_buffer + y * d.pitch;

                for (int x = 0; x < width; x += 8) {
                        comp_type_t r[8];
                        comp_type_t g[8];
                        comp_type_t b[8];
                        OPTIMIZED_FOR (int j = 0; j < 8; ++j) {
                                comp_type_t y = (cfs.y_scale * (*src_y++ - (1<<(depth-4))));
                                comp_type_t cr = *src_cr++ - (1<<(depth-1));
                                comp_type_t cb = *src_cb++ - (1<<(depth-1));
                                comp_type_t rr = YCBCR_TO_R(cfs, y, cb, cr) >> (COMP_BASE-12+depth);
                                comp_type_t gg = YCBCR_TO_G(cfs, y, cb, cr) >> (COMP_BASE-12+depth);
                                comp_type_t bb = YCBCR_TO_B(cfs, y, cb, cr) >> (COMP_BASE-12+depth);
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

static void
yuv444p10le_to_r12l(struct av_conv_data d)
{
        yuv444pXXle_to_r12l(d, DEPTH10);
}

static void
yuv444p12le_to_r12l(struct av_conv_data d)
{
        yuv444pXXle_to_r12l(d, DEPTH12);
}

static void
yuv444p16le_to_r12l(struct av_conv_data d)
{
        yuv444pXXle_to_r12l(d, DEPTH16);
}

#if defined __GNUC__
static inline void yuv444pXXle_to_rg48(struct av_conv_data d, int depth)
        __attribute__((always_inline));
#endif
static inline void
yuv444pXXle_to_rg48(struct av_conv_data d, int depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert((uintptr_t) d.dst_buffer % 2 == 0);
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        const struct color_coeffs cfs = *get_color_coeffs(d.cs_coeffs, depth * d.lmt_rng);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                uint16_t *dst =
                    (uint16_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t y = (cfs.y_scale * (*src_y++ - (1<<(depth-4))));
                        comp_type_t cr = *src_cr++ - (1<<(depth-1));
                        comp_type_t cb = *src_cb++ - (1<<(depth-1));

                        comp_type_t r = YCBCR_TO_R(cfs, y, cb, cr) >> (COMP_BASE-16+depth);
                        comp_type_t g = YCBCR_TO_G(cfs, y, cb, cr) >> (COMP_BASE-16+depth);
                        comp_type_t b = YCBCR_TO_B(cfs, y, cb, cr) >> (COMP_BASE-16+depth);
                        // r g b is now on 16 bit scale

                        *dst++ = CLAMP_FULL(r, 16);
                        *dst++ = CLAMP_FULL(g, 16);
                        *dst++ = CLAMP_FULL(b, 16);
                }
        }
}

static void
yuv444p10le_to_rg48(struct av_conv_data d)
{
        yuv444pXXle_to_rg48(d, DEPTH10);
}

static void
yuv444p12le_to_rg48(struct av_conv_data d)
{
        yuv444pXXle_to_rg48(d, DEPTH12);
}

static void
yuv444p16le_to_rg48(struct av_conv_data d)
{
        yuv444pXXle_to_rg48(d, DEPTH16);
}

#if defined __GNUC__
static inline void gbrpXXle_to_r12l(struct av_conv_data d, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void
gbrpXXle_to_r12l(struct av_conv_data d, unsigned int in_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

#undef S
#define S(x) ((x) >> (in_depth - 12))

        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst =
                    (unsigned char *) d.dst_buffer + y * d.pitch;

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
static inline void gbrpXXle_to_rgb(struct av_conv_data d, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void
gbrpXXle_to_rgb(struct av_conv_data d, unsigned int in_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst =
                    (unsigned char *) d.dst_buffer + y * d.pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_r++ >> (in_depth - 8U);
                        *dst++ = *src_g++ >> (in_depth - 8U);
                        *dst++ = *src_b++ >> (in_depth - 8U);
                }
        }
}

#if defined __GNUC__
static inline void gbrpXXle_to_rgba(struct av_conv_data d, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void
gbrpXXle_to_rgba(struct av_conv_data d, unsigned int in_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert((uintptr_t) d.dst_buffer % 4 == 0);
        assert((uintptr_t) frame->data[0] % 2 == 0);
        assert((uintptr_t) frame->data[1] % 2 == 0);
        assert((uintptr_t) frame->data[2] % 2 == 0);

        const uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << d.rgb_shift[R]) ^
                                    (0xFFU << d.rgb_shift[G]) ^
                                    (0xFFU << d.rgb_shift[B]);

        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                uint32_t *dst = (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ =
                            alpha_mask |
                            (*src_r++ >> (in_depth - 8U)) << d.rgb_shift[0] |
                            (*src_g++ >> (in_depth - 8U)) << d.rgb_shift[1] |
                            (*src_b++ >> (in_depth - 8U)) << d.rgb_shift[2];
                }
        }
}

static void
gbrp10le_to_rgb(struct av_conv_data d)
{
        gbrpXXle_to_rgb(d, DEPTH10);
}

static void
gbrp10le_to_rgba(struct av_conv_data d)
{
        gbrpXXle_to_rgba(d, DEPTH10);
}

static void
gbrp12le_to_r12l(struct av_conv_data d)
{
        gbrpXXle_to_r12l(d, DEPTH12);
}

static void
gbrp12le_to_r10k(struct av_conv_data d)
{
        gbrpXXle_to_r10k(d, DEPTH12);
}

static void
gbrp12le_to_rgb(struct av_conv_data d)
{
        gbrpXXle_to_rgb(d, DEPTH12);
}

static void
gbrp12le_to_rgba(struct av_conv_data d)
{
        gbrpXXle_to_rgba(d, DEPTH12);
}

static void
gbrp16le_to_r12l(struct av_conv_data d)
{
        gbrpXXle_to_r12l(d, DEPTH16);
}

static void
gbrp16le_to_rgb(struct av_conv_data d)
{
        gbrpXXle_to_rgb(d, DEPTH16);
}

static void
gbrp16le_to_rgba(struct av_conv_data d)
{
        gbrpXXle_to_rgba(d, DEPTH16);
}

#if defined __GNUC__
static inline void gbrpXXle_to_rg48(struct av_conv_data d, unsigned int in_depth)
        __attribute__((always_inline));
#endif
static inline void
gbrpXXle_to_rg48(struct av_conv_data d, unsigned int in_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        assert((uintptr_t) d.dst_buffer % 2 == 0);
        assert((uintptr_t) frame->data[0] % 2 == 0);
        assert((uintptr_t) frame->data[1] % 2 == 0);
        assert((uintptr_t) frame->data[2] % 2 == 0);

        for (ptrdiff_t y = 0; y < height; ++y) {
                uint16_t *src_g = (void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (void *) (frame->data[2] + frame->linesize[2] * y);
                uint16_t *dst = (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_r++ << (16U - in_depth);
                        *dst++ = *src_g++ << (16U - in_depth);
                        *dst++ = *src_b++ << (16U - in_depth);
                }
        }
}

static void
gbrp10le_to_rg48(struct av_conv_data d)
{
        gbrpXXle_to_rg48(d, DEPTH10);
}

static void
gbrp12le_to_rg48(struct av_conv_data d)
{
        gbrpXXle_to_rg48(d, DEPTH12);
}

static void
gbrp16le_to_rg48(struct av_conv_data d)
{
        gbrpXXle_to_rg48(d, DEPTH16);
}

static void
rgb48le_to_rgba(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        decoder_t vc_copylineRG48toRGBA = get_decoder_from_to(RG48, RGBA);
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toRGBA(
                    (unsigned char *) d.dst_buffer + y * d.pitch,
                    frame->data[0] + y * frame->linesize[0],
                    vc_get_linesize(width, RGBA), d.rgb_shift[0],
                    d.rgb_shift[1], d.rgb_shift[2]);
        }
}

static void
rgb48le_to_r12l(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        decoder_t vc_copylineRG48toR12L = get_decoder_from_to(RG48, R12L);

        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toR12L(
                    (unsigned char *) d.dst_buffer + y * d.pitch,
                    frame->data[0] + y * frame->linesize[0],
                    vc_get_linesize(width, R12L), d.rgb_shift[0],
                    d.rgb_shift[1], d.rgb_shift[2]);
        }
}

static void
yuv420p_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < (height + 1) / 2; ++y) {
                int scnd_row = y * 2 + 1;
                if (scnd_row == height) {
                        scnd_row = height - 1;
                }
                char *src_y1 = (char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                char *src_y2 = (char *) in_frame->data[0] + in_frame->linesize[0] * scnd_row;
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst1 = d.dst_buffer + (y * 2) * d.pitch;
                char *dst2 = d.dst_buffer + scnd_row * d.pitch;

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

static void
yuv420p_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height / 2; ++y) {
                uint8_t *src_y1 = (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint8_t *src_y2 = (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 =
                    (uint32_t *) (void *) (d.dst_buffer + (y * 2) * d.pitch);
                uint32_t *dst2 =
                    (uint32_t *) (void *) (d.dst_buffer + (y * 2 + 1) * d.pitch);

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

static void
yuv422p_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = d.dst_buffer + d.pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cr++;
                        *dst++ = *src_y++;
                }
        }
}

static void
yuv422p_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst =
                    (uint32_t *) (void *) (d.dst_buffer + y * d.pitch);

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

#if VUYX_PRESENT
static void
yuv444p_to_vuya(struct av_conv_data d)
{
        const int      width    = d.in_frame->width;
        const int      height   = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;
        for (ptrdiff_t y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] +
                                       (in_frame->linesize[0] * y);
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] +
                                        (in_frame->linesize[1] * y);
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] +
                                        (in_frame->linesize[2] * y);
                unsigned char *dst =
                    (unsigned char *) d.dst_buffer + (d.pitch * y);
                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        enum { ALPHA = 0xFF };
                        *dst++ = *src_cr++;
                        *dst++ = *src_cb++;
                        *dst++ = *src_y++;
                        *dst++ = ALPHA;
                }
        }
}
#endif // VUYX_PRESENT

static void
yuv444p_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = d.dst_buffer + d.pitch * y;

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

static void
yuv444p16le_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y + 1;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y + 1;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y + 1;
                unsigned char *dst =
                    (unsigned char *) d.dst_buffer + d.pitch * y;

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

static void
yuv444p_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst =
                    (uint32_t *) (void *) (d.dst_buffer + y * d.pitch);

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
static inline void nv12_to_rgb(struct av_conv_data d, bool rgba)
        __attribute__((always_inline));
#endif
static inline void
nv12_to_rgb(struct av_conv_data d, bool rgba)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        enum {
                S_DEPTH = 8,
        };
        assert((uintptr_t) d.dst_buffer % 4 == 0);

        const uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << d.rgb_shift[R]) ^
                                    (0xFFU << d.rgb_shift[G]) ^
                                    (0xFFU << d.rgb_shift[B]);
        const struct color_coeffs cfs = *get_color_coeffs(d.cs_coeffs, S_DEPTH * d.lmt_rng);

        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cbcr = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                unsigned char *dst =
                    (unsigned char *) d.dst_buffer + d.pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cb = *src_cbcr++ - 128;
                        comp_type_t cr = *src_cbcr++ - 128;
                        comp_type_t y = (*src_y++ - 16) * cfs.y_scale;
                        comp_type_t r = YCBCR_TO_R(cfs, y, cb, cr) >> COMP_BASE;
                        comp_type_t g = YCBCR_TO_G(cfs, y, cb, cr) >> COMP_BASE;
                        comp_type_t b = YCBCR_TO_B(cfs, y, cb, cr) >> COMP_BASE;
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = MK_RGBA(r, g, b, alpha_mask, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL(r, 8);
                                *dst++ = CLAMP_FULL(g, 8);
                                *dst++ = CLAMP_FULL(b, 8);
                        }

                        y = (*src_y++ - 16) * cfs.y_scale;
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = MK_RGBA(r, g, b, alpha_mask, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL(r, 8);
                                *dst++ = CLAMP_FULL(g, 8);
                                *dst++ = CLAMP_FULL(b, 8);
                        }
                }
        }
}

static void nv12_to_rgb24(struct av_conv_data d)
{
        nv12_to_rgb(d, false);
}

static void nv12_to_rgb32(struct av_conv_data d)
{
        nv12_to_rgb(d, true);
}

#if defined __GNUC__
static inline void yuv8p_to_rgb(struct av_conv_data d, int subsampling, bool rgba) __attribute__((always_inline));
#endif
/**
 * Changes pixel format from planar 8-bit 422 and 420 YUV to packed RGB/A.
 */
static inline void yuv8p_to_rgb(struct av_conv_data d, int subsampling, bool rgba)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        enum {
                S_DEPTH = 8,
        };
        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << d.rgb_shift[R]) ^
                              (0xFFU << d.rgb_shift[G]) ^
                              (0xFFU << d.rgb_shift[B]);
        const struct color_coeffs cfs = *get_color_coeffs(d.cs_coeffs, S_DEPTH * d.lmt_rng);

        for(int y = 0; y < height / 2; ++y) {
                unsigned char *src_y1 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                unsigned char *src_y2 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1);
                unsigned char *dst1 =
                    (unsigned char *) d.dst_buffer + d.pitch * (y * 2);
                unsigned char *dst2 =
                    (unsigned char *) d.dst_buffer + d.pitch * (y * 2 + 1);

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

#define WRITE_RES_YUV8P_TO_RGB(DST) {\
                                r >>= COMP_BASE;\
                                g >>= COMP_BASE;\
                                b >>= COMP_BASE;\
                                if (rgba) {\
                                        *((uint32_t *)(void *) DST) = MK_RGBA(r, g, b, alpha_mask, 8);\
                                        DST += 4;\
                                } else {\
                                        *DST++ = CLAMP_FULL(r, 8);\
                                        *DST++ = CLAMP_FULL(g, 8);\
                                        *DST++ = CLAMP_FULL(b, 8);\
                                }\
                        }\

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cb = *src_cb1++ - 128;
                        comp_type_t cr = *src_cr1++ - 128;
                        comp_type_t y = (*src_y1++ - 16) * cfs.y_scale;
                        comp_type_t r = YCBCR_TO_R(cfs, y, cb, cr);
                        comp_type_t g = YCBCR_TO_G(cfs, y, cb, cr);
                        comp_type_t b = YCBCR_TO_B(cfs, y, cb, cr);
                        WRITE_RES_YUV8P_TO_RGB(dst1)

                        y = (*src_y1++ - 16) * cfs.y_scale;
                        r = YCBCR_TO_R(cfs, y, cb, cr);
                        g = YCBCR_TO_G(cfs, y, cb, cr);
                        b = YCBCR_TO_B(cfs, y, cb, cr);
                        WRITE_RES_YUV8P_TO_RGB(dst1)

                        if (subsampling == 422) {
                                cb = *src_cb2++ - 128;
                                cr = *src_cr2++ - 128;
                        }
                        y = (*src_y2++ - 16) * cfs.y_scale;
                        r = YCBCR_TO_R(cfs, y, cb, cr);
                        g = YCBCR_TO_G(cfs, y, cb, cr);
                        b = YCBCR_TO_B(cfs, y, cb, cr);
                        WRITE_RES_YUV8P_TO_RGB(dst2)

                        y = (*src_y2++ - 16) * cfs.y_scale;
                        r = YCBCR_TO_R(cfs, y, cb, cr);
                        g = YCBCR_TO_G(cfs, y, cb, cr);
                        b = YCBCR_TO_B(cfs, y, cb, cr);
                        WRITE_RES_YUV8P_TO_RGB(dst2)
                }
        }
}

static void
yuv420p_to_rgb24(struct av_conv_data d)
{
        yuv8p_to_rgb(d, 420, false);
}

static void
yuv420p_to_rgb32(struct av_conv_data d)
{
        yuv8p_to_rgb(d, 420, true);
}

static void
yuv422p_to_rgb24(struct av_conv_data d)
{
        yuv8p_to_rgb(d, 422, false);
}

static void
yuv422p_to_rgb32(struct av_conv_data d)
{
        yuv8p_to_rgb(d, 422, true);
}

/**
 * Changes pixel format from planar YUV 444 to packed RGB/A.
 */
#if defined __GNUC__
static inline void yuv444p_to_rgb(struct av_conv_data d, bool rgba)
        __attribute__((always_inline));
#endif
static inline void
yuv444p_to_rgb(struct av_conv_data d, bool rgba)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        enum {
                S_DEPTH = 8,
        };
        assert((uintptr_t) d.dst_buffer % 4 == 0);

        uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << d.rgb_shift[R]) ^
                              (0xFFU << d.rgb_shift[G]) ^
                              (0xFFU << d.rgb_shift[B]);
        const struct color_coeffs cfs = *get_color_coeffs(d.cs_coeffs, S_DEPTH * d.lmt_rng);

        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst =
                    (unsigned char *) d.dst_buffer + d.pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ * cfs.y_scale;
                        comp_type_t r = YCBCR_TO_R(cfs, y, cb, cr) >> COMP_BASE;
                        comp_type_t g = YCBCR_TO_G(cfs, y, cb, cr) >> COMP_BASE;
                        comp_type_t b = YCBCR_TO_B(cfs, y, cb, cr) >> COMP_BASE;
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = MK_RGBA(r, g, b, alpha_mask, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP(r, 1, 254);
                                *dst++ = CLAMP(g, 1, 254);
                                *dst++ = CLAMP(b, 1, 254);
                        }
                }
        }
}

static void
yuv444p_to_rgb24(struct av_conv_data d)
{
        yuv444p_to_rgb(d, false);
}

static void
yuv444p_to_rgb32(struct av_conv_data d)
{
        yuv444p_to_rgb(d, true);
}

static void
yuv420p10le_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 =
                    (uint32_t *) (void *) (d.dst_buffer + (y * 2) * d.pitch);
                uint32_t *dst2 = (uint32_t *) (void *) (d.dst_buffer +
                                                        (y * 2 + 1) * d.pitch);

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

static void
yuv422p10le_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst =
                    (uint32_t *) (void *) (d.dst_buffer + y * d.pitch);

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

#if defined __GNUC__
static inline void yuv444p1Xle_to_v210(struct av_conv_data d, int in_depth)
        __attribute__((always_inline));
#endif
static inline void yuv444p1Xle_to_v210(struct av_conv_data d, int in_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst =
                    (uint32_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = ((src_cb[0] >> (in_depth - 10U)) + (src_cb[1] >> (in_depth - 10U))) / 2;
                        w0_0 = w0_0 | (*src_y++ >> (in_depth - 10U)) << 10U;
                        w0_0 = w0_0 | ((src_cr[0] >> (in_depth - 10U)) + (src_cr[1] >> (in_depth - 10U))) / 2 << 20U;
                        src_cb += 2;
                        src_cr += 2;

                        w0_1 = *src_y++;
                        w0_1 = w0_1 | ((src_cb[0] >> (in_depth - 10U)) + (src_cb[1] >> (in_depth - 10U))) / 2 << 10U;
                        w0_1 = w0_1 | (*src_y++ >> (in_depth - 10U)) << 20U;
                        src_cb += 2;

                        w0_2 = ((src_cr[0] >> (in_depth - 10U)) + (src_cr[1] >> (in_depth - 10U))) / 2;
                        w0_2 = w0_2 | (*src_y++ >> (in_depth - 10U)) << 10U;
                        w0_2 = w0_2 | ((src_cb[0] >> (in_depth - 10U)) + (src_cb[1] >> (in_depth - 10U))) / 2 << 20U;
                        src_cr += 2;
                        src_cb += 2;

                        w0_3 = *src_y++;
                        w0_3 = w0_3 | ((src_cr[0] >> (in_depth - 10U)) + (src_cr[1] >> (in_depth - 10U))) / 2 << 10U;
                        w0_3 = w0_3 | ((*src_y++ >> (in_depth - 10U))) << 20U;
                        src_cr += 2;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}

static void
yuv444p10le_to_v210(struct av_conv_data d)
{
        yuv444p1Xle_to_v210(d, DEPTH10);
}

static void
yuv444p12le_to_v210(struct av_conv_data d)
{
        yuv444p1Xle_to_v210(d, DEPTH12);
}

static void
yuv444p16le_to_v210(struct av_conv_data d)
{
        yuv444p1Xle_to_v210(d, DEPTH16);
}

static void yuv420p10le_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst1 =
                    (uint8_t *) (void *) (d.dst_buffer + (y * 2) * d.pitch);
                uint8_t *dst2 =
                    (uint8_t *) (void *) (d.dst_buffer + (y * 2 + 1) * d.pitch);

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

static void yuv422p10le_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst =
                    (uint8_t *) (void *) (d.dst_buffer + y * d.pitch);

                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++ >> 2;
                        *dst++ = *src_y++ >> 2;
                        *dst++ = *src_cr++ >> 2;
                        *dst++ = *src_y++ >> 2;
                }
        }
}

#if defined __GNUC__
static inline void yuv444p1Xle_to_uyvy(struct av_conv_data d, int in_depth)
        __attribute__((always_inline));
#endif
static inline void
yuv444p1Xle_to_uyvy(struct av_conv_data d, int in_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst =
                    (uint8_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = (src_cb[0] + src_cb[1] + 1) / 2 >> (in_depth - 8U);
                        *dst++ = *src_y++ >> (in_depth - 8U);
                        *dst++ = (src_cr[0] + src_cr[1] + 1) / 2 >> (in_depth - 8U);
                        *dst++ = *src_y++ >> (in_depth - 8U);
                        src_cb += 2;
                        src_cr += 2;
                }
        }
}

static void
yuv444p10le_to_uyvy(struct av_conv_data d)
{
        yuv444p1Xle_to_uyvy(d, DEPTH10);
}

static void yuv444p12le_to_uyvy(struct av_conv_data d)
{
        yuv444p1Xle_to_uyvy(d, DEPTH12);
}

#if defined __GNUC__
static inline void yuv444p1Xle_to_y416(struct av_conv_data d, int in_depth)
        __attribute__((always_inline));
#endif
static void
yuv444p1Xle_to_y416(struct av_conv_data d, int in_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) d.dst_buffer % 2 == 0);
        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) in_frame->data[1] % 2 == 0);
        assert((uintptr_t) in_frame->data[2] % 2 == 0);
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint16_t *dst =
                    (uint16_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_cb++ << (16U - in_depth); // U
                        *dst++ = *src_y++ << (16U - in_depth); // Y
                        *dst++ = *src_cr++ << (16U - in_depth); // V
                        *dst++ = 0xFFFFU; // A
                }
        }
}

static void
yuv444p10le_to_y416(struct av_conv_data d)
{
        yuv444p1Xle_to_y416(d, DEPTH10);
}

static void
yuv444p12le_to_y416(struct av_conv_data d)
{
        yuv444p1Xle_to_y416(d, DEPTH12);
}

static void
yuv444p16le_to_y416(struct av_conv_data d)
{
        yuv444p1Xle_to_y416(d, DEPTH16);
}

#if defined __GNUC__
static inline void yuvp10le_to_rgb(struct av_conv_data d, int subsampling,
                                   int out_bit_depth)
    __attribute__((always_inline));
#endif
static inline void
yuvp10le_to_rgb(struct av_conv_data d, int subsampling, int out_bit_depth)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *frame = d.in_frame;

        enum {
                S_DEPTH = 10,
        };
        assert((uintptr_t) d.dst_buffer % 4 == 0);
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        assert(subsampling == 422 || subsampling == 420);
        assert(out_bit_depth == 24 || out_bit_depth == 30 || out_bit_depth == 32);

        const uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << d.rgb_shift[R]) ^
                                    (0xFFU << d.rgb_shift[G]) ^
                                    (0xFFU << d.rgb_shift[B]);
        const int bpp = out_bit_depth == 30 ? 10 : 8;
        const struct color_coeffs cfs = *get_color_coeffs(d.cs_coeffs, S_DEPTH * d.lmt_rng);

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
                unsigned char *dst1 =
                    (unsigned char *) d.dst_buffer + (2 * y) * d.pitch;
                unsigned char *dst2 =
                    (unsigned char *) d.dst_buffer + (2 * y + 1) * d.pitch;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cr = *src_cr1++ - (1<<9);
                        comp_type_t cb = *src_cb1++ - (1<<9);
                        comp_type_t rr = YCBCR_TO_R(cfs, 0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        comp_type_t gg = YCBCR_TO_G(cfs, 0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        comp_type_t bb = YCBCR_TO_B(cfs, 0, cb, cr) >> (COMP_BASE + (10 - bpp));

#                       define WRITE_RES_YUV10P_TO_RGB(Y, DST) {\
                                comp_type_t r = Y + rr;\
                                comp_type_t g = Y + gg;\
                                comp_type_t b = Y + bb;\
                                r = CLAMP_FULL(r, bpp);\
                                g = CLAMP_FULL(g, bpp);\
                                b = CLAMP_FULL(b, bpp);\
                                if (out_bit_depth == 32) {\
                                        *((uint32_t *) (void *) (DST)) = \
                                            alpha_mask | (r << d.rgb_shift[R] |\
                                            g << d.rgb_shift[G] | \
                                            b << d.rgb_shift[B]); \
                                        DST += 4;\
                                } else if (out_bit_depth == 24) {\
                                        *DST++ = r;\
                                        *DST++ = g;\
                                        *DST++ = b;\
                                } else {\
                                        *((uint32_t *)(void *) (DST)) = r >> 2U | (r & 0x3U) << 14 | g >> 4U << 8U | (g & 0xFU) << 20U | b >> 6U << 16U | (b & 0x3FU) << 26U | 0x3U << 24U;\
                                        /*      == htonl(r << 22U | g << 12U | b << 2U) */ \
                                        DST += 4;\
                                }\
                        }

                        comp_type_t y1 = (cfs.y_scale * (*src_y1++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y1, dst1)

                        comp_type_t y11 = (cfs.y_scale * (*src_y1++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y11, dst1)

                        if (subsampling == 422) {
                                cr = *src_cr2++ - (1<<9);
                                cb = *src_cb2++ - (1<<9);
                                rr = YCBCR_TO_R(cfs, 0, cb, cr) >> (COMP_BASE + (10 - bpp));
                                gg = YCBCR_TO_G(cfs, 0, cb, cr) >> (COMP_BASE + (10 - bpp));
                                bb = YCBCR_TO_B(cfs, 0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        }

                        comp_type_t y2 = (cfs.y_scale * (*src_y2++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y2, dst2)

                        comp_type_t y22 = (cfs.y_scale * (*src_y2++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y22, dst2)
                }
        }
}

#define MAKE_YUV_TO_RGB_FUNCTION_NAME(subs, out_bit_depth) yuv ## subs ## p10le_to_rgb ## out_bit_depth

#define MAKE_YUV_TO_RGB_FUNCTION(subs, out_bit_depth) \
        static void MAKE_YUV_TO_RGB_FUNCTION_NAME(subs, out_bit_depth)( \
            struct av_conv_data d) \
        { \
                yuvp10le_to_rgb(d, subs, out_bit_depth); \
        }

MAKE_YUV_TO_RGB_FUNCTION(420, 24)
MAKE_YUV_TO_RGB_FUNCTION(420, 30)
MAKE_YUV_TO_RGB_FUNCTION(420, 32)
MAKE_YUV_TO_RGB_FUNCTION(422, 24)
MAKE_YUV_TO_RGB_FUNCTION(422, 30)
MAKE_YUV_TO_RGB_FUNCTION(422, 32)

#if defined __GNUC__
static inline void yuv444p10le_to_rgb(struct av_conv_data d, bool rgba)
        __attribute__((always_inline));
#endif
static inline void
yuv444p10le_to_rgb(struct av_conv_data d, bool rgba)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        enum {
                S_DEPTH = 10,
        };
        const uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << d.rgb_shift[R]) ^
                                    (0xFFU << d.rgb_shift[G]) ^
                                    (0xFFU << d.rgb_shift[B]);
        const struct color_coeffs cfs = *get_color_coeffs(d.cs_coeffs, S_DEPTH * d.lmt_rng);

        for (int y = 0; y < height; y++) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst =
                    (uint8_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t cb = *src_cb++ - (1 << (S_DEPTH - 1));
                        comp_type_t cr = *src_cr++ - (1 << (S_DEPTH - 1));
                        comp_type_t y =
                            (*src_y++ - (1 << (S_DEPTH - 4))) * cfs.y_scale;
                        comp_type_t r =
                            YCBCR_TO_R(cfs, y, cb, cr) >> (COMP_BASE + 2);
                        comp_type_t g =
                            YCBCR_TO_G(cfs, y, cb, cr) >> (COMP_BASE + 2);
                        comp_type_t b =
                            YCBCR_TO_B(cfs, y, cb, cr) >> (COMP_BASE + 2);
                        if (rgba) {
                                *(uint32_t *)(void *) dst = MK_RGBA(r, g, b, alpha_mask, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL(r, 8);
                                *dst++ = CLAMP_FULL(g, 8);
                                *dst++ = CLAMP_FULL(b, 8);
                        }
                }
        }
}

static void
yuv444p10le_to_rgb24(struct av_conv_data d)
{
        yuv444p10le_to_rgb(d, false);
}

static void
yuv444p10le_to_rgb32(struct av_conv_data d)
{
        yuv444p10le_to_rgb(d, true);
}

#if P210_PRESENT
static void
p210le_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) in_frame->data[1] % 2 == 0);
        assert((uintptr_t) d.dst_buffer % 4 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cbcr = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint32_t *dst =
                    (uint32_t *) (void *) (d.dst_buffer + y * d.pitch);

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

static void
p010le_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) in_frame->data[1] % 2 == 0);
        assert((uintptr_t) d.dst_buffer % 4 == 0 && d.pitch % 4 == 0);
        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *) (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *) (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cbcr = (uint16_t *)(void *) (in_frame->data[1] + in_frame->linesize[1] * y);
                uint32_t *dst1 =
                    (uint32_t *) (void *) (d.dst_buffer + (y * 2) * d.pitch);
                uint32_t *dst2 = (uint32_t *) (void *) (d.dst_buffer +
                                                        (y * 2 + 1) * d.pitch);

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

static void
p010le_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cbcr = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *dst1 =
                    (uint8_t *) (void *) (d.dst_buffer + (y * 2) * d.pitch);
                uint8_t *dst2 =
                    (uint8_t *) (void *) (d.dst_buffer + (y * 2 + 1) * d.pitch);

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

#if P210_PRESENT
static void
p210le_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) in_frame->data[1] % 2 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cbcr = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *dst =
                    (uint8_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst = *src_cbcr++ >> 8;
                        *dst++ = *src_y++ >> 8;
                        *dst++ = *src_cbcr++ >> 8;
                        *dst++ = *src_y++ >> 8;
                }
        }
}
#endif

#if XV3X_PRESENT
static void xv30_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 4 == 0);
        for (ptrdiff_t y = 0; y < height; ++y) {
                uint32_t *src = (void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *dst = (void *) (d.dst_buffer + y * d.pitch);
                int x = 0;
                OPTIMIZED_FOR ( ; x < width - 1; x += 2) {
                        uint32_t in1 = *src++;
                        uint32_t in2 = *src++;
                        *dst++ = (((in1 >> 2U) & 0xFFU) + (((in2 >> 2U) & 0xFFU) + 1)) >> 1;
                        *dst++ = (in1 >> 12U) & 0xFFU;
                        *dst++ = (((in1 >> 22U) & 0xFFU) + (((in2 >> 22U) & 0xFFU) + 1)) >> 1;
                        *dst++ = (in2 >> 12U) & 0xFFU;
                }
                if (x < width) {
                        uint32_t last = *src++;
                        *dst++ = (last >> 2U) & 0xFFU;
                        *dst++ = (last >> 12U) & 0xFFU;
                        *dst++ = (last >> 22U) & 0xFFU;
                        *dst++ = 0;
                }
        }
}

static void
xv30_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 4 == 0);
        assert((uintptr_t) d.dst_buffer % 4 == 0 && d.pitch % 4 == 0);
        for(int y = 0; y < height; ++y) {
                uint32_t *src = (uint32_t *)(void *) (in_frame->data[0] + in_frame->linesize[0] * y);
                uint32_t *dst =
                    (uint32_t *) (void *) (d.dst_buffer + y * d.pitch);

#define FETCH_IN \
                in0 = *src++; \
                in1 = *src++; \
                u = ((in0 & 0x3FFU) + (in1 & 0x3FFU) + 1) >> 1; \
                y0 = (in0 >> 10U) & 0x3FFU; \
                v = ((in0 >> 20U & 0x3FFU) + ((in1 >> 20U & 0x3FFU) + 1)) >> 1; \
                y1 = (in1 >> 10U) & 0x3FFU; \

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t in0, in1;
                        uint32_t u, y0, v, y1;

                        FETCH_IN
                        *dst++ = v << 20U | y0 << 10U | u;
                        uint32_t tmp = y1;
                        FETCH_IN
                        *dst++ = y0 << 20U | u << 10U | tmp;
                        tmp = y1 << 10U | v;
                        FETCH_IN
                        *dst++ = u << 20U | tmp;
                        *dst++ = y1 << 20U | v << 10U | y0;
                }
        }
#undef FETCH_IN
}

static void
xv30_to_y416(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 4 == 0);
        assert((uintptr_t) d.dst_buffer % 2 == 0 && d.pitch % 2 == 0);
        for (ptrdiff_t y = 0; y < height; ++y) {
                uint32_t *src = (void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *dst = (void *)(d.dst_buffer + y * d.pitch);
                OPTIMIZED_FOR (int x = 0; x < width; x += 1) {
                        uint32_t in = *src++;
                        *dst++ = (in & 0x3FFU) << 6U;
                        *dst++ = ((in >> 10U) & 0x3FFU) << 6U;
                        *dst++ = ((in >> 20U) & 0x3FFU) << 6U;
                        *dst++ = 0xFFFFU;
                }
        }
}
#endif // XV3X_PRESENT

#if Y210_PRESENT
static void y210_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) d.dst_buffer % 4 == 0 && d.pitch % 4 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src = (uint16_t *)(void *) (in_frame->data[0] + in_frame->linesize[0] * y);
                uint32_t *dst =
                    (uint32_t *) (void *) (d.dst_buffer + y * d.pitch);

                // Y210 is like YUYV but with 10-bit in high bits of 16-bit container
#define FETCH_IN \
                y0 = *src++ >> 6U; \
                u = *src++ >> 6U; \
                y1 = *src++ >> 6U; \
                v = *src++ >> 6U;

                OPTIMIZED_FOR (int x = 0; x < (width + 5) / 6; ++x) {
                        unsigned y0, u, y1, v;
                        unsigned tmp;

                        FETCH_IN
                        *dst++ = v << 20U | y0 << 10U | u;
                        tmp = y1;
                        FETCH_IN
                        *dst++ = y0 << 20U | u << 10U | tmp;
                        tmp = y1 << 10U | v;
                        FETCH_IN
                        *dst++ = u << 20U | tmp;
                        *dst++ = y1 << 20U | v << 10U | y0;
                }
        }
#undef FETCH_IN
}

static void
y210_to_y416(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) d.dst_buffer % 2 == 0 && d.pitch % 2 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src = (void *) (in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *dst = (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < (width + 1) / 2; ++x) {
                        unsigned y0, u, y1, v;
                        y0 = *src++;
                        u = *src++;
                        y1 = *src++;
                        v = *src++;

                        *dst++ = u;
                        *dst++ = y0;
                        *dst++ = v;
                        *dst++ = 0xFFFFU;
                        *dst++ = u;
                        *dst++ = y1;
                        *dst++ = v;
                        *dst++ = 0xFFFFU;
                }
        }
}

static void y210_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                uint8_t *src = (void *) (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *dst = (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < (width + 1) / 2; ++x) {
                        unsigned y0, u, y1, v;
                        y0 = src[1];
                        u = src[3];
                        y1 = src[5];
                        v = src[7];
                        src += 8;

                        *dst++ = u;
                        *dst++ = y0;
                        *dst++ = v;
                        *dst++ = y1;
                }
        }
}
#endif // Y210_PRESENT

#ifdef HWACC_VDPAU
static void av_vdpau_to_ug_vdpau(struct av_conv_data d)
{
        struct video_frame_callbacks *callbacks = d.in_frame->opaque;

        hw_vdpau_frame *out = (hw_vdpau_frame *)(void *) d.dst_buffer;

        hw_vdpau_frame_init(out);

        hw_vdpau_frame_from_avframe(out, d.in_frame);

        callbacks->recycle = hw_vdpau_recycle_callback; 
        callbacks->copy = hw_vdpau_copy_callback; 
}
#endif

static void hw_drm_recycle_callback(struct video_frame *frame){
        for(unsigned i = 0; i < frame->tile_count; i++){
                struct drm_prime_frame *drm_frame = (struct drm_prime_frame *)(void *) frame->tiles[i].data;
                av_frame_free(&drm_frame->av_frame);
                memset(drm_frame, 0, sizeof(struct drm_prime_frame));
        }

        frame->callbacks.recycle = NULL;
}

static void av_drm_prime_to_ug_drm_prime(struct av_conv_data d)
{
        const AVFrame *in_frame = d.in_frame;

        struct video_frame_callbacks *callbacks = in_frame->opaque;

        struct drm_prime_frame *out =
            (struct drm_prime_frame *) (void *) d.dst_buffer;
        memset(out, 0, sizeof(struct drm_prime_frame));

        assert((uintptr_t) in_frame->data[0] % alignof(AVDRMFrameDescriptor) ==
               0);
        AVDRMFrameDescriptor *av_drm_frame =
            (struct AVDRMFrameDescriptor *) (void *) in_frame->data[0];
        assert(av_drm_frame->nb_layers == 1);
        AVDRMLayerDescriptor *layer = &av_drm_frame->layers[0];

        out->fd_count = av_drm_frame->nb_objects;

        for(int i = 0; i < av_drm_frame->nb_objects; i++){
                out->dmabuf_fds[i] = av_drm_frame->objects[i].fd;
        }

        out->planes = layer->nb_planes;
        out->drm_format = layer->format;

        for(int i = 0; i < layer->nb_planes; i++){
                out->fd_indices[i] = layer->planes[i].object_index;
                out->modifiers[i] = av_drm_frame->objects[layer->planes[i].object_index].format_modifier;
                out->offsets[i] = layer->planes[i].offset;
                out->pitches[i] = layer->planes[i].pitch;
        }

        out->av_frame = av_frame_clone(in_frame);
        callbacks->recycle = hw_drm_recycle_callback;
}

static void ayuv64_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for(int y = 0; y < height; ++y) {
                uint8_t *src = (uint8_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *dst =
                    (uint8_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < ((width + 1) & ~1); ++x) {
                        *dst++ = (src[1] + src[9] / 2);  // U
                        *dst++ = src[3];                 // Y
                        *dst++ = (src[5] + src[13] / 2); // V
                        *dst++ = src[11];                // Y
                }
        }
}

static void ayuv64_to_y416(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) d.dst_buffer % 2 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *dst =
                    (uint16_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = src[2]; // U
                        *dst++ = src[1]; // Y
                        *dst++ = src[3]; // V
                        *dst++ = src[0]; // A
                        src += 4;
                }
        }
}

static void ayuv64_to_v210(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) in_frame->data[0] % 2 == 0);
        assert((uintptr_t) d.dst_buffer % 4 == 0);
        for(int y = 0; y < height; ++y) {
                uint16_t *src = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint32_t *dst =
                    (uint32_t *) (void *) (d.dst_buffer + y * d.pitch);

                OPTIMIZED_FOR (int x = 0; x < (width + 5) / 6; ++x) {
                        uint32_t w;

                        w = ((src[2] >> 6U) + (src[6] >> 6U)) / 2;  // U0
                        w = w | (src[1] >> 6U) << 10U; // U0 Y0a
                        w = w | ((src[3] >> 6U) + (src[7] >> 6U)) / 2 << 20U; // U0 Y0a V0
                        *dst++ = w; // flush output

                        w = src[5]; // Y0b |
                        src += 8; // move to next 2 words
                        w = w | ((src[2] >> 6U) + (src[6] >> 6U)) / 2 << 10U; // Y0b | U1
                        w = w | (src[1] >> 6U) << 20U; // Y0b | U1 Y1a
                        *dst++ = w;

                        w = ((src[3] >> 6U) + (src[7] >> 6U)) / 2; // V1
                        w = w | (src[5] >> 6U) << 10U; // V1 Y1a |
                        src += 8;
                        w = w | ((src[2] >> 6U) + (src[6] >> 6U)) / 2 << 20U; // V1 Y1a | U2
                        *dst++ = w;

                        w = src[1]; // Y2a
                        w = w | ((src[3] >> 6U) + (src[7] >> 6U)) / 2 << 10U; // Y2a V2
                        w = w | ((src[5] >> 6U)) << 20U; // Y2a V2 Y2b |
                        *dst++ = w;
                        src += 8;
                }
        }
}

static void vuya_to_uyvy(struct av_conv_data d) __attribute__((unused));
static void
vuya_to_uyvy(struct av_conv_data d)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        for (ptrdiff_t y = 0; y < height; ++y) {
                unsigned char *src = in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *dst = (void *) (d.dst_buffer + d.pitch * y);

                OPTIMIZED_FOR (int x = 0; x < width / 2; x += 1) {
                        *dst++ = (src[1] + src[5] + 1U) >> 1U;
                        *dst++ = src[2];
                        *dst++ = (src[0] + src[4] + 1U) >> 1U;
                        *dst++ = src[6];
                        src += 8;
                }
                if (width % 2 == 1) {
                        *dst++ = src[1];
                        *dst++ = src[2];
                        *dst++ = src[0];
                        *dst++ = 0;
                }
        }
}

#if defined __GNUC__
static inline void vuyax_to_y416(struct av_conv_data d, bool use_alpha)
        __attribute__((always_inline));
#endif
static inline void vuyax_to_y416(struct av_conv_data d, bool use_alpha)
{
        const int width = d.in_frame->width;
        const int height = d.in_frame->height;
        const AVFrame *in_frame = d.in_frame;

        assert((uintptr_t) d.dst_buffer % 2 == 0);
        for (ptrdiff_t y = 0; y < height; ++y) {
                unsigned char *src = in_frame->data[0] + in_frame->linesize[0] * y;
                uint16_t *dst = (void *) (d.dst_buffer + d.pitch * y);

                OPTIMIZED_FOR (int x = 0; x < width; x += 1) {
                        *dst++ = src[1] << 8U;
                        *dst++ = src[2] << 8U;
                        *dst++ = src[0] << 8U;
                        *dst++ = use_alpha ? src[3] << 8U : 0xFFFF;
                        src += 4;
                }
        }
}
static void vuya_to_y416(struct av_conv_data d) __attribute__((unused));
static void vuya_to_y416(struct av_conv_data d) {
        vuyax_to_y416(d, true);
}
static void vuyx_to_y416(struct av_conv_data d) __attribute__((unused));
static void vuyx_to_y416(struct av_conv_data d) {
        vuyax_to_y416(d, false);
}

typedef void av_to_uv_convert_f(struct av_conv_data d);
typedef av_to_uv_convert_f *av_to_uv_convert_fp;

struct av_to_uv_convert_state {
        av_to_uv_convert_fp convert;
        codec_t src_pixfmt; ///< after av_to_uv conversion (intermediate);
                            ///< VC_NONE if no further conversion needed
        codec_t dst_pixfmt;
        decoder_t dec;
        struct av_to_uv_convert_cuda *cuda_conv_state;
};

struct av_to_uv_conversion {
        enum AVPixelFormat av_codec;
        codec_t uv_codec;
        av_to_uv_convert_fp convert;
};

static const struct av_to_uv_conversion av_to_uv_conversions[] = {
        // 10-bit YUV
        {AV_PIX_FMT_YUV420P10LE, v210, yuv420p10le_to_v210},
        {AV_PIX_FMT_YUV420P10LE, UYVY, yuv420p10le_to_uyvy},
        {AV_PIX_FMT_YUV420P10LE, RGB, yuv420p10le_to_rgb24},
        {AV_PIX_FMT_YUV420P10LE, RGBA, yuv420p10le_to_rgb32},
        {AV_PIX_FMT_YUV420P10LE, R10k, yuv420p10le_to_rgb30},
        {AV_PIX_FMT_YUV422P10LE, v210, yuv422p10le_to_v210},
        {AV_PIX_FMT_YUV422P10LE, UYVY, yuv422p10le_to_uyvy},
        {AV_PIX_FMT_YUV422P10LE, RGB, yuv422p10le_to_rgb24},
        {AV_PIX_FMT_YUV422P10LE, RGBA, yuv422p10le_to_rgb32},
        {AV_PIX_FMT_YUV422P10LE, R10k, yuv422p10le_to_rgb30},
        {AV_PIX_FMT_YUV444P10LE, v210, yuv444p10le_to_v210},
        {AV_PIX_FMT_YUV444P10LE, UYVY, yuv444p10le_to_uyvy},
        {AV_PIX_FMT_YUV444P10LE, R10k, yuv444p10le_to_r10k},
        {AV_PIX_FMT_YUV444P10LE, RGB, yuv444p10le_to_rgb24},
        {AV_PIX_FMT_YUV444P10LE, RGBA, yuv444p10le_to_rgb32},
        {AV_PIX_FMT_YUV444P10LE, R12L, yuv444p10le_to_r12l},
        {AV_PIX_FMT_YUV444P10LE, RG48, yuv444p10le_to_rg48},
        {AV_PIX_FMT_YUV444P10LE, Y416, yuv444p10le_to_y416},
#if P210_PRESENT
        {AV_PIX_FMT_P210LE, v210, p210le_to_v210},
        {AV_PIX_FMT_P210LE, UYVY, p210le_to_uyvy},
#endif
#if XV3X_PRESENT
        {AV_PIX_FMT_XV30,   UYVY, xv30_to_uyvy},
        {AV_PIX_FMT_XV30,   v210, xv30_to_v210},
        {AV_PIX_FMT_XV30,   Y416, xv30_to_y416},
        {AV_PIX_FMT_Y212,   UYVY, y210_to_uyvy},
        {AV_PIX_FMT_Y212,   v210, y210_to_v210},
        {AV_PIX_FMT_Y212,   Y416, y210_to_y416},
        {AV_PIX_FMT_Y212,   Y216, memcpy_data},
#endif
#if Y210_PRESENT
        {AV_PIX_FMT_Y210,   UYVY, y210_to_uyvy},
        {AV_PIX_FMT_Y210,   v210, y210_to_v210},
        {AV_PIX_FMT_Y210,   Y416, y210_to_y416},
        {AV_PIX_FMT_Y210,   Y216, memcpy_data},
#endif
        {AV_PIX_FMT_P010LE, v210, p010le_to_v210},
        {AV_PIX_FMT_P010LE, UYVY, p010le_to_uyvy},
        // 8-bit YUV
        {AV_PIX_FMT_YUV420P, v210, yuv420p_to_v210},
        {AV_PIX_FMT_YUV420P, UYVY, yuv420p_to_uyvy},
        {AV_PIX_FMT_YUV420P, RGB, yuv420p_to_rgb24},
        {AV_PIX_FMT_YUV420P, RGBA, yuv420p_to_rgb32},
        {AV_PIX_FMT_YUV422P, v210, yuv422p_to_v210},
        {AV_PIX_FMT_YUV422P, UYVY, yuv422p_to_uyvy},
        {AV_PIX_FMT_YUV422P, RGB, yuv422p_to_rgb24},
        {AV_PIX_FMT_YUV422P, RGBA, yuv422p_to_rgb32},
        {AV_PIX_FMT_YUV444P, v210, yuv444p_to_v210},
        {AV_PIX_FMT_YUV444P, UYVY, yuv444p_to_uyvy},
        {AV_PIX_FMT_YUV444P, RGB, yuv444p_to_rgb24},
        {AV_PIX_FMT_YUV444P, RGBA, yuv444p_to_rgb32},
        // 8-bit YUV - this should be supposedly full range JPEG but lavd decoder doesn't honor
        // GPUJPEG's SPIFF header indicating YUV BT.709 limited range. The YUVJ pixel formats
        // are detected only for GPUJPEG generated JPEGs.
        {AV_PIX_FMT_YUVJ420P, v210, yuv420p_to_v210},
        {AV_PIX_FMT_YUVJ420P, UYVY, yuv420p_to_uyvy},
        {AV_PIX_FMT_YUVJ420P, RGB, yuv420p_to_rgb24},
        {AV_PIX_FMT_YUVJ420P, RGBA, yuv420p_to_rgb32},
        {AV_PIX_FMT_YUVJ422P, v210, yuv422p_to_v210},
        {AV_PIX_FMT_YUVJ422P, UYVY, yuv422p_to_uyvy},
        {AV_PIX_FMT_YUVJ422P, RGB, yuv422p_to_rgb24},
        {AV_PIX_FMT_YUVJ422P, RGBA, yuv422p_to_rgb32},
        {AV_PIX_FMT_YUVJ444P, v210, yuv444p_to_v210},
        {AV_PIX_FMT_YUVJ444P, UYVY, yuv444p_to_uyvy},
        {AV_PIX_FMT_YUVJ444P, RGB, yuv444p_to_rgb24},
        {AV_PIX_FMT_YUVJ444P, RGBA, yuv444p_to_rgb32},
#if VUYX_PRESENT
        {AV_PIX_FMT_VUYA, UYVY, vuya_to_uyvy},
        {AV_PIX_FMT_VUYX, UYVY, vuya_to_uyvy},
        {AV_PIX_FMT_VUYA, Y416, vuya_to_y416},
        {AV_PIX_FMT_VUYX, Y416, vuyx_to_y416},
        {AV_PIX_FMT_YUV444P, VUYA, yuv444p_to_vuya},
        {AV_PIX_FMT_YUVJ444P, VUYA, yuv444p_to_vuya},
#endif
        // 8-bit YUV (NV12)
        {AV_PIX_FMT_NV12, UYVY, nv12_to_uyvy},
        {AV_PIX_FMT_NV12, RGB, nv12_to_rgb24},
        {AV_PIX_FMT_NV12, RGBA, nv12_to_rgb32},
        // 12-bit YUV
        {AV_PIX_FMT_YUV444P12LE, R10k, yuv444p12le_to_r10k},
        {AV_PIX_FMT_YUV444P12LE, R12L, yuv444p12le_to_r12l},
        {AV_PIX_FMT_YUV444P12LE, RG48, yuv444p12le_to_rg48},
        {AV_PIX_FMT_YUV444P12LE, UYVY, yuv444p12le_to_uyvy},
        {AV_PIX_FMT_YUV444P12LE, v210, yuv444p12le_to_v210},
        {AV_PIX_FMT_YUV444P12LE, Y416, yuv444p12le_to_y416},
        // 16-bit YUV
        {AV_PIX_FMT_YUV444P16LE, R10k, yuv444p16le_to_r10k},
        {AV_PIX_FMT_YUV444P16LE, R12L, yuv444p16le_to_r12l},
        {AV_PIX_FMT_YUV444P16LE, RG48, yuv444p16le_to_rg48},
        {AV_PIX_FMT_YUV444P16LE, UYVY, yuv444p16le_to_uyvy},
        {AV_PIX_FMT_YUV444P16LE, v210, yuv444p16le_to_v210},
        {AV_PIX_FMT_YUV444P16LE, Y416, yuv444p16le_to_y416},
        {AV_PIX_FMT_AYUV64, UYVY, ayuv64_to_uyvy},
        {AV_PIX_FMT_AYUV64, v210, ayuv64_to_v210},
        {AV_PIX_FMT_AYUV64, Y416, ayuv64_to_y416},
        // RGB
        {AV_PIX_FMT_GBRAP, RGB, gbrap_to_rgb},
        {AV_PIX_FMT_GBRAP, RGBA, gbrap_to_rgba},
        {AV_PIX_FMT_GBRP, RGB, gbrp_to_rgb},
        {AV_PIX_FMT_GBRP, RGBA, gbrp_to_rgba},
        {AV_PIX_FMT_RGB24, UYVY, rgb24_to_uyvy},
        {AV_PIX_FMT_RGB24, RGBA, rgb24_to_rgb32},
        {AV_PIX_FMT_GBRP10LE, R10k, gbrp10le_to_r10k},
        {AV_PIX_FMT_GBRP10LE, RGB, gbrp10le_to_rgb},
        {AV_PIX_FMT_GBRP10LE, RGBA, gbrp10le_to_rgba},
        {AV_PIX_FMT_GBRP10LE, RG48, gbrp10le_to_rg48},
        {AV_PIX_FMT_GBRP12LE, R12L, gbrp12le_to_r12l},
        {AV_PIX_FMT_GBRP12LE, R10k, gbrp12le_to_r10k},
        {AV_PIX_FMT_GBRP12LE, RGB, gbrp12le_to_rgb},
        {AV_PIX_FMT_GBRP12LE, RGBA, gbrp12le_to_rgba},
        {AV_PIX_FMT_GBRP12LE, RG48, gbrp12le_to_rg48},
        {AV_PIX_FMT_GBRP16LE, R12L, gbrp16le_to_r12l},
        {AV_PIX_FMT_GBRP16LE, R10k, gbrp16le_to_r10k},
        {AV_PIX_FMT_GBRP16LE, RG48, gbrp16le_to_rg48},
        {AV_PIX_FMT_GBRP12LE, RGB, gbrp16le_to_rgb},
        {AV_PIX_FMT_GBRP12LE, RGBA, gbrp16le_to_rgba},
        {AV_PIX_FMT_RGB48LE, R12L, rgb48le_to_r12l},
        {AV_PIX_FMT_RGB48LE, RGBA, rgb48le_to_rgba},
#ifdef HWACC_VDPAU
        // HW acceleration
        {AV_PIX_FMT_VDPAU, HW_VDPAU, av_vdpau_to_ug_vdpau},
#endif
        {AV_PIX_FMT_DRM_PRIME, DRM_PRIME, av_drm_prime_to_ug_drm_prime},
};
#define AV_TO_UV_CONVERSION_COUNT (sizeof av_to_uv_conversions / sizeof av_to_uv_conversions[0])
static const struct av_to_uv_conversion *av_to_uv_conversions_end = av_to_uv_conversions + AV_TO_UV_CONVERSION_COUNT;

static QSORT_S_COMP_DEFINE(compare_convs, a, b, orig_c) {
        const struct av_to_uv_conversion *conv_a = a;
        const struct av_to_uv_conversion *conv_b = b;
        const struct pixfmt_desc *src_desc = orig_c;
        struct pixfmt_desc desc_a = get_pixfmt_desc(conv_a->uv_codec);
        struct pixfmt_desc desc_b = get_pixfmt_desc(conv_b->uv_codec);

        int ret = compare_pixdesc(&desc_a, &desc_b, src_desc);
        return ret != 0 ? ret : (int) conv_a->uv_codec - (int) conv_b->uv_codec;
}

static QSORT_S_COMP_DEFINE(compare_codecs, a, b, orig_c)
{
        const codec_t            *codec_a  = a;
        const codec_t            *codec_b  = b;
        const struct pixfmt_desc *src_desc = orig_c;
        const struct pixfmt_desc  desc_a   = get_pixfmt_desc(*codec_a);
        const struct pixfmt_desc  desc_b   = get_pixfmt_desc(*codec_b);

        const int ret = compare_pixdesc(&desc_a, &desc_b, src_desc);
        return ret != 0 ? ret : (int) *codec_a - (int) *codec_b;
}

static decoder_t get_av_and_uv_conversion(enum AVPixelFormat av_codec, codec_t uv_codec,
                codec_t *intermediate_c, av_to_uv_convert_fp *av_convert) {
        struct av_to_uv_conversion convs[AV_TO_UV_CONVERSION_COUNT];
        size_t convs_count = 0;
        for (const struct av_to_uv_conversion *c = av_to_uv_conversions; c < av_to_uv_conversions_end; c++) {
                if (c->av_codec == av_codec) {
                        memcpy(convs + convs_count++, c, sizeof av_to_uv_conversions[0]);
                }
        }
        struct pixfmt_desc src_desc = av_pixfmt_get_desc(av_codec);
        qsort_s(convs, convs_count, sizeof convs[0], compare_convs, &src_desc);
        for (size_t i = 0; i < convs_count; ++i) {
                decoder_t dec = get_decoder_from_to(convs[i].uv_codec, uv_codec);
                if (dec != NULL) {
                        if (av_convert) {
                                *av_convert = convs[i].convert;
                        }
                        if (intermediate_c) {
                                *intermediate_c = convs[i].uv_codec;
                        }
                        return dec;
                }
        }
        return NULL;
}

void
av_to_uv_conversion_destroy(av_to_uv_convert_t **s)
{
        if (s == NULL || *s == NULL) {
                return;
        }
        av_to_uv_conversion_cuda_destroy(&(*s)->cuda_conv_state);
        free(*s);
        *s = NULL;
}

static bool
from_lavc_cuda_conv_enabled()
{
        if (!cuda_devices_explicit) {
                return false;
        }
        struct av_to_uv_convert_cuda *s =
            get_av_to_uv_cuda_conversion(AV_PIX_FMT_YUV422P, UYVY);
        if (s == NULL) {
                return false;
        }
        av_to_uv_conversion_cuda_destroy(&s);
        return true;
}

static enum AVPixelFormat
get_first_supported_cuda(const enum AVPixelFormat *fmts)
{
        for (; *fmts != AV_PIX_FMT_NONE; fmts++) {
                for (unsigned i = 0;
                     i < sizeof from_lavc_cuda_supp_formats /
                             sizeof from_lavc_cuda_supp_formats[0];
                     ++i) {
                        if (*fmts == from_lavc_cuda_supp_formats[i]) {
                                return *fmts;
                        }
                }
        }
        return AV_PIX_FMT_NONE;
}

static av_to_uv_convert_t *
get_av_to_uv_conversion_int(int av_codec, codec_t uv_codec)
{
        av_to_uv_convert_t *ret = calloc(1, sizeof *ret);
        ret->dst_pixfmt = uv_codec;

        if (from_lavc_cuda_conv_enabled()) {
                enum AVPixelFormat f[2] = { av_codec, AV_PIX_FMT_NONE };
                if (get_first_supported_cuda(f) != AV_PIX_FMT_NONE) {
                        ret->cuda_conv_state =
                            get_av_to_uv_cuda_conversion(av_codec, uv_codec);
                        if (ret->cuda_conv_state != NULL) {
                                MSG(NOTICE, "Using CUDA FFmpeg conversions.\n");
                                return ret;
                        }
                        MSG(ERROR, "Unable to initialize CUDA conv state!\n");
                }
        }

        codec_t mapped_pix_fmt = get_av_to_ug_pixfmt(av_codec);
        if (mapped_pix_fmt == uv_codec) {
                if (codec_is_planar(uv_codec)) {
                        log_msg(LOG_LEVEL_ERROR, "Planar pixfmts not support here, please report a bug!\n");
                        free(ret);
                        return NULL;
                }
                return ret;
        }
        if (mapped_pix_fmt != VC_NONE) {
                decoder_t dec = get_decoder_from_to(mapped_pix_fmt, uv_codec);
                if (dec) {
                        ret->dec = dec;
                        return ret;
                }
        }

        for (const struct av_to_uv_conversion *conversions = av_to_uv_conversions;
                        conversions < av_to_uv_conversions_end; conversions++) {
                if (conversions->av_codec == av_codec &&
                                conversions->uv_codec == uv_codec) {
                        ret->convert = conversions->convert;
                        watch_pixfmt_degrade(MOD_NAME, av_pixfmt_get_desc(av_codec), get_pixfmt_desc(uv_codec));
                        return ret;
                }
        }

        av_to_uv_convert_fp av_convert = NULL;
        codec_t intermediate = VC_NONE;
        decoder_t dec = get_av_and_uv_conversion(av_codec, uv_codec,
                &intermediate, &av_convert);
        if (!dec) {
                free(ret);
                return NULL;
        }
        ret->dec = dec;
        ret->convert = av_convert;
        ret->src_pixfmt = intermediate;
        watch_pixfmt_degrade(MOD_NAME, av_pixfmt_get_desc(av_codec), get_pixfmt_desc(intermediate));
        watch_pixfmt_degrade(MOD_NAME, get_pixfmt_desc(intermediate), get_pixfmt_desc(uv_codec));

        return ret;
}

av_to_uv_convert_t *
get_av_to_uv_conversion(int av_codec, codec_t uv_codec)
{
        av_to_uv_convert_t *ret =
            get_av_to_uv_conversion_int(av_codec, uv_codec);
        if (ret == NULL) {
                return NULL;
        }
        MSG(VERBOSE, "converting %s to %s over %s\n",
            av_get_pix_fmt_name(av_codec), get_codec_name(ret->dst_pixfmt),
            get_codec_name(ret->src_pixfmt));
        return ret;
}

/**
 * Returns first AVPixelFormat convertible to *ugc. If !*ugc, finds (probes)
 * best UltraGrid codec to which can be one of fmt converted and returns
 * AV_PIX_FMT_NONE.
 *
 * @param[in,out] ugc        if zero, probing the codec, if nonzero, only finding matching AVPixelFormat
 * @retval AV_PIX_FMT_NONE   if !*ugc
 * @retval !=AV_PIX_FMT_NONE if *ugc is non-zero
 */
static enum AVPixelFormat get_ug_codec_to_av(const enum AVPixelFormat *fmt, codec_t *ugc, bool use_hwaccel) {
        // directly mapped UG codecs
        for (const enum AVPixelFormat *fmt_it = fmt; *fmt_it != AV_PIX_FMT_NONE; fmt_it++) {
                //If hwaccel is not enabled skip hw accel pixfmts even if there are convert functions
                if (!use_hwaccel && (av_pix_fmt_desc_get(*fmt_it)->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                        continue;
                }

                codec_t mapped_pix_fmt = get_av_to_ug_pixfmt(*fmt_it);
                if (mapped_pix_fmt != VIDEO_CODEC_NONE) {
                        if (*ugc == VIDEO_CODEC_NONE) { // just probing internal format
                                *ugc = mapped_pix_fmt;
                                return AV_PIX_FMT_NONE;
                        }
                        if (*ugc == mapped_pix_fmt || get_decoder_from_to(mapped_pix_fmt, *ugc)) { // either mapped or convertible
                                return *fmt_it;
                        }
                }

                if (*ugc != VIDEO_CODEC_NONE) {
                        for (const struct av_to_uv_conversion *c = av_to_uv_conversions; c < av_to_uv_conversions_end; c++) {
                                if (c->av_codec == *fmt_it && *ugc == c->uv_codec) {
                                        return *fmt_it;
                                }
                        }

                        // AV+UV conversion needed
                        codec_t c;
                        if (get_av_and_uv_conversion(*fmt_it, *ugc, &c, NULL)) {
                                log_msg(LOG_LEVEL_VERBOSE, __FILE__ ": selected conversion from %s to %s with %s intermediate.\n",
                                                av_get_pix_fmt_name(*fmt_it), get_codec_name(*ugc), get_codec_name(c));
                                return *fmt_it;
                        }
                } else { // probe
                        struct av_to_uv_conversion usable_convs[sizeof av_to_uv_conversions / sizeof av_to_uv_conversions[0]];
                        int usable_convs_count = 0;
                        for (const struct av_to_uv_conversion *c = av_to_uv_conversions; c < av_to_uv_conversions_end; c++) {
                                if (c->av_codec == *fmt_it) {
                                        memcpy(usable_convs + usable_convs_count++, c, sizeof av_to_uv_conversions[0]);
                                }
                        }
                        if (usable_convs_count == 0) {
                                continue;
                        }
                        struct pixfmt_desc src_desc = av_pixfmt_get_desc(*fmt_it);
                        qsort_s(usable_convs, usable_convs_count, sizeof usable_convs[0], compare_convs, &src_desc);
                        *ugc = usable_convs[0].uv_codec;
                        return AV_PIX_FMT_NONE;
                }
        }
        return AV_PIX_FMT_NONE;
}

static codec_t
probe_cuda_ug_to_av(const enum AVPixelFormat *fmts)
{
        const enum AVPixelFormat sel_fmt = get_first_supported_cuda(fmts);
        if (sel_fmt == AV_PIX_FMT_NONE) {
                return VIDEO_CODEC_NONE;
        }

        codec_t ug_codecs[VC_COUNT];
        int     ugc_count = 0;
        for (codec_t c = VC_FIRST; c < VC_END; ++c) {
                if (!is_codec_opaque(c)) {
                        ug_codecs[ugc_count++] = c;
                }
        }

        struct pixfmt_desc src_desc = av_pixfmt_get_desc(sel_fmt);

        qsort_s(ug_codecs, ugc_count, sizeof ug_codecs[0], compare_codecs,
                &src_desc);
        return ug_codecs[0];
}

codec_t get_best_ug_codec_to_av(const enum AVPixelFormat *fmt, bool use_hwaccel) {
        if (from_lavc_cuda_conv_enabled() && probe_cuda_ug_to_av(fmt) != VC_NONE) {
                return probe_cuda_ug_to_av(fmt);
        }
        codec_t c = VIDEO_CODEC_NONE;
        get_ug_codec_to_av(fmt, &c, use_hwaccel);
        return c;
}

enum AVPixelFormat lavd_get_av_to_ug_codec(const enum AVPixelFormat *fmt, codec_t c, bool use_hwaccel) {
        if (from_lavc_cuda_conv_enabled() &&
            get_first_supported_cuda(fmt) != AV_PIX_FMT_NONE) {
                return get_first_supported_cuda(fmt);
        }
        return get_ug_codec_to_av(fmt, &c, use_hwaccel);
}

enum AVPixelFormat
pick_av_convertible_to_ug(codec_t color_spec, av_to_uv_convert_t **av_conv)
{
        *av_conv = calloc(1, sizeof **av_conv);
        (*av_conv)->dst_pixfmt = color_spec;

        if (get_ug_to_av_pixfmt(color_spec) != AV_PIX_FMT_NONE) {
                return get_ug_to_av_pixfmt(color_spec);
        }

        struct pixfmt_desc out_desc = get_pixfmt_desc(color_spec);
        decoder_t dec;
        if (out_desc.rgb) {
                if (out_desc.depth > 8 && (dec = get_decoder_from_to(RG48, color_spec))) {
                        (*av_conv)->dec = dec;
                        return AV_PIX_FMT_RGB48LE;
                } else if ((dec = get_decoder_from_to(RGB, color_spec))) {
                        (*av_conv)->dec = dec;
                        return AV_PIX_FMT_RGB24;
                }
        }
#if XV3X_PRESENT
        if (out_desc.depth > 8 && (dec = get_decoder_from_to(Y416, color_spec))) {
                (*av_conv)->dec = dec;
                return AV_PIX_FMT_XV36;
        }
#endif
        if ((dec = get_decoder_from_to(UYVY, color_spec))) {
                (*av_conv)->dec = dec;
                return AV_PIX_FMT_UYVY422;
        }
        for (const struct av_to_uv_conversion *c = av_to_uv_conversions; c < av_to_uv_conversions_end; c++) {
                if (c->uv_codec == color_spec) { // pick any (first usable)
                        memset(*av_conv, 0, sizeof **av_conv);
                        (*av_conv)->convert = c->convert;
                        return c->av_codec;
                }
        }
        free(*av_conv);
        *av_conv = NULL;
        return AV_PIX_FMT_NONE;
}

static void
do_av_to_uv_convert(const av_to_uv_convert_t *s, char *__restrict dst_buffer,
                    AVFrame *__restrict inf, int                     pitch,
                    const int *__restrict rgb_shift)
{
        int lmt_rng = 1; // 0 if src YCbCr is full-range (aka JPEG), 1 otherwise
        const codec_t av_to_uv_pf = s->src_pixfmt != VC_NONE
                                        ? s->src_pixfmt
                                        : s->dst_pixfmt;
        const enum colorspace cs_coeffs = get_cs_for_conv(
            inf, av_to_uv_pf, &lmt_rng);

        unsigned char *dec_input = inf->data[0];
        size_t src_linesize = inf->linesize[0];
        unsigned char *tmp = NULL;
        if (s->convert) {
                DEBUG_TIMER_START(lavd_av_to_uv);
                if (!s->dec) {
                        s->convert((struct av_conv_data){
                            dst_buffer,
                            inf,
                            pitch,
                            { rgb_shift[0], rgb_shift[1], rgb_shift[2] },
                            cs_coeffs,
                            lmt_rng,
                        });
                        DEBUG_TIMER_STOP(lavd_av_to_uv);
                        return;
                }
                src_linesize = vc_get_linesize(inf->width, s->src_pixfmt);
                dec_input = tmp = malloc(
                    vc_get_datalen(inf->width, inf->height, s->src_pixfmt) +
                    MAX_PADDING);
                s->convert((struct av_conv_data){
                    (char *) dec_input,
                    inf,
                    src_linesize,
                    DEFAULT_RGB_SHIFT_INIT,
                    cs_coeffs,
                    lmt_rng,
                });
                DEBUG_TIMER_STOP(lavd_av_to_uv);
        }
        if (s->dec) {
                DEBUG_TIMER_START(lavd_dec);
                int dst_size = vc_get_size(inf->width, s->dst_pixfmt);
                for (ptrdiff_t i = 0; i < inf->height; ++i) {
                        s->dec((unsigned char *) dst_buffer + i * pitch,
                               dec_input + i * src_linesize, dst_size,
                               rgb_shift[0], rgb_shift[1], rgb_shift[2]);
                }
                DEBUG_TIMER_STOP(lavd_dec);
                free(tmp);
                return;
        }

        // memcpy only
        int linesize = vc_get_linesize(inf->width, s->dst_pixfmt);
        for (ptrdiff_t i = 0; i < inf->height; ++i) {
                memcpy(dst_buffer + i * pitch, inf->data[0] + i * inf->linesize[0], linesize);
        }
}

struct convert_task_data {
        const av_to_uv_convert_t *convert;
        unsigned char            *out_data;
        AVFrame                  *in_frame;
        int                       pitch;
        const int                *rgb_shift;
};

static void *
convert_task(void *arg)
{
        struct convert_task_data *d = arg;
        do_av_to_uv_convert(d->convert, (char *) d->out_data, d->in_frame,
                            d->pitch, d->rgb_shift);
        return NULL;
}

/**
 * @return check color space/range if we can correctly convert
 */
static void
check_constraints(AVFrame *f, bool dst_rgb)
{
        const struct AVPixFmtDescriptor *avd = av_pix_fmt_desc_get(f->format);
        const bool src_rgb = (avd->flags & AV_PIX_FMT_FLAG_RGB) != 0;
        if (f->color_range == AVCOL_RANGE_JPEG && !src_rgb && !dst_rgb) {
                MSG_ONCE(WARNING, "Full-range YCbCr may be clipped!\n");
        }
}

/**
 * @param av_to_uv_pf  intermediate PF (after av_to_uv) because eventual
 * uv_to_uv conv do not implement other CS coeffs than BT.709
 * @note ensure that used AVFrame properties copied by av_to_uv_convert
 */
static enum colorspace
get_cs_for_conv(AVFrame *f, codec_t av_to_uv_pf, int *lmt_rng)
{
        const struct AVPixFmtDescriptor *avd = av_pix_fmt_desc_get(f->format);
        const bool src_rgb = (avd->flags & AV_PIX_FMT_FLAG_RGB) != 0;
        const bool dst_rgb = codec_is_a_rgb(av_to_uv_pf);
        const bool src_601 = f->colorspace == AVCOL_SPC_BT470BG ||
                             f->colorspace == AVCOL_SPC_SMPTE170M ||
                             f->colorspace == AVCOL_SPC_SMPTE240M;
        *lmt_rng = !src_rgb && f->color_range == AVCOL_RANGE_JPEG ? 0 : 1;
        if (src_rgb) {
                return CS_DFL; // either no CS conv or to default YUV
        }
        // from now src is YUV
        if (!dst_rgb) { // dst is YUV -> no CS conv!
                if (f->colorspace != AVCOL_SPC_RGB &&
                    f->colorspace != AVCOL_SPC_BT709 &&
                    f->colorspace != AVCOL_SPC_UNSPECIFIED && !src_601) {
                        MSG(WARNING,
                            "Input color space %s is not supported by "
                            "UltraGrid!\n",
                            av_color_space_name(f->colorspace));
                }
                const bool have_pp = tok_in_argv(uv_argv, "y601_to_y709");
                if (src_601 && get_default_cs() != CS_601 && !have_pp) {
                        MSG_ONCE(WARNING,
                            "Got %s CS but not converted - consider \"--param "
                            "color-601\" as a hint for supported displays or "
                            "\"-p matrix2:y601_to_y709\"\n",
                            av_color_space_name(f->colorspace));
                }
                return CS_DFL; // doesn't matter - won't be used anyways
        }
        if (src_601) {
                return CS_601;
        }
        if (f->colorspace == AVCOL_SPC_BT709) {
                return CS_709;
        }
        MSG(WARNING,
            "Suspicious (unexpected) color space %s, using default "
            "coeffs. Please report.!\n",
            av_color_space_name(f->colorspace));
        return CS_DFL;
}

void
av_to_uv_convert(const av_to_uv_convert_t *convert,
                 char *dst, AVFrame *in, int pitch,
                 const int rgb_shift[3])
{
        check_constraints(in, codec_is_a_rgb(convert->dst_pixfmt));
        if (convert->cuda_conv_state != NULL) {
                av_to_uv_convert_cuda(convert->cuda_conv_state, dst, in,
                                      in->width, in->height, pitch, rgb_shift);
                return;
        }

        if (codec_is_const_size(convert->dst_pixfmt)) { // VAAPI etc
                do_av_to_uv_convert(convert, dst, in, pitch, rgb_shift);
                return;
        }

        const int cpu_count = get_cpu_core_count();

        struct convert_task_data d[cpu_count];
        AVFrame                  parts[cpu_count];
        for (int i = 0; i < cpu_count; ++i) {
                int row_height = (in->height / cpu_count) & ~1; // needs to be even
                unsigned char *part_dst =
                    (unsigned char *) dst + (size_t) i * row_height * pitch;

                // copy used props - av_frame_copy_props() can be used as well
                // *but* AVFrame must have been alloced by av_frame_alloc()
                // (but unsure if there isn't higher overhead of the calls)
                memcpy(parts[i].linesize, in->linesize, sizeof in->linesize);
                parts[i].colorspace  = in->colorspace;
                parts[i].color_range = in->color_range;
                parts[i].format      = in->format;

                const AVPixFmtDescriptor *fmt_desc =
                    av_pix_fmt_desc_get(in->format);
                for (int plane = 0; plane < AV_NUM_DATA_POINTERS; ++plane) {
                        if (in->data[plane] == NULL) {
                                break;
                        }
                        parts[i].data[plane] =
                            in->data[plane] +
                            ((i * row_height * in->linesize[plane]) >>
                             (plane == 0 ? 0 : fmt_desc->log2_chroma_h));
                }
                if (i == cpu_count - 1) {
                        row_height = in->height - row_height * (cpu_count - 1);
                }
                parts[i].width = in->width;
                parts[i].height = row_height;
                d[i] =
                    (struct convert_task_data){ convert,  part_dst,   &parts[i],
                                                pitch, rgb_shift };
        }
        task_run_parallel(convert_task, cpu_count, d, sizeof d[0], NULL);
}

int
from_lavc_pf_priority(struct pixfmt_desc internal, codec_t ugc)
{
        bool found_a_conversion = false;
        for (unsigned i = 0; i < ARR_COUNT(av_to_uv_conversions); i++) {
                if (av_to_uv_conversions[i].uv_codec == ugc) {
                        found_a_conversion = true;
                        break;
                }
        }
        if (!found_a_conversion) {
                return VDEC_PRIO_NA;
        }
        if (internal.depth == 0) { // unspecified internal format
                return VDEC_PRIO_LOW;
        }
        for (unsigned i = 0; i < ARR_COUNT(av_to_uv_conversions); i++) {
                if (av_to_uv_conversions[i].uv_codec != ugc) {
                        continue;
                }
                if (pixdesc_equals(
                        av_pixfmt_get_desc(av_to_uv_conversions[i].av_codec),
                        internal)) {
                        // conv from AV PF with same props as internal
                        return VDEC_PRIO_NORMAL;
                }
        }
        // the conversion may be not direct but over intermediate UG codec_t
        return VDEC_PRIO_NOT_PREFERRED;
}

#pragma GCC diagnostic pop

/* vi: set expandtab sw=8: */
