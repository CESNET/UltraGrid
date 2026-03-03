/**
 * @file   from_planar.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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


#include "from_planar.h"

#include <assert.h>        // for assert
#ifdef __SSE3__
#include <emmintrin.h>     // for __m128i, _mm_or_si128, _mm_unpacklo_epi8
#include <pmmintrin.h>
#endif
#include <stdint.h>        // for uint32_t, uintptr_t, uint16_t, uint64_t
#include <string.h>        // for memcpy

#include "compat/c23.h"    // for size_t, NULL, countof, nullptr, ptrdiff_t
#include "types.h"         // for depth
#include "utils/macros.h"  // for ALWAYS_INLINE, OPTIMIZED_FOR
#include "utils/misc.h"    // for get_cpu_core_count
#include "utils/worker.h"  // for task_run_parallel

// shortcuts
#define R R_SHIFT_IDX
#define G G_SHIFT_IDX
#define B B_SHIFT_IDX

ALWAYS_INLINE static inline  void
gbrpXXle_to_r12l(struct from_planar_data d, const int in_depth, int rind, int gind, int bind)
{
        assert((uintptr_t) d.in_linesize[0] % 2 == 0);
        assert((uintptr_t) d.in_linesize[1] % 2 == 0);
        assert((uintptr_t) d.in_linesize[2] % 2 == 0);

#define S(x) ((x) >> (in_depth - 12))
        // clang-format off
        for (size_t y = 0; y < (size_t) d.height; ++y) {
                const uint16_t *src_r = (const void *) (d.in_data[rind] + (d.in_linesize[rind] * y));
                const uint16_t *src_g = (const void *) (d.in_data[gind] + (d.in_linesize[gind] * y));
                const uint16_t *src_b = (const void *) (d.in_data[bind] + (d.in_linesize[bind] * y));
                unsigned char *dst =
                    (unsigned char *) d.out_data + (y * d.out_pitch);

                for (int x = 0; x < d.width; x += 8) {
                        uint16_t tmpbuf[3][8];
                        if (x + 8 >= d.width) {
                                size_t remains = sizeof(uint16_t) * (d.width - x);
                                memcpy(tmpbuf[0], src_r, remains);
                                memcpy(tmpbuf[1], src_g, remains);
                                memcpy(tmpbuf[2], src_b, remains);
                                src_r = tmpbuf[0];
                                src_g = tmpbuf[1];
                                src_b = tmpbuf[2];
                        }
                        dst[0] = S(*src_r) & 0xff;
                        dst[1] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[2] = S(*src_g++) >> 4;
                        dst[3] = S(*src_b) & 0xff;
                        dst[4 + 0] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[4 + 1] = S(*src_r++) >> 4;
                        dst[4 + 2] = S(*src_g) & 0xff;
                        dst[4 + 3] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[8 + 0] = S(*src_b++) >> 4;
                        dst[8 + 1] = S(*src_r) & 0xff;
                        dst[8 + 2] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[8 + 3] = S(*src_g++) >> 4;
                        dst[12 + 0] = S(*src_b) & 0xff;
                        dst[12 + 1] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[12 + 2] = S(*src_r++) >> 4;
                        dst[12 + 3] = S(*src_g) & 0xff;
                        dst[16 + 0] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[16 + 1] = S(*src_b++) >> 4;
                        dst[16 + 2] = S(*src_r) & 0xff;
                        dst[16 + 3] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[20 + 0] = S(*src_g++) >> 4;
                        dst[20 + 1] = S(*src_b) & 0xff;
                        dst[20 + 2] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[20 + 3] = S(*src_r++) >> 4;;
                        dst[24 + 0] = S(*src_g) & 0xff;
                        dst[24 + 1] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[24 + 2] = S(*src_b++) >> 4;
                        dst[24 + 3] = S(*src_r) & 0xff;
                        dst[28 + 0] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[28 + 1] = S(*src_g++) >> 4;
                        dst[28 + 2] = S(*src_b) & 0xff;
                        dst[28 + 3] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[32 + 0] = S(*src_r++) >> 4;
                        dst[32 + 1] = S(*src_g) & 0xff;
                        dst[32 + 2] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[32 + 3] = S(*src_b++) >> 4;
                        dst += 36;
                }
        }
        // clang-format on
#undef S
}

/**
 * test with:
 * @code{.sh}
 * uv -t testcard:c=R12L -c lavc:e=libx265 -p change_pixfmt:RGBA -d gl
 * uv -t testcard:s=511x512c=R12L -c lavc:e=libx265 -p change_pixfmt:RGBA -d gl # irregular sz
 * # optionally also `--param decoder-use-codec=R12L` to ensure decoded codec
 * @endcode
 */
void
gbrp12le_to_r12l(struct from_planar_data d)
{
        gbrpXXle_to_r12l(d, DEPTH12, 2, 0, 1);
}

void
gbrp16le_to_r12l(struct from_planar_data d)
{
        gbrpXXle_to_r12l(d, DEPTH16, 2, 0, 1);
}

void
rgbpXXle_to_r12l(struct from_planar_data d)
{
        gbrpXXle_to_r12l(d, d.in_depth, 0, 1, 2);
}

static void
rgbpXXle_to_rg48_int(struct from_planar_data d, const int in_depth, int rind, int gind, int bind)
{
        assert((uintptr_t) d.out_data % 2 == 0);
        assert((uintptr_t) d.in_data[0] % 2 == 0);
        assert((uintptr_t) d.in_data[1] % 2 == 0);
        assert((uintptr_t) d.in_data[2] % 2 == 0);

        for (ptrdiff_t y = 0; y < d.height; ++y) {
                const uint16_t *src_r = (const void *) (d.in_data[rind] + (d.in_linesize[rind] * y));
                const uint16_t *src_g = (const void *) (d.in_data[gind] + (d.in_linesize[gind] * y));
                const uint16_t *src_b = (const void *) (d.in_data[bind] + (d.in_linesize[bind] * y));
                uint16_t *dst = (void *) (d.out_data + (y * d.out_pitch));

                for (int x = 0; x < d.width; ++x) {
                        *dst++ = *src_r++ << (16U - in_depth);
                        *dst++ = *src_g++ << (16U - in_depth);
                        *dst++ = *src_b++ << (16U - in_depth);
                }
        }
}

void
gbrp10le_to_rg48(struct from_planar_data d)
{
        rgbpXXle_to_rg48_int(d, DEPTH10, 2, 0, 1);
}

void
gbrp12le_to_rg48(struct from_planar_data d)
{
        rgbpXXle_to_rg48_int(d, DEPTH12, 2, 0, 1);
}

void
gbrp16le_to_rg48(struct from_planar_data d)
{
        rgbpXXle_to_rg48_int(d, DEPTH16, 2, 0, 1);
}

void
rgbpXXle_to_rg48(struct from_planar_data d)
{
        rgbpXXle_to_rg48_int(d, d.in_depth, 0, 1, 2);
}

ALWAYS_INLINE static inline void
gbrpXXle_to_r10k(struct from_planar_data d, const unsigned int in_depth,
                 int rind, int gind, int bind)
{
        // __builtin_trap();
        assert((uintptr_t) d.in_linesize[0] % 2 == 0);
        assert((uintptr_t) d.in_linesize[1] % 2 == 0);
        assert((uintptr_t) d.in_linesize[2] % 2 == 0);

        for (size_t y = 0; y < (size_t) d.height; ++y) {
                const uint16_t *src_r = (const void *) (d.in_data[rind] + (d.in_linesize[rind] * y));
                const uint16_t *src_g = (const void *) (d.in_data[gind] + (d.in_linesize[gind] * y));
                const uint16_t *src_b = (const void *) (d.in_data[bind] + (d.in_linesize[bind] * y));
                unsigned char *dst = d.out_data + (y * d.out_pitch);

                const unsigned width = d.width;
                OPTIMIZED_FOR (unsigned x = 0; x < width; ++x) {
                        *dst++ = *src_r >> (in_depth - 8U);
                        *dst++ = ((*src_r++ >> (in_depth - 10U)) & 0x3U) << 6U | *src_g >> (in_depth - 6U);
                        *dst++ = ((*src_g++ >> (in_depth - 10U)) & 0xFU) << 4U | *src_b >> (in_depth - 4U);
                        *dst++ = ((*src_b++ >> (in_depth - 10U)) & 0x3FU) << 2U | 0x3U;
                }
        }
}

void
gbrp10le_to_r10k(struct from_planar_data d)
{
        gbrpXXle_to_r10k(d, DEPTH10, 2, 0, 1);
}

void
gbrp12le_to_r10k(struct from_planar_data d)
{
        gbrpXXle_to_r10k(d, DEPTH12, 2, 0, 1);
}

void
gbrp16le_to_r10k(struct from_planar_data d)
{
        gbrpXXle_to_r10k(d, DEPTH16, 2, 0, 1);
}

void
rgbpXXle_to_r10k(struct from_planar_data d)
{
        gbrpXXle_to_r10k(d,d.in_depth, 0, 1, 2);
}

struct convert_task_data {
        decode_planar_func_t *convert;
        struct from_planar_data d;
};

static void *
convert_task(void *arg)
{
        struct convert_task_data *d = arg;
        d->convert(d->d);
        return nullptr;
}

// destiled from av_to_uv_convert
void
decode_planar_parallel(decode_planar_func_t         *dec,
                       const struct from_planar_data d, int num_threads)
{
        const unsigned cpu_count =
            num_threads == 0 ? get_cpu_core_count() : num_threads;

        struct convert_task_data data[cpu_count];
        for (size_t i = 0; i < cpu_count; ++i) {
                unsigned row_height = (d.height / cpu_count) & ~1; // needs to be even
                data[i].convert     = dec;
                data[i].d           = d;
                data[i].d.out_data  = d.out_data + (i * row_height * d.out_pitch);

                for (unsigned plane = 0; plane < countof(d.in_data); ++plane) {
                        data[i].d.in_data[plane] =
                            d.in_data[plane] +
                            ((i * row_height * d.in_linesize[plane]) >>
                             (plane == 0 ? 0 : d.log2_chroma_h));
                }
                if (i == cpu_count - 1) {
                        row_height = d.height - (row_height * (cpu_count - 1));
                }
                data[i].d.height = row_height;
        }
        task_run_parallel(convert_task, (int) cpu_count, data, sizeof data[0], NULL);
}

void
yuv422p10le_to_v210(const struct from_planar_data d)
{
        const int width = d.width;
        const int height = d.height;

        for(int y = 0; y < height; ++y) {
                const uint16_t *src_y =  (const void *)(d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_cb = (const void *)(d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_cr = (const void *)(d.in_data[2] + d.in_linesize[2] * y);
                uint32_t *dst =
                    (void *) (d.out_data + y * d.out_pitch);

                for (int x = 0; x < width / 6; ++x) {
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

static void
gbrap_to_rgb_rgba(const struct from_planar_data d, int rind, int gind, int bind, int aind)
{
        const int width = d.width;
        const int height = d.height;
        const int out_comp_count = aind == -1 ? 3 : 4;

        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                        uint8_t *buf = d.out_data + (y * d.out_pitch) + (x * out_comp_count);
                        int src_idx = (y * d.in_linesize[0]) + x;
                        buf[0] = d.in_data[rind][src_idx]; // R
                        buf[1] = d.in_data[gind][src_idx]; // G
                        buf[2] = d.in_data[bind][src_idx]; // B
                        if (out_comp_count == 4) {
                                buf[3] = d.in_data[aind][src_idx]; // A
                        }
                }
        }
}

void
gbrap_to_rgba(const struct from_planar_data d)
{
        gbrap_to_rgb_rgba(d, 2, 0, 1, 3);
}

void
gbrap_to_rgb(const struct from_planar_data d)
{
        gbrap_to_rgb_rgba(d, 2, 0, 1, -1);
}

void
yuv420_to_i420(const struct from_planar_data d)
{
        assert(d.width % 2 == 0);
        assert(d.height % 2 == 0);
        // d.out_pitch ignored
        const size_t dst_y_linesize = d.width;
        const size_t dst_uv_linesize = d.width  / 2 ;
        unsigned char *dst_y = d.out_data;
        unsigned char *dst_u = dst_y + ((size_t) dst_y_linesize * d.height);
        unsigned char *dst_v = dst_u + ((size_t) dst_uv_linesize * (d.height / 2));
        for (size_t y = 0; y < (size_t) d.height / 2; y += 1) {
                memcpy(dst_y, d.in_data[0] + (2 * y * d.in_linesize[0]), dst_y_linesize);
                dst_y += dst_y_linesize;
                memcpy(dst_y, d.in_data[0] + ((2 * y + 1) * d.in_linesize[0]), dst_y_linesize);
                dst_y += dst_y_linesize;

                memcpy(dst_u, d.in_data[1] + (y * d.in_linesize[1]), dst_uv_linesize);
                dst_u += dst_uv_linesize;
                memcpy(dst_v, d.in_data[2] + (y * d.in_linesize[2]), dst_uv_linesize);
                dst_v += dst_uv_linesize;
        }
}

static void
yuv422p_to_uyvy_yuyv(const struct from_planar_data d, bool yuyv)
{
        (void) yuyv;
        for (size_t y = 0; y < (size_t) d.height; ++y) {
                const unsigned char *src_y =  d.in_data[0] + (d.in_linesize[0] * y);
                const unsigned char *src_cb = d.in_data[1] + (d.in_linesize[1] * y);
                const unsigned char *src_cr = d.in_data[2] + (d.in_linesize[2] * y);
                unsigned char *dst = d.out_data + (d.out_pitch * y);

                const unsigned width = d.width;
                OPTIMIZED_FOR (unsigned x = 0; x < width / 2; ++x) {
                        if (yuyv) {
                                *dst++ = *src_y++;
                                *dst++ = *src_cb++;
                                *dst++ = *src_y++;
                                *dst++ = *src_cr++;
                        } else {
                                *dst++ = *src_cb++;
                                *dst++ = *src_y++;
                                *dst++ = *src_cr++;
                                *dst++ = *src_y++;
                        }
                }
        }
}

void
yuv422p_to_uyvy(const struct from_planar_data d)
{
        yuv422p_to_uyvy_yuyv(d, false);
}

static void yuv422pXXle_to_uyvy_int(const struct from_planar_data d, int in_depth)
{
        for(int y = 0; y < d.height; ++y) {
                const uint16_t *src_y =  (const void *)(d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_cb = (const void *)(d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_cr = (const void *)(d.in_data[2] + d.in_linesize[2] * y);
                uint8_t *dst =
                    (uint8_t *) (void *) (d.out_data + y * d.out_pitch);

                for(int x = 0; x < d.width / 2; ++x) {
                        *dst++ = *src_cb++ >> (in_depth - 8);
                        *dst++ = *src_y++  >> (in_depth - 8);
                        *dst++ = *src_cr++ >> (in_depth - 8);
                        *dst++ = *src_y++  >> (in_depth - 8);
                }
        }
}

void yuv422p10le_to_uyvy(const struct from_planar_data d)
{
        yuv422pXXle_to_uyvy_int(d, DEPTH10);
}

void
yuv422pXX_to_uyvy(const struct from_planar_data d)
{
        if (d.in_depth == DEPTH8) {
                yuv422p_to_uyvy_yuyv(d, false);
        } else {
                yuv422pXXle_to_uyvy_int(d, d.in_depth);
        }
}


void
yuv422p_to_yuyv(const struct from_planar_data d)
{
        yuv422p_to_uyvy_yuyv(d, true);
}

static void
gbrpXXle_to_rgb(const struct from_planar_data d, unsigned int in_depth, int rind, int gind, int bind)
{
        assert((uintptr_t) d.in_linesize[0] % 2 == 0);
        assert((uintptr_t) d.in_linesize[1] % 2 == 0);
        assert((uintptr_t) d.in_linesize[2] % 2 == 0);

        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_r = (const void *) (d.in_data[rind] + ((d.in_linesize[rind] * y)));
                const uint16_t *src_g = (const void *) (d.in_data[gind] + ((d.in_linesize[gind] * y)));
                const uint16_t *src_b = (const void *) (d.in_data[bind] + ((d.in_linesize[bind] * y)));
                unsigned char *dst =
                    (unsigned char *) d.out_data + (y * d.out_pitch);

                const unsigned width = d.width;
                OPTIMIZED_FOR (unsigned x = 0; x < width; ++x) {
                        *dst++ = *src_r++ >> (in_depth - 8U);
                        *dst++ = *src_g++ >> (in_depth - 8U);
                        *dst++ = *src_b++ >> (in_depth - 8U);
                }
        }
}

static void
gbrpXXle_to_rgba(const struct from_planar_data d, unsigned int in_depth)
{
        assert((uintptr_t) d.out_data  % 4 == 0);
        assert((uintptr_t) d.in_data[0] % 2 == 0);
        assert((uintptr_t) d.in_data[1] % 2 == 0);
        assert((uintptr_t) d.in_data[2] % 2 == 0);

        enum { R = 0, G = 1, B = 2 };

        const uint32_t alpha_mask = 0xFFFFFFFFU ^ (0xFFU << d.rgb_shift[R]) ^
                                    (0xFFU << d.rgb_shift[G]) ^
                                    (0xFFU << d.rgb_shift[B]);

        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_g = (const void *) (d.in_data[0] + (d.in_linesize[0] * y));
                const uint16_t *src_b = (const void *) (d.in_data[1] + (d.in_linesize[1] * y));
                const uint16_t *src_r = (const void *) (d.in_data[2] + (d.in_linesize[2] * y));
                uint32_t *dst = (void *) (d.out_data+ (y * d.out_pitch));

                const int width = d.width;
                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ =
                            alpha_mask |
                            (*src_r++ >> (in_depth - 8U)) << d.rgb_shift[0] |
                            (*src_g++ >> (in_depth - 8U)) << d.rgb_shift[1] |
                            (*src_b++ >> (in_depth - 8U)) << d.rgb_shift[2];
                }
        }
}

void
gbrp10le_to_rgb(const struct from_planar_data d)
{
        gbrpXXle_to_rgb(d, DEPTH10, 2, 0, 1);
}

void
gbrp10le_to_rgba(const struct from_planar_data d)
{
        gbrpXXle_to_rgba(d, DEPTH10);
}

void
gbrp12le_to_rgb(const struct from_planar_data d)
{
        gbrpXXle_to_rgb(d, DEPTH12, 2, 0, 1);
}

void
gbrp12le_to_rgba(const struct from_planar_data d)
{
        gbrpXXle_to_rgba(d, DEPTH12);
}

void
gbrp16le_to_rgb(const struct from_planar_data d)
{
        gbrpXXle_to_rgb(d, DEPTH16, 2, 0, 1);
}

void
gbrp16le_to_rgba(const struct from_planar_data d)
{
        gbrpXXle_to_rgba(d, DEPTH16);
}

void
rgbpXX_to_rgb(const struct from_planar_data d)
{
        if (d.in_depth == DEPTH8) {
                gbrap_to_rgb_rgba(d, 0, 1, 2, -1);
        } else {
                gbrpXXle_to_rgb(d, d.in_depth, 0, 1, 2);
        }
}

void
yuv420p_to_uyvy(const struct from_planar_data d)
{
        const unsigned width = d.width;
        for(int y = 0; y < (d.height + 1) / 2; ++y) {
                int  scnd_row = y * 2 + 1;
                if (scnd_row == d.height) {
                        scnd_row = d.height - 1;
                }
                const char *src_y1 = (const char *) d.in_data[0] + d.in_linesize[0] * y * 2;
                const char *src_y2 = (const char *) d.in_data[0] + d.in_linesize[0] * scnd_row;
                const char *src_cb = (const char *) d.in_data[1] + d.in_linesize[1] * y;
                const char *src_cr = (const char *) d.in_data[2] + d.in_linesize[2] * y;
                char *dst1 = (char *) d.out_data + (y * 2) * d.out_pitch;
                char *dst2 = (char *) d.out_data + scnd_row * d.out_pitch;

                unsigned x = 0;

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


                OPTIMIZED_FOR (; x < width - 15; x += 16){
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

