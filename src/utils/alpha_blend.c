/**
 * @file   utils/alpha_blend.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Alpha blending of 16-bit RGBA overlay onto native video formats
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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
#endif

#include <string.h>           // for memcpy

#include "color_space.h"      // for get_color_coeffs, RGB_TO_*, COMP_BASE, LIMIT_*
#include "types.h"            // for DEPTH8, DEPTH10, DEPTH16
#include "utils/alpha_blend.h"
#include "utils/macros.h"     // for CLAMP

/* For 16-bit RGB input through DEPTH-N coefficients, the right-shift accounts
 * for the difference between input bit depth (16) and output bit depth (N).
 * Pattern matches vc_copylineRG48toV210 in pixfmt_conv.c. */
#define COMP_OFF_8  (COMP_BASE + 8)
#define COMP_OFF_10 (COMP_BASE + 6)
#define COMP_OFF_16 (COMP_BASE + 0)

/* Limited-range YCbCr offsets: Y zero is at 16, CbCr zero is at 128 (8-bit).
 * For other depths use (1 << (depth - 4)) and (1 << (depth - 1)) respectively. */
#define Y_OFFSET_8     (1 << (DEPTH8 - 4))
#define CBCR_OFFSET_8  (1 << (DEPTH8 - 1))
#define Y_OFFSET_10    (1 << (DEPTH10 - 4))
#define CBCR_OFFSET_10 (1 << (DEPTH10 - 1))
#define Y_OFFSET_16    (1 << (DEPTH16 - 4))
#define CBCR_OFFSET_16 (1 << (DEPTH16 - 1))

/* Unsigned types and a literal divisor let the compiler emit magic-multiply
 * for the divide instead of idiv. The dividend max is bounded (max*max fits
 * in 32 bits at every used depth) so unsigned is always safe. */
#define BLEND_N(src, dst, a, max) \
        (((src) * (a) + (dst) * ((max) - (a))) / (max))

static inline uint8_t blend8(unsigned src, unsigned dst, unsigned a)
{
        return BLEND_N(src, dst, a, 255u);
}

static inline uint16_t blend10(unsigned src, unsigned dst, unsigned a)
{
        return BLEND_N(src, dst, a, 1023u);
}

/* 16-bit blend needs 64-bit accumulation: 65535 * 65535 already overflows
 * uint32_t and the BLEND_N expansion adds two such products. */
static inline uint16_t blend16(unsigned src, unsigned dst, unsigned a)
{
        return ((uint64_t)src * a + (uint64_t)dst * (65535u - a)) / 65535u;
}

void alpha_blend_rgba(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width)
{
        for (int x = 0; x < width; x++) {
                unsigned r = rgba16[0] >> 8;
                unsigned g = rgba16[1] >> 8;
                unsigned b = rgba16[2] >> 8;
                unsigned a = rgba16[3] >> 8;

                dst[0] = blend8(r, dst[0], a);
                dst[1] = blend8(g, dst[1], a);
                dst[2] = blend8(b, dst[2], a);
                /* Porter-Duff "over": alpha_out = alpha_src + alpha_dst*(1-alpha_src) */
                dst[3] = a + (dst[3] * (255u - a)) / 255u;

                rgba16 += 4;
                dst += 4;
        }
}

void alpha_blend_rgb(uint8_t * __restrict dst,
                     const uint16_t * __restrict rgba16, int width)
{
        for (int x = 0; x < width; x++) {
                unsigned r = rgba16[0] >> 8;
                unsigned g = rgba16[1] >> 8;
                unsigned b = rgba16[2] >> 8;
                unsigned a = rgba16[3] >> 8;

                dst[0] = blend8(r, dst[0], a);
                dst[1] = blend8(g, dst[1], a);
                dst[2] = blend8(b, dst[2], a);

                rgba16 += 4;
                dst += 3;
        }
}

/* 4:2:2 packed YUV holds 2 pixels in 4 bytes with Y per-pixel and Cb/Cr
 * shared across the pair. UYVY and YUYV differ only in pack order. */
struct yuv422_pair {
        uint8_t y0, y1, cb, cr;
        uint8_t a_y0, a_y1, a_chroma;
};

static struct yuv422_pair convert_pair_16bit_rgb_to_yuv422(
        const uint16_t *rgba16, const struct color_coeffs *cfs)
{
        uint16_t r0 = rgba16[0], g0 = rgba16[1], b0 = rgba16[2];
        uint16_t a0 = rgba16[3];
        uint16_t r1 = rgba16[4], g1 = rgba16[5], b1 = rgba16[6];
        uint16_t a1 = rgba16[7];

        int y0 = (RGB_TO_Y(*cfs, r0, g0, b0) >> COMP_OFF_8) + Y_OFFSET_8;
        int y1 = (RGB_TO_Y(*cfs, r1, g1, b1) >> COMP_OFF_8) + Y_OFFSET_8;

        int32_t cb_sum = RGB_TO_CB(*cfs, r0, g0, b0)
                       + RGB_TO_CB(*cfs, r1, g1, b1);
        int32_t cr_sum = RGB_TO_CR(*cfs, r0, g0, b0)
                       + RGB_TO_CR(*cfs, r1, g1, b1);
        int cb = ((cb_sum / 2) >> COMP_OFF_8) + CBCR_OFFSET_8;
        int cr = ((cr_sum / 2) >> COMP_OFF_8) + CBCR_OFFSET_8;

        struct yuv422_pair p;
        p.y0 = CLAMP(y0, LIMIT_LO(DEPTH8), LIMIT_HI_Y(DEPTH8));
        p.y1 = CLAMP(y1, LIMIT_LO(DEPTH8), LIMIT_HI_Y(DEPTH8));
        p.cb = CLAMP(cb, LIMIT_LO(DEPTH8), LIMIT_HI_CBCR(DEPTH8));
        p.cr = CLAMP(cr, LIMIT_LO(DEPTH8), LIMIT_HI_CBCR(DEPTH8));
        p.a_y0 = a0 >> 8;
        p.a_y1 = a1 >> 8;
        p.a_chroma = (p.a_y0 + p.a_y1 + 1) >> 1;
        return p;
}

/*
 * UYVY: byte order U Y0 V Y1. Width must be even (caller's responsibility);
 * an odd trailing pixel is silently dropped.
 */
void alpha_blend_uyvy(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width)
{
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH8);
        for (int x = 0; x + 2 <= width; x += 2) {
                struct yuv422_pair p = convert_pair_16bit_rgb_to_yuv422(rgba16, &cfs);
                dst[0] = blend8(p.cb, dst[0], p.a_chroma);
                dst[1] = blend8(p.y0, dst[1], p.a_y0);
                dst[2] = blend8(p.cr, dst[2], p.a_chroma);
                dst[3] = blend8(p.y1, dst[3], p.a_y1);
                rgba16 += 8;
                dst += 4;
        }
}

/*
 * YUYV: byte order Y0 U Y1 V. Same conversion as UYVY, different pack order.
 */
void alpha_blend_yuyv(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width)
{
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH8);
        for (int x = 0; x + 2 <= width; x += 2) {
                struct yuv422_pair p = convert_pair_16bit_rgb_to_yuv422(rgba16, &cfs);
                dst[0] = blend8(p.y0, dst[0], p.a_y0);
                dst[1] = blend8(p.cb, dst[1], p.a_chroma);
                dst[2] = blend8(p.y1, dst[2], p.a_y1);
                dst[3] = blend8(p.cr, dst[3], p.a_chroma);
                rgba16 += 8;
                dst += 4;
        }
}

/*
 * Y416: 16-bit YUV 4:4:4 + alpha, layout U(2) Y(2) V(2) A(2) little-endian.
 * Full chroma resolution.
 */
void alpha_blend_y416(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width)
{
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH16);
        for (int x = 0; x < width; x++) {
                uint16_t r = rgba16[0], g = rgba16[1], b = rgba16[2];
                uint16_t a = rgba16[3];
                int inv = 65535 - a;

                int y  = (RGB_TO_Y(cfs, r, g, b)  >> COMP_OFF_16) + Y_OFFSET_16;
                int cb = (RGB_TO_CB(cfs, r, g, b) >> COMP_OFF_16) + CBCR_OFFSET_16;
                int cr = (RGB_TO_CR(cfs, r, g, b) >> COMP_OFF_16) + CBCR_OFFSET_16;
                y  = CLAMP(y,  LIMIT_LO(DEPTH16), LIMIT_HI_Y(DEPTH16));
                cb = CLAMP(cb, LIMIT_LO(DEPTH16), LIMIT_HI_CBCR(DEPTH16));
                cr = CLAMP(cr, LIMIT_LO(DEPTH16), LIMIT_HI_CBCR(DEPTH16));

                uint16_t du, dy, dv, da;
                memcpy(&du, dst + 0, 2);
                memcpy(&dy, dst + 2, 2);
                memcpy(&dv, dst + 4, 2);
                memcpy(&da, dst + 6, 2);

                uint16_t ou = (uint16_t)(((int64_t)cb * a + (int64_t)du * inv) / 65535);
                uint16_t oy = (uint16_t)(((int64_t)y  * a + (int64_t)dy * inv) / 65535);
                uint16_t ov = (uint16_t)(((int64_t)cr * a + (int64_t)dv * inv) / 65535);
                /* Porter-Duff "over": alpha_out = alpha_src + alpha_dst*(1-alpha_src) */
                uint16_t oa = (uint16_t)(a + ((int64_t)da * inv) / 65535);
                memcpy(dst + 0, &ou, 2);
                memcpy(dst + 2, &oy, 2);
                memcpy(dst + 4, &ov, 2);
                memcpy(dst + 6, &oa, 2);

                rgba16 += 4;
                dst += 8;
        }
}

/* 4:2:0 chroma sample averaged from a 2x2 RGB block. Sums are int64_t because
 * a single |CB raw| can reach ~943M; four summed exceeds INT32_MAX (~2.15B). */
struct yuv420_chroma {
        uint8_t cb, cr, a;
};

static struct yuv420_chroma convert_quad_16bit_rgb_to_yuv420(
        const uint16_t *p00, const uint16_t *p01,
        const uint16_t *p10, const uint16_t *p11,
        const struct color_coeffs *cfs)
{
        int64_t cb_sum = (int64_t)RGB_TO_CB(*cfs, p00[0], p00[1], p00[2])
                       + RGB_TO_CB(*cfs, p01[0], p01[1], p01[2])
                       + RGB_TO_CB(*cfs, p10[0], p10[1], p10[2])
                       + RGB_TO_CB(*cfs, p11[0], p11[1], p11[2]);
        int64_t cr_sum = (int64_t)RGB_TO_CR(*cfs, p00[0], p00[1], p00[2])
                       + RGB_TO_CR(*cfs, p01[0], p01[1], p01[2])
                       + RGB_TO_CR(*cfs, p10[0], p10[1], p10[2])
                       + RGB_TO_CR(*cfs, p11[0], p11[1], p11[2]);
        int cb = (int)((cb_sum / 4) >> COMP_OFF_8) + CBCR_OFFSET_8;
        int cr = (int)((cr_sum / 4) >> COMP_OFF_8) + CBCR_OFFSET_8;

        unsigned a_sum = (unsigned)(p00[3] >> 8) + (p01[3] >> 8)
                       + (p10[3] >> 8) + (p11[3] >> 8);

        struct yuv420_chroma c;
        c.cb = CLAMP(cb, LIMIT_LO(DEPTH8), LIMIT_HI_CBCR(DEPTH8));
        c.cr = CLAMP(cr, LIMIT_LO(DEPTH8), LIMIT_HI_CBCR(DEPTH8));
        c.a  = (a_sum + 2) >> 2;
        return c;
}

/* Compute one 8-bit Y from a 16-bit RGB pixel. */
static inline uint8_t
rgb16_to_y8(const struct color_coeffs *cfs, const uint16_t *p)
{
        int y = (RGB_TO_Y(*cfs, p[0], p[1], p[2]) >> COMP_OFF_8) + Y_OFFSET_8;
        return CLAMP(y, LIMIT_LO(DEPTH8), LIMIT_HI_Y(DEPTH8));
}

/*
 * I420: 8-bit YUV 4:2:0 planar. Y plane is full resolution; U and V planes
 * each (width/2) x (height/2). width and height must be even (caller
 * responsibility); odd dimensions silently truncate the last column/row.
 *
 * Fused 2x2-block iteration: the four overlay pixels of each chroma cell
 * feed both the four Y outputs and the single CbCr pair, so each source
 * pixel is read once instead of twice (vs. separate Y-plane and UV-plane
 * passes — half the source DRAM traffic).
 */
void alpha_blend_i420(uint8_t * __restrict dst_y, int y_stride,
                      uint8_t * __restrict dst_u,
                      uint8_t * __restrict dst_v, int uv_stride,
                      const uint16_t * __restrict rgba16, int src_pixel_stride,
                      int width, int height)
{
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH8);
        /* uint16_t elements per overlay row, not bytes (the pointer it
         * advances is uint16_t *). Byte stride = src_row_elems * 2. */
        const size_t src_row_elems = (size_t)src_pixel_stride * 4;
        const int uv_w     = width / 2;
        const int chroma_h = height / 2;

        for (int cy = 0; cy < chroma_h; cy++) {
                const uint16_t *r0 = rgba16 + (size_t)(cy * 2)     * src_row_elems;
                const uint16_t *r1 = rgba16 + (size_t)(cy * 2 + 1) * src_row_elems;
                uint8_t *dy0 = dst_y + (size_t)(cy * 2)     * y_stride;
                uint8_t *dy1 = dst_y + (size_t)(cy * 2 + 1) * y_stride;
                uint8_t *du  = dst_u + (size_t)cy * uv_stride;
                uint8_t *dv  = dst_v + (size_t)cy * uv_stride;

                for (int col = 0; col < uv_w; col++) {
                        const uint16_t *p00 = r0;
                        const uint16_t *p01 = r0 + 4;
                        const uint16_t *p10 = r1;
                        const uint16_t *p11 = r1 + 4;
                        const int dx0 = col * 2;
                        const int dx1 = dx0 + 1;

                        dy0[dx0] = blend8(rgb16_to_y8(&cfs, p00),
                                          dy0[dx0], p00[3] >> 8);
                        dy0[dx1] = blend8(rgb16_to_y8(&cfs, p01),
                                          dy0[dx1], p01[3] >> 8);
                        dy1[dx0] = blend8(rgb16_to_y8(&cfs, p10),
                                          dy1[dx0], p10[3] >> 8);
                        dy1[dx1] = blend8(rgb16_to_y8(&cfs, p11),
                                          dy1[dx1], p11[3] >> 8);

                        struct yuv420_chroma c = convert_quad_16bit_rgb_to_yuv420(
                                p00, p01, p10, p11, &cfs);
                        du[col] = blend8(c.cb, du[col], c.a);
                        dv[col] = blend8(c.cr, dv[col], c.a);

                        r0 += 8;
                        r1 += 8;
                }
        }
}

/* RG48: 16-bit RGB, 6 bytes per pixel, little-endian. No color conversion;
 * the 16-bit overlay components write through directly. */
void alpha_blend_rg48(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width)
{
        for (int x = 0; x < width; x++) {
                unsigned r = rgba16[0];
                unsigned g = rgba16[1];
                unsigned b = rgba16[2];
                unsigned a = rgba16[3];

                uint16_t dr, dg, db;
                memcpy(&dr, dst + 0, 2);
                memcpy(&dg, dst + 2, 2);
                memcpy(&db, dst + 4, 2);

                dr = blend16(r, dr, a);
                dg = blend16(g, dg, a);
                db = blend16(b, db, a);

                memcpy(dst + 0, &dr, 2);
                memcpy(dst + 2, &dg, 2);
                memcpy(dst + 4, &db, 2);

                rgba16 += 4;
                dst += 6;
        }
}

/* v210: 10-bit YUV 4:2:2 packed; 6 pixels per 16 bytes (4 uint32_t words).
 * Components occupy 10-bit slots at shifts 0/10/20; bits 30-31 are padding.
 * dst is uint8_t* with no alignment guarantee, hence memcpy for word access. */
#define V210_PIXELS_PER_GROUP 6
#define V210_BYTES_PER_GROUP  16
#define V210_COMP_BITS        10
#define V210_COMP_MASK        ((1u << V210_COMP_BITS) - 1)

static inline uint32_t v210_pack3(unsigned a, unsigned b, unsigned c)
{
        return (a & V210_COMP_MASK)
             | ((b & V210_COMP_MASK) << V210_COMP_BITS)
             | ((c & V210_COMP_MASK) << (2 * V210_COMP_BITS));
}

void alpha_blend_v210(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width)
{
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, DEPTH10);

        for (int x = 0; x + V210_PIXELS_PER_GROUP <= width;
             x += V210_PIXELS_PER_GROUP) {
                int y[6], a_y[6];
                int cb[3], cr[3], a_c[3];

                for (int i = 0; i < 6; i++) {
                        uint16_t r = rgba16[i*4 + 0];
                        uint16_t g = rgba16[i*4 + 1];
                        uint16_t b = rgba16[i*4 + 2];
                        int yv = (RGB_TO_Y(cfs, r, g, b) >> COMP_OFF_10)
                               + Y_OFFSET_10;
                        y[i] = CLAMP(yv, LIMIT_LO(DEPTH10), LIMIT_HI_Y(DEPTH10));
                        a_y[i] = rgba16[i*4 + 3] >> 6;  /* 16-bit -> 10-bit */
                }
                for (int p = 0; p < 3; p++) {
                        const uint16_t *p0 = rgba16 + (p*2)     * 4;
                        const uint16_t *p1 = rgba16 + (p*2 + 1) * 4;
                        int32_t cb_sum = RGB_TO_CB(cfs, p0[0], p0[1], p0[2])
                                       + RGB_TO_CB(cfs, p1[0], p1[1], p1[2]);
                        int32_t cr_sum = RGB_TO_CR(cfs, p0[0], p0[1], p0[2])
                                       + RGB_TO_CR(cfs, p1[0], p1[1], p1[2]);
                        int cbv = ((cb_sum / 2) >> COMP_OFF_10) + CBCR_OFFSET_10;
                        int crv = ((cr_sum / 2) >> COMP_OFF_10) + CBCR_OFFSET_10;
                        cb[p] = CLAMP(cbv, LIMIT_LO(DEPTH10), LIMIT_HI_CBCR(DEPTH10));
                        cr[p] = CLAMP(crv, LIMIT_LO(DEPTH10), LIMIT_HI_CBCR(DEPTH10));
                        a_c[p] = (a_y[p*2] + a_y[p*2 + 1] + 1) >> 1;
                }

                uint32_t d0, d1, d2, d3;
                memcpy(&d0, dst + 0,  4);
                memcpy(&d1, dst + 4,  4);
                memcpy(&d2, dst + 8,  4);
                memcpy(&d3, dst + 12, 4);

                /* Word layout: 0=Cb0|Y0|Cr0  1=Y1|Cb1|Y2  2=Cr1|Y3|Cb2  3=Y4|Cr2|Y5 */
                int dst_cb0 = (d0 >>  0) & V210_COMP_MASK;
                int dst_y0  = (d0 >> 10) & V210_COMP_MASK;
                int dst_cr0 = (d0 >> 20) & V210_COMP_MASK;
                int dst_y1  = (d1 >>  0) & V210_COMP_MASK;
                int dst_cb1 = (d1 >> 10) & V210_COMP_MASK;
                int dst_y2  = (d1 >> 20) & V210_COMP_MASK;
                int dst_cr1 = (d2 >>  0) & V210_COMP_MASK;
                int dst_y3  = (d2 >> 10) & V210_COMP_MASK;
                int dst_cb2 = (d2 >> 20) & V210_COMP_MASK;
                int dst_y4  = (d3 >>  0) & V210_COMP_MASK;
                int dst_cr2 = (d3 >> 10) & V210_COMP_MASK;
                int dst_y5  = (d3 >> 20) & V210_COMP_MASK;

                d0 = v210_pack3(blend10(cb[0], dst_cb0, a_c[0]),
                                blend10(y[0],  dst_y0,  a_y[0]),
                                blend10(cr[0], dst_cr0, a_c[0]));
                d1 = v210_pack3(blend10(y[1],  dst_y1,  a_y[1]),
                                blend10(cb[1], dst_cb1, a_c[1]),
                                blend10(y[2],  dst_y2,  a_y[2]));
                d2 = v210_pack3(blend10(cr[1], dst_cr1, a_c[1]),
                                blend10(y[3],  dst_y3,  a_y[3]),
                                blend10(cb[2], dst_cb2, a_c[2]));
                d3 = v210_pack3(blend10(y[4],  dst_y4,  a_y[4]),
                                blend10(cr[2], dst_cr2, a_c[2]),
                                blend10(y[5],  dst_y5,  a_y[5]));

                memcpy(dst + 0,  &d0, 4);
                memcpy(dst + 4,  &d1, 4);
                memcpy(dst + 8,  &d2, 4);
                memcpy(dst + 12, &d3, 4);

                rgba16 += V210_PIXELS_PER_GROUP * 4;
                dst += V210_BYTES_PER_GROUP;
        }
}

/*
 * R10k: 10-bit RGB packed in 4 bytes per pixel. Layout (matches
 * vc_copylineRG48toR10k in pixfmt_conv.c):
 *   byte 0:  r[9:2]
 *   byte 1:  g[9:4]            in low 6 bits, r[1:0] in high 2 bits
 *   byte 2:  b[9:6]            in low 4 bits, g[3:0] in high 4 bits
 *   byte 3:  pad (0b11)        in low 2 bits, b[5:0] in high 6 bits
 */
#define R10K_PAD 0x3u

static inline uint32_t r10k_pack3(unsigned r, unsigned g, unsigned b)
{
        return ((r >> 2) & 0xFFu)
             | (((g >> 4) & 0x3Fu) << 8)
             | ((r & 0x3u)  << 14)
             | (((b >> 6) & 0xFu)  << 16)
             | ((g & 0xFu)  << 20)
             | (R10K_PAD    << 24)
             | ((b & 0x3Fu) << 26);
}

static inline void r10k_unpack3(uint32_t d, unsigned *r, unsigned *g, unsigned *b)
{
        *r = ((d >> 0)  & 0xFFu) << 2 | ((d >> 14) & 0x3u);
        *g = ((d >> 8)  & 0x3Fu) << 4 | ((d >> 20) & 0xFu);
        *b = ((d >> 16) & 0xFu)  << 6 | ((d >> 26) & 0x3Fu);
}

void alpha_blend_r10k(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width)
{
        for (int x = 0; x < width; x++) {
                unsigned r = rgba16[0] >> 6;
                unsigned g = rgba16[1] >> 6;
                unsigned b = rgba16[2] >> 6;
                unsigned a = rgba16[3] >> 6;

                uint32_t d;
                memcpy(&d, dst, 4);
                unsigned dr, dg, db;
                r10k_unpack3(d, &dr, &dg, &db);

                d = r10k_pack3(blend10(r, dr, a),
                               blend10(g, dg, a),
                               blend10(b, db, a));
                memcpy(dst, &d, 4);

                rgba16 += 4;
                dst += 4;
        }
}

/*
 * R12L: 12-bit RGB packed, 8 pixels in 36 bytes (4 sub-blocks of 9 bytes,
 * each holding 2 pixels = 6 components × 12 bits = 72 bits). Layout per
 * 9-byte sub-block:
 *   byte 0:                   P0_R[7:0]
 *   byte 1: P0_G[3:0] << 4 |  P0_R[11:8]
 *   byte 2:                   P0_G[11:4]
 *   byte 3:                   P0_B[7:0]
 *   byte 4: P1_R[3:0] << 4 |  P0_B[11:8]
 *   byte 5:                   P1_R[11:4]
 *   byte 6:                   P1_G[7:0]
 *   byte 7: P1_B[3:0] << 4 |  P1_G[11:8]
 *   byte 8:                   P1_B[11:4]
 */
#define R12L_PIXELS_PER_BLOCK 8
#define R12L_BYTES_PER_BLOCK  36
#define R12L_PIXELS_PER_PAIR  2
#define R12L_BYTES_PER_PAIR   9
#define R12L_PAIRS_PER_BLOCK  (R12L_PIXELS_PER_BLOCK / R12L_PIXELS_PER_PAIR)

static inline unsigned blend12(unsigned src, unsigned dst, unsigned a)
{
        return BLEND_N(src, dst, a, 4095u);
}

/* Blend a single 9-byte sub-block (2 RGBA pixels). */
static inline void r12l_blend_pair(uint8_t *dst, const uint16_t *rgba16)
{
        unsigned p0_r = rgba16[0] >> 4, p0_g = rgba16[1] >> 4;
        unsigned p0_b = rgba16[2] >> 4, p0_a = rgba16[3] >> 4;
        unsigned p1_r = rgba16[4] >> 4, p1_g = rgba16[5] >> 4;
        unsigned p1_b = rgba16[6] >> 4, p1_a = rgba16[7] >> 4;

        unsigned d_p0_r = dst[0] | ((dst[1] & 0xFu) << 8);
        unsigned d_p0_g = (dst[1] >> 4) | (dst[2] << 4);
        unsigned d_p0_b = dst[3] | ((dst[4] & 0xFu) << 8);
        unsigned d_p1_r = (dst[4] >> 4) | (dst[5] << 4);
        unsigned d_p1_g = dst[6] | ((dst[7] & 0xFu) << 8);
        unsigned d_p1_b = (dst[7] >> 4) | (dst[8] << 4);

        unsigned o_p0_r = blend12(p0_r, d_p0_r, p0_a);
        unsigned o_p0_g = blend12(p0_g, d_p0_g, p0_a);
        unsigned o_p0_b = blend12(p0_b, d_p0_b, p0_a);
        unsigned o_p1_r = blend12(p1_r, d_p1_r, p1_a);
        unsigned o_p1_g = blend12(p1_g, d_p1_g, p1_a);
        unsigned o_p1_b = blend12(p1_b, d_p1_b, p1_a);

        dst[0] = o_p0_r & 0xFFu;
        dst[1] = ((o_p0_r >> 8) & 0xFu) | ((o_p0_g & 0xFu) << 4);
        dst[2] = (o_p0_g >> 4) & 0xFFu;
        dst[3] = o_p0_b & 0xFFu;
        dst[4] = ((o_p0_b >> 8) & 0xFu) | ((o_p1_r & 0xFu) << 4);
        dst[5] = (o_p1_r >> 4) & 0xFFu;
        dst[6] = o_p1_g & 0xFFu;
        dst[7] = ((o_p1_g >> 8) & 0xFu) | ((o_p1_b & 0xFu) << 4);
        dst[8] = (o_p1_b >> 4) & 0xFFu;
}

void alpha_blend_r12l(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width)
{
        int x = 0;
        for (; x + R12L_PIXELS_PER_BLOCK <= width;
             x += R12L_PIXELS_PER_BLOCK) {
                for (int p = 0; p < R12L_PAIRS_PER_BLOCK; p++) {
                        r12l_blend_pair(dst    + p * R12L_BYTES_PER_PAIR,
                                        rgba16 + p * R12L_PIXELS_PER_PAIR * 4);
                }
                dst    += R12L_BYTES_PER_BLOCK;
                rgba16 += R12L_PIXELS_PER_BLOCK * 4;
        }
        /* Any remaining 2-pixel pairs (R12L data is byte-aligned only at the
         * pair boundary, so a single trailing pixel is silently dropped). */
        while (x + R12L_PIXELS_PER_PAIR <= width) {
                r12l_blend_pair(dst, rgba16);
                dst    += R12L_BYTES_PER_PAIR;
                rgba16 += R12L_PIXELS_PER_PAIR * 4;
                x      += R12L_PIXELS_PER_PAIR;
        }
}
