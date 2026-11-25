/**
 * @file   test/test_alpha_blend.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Unit tests for alpha_blend.c, registered via run_tests.c
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

#include <stdint.h>
#include <string.h>

#include "test_alpha_blend.h"
#include "unit_common.h"
#include "utils/alpha_blend.h"

static void mk_pixel16(uint16_t *p, int r, int g, int b, int a)
{
        p[0] = r; p[1] = g; p[2] = b; p[3] = a;
}

/*
 * Property: alpha=0 in the overlay leaves dst exactly unchanged.
 */
int alpha_blend_test_rgba_alpha_zero(void)
{
        uint8_t dst[12]  = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120};
        uint8_t orig[12];
        memcpy(orig, dst, sizeof dst);

        uint16_t src[12];
        for (int i = 0; i < 3; i++) {
                /* white overlay, alpha=0 - should not affect dst */
                mk_pixel16(src + i*4, 65535, 65535, 65535, 0);
        }
        alpha_blend_rgba(dst, src, 3);

        for (size_t i = 0; i < sizeof dst; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        }
        return 0;
}

/*
 * Property: alpha=max replaces dst with the source RGB. Alpha channel becomes
 * fully opaque (Porter-Duff "over" with src.a=1 -> out.a=1).
 */
int alpha_blend_test_rgba_alpha_max(void)
{
        uint8_t dst[4] = {0, 0, 0, 0};
        uint16_t src[4];
        mk_pixel16(src, 65535, 65535, 65535, 65535);  /* white opaque */
        alpha_blend_rgba(dst, src, 1);
        ASSERT_EQUAL_MESSAGE("R", 255, dst[0]);
        ASSERT_EQUAL_MESSAGE("G", 255, dst[1]);
        ASSERT_EQUAL_MESSAGE("B", 255, dst[2]);
        ASSERT_EQUAL_MESSAGE("A", 255, dst[3]);
        return 0;
}

/*
 * White overlay onto black dst at alpha~50% -> RGB ~128.
 * alpha8 = 32768>>8 = 128, out = (255*128 + 0*127)/255 = 128.
 */
int alpha_blend_test_rgba_half_alpha(void)
{
        uint8_t dst[4] = {0, 0, 0, 0};
        uint16_t src[4];
        mk_pixel16(src, 65535, 65535, 65535, 32768);
        alpha_blend_rgba(dst, src, 1);
        ASSERT_EQUAL_MESSAGE("R", 128, dst[0]);
        ASSERT_EQUAL_MESSAGE("G", 128, dst[1]);
        ASSERT_EQUAL_MESSAGE("B", 128, dst[2]);
        return 0;
}

/* RGB destination: 3 bytes per pixel, no alpha channel in dst. */

int alpha_blend_test_rgb_alpha_zero(void)
{
        uint8_t dst[9] = {10, 20, 30, 40, 50, 60, 70, 80, 90};
        uint8_t orig[9]; memcpy(orig, dst, sizeof dst);
        uint16_t src[12];
        for (int i = 0; i < 3; i++) {
                mk_pixel16(src + i*4, 65535, 65535, 65535, 0);
        }
        alpha_blend_rgb(dst, src, 3);
        for (size_t i = 0; i < sizeof dst; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        }
        return 0;
}

int alpha_blend_test_rgb_alpha_max(void)
{
        uint8_t dst[3] = {0, 0, 0};
        uint16_t src[4];
        mk_pixel16(src, 65535, 32768, 0, 65535);
        alpha_blend_rgb(dst, src, 1);
        ASSERT_EQUAL_MESSAGE("R", 255, dst[0]);
        ASSERT_EQUAL_MESSAGE("G", 128, dst[1]);
        ASSERT_EQUAL_MESSAGE("B",   0, dst[2]);
        return 0;
}

/*
 * UYVY: 8-bit YUV 4:2:2, byte order U Y0 V Y1 (2 pixels in 4 bytes).
 */

int alpha_blend_test_uyvy_alpha_zero(void)
{
        uint8_t dst[8] = {100, 110, 120, 130, 140, 150, 160, 170};
        uint8_t orig[8]; memcpy(orig, dst, sizeof dst);
        uint16_t src[8];
        mk_pixel16(src + 0, 65535, 65535, 65535, 0);
        mk_pixel16(src + 4, 65535, 65535, 65535, 0);
        alpha_blend_uyvy(dst, src, 2);
        for (size_t i = 0; i < sizeof dst; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        }
        return 0;
}

/*
 * White overlay onto black dst at alpha=max -> BT.709 white limited-range:
 * Y=235, Cb=Cr=128.
 */
int alpha_blend_test_uyvy_alpha_max_white(void)
{
        uint8_t dst[4] = {0, 0, 0, 0};
        uint16_t src[8];
        mk_pixel16(src + 0, 65535, 65535, 65535, 65535);
        mk_pixel16(src + 4, 65535, 65535, 65535, 65535);
        alpha_blend_uyvy(dst, src, 2);
        ASSERT_EQUAL_MESSAGE("U",  128, dst[0]);
        ASSERT_EQUAL_MESSAGE("Y0", 235, dst[1]);
        ASSERT_EQUAL_MESSAGE("V",  128, dst[2]);
        ASSERT_EQUAL_MESSAGE("Y1", 235, dst[3]);
        return 0;
}

/*
 * Pure red overlay onto black at alpha=max. BT.709 red as 16-bit input through
 * DEPTH8 coefficients with COMP_OFF=22 produces Y=62 Cb=102 Cr=240. The Y
 * value is 62 (not 63) because UltraGrid's RGB_TO_Y uses a truncating
 * right-shift, not rounding division.
 */
int alpha_blend_test_uyvy_alpha_max_red(void)
{
        uint8_t dst[4] = {0, 0, 0, 0};
        uint16_t src[8];
        mk_pixel16(src + 0, 65535, 0, 0, 65535);
        mk_pixel16(src + 4, 65535, 0, 0, 65535);
        alpha_blend_uyvy(dst, src, 2);
        ASSERT_EQUAL_MESSAGE("U=Cb", 102, dst[0]);
        ASSERT_EQUAL_MESSAGE("Y0",    62, dst[1]);
        ASSERT_EQUAL_MESSAGE("V=Cr", 240, dst[2]);
        ASSERT_EQUAL_MESSAGE("Y1",    62, dst[3]);
        return 0;
}

/*
 * YUYV: 8-bit YUV 4:2:2, byte order Y0 U Y1 V (same conversion as UYVY,
 * different pack).
 */

int alpha_blend_test_yuyv_alpha_zero(void)
{
        uint8_t dst[8] = {100, 110, 120, 130, 140, 150, 160, 170};
        uint8_t orig[8]; memcpy(orig, dst, sizeof dst);
        uint16_t src[8];
        mk_pixel16(src + 0, 65535, 65535, 65535, 0);
        mk_pixel16(src + 4, 65535, 65535, 65535, 0);
        alpha_blend_yuyv(dst, src, 2);
        for (size_t i = 0; i < sizeof dst; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        }
        return 0;
}

int alpha_blend_test_yuyv_alpha_max_white(void)
{
        uint8_t dst[4] = {0, 0, 0, 0};
        uint16_t src[8];
        mk_pixel16(src + 0, 65535, 65535, 65535, 65535);
        mk_pixel16(src + 4, 65535, 65535, 65535, 65535);
        alpha_blend_yuyv(dst, src, 2);
        ASSERT_EQUAL_MESSAGE("Y0", 235, dst[0]);
        ASSERT_EQUAL_MESSAGE("U",  128, dst[1]);
        ASSERT_EQUAL_MESSAGE("Y1", 235, dst[2]);
        ASSERT_EQUAL_MESSAGE("V",  128, dst[3]);
        return 0;
}

int alpha_blend_test_yuyv_alpha_max_red(void)
{
        uint8_t dst[4] = {0, 0, 0, 0};
        uint16_t src[8];
        mk_pixel16(src + 0, 65535, 0, 0, 65535);
        mk_pixel16(src + 4, 65535, 0, 0, 65535);
        alpha_blend_yuyv(dst, src, 2);
        ASSERT_EQUAL_MESSAGE("Y0",    62, dst[0]);
        ASSERT_EQUAL_MESSAGE("U=Cb", 102, dst[1]);
        ASSERT_EQUAL_MESSAGE("Y1",    62, dst[2]);
        ASSERT_EQUAL_MESSAGE("V=Cr", 240, dst[3]);
        return 0;
}

/*
 * Y416: 16-bit YUV 4:4:4 with alpha, byte order U(2) Y(2) V(2) A(2)
 * little-endian, 8 bytes per pixel. Full chroma resolution (no subsampling).
 */

int alpha_blend_test_y416_alpha_zero(void)
{
        uint8_t dst[16];
        for (size_t i = 0; i < sizeof dst; i++) dst[i] = (uint8_t)(i * 17);
        uint8_t orig[16]; memcpy(orig, dst, sizeof dst);
        uint16_t src[8];
        mk_pixel16(src + 0, 65535, 65535, 65535, 0);
        mk_pixel16(src + 4, 65535, 65535, 65535, 0);
        alpha_blend_y416(dst, src, 2);
        for (size_t i = 0; i < sizeof dst; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        }
        return 0;
}

/*
 * White overlay onto zero dst at alpha=max -> BT.709 white at 16-bit limited.
 * Y nominally 235*256 = 60160 but integer truncation produces 60159.
 * Cb = Cr = 128*256 = 32768. Alpha = full opacity (65535).
 */
int alpha_blend_test_y416_alpha_max_white(void)
{
        uint8_t dst[8] = {0,0, 0,0, 0,0, 0,0};
        uint16_t src[4];
        mk_pixel16(src, 65535, 65535, 65535, 65535);
        alpha_blend_y416(dst, src, 1);
        uint16_t u, y, v, a;
        memcpy(&u, dst + 0, 2);
        memcpy(&y, dst + 2, 2);
        memcpy(&v, dst + 4, 2);
        memcpy(&a, dst + 6, 2);
        ASSERT_EQUAL_MESSAGE("U", 32768, u);
        ASSERT_EQUAL_MESSAGE("Y", 60159, y);
        ASSERT_EQUAL_MESSAGE("V", 32768, v);
        ASSERT_EQUAL_MESSAGE("A", 65535, a);
        return 0;
}

/*
 * I420: 8-bit YUV 4:2:0 planar. Y plane full resolution, U and V planes
 * each at half resolution (one chroma sample per 2x2 Y block). Chroma
 * alpha is the average of the 4 source pixels' alphas in each 2x2 block.
 */

int alpha_blend_test_i420_alpha_zero(void)
{
        enum { W = 4, H = 2 };
        uint8_t dst_y[W * H]; uint8_t dst_u[W / 2 * H / 2]; uint8_t dst_v[W / 2 * H / 2];
        uint8_t orig_y[W * H], orig_u[W / 2 * H / 2], orig_v[W / 2 * H / 2];
        for (int i = 0; i < W * H; i++) dst_y[i] = orig_y[i] = (uint8_t)(100 + i);
        for (int i = 0; i < W / 2 * H / 2; i++) {
                dst_u[i] = orig_u[i] = 50;
                dst_v[i] = orig_v[i] = 200;
        }
        uint16_t src[W * H * 4];
        for (int i = 0; i < W * H; i++) {
                mk_pixel16(src + i*4, 65535, 0, 0, 0);  /* red, alpha=0 */
        }
        alpha_blend_i420(dst_y, W, dst_u, dst_v, W / 2, src, W, W, H);
        for (int i = 0; i < W * H; i++) ASSERT_EQUAL_MESSAGE("Y unchanged", orig_y[i], dst_y[i]);
        for (int i = 0; i < W / 2 * H / 2; i++) ASSERT_EQUAL_MESSAGE("U unchanged", orig_u[i], dst_u[i]);
        for (int i = 0; i < W / 2 * H / 2; i++) ASSERT_EQUAL_MESSAGE("V unchanged", orig_v[i], dst_v[i]);
        return 0;
}

int alpha_blend_test_i420_alpha_max_white(void)
{
        enum { W = 2, H = 2 };
        uint8_t dst_y[W * H] = {0};
        uint8_t dst_u[1] = {0};
        uint8_t dst_v[1] = {0};
        uint16_t src[W * H * 4];
        for (int i = 0; i < W * H; i++) {
                mk_pixel16(src + i*4, 65535, 65535, 65535, 65535);
        }
        alpha_blend_i420(dst_y, W, dst_u, dst_v, W / 2, src, W, W, H);
        ASSERT_EQUAL_MESSAGE("Y[0]", 235, dst_y[0]);
        ASSERT_EQUAL_MESSAGE("Y[1]", 235, dst_y[1]);
        ASSERT_EQUAL_MESSAGE("Y[2]", 235, dst_y[2]);
        ASSERT_EQUAL_MESSAGE("Y[3]", 235, dst_y[3]);
        ASSERT_EQUAL_MESSAGE("U", 128, dst_u[0]);
        ASSERT_EQUAL_MESSAGE("V", 128, dst_v[0]);
        return 0;
}

/*
 * Validates the 2x2 chroma alpha averaging. Two pixels in the 2x2 block have
 * alpha=255 (full source), the other two have alpha=0 (full dst). The chroma
 * sample's averaged alpha is (255 + 255 + 0 + 0 + 2) >> 2 = 128. Y values use
 * per-pixel alpha, so two Y pixels stay at dst (=10) and the other two
 * become 235 (white). Chroma is averaged by alpha=128:
 *   U_out = (128 * 128 + 50 * 127) / 255 = 89
 *   V_out = (128 * 128 + 200 * 127) / 255 = 163
 */
int alpha_blend_test_i420_chroma_alpha_averaging(void)
{
        enum { W = 2, H = 2 };
        uint8_t dst_y[W * H] = {10, 10, 10, 10};
        uint8_t dst_u[1] = {50};
        uint8_t dst_v[1] = {200};
        uint16_t src[W * H * 4];
        /* 2x2 block: top-left and bottom-right opaque white, others transparent */
        mk_pixel16(src + 0*4,  65535, 65535, 65535, 65535);  /* top-left  */
        mk_pixel16(src + 1*4,  65535, 65535, 65535, 0);      /* top-right */
        mk_pixel16(src + 2*4,  65535, 65535, 65535, 0);      /* bot-left  */
        mk_pixel16(src + 3*4,  65535, 65535, 65535, 65535);  /* bot-right */
        alpha_blend_i420(dst_y, W, dst_u, dst_v, W / 2, src, W, W, H);
        ASSERT_EQUAL_MESSAGE("Y top-left  (alpha=max)", 235, dst_y[0]);
        ASSERT_EQUAL_MESSAGE("Y top-right (alpha=0)",    10, dst_y[1]);
        ASSERT_EQUAL_MESSAGE("Y bot-left  (alpha=0)",    10, dst_y[2]);
        ASSERT_EQUAL_MESSAGE("Y bot-right (alpha=max)", 235, dst_y[3]);
        ASSERT_EQUAL_MESSAGE("U (averaged alpha)",       89, dst_u[0]);
        ASSERT_EQUAL_MESSAGE("V (averaged alpha)",      163, dst_v[0]);
        return 0;
}

/*
 * Sub-region: a 2x2 overlay blended into the upper-left quadrant of a 4x4
 * destination. Outside the blended region every byte must remain untouched,
 * which exercises the new y_stride / uv_stride / src_stride parameters.
 */
int alpha_blend_test_i420_subregion_strides(void)
{
        enum { DW = 4, DH = 4, OW = 2, OH = 2 };
        uint8_t dst_y[DW * DH];
        uint8_t dst_u[(DW / 2) * (DH / 2)];
        uint8_t dst_v[(DW / 2) * (DH / 2)];
        for (int i = 0; i < DW * DH; i++)            dst_y[i] = (uint8_t)(50 + i);
        for (int i = 0; i < (DW/2) * (DH/2); i++) {  dst_u[i] = 70; dst_v[i] = 180; }

        /* 2x2 opaque-white overlay laid out at OW pixel stride. */
        uint16_t src[OW * OH * 4];
        for (int i = 0; i < OW * OH; i++)
                mk_pixel16(src + i * 4, 65535, 65535, 65535, 65535);

        alpha_blend_i420(dst_y + 0, DW,                  /* upper-left corner of Y plane */
                         dst_u + 0, dst_v + 0, DW / 2,
                         src, OW,
                         OW, OH);

        /* Top half of Y plane: opaque white -> 235. Bottom half untouched. */
        for (int x = 0; x < OW; x++) {
                ASSERT_EQUAL_MESSAGE("Y top-row blended",    235, dst_y[x]);
                ASSERT_EQUAL_MESSAGE("Y second-row blended", 235, dst_y[DW + x]);
        }
        for (int x = OW; x < DW; x++) {
                ASSERT_EQUAL_MESSAGE("Y row 0 right untouched", (uint8_t)(50 + x),         dst_y[x]);
                ASSERT_EQUAL_MESSAGE("Y row 1 right untouched", (uint8_t)(50 + DW + x),    dst_y[DW + x]);
        }
        for (int x = 0; x < DW; x++) {
                ASSERT_EQUAL_MESSAGE("Y row 2 untouched", (uint8_t)(50 + 2*DW + x), dst_y[2*DW + x]);
                ASSERT_EQUAL_MESSAGE("Y row 3 untouched", (uint8_t)(50 + 3*DW + x), dst_y[3*DW + x]);
        }

        /* U/V: only the (0,0) chroma sample (covering the blended 2x2 block)
         * is touched. The (1,0), (0,1), (1,1) samples must stay at baseline. */
        ASSERT_EQUAL_MESSAGE("U(0,0) blended", 128, dst_u[0]);
        ASSERT_EQUAL_MESSAGE("V(0,0) blended", 128, dst_v[0]);
        ASSERT_EQUAL_MESSAGE("U(1,0) untouched",  70, dst_u[1]);
        ASSERT_EQUAL_MESSAGE("V(1,0) untouched", 180, dst_v[1]);
        ASSERT_EQUAL_MESSAGE("U(0,1) untouched",  70, dst_u[2]);
        ASSERT_EQUAL_MESSAGE("V(0,1) untouched", 180, dst_v[2]);
        ASSERT_EQUAL_MESSAGE("U(1,1) untouched",  70, dst_u[3]);
        ASSERT_EQUAL_MESSAGE("V(1,1) untouched", 180, dst_v[3]);
        return 0;
}

/*
 * RG48: 16-bit RGB, 6 bytes per pixel little-endian, no color conversion.
 * 16-bit overlay components map directly to dst components (just truncate
 * 16->16 = pass-through), alpha-blended at full 16-bit precision.
 */
int alpha_blend_test_rg48_alpha_zero(void)
{
        uint8_t dst[6 * 2];
        for (size_t i = 0; i < sizeof dst; i++) dst[i] = (uint8_t)(0x10 + i);
        uint8_t orig[6 * 2];
        memcpy(orig, dst, sizeof dst);

        uint16_t src[2 * 4];
        mk_pixel16(src + 0*4, 65535, 0, 65535, 0);   /* magenta, alpha=0 */
        mk_pixel16(src + 1*4, 12345, 6789, 1111, 0); /* arbitrary, alpha=0 */

        alpha_blend_rg48(dst, src, 2);
        for (size_t i = 0; i < sizeof dst; i++)
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        return 0;
}

int alpha_blend_test_rg48_alpha_max_white(void)
{
        uint8_t dst[6 * 2] = {0};
        uint16_t src[2 * 4];
        mk_pixel16(src + 0*4, 65535, 65535, 65535, 65535);
        mk_pixel16(src + 1*4, 65535,     0,     0, 65535);

        alpha_blend_rg48(dst, src, 2);

        uint16_t r0, g0, b0, r1, g1, b1;
        memcpy(&r0, dst + 0,  2); memcpy(&g0, dst + 2, 2); memcpy(&b0, dst + 4, 2);
        memcpy(&r1, dst + 6,  2); memcpy(&g1, dst + 8, 2); memcpy(&b1, dst + 10, 2);
        ASSERT_EQUAL_MESSAGE("p0 R", 65535, r0);
        ASSERT_EQUAL_MESSAGE("p0 G", 65535, g0);
        ASSERT_EQUAL_MESSAGE("p0 B", 65535, b0);
        ASSERT_EQUAL_MESSAGE("p1 R", 65535, r1);
        ASSERT_EQUAL_MESSAGE("p1 G",     0, g1);
        ASSERT_EQUAL_MESSAGE("p1 B",     0, b1);
        return 0;
}

/* v210 component extraction at 10-bit, low-bit shift positions 0/10/20. */
#define V210_GET(w, shift) (((w) >> (shift)) & 0x3FFu)

int alpha_blend_test_v210_alpha_zero(void)
{
        /* alpha_blend_v210 strips bits 30-31 (v210 padding) on round-trip,
         * so dst must be initialised with valid v210 (zero padding) for the
         * "byte unchanged" assertion to be meaningful. */
        uint8_t dst[16];
        uint32_t w[4] = {
                (100u << 0) | (200u << 10) | (300u << 20),
                (400u << 0) | (500u << 10) | (600u << 20),
                (700u << 0) | (800u << 10) | (900u << 20),
                (150u << 0) | (250u << 10) | (350u << 20),
        };
        memcpy(dst, w, sizeof dst);
        uint8_t orig[16]; memcpy(orig, dst, sizeof dst);
        uint16_t src[24];
        for (int i = 0; i < 6; i++) {
                mk_pixel16(src + i*4, 65535, 65535, 65535, 0);
        }
        alpha_blend_v210(dst, src, 6);
        for (size_t i = 0; i < sizeof dst; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        }
        return 0;
}

/* White overlay at alpha=max -> 10-bit limited Y=940, Cb=Cr=512. */
int alpha_blend_test_v210_alpha_max_white(void)
{
        uint8_t dst[16] = {0};
        uint16_t src[24];
        for (int i = 0; i < 6; i++) {
                mk_pixel16(src + i*4, 65535, 65535, 65535, 65535);
        }
        alpha_blend_v210(dst, src, 6);

        uint32_t w[4];
        memcpy(w, dst, sizeof w);

        /* Word layout: 0=Cb0 Y0 Cr0   1=Y1 Cb1 Y2   2=Cr1 Y3 Cb2   3=Y4 Cr2 Y5 */
        int y[6]  = { V210_GET(w[0], 10), V210_GET(w[1], 0),  V210_GET(w[1], 20),
                      V210_GET(w[2], 10), V210_GET(w[3], 0),  V210_GET(w[3], 20) };
        int cb[3] = { V210_GET(w[0], 0),  V210_GET(w[1], 10), V210_GET(w[2], 20) };
        int cr[3] = { V210_GET(w[0], 20), V210_GET(w[2], 0),  V210_GET(w[3], 10) };

        for (int i = 0; i < 6; i++) ASSERT_EQUAL_MESSAGE("Y",  940, y[i]);
        for (int i = 0; i < 3; i++) ASSERT_EQUAL_MESSAGE("Cb", 512, cb[i]);
        for (int i = 0; i < 3; i++) ASSERT_EQUAL_MESSAGE("Cr", 512, cr[i]);
        for (int i = 0; i < 4; i++) ASSERT_EQUAL_MESSAGE("padding", 0u, w[i] >> 30);
        return 0;
}

/* R10k: see alpha_blend_r10k in src/utils/alpha_blend.c for layout. */

int alpha_blend_test_r10k_alpha_zero(void)
{
        /* Pad bits (24-25) must be 0b11 for valid R10k; the rest is arbitrary. */
        uint8_t dst[8] = {0xAA, 0xBB, 0xCC, 0x03,  0x11, 0x22, 0x33, 0x03};
        uint8_t orig[8]; memcpy(orig, dst, sizeof dst);
        uint16_t src[8];
        mk_pixel16(src + 0, 65535, 65535, 65535, 0);
        mk_pixel16(src + 4, 65535, 0, 0, 0);
        alpha_blend_r10k(dst, src, 2);
        for (size_t i = 0; i < sizeof dst; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        }
        return 0;
}

/* White overlay at alpha=max -> r=g=b=1023, padding=0x3. All 4 bytes 0xFF. */
int alpha_blend_test_r10k_alpha_max_white(void)
{
        uint8_t dst[4] = {0, 0, 0, 0};
        uint16_t src[4];
        mk_pixel16(src, 65535, 65535, 65535, 65535);
        alpha_blend_r10k(dst, src, 1);

        uint32_t w; memcpy(&w, dst, 4);
        unsigned r = ((w >> 0)  & 0xFFu) << 2 | ((w >> 14) & 0x3u);
        unsigned g = ((w >> 8)  & 0x3Fu) << 4 | ((w >> 20) & 0xFu);
        unsigned b = ((w >> 16) & 0xFu)  << 6 | ((w >> 26) & 0x3Fu);
        unsigned pad = (w >> 24) & 0x3u;
        ASSERT_EQUAL_MESSAGE("R",   1023u, r);
        ASSERT_EQUAL_MESSAGE("G",   1023u, g);
        ASSERT_EQUAL_MESSAGE("B",   1023u, b);
        ASSERT_EQUAL_MESSAGE("pad", 0x3u,  pad);
        return 0;
}

/* R12L: see alpha_blend_r12l in src/utils/alpha_blend.c for layout. */

/* Unpack a 12-bit pixel from a 9-byte R12L sub-block. */
static void r12l_unpack_pixel(const uint8_t *block, int pixel,
                              unsigned *r, unsigned *g, unsigned *b)
{
        if (pixel == 0) {
                *r = block[0] | ((block[1] & 0xFu) << 8);
                *g = (block[1] >> 4) | (block[2] << 4);
                *b = block[3] | ((block[4] & 0xFu) << 8);
        } else {
                *r = (block[4] >> 4) | (block[5] << 4);
                *g = block[6] | ((block[7] & 0xFu) << 8);
                *b = (block[7] >> 4) | (block[8] << 4);
        }
}

int alpha_blend_test_r12l_alpha_zero(void)
{
        /* R12L has no padding bits; arbitrary fill is valid. */
        uint8_t dst[36];
        for (size_t i = 0; i < sizeof dst; i++) dst[i] = 0xAAu;
        uint8_t orig[36]; memcpy(orig, dst, sizeof dst);
        uint16_t src[8 * 4];
        for (int i = 0; i < 8; i++) {
                mk_pixel16(src + i*4, 65535, 65535, 65535, 0);
        }
        alpha_blend_r12l(dst, src, 8);
        for (size_t i = 0; i < sizeof dst; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], dst[i]);
        }
        return 0;
}

/*
 * Asymmetric per-pixel values verify the byte layout: each pixel's r/g/b
 * differ from neighbours, so a stride or pair-order bug would mis-place
 * a value. Spot-checks block 0 P0+P1 and block 3 P1 (last pixel).
 */
int alpha_blend_test_r12l_alpha_max_white(void)
{
        uint8_t dst[36] = {0};
        uint16_t src[8 * 4];
        for (int i = 0; i < 8; i++) {
                /* 16-bit values that down-shift by 4 to recognisable 12-bit
                 * triples: pixel i gets r=0x100*(i+1), g=0x010*(i+1), b=0x001*(i+1). */
                mk_pixel16(src + i*4,
                           (0x100u * (i + 1)) << 4,
                           (0x010u * (i + 1)) << 4,
                           (0x001u * (i + 1)) << 4,
                           65535);
        }
        alpha_blend_r12l(dst, src, 8);

        unsigned r, g, b;
        r12l_unpack_pixel(dst + 0,  0, &r, &g, &b);
        ASSERT_EQUAL_MESSAGE("blk0 P0 R", 0x100u, r);
        ASSERT_EQUAL_MESSAGE("blk0 P0 G", 0x010u, g);
        ASSERT_EQUAL_MESSAGE("blk0 P0 B", 0x001u, b);

        r12l_unpack_pixel(dst + 0,  1, &r, &g, &b);
        ASSERT_EQUAL_MESSAGE("blk0 P1 R", 0x200u, r);
        ASSERT_EQUAL_MESSAGE("blk0 P1 G", 0x020u, g);
        ASSERT_EQUAL_MESSAGE("blk0 P1 B", 0x002u, b);

        r12l_unpack_pixel(dst + 27, 1, &r, &g, &b);  /* last pixel of block 3 */
        ASSERT_EQUAL_MESSAGE("blk3 P1 R", 0x800u, r);
        ASSERT_EQUAL_MESSAGE("blk3 P1 G", 0x080u, g);
        ASSERT_EQUAL_MESSAGE("blk3 P1 B", 0x008u, b);
        return 0;
}
