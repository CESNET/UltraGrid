/**
 * @file   test/test_overlay.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Unit tests for the vo_postprocess/overlay module - alpha blending,
 *         PAM loading, layout, scaling, soft edges, config parsing and file
 *         watching - squashed into a single file exposing one test case
 *         (overlay_test) to run_tests.c.
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

#include "config.h"

#ifdef HAVE_OVERLAY

#include <fcntl.h>
#include <stdbool.h>
#include <libavutil/avutil.h> // for av_log_set_level
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <sys/utime.h>
#else
#include <sys/time.h>
#endif
#include <unistd.h>

#include "unit_common.h"
#include "utils/fs.h"
#include "vo_postprocess/overlay/alpha_blend.h"
#include "vo_postprocess/overlay/overlay_config.h"
#include "vo_postprocess/overlay/overlay_layout.h"
#include "vo_postprocess/overlay/overlay_pam.h"
#include "vo_postprocess/overlay/overlay_scale.h"
#include "vo_postprocess/overlay/overlay_soft_edge.h"
#include "vo_postprocess/overlay/overlay_watch.h"

int overlay_test(void);

extern const char *g_test_opts; // set by run_tests.c from "overlay_test:<opts>"

/*
 * ---- test_alpha_blend (formerly test/test_alpha_blend.c) ----
 */

static void mk_pixel16(uint16_t *p, int r, int g, int b, int a)
{
        p[0] = r; p[1] = g; p[2] = b; p[3] = a;
}

/*
 * Property: alpha=0 in the overlay leaves dst exactly unchanged.
 */
static int alpha_blend_test_rgba_alpha_zero(void)
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
static int alpha_blend_test_rgba_alpha_max(void)
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
static int alpha_blend_test_rgba_half_alpha(void)
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

static int alpha_blend_test_rgb_alpha_zero(void)
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

static int alpha_blend_test_rgb_alpha_max(void)
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

static int alpha_blend_test_uyvy_alpha_zero(void)
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
static int alpha_blend_test_uyvy_alpha_max_white(void)
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
static int alpha_blend_test_uyvy_alpha_max_red(void)
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

static int alpha_blend_test_yuyv_alpha_zero(void)
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

static int alpha_blend_test_yuyv_alpha_max_white(void)
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

static int alpha_blend_test_yuyv_alpha_max_red(void)
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

static int alpha_blend_test_y416_alpha_zero(void)
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
static int alpha_blend_test_y416_alpha_max_white(void)
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

static int alpha_blend_test_i420_alpha_zero(void)
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

static int alpha_blend_test_i420_alpha_max_white(void)
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
static int alpha_blend_test_i420_chroma_alpha_averaging(void)
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
static int alpha_blend_test_i420_subregion_strides(void)
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
static int alpha_blend_test_rg48_alpha_zero(void)
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

static int alpha_blend_test_rg48_alpha_max_white(void)
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

static int alpha_blend_test_v210_alpha_zero(void)
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
static int alpha_blend_test_v210_alpha_max_white(void)
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

/* R10k: see alpha_blend_r10k in src/vo_postprocess/overlay/alpha_blend.c for layout. */

static int alpha_blend_test_r10k_alpha_zero(void)
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
static int alpha_blend_test_r10k_alpha_max_white(void)
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

/* R12L: see alpha_blend_r12l in src/vo_postprocess/overlay/alpha_blend.c for layout. */

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

static int alpha_blend_test_r12l_alpha_zero(void)
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
static int alpha_blend_test_r12l_alpha_max_white(void)
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

/*
 * ---- test_overlay_pam (formerly test/test_overlay_pam.c) ----
 */

/* Write a temp PAM file with the given header + data and return its path via
 * *path_out (caller will unlink). Returns true on success. */
static bool write_temp_pam(const char *header, const unsigned char *data,
                           size_t data_len, const char **path_out)
{
        FILE *f = get_temp_file(path_out);
        if (!f) return false;
        const size_t header_len = strlen(header);
        if (fwrite(header, 1, header_len, f) != header_len ||
            fwrite(data, 1, data_len, f) != data_len) {
                fclose(f);
                unlink(*path_out);
                return false;
        }
        if (fclose(f) != 0) {
                unlink(*path_out);
                return false;
        }
        return true;
}

/*
 * 8-bit RGBA PAM (maxval=255) loads with 8-bit values bit-replicated to 16-bit:
 * value 0xAB at 8-bit becomes 0xABAB at 16-bit.
 */
static int overlay_pam_test_load_8bit_rgba(void)
{
        const char *header =
                "P7\n"
                "WIDTH 2\n"
                "HEIGHT 1\n"
                "DEPTH 4\n"
                "MAXVAL 255\n"
                "TUPLTYPE RGB_ALPHA\n"
                "ENDHDR\n";
        const unsigned char data[] = {
                0xFF, 0x00, 0x00, 0xFF,   /* opaque red   */
                0xAB, 0xCD, 0xEF, 0x80,   /* arbitrary, half-alpha */
        };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned true", ok);
        ASSERT_EQUAL_MESSAGE("width",  2, w);
        ASSERT_EQUAL_MESSAGE("height", 1, h);

        ASSERT_EQUAL_MESSAGE("p0 R", 0xFFFFu, (unsigned)out[0]);
        ASSERT_EQUAL_MESSAGE("p0 G", 0x0000u, (unsigned)out[1]);
        ASSERT_EQUAL_MESSAGE("p0 B", 0x0000u, (unsigned)out[2]);
        ASSERT_EQUAL_MESSAGE("p0 A", 0xFFFFu, (unsigned)out[3]);
        ASSERT_EQUAL_MESSAGE("p1 R", 0xABABu, (unsigned)out[4]);
        ASSERT_EQUAL_MESSAGE("p1 G", 0xCDCDu, (unsigned)out[5]);
        ASSERT_EQUAL_MESSAGE("p1 B", 0xEFEFu, (unsigned)out[6]);
        ASSERT_EQUAL_MESSAGE("p1 A", 0x8080u, (unsigned)out[7]);

        free(out);
        return 0;
}

/* 3-channel RGB PAM (DEPTH 3, no alpha) gets alpha=65535 added. */
static int overlay_pam_test_load_8bit_rgb_adds_alpha(void)
{
        const char *header =
                "P7\n"
                "WIDTH 1\n"
                "HEIGHT 1\n"
                "DEPTH 3\n"
                "MAXVAL 255\n"
                "TUPLTYPE RGB\n"
                "ENDHDR\n";
        const unsigned char data[] = { 0x12, 0x34, 0x56 };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned true", ok);
        ASSERT_EQUAL_MESSAGE("R", 0x1212u, (unsigned)out[0]);
        ASSERT_EQUAL_MESSAGE("G", 0x3434u, (unsigned)out[1]);
        ASSERT_EQUAL_MESSAGE("B", 0x5656u, (unsigned)out[2]);
        ASSERT_EQUAL_MESSAGE("A defaults to opaque", 0xFFFFu, (unsigned)out[3]);
        free(out);
        return 0;
}

/* 16-bit RGBA PAM (maxval=65535): samples are 2 bytes big-endian. Data is read
 * verbatim into the uint16_t buffer with byte-swap on little-endian hosts. */
static int overlay_pam_test_load_16bit_rgba(void)
{
        const char *header =
                "P7\n"
                "WIDTH 1\n"
                "HEIGHT 1\n"
                "DEPTH 4\n"
                "MAXVAL 65535\n"
                "TUPLTYPE RGB_ALPHA\n"
                "ENDHDR\n";
        /* Big-endian 16-bit samples: R=0x1234, G=0x5678, B=0x9ABC, A=0xDEF0 */
        const unsigned char data[] = {
                0x12, 0x34,  0x56, 0x78,  0x9A, 0xBC,  0xDE, 0xF0,
        };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned true", ok);
        ASSERT_EQUAL_MESSAGE("R", 0x1234u, (unsigned)out[0]);
        ASSERT_EQUAL_MESSAGE("G", 0x5678u, (unsigned)out[1]);
        ASSERT_EQUAL_MESSAGE("B", 0x9ABCu, (unsigned)out[2]);
        ASSERT_EQUAL_MESSAGE("A", 0xDEF0u, (unsigned)out[3]);
        free(out);
        return 0;
}

static int overlay_pam_test_rejects_missing_file(void)
{
        uint16_t *out = (uint16_t *)0xDEADBEEF;
        int w = 999, h = 999;

        // pam_read writes to stderr unconditionally
        int stderr_fileno_save = ret_fd_dup_and_silence(STDERR_FILENO);
        bool ok = overlay_load_pam_rgba16("/nonexistent/path/no.pam",
                                          &out, &w, &h);
        fd_restore(stderr_fileno_save, STDERR_FILENO);

        ASSERT_MESSAGE("load returned false", !ok);
        /* outputs untouched per contract */
        ASSERT_EQUAL_MESSAGE("out untouched", (uintptr_t)0xDEADBEEF, (uintptr_t)out);
        ASSERT_EQUAL_MESSAGE("w untouched",   999, w);
        ASSERT_EQUAL_MESSAGE("h untouched",   999, h);
        return 0;
}

static int overlay_pam_test_rejects_grayscale(void)
{
        const char *header =
                "P7\n"
                "WIDTH 1\n"
                "HEIGHT 1\n"
                "DEPTH 1\n"
                "MAXVAL 255\n"
                "TUPLTYPE GRAYSCALE\n"
                "ENDHDR\n";
        const unsigned char data[] = { 0x80 };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned false", !ok);
        return 0;
}

static int overlay_pam_test_rejects_intermediate_maxval(void)
{
        /* 10-bit (maxval=1023) is rejected; values would otherwise be
         * silently treated as raw 16-bit, producing a darker overlay. */
        const char *header =
                "P7\n"
                "WIDTH 1\n"
                "HEIGHT 1\n"
                "DEPTH 4\n"
                "MAXVAL 1023\n"
                "TUPLTYPE RGB_ALPHA\n"
                "ENDHDR\n";
        const unsigned char data[] = {
                0x03, 0xFF,  0x03, 0xFF,  0x03, 0xFF,  0x03, 0xFF,
        };
        const char *path = NULL;
        ASSERT_MESSAGE("write temp file",
                       write_temp_pam(header, data, sizeof data, &path));

        uint16_t *out = NULL;
        int w = 0, h = 0;
        bool ok = overlay_load_pam_rgba16(path, &out, &w, &h);
        unlink(path);

        ASSERT_MESSAGE("load returned false", !ok);
        return 0;
}

/*
 * ---- test_overlay_layout (formerly test/test_overlay_layout.c) ----
 */

/* Center: a 200x100 overlay in 1920x1080 frame -> (860, 490, 200, 100). */
static int overlay_layout_test_center(void)
{
        struct overlay_rect r = overlay_calc_rect(OVERLAY_POS_CENTER,
                                                  0, 0, 1920, 1080,
                                                  200, 100, 1, 1);
        ASSERT_EQUAL_MESSAGE("x",      860, r.x);
        ASSERT_EQUAL_MESSAGE("y",      490, r.y);
        ASSERT_EQUAL_MESSAGE("width",  200, r.width);
        ASSERT_EQUAL_MESSAGE("height", 100, r.height);
        return 0;
}

static int overlay_layout_test_corners(void)
{
        struct overlay_rect r;

        r = overlay_calc_rect(OVERLAY_POS_TOP_LEFT, 0, 0,
                              1920, 1080, 200, 100, 1, 1);
        ASSERT_EQUAL_MESSAGE("TL x", 0, r.x);
        ASSERT_EQUAL_MESSAGE("TL y", 0, r.y);

        r = overlay_calc_rect(OVERLAY_POS_TOP_RIGHT, 0, 0,
                              1920, 1080, 200, 100, 1, 1);
        ASSERT_EQUAL_MESSAGE("TR x", 1720, r.x);
        ASSERT_EQUAL_MESSAGE("TR y",    0, r.y);

        r = overlay_calc_rect(OVERLAY_POS_BOTTOM_LEFT, 0, 0,
                              1920, 1080, 200, 100, 1, 1);
        ASSERT_EQUAL_MESSAGE("BL x",    0, r.x);
        ASSERT_EQUAL_MESSAGE("BL y",  980, r.y);

        r = overlay_calc_rect(OVERLAY_POS_BOTTOM_RIGHT, 0, 0,
                              1920, 1080, 200, 100, 1, 1);
        ASSERT_EQUAL_MESSAGE("BR x", 1720, r.x);
        ASSERT_EQUAL_MESSAGE("BR y",  980, r.y);
        return 0;
}

/*
 * Custom: positive values are absolute; negative values count from the
 * right/bottom edges (so the overlay's right edge sits N pixels from the
 * frame's right edge for x = -N).
 */
static int overlay_layout_test_custom_negative_from_edge(void)
{
        /* x=-10, y=-10 with overlay 200x100 in 1920x1080:
         * x = 1920 + (-10) - 200 = 1710
         * y = 1080 + (-10) - 100 = 970 */
        struct overlay_rect r = overlay_calc_rect(OVERLAY_POS_CUSTOM,
                                                  -10, -10, 1920, 1080,
                                                  200, 100, 1, 1);
        ASSERT_EQUAL_MESSAGE("x", 1710, r.x);
        ASSERT_EQUAL_MESSAGE("y",  970, r.y);
        return 0;
}

/*
 * Block alignment: x and width are snapped down to a multiple of block_pixels.
 * v210's block is 6 pixels; an overlay at x=863 / width=200 should snap to
 * x=858 / width=198.
 */
static int overlay_layout_test_block_pixel_alignment(void)
{
        struct overlay_rect r = overlay_calc_rect(OVERLAY_POS_CUSTOM,
                                                  863, 100, 1920, 1080,
                                                  200, 100, 6, 1);
        ASSERT_EQUAL_MESSAGE("x snapped",     858, r.x);
        ASSERT_EQUAL_MESSAGE("width snapped", 198, r.width);
        ASSERT_EQUAL_MESSAGE("y unchanged",   100, r.y);
        ASSERT_EQUAL_MESSAGE("height",        100, r.height);
        return 0;
}

/* Overlay larger than frame, top-left positioning: visible rect is the
 * overlay's top-left corner; src_x/src_y stay at 0. */
static int overlay_layout_test_overlay_larger_than_frame(void)
{
        struct overlay_rect r = overlay_calc_rect(OVERLAY_POS_TOP_LEFT,
                                                  0, 0, 100, 50,
                                                  200, 200, 1, 1);
        ASSERT_EQUAL_MESSAGE("x",       0, r.x);
        ASSERT_EQUAL_MESSAGE("y",       0, r.y);
        ASSERT_EQUAL_MESSAGE("width",  100, r.width);
        ASSERT_EQUAL_MESSAGE("height",  50, r.height);
        ASSERT_EQUAL_MESSAGE("src_x",   0, r.src_x);
        ASSERT_EQUAL_MESSAGE("src_y",   0, r.src_y);
        return 0;
}

/* Centred 200x200 overlay on a 100x50 frame: visible region maps to the
 * centre slice of the overlay, not the top-left corner. */
static int overlay_layout_test_oversized_center(void)
{
        struct overlay_rect r = overlay_calc_rect(OVERLAY_POS_CENTER,
                                                  0, 0, 100, 50,
                                                  200, 200, 1, 1);
        ASSERT_EQUAL_MESSAGE("x",       0, r.x);
        ASSERT_EQUAL_MESSAGE("y",       0, r.y);
        ASSERT_EQUAL_MESSAGE("width",  100, r.width);
        ASSERT_EQUAL_MESSAGE("height",  50, r.height);
        ASSERT_EQUAL_MESSAGE("src_x",  50, r.src_x);  /* (200-100)/2 */
        ASSERT_EQUAL_MESSAGE("src_y",  75, r.src_y);  /* (200-50)/2 */
        return 0;
}

/* Right-aligned 200x200 overlay on a 100x50 frame: visible region is
 * the right slice (last 100 columns) and bottom slice (last 50 rows). */
static int overlay_layout_test_oversized_right(void)
{
        struct overlay_rect r = overlay_calc_rect(OVERLAY_POS_BOTTOM_RIGHT,
                                                  0, 0, 100, 50,
                                                  200, 200, 1, 1);
        ASSERT_EQUAL_MESSAGE("x",       0, r.x);
        ASSERT_EQUAL_MESSAGE("y",       0, r.y);
        ASSERT_EQUAL_MESSAGE("width",  100, r.width);
        ASSERT_EQUAL_MESSAGE("height",  50, r.height);
        ASSERT_EQUAL_MESSAGE("src_x", 100, r.src_x);  /* 200-100 */
        ASSERT_EQUAL_MESSAGE("src_y", 150, r.src_y);  /* 200-50 */
        return 0;
}

/* Custom-positive offset on an oversized overlay: the right portion of
 * the overlay is clipped, but the visible portion still starts at the
 * overlay's column 0 (no slicing on the left). */
static int overlay_layout_test_oversized_custom_positive(void)
{
        struct overlay_rect r = overlay_calc_rect(OVERLAY_POS_CUSTOM,
                                                  20, 10, 100, 50,
                                                  200, 200, 1, 1);
        ASSERT_EQUAL_MESSAGE("x",      20, r.x);
        ASSERT_EQUAL_MESSAGE("y",      10, r.y);
        ASSERT_EQUAL_MESSAGE("width",  80, r.width);   /* 100 - 20 */
        ASSERT_EQUAL_MESSAGE("height", 40, r.height);  /* 50 - 10 */
        ASSERT_EQUAL_MESSAGE("src_x",   0, r.src_x);
        ASSERT_EQUAL_MESSAGE("src_y",   0, r.src_y);
        return 0;
}

/* Vertical block alignment for chroma-subsampled formats: an odd custom_y
 * snaps the rect origin down, and an odd remaining height snaps the height
 * down too, so the chroma sample 2x2 grid is preserved. */
static int overlay_layout_test_block_lines_alignment(void)
{
        /* y=3 with block_lines=2 -> y=2; height=5 -> 4. */
        struct overlay_rect r = overlay_calc_rect(OVERLAY_POS_CUSTOM,
                                                  10, 3, 1920, 1080,
                                                  100, 5, 2, 2);
        ASSERT_EQUAL_MESSAGE("x snapped",      10, r.x);
        ASSERT_EQUAL_MESSAGE("y snapped",       2, r.y);
        ASSERT_EQUAL_MESSAGE("width snapped", 100, r.width);
        ASSERT_EQUAL_MESSAGE("height snapped",  4, r.height);
        return 0;
}

/*
 * ---- test_overlay_scale (formerly test/test_overlay_scale.c) ----
 */

static void
fill_solid_red(uint16_t *buf, int w, int h)
{
        for (int i = 0; i < w * h; i++) {
                buf[i*4 + 0] = 65535;  /* R */
                buf[i*4 + 1] = 0;      /* G */
                buf[i*4 + 2] = 0;      /* B */
                buf[i*4 + 3] = 65535;  /* A */
        }
}

/* Identity scale (same dims) should preserve every pixel exactly. */
static int overlay_scale_test_identity(void)
{
        enum { W = 4, H = 2 };
        uint16_t src[W * H * 4];
        for (int i = 0; i < W * H * 4; i++) src[i] = (uint16_t)(i * 257);

        uint16_t *dst = overlay_scale_rgba16(src, W, H, W, H);
        ASSERT_MESSAGE("not NULL", dst != NULL);
        for (int i = 0; i < W * H * 4; i++) {
                ASSERT_EQUAL_MESSAGE("byte equal", src[i], dst[i]);
        }
        free(dst);
        return 0;
}

/* Upscale a solid-colour overlay: every output pixel is still that colour
 * (within +/- 64 LSB for libswscale filter rounding). */
static int overlay_scale_test_upscale_solid_colour(void)
{
        enum { SW = 4, SH = 4, DW = 16, DH = 16 };
        uint16_t src[SW * SH * 4];
        fill_solid_red(src, SW, SH);

        uint16_t *dst = overlay_scale_rgba16(src, SW, SH, DW, DH);
        ASSERT_MESSAGE("not NULL", dst != NULL);
        /* libswscale bilinear has up to a few LSB of filter rounding even
         * on constant-colour input. Allow a small tolerance — the real
         * contract is "the colour stays close to red", not "no rounding". */
        const int TOL = 64;  /* ~0.1% of full range */
        for (int i = 0; i < DW * DH; i++) {
                ASSERT_MESSAGE("R close to max",
                               dst[i*4 + 0] >= 65535 - TOL);
                ASSERT_MESSAGE("G close to 0",
                               dst[i*4 + 1] <= TOL);
                ASSERT_MESSAGE("B close to 0",
                               dst[i*4 + 2] <= TOL);
                ASSERT_MESSAGE("A close to max",
                               dst[i*4 + 3] >= 65535 - TOL);
        }
        free(dst);
        return 0;
}

/* Downscale a 4x4 half-red half-blue checkerboard to 2x2: each output
 * sample averages a 2x2 input block, so the corners come out as a
 * mid-tone purple. We only sanity-check that the channels mix instead
 * of asserting a precise filter output. */
static int overlay_scale_test_downscale_average(void)
{
        enum { SW = 4, SH = 4, DW = 2, DH = 2 };
        uint16_t src[SW * SH * 4];
        for (int y = 0; y < SH; y++) {
                for (int x = 0; x < SW; x++) {
                        const int even = ((x ^ y) & 1) == 0;
                        const int idx = (y * SW + x) * 4;
                        src[idx + 0] = even ? 65535 : 0;       /* R */
                        src[idx + 1] = 0;                       /* G */
                        src[idx + 2] = even ? 0 : 65535;       /* B */
                        src[idx + 3] = 65535;
                }
        }

        uint16_t *dst = overlay_scale_rgba16(src, SW, SH, DW, DH);
        ASSERT_MESSAGE("not NULL", dst != NULL);
        for (int i = 0; i < DW * DH; i++) {
                /* Each channel should be in the mixed range, not pure
                 * 0 or 65535. The exact value depends on the libswscale
                 * filter taps; we just assert "neither original colour". */
                ASSERT_MESSAGE("R between extremes",
                               dst[i*4 + 0] > 0 && dst[i*4 + 0] < 65535);
                ASSERT_MESSAGE("B between extremes",
                               dst[i*4 + 2] > 0 && dst[i*4 + 2] < 65535);
                ASSERT_EQUAL_MESSAGE("A still max", 65535, dst[i*4 + 3]);
        }
        free(dst);
        return 0;
}

static int overlay_scale_test_returns_null_on_bad_dims(void)
{
        enum { W = 4, H = 4 };
        uint16_t src[W * H * 4];
        fill_solid_red(src, W, H);

        ASSERT_MESSAGE("zero dst_w",  overlay_scale_rgba16(src, W, H, 0,  4) == NULL);
        ASSERT_MESSAGE("zero dst_h",  overlay_scale_rgba16(src, W, H, 4,  0) == NULL);
        ASSERT_MESSAGE("neg dst_w",   overlay_scale_rgba16(src, W, H, -1, 4) == NULL);
        ASSERT_MESSAGE("zero src_w",  overlay_scale_rgba16(src, 0, H, 4, 4) == NULL);
        ASSERT_MESSAGE("NULL src",    overlay_scale_rgba16(NULL, W, H, 4, 4) == NULL);
        return 0;
}

static int overlay_scale_test_source_buffer_unchanged(void)
{
        enum { W = 4, H = 4 };
        uint16_t src[W * H * 4];
        fill_solid_red(src, W, H);
        uint16_t orig[W * H * 4];
        memcpy(orig, src, sizeof src);

        uint16_t *dst = overlay_scale_rgba16(src, W, H, 16, 16);
        ASSERT_MESSAGE("not NULL", dst != NULL);
        for (int i = 0; i < W * H * 4; i++) {
                ASSERT_EQUAL_MESSAGE("src untouched", orig[i], src[i]);
        }
        free(dst);
        return 0;
}

/*
 * The opaque scaler holds a cached SwsContext across calls. The cache
 * effect itself isn't observable from the API surface — output is always
 * correct regardless — so these tests prove the contract of the new API
 * (create/destroy, dimensions can change between calls) and rely on the
 * implementation review for the cache hit.
 */

static int overlay_scaler_test_create_destroy(void)
{
        struct overlay_scaler *s = overlay_scaler_create(OVERLAY_SCALE_LANCZOS);
        ASSERT_MESSAGE("create non-NULL", s != NULL);
        overlay_scaler_destroy(s);
        /* destroy(NULL) is a documented no-op so callers don't need
         * to guard their cleanup paths. */
        overlay_scaler_destroy(NULL);
        return 0;
}

static int overlay_scaler_test_reuses_context_same_dims(void)
{
        enum { W = 4, H = 4, DW = 16, DH = 16 };
        uint16_t src[W * H * 4];
        fill_solid_red(src, W, H);

        struct overlay_scaler *sc = overlay_scaler_create(OVERLAY_SCALE_LANCZOS);
        ASSERT_MESSAGE("create", sc != NULL);

        uint16_t *out1 = overlay_scaler_scale(sc, src, W, H, DW, DH);
        uint16_t *out2 = overlay_scaler_scale(sc, src, W, H, DW, DH);
        ASSERT_MESSAGE("call 1 ok", out1 != NULL);
        ASSERT_MESSAGE("call 2 ok", out2 != NULL);
        for (int i = 0; i < DW * DH * 4; i++) {
                ASSERT_EQUAL_MESSAGE("identical output", out1[i], out2[i]);
        }
        free(out1); free(out2);
        overlay_scaler_destroy(sc);
        return 0;
}

static int overlay_scaler_test_rebuilds_context_on_dim_change(void)
{
        enum { W = 4, H = 4 };
        uint16_t src[W * H * 4];
        fill_solid_red(src, W, H);

        struct overlay_scaler *sc = overlay_scaler_create(OVERLAY_SCALE_LANCZOS);
        uint16_t *small = overlay_scaler_scale(sc, src, W, H, 8, 8);
        uint16_t *big   = overlay_scaler_scale(sc, src, W, H, 16, 16);
        ASSERT_MESSAGE("8x8 ok",   small != NULL);
        ASSERT_MESSAGE("16x16 ok", big   != NULL);
        const int TOL = 64;
        for (int i = 0; i < 8 * 8 * 4; i += 4) {
                ASSERT_MESSAGE("8x8 R near max",   small[i + 0] >= 65535 - TOL);
                ASSERT_MESSAGE("8x8 G near 0",     small[i + 1] <= TOL);
        }
        for (int i = 0; i < 16 * 16 * 4; i += 4) {
                ASSERT_MESSAGE("16x16 R near max", big[i + 0] >= 65535 - TOL);
                ASSERT_MESSAGE("16x16 G near 0",   big[i + 1] <= TOL);
        }
        free(small); free(big);
        overlay_scaler_destroy(sc);
        return 0;
}

/* scale_into writes through caller-provided dst buffer; no allocation
 * happens on the scaler side, so the same dst pointer is filled by
 * successive calls. */
static int overlay_scaler_test_scale_into_no_alloc(void)
{
        enum { W = 4, H = 4, DW = 16, DH = 16 };
        uint16_t src[W * H * 4];
        fill_solid_red(src, W, H);

        uint16_t *dst = malloc(DW * DH * 4 * sizeof *dst);
        ASSERT_MESSAGE("dst allocated", dst != NULL);
        memset(dst, 0xAA, DW * DH * 4 * sizeof *dst);

        struct overlay_scaler *sc = overlay_scaler_create(OVERLAY_SCALE_LANCZOS);
        ASSERT_MESSAGE("call ok", overlay_scaler_scale_into(sc, dst, src, W, H, DW, DH));

        /* Output must have replaced the 0xAA pattern with red. */
        const int TOL = 64;
        for (int i = 0; i < DW * DH; i++) {
                ASSERT_MESSAGE("R near max",   dst[i*4 + 0] >= 65535 - TOL);
                ASSERT_MESSAGE("G near 0",     dst[i*4 + 1] <= TOL);
                ASSERT_MESSAGE("B near 0",     dst[i*4 + 2] <= TOL);
                ASSERT_MESSAGE("A near max",   dst[i*4 + 3] >= 65535 - TOL);
        }

        /* Second call into the same buffer with new content still works. */
        for (int i = 0; i < W * H; i++) {
                src[i*4 + 0] = 0;
                src[i*4 + 1] = 65535;  /* now green */
                src[i*4 + 2] = 0;
                src[i*4 + 3] = 65535;
        }
        ASSERT_MESSAGE("second call ok",
                       overlay_scaler_scale_into(sc, dst, src, W, H, DW, DH));
        for (int i = 0; i < DW * DH; i++) {
                ASSERT_MESSAGE("R near 0",   dst[i*4 + 0] <= TOL);
                ASSERT_MESSAGE("G near max", dst[i*4 + 1] >= 65535 - TOL);
        }

        free(dst);
        overlay_scaler_destroy(sc);
        return 0;
}

/* Nearest-neighbour upscale of a 2x2 distinct-colour block must produce
 * sharp 8x8 output (no inter-pixel mixing): the top-left quadrant is
 * pure red, top-right pure green, etc. Bilinear/Lanczos would smear. */
static int overlay_scaler_test_filter_nearest(void)
{
        enum { W = 2, H = 2, DW = 8, DH = 8 };
        uint16_t src[W * H * 4] = {
                65535,     0,     0, 65535,   /* (0,0) red */
                    0, 65535,     0, 65535,   /* (1,0) green */
                    0,     0, 65535, 65535,   /* (0,1) blue */
                65535, 65535, 65535, 65535,   /* (1,1) white */
        };

        struct overlay_scaler *sc = overlay_scaler_create(OVERLAY_SCALE_NEAREST);
        uint16_t *dst = overlay_scaler_scale(sc, src, W, H, DW, DH);
        ASSERT_MESSAGE("call ok", dst != NULL);

        /* SWS_POINT preserves the per-pixel source colour but adds a few
         * LSB of rounding at component boundaries — much sharper than
         * bilinear/lanczos but not bit-exact. Tolerance reflects that. */
        const int TOL = 64;
        ASSERT_MESSAGE("(0,0) R near max",   dst[0] >= 65535 - TOL);
        ASSERT_MESSAGE("(0,0) G near 0",     dst[1] <= TOL);
        ASSERT_MESSAGE("(0,0) B near 0",     dst[2] <= TOL);
        ASSERT_MESSAGE("(DW-1,0) R near 0",   dst[(DW - 1) * 4 + 0] <= TOL);
        ASSERT_MESSAGE("(DW-1,0) G near max", dst[(DW - 1) * 4 + 1] >= 65535 - TOL);

        free(dst);
        overlay_scaler_destroy(sc);
        return 0;
}

/* Bilinear upscale of solid-colour input still gives that solid colour
 * (within filter tolerance). Just proves the bilinear path works. */
static int overlay_scaler_test_filter_bilinear(void)
{
        enum { W = 4, H = 4, DW = 16, DH = 16 };
        uint16_t src[W * H * 4];
        fill_solid_red(src, W, H);

        struct overlay_scaler *sc = overlay_scaler_create(OVERLAY_SCALE_BILINEAR);
        uint16_t *dst = overlay_scaler_scale(sc, src, W, H, DW, DH);
        ASSERT_MESSAGE("call ok", dst != NULL);

        const int TOL = 64;
        for (int i = 0; i < DW * DH; i++) {
                ASSERT_MESSAGE("R near max", dst[i*4 + 0] >= 65535 - TOL);
                ASSERT_MESSAGE("G near 0",   dst[i*4 + 1] <= TOL);
        }
        free(dst);
        overlay_scaler_destroy(sc);
        return 0;
}

/*
 * ---- test_overlay_soft_edge (formerly test/test_overlay_soft_edge.c) ----
 */

/* Fill an opaque-white WxH RGBA16 buffer (alpha = 65535). */
static void
fill_opaque_white(uint16_t *buf, int w, int h)
{
        for (int i = 0; i < w * h; i++) {
                buf[i*4 + 0] = 65535;
                buf[i*4 + 1] = 65535;
                buf[i*4 + 2] = 65535;
                buf[i*4 + 3] = 65535;
        }
}

/* edge_w=0: function leaves the buffer untouched. */
static int overlay_soft_edge_test_zero_width_is_noop(void)
{
        enum { W = 8, H = 8 };
        uint16_t buf[W * H * 4];
        fill_opaque_white(buf, W, H);
        uint16_t orig[W * H * 4];
        memcpy(orig, buf, sizeof buf);

        overlay_apply_soft_edge(buf, W, H, 0);
        for (int i = 0; i < W * H * 4; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], buf[i]);
        }
        return 0;
}

/* The outermost row/column has distance 0 from an edge: alpha -> 0. */
static int overlay_soft_edge_test_edge_pixel_zeroed(void)
{
        enum { W = 8, H = 8 };
        uint16_t buf[W * H * 4];
        fill_opaque_white(buf, W, H);

        overlay_apply_soft_edge(buf, W, H, 3);

        ASSERT_EQUAL_MESSAGE("top-left",     0, buf[(0  * W + 0)     * 4 + 3]);
        ASSERT_EQUAL_MESSAGE("top-right",    0, buf[(0  * W + W - 1) * 4 + 3]);
        ASSERT_EQUAL_MESSAGE("bot-left",     0, buf[((H - 1) * W + 0)     * 4 + 3]);
        ASSERT_EQUAL_MESSAGE("bot-right",    0, buf[((H - 1) * W + W - 1) * 4 + 3]);
        ASSERT_EQUAL_MESSAGE("top edge mid", 0, buf[(0     * W + W / 2) * 4 + 3]);
        return 0;
}

/* For edge_w=4 on a wide-enough buffer, alpha at distance d from the nearest
 * edge is round(65535 * d / edge_w) for d in 1..edge_w-1, clamped at 65535. */
static int overlay_soft_edge_test_linear_ramp(void)
{
        enum { W = 16, H = 16, EDGE = 4 };
        uint16_t buf[W * H * 4];
        fill_opaque_white(buf, W, H);

        overlay_apply_soft_edge(buf, W, H, EDGE);

        /* Row of pixels stepping inward from the left edge, on a row that's
         * far from the top/bottom (so vertical distance is >= EDGE). */
        const int y = H / 2;
        const int expected[] = { 0, 16383, 32767, 49151, 65535, 65535 };
        for (int x = 0; x < (int)(sizeof expected / sizeof expected[0]); x++) {
                char msg[32];
                snprintf(msg, sizeof msg, "x=%d", x);
                ASSERT_EQUAL_MESSAGE(msg, expected[x],
                                     buf[(y * W + x) * 4 + 3]);
        }
        return 0;
}

/* Pixels at distance >= edge_w from every edge keep alpha = 65535. */
static int overlay_soft_edge_test_centre_untouched(void)
{
        enum { W = 20, H = 20, EDGE = 4 };
        uint16_t buf[W * H * 4];
        fill_opaque_white(buf, W, H);

        overlay_apply_soft_edge(buf, W, H, EDGE);

        for (int y = EDGE; y < H - EDGE; y++) {
                for (int x = EDGE; x < W - EDGE; x++) {
                        ASSERT_EQUAL_MESSAGE("centre alpha unchanged",
                                             65535, buf[(y * W + x) * 4 + 3]);
                }
        }
        return 0;
}

/* Only alpha is modified — RGB components stay put. */
static int overlay_soft_edge_test_rgb_components_unchanged(void)
{
        enum { W = 8, H = 8 };
        uint16_t buf[W * H * 4];
        for (int i = 0; i < W * H; i++) {
                buf[i*4 + 0] = (uint16_t)(1000 + i);
                buf[i*4 + 1] = (uint16_t)(2000 + i);
                buf[i*4 + 2] = (uint16_t)(3000 + i);
                buf[i*4 + 3] = 65535;
        }

        overlay_apply_soft_edge(buf, W, H, 3);

        for (int i = 0; i < W * H; i++) {
                ASSERT_EQUAL_MESSAGE("R untouched", (uint16_t)(1000 + i), buf[i*4 + 0]);
                ASSERT_EQUAL_MESSAGE("G untouched", (uint16_t)(2000 + i), buf[i*4 + 1]);
                ASSERT_EQUAL_MESSAGE("B untouched", (uint16_t)(3000 + i), buf[i*4 + 2]);
        }
        return 0;
}

/* edge_w larger than min(W,H)/2 must not over-attenuate the centre below
 * what it would be at edge_w = min(W,H)/2 (centre stays > 0). */
static int overlay_soft_edge_test_oversized_width_clamps(void)
{
        enum { W = 4, H = 4 };
        uint16_t buf[W * H * 4];
        fill_opaque_white(buf, W, H);

        /* edge_w=100 on a 4x4 buffer would over-shoot if no clamp. */
        overlay_apply_soft_edge(buf, W, H, 100);

        /* The four corner-adjacent inner pixels (x=1,y=1 etc) have distance
         * 1 from the nearest edge, so they should be > 0 (not negative or
         * wrapped). */
        ASSERT_MESSAGE("inner pixel > 0",     buf[(1 * W + 1) * 4 + 3] > 0);
        ASSERT_MESSAGE("inner pixel <= max",  buf[(1 * W + 1) * 4 + 3] <= 65535);
        return 0;
}

/* Non-square buffer: clamp must be on the SHORT axis, not just width. A
 * 4x20 buffer with edge_w=10 must clamp to MIN(4,20)/2 = 2, not 10. */
static int overlay_soft_edge_test_non_square(void)
{
        enum { W = 4, H = 20 };
        uint16_t buf[W * H * 4];
        fill_opaque_white(buf, W, H);

        overlay_apply_soft_edge(buf, W, H, 10);

        /* Centre column (x=2) at y=10 has dx=1 (clamped edge_w=2) so
         * alpha = 65535*1/2 = 32767. If clamp had used width instead of
         * MIN(w,h) we'd see 65535*1/4 = 16383. */
        ASSERT_EQUAL_MESSAGE("centre col mid-row",
                             32767, buf[(10 * W + 2) * 4 + 3]);
        return 0;
}

/* edge_w exactly at the clamp boundary (= min(w,h)/2): the clamp is a
 * no-op, the deepest interior pixel sits at d = edge_w - 1, and the
 * d == edge_w branch is unreachable. */
static int overlay_soft_edge_test_exact_half_dimension(void)
{
        enum { W = 8, H = 8, EDGE = 4 };
        uint16_t buf[W * H * 4];
        fill_opaque_white(buf, W, H);

        overlay_apply_soft_edge(buf, W, H, EDGE);

        /* (3,3) and (4,4) are the deepest pixels: dx=3, dy=3, d=3. */
        ASSERT_EQUAL_MESSAGE("(3,3) deepest", 49151,
                             buf[(3 * W + 3) * 4 + 3]);
        ASSERT_EQUAL_MESSAGE("(4,4) deepest", 49151,
                             buf[(4 * W + 4) * 4 + 3]);
        return 0;
}

/* Already-translucent input: the ramp must SCALE existing alpha, not
 * clobber it to a fixed ramp value. alpha=32768 with d=1, edge_w=4 gives
 * floor(32768 * 1 / 4) = 8192 — not 16384 (which is what 65535*1/4 / 2
 * would give if the function rebuilt alpha from scratch). */
static int overlay_soft_edge_test_scales_existing_alpha(void)
{
        enum { W = 8, H = 8, EDGE = 4 };
        uint16_t buf[W * H * 4];
        for (int i = 0; i < W * H; i++) {
                buf[i*4 + 0] = 0;
                buf[i*4 + 1] = 0;
                buf[i*4 + 2] = 0;
                buf[i*4 + 3] = 32768;
        }

        overlay_apply_soft_edge(buf, W, H, EDGE);

        ASSERT_EQUAL_MESSAGE("d=0 -> 0",   0,    buf[(W/2 * W + 0) * 4 + 3]);
        ASSERT_EQUAL_MESSAGE("d=1 -> 8192", 8192, buf[(W/2 * W + 1) * 4 + 3]);
        ASSERT_EQUAL_MESSAGE("d=2 -> 16384", 16384, buf[(W/2 * W + 2) * 4 + 3]);
        ASSERT_EQUAL_MESSAGE("d=3 -> 24576", 24576, buf[(W/2 * W + 3) * 4 + 3]);
        return 0;
}

/* 1xN buffer: max_edge = MIN(1, N)/2 = 0, so the function returns without
 * touching the buffer (and without dividing by zero). */
static int overlay_soft_edge_test_degenerate_one_row(void)
{
        enum { W = 8, H = 1 };
        uint16_t buf[W * H * 4];
        fill_opaque_white(buf, W, H);
        uint16_t orig[W * H * 4];
        memcpy(orig, buf, sizeof buf);

        overlay_apply_soft_edge(buf, W, H, 4);
        for (int i = 0; i < W * H * 4; i++) {
                ASSERT_EQUAL_MESSAGE("byte unchanged", orig[i], buf[i]);
        }
        return 0;
}

/*
 * ---- test_overlay_watch (formerly test/test_overlay_watch.c) ----
 */

static bool write_file(const char **path_out, const char *body, size_t len)
{
        FILE *f = get_temp_file(path_out);
        if (!f) return false;
        bool ok = fwrite(body, 1, len, f) == len;
        if (fclose(f) != 0) ok = false;
        if (!ok) unlink(*path_out);
        return ok;
}

static bool overwrite_file(const char *path, const char *body, size_t len)
{
        FILE *f = fopen(path, "wb");
        if (!f) return false;
        bool ok = fwrite(body, 1, len, f) == len;
        if (fclose(f) != 0) ok = false;
        return ok;
}

/* Stable file: init then immediate poll reports no change. */
static int overlay_watch_test_init_no_change(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("not changed", !overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* File grows -> change detected (size differs even if mtime resolution is coarse). */
static int overlay_watch_test_detects_size_change(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("rewrite bigger",
                       overwrite_file(path, "hello world", 11));
        ASSERT_MESSAGE("changed", overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* Same size, different mtime -> change detected. Force the baseline mtime
 * into the past (Windows has neither utimes() nor utimensat(), so use the
 * CRT _utime() there; 1-s resolution suffices for a backdate to 1000000). */
static int overlay_watch_test_detects_mtime_change(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);

#ifdef _WIN32
        struct _utimbuf tb;
        tb.actime = 1000000;   /* atime */
        tb.modtime = 1000000;  /* mtime: well in the past */
        ASSERT_EQUAL_MESSAGE("_utime", 0, _utime(path, &tb));
#else
        struct timeval tv[2];
        tv[0].tv_sec = 1000000; tv[0].tv_usec = 0;  /* atime */
        tv[1].tv_sec = 1000000; tv[1].tv_usec = 0;  /* mtime: well in the past */
        ASSERT_EQUAL_MESSAGE("utimes", 0, utimes(path, tv));
#endif

        ASSERT_MESSAGE("changed", overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* Acking against a missing file is a no-op: baseline survives so the next
 * appearance still triggers reload. Avoids dropping a real edit if the file
 * is briefly absent (e.g. atomic-write rename) at ack time. */
static int overlay_watch_test_ack_on_missing_file_preserves_baseline(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);

        ASSERT_EQUAL_MESSAGE("unlink", 0, unlink(path));
        overlay_watch_ack(&w, path);

        ASSERT_MESSAGE("recreate",
                       overwrite_file(path, "different content", 17));
        ASSERT_MESSAGE("appearance still triggers",
                       overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/*
 * The baseline must NOT advance until the caller explicitly acknowledges
 * the change (after a successful reload). Otherwise a failed reload would
 * eat the trigger and a later valid mutation would go undetected.
 */
static int overlay_watch_test_changed_does_not_consume_baseline(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("rewrite #1",
                       overwrite_file(path, "broken!!!", 9));

        /* First poll: change detected. Caller's reload (simulated) fails,
         * so it does NOT call overlay_watch_ack. */
        ASSERT_MESSAGE("first poll changed",
                       overlay_watch_changed(&w, path));

        /* Second poll without ack: still reports change so the caller can
         * retry, even though mtime/size haven't moved further. */
        ASSERT_MESSAGE("second poll still changed (not acked)",
                       overlay_watch_changed(&w, path));

        /* And after a fresh mutation, change is still reported. */
        ASSERT_MESSAGE("rewrite #2",
                       overwrite_file(path, "fixed content version 2", 23));
        ASSERT_MESSAGE("after mutation: changed",
                       overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

static int overlay_watch_test_ack_commits_baseline(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "hello", 5));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("rewrite",
                       overwrite_file(path, "version 2 here", 14));
        ASSERT_MESSAGE("changed", overlay_watch_changed(&w, path));

        overlay_watch_ack(&w, path);
        ASSERT_MESSAGE("after ack: quiet",
                       !overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* Missing file (transiently absent during atomic-write) must not trigger a
 * spurious reload — caller would crash trying to load nothing. */
static int overlay_watch_test_missing_file_no_change(void)
{
        struct overlay_watch w;
        overlay_watch_init(&w, "/nonexistent/path/no.pam");
        ASSERT_MESSAGE("missing file: no change",
                       !overlay_watch_changed(&w, "/nonexistent/path/no.pam"));
        return 0;
}

/* File initially missing, then created -> first poll after creation reports
 * change (so the caller's reload kicks in once the file appears). */
static int overlay_watch_test_file_appears_after_init(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write+remove", write_file(&path, "x", 1));
        ASSERT_EQUAL_MESSAGE("unlink", 0, unlink(path));

        struct overlay_watch w;
        overlay_watch_init(&w, path);
        ASSERT_MESSAGE("missing while absent",
                       !overlay_watch_changed(&w, path));

        ASSERT_MESSAGE("recreate", overwrite_file(path, "appeared", 8));
        ASSERT_MESSAGE("change reported on appearance",
                       overlay_watch_changed(&w, path));
        overlay_watch_ack(&w, path);
        ASSERT_MESSAGE("then quiet",
                       !overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/* Atomic-write replacement: write a sibling temp file then rename() over the
 * watched path. The new inode has a different mtime/size and the watcher
 * must detect it. */
static int overlay_watch_test_detects_atomic_rename(void)
{
        const char *path = NULL;
        ASSERT_MESSAGE("write file", write_file(&path, "v1", 2));

        struct overlay_watch w;
        overlay_watch_init(&w, path);

        const char *staging = NULL;
        ASSERT_MESSAGE("write staging", write_file(&staging, "version2", 8));
        ASSERT_EQUAL_MESSAGE("rename", 0, rename(staging, path));

        ASSERT_MESSAGE("change detected after rename",
                       overlay_watch_changed(&w, path));

        unlink(path);
        return 0;
}

/*
 * ---- test_overlay_config (formerly test/test_overlay_config.c) ----
 */

/* file=PATH alone: position defaults to center; help is false. */
static int overlay_config_test_minimal_file_only(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse",
                       overlay_config_parse("file=/tmp/x.pam", &c));
        ASSERT_MESSAGE("file matches",
                       strcmp(c.file, "/tmp/x.pam") == 0);
        ASSERT_EQUAL_MESSAGE("position default",
                             OVERLAY_POS_CENTER, (int)c.position);
        ASSERT_MESSAGE("not help", !c.help);
        return 0;
}

/* All five preset position keywords map to their enum values. */
static int overlay_config_test_position_keywords(void)
{
        struct overlay_config c;
        const struct {
                const char *kw;
                enum overlay_position pos;
        } cases[] = {
                {"center",       OVERLAY_POS_CENTER},
                {"top_left",     OVERLAY_POS_TOP_LEFT},
                {"top_right",    OVERLAY_POS_TOP_RIGHT},
                {"bottom_left",  OVERLAY_POS_BOTTOM_LEFT},
                {"bottom_right", OVERLAY_POS_BOTTOM_RIGHT},
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
                char buf[64];
                snprintf(buf, sizeof buf, "file=a.pam:position=%s",
                         cases[i].kw);
                ASSERT_MESSAGE(cases[i].kw, overlay_config_parse(buf, &c));
                ASSERT_EQUAL_MESSAGE(cases[i].kw, (int)cases[i].pos,
                                     (int)c.position);
        }
        return 0;
}

/* custom_x / custom_y promote position to OVERLAY_POS_CUSTOM and parse signed
 * integers (negative = from-edge). */
static int overlay_config_test_custom_xy(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse",
                       overlay_config_parse(
                               "file=a.pam:custom_x=10:custom_y=-20", &c));
        ASSERT_EQUAL_MESSAGE("position", OVERLAY_POS_CUSTOM, (int)c.position);
        ASSERT_EQUAL_MESSAGE("x",  10, c.custom_x);
        ASSERT_EQUAL_MESSAGE("y", -20, c.custom_y);
        return 0;
}

/* "help" alone is allowed with no file (the postprocessor prints usage). */
static int overlay_config_test_help(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse help", overlay_config_parse("help", &c));
        ASSERT_MESSAGE("help flag", c.help);
        return 0;
}

static int overlay_config_test_rejects_missing_file(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("no file: reject",
                       !overlay_config_parse("position=center", &c));
        return 0;
}

static int overlay_config_test_rejects_unknown_key(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("typo: reject",
                       !overlay_config_parse("file=a.pam:positon=center", &c));
        return 0;
}

static int overlay_config_test_rejects_bad_position(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("middle: reject",
                       !overlay_config_parse("file=a.pam:position=middle", &c));
        return 0;
}

static int overlay_config_test_rejects_non_integer_xy(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("alpha x: reject",
                       !overlay_config_parse("file=a.pam:custom_x=abc", &c));
        ASSERT_MESSAGE("trailing junk: reject",
                       !overlay_config_parse("file=a.pam:custom_x=10x", &c));
        return 0;
}

static int overlay_config_test_rejects_null_and_empty_value(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("NULL opts: reject", !overlay_config_parse(NULL, &c));
        ASSERT_MESSAGE("empty position value: reject",
                       !overlay_config_parse("file=a.pam:position=", &c));
        ASSERT_MESSAGE("empty file value: reject",
                       !overlay_config_parse("file=", &c));
        return 0;
}

static int overlay_config_test_soft_edge(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse",
                       overlay_config_parse("file=a.pam:soft_edge=12", &c));
        ASSERT_EQUAL_MESSAGE("soft_edge", 12, c.soft_edge);

        ASSERT_MESSAGE("default zero",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_EQUAL_MESSAGE("default", 0, c.soft_edge);

        ASSERT_MESSAGE("negative rejected",
                       !overlay_config_parse("file=a.pam:soft_edge=-3", &c));
        return 0;
}

static int overlay_config_test_scale(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse 320x240",
                       overlay_config_parse("file=a.pam:scale=320x240", &c));
        ASSERT_EQUAL_MESSAGE("scale_w", 320, c.scale_w);
        ASSERT_EQUAL_MESSAGE("scale_h", 240, c.scale_h);

        ASSERT_MESSAGE("default zero",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_EQUAL_MESSAGE("default w", 0, c.scale_w);
        ASSERT_EQUAL_MESSAGE("default h", 0, c.scale_h);

        ASSERT_MESSAGE("missing 'x' rejected",
                       !overlay_config_parse("file=a.pam:scale=320", &c));
        ASSERT_MESSAGE("zero rejected",
                       !overlay_config_parse("file=a.pam:scale=0x240", &c));
        ASSERT_MESSAGE("negative rejected",
                       !overlay_config_parse("file=a.pam:scale=-10x240", &c));
        return 0;
}

/* scale=frame parses, sets scale_to_frame=true, and zeroes the
 * numeric scale dimensions. */
static int overlay_config_test_scale_frame(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse scale=frame",
                       overlay_config_parse("file=a.pam:scale=frame", &c));
        ASSERT_MESSAGE("scale_to_frame set",       c.scale_to_frame);
        ASSERT_EQUAL_MESSAGE("scale_w zero", 0, c.scale_w);
        ASSERT_EQUAL_MESSAGE("scale_h zero", 0, c.scale_h);

        ASSERT_MESSAGE("default off",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_MESSAGE("scale_to_frame default false", !c.scale_to_frame);
        return 0;
}

/* Last-one-wins between scale=WxH and scale=frame. */
static int overlay_config_test_scale_frame_overrides_wxh(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("frame after WxH",
                       overlay_config_parse(
                               "file=a.pam:scale=320x240:scale=frame", &c));
        ASSERT_MESSAGE("scale_to_frame set",       c.scale_to_frame);
        ASSERT_EQUAL_MESSAGE("scale_w cleared", 0, c.scale_w);
        ASSERT_EQUAL_MESSAGE("scale_h cleared", 0, c.scale_h);

        ASSERT_MESSAGE("WxH after frame",
                       overlay_config_parse(
                               "file=a.pam:scale=frame:scale=320x240", &c));
        ASSERT_MESSAGE("scale_to_frame cleared",     !c.scale_to_frame);
        ASSERT_EQUAL_MESSAGE("scale_w set",   320, c.scale_w);
        ASSERT_EQUAL_MESSAGE("scale_h set",   240, c.scale_h);
        return 0;
}

static int overlay_config_test_perf(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("parse perf flag",
                       overlay_config_parse("file=a.pam:perf", &c));
        ASSERT_MESSAGE("perf set", c.perf);

        ASSERT_MESSAGE("default off",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_MESSAGE("perf default false", !c.perf);
        return 0;
}

static int overlay_config_test_scale_filter(void)
{
        struct overlay_config c;
        ASSERT_MESSAGE("default bicubic",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_EQUAL_MESSAGE("default", OVERLAY_SCALE_BICUBIC, (int)c.scale_filter);

        const struct {
                const char *name;
                enum overlay_scale_filter expected;
        } cases[] = {
                {"nearest",       OVERLAY_SCALE_NEAREST},
                {"fast_bilinear", OVERLAY_SCALE_FAST_BILINEAR},
                {"bilinear",      OVERLAY_SCALE_BILINEAR},
                {"bicubic",       OVERLAY_SCALE_BICUBIC},
                {"lanczos",       OVERLAY_SCALE_LANCZOS},
        };
        for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
                char buf[64];
                snprintf(buf, sizeof buf, "file=a.pam:scale_filter=%s",
                         cases[i].name);
                ASSERT_MESSAGE(cases[i].name, overlay_config_parse(buf, &c));
                ASSERT_EQUAL_MESSAGE(cases[i].name, (int)cases[i].expected,
                                     (int)c.scale_filter);
        }
        ASSERT_MESSAGE("unknown filter rejected",
                       !overlay_config_parse("file=a.pam:scale_filter=bogus", &c));
        return 0;
}

static int overlay_config_test_blend_threads(void)
{
        struct overlay_config c;

        /* Unset -> auto-default of min(ncpu, 8). On any host that's >= 1. */
        ASSERT_MESSAGE("default auto",
                       overlay_config_parse("file=a.pam", &c));
        ASSERT_MESSAGE("auto >= 1", c.blend_threads >= 1);
        ASSERT_MESSAGE("auto <= 8", c.blend_threads <= 8);

        ASSERT_MESSAGE("explicit 4",
                       overlay_config_parse("file=a.pam:blend_threads=4", &c));
        ASSERT_EQUAL_MESSAGE("4", 4, c.blend_threads);

        /* User who wants single-threaded passes 1 explicitly. */
        ASSERT_MESSAGE("explicit 1",
                       overlay_config_parse("file=a.pam:blend_threads=1", &c));
        ASSERT_EQUAL_MESSAGE("1", 1, c.blend_threads);

        ASSERT_MESSAGE("negative rejected",
                       !overlay_config_parse("file=a.pam:blend_threads=-1", &c));
        return 0;
}

static int overlay_config_test_rejects_oversize_options(void)
{
        char giant[MAX_PATH_SIZE + 512];
        memset(giant, 'a', sizeof giant - 1);
        giant[sizeof giant - 1] = '\0';
        memcpy(giant, "file=", 5);

        struct overlay_config c;
        ASSERT_MESSAGE("options too long: reject",
                       !overlay_config_parse(giant, &c));
        return 0;
}

#define DEFINE_SUBTEST(func) { #func, func }

static const struct {
        const char *name;
        int (*test)(void);
} subtests[] = {
        DEFINE_SUBTEST(alpha_blend_test_rgba_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_rgba_alpha_max),
        DEFINE_SUBTEST(alpha_blend_test_rgba_half_alpha),
        DEFINE_SUBTEST(alpha_blend_test_rgb_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_rgb_alpha_max),
        DEFINE_SUBTEST(alpha_blend_test_uyvy_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_uyvy_alpha_max_white),
        DEFINE_SUBTEST(alpha_blend_test_uyvy_alpha_max_red),
        DEFINE_SUBTEST(alpha_blend_test_yuyv_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_yuyv_alpha_max_white),
        DEFINE_SUBTEST(alpha_blend_test_yuyv_alpha_max_red),
        DEFINE_SUBTEST(alpha_blend_test_y416_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_y416_alpha_max_white),
        DEFINE_SUBTEST(alpha_blend_test_i420_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_i420_alpha_max_white),
        DEFINE_SUBTEST(alpha_blend_test_i420_chroma_alpha_averaging),
        DEFINE_SUBTEST(alpha_blend_test_i420_subregion_strides),
        DEFINE_SUBTEST(alpha_blend_test_rg48_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_rg48_alpha_max_white),
        DEFINE_SUBTEST(alpha_blend_test_v210_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_v210_alpha_max_white),
        DEFINE_SUBTEST(alpha_blend_test_r10k_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_r10k_alpha_max_white),
        DEFINE_SUBTEST(alpha_blend_test_r12l_alpha_zero),
        DEFINE_SUBTEST(alpha_blend_test_r12l_alpha_max_white),
        DEFINE_SUBTEST(overlay_pam_test_load_8bit_rgba),
        DEFINE_SUBTEST(overlay_pam_test_load_8bit_rgb_adds_alpha),
        DEFINE_SUBTEST(overlay_pam_test_load_16bit_rgba),
        DEFINE_SUBTEST(overlay_pam_test_rejects_missing_file),
        DEFINE_SUBTEST(overlay_pam_test_rejects_grayscale),
        DEFINE_SUBTEST(overlay_pam_test_rejects_intermediate_maxval),
        DEFINE_SUBTEST(overlay_layout_test_center),
        DEFINE_SUBTEST(overlay_layout_test_corners),
        DEFINE_SUBTEST(overlay_layout_test_custom_negative_from_edge),
        DEFINE_SUBTEST(overlay_layout_test_block_pixel_alignment),
        DEFINE_SUBTEST(overlay_layout_test_overlay_larger_than_frame),
        DEFINE_SUBTEST(overlay_layout_test_oversized_center),
        DEFINE_SUBTEST(overlay_layout_test_oversized_right),
        DEFINE_SUBTEST(overlay_layout_test_oversized_custom_positive),
        DEFINE_SUBTEST(overlay_layout_test_block_lines_alignment),
        DEFINE_SUBTEST(overlay_scale_test_identity),
        DEFINE_SUBTEST(overlay_scale_test_upscale_solid_colour),
        DEFINE_SUBTEST(overlay_scale_test_downscale_average),
        DEFINE_SUBTEST(overlay_scale_test_returns_null_on_bad_dims),
        DEFINE_SUBTEST(overlay_scale_test_source_buffer_unchanged),
        DEFINE_SUBTEST(overlay_scaler_test_create_destroy),
        DEFINE_SUBTEST(overlay_scaler_test_reuses_context_same_dims),
        DEFINE_SUBTEST(overlay_scaler_test_rebuilds_context_on_dim_change),
        DEFINE_SUBTEST(overlay_scaler_test_scale_into_no_alloc),
        DEFINE_SUBTEST(overlay_scaler_test_filter_nearest),
        DEFINE_SUBTEST(overlay_scaler_test_filter_bilinear),
        DEFINE_SUBTEST(overlay_soft_edge_test_zero_width_is_noop),
        DEFINE_SUBTEST(overlay_soft_edge_test_edge_pixel_zeroed),
        DEFINE_SUBTEST(overlay_soft_edge_test_linear_ramp),
        DEFINE_SUBTEST(overlay_soft_edge_test_centre_untouched),
        DEFINE_SUBTEST(overlay_soft_edge_test_rgb_components_unchanged),
        DEFINE_SUBTEST(overlay_soft_edge_test_oversized_width_clamps),
        DEFINE_SUBTEST(overlay_soft_edge_test_non_square),
        DEFINE_SUBTEST(overlay_soft_edge_test_exact_half_dimension),
        DEFINE_SUBTEST(overlay_soft_edge_test_scales_existing_alpha),
        DEFINE_SUBTEST(overlay_soft_edge_test_degenerate_one_row),
        DEFINE_SUBTEST(overlay_watch_test_init_no_change),
        DEFINE_SUBTEST(overlay_watch_test_detects_size_change),
        DEFINE_SUBTEST(overlay_watch_test_detects_mtime_change),
        DEFINE_SUBTEST(overlay_watch_test_ack_on_missing_file_preserves_baseline),
        DEFINE_SUBTEST(overlay_watch_test_changed_does_not_consume_baseline),
        DEFINE_SUBTEST(overlay_watch_test_ack_commits_baseline),
        DEFINE_SUBTEST(overlay_watch_test_missing_file_no_change),
        DEFINE_SUBTEST(overlay_watch_test_file_appears_after_init),
        DEFINE_SUBTEST(overlay_watch_test_detects_atomic_rename),
        DEFINE_SUBTEST(overlay_config_test_minimal_file_only),
        DEFINE_SUBTEST(overlay_config_test_position_keywords),
        DEFINE_SUBTEST(overlay_config_test_custom_xy),
        DEFINE_SUBTEST(overlay_config_test_help),
        DEFINE_SUBTEST(overlay_config_test_rejects_missing_file),
        DEFINE_SUBTEST(overlay_config_test_rejects_unknown_key),
        DEFINE_SUBTEST(overlay_config_test_rejects_bad_position),
        DEFINE_SUBTEST(overlay_config_test_rejects_non_integer_xy),
        DEFINE_SUBTEST(overlay_config_test_rejects_null_and_empty_value),
        DEFINE_SUBTEST(overlay_config_test_soft_edge),
        DEFINE_SUBTEST(overlay_config_test_scale),
        DEFINE_SUBTEST(overlay_config_test_scale_frame),
        DEFINE_SUBTEST(overlay_config_test_scale_frame_overrides_wxh),
        DEFINE_SUBTEST(overlay_config_test_perf),
        DEFINE_SUBTEST(overlay_config_test_scale_filter),
        DEFINE_SUBTEST(overlay_config_test_blend_threads),
        DEFINE_SUBTEST(overlay_config_test_rejects_oversize_options),
};

/*
 * Aggregated test case: runs every subtest above. Subtests report their
 * own failures via the ASSERT macros; here we just collect the results.
 * Returns 0 when all subtests pass, negative otherwise.
 */
int overlay_test(void)
{
        if (strcmp(g_test_opts, "help") == 0) {
                printf("Usage: run_tests overlay_test[:help|<subtest>]\n\n"
                       "Runs all subtests, a single subtest selected by name,\n"
                       "or lists the subtests with :help.\n"
                       "Available subtests:\n");
                for (size_t i = 0; i < sizeof subtests / sizeof subtests[0]; ++i) {
                        printf(" - %s\n", subtests[i].name);
                }
                return 1; // help requested, runner reports "--"
        }

        bool single_test = strlen(g_test_opts) > 0;

        int av_ll_orig = av_log_get_level();
        av_log_set_level(AV_LOG_WARNING);
        bool ok = true;
        int nr_run = 0;
        for (size_t i = 0; i < sizeof subtests / sizeof subtests[0]; i++) {
                if (single_test && strcmp(subtests[i].name, g_test_opts) != 0) {
                        continue;
                }
                nr_run += 1;
                int ret = subtests[i].test();
                if (ret != 0) {
                        fprintf(stderr, "overlay subtest %s failed\n",
                                subtests[i].name);
                        ok = false;
                }
        }
        av_log_set_level(av_ll_orig);

        if (!nr_run) {
                fprintf(stderr,
                        "No overlay subtest run! Requested: \"%s\", "
                        "run \"overlay_test:help\" for the list.\n",
                        g_test_opts);
                ok = 0;
        }

        return ok ? 0 : -1;
}

#endif // defined HAVE_OVERLAY
