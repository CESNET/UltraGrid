/**
 * @file   test/test_overlay_soft_edge.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Unit tests for utils/overlay_soft_edge.c
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

#include "test_overlay_soft_edge.h"
#include "unit_common.h"
#include "utils/overlay_soft_edge.h"

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
int overlay_soft_edge_test_zero_width_is_noop(void)
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
int overlay_soft_edge_test_edge_pixel_zeroed(void)
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
int overlay_soft_edge_test_linear_ramp(void)
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
int overlay_soft_edge_test_centre_untouched(void)
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
int overlay_soft_edge_test_rgb_components_unchanged(void)
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
int overlay_soft_edge_test_oversized_width_clamps(void)
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
int overlay_soft_edge_test_non_square(void)
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
int overlay_soft_edge_test_exact_half_dimension(void)
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
int overlay_soft_edge_test_scales_existing_alpha(void)
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
int overlay_soft_edge_test_degenerate_one_row(void)
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
