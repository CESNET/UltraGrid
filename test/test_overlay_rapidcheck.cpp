/**
 * @file   test/test_overlay_rapidcheck.cpp
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Property-based tests for utils/overlay_layout.c.
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

#include <rapidcheck.h>

extern "C" {
#include "utils/overlay_layout.h"
}

namespace {

/* Frame and overlay dimensions in a range that exercises both
 * "overlay fits" and "overlay larger than frame" cases. */
auto gen_frame_dim()   { return rc::gen::inRange(2, 4096); }
auto gen_overlay_dim() { return rc::gen::inRange(1, 4096); }
auto gen_block()       { return rc::gen::elementOf(std::vector<int>{1, 2, 4, 6, 8}); }

auto gen_position()
{
        return rc::gen::elementOf(std::vector<enum overlay_position>{
                OVERLAY_POS_CENTER,
                OVERLAY_POS_TOP_LEFT,
                OVERLAY_POS_TOP_RIGHT,
                OVERLAY_POS_BOTTOM_LEFT,
                OVERLAY_POS_BOTTOM_RIGHT,
        });
}

} // namespace

bool test_overlay_layout_properties()
{
        bool ok = true;

        /* Preset positions never produce a rect that escapes the frame. */
        ok &= rc::check("overlay_calc_rect: preset stays inside frame", []() {
                const int fw = *gen_frame_dim();
                const int fh = *gen_frame_dim();
                const int ow = *gen_overlay_dim();
                const int oh = *gen_overlay_dim();
                const int bp = *gen_block();
                const int bl = *gen_block();
                const auto pos = *gen_position();

                struct overlay_rect r = overlay_calc_rect(
                        pos, 0, 0, fw, fh, ow, oh, bp, bl);

                RC_ASSERT(r.x >= 0);
                RC_ASSERT(r.y >= 0);
                RC_ASSERT(r.width  >= 0);
                RC_ASSERT(r.height >= 0);
                RC_ASSERT(r.x + r.width  <= fw);
                RC_ASSERT(r.y + r.height <= fh);
        });

        /* Block-grid snapping: the rect's x/y/width/height must align to
         * the block grid that the caller declared. */
        ok &= rc::check("overlay_calc_rect: snaps to block grid", []() {
                const int fw = *gen_frame_dim();
                const int fh = *gen_frame_dim();
                const int ow = *gen_overlay_dim();
                const int oh = *gen_overlay_dim();
                const int bp = *gen_block();
                const int bl = *gen_block();
                const auto pos = *gen_position();

                struct overlay_rect r = overlay_calc_rect(
                        pos, 0, 0, fw, fh, ow, oh, bp, bl);

                if (bp > 1) {
                        RC_ASSERT(r.x % bp == 0);
                        RC_ASSERT(r.width % bp == 0);
                }
                if (bl > 1) {
                        RC_ASSERT(r.y % bl == 0);
                        RC_ASSERT(r.height % bl == 0);
                }
        });

        /* Custom position with non-negative offsets: the rect's origin is
         * the snapped offset (or zero if it would push past the frame). */
        ok &= rc::check("overlay_calc_rect: custom positive offset honoured",
                        []() {
                const int fw = *gen_frame_dim();
                const int fh = *gen_frame_dim();
                const int ow = *gen_overlay_dim();
                const int oh = *gen_overlay_dim();
                const int cx = *rc::gen::inRange(0, 4096);
                const int cy = *rc::gen::inRange(0, 4096);

                struct overlay_rect r = overlay_calc_rect(
                        OVERLAY_POS_CUSTOM, cx, cy, fw, fh, ow, oh, 1, 1);

                RC_ASSERT(r.x >= 0 && r.y >= 0);
                RC_ASSERT(r.x + r.width  <= fw);
                RC_ASSERT(r.y + r.height <= fh);
                if (cx < fw) RC_ASSERT(r.x == cx);
                if (cy < fh) RC_ASSERT(r.y == cy);
        });

        /* Negative custom offset counts from the opposite edge. Per
         * overlay_layout.c the rule is x = fw + cx - ow, then clamp to
         * [0, fw]. With block_pixels=block_lines=1 no snapping happens,
         * so we can verify the formula exactly. */
        ok &= rc::check("overlay_calc_rect: negative custom offset from edge",
                        []() {
                const int fw = *gen_frame_dim();
                const int fh = *gen_frame_dim();
                const int ow = *rc::gen::inRange(1, 256);
                const int oh = *rc::gen::inRange(1, 256);
                const int cx = *rc::gen::inRange(-256, 0);
                const int cy = *rc::gen::inRange(-256, 0);

                struct overlay_rect r = overlay_calc_rect(
                        OVERLAY_POS_CUSTOM, cx, cy, fw, fh, ow, oh, 1, 1);

                RC_ASSERT(r.x >= 0 && r.y >= 0);
                RC_ASSERT(r.x + r.width  <= fw);
                RC_ASSERT(r.y + r.height <= fh);

                /* Exact formula: when (fw + cx - ow) is in-range, that's
                 * the placement; otherwise it clamps to 0. */
                const int ideal_x = fw + cx - ow;
                const int ideal_y = fh + cy - oh;
                RC_ASSERT(r.x == (ideal_x > 0 ? ideal_x : 0));
                RC_ASSERT(r.y == (ideal_y > 0 ? ideal_y : 0));
        });

        /* Center positioning is symmetric: x*2 + width is roughly fw
         * (within one block_pixels of slack from the snap). Skip when
         * the rect collapses to zero — block-snap can push a small
         * overlay past zero, in which case "centered" is meaningless. */
        ok &= rc::check("overlay_calc_rect: CENTER places near middle", []() {
                const int fw = *rc::gen::inRange(64, 4096);
                const int fh = *rc::gen::inRange(64, 4096);
                const int ow = *rc::gen::inRange(2, fw);
                const int oh = *rc::gen::inRange(2, fh);
                const int bp = *gen_block();
                const int bl = *gen_block();

                struct overlay_rect r = overlay_calc_rect(
                        OVERLAY_POS_CENTER, 0, 0, fw, fh, ow, oh, bp, bl);

                RC_PRE(r.width  > 0);
                RC_PRE(r.height > 0);
                /* Centre placement combines three error sources, each
                 * bounded by < block:
                 *   1. width snap drops up to (bp-1) from overlay_w
                 *   2. y = (fh - overlay_h)/2 truncation loses 0.5 px
                 *      when fh - overlay_h is odd (counted as 1)
                 *   3. y snap drops up to (bp-1) from the ideal y
                 * Total asymmetry magnitude: |a + p + 2q| < 3*block,
                 * where a = width snap loss, p = parity loss, q = y
                 * snap loss. */
                const int slack_x = 3 * bp;
                const int slack_y = 3 * bl;
                RC_ASSERT(std::abs((fw - r.width)  - 2 * r.x) <= slack_x);
                RC_ASSERT(std::abs((fh - r.height) - 2 * r.y) <= slack_y);
        });

        return ok;
}
