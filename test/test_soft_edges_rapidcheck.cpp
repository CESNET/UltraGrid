/**
 * @file   test/test_soft_edges_rapidcheck.cpp
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Property-based tests for utils/overlay_soft_edge.c.
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

#include <cstdint>
#include <vector>

extern "C" {
#include "utils/overlay_soft_edge.h"
}

namespace {

auto gen_dim()    { return rc::gen::inRange(1, 256); }
auto gen_edge_w() { return rc::gen::inRange(0, 64); }

std::vector<uint16_t> gen_overlay(int w, int h)
{
        return *rc::gen::container<std::vector<uint16_t>>(
                static_cast<size_t>(w) * h * 4,
                rc::gen::arbitrary<uint16_t>());
}

inline uint16_t alpha_at(const std::vector<uint16_t> &v,
                         int w, int x, int y)
{
        return v[(static_cast<size_t>(y) * w + x) * 4 + 3];
}

} // namespace

bool test_soft_edge_properties()
{
        bool ok = true;

        /* edge_w == 0 is documented as a no-op. */
        ok &= rc::check("soft_edge: edge_w=0 is a no-op", []() {
                const int w = *gen_dim();
                const int h = *gen_dim();
                auto buf = gen_overlay(w, h);
                auto before = buf;
                overlay_apply_soft_edge(buf.data(), w, h, 0);
                RC_ASSERT(buf == before);
        });

        /* RGB channels are never touched. */
        ok &= rc::check("soft_edge: RGB channels untouched", []() {
                const int w = *gen_dim();
                const int h = *gen_dim();
                const int e = *gen_edge_w();
                auto buf = gen_overlay(w, h);
                auto before = buf;
                overlay_apply_soft_edge(buf.data(), w, h, e);
                for (size_t i = 0; i + 3 < buf.size(); i += 4) {
                        RC_ASSERT(buf[i + 0] == before[i + 0]);
                        RC_ASSERT(buf[i + 1] == before[i + 1]);
                        RC_ASSERT(buf[i + 2] == before[i + 2]);
                }
        });

        /* Alpha never increases. The fade only attenuates. */
        ok &= rc::check("soft_edge: alpha never increases", []() {
                const int w = *gen_dim();
                const int h = *gen_dim();
                const int e = *gen_edge_w();
                auto buf = gen_overlay(w, h);
                auto before = buf;
                overlay_apply_soft_edge(buf.data(), w, h, e);
                for (size_t i = 3; i < buf.size(); i += 4) {
                        RC_ASSERT(buf[i] <= before[i]);
                }
        });

        /* Pixels at distance >= edge_w are unchanged. Distance d to the
         * nearest edge of an int x in [0, w-1] is min(x, w-1-x); same for y. */
        ok &= rc::check("soft_edge: interior pixels unchanged", []() {
                const int w = *rc::gen::inRange(8, 256);
                const int h = *rc::gen::inRange(8, 256);
                /* Pick edge_w small enough that an interior region exists. */
                const int e = *rc::gen::inRange(1, std::min(w, h) / 2);
                auto buf = gen_overlay(w, h);
                auto before = buf;
                overlay_apply_soft_edge(buf.data(), w, h, e);
                for (int y = 0; y < h; y++) {
                        for (int x = 0; x < w; x++) {
                                const int dx = std::min(x, w - 1 - x);
                                const int dy = std::min(y, h - 1 - y);
                                if (std::min(dx, dy) >= e) {
                                        RC_ASSERT(alpha_at(buf,    w, x, y)
                                                  == alpha_at(before, w, x, y));
                                }
                        }
                }
        });

        /* Outer row/column ends up at alpha = 0 when edge_w > 0 and the
         * overlay is large enough that the ramp actually fires. */
        ok &= rc::check("soft_edge: outer ring is zero", []() {
                const int w = *rc::gen::inRange(4, 256);
                const int h = *rc::gen::inRange(4, 256);
                const int e = *rc::gen::inRange(1, std::min(w, h) / 2);
                auto buf = gen_overlay(w, h);
                overlay_apply_soft_edge(buf.data(), w, h, e);
                for (int x = 0; x < w; x++) {
                        RC_ASSERT(alpha_at(buf, w, x, 0)     == 0);
                        RC_ASSERT(alpha_at(buf, w, x, h - 1) == 0);
                }
                for (int y = 0; y < h; y++) {
                        RC_ASSERT(alpha_at(buf, w, 0,     y) == 0);
                        RC_ASSERT(alpha_at(buf, w, w - 1, y) == 0);
                }
        });

        return ok;
}
