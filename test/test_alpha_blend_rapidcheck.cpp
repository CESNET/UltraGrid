/**
 * @file   test/test_alpha_blend_rapidcheck.cpp
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Property-based tests for utils/alpha_blend.c — every codec.
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
#include <cstring>
#include <vector>

extern "C" {
#include "utils/alpha_blend.h"
}

namespace {

/* Generate a 16-bit RGBA overlay row with all alphas forced to a chosen
 * value. Component layout matches alpha_blend.h: 4 uint16_t per pixel
 * (R, G, B, A). */
std::vector<uint16_t> rgba16_with_alpha(int width, uint16_t alpha)
{
        auto pixels = *rc::gen::container<std::vector<uint16_t>>(
                width * 4U, rc::gen::arbitrary<uint16_t>());
        for (int i = 0; i < width; i++) pixels[i * 4 + 3] = alpha;
        return pixels;
}

std::vector<uint8_t> dst_bytes(size_t n)
{
        return *rc::gen::container<std::vector<uint8_t>>(
                n, rc::gen::arbitrary<uint8_t>());
}

/* Place the dst buffer between two pages of guard bytes; assert the
 * guards are untouched after the blend. Catches off-by-one writes that
 * a strict bounds check would miss. */
struct guarded_buffer {
        static constexpr size_t GUARD = 32;
        std::vector<uint8_t> buf;       /* size = GUARD + payload + GUARD */
        size_t payload_size;

        explicit guarded_buffer(size_t payload)
            : buf(GUARD + payload + GUARD, 0xAB)
            , payload_size(payload)
        {
                /* Random payload bytes; guards stay 0xAB. */
                auto fill = *rc::gen::container<std::vector<uint8_t>>(
                        payload, rc::gen::arbitrary<uint8_t>());
                std::memcpy(buf.data() + GUARD, fill.data(), payload);
        }

        uint8_t *dst() { return buf.data() + GUARD; }

        bool guards_intact() const
        {
                for (size_t i = 0; i < GUARD; i++) {
                        if (buf[i] != 0xAB) return false;
                        if (buf[GUARD + payload_size + i] != 0xAB) return false;
                }
                return true;
        }
};

/* width range used by the per-codec generators. Wide enough to exercise
 * group packing (R12L 8-pixel groups, v210 6-pixel groups) without
 * blowing test time. */
auto gen_width()       { return rc::gen::inRange(1, 256); }
auto gen_width_even()  { return rc::gen::map(rc::gen::inRange(1, 128), [](int n){ return n*2; }); }
auto gen_width_v210()  { return rc::gen::map(rc::gen::inRange(1, 32),  [](int n){ return n*6; }); }
auto gen_width_r12l()  { return rc::gen::map(rc::gen::inRange(1, 32),  [](int n){ return n*8; }); }
auto gen_width_i420()  { return rc::gen::map(rc::gen::inRange(1, 64),  [](int n){ return n*2; }); }
auto gen_height_i420() { return rc::gen::map(rc::gen::inRange(1, 64),  [](int n){ return n*2; }); }

/* Bytes per pixel for each packed codec. */
constexpr int BPP_RGBA = 4;
constexpr int BPP_RGB  = 3;
constexpr int BPP_UYVY = 2;
constexpr int BPP_YUYV = 2;
constexpr int BPP_Y416 = 8;
constexpr int BPP_R10K = 4;
constexpr int BPP_RG48 = 6;

inline int bytes_v210(int width) { return (width / 6) * 16; }
inline int bytes_r12l(int width) { return (width / 8) * 36; }

/* Run an alpha=0 identity check for a single-plane blend.
 *
 * For codecs without format-reserved bits the property is "byte-identical
 * dst, including the very first call". This is the property the original
 * RapidCheck integration found (and fixed) the RGBA dst[3] clobber with;
 * the canonicalize-then-no-op variant alone would not have caught it.
 *
 * For codecs with reserved bits (R10k pad, v210 bits 30-31) the first
 * call legally canonicalises those bits — pixel data is preserved but
 * byte representation isn't. The two-call variant verifies a converged
 * dst stays converged. */
template <typename Blend>
bool check_alpha_zero(const char *label, int bytes_per_row,
                      Blend blend, decltype(gen_width()) width_gen)
{
        return rc::check(label, [=]() {
                const int w = *width_gen;
                const auto src = rgba16_with_alpha(w, 0);
                guarded_buffer g(w * bytes_per_row);
                std::vector<uint8_t> before(g.dst(),
                                            g.dst() + g.payload_size);
                blend(g.dst(), src.data(), w);
                std::vector<uint8_t> after(g.dst(),
                                           g.dst() + g.payload_size);
                RC_ASSERT(before == after);
                RC_ASSERT(g.guards_intact());
        });
}

/* Two-call variant for codecs with format-reserved bits. */
template <typename Blend>
bool check_alpha_zero_canonical(const char *label, int bytes_per_row,
                                Blend blend,
                                decltype(gen_width()) width_gen)
{
        return rc::check(label, [=]() {
                const int w = *width_gen;
                const auto src = rgba16_with_alpha(w, 0);
                guarded_buffer g(w * bytes_per_row);
                blend(g.dst(), src.data(), w);
                std::vector<uint8_t> canonical(g.dst(),
                                               g.dst() + g.payload_size);
                blend(g.dst(), src.data(), w);
                std::vector<uint8_t> after(g.dst(),
                                           g.dst() + g.payload_size);
                RC_ASSERT(canonical == after);
                RC_ASSERT(g.guards_intact());
        });
}

/* No-overrun + bounds check (every output byte stays in-range, which
 * for uint8_t is automatic — the meaningful check is the guard pages). */
template <typename Blend>
bool check_no_overrun(const char *label, int bytes_per_row,
                      Blend blend, decltype(gen_width()) width_gen)
{
        return rc::check(label, [=]() {
                const int w = *width_gen;
                const auto src = rgba16_with_alpha(w,
                        *rc::gen::arbitrary<uint16_t>());
                guarded_buffer g(w * bytes_per_row);
                blend(g.dst(), src.data(), w);
                RC_ASSERT(g.guards_intact());
        });
}

/* Monotonicity in alpha: increasing the source alpha can only move dst
 * monotonically toward the (codec-specific) src representation. For all
 * codecs this means: with low alpha, |dst - src_repr| >= |dst' - src_repr|
 * where dst' is the result with higher alpha. We don't have src_repr
 * cheaply for converted codecs; instead check the RGB-only codecs where
 * dst byte is a direct blend of src component and original dst. */
bool check_rgba_monotonicity()
{
        return rc::check("alpha_blend_rgba: dst moves monotonically with alpha",
                         []() {
                const int w = *gen_width();
                /* Independent RGB; alpha bumped from a_lo to a_hi. */
                auto src = rgba16_with_alpha(w, 0);
                for (int i = 0; i < w; i++) {
                        src[i*4+0] = *rc::gen::arbitrary<uint16_t>();
                        src[i*4+1] = *rc::gen::arbitrary<uint16_t>();
                        src[i*4+2] = *rc::gen::arbitrary<uint16_t>();
                }
                const uint16_t a_lo = *rc::gen::inRange(0, 32768);
                const uint16_t a_hi = *rc::gen::inRange(32768, 65536);
                const auto dst_orig = dst_bytes(w * BPP_RGBA);

                std::vector<uint8_t> dst_lo = dst_orig;
                std::vector<uint8_t> dst_hi = dst_orig;
                for (int i = 0; i < w; i++) src[i*4+3] = a_lo;
                alpha_blend_rgba(dst_lo.data(), src.data(), w);
                for (int i = 0; i < w; i++) src[i*4+3] = a_hi;
                alpha_blend_rgba(dst_hi.data(), src.data(), w);

                /* d_lo and d_hi both lie on the segment [d_orig, s_byte];
                 * d_hi must be at least as close to s_byte as d_lo. */
                for (int i = 0; i < w; i++) {
                        for (int c = 0; c < 3; c++) {
                                const int s_byte = src[i*4+c] >> 8;
                                const int dist_lo =
                                        std::abs(int(dst_lo[i*4+c]) - s_byte);
                                const int dist_hi =
                                        std::abs(int(dst_hi[i*4+c]) - s_byte);
                                RC_ASSERT(dist_hi <= dist_lo);
                        }
                }
        });
}

} // namespace

bool test_alpha_blend_properties()
{
        bool ok = true;

        /* alpha=0 identity — the bug-finder per the original RapidCheck
         * integration (it caught dst[3] being clobbered by RGBA blend). */
        ok &= check_alpha_zero("alpha_blend_rgba: alpha=0 preserves dst",
                               BPP_RGBA, alpha_blend_rgba, gen_width());
        ok &= check_alpha_zero("alpha_blend_rgb:  alpha=0 preserves dst",
                               BPP_RGB,  alpha_blend_rgb,  gen_width());
        ok &= check_alpha_zero("alpha_blend_uyvy: alpha=0 preserves dst",
                               BPP_UYVY, alpha_blend_uyvy, gen_width_even());
        ok &= check_alpha_zero("alpha_blend_yuyv: alpha=0 preserves dst",
                               BPP_YUYV, alpha_blend_yuyv, gen_width_even());
        ok &= check_alpha_zero("alpha_blend_y416: alpha=0 preserves dst",
                               BPP_Y416, alpha_blend_y416, gen_width());
        /* R10k always packs the 0x3 reserved pad bits even with alpha=0
         * — use the canonicalize-then-no-op variant. */
        ok &= check_alpha_zero_canonical(
                "alpha_blend_r10k: alpha=0 preserves dst (canonical)",
                BPP_R10K, alpha_blend_r10k, gen_width());
        ok &= check_alpha_zero("alpha_blend_rg48: alpha=0 preserves dst",
                               BPP_RG48, alpha_blend_rg48, gen_width());

        /* v210: 6-pixel groups (16 bytes/group); reserved bits 30-31
         * always pack to zero, so use the canonicalize-then-no-op
         * pattern. Inline because the helper templates assume
         * bytes-per-pixel rather than bytes-per-group. */
        ok &= rc::check("alpha_blend_v210: alpha=0 preserves dst (canonical)",
                        []() {
                const int w = *gen_width_v210();
                const auto src = rgba16_with_alpha(w, 0);
                guarded_buffer g(bytes_v210(w));
                alpha_blend_v210(g.dst(), src.data(), w);
                std::vector<uint8_t> canonical(g.dst(),
                                               g.dst() + g.payload_size);
                alpha_blend_v210(g.dst(), src.data(), w);
                std::vector<uint8_t> after(g.dst(),
                                           g.dst() + g.payload_size);
                RC_ASSERT(canonical == after);
                RC_ASSERT(g.guards_intact());
        });

        /* R12L: 8-pixel groups (36 bytes), partial trailing pairs OK. No
         * reserved bits — strict byte identity holds. */
        ok &= rc::check("alpha_blend_r12l: alpha=0 preserves dst", []() {
                const int w = *gen_width_r12l();
                const auto src = rgba16_with_alpha(w, 0);
                guarded_buffer g(bytes_r12l(w));
                std::vector<uint8_t> before(g.dst(),
                                            g.dst() + g.payload_size);
                alpha_blend_r12l(g.dst(), src.data(), w);
                std::vector<uint8_t> after(g.dst(),
                                           g.dst() + g.payload_size);
                RC_ASSERT(before == after);
                RC_ASSERT(g.guards_intact());
        });

        /* I420 alpha=0 identity across all three planes. */
        ok &= rc::check("alpha_blend_i420: alpha=0 preserves all planes", []() {
                const int w = *gen_width_i420();
                const int h = *gen_height_i420();
                const auto src = [&]() {
                        auto v = *rc::gen::container<std::vector<uint16_t>>(
                                static_cast<size_t>(w) * h * 4,
                                rc::gen::arbitrary<uint16_t>());
                        for (int i = 0; i < w * h; i++) v[i * 4 + 3] = 0;
                        return v;
                }();
                guarded_buffer y_g(static_cast<size_t>(w) * h);
                guarded_buffer u_g(static_cast<size_t>(w / 2) * (h / 2));
                guarded_buffer v_g(static_cast<size_t>(w / 2) * (h / 2));
                std::vector<uint8_t> y0(y_g.dst(), y_g.dst() + y_g.payload_size);
                std::vector<uint8_t> u0(u_g.dst(), u_g.dst() + u_g.payload_size);
                std::vector<uint8_t> v0(v_g.dst(), v_g.dst() + v_g.payload_size);
                alpha_blend_i420(y_g.dst(), w, u_g.dst(), v_g.dst(), w / 2,
                                 src.data(), w, w, h);
                std::vector<uint8_t> y1(y_g.dst(), y_g.dst() + y_g.payload_size);
                std::vector<uint8_t> u1(u_g.dst(), u_g.dst() + u_g.payload_size);
                std::vector<uint8_t> v1(v_g.dst(), v_g.dst() + v_g.payload_size);
                RC_ASSERT(y0 == y1);
                RC_ASSERT(u0 == u1);
                RC_ASSERT(v0 == v1);
                RC_ASSERT(y_g.guards_intact() && u_g.guards_intact()
                          && v_g.guards_intact());
        });

        /* Per-codec no-overrun. */
        ok &= check_no_overrun("alpha_blend_rgba: no buffer overrun",
                               BPP_RGBA, alpha_blend_rgba, gen_width());
        ok &= check_no_overrun("alpha_blend_rgb:  no buffer overrun",
                               BPP_RGB,  alpha_blend_rgb,  gen_width());
        ok &= check_no_overrun("alpha_blend_uyvy: no buffer overrun",
                               BPP_UYVY, alpha_blend_uyvy, gen_width_even());
        ok &= check_no_overrun("alpha_blend_yuyv: no buffer overrun",
                               BPP_YUYV, alpha_blend_yuyv, gen_width_even());
        ok &= check_no_overrun("alpha_blend_y416: no buffer overrun",
                               BPP_Y416, alpha_blend_y416, gen_width());
        ok &= check_no_overrun("alpha_blend_r10k: no buffer overrun",
                               BPP_R10K, alpha_blend_r10k, gen_width());
        ok &= check_no_overrun("alpha_blend_rg48: no buffer overrun",
                               BPP_RG48, alpha_blend_rg48, gen_width());

        ok &= check_rgba_monotonicity();

        return ok;
}
