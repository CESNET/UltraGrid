/**
 * @file   color_space.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022-2025 CESNET, zájmové sdružení právnických osob
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

#ifndef COLOR_H_CD26B745_C30E_4DA3_8280_C9492B6BFF25
#define COLOR_H_CD26B745_C30E_4DA3_8280_C9492B6BFF25

#ifdef __cplusplus
#include <cstdint>
#else
#include <assert.h>
#include <stdint.h>
#endif

#include "utils/macros.h" // CLAMP

/**
 * @file
 * @brief Color space coedfficients and limits
 *
 * RGB should use SDI full range [1<<(depth-8)..255<<(depth-8)-1], YCbCr
 * limited, see [limits].
 *
 * The coefficients are scaled by 1<<COMP_BASE.
 *
 * limited footroom is (1<<(limited_depth - 4)), headroom 235*(limited_depth-8)
 * /luma/, 240*(limited_depth-8)/255 /chroma/; full-range limits
 * [2^(depth-8)..255*2^(depth-8)-1] (excludes vals with 0x00 and 0xFF MSB).
 *
 * matrix Y = [ 0.182586, 0.614231, 0.062007; -0.100643, -0.338572, 0.4392157; 0.4392157, -0.398942, -0.040274 ]
 * * [coefficients]: https://gist.github.com/yohhoy/dafa5a47dade85d8b40625261af3776a "Rec. 709 coefficients"
 * * [limits]:       https://tech.ebu.ch/docs/r/r103.pdf                             "SDI limits"
 *
 * @todo
 * Use this transformations in all conversions.
 */
typedef int32_t comp_type_t; // int32_t provides much better performance than int_fast32_t
#define COMP_BASE (sizeof(comp_type_t) == 4 ? 14 : 18) // computation will be less precise when comp_type_t is 32 bit
static_assert(sizeof(comp_type_t) * 8 >= COMP_BASE + 18, "comp_type_t not wide enough (we are computing in up to 16 bits!)");

#define KR_601 .299
#define KB_601 .114
#define KR_709 .212639
#define KB_709 .072192
#define KR_2020 .262700
#define KB_2020 .059302
#define KR_P3 .228975
#define KB_P3 .079287

#ifdef YCBCR_FULL
#define LIMIT_LO(depth) 0
#define LIMIT_HI_Y(depth) ((1<<(depth))-1)
#define LIMIT_HI_CBCR(depth) ((1<<(depth))-1)
#else
#define LIMIT_LO(depth) (1<<((depth)-4))
#define LIMIT_HI_Y(depth) (235 * (1<<((depth)-8)))
#define LIMIT_HI_CBCR(depth) (240 * (1<<((depth)-8)))
#endif
// TODO: remove
#define CLAMP_LIMITED_Y(val, depth) (val)
#define CLAMP_LIMITED_CBCR(val, depth) (val)

#define FULL_FOOT(depth) (1 << ((depth) - 8))
#define FULL_HEAD(depth) ((255<<((depth)-8))-1)
#define CLAMP_FULL(val, depth) CLAMP((val), FULL_FOOT(depth), FULL_HEAD(depth))

#define RGB_TO_Y(t, r, g, b) ((r) * (t).y_r + (g) * (t).y_g + (b) * (t).y_b)
#define RGB_TO_CB(t, r, g, b) \
        ((r) * (t).cb_r + (g) * (t).cb_g + (b) * (t).cb_b)
#define RGB_TO_CR(t, r, g, b) \
        ((r) * (t).cr_r + (g) * (t).cr_g + (b) * (t).cr_b)
/// @param y_scaled Y scaled (multiplied) by Y_SCALE()
#define YCBCR_TO_R(t, y_scaled, cb, cr) ((y_scaled) + (cr) * (t).r_cr)
#define YCBCR_TO_G(t, y_scaled, cb, cr) \
        ((y_scaled) + (cb) * (t).g_cb + (cr) * (t).g_cr)
#define YCBCR_TO_B(t, y_scaled, cb, cr) ((y_scaled) + (cb) * (t).b_cb)

/**
 * @param alpha_mask alpha mask already positioned at target bit offset
 */
#define FORMAT_RGBA(r, g, b, rshift, gshift, bshift, alpha_mask, depth) \
        ((alpha_mask) | (CLAMP_FULL((r), (depth)) << (rshift) | \
                         CLAMP_FULL((g), (depth)) << (gshift) | \
                         CLAMP_FULL((b), (depth)) << (bshift)))

#define MK_MONOCHROME(val) \
        FORMAT_RGBA((val), (val), (val), 0, 8, 16, 0xFF000000, 8)
#define RGBA_BLACK MK_MONOCHROME(0x00)
#define RGBA_GRAY  MK_MONOCHROME(0x80)
#define RGBA_WHITE MK_MONOCHROME(0xFF)

#ifdef __cplusplus
extern "C" {
#endif

enum colorspace {
        CS_DFL = 0,
        CS_601 = 1,
        CS_709 = 2,
};

struct color_coeffs {
        // shorts are used the compiler can use 2-byte words in the vecotred
        // instruction, which is faster (and the values fit)
        short y_r,  y_g,  y_b;
        short cb_r, cb_g, cb_b;
        short cr_r, cr_g, cr_b;

        // the shorts below doesn't seem to be necessary - it seems like the
        // compiler doesn't vectorise those conversions (in contrary to the
        // above coeffs)
        short y_scale;
        short r_cr, g_cb, g_cr;
        int   b_cb; // is 34712 for 709  so doesn't fit to 16-bit short
};
const struct color_coeffs *get_color_coeffs(enum colorspace cs,
                                            int             ycbcr_bit_depth);
struct color_coeffs        compute_color_coeffs(double kr, double kb,
                                                int ycbcr_bit_depth);
enum colorspace            get_default_cs(void);

#ifdef __cplusplus
} // extern "C"
#endif


#endif // !defined COLOR_H_CD26B745_C30E_4DA3_8280_C9492B6BFF25
