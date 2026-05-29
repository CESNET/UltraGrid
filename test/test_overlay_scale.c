/**
 * @file   test/test_overlay_scale.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Unit tests for utils/overlay_scale.c
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
#include <stdlib.h>
#include <string.h>

#include "test_overlay_scale.h"
#include "unit_common.h"
#include "utils/overlay_scale.h"

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
int overlay_scale_test_identity(void)
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
int overlay_scale_test_upscale_solid_colour(void)
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
int overlay_scale_test_downscale_average(void)
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

int overlay_scale_test_returns_null_on_bad_dims(void)
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

int overlay_scale_test_source_buffer_unchanged(void)
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

int overlay_scaler_test_create_destroy(void)
{
        struct overlay_scaler *s = overlay_scaler_create(OVERLAY_SCALE_LANCZOS);
        ASSERT_MESSAGE("create non-NULL", s != NULL);
        overlay_scaler_destroy(s);
        /* destroy(NULL) is a documented no-op so callers don't need
         * to guard their cleanup paths. */
        overlay_scaler_destroy(NULL);
        return 0;
}

int overlay_scaler_test_reuses_context_same_dims(void)
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

int overlay_scaler_test_rebuilds_context_on_dim_change(void)
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
int overlay_scaler_test_scale_into_no_alloc(void)
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
int overlay_scaler_test_filter_nearest(void)
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
int overlay_scaler_test_filter_bilinear(void)
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
