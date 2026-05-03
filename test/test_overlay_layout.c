/**
 * @file   test/test_overlay_layout.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Unit tests for utils/overlay_layout.c
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

#include "test_overlay_layout.h"
#include "unit_common.h"
#include "utils/overlay_layout.h"

/* Center: a 200x100 overlay in 1920x1080 frame -> (860, 490, 200, 100). */
int overlay_layout_test_center(void)
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

int overlay_layout_test_corners(void)
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
int overlay_layout_test_custom_negative_from_edge(void)
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
int overlay_layout_test_block_pixel_alignment(void)
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
int overlay_layout_test_overlay_larger_than_frame(void)
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
int overlay_layout_test_oversized_center(void)
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
int overlay_layout_test_oversized_right(void)
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
int overlay_layout_test_oversized_custom_positive(void)
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
int overlay_layout_test_block_lines_alignment(void)
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
