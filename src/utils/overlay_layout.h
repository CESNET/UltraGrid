/**
 * @file   utils/overlay_layout.h
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Overlay positioning math for the overlay postprocessor.
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

#ifndef UTILS_OVERLAY_LAYOUT_H_5C2D8A1B_3E7F_4A9B_8D6C_2F5E1A8B7C3D
#define UTILS_OVERLAY_LAYOUT_H_5C2D8A1B_3E7F_4A9B_8D6C_2F5E1A8B7C3D

#ifdef __cplusplus
extern "C" {
#endif

enum overlay_position {
        OVERLAY_POS_CENTER = 0,
        OVERLAY_POS_TOP_LEFT,
        OVERLAY_POS_TOP_RIGHT,
        OVERLAY_POS_BOTTOM_LEFT,
        OVERLAY_POS_BOTTOM_RIGHT,
        OVERLAY_POS_CUSTOM,
};

struct overlay_rect {
        int x, y;          ///< origin within the frame
        int width, height; ///< blend region (overlay clipped to frame bounds)
        int src_x, src_y;  ///< origin within the overlay buffer (non-zero
                           ///< when the overlay is larger than the frame
                           ///< and the visible region is the centre/right
                           ///< slice of the overlay)
};

/*
 * Compute the blend rectangle for an overlay of size (overlay_w, overlay_h)
 * placed onto a frame of size (frame_w, frame_h).
 *
 * - pos selects a preset position; OVERLAY_POS_CUSTOM uses (custom_x, custom_y)
 *   instead, with negative values counting from the right/bottom edges.
 * - block_pixels / block_lines snap x/width and y/height down to the
 *   codec's pixel-block grid. Use get_pf_block_pixels(codec) for the
 *   horizontal block; pass block_lines=2 for chroma-vertically-subsampled
 *   formats (I420), 1 otherwise. Values <= 1 are a no-op for that axis.
 * - The returned rect is clipped to {0..frame_w, 0..frame_h}; an overlay that
 *   doesn't fit returns width or height of 0.
 * - When the overlay is larger than the frame, src_x/src_y describe which
 *   part of the overlay maps to the visible rect (left/centre/right slice
 *   per the chosen position). When the overlay fits, both are 0.
 */
struct overlay_rect overlay_calc_rect(enum overlay_position pos,
                                      int custom_x, int custom_y,
                                      int frame_w, int frame_h,
                                      int overlay_w, int overlay_h,
                                      int block_pixels, int block_lines);

#ifdef __cplusplus
}
#endif

#endif
