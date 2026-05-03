/**
 * @file   utils/overlay_layout.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Overlay positioning math
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

#include "utils/macros.h"
#include "utils/overlay_layout.h"

/* 1D placement: the overlay's left edge wants to be at signed
 * coordinate `pos` within a frame of size `frame_size`. Compute the
 * intersection of [pos, pos + overlay_size] with [0, frame_size].
 *
 * Returns the frame-side offset, the offset *into* the overlay buffer
 * (non-zero when pos < 0 — i.e. the overlay extends past the left/top
 * edge), and the visible length. When the overlay fits cleanly inside
 * the frame, src_off is always 0. */
struct axis {
        int frame_off;
        int src_off;
        int length;
};

static struct axis
calc_axis(int pos, int frame_size, int overlay_size, int block)
{
        int frame_off = pos < 0 ? 0 : pos;
        int end       = pos + overlay_size;
        if (end > frame_size) end = frame_size;
        int length    = end - frame_off;
        if (length < 0) length = 0;
        int src_off   = frame_off - pos;  /* 0 if pos >= 0 */

        if (block > 1) {
                /* Snap frame_off down to a block boundary. When the
                 * overlay extends past the left edge (src_off > 0) we
                 * have cushion: shift src_off back the same amount and
                 * gain `shift` columns of length. Without cushion the
                 * overlay just appears shifted-left in the frame by
                 * up to (block-1) pixels — a cosmetic artifact of the
                 * codec's pixel-block grid, same as the legacy
                 * behaviour before src_x/src_y were tracked. */
                int shift = frame_off % block;
                if (shift > 0) {
                        frame_off -= shift;
                        if (src_off >= shift) {
                                src_off -= shift;
                                length  += shift;
                        }
                }
                /* length only grows in the cushion branch, never
                 * shrinks; the earlier `length < 0` guard already
                 * pinned it to >= 0. Just snap to a block multiple. */
                length -= length % block;
        }
        return (struct axis){frame_off, src_off, length};
}

struct overlay_rect overlay_calc_rect(enum overlay_position pos,
                                      int custom_x, int custom_y,
                                      int frame_w, int frame_h,
                                      int overlay_w, int overlay_h,
                                      int block_pixels, int block_lines)
{
        int ideal_x = 0, ideal_y = 0;
        switch (pos) {
        case OVERLAY_POS_CENTER:
                ideal_x = (frame_w - overlay_w) / 2;
                ideal_y = (frame_h - overlay_h) / 2;
                break;
        case OVERLAY_POS_TOP_LEFT:
                ideal_x = 0; ideal_y = 0;
                break;
        case OVERLAY_POS_TOP_RIGHT:
                ideal_x = frame_w - overlay_w; ideal_y = 0;
                break;
        case OVERLAY_POS_BOTTOM_LEFT:
                ideal_x = 0; ideal_y = frame_h - overlay_h;
                break;
        case OVERLAY_POS_BOTTOM_RIGHT:
                ideal_x = frame_w - overlay_w;
                ideal_y = frame_h - overlay_h;
                break;
        case OVERLAY_POS_CUSTOM:
                /* Negative values count from the right/bottom edge: the
                 * overlay's right edge sits |custom_x| pixels from the
                 * frame's right edge for x = -|custom_x|. */
                ideal_x = custom_x < 0 ? frame_w + custom_x - overlay_w
                                       : custom_x;
                ideal_y = custom_y < 0 ? frame_h + custom_y - overlay_h
                                       : custom_y;
                break;
        }

        const struct axis ax = calc_axis(ideal_x, frame_w, overlay_w,
                                         block_pixels);
        const struct axis ay = calc_axis(ideal_y, frame_h, overlay_h,
                                         block_lines);
        return (struct overlay_rect){
                .x      = ax.frame_off,
                .y      = ay.frame_off,
                .width  = ax.length,
                .height = ay.length,
                .src_x  = ax.src_off,
                .src_y  = ay.src_off,
        };
}
