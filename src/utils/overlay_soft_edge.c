/**
 * @file   utils/overlay_soft_edge.c
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Linear alpha-edge fade for the overlay buffer.
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
#include "utils/overlay_soft_edge.h"

void overlay_apply_soft_edge(uint16_t *rgba16,
                             int width, int height, int edge_w)
{
        if (edge_w <= 0 || width <= 0 || height <= 0) return;

        /* Clamp so the centre always keeps non-zero alpha. For a 4-wide
         * buffer the deepest interior pixel is at distance 2 from each
         * edge, so the maximum useful edge_w is width/2 (and height/2). */
        const int max_edge = MIN(width, height) / 2;
        if (edge_w > max_edge) edge_w = max_edge;
        /* Reachable for a 1xN or Nx1 buffer (max_edge=0); also prevents
         * the divide-by-zero in the inner loop. */
        if (edge_w == 0) return;

        for (int y = 0; y < height; y++) {
                const int dy = MIN(y, height - 1 - y);
                for (int x = 0; x < width; x++) {
                        const int dx = MIN(x, width - 1 - x);
                        const int d = MIN(dx, dy);
                        if (d >= edge_w) continue;
                        uint16_t *a = &rgba16[(y * width + x) * 4 + 3];
                        /* Linear ramp: alpha *= d / edge_w. uint32_t holds
                         * 65535 * (edge_w - 1) since edge_w is clamped to
                         * MIN(w, h)/2 — well below 2^16. */
                        *a = (uint16_t)(((uint32_t)*a * (uint32_t)d) / (uint32_t)edge_w);
                }
        }
}
