/**
 * @file   utils/overlay_soft_edge.h
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

#ifndef UTILS_OVERLAY_SOFT_EDGE_H_4D2A8B7F_3E1C_4F6A_9B5D_8C2E1F4A6B3D
#define UTILS_OVERLAY_SOFT_EDGE_H_4D2A8B7F_3E1C_4F6A_9B5D_8C2E1F4A6B3D

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Apply a linear alpha ramp around the edges of a 16-bit RGBA overlay.
 * For every pixel, alpha is multiplied by min(d, edge_w) / edge_w where
 * d is the distance to the nearest edge — pixels on the outer row/column
 * end up at alpha=0, and pixels at distance >= edge_w are unchanged.
 *
 * RGB components are not touched. edge_w == 0 is a no-op. edge_w larger
 * than min(width, height) / 2 is silently clamped to that limit (so the
 * centre still keeps non-zero alpha for pathological config values).
 */
void overlay_apply_soft_edge(uint16_t *rgba16,
                             int width, int height, int edge_w);

#ifdef __cplusplus
}
#endif

#endif
