/**
 * @file   test/test_overlay_soft_edge.h
 * @author Ben Roeder     <ben@sohonet.com>
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
#ifndef TEST_OVERLAY_SOFT_EDGE_H_8C2A1F6D_4B5E_4E3F_9A7C_3D8B5E1F2A4C
#define TEST_OVERLAY_SOFT_EDGE_H_8C2A1F6D_4B5E_4E3F_9A7C_3D8B5E1F2A4C

int overlay_soft_edge_test_zero_width_is_noop(void);
int overlay_soft_edge_test_edge_pixel_zeroed(void);
int overlay_soft_edge_test_linear_ramp(void);
int overlay_soft_edge_test_centre_untouched(void);
int overlay_soft_edge_test_rgb_components_unchanged(void);
int overlay_soft_edge_test_oversized_width_clamps(void);
int overlay_soft_edge_test_non_square(void);
int overlay_soft_edge_test_exact_half_dimension(void);
int overlay_soft_edge_test_scales_existing_alpha(void);
int overlay_soft_edge_test_degenerate_one_row(void);

#endif
