/**
 * @file   test/test_overlay_scale.h
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
#ifndef TEST_OVERLAY_SCALE_H_2A1F8C7B_4E5D_4F3A_9B2D_5E8A1F7B4C3D
#define TEST_OVERLAY_SCALE_H_2A1F8C7B_4E5D_4F3A_9B2D_5E8A1F7B4C3D

int overlay_scale_test_identity(void);
int overlay_scale_test_upscale_solid_colour(void);
int overlay_scale_test_downscale_average(void);
int overlay_scale_test_returns_null_on_bad_dims(void);
int overlay_scale_test_source_buffer_unchanged(void);
int overlay_scaler_test_create_destroy(void);
int overlay_scaler_test_reuses_context_same_dims(void);
int overlay_scaler_test_rebuilds_context_on_dim_change(void);
int overlay_scaler_test_scale_into_no_alloc(void);
int overlay_scaler_test_filter_nearest(void);
int overlay_scaler_test_filter_bilinear(void);

#endif
