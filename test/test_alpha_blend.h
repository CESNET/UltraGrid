/**
 * @file   test/test_alpha_blend.h
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
#ifndef TEST_ALPHA_BLEND_H_98D2F4A0_3E81_4B6F_9C2A_5D8E1B7F0A12
#define TEST_ALPHA_BLEND_H_98D2F4A0_3E81_4B6F_9C2A_5D8E1B7F0A12

/* Returns 0 on success, negative on failure. Matches the run_tests.c contract. */
int alpha_blend_test_rgba_alpha_zero(void);
int alpha_blend_test_rgba_alpha_max(void);
int alpha_blend_test_rgba_half_alpha(void);
int alpha_blend_test_rgb_alpha_zero(void);
int alpha_blend_test_rgb_alpha_max(void);
int alpha_blend_test_uyvy_alpha_zero(void);
int alpha_blend_test_uyvy_alpha_max_white(void);
int alpha_blend_test_uyvy_alpha_max_red(void);
int alpha_blend_test_yuyv_alpha_zero(void);
int alpha_blend_test_yuyv_alpha_max_white(void);
int alpha_blend_test_yuyv_alpha_max_red(void);
int alpha_blend_test_y416_alpha_zero(void);
int alpha_blend_test_y416_alpha_max_white(void);
int alpha_blend_test_i420_alpha_zero(void);
int alpha_blend_test_i420_alpha_max_white(void);
int alpha_blend_test_i420_chroma_alpha_averaging(void);
int alpha_blend_test_i420_subregion_strides(void);
int alpha_blend_test_rg48_alpha_zero(void);
int alpha_blend_test_rg48_alpha_max_white(void);
int alpha_blend_test_v210_alpha_zero(void);
int alpha_blend_test_v210_alpha_max_white(void);
int alpha_blend_test_r10k_alpha_zero(void);
int alpha_blend_test_r10k_alpha_max_white(void);
int alpha_blend_test_r12l_alpha_zero(void);
int alpha_blend_test_r12l_alpha_max_white(void);

#endif // defined TEST_ALPHA_BLEND_H_*
