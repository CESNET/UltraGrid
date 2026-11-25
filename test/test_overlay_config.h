/**
 * @file   test/test_overlay_config.h
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
#ifndef TEST_OVERLAY_CONFIG_H_3B7E8C2A_5D9F_4A1E_8B6D_4C2F7A3E1D9B
#define TEST_OVERLAY_CONFIG_H_3B7E8C2A_5D9F_4A1E_8B6D_4C2F7A3E1D9B

int overlay_config_test_minimal_file_only(void);
int overlay_config_test_position_keywords(void);
int overlay_config_test_custom_xy(void);
int overlay_config_test_help(void);
int overlay_config_test_rejects_missing_file(void);
int overlay_config_test_rejects_unknown_key(void);
int overlay_config_test_rejects_bad_position(void);
int overlay_config_test_rejects_non_integer_xy(void);
int overlay_config_test_rejects_null_and_empty_value(void);
int overlay_config_test_rejects_oversize_options(void);
int overlay_config_test_soft_edge(void);
int overlay_config_test_scale(void);
int overlay_config_test_scale_frame(void);
int overlay_config_test_scale_frame_overrides_wxh(void);
int overlay_config_test_perf(void);
int overlay_config_test_scale_filter(void);
int overlay_config_test_blend_threads(void);

#endif
