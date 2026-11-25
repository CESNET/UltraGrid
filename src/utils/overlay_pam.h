/**
 * @file   utils/overlay_pam.h
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  PAM image loader normalised to 16-bit RGBA, used by the overlay
 *         postprocessor.
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

#ifndef UTILS_OVERLAY_PAM_H_3F8E2A1D_6B4C_4A9F_92E0_7D5C8E3A2B6F
#define UTILS_OVERLAY_PAM_H_3F8E2A1D_6B4C_4A9F_92E0_7D5C8E3A2B6F

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Load a PAM image from path and normalise to 16-bit RGBA. Accepts 8-bit
 * (maxval=255) or 16-bit (maxval=65535) PAM in RGB (3-channel) or RGBA
 * (4-channel) layout. Other maxvals are rejected. 3-channel images receive
 * alpha=65535. 8-bit values are bit-replicated to 16-bit (val<<8|val).
 *
 * On success: *out_data points to an allocated uint16_t buffer with
 * width*height*4 components; *out_width and *out_height are set; the caller
 * must free(*out_data). On failure returns false and leaves outputs untouched.
 */
bool overlay_load_pam_rgba16(const char *path, uint16_t **out_data,
                             int *out_width, int *out_height);

#ifdef __cplusplus
}
#endif

#endif
