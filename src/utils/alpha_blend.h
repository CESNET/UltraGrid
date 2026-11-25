/**
 * @file   utils/alpha_blend.h
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Alpha blending of 16-bit RGBA overlay onto native video formats
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

#ifndef UTILS_ALPHA_BLEND_H_5A8F2C1B_4E73_4D8A_B6F0_2C9E1F4A8B3D
#define UTILS_ALPHA_BLEND_H_5A8F2C1B_4E73_4D8A_B6F0_2C9E1F4A8B3D

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Blend a 16-bit RGBA overlay (uint16_t per component, range 0-65535) onto
 * an 8-bit RGBA destination buffer. width is the number of pixels.
 */
void alpha_blend_rgba(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width);
void alpha_blend_rgb(uint8_t * __restrict dst,
                     const uint16_t * __restrict rgba16, int width);
void alpha_blend_uyvy(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width);
void alpha_blend_yuyv(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width);
void alpha_blend_y416(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width);

/*
 * v210: 10-bit YUV 4:2:2 packed, 6 pixels per 16 bytes (4 uint32_t words).
 * width must be a multiple of 6; remaining pixels are silently dropped.
 */
void alpha_blend_v210(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width);

/*
 * R10k: 10-bit RGB packed in 4 bytes per pixel. No color space conversion
 * (RGB destination); 16-bit input scaled to 10-bit by right-shifting 6.
 */
void alpha_blend_r10k(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width);

/*
 * R12L: 12-bit RGB packed, 8 pixels in 36 bytes. No color space conversion;
 * 16-bit input scaled to 12-bit by right-shifting 4. Width should be a
 * multiple of 8; remaining pixels (1-7) are processed in 2-pixel pair
 * sub-blocks, with any final odd pixel silently dropped.
 */
void alpha_blend_r12l(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width);

/*
 * RG48: 16-bit RGB, 6 bytes per pixel little-endian. No color conversion;
 * 16-bit overlay components write through directly to the destination.
 * dst is uint8_t* because the destination buffer has no 2-byte alignment
 * guarantee — components are loaded/stored via memcpy.
 */
void alpha_blend_rg48(uint8_t * __restrict dst,
                      const uint16_t * __restrict rgba16, int width);

/*
 * I420 planar 4:2:0: width must be even (odd column truncated; height likewise
 * truncated to an even count for the chroma pass).
 *
 * Plane and overlay strides are explicit so the call can blend into a
 * sub-region of a larger frame:
 *   - y_stride:         bytes per row in the Y plane
 *   - uv_stride:        bytes per row in each of the U and V planes
 *   - src_pixel_stride: pixels per row of the rgba16 overlay buffer (the
 *                       byte advance is src_pixel_stride * 4 * sizeof(uint16_t))
 * For the simple "full-frame" case, pass y_stride=width, uv_stride=width/2,
 * src_pixel_stride=width.
 */
void alpha_blend_i420(uint8_t * __restrict dst_y, int y_stride,
                      uint8_t * __restrict dst_u,
                      uint8_t * __restrict dst_v, int uv_stride,
                      const uint16_t * __restrict rgba16, int src_pixel_stride,
                      int width, int height);

#ifdef __cplusplus
}
#endif

#endif
