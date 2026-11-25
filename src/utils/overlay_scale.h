/**
 * @file   utils/overlay_scale.h
 * @author Ben Roeder     <ben@sohonet.com>
 * @brief  Resize a 16-bit RGBA overlay via libswscale.
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

#ifndef UTILS_OVERLAY_SCALE_H_5C8F2A1B_3D6E_4A8F_9C7D_2E5B1F4A8C3D
#define UTILS_OVERLAY_SCALE_H_5C8F2A1B_3D6E_4A8F_9C7D_2E5B1F4A8C3D

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Resampling filter for the upscale path. Trade-off summary:
 *   NEAREST       — fastest, blocky, no inter-pixel mixing
 *   FAST_BILINEAR — very fast, mediocre quality (libswscale's quick path)
 *   BILINEAR      — soft, smooth, half the cost of LANCZOS
 *   BICUBIC       — sharp + smooth, balanced (default)
 *   LANCZOS       — sharpest, can ring slightly, slowest
 *
 * Default is BICUBIC: it stays comfortably inside a 4K30 frame budget
 * where LANCZOS does not, and the visual difference for typical overlay
 * content is below threshold. Users who need sharper resampling can
 * opt into LANCZOS explicitly.
 */
enum overlay_scale_filter {
        OVERLAY_SCALE_BICUBIC = 0,
        OVERLAY_SCALE_LANCZOS,
        OVERLAY_SCALE_BILINEAR,
        OVERLAY_SCALE_FAST_BILINEAR,
        OVERLAY_SCALE_NEAREST,
};

/*
 * One-shot scale: builds a fresh SwsContext, scales, frees the context.
 * Convenient for tests and infrequent reloads, but rebuilds the filter
 * tables on every call. Use overlay_scaler_scale() when scaling at
 * video rate.
 *
 * Scale a 16-bit RGBA overlay (4 components per pixel, range 0-65535) from
 * src_w x src_h to dst_w x dst_h via libswscale (bicubic). Returns a freshly
 * malloc'd buffer of dst_w * dst_h * 4 uint16_t that the caller must free, or
 * NULL on bad input or libswscale failure. The source buffer is read-only
 * and unchanged on success.
 */
uint16_t *overlay_scale_rgba16(const uint16_t *src,
                               int src_w, int src_h,
                               int dst_w, int dst_h);

/*
 * Reusable scaler with a cached SwsContext. The context is rebuilt only
 * when src/dst dimensions change between calls; for a steady stream of
 * scale operations at the same dimensions (the postprocessor's
 * hot-reload path) the per-call cost drops to one sws_scale().
 */
struct overlay_scaler;

struct overlay_scaler *overlay_scaler_create(enum overlay_scale_filter filter);

/* Allocates a new dst buffer; same return contract as
 * overlay_scale_rgba16(). Convenient for tests; for the postprocess
 * hot path use overlay_scaler_scale_into() to avoid the per-call malloc. */
uint16_t *overlay_scaler_scale(struct overlay_scaler *s,
                               const uint16_t *src,
                               int src_w, int src_h,
                               int dst_w, int dst_h);

/* Scale into a caller-provided dst buffer of size
 * dst_w * dst_h * 4 * sizeof(uint16_t). No allocation in the scaler.
 * Returns true on success. The buffer reuse lets the postprocessor
 * avoid a malloc(33 MB) + free(33 MB) per reload at 4K. */
bool overlay_scaler_scale_into(struct overlay_scaler *s, uint16_t *dst,
                               const uint16_t *src,
                               int src_w, int src_h,
                               int dst_w, int dst_h);

/* No-op when s is NULL. */
void overlay_scaler_destroy(struct overlay_scaler *s);

#ifdef __cplusplus
}
#endif

#endif
