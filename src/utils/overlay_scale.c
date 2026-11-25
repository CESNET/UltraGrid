/**
 * @file   utils/overlay_scale.c
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdlib.h>
#include <string.h>

#include <libavutil/opt.h>
#include <libswscale/swscale.h>

#include "utils/overlay_scale.h"

/* libavutil's RGBA64LE matches our 16-bit RGBA layout (R then G then B then
 * A, native uint16_t per component) on all little-endian targets. There's
 * no codec_t round-trip via get_ug_to_av_pixfmt because UltraGrid has no
 * codec for 16-bit RGBA — the closest is RG48 (no alpha). */
#define RGBA16_PIX_FMT AV_PIX_FMT_RGBA64LE

struct overlay_scaler {
        struct SwsContext *ctx;
        int src_w, src_h, dst_w, dst_h;  /* cache key for ctx */
        int sws_flags;                    /* libswscale filter flag */
};

static int
filter_to_sws_flags(enum overlay_scale_filter f)
{
        switch (f) {
        case OVERLAY_SCALE_NEAREST:        return SWS_POINT;
        case OVERLAY_SCALE_FAST_BILINEAR:  return SWS_FAST_BILINEAR;
        case OVERLAY_SCALE_BILINEAR:       return SWS_BILINEAR;
        case OVERLAY_SCALE_LANCZOS:        return SWS_LANCZOS;
        case OVERLAY_SCALE_BICUBIC:
        default:                           return SWS_BICUBIC;
        }
}

struct overlay_scaler *
overlay_scaler_create(enum overlay_scale_filter filter)
{
        struct overlay_scaler *s = calloc(1, sizeof *s);
        if (s == NULL) return NULL;
        s->sws_flags = filter_to_sws_flags(filter);
        return s;
}

void
overlay_scaler_destroy(struct overlay_scaler *s)
{
        if (s == NULL) return;
        if (s->ctx != NULL) sws_freeContext(s->ctx);
        free(s);
}

bool
overlay_scaler_scale_into(struct overlay_scaler *s, uint16_t *dst,
                          const uint16_t *src,
                          int src_w, int src_h,
                          int dst_w, int dst_h)
{
        if (s == NULL || dst == NULL || src == NULL
            || src_w <= 0 || src_h <= 0
            || dst_w <= 0 || dst_h <= 0) {
                return false;
        }

        if (src_w == dst_w && src_h == dst_h) {
                memcpy(dst, src, (size_t)dst_w * dst_h * 4 * sizeof *src);
                return true;
        }

        if (s->ctx == NULL || src_w != s->src_w || src_h != s->src_h
            || dst_w != s->dst_w || dst_h != s->dst_h) {
                if (s->ctx != NULL) sws_freeContext(s->ctx);
                s->ctx = sws_getContext(
                        src_w, src_h, RGBA16_PIX_FMT,
                        dst_w, dst_h, RGBA16_PIX_FMT,
                        s->sws_flags, NULL, NULL, NULL);
                if (s->ctx == NULL) return false;
                s->src_w = src_w; s->src_h = src_h;
                s->dst_w = dst_w; s->dst_h = dst_h;
        }

        const uint8_t *src_planes[4] = { (const uint8_t *)src, NULL, NULL, NULL };
        const int      src_stride[4] = { src_w * 4 * (int)sizeof *src, 0, 0, 0 };
        uint8_t       *dst_planes[4] = { (uint8_t *)dst,       NULL, NULL, NULL };
        const int      dst_stride[4] = { dst_w * 4 * (int)sizeof *dst, 0, 0, 0 };

        sws_scale(s->ctx, src_planes, src_stride, 0, src_h,
                  dst_planes, dst_stride);
        return true;
}

uint16_t *
overlay_scaler_scale(struct overlay_scaler *s,
                     const uint16_t *src,
                     int src_w, int src_h,
                     int dst_w, int dst_h)
{
        if (dst_w <= 0 || dst_h <= 0) return NULL;
        uint16_t *dst = malloc((size_t)dst_w * dst_h * 4 * sizeof *dst);
        if (dst == NULL) return NULL;
        if (!overlay_scaler_scale_into(s, dst, src, src_w, src_h, dst_w, dst_h)) {
                free(dst);
                return NULL;
        }
        return dst;
}

/* One-shot wrapper: keeps the existing simple-call API with a Lanczos
 * default. Builds and frees a SwsContext on every call — fine for tests
 * and infrequent reloads. Use overlay_scaler_create()+_scale_into for
 * the hot-reload path. */
uint16_t *overlay_scale_rgba16(const uint16_t *src,
                               int src_w, int src_h,
                               int dst_w, int dst_h)
{
        struct overlay_scaler s = {0};
        s.sws_flags = filter_to_sws_flags(OVERLAY_SCALE_BICUBIC);
        uint16_t *dst = overlay_scaler_scale(&s, src, src_w, src_h, dst_w, dst_h);
        if (s.ctx != NULL) sws_freeContext(s.ctx);
        return dst;
}
