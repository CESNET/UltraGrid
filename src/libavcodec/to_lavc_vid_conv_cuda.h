/**
 * @file   libavcodec/to_lavc_vid_conv_cuda.h
 *
 * This file contains CUDA-accelerated conversions from UltraGrid to
 FFmpeg pixel formats.
 * @sa to_lavc_vid_conv.h
 */
/*
 * Copyright (c) 2024 CESNET, z. s. p. o.
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

#ifndef LIBAVCODEC_TO_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9
#define LIBAVCODEC_TO_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9

#include <libavutil/pixfmt.h>
#include <stdio.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "types.h"

struct AVFrame;
struct to_lavc_vid_conv_cuda;

#ifdef __cplusplus
extern "C" {
#endif

/// @note needs to support conversion for all src codec_t
static const enum AVPixelFormat to_lavc_cuda_supp_formats[] = { AV_PIX_FMT_YUV444P };

#ifdef HAVE_CUDA
struct to_lavc_vid_conv_cuda *
to_lavc_vid_conv_cuda_init(codec_t in_pixfmt, int width, int height,
                           enum AVPixelFormat out_pixfmt);
struct AVFrame *to_lavc_vid_conv_cuda(struct to_lavc_vid_conv_cuda *state,
                                      const char                   *in_data);
void to_lavc_vid_conv_cuda_destroy(struct to_lavc_vid_conv_cuda **state);

#else
static struct to_lavc_vid_conv_cuda *
to_lavc_vid_conv_cuda_init(codec_t in_pixfmt, int width, int height,
                           enum AVPixelFormat out_pixfmt)
{
        (void) in_pixfmt, (void) width, (void) height, (void) out_pixfmt;
        fprintf(stderr, "ERROR: CUDA support not compiled in!\n");
        return NULL;
}
static struct AVFrame *
to_lavc_vid_conv_cuda(struct to_lavc_vid_conv_cuda *state, const char *in_data)
{
        (void) state, (void) in_data;
        return NULL;
}

static void
to_lavc_vid_conv_cuda_destroy(struct to_lavc_vid_conv_cuda **state)
{
        (void) state;
}
#endif

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_TO_LAVC_VID_CONV_49C12A96_D7A3_11EE_9446_F0DEF1A0ACC9
