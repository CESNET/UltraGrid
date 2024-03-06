/**
 * @file   libavcodec/from_lavc_vid_conv_cuda.h
 *
 * This file contains CUDA-accelerated conversions from FFmpeg to UltraGrid
 * pixfmts.
 * @sa from_lavc_vid_conv.h
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

#ifndef LIBAVCODEC_FROM_LAVC_VID_CONV_CUDA_H_75888464_D7A1_11EE_BE41_F0DEF1A0ACC9
#define LIBAVCODEC_FROM_LAVC_VID_CONV_CUDA_H_75888464_D7A1_11EE_BE41_F0DEF1A0ACC9

struct AVFrame;

#include <libavutil/pixfmt.h>
#include <stdio.h>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "types.h" // codec_t

#ifdef __cplusplus
extern "C" {
#endif

/// @note needs to support conversion for all dst codec_t
static const enum AVPixelFormat from_lavc_cuda_supp_formats[] = {
        AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV444P
};

struct av_to_uv_convert_cuda;

#ifdef HAVE_CUDA
struct av_to_uv_convert_cuda *
get_av_to_uv_cuda_conversion(enum AVPixelFormat av_codec, codec_t uv_codec);
void av_to_uv_convert_cuda(struct av_to_uv_convert_cuda *state,
                           char *__restrict dst_buffer,
                           struct AVFrame *__restrict in_frame, int width,
                           int height, int pitch,
                           const int *__restrict rgb_shift);
void av_to_uv_conversion_cuda_destroy(struct av_to_uv_convert_cuda **state);

#else
static struct av_to_uv_convert_cuda *
get_av_to_uv_cuda_conversion(int av_codec, codec_t uv_codec)
{
        (void) av_codec, (void) uv_codec;
        fprintf(stderr, "ERROR: CUDA support not compiled in!\n");
        return NULL;
}

static void
av_to_uv_convert_cuda(struct av_to_uv_convert_cuda *state,
                      char *__restrict dst_buffer,
                      struct AVFrame *__restrict in_frame, int width,
                      int height, int pitch, const int *__restrict rgb_shift)
{
        (void) state, (void) dst_buffer, (void) in_frame, (void) width,
            (void) height, (void) pitch, (void) rgb_shift;
}

static void
av_to_uv_conversion_cuda_destroy(struct av_to_uv_convert_cuda **state)
{
        (void) state;
}
#endif

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_FROM_LAVC_VID_CONV_CUDA_H_75888464_D7A1_11EE_BE41_F0DEF1A0ACC9

