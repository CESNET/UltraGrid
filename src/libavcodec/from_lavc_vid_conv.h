/**
 * @file   libavcodec/from_lavc_vid_conv.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * This file contains conversions from FFmpeg to UltraGrid pixfmts.
 * @sa to_lavc_vid_conv.h
 */
/*
 * Copyright (c) 2013-2023 CESNET, z. s. p. o.
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

#ifndef LIBAVCODEC_FROM_LAVC_VID_CONV_H_97E7417B_773A_453F_BB1A_37841E167152
#define LIBAVCODEC_FROM_LAVC_VID_CONV_H_97E7417B_773A_453F_BB1A_37841E167152

#include "libavcodec/lavc_common.h"

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct av_to_uv_convert_state;
typedef struct av_to_uv_convert_state av_to_uv_convert_t;

av_to_uv_convert_t *get_av_to_uv_conversion(int av_codec, codec_t uv_codec);
void av_to_uv_convert(const av_to_uv_convert_t *s, char * __restrict dst_buffer, AVFrame * __restrict in_frame, int width, int height, int pitch, const int * __restrict rgb_shift);
void parallel_convert(codec_t out_codec, const av_to_uv_convert_t *convert,
                      char *dst, AVFrame *in, int width, int height, int pitch,
                      int rgb_shift[3]);
void av_to_uv_conversion_destroy(av_to_uv_convert_t **);

codec_t get_best_ug_codec_to_av(const enum AVPixelFormat *fmt, bool use_hwaccel);
enum AVPixelFormat lavd_get_av_to_ug_codec(const enum AVPixelFormat *fmt, codec_t c, bool use_hwaccel);
enum AVPixelFormat pick_av_convertible_to_ug(codec_t              color_spec,
                                             av_to_uv_convert_t **av_conv);

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_FROM_LAVC_VID_CONV_H_97E7417B_773A_453F_BB1A_37841E167152

