/**
 * @file   libavcodec/to_lavc_vid_conv.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2022 CESNET, z. s. p. o.
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

#ifndef LIBAVCODEC_TO_LAVC_VID_CONV_0C22E28C_A3F1_489D_87DC_E56D76E3598B
#define LIBAVCODEC_TO_LAVC_VID_CONV_0C22E28C_A3F1_489D_87DC_E56D76E3598B

#include "libavcodec/lavc_common.h"
#include "video_codec.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void uv_to_av_convert(AVFrame * __restrict out_frame, const unsigned char * __restrict in_data, int width, int height);
typedef uv_to_av_convert *pixfmt_callback_t;

int get_available_pix_fmts(codec_t in_codec, int requested_subsampling, codec_t force_conv_to, enum AVPixelFormat fmts[AV_PIX_FMT_NB]);
decoder_t get_decoder_from_uv_to_uv(codec_t in, enum AVPixelFormat av, codec_t *out);
pixfmt_callback_t select_pixfmt_callback(enum AVPixelFormat fmt, codec_t src);

pixfmt_callback_t get_uv_to_av_conversion(codec_t uv_codec, int av_codec);

/**
 * Returns AV format details for given pair UV,AV codec (must be unique then)
 */
void get_av_pixfmt_details(codec_t uv_codec, int av_codec, enum AVColorSpace *colorspace, enum AVColorRange *color_range);

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_TO_LAVC_VID_CONV_0C22E28C_A3F1_489D_87DC_E56D76E3598B

