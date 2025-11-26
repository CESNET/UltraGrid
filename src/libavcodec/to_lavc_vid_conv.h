/**
 * @file   libavcodec/to_lavc_vid_conv.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * This file contains conversions from UltraGrid to FFmpeg pixel formats.
 * @sa from_lavc_vid_conv.h
 */
/*
 * Copyright (c) 2013-2025 CESNET, zájmové sdružení právnických osob
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
#include "types.h"             // for codec_t

#ifdef __cplusplus
extern "C" {
#endif

struct to_lavc_vid_conv;
struct to_lavc_vid_conv *to_lavc_vid_conv_init(codec_t in_pixfmt, int width, int height, enum AVPixelFormat out_pixfmt, int thread_count);
struct AVFrame *to_lavc_vid_conv(struct to_lavc_vid_conv *state, char *in_data);
void to_lavc_vid_conv_destroy(struct to_lavc_vid_conv **state);

struct to_lavc_req_prop {
        int subsampling; ///< 4440, 4220, 4200 or 0 (no subsampling explicitly requested)
        int depth;
        int rgb; ///< -1 no request; 0 false; 1 true
        codec_t force_conv_to;  // if non-zero, use only this codec as a target
                                // of UG conversions (before FFMPEG conversion)
                                // or (likely) no conversion at all

};
#define TO_LAVC_REQ_PROP_INIT 0, 0, -1, VIDEO_CODEC_NONE
int get_available_pix_fmts(codec_t in_codec, struct to_lavc_req_prop req_prop,
                           enum AVPixelFormat fmts[AV_PIX_FMT_NB])
    __attribute__((warn_unused_result));

/// @returns colorspace/range details for given av_codec
void get_av_pixfmt_details(enum AVPixelFormat av_codec, enum AVColorSpace *colorspace, enum AVColorRange *color_range);

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_TO_LAVC_VID_CONV_0C22E28C_A3F1_489D_87DC_E56D76E3598B

