/**
 * @file   libavcodec/from_lavc_vid_conv.h
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

#ifndef LIBAVCODEC_FROM_LAVC_VID_CONV_H_97E7417B_773A_453F_BB1A_37841E167152
#define LIBAVCODEC_FROM_LAVC_VID_CONV_H_97E7417B_773A_453F_BB1A_37841E167152

#include "libavcodec_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void av_to_uv_convert(char * __restrict dst_buffer, AVFrame * __restrict in_frame, int width, int height, int pitch, const int * __restrict rgb_shift);
typedef av_to_uv_convert *av_to_uv_convert_p;

struct av_to_uv_conversion {
        int av_codec;
        codec_t uv_codec;
        av_to_uv_convert_p convert;
        bool native; ///< there is a 1:1 mapping between the FFMPEG and UV codec (matching
                     ///< color space, channel count (w/wo alpha), bit-depth,
                     ///< subsampling etc.). Supported out are: RGB, UYVY, v210 (in future
                     ///< also 10,12 bit RGB). Subsampling doesn't need to be respected (we do
                     ///< not have codec for eg. 4:4:4 UYVY).
};

av_to_uv_convert_p get_av_to_uv_conversion(int av_codec, codec_t uv_codec);
const struct av_to_uv_conversion *get_av_to_uv_conversions(void);

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_FROM_LAVC_VID_CONV_H_97E7417B_773A_453F_BB1A_37841E167152

