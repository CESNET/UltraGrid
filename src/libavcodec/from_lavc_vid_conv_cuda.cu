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

#include <cassert>

#include "debug.h"
#include "video_codec.h"
#include "from_lavc_vid_conv_cuda.h"
#include "libavcodec/lavc_common.h"

struct av_to_uv_convert_cuda {
        enum AVPixelFormat in_codec;
        codec_t out_codec;
};

struct av_to_uv_convert_cuda *
get_av_to_uv_cuda_conversion(enum AVPixelFormat av_codec, codec_t uv_codec)
{
        assert(av_codec == AV_PIX_FMT_YUV422P);
        auto *ret = new struct av_to_uv_convert_cuda();
        log_msg(LOG_LEVEL_VERBOSE, "[%s] converting from %s to %s\n",
                __FILE__, av_get_pix_fmt_name(av_codec),
                get_codec_name(uv_codec));
        ret->in_codec  = av_codec;
        ret->out_codec = uv_codec;
        return ret;
}

void
av_to_uv_convert_cuda(struct av_to_uv_convert_cuda *state,
                      char *__restrict dst_buffer, AVFrame *__restrict in_frame,
                      int width, int height, int pitch,
                      const int *__restrict rgb_shift)
{
        for (size_t y = 0; y < height; ++y) {
                char *src_y =
                    (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cb =
                    (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr =
                    (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;

                for (int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cr++;
                        *dst++ = *src_y++;
                }
        }
}

void
av_to_uv_conversion_cuda_destroy(struct av_to_uv_convert_cuda **s)
{
        if (s == nullptr || *s == nullptr) {
                return;
        }
        delete *s;
        *s = nullptr;
}
