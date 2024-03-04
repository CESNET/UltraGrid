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
#include <cstddef>
#include <cstdio>
#include <memory>

using std::shared_ptr;

#define TOREMOVE_TEST 1

#include "libavcodec/lavc_common.h"
#include "libavcodec/to_lavc_vid_conv.h"
#include "to_lavc_vid_conv_cuda.h"

struct to_lavc_vid_conv_cuda {
        shared_ptr<struct AVFrame> out_frame;
};

struct to_lavc_vid_conv_cuda *
to_lavc_vid_conv_cuda_init(codec_t in_pixfmt, int width, int height,
                           enum AVPixelFormat out_pixfmt)
{
#ifdef TOREMOVE_TEST
        assert(in_pixfmt == UYVY && out_pixfmt == AV_PIX_FMT_YUV444P);
        auto *s      = new struct to_lavc_vid_conv_cuda();
        s->out_frame = shared_ptr<AVFrame>(
            av_frame_alloc(), [](struct AVFrame *f) { av_frame_free(&f); });
        s->out_frame->pts    = -1;
        s->out_frame->format = out_pixfmt;
        s->out_frame->width  = width;
        s->out_frame->height = height;
        get_av_pixfmt_details(out_pixfmt, &s->out_frame->colorspace,
                              &s->out_frame->color_range);
        int ret = av_frame_get_buffer(s->out_frame.get(), 0);
        assert(ret == 0);
        return s;
#else
        fprintf(stderr, "TODO: implement!\n");
        return nullptr;
#endif
}

struct AVFrame *
to_lavc_vid_conv_cuda(struct to_lavc_vid_conv_cuda *s, const char *in_data)
{
#ifdef TOREMOVE_TEST
        for (size_t y = 0; y < s->out_frame->height; ++y) {
                unsigned char *dst_y =
                    s->out_frame->data[0] + s->out_frame->linesize[0] * y;
                unsigned char *dst_cb =
                    s->out_frame->data[1] + s->out_frame->linesize[1] * y;
                unsigned char *dst_cr =
                    s->out_frame->data[2] + s->out_frame->linesize[2] * y;

                for (int x = 0; x < s->out_frame->width; x += 2) {
                        *dst_cb++ = *in_data;
                        *dst_cb++ = *in_data++;
                        *dst_y++  = *in_data++;
                        *dst_cr++ = *in_data;
                        *dst_cr++ = *in_data++;
                        *dst_y++  = *in_data++;
                }
        }

        return s->out_frame.get();
#else
        return nullptr;
#endif
}

void
to_lavc_vid_conv_cuda_destroy(struct to_lavc_vid_conv_cuda **state)
{
        if (state == nullptr) {
                return;
        }
        delete *state;
        *state = nullptr;
}
