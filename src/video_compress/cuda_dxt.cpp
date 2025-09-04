/**
 * @file   video_compress/cuda_dxt.cpp
 * @author Martin Pulec  <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2025 CESNET
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

#include <cstdio>                      // for printf, stderr
#include <cstdlib>                    // for free, malloc
#include <memory>                     // for shared_ptr

#include "../cuda_dxt/cuda_dxt.h"
#include "compat/strings.h"            // for strcasecmp
#include "cuda_wrapper.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "pixfmt_conv.h"               // for get_decoder_from_to, decoder_t
#include "types.h"                     // for tile, video_frame, video_desc
#include "utils/video_frame_pool.h"
#include "video_codec.h"               // for codec_is_a_rgb, get_bits_per_c...
#include "video_compress.h"
#include "video_frame.h"               // for vf_get_tile, video_desc_from_f...

#define MOD_NAME "[CUDA DXT] "

#define CHECK_CUDA(cmd, msg, action) \
        if ((cmd) != CUDA_WRAPPER_SUCCESS) { \
                MSG(ERROR, "%s: %s\n", msg, cuda_wrapper_last_error_string()); \
                action; \
        }


using namespace std;

namespace {

struct cuda_buffer_data_allocator : public video_frame_pool_allocator {
        void *allocate(size_t size) override {
                void *ptr;
                if (CUDA_WRAPPER_SUCCESS != cuda_wrapper_malloc_host(&ptr,
                                        size)) {
                        return NULL;
                }
                return ptr;
        }
        void deallocate(void *ptr) override {
                cuda_wrapper_free(ptr);
        }
        video_frame_pool_allocator *clone() const override {
                return new cuda_buffer_data_allocator(*this);
        }
};

struct state_video_compress_cuda_dxt {
        struct video_desc   saved_desc;
        char               *in_buffer;      ///< for decoded data
        char               *cuda_uyvy_buffer; ///< same as in_buffer but in device memory
        char               *cuda_in_buffer;  ///< same as in_buffer but in device memory
        char               *cuda_out_buffer;  ///< same as in_buffer but in device memory
        codec_t             in_codec;
        codec_t             out_codec;
        decoder_t           decoder;

        video_frame_pool pool{0, cuda_buffer_data_allocator()};
};

static void cuda_dxt_compress_done(void *state);

void *cuda_dxt_compress_init(struct module *parent,
                const char *fmt)
{
        (void) parent;
        state_video_compress_cuda_dxt *s =
                new state_video_compress_cuda_dxt();
        s->out_codec = DXT1;

        if (fmt && fmt[0] != '\0') {
                if (strcasecmp(fmt, "DXT5") == 0) {
                        s->out_codec = DXT5;
                } else if (strcasecmp(fmt, "DXT1") == 0) {
                        s->out_codec = DXT1;
                } else {
                        printf("usage:\n"
                               "\t-c cuda_dxt[:DXT1|:DXT5]\n");
                        delete s;
                        return NULL;
                }
        }

        return s;
}

static void cleanup(struct state_video_compress_cuda_dxt *s)
{
        if (s->in_buffer) {
                free(s->in_buffer);
                s->in_buffer = NULL;
        }
        if (s->cuda_uyvy_buffer) {
                cuda_wrapper_free(s->cuda_uyvy_buffer);
                s->cuda_uyvy_buffer = NULL;
        }
        if (s->cuda_in_buffer) {
                cuda_wrapper_free(s->cuda_in_buffer);
                s->cuda_in_buffer = NULL;
        }
        if (s->cuda_out_buffer) {
                cuda_wrapper_free(s->cuda_out_buffer);
                s->cuda_out_buffer = NULL;
        }
}

static bool configure_with(struct state_video_compress_cuda_dxt *s, struct video_desc desc)
{
        cleanup(s);

        if (get_bits_per_component(desc.color_spec) > 8) {
                LOG(LOG_LEVEL_NOTICE) << "[CUDA DXT] Converting from " << get_bits_per_component(desc.color_spec) <<
                        " to 8 bits. You may directly capture 8-bit signal to improve performance.\n";
        }

        codec_t supported_codecs[] = { RGB, UYVY, VIDEO_CODEC_NONE };
        s->decoder = get_best_decoder_from(desc.color_spec, supported_codecs, &s->in_codec);
        if (!s->decoder) {
                MSG(ERROR, "Unsupported codec: %s\n",
                    get_codec_name(desc.color_spec));
                return false;
        }

        if (s->in_codec == UYVY) {
                CHECK_CUDA(cuda_wrapper_malloc((void **) &s->cuda_uyvy_buffer,
                                               desc.width * desc.height * 2),
                           "Could not allocate CUDA UYVY buffer", return false);
        }

        s->in_buffer = (char *) malloc(desc.width * desc.height * 3);

        CHECK_CUDA(cuda_wrapper_malloc((void **) &s->cuda_in_buffer,
                                       desc.width * desc.height * 3),
                   "Could not allocate CUDA output buffer", return false);

        struct video_desc compressed_desc = desc;
        compressed_desc.color_spec = s->out_codec;
        compressed_desc.tile_count = 1;
        size_t data_len = desc.width * desc.height / (s->out_codec == DXT1 ? 2 : 1);

        s->pool.reconfigure(compressed_desc, data_len);

        CHECK_CUDA(cuda_wrapper_malloc((void **) &s->cuda_out_buffer, data_len),
                   "Could not allocate CUDA output buffer", return false);

        return true;
}

shared_ptr<video_frame> cuda_dxt_compress_tile(void *state, shared_ptr<video_frame> tx)
{
        if (!tx) {
                return {};
        }

        auto *s = (struct state_video_compress_cuda_dxt *) state;

        cuda_wrapper_set_device(cuda_devices[0]);

        if (!video_desc_eq_excl_param(video_desc_from_frame(tx.get()),
                                s->saved_desc, PARAM_TILE_COUNT)) {
                if(configure_with(s, video_desc_from_frame(tx.get()))) {
                        s->saved_desc = video_desc_from_frame(tx.get());
                } else {
                        MSG(ERROR, "Reconfiguration failed!\n");
                        return NULL;
                }
        }

        char *in_buffer;
        if (tx->color_spec == s->in_codec) {
                in_buffer = tx->tiles[0].data;
        } else {
                unsigned char *line1 = (unsigned char *) tx->tiles[0].data;
                unsigned char *line2 = (unsigned char *) s->in_buffer;

                for (int i = 0; i < (int) tx->tiles[0].height; ++i) {
                        s->decoder(line2, line1, vc_get_linesize(tx->tiles[0].width,
                                                s->in_codec), 0, 8, 16);
                        line1 += vc_get_linesize(tx->tiles[0].width, tx->color_spec);
                        line2 += vc_get_linesize(tx->tiles[0].width, s->in_codec);
                }
                in_buffer = s->in_buffer;
        }

        if (s->in_codec == UYVY) {
                CHECK_CUDA(cuda_wrapper_memcpy(
                               s->cuda_uyvy_buffer, in_buffer,
                               tx->tiles[0].width * tx->tiles[0].height * 2,
                               CUDA_WRAPPER_MEMCPY_HOST_TO_DEVICE),
                           "Memcpy failed", return nullptr);

                CHECK_CUDA(cuda_yuv422_to_yuv444(
                               s->cuda_uyvy_buffer, s->cuda_in_buffer,
                               tx->tiles[0].width * tx->tiles[0].height, 0),
                           "Kernel failed", return nullptr);
        } else {
                CHECK_CUDA(cuda_wrapper_memcpy(
                               s->cuda_in_buffer, in_buffer,
                               tx->tiles[0].width * tx->tiles[0].height * 3,
                               CUDA_WRAPPER_MEMCPY_HOST_TO_DEVICE),
                           "Memcpy failed", return nullptr);
        }

        int (*cuda_dxt_enc_func)(const void * src, void * out, int size_x, int size_y,
                        cuda_wrapper_stream_t stream);

        if (s->out_codec == DXT1) {
                if (s->in_codec == RGB) {
                        cuda_dxt_enc_func = cuda_rgb_to_dxt1;
                } else {
                        cuda_dxt_enc_func = cuda_yuv_to_dxt1;
                }
        } else {
                if (s->in_codec == RGB) {
                        cuda_dxt_enc_func = cuda_rgb_to_dxt6;
                } else {
                        cuda_dxt_enc_func = cuda_yuv_to_dxt6;
                }
        }
        CHECK_CUDA(cuda_dxt_enc_func(s->cuda_in_buffer, s->cuda_out_buffer,
                                     s->saved_desc.width, s->saved_desc.height,
                                     0),
                   "Encoding failed", return nullptr);

        shared_ptr<video_frame> out = s->pool.get_frame();
        CHECK_CUDA(cuda_wrapper_memcpy(out->tiles[0].data, s->cuda_out_buffer,
                                       out->tiles[0].data_len,
                                       CUDA_WRAPPER_MEMCPY_DEVICE_TO_HOST),
                   "Memcpy failed", return nullptr);

        return out;
}

static void cuda_dxt_compress_done(void *state)
{
        auto *s = (struct state_video_compress_cuda_dxt *) state;
        cleanup(s);
        delete s;
}

const struct video_compress_info cuda_dxt_info = {
        cuda_dxt_compress_init,
        cuda_dxt_compress_done,
        NULL,
        cuda_dxt_compress_tile,
        NULL,
        NULL,
        NULL,
        NULL,
        NULL
};

REGISTER_MODULE(cuda_dxt, &cuda_dxt_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace

