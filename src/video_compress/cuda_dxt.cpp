/**
 * @file   video_compress/cuda_dxt.cpp
 * @author Martin Pulec  <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2013 CESNET z.s.p.o.
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
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video_compress/cuda_dxt.h"

#include "cuda_dxt/cuda_dxt.h"

#include "host.h"
#include "module.h"
#include "video.h"

struct state_video_compress_cuda_dxt {
        state_video_compress_cuda_dxt() {
                memset(&saved_desc, 0, sizeof(saved_desc));
                out[0] = out[1] = NULL;
                in_buffer = NULL;
                cuda_in_buffer = NULL;
                cuda_uyvy_buffer = NULL;
        }
        struct module       module_data;
        struct video_desc   saved_desc;
        char               *in_buffer;      ///< for decoded data
        char               *cuda_uyvy_buffer; ///< same as in_buffer but in device memory
        char               *cuda_in_buffer;  ///< same as in_buffer but in device memory
        struct tile        *out[2];
        codec_t             in_codec;
        codec_t             out_codec;
        decoder_t           decoder;
};

static void cuda_dxt_compress_done(struct module *mod);

struct module *cuda_dxt_compress_init(struct module *parent, const char *fmt)
{
        state_video_compress_cuda_dxt *s =
                new state_video_compress_cuda_dxt;

        s->out_codec = DXT1;

        if (fmt && fmt[0] != '\0') {
                if (strcasecmp(fmt, "DXT5") == 0) {
                        s->out_codec = DXT5;
                } else if (strcasecmp(fmt, "DXT1") == 0) {
                        s->out_codec = DXT1;
                } else {
                        printf("usage:\n"
                               "\t-c cuda_dxt[:DXT1|:DXT5]\n");
                        return NULL;
                }
        }

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = cuda_dxt_compress_done;
        module_register(&s->module_data, parent);

        return &s->module_data;
}

static void cleanup(struct state_video_compress_cuda_dxt *s)
{
        if (s->in_buffer) {
                free(s->in_buffer);
                s->in_buffer = NULL;
        }
        if (s->cuda_uyvy_buffer) {
                cudaFree(s->cuda_uyvy_buffer);
                s->cuda_uyvy_buffer = NULL;
        }
        if (s->cuda_in_buffer) {
                cudaFree(s->cuda_in_buffer);
                s->cuda_in_buffer = NULL;
        }
        for (int i = 0; i < 2; ++i) {
                if (s->out[i] != NULL) {
                        cudaFree(s->out[i]->data);
                        s->out[i]->data = NULL;
                }
        }
}

static bool configure_with(struct state_video_compress_cuda_dxt *s, struct video_desc desc)
{
        cleanup(s);

        if (desc.color_spec == RGB || desc.color_spec == UYVY) {
                s->in_codec = desc.color_spec;
        } else if (get_decoder_from_to(desc.color_spec, RGB, &s->decoder)) {
                s->in_codec = RGB;
        } else if (get_decoder_from_to(desc.color_spec, UYVY, &s->decoder)) {
                s->in_codec = UYVY;
        } else {
                fprintf(stderr, "Unsupported codec: %s\n", get_codec_name(desc.color_spec));
                return false;
        }

        if (s->in_codec == UYVY) {
                if (cudaSuccess != cudaMalloc((void **) &s->cuda_uyvy_buffer,
                                        desc.width * desc.height * 2)) {
                        fprintf(stderr, "Could not allocate CUDA UYVY buffer.\n");
                        return false;
                }
        }

        s->in_buffer = (char *) malloc(desc.width * desc.height * 3);

        if (cudaSuccess != cudaMalloc((void **) &s->cuda_in_buffer,
                                desc.width * desc.height * 3)) {
                fprintf(stderr, "Could not allocate CUDA output buffer.\n");
                return false;
        }

        for (int i = 0; i < 2; ++i) {
                struct video_desc compressed_desc = desc;
                compressed_desc.color_spec = s->out_codec;

                s->out[i] = tile_alloc_desc(compressed_desc);
                s->out[i]->data_len = desc.width * desc.height / (s->out_codec == DXT1 ? 2 : 1);
                if (cudaSuccess != cudaMallocHost((void **) &s->out[i]->data,
                                        s->out[i]->data_len)) {
                        fprintf(stderr, "Could not allocate CUDA output buffer.\n");
                        return false;
                }
        }

        return true;
}

struct tile *cuda_dxt_compress_tile(struct module *mod, struct tile *tx, struct video_desc *desc,
                int buffer)
{
        struct state_video_compress_cuda_dxt *s =
                (struct state_video_compress_cuda_dxt *) mod->priv_data;

        cudaSetDevice(cuda_devices[0]);

        if (!video_desc_eq(*desc, s->saved_desc)) {
                if(configure_with(s, *desc)) {
                        s->saved_desc = *desc;
                } else {
                        fprintf(stderr, "[CUDA DXT] Reconfiguration failed!\n");
                        return NULL;
                }
        }

        char *in_buffer;
        if (desc->color_spec == s->in_codec) {
                in_buffer = tx->data;
        } else {
                unsigned char *line1 = (unsigned char *) tx->data;
                unsigned char *line2 = (unsigned char *) s->in_buffer;

                for (int i = 0; i < (int) tx->height; ++i) {
                        s->decoder(line2, line1, vc_get_linesize(tx->width, s->in_codec),
                                        0, 8, 16);
                        line1 += vc_get_linesize(tx->width, desc->color_spec);
                        line2 += vc_get_linesize(tx->width, s->in_codec);
                }
                in_buffer = s->in_buffer;
        }

        if (s->in_codec == UYVY) {
                if (cudaMemcpy(s->cuda_uyvy_buffer, in_buffer, desc->width * desc->height * 2,
                                        cudaMemcpyHostToDevice) != cudaSuccess) {
                        fprintf(stderr, "Memcpy failed: %s\n",
                                        cudaGetErrorString(cudaGetLastError()));
                        return NULL;
                }
                if (cuda_yuv422_to_yuv444(s->cuda_uyvy_buffer, s->cuda_in_buffer,
                                        desc->width * desc->height, 0) != 0) {
                        fprintf(stderr, "UYVY kernel failed: %s\n",
                                        cudaGetErrorString(cudaGetLastError()));
                }
        } else {
                if (cudaMemcpy(s->cuda_in_buffer, in_buffer, desc->width * desc->height * 3,
                                        cudaMemcpyHostToDevice) != cudaSuccess) {
                        fprintf(stderr, "Memcpy failed: %s\n",
                                        cudaGetErrorString(cudaGetLastError()));
                        return NULL;
                }
        }

        int (*cuda_dxt_enc_func)(const void * src, void * out, int size_x, int size_y, cudaStream_t stream);

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
        int ret = cuda_dxt_enc_func(s->cuda_in_buffer, s->out[buffer]->data, s->saved_desc.width,
                        s->saved_desc.height, 0);
        if (ret != 0) {
                fprintf(stderr, "Encoding failed: %s\n",
                                cudaGetErrorString(cudaGetLastError()));
                return NULL;
        }

        desc->color_spec = s->out_codec;
        return s->out[buffer];
}

static void cuda_dxt_compress_done(struct module *mod)
{
        struct state_video_compress_cuda_dxt *s =
                (struct state_video_compress_cuda_dxt *) mod->priv_data;

        cleanup(s);

        delete s;
}

