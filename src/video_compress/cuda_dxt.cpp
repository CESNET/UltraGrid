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
#include "cuda_wrapper.h"

#include "host.h"
#include "module.h"
#include "video.h"
#include "video_compress.h"

struct state_video_compress_cuda_dxt {
        state_video_compress_cuda_dxt() {
                memset(&saved_desc, 0, sizeof(saved_desc));
                out[0] = out[1] = NULL;
                in_buffer = NULL;
                cuda_in_buffer = NULL;
                cuda_uyvy_buffer = NULL;
                cuda_out_buffer = NULL;
        }
        struct module       module_data;
        struct video_desc   saved_desc;
        char               *in_buffer;      ///< for decoded data
        char               *cuda_uyvy_buffer; ///< same as in_buffer but in device memory
        char               *cuda_in_buffer;  ///< same as in_buffer but in device memory
        char               *cuda_out_buffer;  ///< same as in_buffer but in device memory
        struct video_frame *out[2];
        codec_t             in_codec;
        codec_t             out_codec;
        decoder_t           decoder;
};

static void cuda_dxt_compress_done(struct module *mod);

struct module *cuda_dxt_compress_init(struct module *parent,
                const struct video_compress_params *params)
{
        state_video_compress_cuda_dxt *s =
                new state_video_compress_cuda_dxt;
        const char *fmt = params->cfg;
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
        for (int i = 0; i < 2; ++i) {
                if (s->out[i] != NULL) {
                        cuda_wrapper_free(s->out[i]->tiles[0].data);
                        s->out[i]->tiles[0].data = NULL;
                }
        }
}

static bool configure_with(struct state_video_compress_cuda_dxt *s, struct video_desc desc)
{
        cleanup(s);

        if (desc.color_spec == RGB || desc.color_spec == UYVY) {
                s->in_codec = desc.color_spec;
        } else if ((s->decoder = get_decoder_from_to(desc.color_spec, RGB))) {
                s->in_codec = RGB;
        } else if ((s->decoder = get_decoder_from_to(desc.color_spec, UYVY))) {
                s->in_codec = UYVY;
        } else {
                fprintf(stderr, "Unsupported codec: %s\n", get_codec_name(desc.color_spec));
                return false;
        }

        if (s->in_codec == UYVY) {
                if (CUDA_WRAPPER_SUCCESS != cuda_wrapper_malloc((void **) &s->cuda_uyvy_buffer,
                                        desc.width * desc.height * 2)) {
                        fprintf(stderr, "Could not allocate CUDA UYVY buffer.\n");
                        return false;
                }
        }

        s->in_buffer = (char *) malloc(desc.width * desc.height * 3);

        if (CUDA_WRAPPER_SUCCESS != cuda_wrapper_malloc((void **) &s->cuda_in_buffer,
                                desc.width * desc.height * 3)) {
                fprintf(stderr, "Could not allocate CUDA output buffer.\n");
                return false;
        }

        for (int i = 0; i < 2; ++i) {
                struct video_desc compressed_desc = desc;
                compressed_desc.color_spec = s->out_codec;
                compressed_desc.tile_count = 1;

                s->out[i] = vf_alloc_desc(compressed_desc);
                s->out[i]->tiles[0].data_len = desc.width * desc.height / (s->out_codec == DXT1 ? 2 : 1);
                if (CUDA_WRAPPER_SUCCESS != cuda_wrapper_malloc_host((void **) &s->out[i]->tiles[0].data,
                                        s->out[i]->tiles[0].data_len)) {
                        fprintf(stderr, "Could not allocate CUDA output host buffer.\n");
                        return false;
                }
        }
        if (CUDA_WRAPPER_SUCCESS != cuda_wrapper_malloc((void **)
                                &s->cuda_out_buffer,
                                s->out[0]->tiles[0].data_len)) {
                fprintf(stderr, "Could not allocate CUDA output buffer.\n");
                return false;
        }

        return true;
}

struct video_frame *cuda_dxt_compress_tile(struct module *mod, struct video_frame *tx,
                int tile_idx, int buffer)
{
        struct state_video_compress_cuda_dxt *s =
                (struct state_video_compress_cuda_dxt *) mod->priv_data;

        cuda_wrapper_set_device(cuda_devices[0]);

        if (!video_desc_eq_excl_param(video_desc_from_frame(tx),
                                s->saved_desc, PARAM_TILE_COUNT)) {
                if(configure_with(s, video_desc_from_frame(tx))) {
                        s->saved_desc = video_desc_from_frame(tx);
                } else {
                        fprintf(stderr, "[CUDA DXT] Reconfiguration failed!\n");
                        return NULL;
                }
        }

        char *in_buffer;
        if (tx->color_spec == s->in_codec) {
                in_buffer = tx->tiles[tile_idx].data;
        } else {
                unsigned char *line1 = (unsigned char *) tx->tiles[tile_idx].data;
                unsigned char *line2 = (unsigned char *) s->in_buffer;

                for (int i = 0; i < (int) tx->tiles[tile_idx].height; ++i) {
                        s->decoder(line2, line1, vc_get_linesize(tx->tiles[tile_idx].width,
                                                s->in_codec), 0, 8, 16);
                        line1 += vc_get_linesize(tx->tiles[tile_idx].width, tx->color_spec);
                        line2 += vc_get_linesize(tx->tiles[tile_idx].width, s->in_codec);
                }
                in_buffer = s->in_buffer;
        }

        if (s->in_codec == UYVY) {
                if (cuda_wrapper_memcpy(s->cuda_uyvy_buffer, in_buffer, tx->tiles[tile_idx].width *
                                        tx->tiles[tile_idx].height * 2,
                                        CUDA_WRAPPER_MEMCPY_HOST_TO_DEVICE) != CUDA_WRAPPER_SUCCESS) {
                        fprintf(stderr, "Memcpy failed: %s\n", cuda_wrapper_last_error_string());
                        return NULL;
                }
                if (cuda_yuv422_to_yuv444(s->cuda_uyvy_buffer, s->cuda_in_buffer,
                                        tx->tiles[tile_idx].width *
                                        tx->tiles[tile_idx].height, 0) != CUDA_WRAPPER_SUCCESS) {
                        fprintf(stderr, "Kernel failed: %s\n", cuda_wrapper_last_error_string());
                }
        } else {
                if (cuda_wrapper_memcpy(s->cuda_in_buffer, in_buffer, tx->tiles[tile_idx].width *
                                        tx->tiles[tile_idx].height * 3,
                                        CUDA_WRAPPER_MEMCPY_HOST_TO_DEVICE) != CUDA_WRAPPER_SUCCESS) {
                        fprintf(stderr, "Memcpy failed: %s\n", cuda_wrapper_last_error_string());
                        return NULL;
                }
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
        int ret = cuda_dxt_enc_func(s->cuda_in_buffer, s->cuda_out_buffer,
                        s->saved_desc.width, s->saved_desc.height, 0);
        if (ret != 0) {
                fprintf(stderr, "Encoding failed: %s\n", cuda_wrapper_last_error_string());
                return NULL;
        }

        if (cuda_wrapper_memcpy(s->out[buffer]->tiles[0].data,
                                s->cuda_out_buffer,
                                s->out[buffer]->tiles[0].data_len,
                                CUDA_WRAPPER_MEMCPY_DEVICE_TO_HOST) != CUDA_WRAPPER_SUCCESS) {
                fprintf(stderr, "Memcpy failed: %s\n", cuda_wrapper_last_error_string());
                return NULL;
        }

        return s->out[buffer];
}

static void cuda_dxt_compress_done(struct module *mod)
{
        struct state_video_compress_cuda_dxt *s =
                (struct state_video_compress_cuda_dxt *) mod->priv_data;

        cleanup(s);

        delete s;
}

