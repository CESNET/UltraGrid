/**
 * @file   video_decompress/gpujpeg.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2020 CESNET, z. s. p. o.
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
#include "debug.h"
#include "host.h"
#include "video.h"
#include "video_decompress.h"

#include <cuda_runtime.h>
#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_version.h>
//#include "compat/platform_semaphore.h"
#include <pthread.h>
#include <stdlib.h>
#include "lib_common.h"

#define MOD_NAME "[GPUJPEG dec.] "

struct state_decompress_gpujpeg {
        struct gpujpeg_decoder *decoder;

        struct video_desc desc;
        int rshift, gshift, bshift;
        int pitch;
        codec_t out_codec;

        uint8_t *cuda_tmp_buf;
        bool unstripe;
};

static int configure_with(struct state_decompress_gpujpeg *s, struct video_desc desc);

static int configure_with(struct state_decompress_gpujpeg *s, struct video_desc desc)
{
        s->desc = desc;

        s->decoder = gpujpeg_decoder_create(NULL);
        if(!s->decoder) {
                return FALSE;
        }

        // setting verbosity - a bit tricky now, gpujpeg_decoder_init needs to be called with some "valid" data
        // otherwise, parameter setting is unneeded - it is done automaticaly by the image
        struct gpujpeg_parameters param;
        gpujpeg_set_default_parameters(&param);
        param.color_space_internal = GPUJPEG_YCBCR_BT709; // see comment bellow
        param.verbose = MAX(0, log_level - LOG_LEVEL_INFO);
        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);
        param_image.width = desc.width; // size must be non-zero in order the init to succeed
        param_image.height = desc.height;
        param_image.color_space = GPUJPEG_YCBCR_BT709; // assume now BT.709 as default - this is mainly applicable for FFmpeg-encoded
                                                       // JPEGs that doesn't indicate explicitly color spec (no JFIF marker, only CS=ITU601
                                                       // for BT.601 limited range - not enabled by UG encoder because FFmpeg emits it also for 709)
        int rc = gpujpeg_decoder_init(s->decoder, &param, &param_image);
        assert(rc == 0);

        switch (s->out_codec) {
        case CUDA_I420:
        case I420:
                gpujpeg_decoder_set_output_format(s->decoder, GPUJPEG_YCBCR_BT709,
                                GPUJPEG_420_U8_P0P1P2);
                break;
        case CUDA_RGBA:
        case RGBA:
                gpujpeg_decoder_set_output_format(s->decoder, GPUJPEG_RGB,
                                s->out_codec == CUDA_RGBA || (s->out_codec == RGBA && s->rshift == 0 && s->gshift == 8 && s->bshift == 16 && vc_get_linesize(desc.width, RGBA) == s->pitch) ?
                                GPUJPEG_444_U8_P012A : GPUJPEG_444_U8_P012);
                break;
        case RGB:
                gpujpeg_decoder_set_output_format(s->decoder, GPUJPEG_RGB,
                                GPUJPEG_444_U8_P012);
                break;
        case UYVY:
                gpujpeg_decoder_set_output_format(s->decoder, GPUJPEG_YCBCR_BT709,
                                GPUJPEG_422_U8_P1020);
                break;
        case VIDEO_CODEC_NONE:
                break;
        default:
                assert("Invalid codec!" && 0);
        }

        if (cudaMalloc((void **) &s->cuda_tmp_buf, desc.width * desc.height * 4) != cudaSuccess) {
                log_msg(LOG_LEVEL_WARNING, "Cannot allocate CUDA buffer!\n");
        }

        return TRUE;
}

static void * gpujpeg_decompress_init(void)
{
        if (gpujpeg_version() >> 8 != GPUJPEG_VERSION_INT >> 8) {
                char ver_req[128] = "";
                char ver_lib[128] = "";
                strncpy(ver_req, gpujpeg_version_to_string(GPUJPEG_VERSION_INT), sizeof ver_req - 1);
                strncpy(ver_lib, gpujpeg_version_to_string(gpujpeg_version()), sizeof ver_lib - 1);
                log_msg(LOG_LEVEL_WARNING, "GPUJPEG API version mismatch! (compiled: %s, library present: %s, required same minor version)\n",
                                ver_req, ver_lib);
        }

        struct state_decompress_gpujpeg *s = (struct state_decompress_gpujpeg *) calloc(1, sizeof(struct state_decompress_gpujpeg));

        int ret;
        printf("Initializing CUDA device %d...\n", cuda_devices[0]);
        ret = gpujpeg_init_device(cuda_devices[0], TRUE);
        if(ret != 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "initializing CUDA device %d failed.\n", cuda_devices[0]);
                free(s);
                return NULL;
        }

        if (get_commandline_param("unstripe") != NULL) {
                s->unstripe = true;
        }

        return s;
}

static int gpujpeg_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_gpujpeg *s = (struct state_decompress_gpujpeg *) state;
        
        assert(out_codec == I420 || out_codec == RGB || out_codec == RGBA
                        || out_codec == UYVY || out_codec == VIDEO_CODEC_NONE
                        || out_codec == CUDA_I420 || out_codec == CUDA_RGBA);

        if(s->out_codec == out_codec &&
                        s->pitch == pitch &&
                        s->rshift == rshift &&
                        s->gshift == gshift &&
                        s->bshift == bshift &&
                        video_desc_eq_excl_param(s->desc, desc, PARAM_INTERLACING)) {
                return TRUE;
        } else {
                s->out_codec = out_codec;
                s->pitch = pitch;
                s->rshift = rshift;
                s->gshift = gshift;
                s->bshift = bshift;
                if(s->decoder) {
                        gpujpeg_decoder_destroy(s->decoder);
                }
                cudaFree(s->cuda_tmp_buf);
                s->cuda_tmp_buf = NULL;
                return configure_with(s, desc);
        }
}

static decompress_status gpujpeg_probe_internal_codec(unsigned char *buffer, size_t len, codec_t *internal_codec) {
        *internal_codec = VIDEO_CODEC_NONE;
	struct gpujpeg_image_parameters params = { 0 };
	if (gpujpeg_decoder_get_image_info(buffer, len, &params, NULL, MAX(0, log_level - LOG_LEVEL_INFO)) != 0) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "probe - cannot get image info!\n");
		return DECODER_GOT_FRAME;
	}

	if (!params.color_space) {
                params.color_space = GPUJPEG_YCBCR_BT601_256LVLS;
	}

	switch ( params.color_space ) {
	case GPUJPEG_RGB:
		*internal_codec = RGB;
		break;
	case GPUJPEG_YUV:
	case GPUJPEG_YCBCR_BT601:
	case GPUJPEG_YCBCR_BT601_256LVLS:
	case GPUJPEG_YCBCR_BT709:
                *internal_codec = params.pixel_format == GPUJPEG_420_U8_P0P1P2 ? I420 : UYVY;
		break;
	default:
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "probe - unhandled color space: %s\n",
                                gpujpeg_color_space_get_name(params.color_space));
		return DECODER_GOT_FRAME;
	}

	log_msg(LOG_LEVEL_VERBOSE, "JPEG color space: %s\n", gpujpeg_color_space_get_name(params.color_space));
	return DECODER_GOT_CODEC;
}

#define IS_I420(c) (c == I420 || c == CUDA_I420)

static decompress_status gpujpeg_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, codec_t *internal_codec,
                const int *pitches)
{
        UNUSED(frame_seq);
        UNUSED(callbacks);
        struct state_decompress_gpujpeg *s = (struct state_decompress_gpujpeg *) state;
        int ret;
        struct gpujpeg_decoder_output decoder_output;
        int linesize;

        if (s->out_codec == VIDEO_CODEC_NONE) {
                return gpujpeg_probe_internal_codec(buffer, src_len, internal_codec);
        }

        linesize = vc_get_linesize(s->desc.width, s->out_codec);
        
        gpujpeg_set_device(cuda_devices[0]);

        if (s->unstripe) {
                int pitches_buf[4];
                if (!pitches) {
                        pitches = pitches_buf;
                        if (IS_I420(s->out_codec)) {
                                pitches_buf[0] = s->desc.width * 8;
                                pitches_buf[1] =
                                        pitches_buf[2] = s->desc.width * 8 / 2;
                        } else { // (CUDA_)RGBA
                                pitches_buf[0] = 4 * s->desc.width * 8;
                        }
                }
                assert(s->cuda_tmp_buf != NULL);
                gpujpeg_decoder_output_set_custom_cuda (&decoder_output, s->cuda_tmp_buf);
                if (gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output) != 0) {
                        return DECODER_NO_FRAME;
                }
                if (!IS_I420(s->out_codec)) {
                        for (int i = 0; i < 8; ++i) {
                                if (cudaMemcpy2D(dst + 4 * i * s->desc.width, pitches[0],
                                                        s->cuda_tmp_buf + i * 4 * s->desc.width * (s->desc.height / 8), 4 * s->desc.width, 4 * s->desc.width, s->desc.height / 8, cudaMemcpyDefault) != cudaSuccess) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "cudaMemcpy2D failed: %s!\n", cudaGetErrorString(cudaGetLastError()));
                                }
                        }
                } else { ///@todo irregular dimensions (%16 != 0)
                        unsigned char *src = s->cuda_tmp_buf;
                        for (int i = 0; i < 8; ++i) { // y
                                if (cudaMemcpy2D(dst + i * s->desc.width, s->desc.width * 8,
                                                        src + i * s->desc.width * (s->desc.height / 8), s->desc.width, s->desc.width, s->desc.height / 8, cudaMemcpyDefault) != cudaSuccess) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "cudaMemcpy2D failed: %s!\n", cudaGetErrorString(cudaGetLastError()));
                                }
                        }
                        src += s->desc.width * s->desc.height;
                        dst += pitches[0] * (s->desc.height / 8);
                        for (int n = 0; n < 2; ++n) { // uv
                                for (int i = 0; i < 8; ++i) { // u
                                        if (cudaMemcpy2D(dst + i * s->desc.width / 2, pitches[1],
                                                                src + i * s->desc.width / 2 * (s->desc.height / 2 / 8), s->desc.width / 2, s->desc.width / 2, s->desc.height / 2 / 8, cudaMemcpyDefault) != cudaSuccess) {
                                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "cudaMemcpy2D failed: %s!\n", cudaGetErrorString(cudaGetLastError()));
                                        }
                                }
                                src += (s->desc.width / 2) * (s->desc.height / 2);
                                dst += pitches[1] * (s->desc.height / 8 / 2);
                        }
                }
        } else if (pitches != NULL) {
                assert(s->out_codec == I420 || s->out_codec == CUDA_I420);
                assert(s->cuda_tmp_buf != NULL);
                gpujpeg_decoder_output_set_custom_cuda (&decoder_output, s->cuda_tmp_buf);
                if (gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output) != 0) {
                        return DECODER_NO_FRAME;
                }
                if (cudaMemcpy2D(dst, pitches[0],
                                        s->cuda_tmp_buf, s->desc.width,
                                        s->desc.width, s->desc.height,
                                        cudaMemcpyDefault) != cudaSuccess) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "cudaMemcpy2D Y failed: %s!\n", cudaGetErrorString(cudaGetLastError()));
                }
                if (cudaMemcpy2D(dst + pitches[0] * s->desc.height, pitches[1],
                                        s->cuda_tmp_buf + s->desc.width * s->desc.height, (s->desc.width + 1) / 2,
                                        (s->desc.width + 1) / 2, (s->desc.height + 1) / 2,
                                        cudaMemcpyDefault) != cudaSuccess) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "cudaMemcpy2D Cb failed: %s!\n", cudaGetErrorString(cudaGetLastError()));
                }
                if (cudaMemcpy2D(dst + pitches[0] * s->desc.height + pitches[1] * ((s->desc.height + 1) / 2), pitches[2],
                                        s->cuda_tmp_buf + s->desc.width * s->desc.height + ((s->desc.width + 1) / 2) * ((s->desc.height + 1) / 2), (s->desc.width + 1) / 2,
                                        (s->desc.width + 1) / 2, (s->desc.height + 1) / 2,
                                        cudaMemcpyDefault) != cudaSuccess) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "cudaMemcpy2D Cr failed: %s!\n", cudaGetErrorString(cudaGetLastError()));
                }

        } else if (s->out_codec == CUDA_I420 || s->out_codec == CUDA_RGBA) {
                gpujpeg_decoder_output_set_custom_cuda (&decoder_output, dst);
                if (gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output) != 0) {
                        return DECODER_NO_FRAME;
                }
        } else if (s->pitch == linesize && (s->out_codec == UYVY || s->out_codec == RGB
                                || (s->out_codec == RGBA && s->rshift == 0 && s->gshift == 8 && s->bshift == 16)
                        )) {
                gpujpeg_decoder_output_set_custom(&decoder_output, dst);
                //int data_decompressed_size = decoder_output.data_size;
                    
                ret = gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output);
                if (ret != 0) return DECODER_NO_FRAME;
        } else {
                unsigned char *line_src, *line_dst;
                
                gpujpeg_decoder_output_set_default(&decoder_output);
                decoder_output.type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
                //int data_decompressed_size = decoder_output.data_size;
                    
                ret = gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output);

                if (ret != 0) return DECODER_NO_FRAME;
                
                line_dst = dst;
                line_src = decoder_output.data;
                for (unsigned i = 0u; i < s->desc.height; i++) {
                        if (s->out_codec == RGBA) {
                                vc_copylineRGBtoRGBA(line_dst, line_src, linesize,
                                                s->rshift, s->gshift, s->bshift);
                        } else {
                                assert(s->out_codec == UYVY || s->out_codec == I420);
                                memcpy(line_dst, line_src, linesize);
                        }
                                
                        line_dst += s->pitch;
                        line_src += vc_get_linesize(s->desc.width, s->out_codec);
                }
        }

        return DECODER_GOT_FRAME;
}

static int gpujpeg_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decompress *s = (struct state_decompress *) state;
        UNUSED(s);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = FALSE;
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

static void gpujpeg_decompress_done(void *state)
{
        struct state_decompress_gpujpeg *s = (struct state_decompress_gpujpeg *) state;

        if(s->decoder) {
                gpujpeg_decoder_destroy(s->decoder);
        }
        cudaFree(s->cuda_tmp_buf);
        free(s);
}

static const struct decode_from_to *gpujpeg_decompress_get_decoders() {
        static const struct decode_from_to ret[] = {
		{ JPEG, VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 50 }, // for probe
		{ JPEG, RGB, RGB, 300 },
		{ JPEG, RGB, RGBA, 300  },
		{ JPEG, UYVY, UYVY, 300 },
		{ JPEG, I420, I420, 300 },
		{ JPEG, I420, UYVY, 500 },
		{ JPEG, RGB, UYVY, 700 },
		{ JPEG, UYVY, RGB, 700 },
		{ JPEG, UYVY, RGBA, 700 },
		{ JPEG, VIDEO_CODEC_NONE, RGB, 900 },
		{ JPEG, VIDEO_CODEC_NONE, UYVY, 900 },
		{ JPEG, VIDEO_CODEC_NONE, RGBA, 900 },
#if GPUJPEG_VERSION_INT >= GPUJPEG_MK_VERSION_INT(0, 13, 0)
		{ JPEG, VIDEO_CODEC_NONE, I420, 900 },
#endif
                // decoding from FFmpeg MJPG has lower priority than libavcodec
                // decoder because those files doesn't has much independent
                // segments (1 per MCU row -> 68 for HD) -> lavd may be better
		{ MJPG, VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 90 },
		{ MJPG, RGB, RGB, 600 },
		{ MJPG, RGB, RGBA, 600 },
		{ MJPG, UYVY, UYVY, 600 },
		{ MJPG, I420, I420, 600 },
		{ MJPG, I420, UYVY, 700 },
		{ MJPG, RGB, UYVY, 800 },
		{ MJPG, UYVY, RGB, 800 },
		{ MJPG, UYVY, RGBA, 800 },
		{ MJPG, VIDEO_CODEC_NONE, RGB, 920 },
		{ MJPG, VIDEO_CODEC_NONE, UYVY, 920 },
		{ MJPG, VIDEO_CODEC_NONE, RGBA, 920 },
#if GPUJPEG_VERSION_INT >= GPUJPEG_MK_VERSION_INT(0, 13, 0)
		{ MJPG, VIDEO_CODEC_NONE, I420, 920 },
#endif
                // for VRG
		{ JPEG, VIDEO_CODEC_NONE, CUDA_I420, 200 },
		{ MJPG, VIDEO_CODEC_NONE, CUDA_I420, 200 },
		{ JPEG, VIDEO_CODEC_NONE, CUDA_RGBA, 200 },
		{ MJPG, VIDEO_CODEC_NONE, CUDA_RGBA, 200 },

		{ VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 0 },
        };
        return ret;
}

static const struct video_decompress_info gpujpeg_info = {
        gpujpeg_decompress_init,
        gpujpeg_decompress_reconfigure,
        gpujpeg_decompress,
        gpujpeg_decompress_get_property,
        gpujpeg_decompress_done,
        gpujpeg_decompress_get_decoders,
};

REGISTER_MODULE(gpujpeg, &gpujpeg_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

/* vi: set expandtab sw=8: */
