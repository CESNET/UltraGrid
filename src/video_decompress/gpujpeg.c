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

#include <libgpujpeg/gpujpeg_decoder.h>
#include <libgpujpeg/gpujpeg_version.h>
//#include "compat/platform_semaphore.h"
#include <pthread.h>
#include <stdlib.h>
#include "lib_common.h"

#if LIBGPUJPEG_API_VERSION >= 7
#define GJ_RGBA_SUPP 1
#else
#define GJ_RGBA_SUPP 0
#endif

// compat
#if LIBGPUJPEG_API_VERSION <= 2
#define GPUJPEG_444_U8_P012 GPUJPEG_4_4_4
#define GPUJPEG_422_U8_P1020 GPUJPEG_4_2_2
#endif

#if LIBGPUJPEG_API_VERSION < 9
#define GPUJPEG_444_U8_P012A GPUJPEG_444_U8_P012Z
#endif

#define MOD_NAME "[GPUJPEG dec.] "

struct state_decompress_gpujpeg {
        struct gpujpeg_decoder *decoder;

        struct video_desc desc;
        int rshift, gshift, bshift;
        int pitch;
        codec_t out_codec;
};

static int configure_with(struct state_decompress_gpujpeg *s, struct video_desc desc);

static int configure_with(struct state_decompress_gpujpeg *s, struct video_desc desc)
{
        s->desc = desc;


#if LIBGPUJPEG_API_VERSION <= 2
        s->decoder = gpujpeg_decoder_create();
#else
        s->decoder = gpujpeg_decoder_create(NULL);
#endif
        if(!s->decoder) {
                return FALSE;
        }

        struct gpujpeg_parameters param;
        gpujpeg_set_default_parameters(&param);
        param.verbose = MAX(0, log_level - LOG_LEVEL_INFO);
        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);
        gpujpeg_decoder_init(s->decoder, &param, &param_image);

        switch (s->out_codec) {
        case I420:
                gpujpeg_decoder_set_output_format(s->decoder, GPUJPEG_YCBCR_JPEG,
                                GPUJPEG_420_U8_P0P1P2);
                break;
        case RGBA:
#if GJ_RGBA_SUPP == 1
                gpujpeg_decoder_set_output_format(s->decoder, GPUJPEG_RGB,
                                s->out_codec == RGBA && s->rshift == 0 && s->gshift == 8 && s->bshift == 16 && vc_get_linesize(desc.width, RGBA) == s->pitch ?
                                GPUJPEG_444_U8_P012A : GPUJPEG_444_U8_P012);
                break;
#endif
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

        return TRUE;
}

static void * gpujpeg_decompress_init(void)
{
#if LIBGPUJPEG_API_VERSION >= 7
        if (gpujpeg_version() != LIBGPUJPEG_API_VERSION) {
                log_msg(LOG_LEVEL_WARNING, "GPUJPEG API version mismatch! (%d vs %d)\n",
                                gpujpeg_version(), LIBGPUJPEG_API_VERSION);
        }
#endif
        struct state_decompress_gpujpeg *s;

        s = (struct state_decompress_gpujpeg *) calloc(1, sizeof(struct state_decompress_gpujpeg));

        int ret;
        printf("Initializing CUDA device %d...\n", cuda_devices[0]);
        ret = gpujpeg_init_device(cuda_devices[0], TRUE);
        if(ret != 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "initializing CUDA device %d failed.\n", cuda_devices[0]);
                free(s);
                return NULL;
        }


        return s;
}

static int gpujpeg_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_gpujpeg *s = (struct state_decompress_gpujpeg *) state;
        
        assert(out_codec == I420 || out_codec == RGB || out_codec == RGBA
                        || out_codec == UYVY || out_codec == VIDEO_CODEC_NONE);

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
                return configure_with(s, desc);
        }
}

#if LIBGPUJPEG_API_VERSION >= 4
static decompress_status gpujpeg_probe_internal_codec(unsigned char *buffer, size_t len, codec_t *internal_codec) {
        *internal_codec = VIDEO_CODEC_NONE;
	struct gpujpeg_image_parameters params = { 0 };
#if LIBGPUJPEG_API_VERSION >= 11
	if (gpujpeg_decoder_get_image_info(buffer, len, &params, NULL, MAX(0, log_level - LOG_LEVEL_INFO)) != 0) {
#elif LIBGPUJPEG_API_VERSION >= 6
	if (gpujpeg_decoder_get_image_info(buffer, len, &params, NULL) != 0) {
#elif LIBGPUJPEG_API_VERSION >= 5
	if (gpujpeg_reader_get_image_info(buffer, len, &params, NULL) != 0) {
#else
	if (gpujpeg_decoder_get_image_info(buffer, len, &params) != 0) {
#endif
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
#if LIBGPUJPEG_API_VERSION < 8
                *internal_codec = UYVY;
#else
                *internal_codec = params.pixel_format == GPUJPEG_420_U8_P0P1P2 ? I420 : UYVY;
#endif
		break;
	default:
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "probe - unhandled color space: %s\n",
                                gpujpeg_color_space_get_name(params.color_space));
		return DECODER_GOT_FRAME;
	}

	log_msg(LOG_LEVEL_VERBOSE, "JPEG color space: %s\n", gpujpeg_color_space_get_name(params.color_space));
	return DECODER_GOT_CODEC;
}
#endif

static decompress_status gpujpeg_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, codec_t *internal_codec)
{
        UNUSED(frame_seq);
        UNUSED(callbacks);
        struct state_decompress_gpujpeg *s = (struct state_decompress_gpujpeg *) state;
        int ret;
        struct gpujpeg_decoder_output decoder_output;
        int linesize;

        if (s->out_codec == VIDEO_CODEC_NONE) {
#if LIBGPUJPEG_API_VERSION >= 4
                return gpujpeg_probe_internal_codec(buffer, src_len, internal_codec);
#else
                assert("Old GPUJPEG, cannot probe!" && 0);
#endif
        }

        linesize = vc_get_linesize(s->desc.width, s->out_codec);
        
        gpujpeg_set_device(cuda_devices[0]);

        if (s->pitch == linesize && (s->out_codec == UYVY || s->out_codec == RGB
#if GJ_RGBA_SUPP == 1
                                || (s->out_codec == RGBA && s->rshift == 0 && s->gshift == 8 && s->bshift == 16)
#endif
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
        free(s);
}

static const struct decode_from_to *gpujpeg_decompress_get_decoders() {
        static const struct decode_from_to ret[] = {
#if LIBGPUJPEG_API_VERSION >= 4
		{ JPEG, VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 50 },
#endif
		{ JPEG, RGB, RGB, 300 },
		{ JPEG, RGB, RGBA, 300 + (1 - GJ_RGBA_SUPP) * 50 }, // 300 when GJ support RGBA natively,
                                                                    // 350 when using CPU conversion
		{ JPEG, UYVY, UYVY, 300 },
		{ JPEG, I420, I420, 300 },
		{ JPEG, I420, UYVY, 500 },
		{ JPEG, RGB, UYVY, 700 },
		{ JPEG, UYVY, RGB, 700 },
		{ JPEG, UYVY, RGBA, 700  + (1 - GJ_RGBA_SUPP) * 50},
		{ JPEG, VIDEO_CODEC_NONE, RGB, 900 },
		{ JPEG, VIDEO_CODEC_NONE, UYVY, 900 },
		{ JPEG, VIDEO_CODEC_NONE, RGBA, 900 +  (1 - GJ_RGBA_SUPP) * 50},
#if LIBGPUJPEG_API_VERSION > 6
                // decoding from FFmpeg MJPG has lower priority than libavcodec
                // decoder because those files doesn't has much independent
                // segments (1 per MCU row -> 68 for HD) -> lavd may be better
		{ MJPG, VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 90 },
		{ MJPG, RGB, RGB, 600 },
		{ MJPG, RGB, RGBA, 600 + (1 - GJ_RGBA_SUPP) * 50 },
		{ MJPG, UYVY, UYVY, 600 },
		{ MJPG, I420, I420, 600 },
		{ MJPG, I420, UYVY, 700 },
		{ MJPG, RGB, UYVY, 800 },
		{ MJPG, UYVY, RGB, 800 },
		{ MJPG, UYVY, RGBA, 800  + (1 - GJ_RGBA_SUPP) * 50},
		{ MJPG, VIDEO_CODEC_NONE, RGB, 920 },
		{ MJPG, VIDEO_CODEC_NONE, UYVY, 920 },
		{ MJPG, VIDEO_CODEC_NONE, RGBA, 920 +  (1 - GJ_RGBA_SUPP) * 50},
#endif
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
