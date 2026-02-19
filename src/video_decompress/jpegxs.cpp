/**
 * @file   video_decompress/jpegxs.cpp
 * @author Jan Frejlach     <536577@mail.muni.cz>
 */
/*
 * Copyright (c) 2026 CESNET
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

#include <assert.h>
#include <svt-jpegxs/SvtJpegxsDec.h>
#include <svt-jpegxs/SvtJpegxsImageBufferTools.h>

#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_decompress.h"
#include "jpegxs/jpegxs_conv.h"

#define MOD_NAME "[JPEG XS dec.] "

struct state_decompress_jpegxs {
        ~state_decompress_jpegxs() {
                if (frame_pool) {
                        svt_jpeg_xs_frame_pool_free(frame_pool);
                }
                if (configured) {
                        svt_jpeg_xs_decoder_close(&decoder);
                }
        }
        svt_jpeg_xs_decoder_api_t decoder{};
        svt_jpeg_xs_image_config_t image_config{};
        svt_jpeg_xs_frame_pool_t *frame_pool{};
        
        bool configured = 0;

        void (*convert_from_planar)(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) = nullptr;

        struct video_desc desc{};
        int rshift, gshift, bshift;
        int pitch;
        codec_t out_codec;
};

static void *jpegxs_decompress_init(void) {
        struct state_decompress_jpegxs *s = new state_decompress_jpegxs();

        return s;
}

static bool configure_with(struct state_decompress_jpegxs *s, unsigned char *bitstream_buffer, size_t codestream_size)
{
        s->decoder.verbose = VERBOSE_NONE;
        s->decoder.threads_num = 10;
        s->decoder.use_cpu_flags = CPU_FLAGS_ALL;
        s->decoder.proxy_mode = proxy_mode_full;

        SvtJxsErrorType_t err = svt_jpeg_xs_decoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &s->decoder, bitstream_buffer, codestream_size, &s->image_config);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize JPEG XS decoder\n");
                return false;
        }

        s->frame_pool = svt_jpeg_xs_frame_pool_alloc(&s->image_config, 0, 1);
        if (!s->frame_pool) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to allocate JPEG XS frame pool\n");
                return false;
        }

        s->configured = true;
        return true;
}

static int jpegxs_decompress_reconfigure(void *state, struct video_desc desc,
        int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_jpegxs *s = (struct state_decompress_jpegxs *) state;

        assert(get_jpegxs_to_uv_conversion(out_codec) || out_codec == VIDEO_CODEC_NONE);

        if (s->out_codec == out_codec &&
                s->pitch == pitch &&
                s->rshift == rshift &&
                s->gshift == gshift &&
                s->bshift == bshift &&
                video_desc_eq_excl_param(s->desc, desc, PARAM_INTERLACING)) {
                return true;
        }

        s->out_codec = out_codec;
        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        s->desc = desc;

        if (s->out_codec != VIDEO_CODEC_NONE) {
                const struct jpegxs_to_uv_conversion *conv = get_jpegxs_to_uv_conversion(s->out_codec);
                if (!conv || !conv->convert) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported codec: %s\n", get_codec_name(s->out_codec));
                        return false;
                }
                s->convert_from_planar = conv->convert;
        }

        if (s->configured) {
                svt_jpeg_xs_decoder_close(&s->decoder);
                s->configured = false;
        }

        return true;
}

static decompress_status jpegxs_probe_internal_codec(struct state_decompress_jpegxs *s, struct pixfmt_desc *internal_prop, unsigned char *buffer, size_t buffer_size)
{
        uint32_t size;
        SvtJxsErrorType_t err = svt_jpeg_xs_decoder_get_single_frame_size(buffer, buffer_size, &s->image_config, &size, 0);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get frame size from bitstream, error code: %x\n", err);
                abort();
        }
        assert(buffer_size == size);

        internal_prop->depth = s->image_config.bit_depth;
        switch (s->image_config.format) {
        case COLOUR_FORMAT_PLANAR_YUV420:
                internal_prop->subsampling = 4200;
                internal_prop->rgb = false;
                break;
        case COLOUR_FORMAT_PLANAR_YUV422:
                internal_prop->subsampling = 4220;
                internal_prop->rgb = false;
                break;
        case COLOUR_FORMAT_PLANAR_YUV444_OR_RGB:
                internal_prop->subsampling = 4440;
                internal_prop->rgb = true;
                break;
        default:
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to probe JPEG XS codec - unsupported colour format\n");
                abort();
        }

        return DECODER_GOT_CODEC;
}

static decompress_status jpegxs_decompress(void *state, unsigned char *dst, unsigned char *buffer, 
        unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, struct pixfmt_desc *internal_prop)
{
        UNUSED(frame_seq);
        UNUSED(callbacks);
        auto *s = (struct state_decompress_jpegxs *) state;

        if (s->out_codec == VIDEO_CODEC_NONE) {
                return jpegxs_probe_internal_codec(s, internal_prop, buffer, src_len);
        }

        if (!s->configured) {
                if (!configure_with(s, buffer, src_len)) {
                        return DECODER_NO_FRAME;
                }
        }

        svt_jpeg_xs_bitstream_buffer_t bitstream;
        bitstream.buffer = buffer;
        bitstream.used_size = src_len;
        bitstream.allocation_size = src_len;

        svt_jpeg_xs_frame_t dec_input;
        SvtJxsErrorType_t err = svt_jpeg_xs_frame_pool_get(s->frame_pool, &dec_input, /*blocking*/ 1);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Failed to get frame from JPEG XS pool, error code: %x\n", err);
                return DECODER_NO_FRAME;
        }
        dec_input.bitstream = bitstream;

        err = svt_jpeg_xs_decoder_send_frame(&s->decoder, &dec_input, 1 /*blocking*/);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to send frame to decoder, error code: %x\n", err);
                return DECODER_NO_FRAME;
        }

        svt_jpeg_xs_frame_t dec_output;
        err = svt_jpeg_xs_decoder_get_frame(&s->decoder, &dec_output, 1 /*blocking*/);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get encoded packet, error code: %x\n", err);
                return DECODER_NO_FRAME;
        }

        s->convert_from_planar(&dec_output.image, s->image_config.width, s->image_config.height, dst);
        svt_jpeg_xs_frame_pool_release(s->frame_pool, &dec_output);
        return DECODER_GOT_FRAME;
}

static int jpegxs_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decompress *s = (struct state_decompress *) state;
        UNUSED(s);
        int ret = false;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = false;
                                *len = sizeof(int);
                                ret = true;
                        }
                        break;
                default:
                        ret = false;
        }

        return ret;
}

static void jpegxs_decompress_done(void *state) {
       delete (struct state_decompress_jpegxs *) state;
}

static int jpegxs_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc)
{
        UNUSED(internal);

        if (compression != JPEG_XS) {
                return VDEC_PRIO_NA;
        }

        if (ugc == VC_NONE) {
                return VDEC_PRIO_PROBE_HI;
        }

        // supported output formats
        if (get_jpegxs_to_uv_conversion(ugc) != nullptr) {
                return VDEC_PRIO_PREFERRED;
        }

        return VDEC_PRIO_NA;
}

static const struct video_decompress_info jpegxs_info = {
        jpegxs_decompress_init,
        jpegxs_decompress_reconfigure,
        jpegxs_decompress,
        jpegxs_decompress_get_property,
        jpegxs_decompress_done,
        jpegxs_decompress_get_priority,
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
