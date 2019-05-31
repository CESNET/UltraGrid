/**
 * @file   video_decompress/cineform.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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
#include "lib_common.h"
#include "tv.h"
#include "utils/resource_manager.h"
#include "video.h"
#include "video_decompress.h"

#include "CFHDTypes.h"
#include "CFHDDecoder.h"

#include <mutex>
#include <vector>

struct state_cineform_decompress {
        int              width, height;
        int              pitch;
        int              rshift, gshift, bshift;
        int              max_compressed_len;
        codec_t          in_codec;
        codec_t          out_codec;
        CFHD_PixelFormat decode_codec;
        int              decode_linesize;
        void (*convert)(unsigned char *dst_buffer,
                        unsigned char *src_buffer,
                        int width, int height, int pitch);
        std::vector<unsigned char> conv_buf;

        unsigned         last_frame_seq:22; // This gives last sucessfully decoded frame seq number. It is the buffer number from the packet format header, uses 22 bits.
        bool             last_frame_seq_initialized;
        bool             prepared_to_decode;

        CFHD_DecoderRef decoderRef;

        struct video_desc saved_desc;
};

static void * cineform_decompress_init(void)
{
        struct state_cineform_decompress *s;

        s = new state_cineform_decompress();

        s->width = s->height = s->pitch = s->decode_linesize = 0;
        s->convert = nullptr;
        s->prepared_to_decode = false;

        CFHD_Error status;
        status = CFHD_OpenDecoder(&s->decoderRef, nullptr);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to open decoder\n");
        }

        return s;
}

static void cineform_decompress_done(void *state)
{
        struct state_cineform_decompress *s =
                (struct state_cineform_decompress *) state;

        CFHD_CloseDecoder(s->decoderRef);
        delete s;
}

static void rg48_to_r12l(unsigned char *dst_buffer,
                unsigned char *src_buffer,
                int width, int height, int pitch)
{
        int src_pitch = vc_get_linesize(width, RG48);
        int dst_len = vc_get_linesize(width, R12L);

        for(int i = 0; i < height; i++){
                vc_copylineRG48toR12L(dst_buffer, src_buffer, dst_len);
                src_buffer += src_pitch;
                dst_buffer += pitch;
        }
}

static const struct {
        codec_t ug_codec;
        CFHD_PixelFormat cfhd_pixfmt;
        void (*convert)(unsigned char *dst_buffer,
                        unsigned char *src_buffer,
                        int width, int height, int pitch);
} decode_codecs[] = {
        {R12L, CFHD_PIXEL_FORMAT_RG48, rg48_to_r12l},
        {UYVY, CFHD_PIXEL_FORMAT_2VUY, nullptr},
};

static bool configure_with(struct state_cineform_decompress *s,
                struct video_desc desc)
{
        s->last_frame_seq_initialized = false;
        s->prepared_to_decode = false;
        s->saved_desc = desc;

        if(s->out_codec == VIDEO_CODEC_NONE){
                log_msg(LOG_LEVEL_DEBUG, "[cineform] Will probe for internal format.\n");
                return true;
        }

        for(const auto& i : decode_codecs){
                if(i.ug_codec == s->out_codec){
                        s->decode_codec = i.cfhd_pixfmt;
                        s->convert = i.convert;
                        CFHD_GetImagePitch(desc.width, i.cfhd_pixfmt, &s->decode_linesize);
                        if(i.ug_codec == R12L){
                                log_msg(LOG_LEVEL_NOTICE, "[cineform] Decoding to 12-bit RGB.\n");
                        }
                        return true;
                }
        }

        return false;
}

static int cineform_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_cineform_decompress *s =
                (struct state_cineform_decompress *) state;

        assert(out_codec == UYVY ||
                        out_codec == RGB ||
                        out_codec == v210 ||
                        out_codec == R12L ||
                        out_codec == VIDEO_CODEC_NONE);

        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        s->in_codec = desc.color_spec;
        s->out_codec = out_codec;
        s->width = desc.width;
        s->height = desc.height;

        return configure_with(s, desc);
}

static bool prepare(struct state_cineform_decompress *s,
                unsigned char *src,
                unsigned int src_len)
{
        if(s->prepared_to_decode){
                return true;
        }

        CFHD_Error status;

        int actualWidth;
        int actualHeight;
        CFHD_PixelFormat actualFormat;
        status = CFHD_PrepareToDecode(s->decoderRef,
                        s->saved_desc.width,
                        s->saved_desc.height,
                        s->decode_codec,
                        CFHD_DECODED_RESOLUTION_FULL,
                        CFHD_DECODING_FLAGS_NONE,
                        src,
                        src_len,
                        &actualWidth,
                        &actualHeight,
                        &actualFormat
                        );
        assert(actualWidth == s->width);
        assert(actualHeight == s->height);
        assert(actualFormat == s->decode_codec);
        if(s->convert){
                int actualPitch;
                CFHD_GetImagePitch(actualWidth, actualFormat, &actualPitch);
                assert(actualPitch == s->decode_linesize);
                s->conv_buf.resize(s->height * s->decode_linesize);
        } else {
                s->conv_buf.clear();
        }
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to prepare for decoding\n");
                return false;
        } 

        s->prepared_to_decode = true;
        return true;
}

static decompress_status probe_internal(struct state_cineform_decompress *s,
                                        unsigned char *src,
                                        unsigned src_len,
                                        codec_t *internal_codec)
{
        CFHD_Error status;
        CFHD_PixelFormat fmt_list[64];
        int count = 0;

        status = CFHD_GetOutputFormats(s->decoderRef,
                                       src,
                                       src_len,
                                       fmt_list,
                                       sizeof(fmt_list) / sizeof(fmt_list[0]),
                                       &count);
        log_msg(LOG_LEVEL_DEBUG, "[cineform] probing...\n");

        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] probe failed\n");
                return DECODER_NO_FRAME;
        }

        for(int i = 0; i < count; i++){
                for(const auto &codec : decode_codecs){
                        if(codec.cfhd_pixfmt == fmt_list[i]){
                                *internal_codec = codec.ug_codec;
                                return DECODER_GOT_CODEC;
                        }
                }
        }

        //Unknown internal format. This should never happen since cineform can
        //decode any internal codec to CFHD_PIXEL_FORMAT_YUY2.
        //Here we just select UYVY and hope for the best.
        *internal_codec = UYVY;
        return DECODER_GOT_CODEC;
}

static decompress_status cineform_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks,
                codec_t *internal_codec)
{
        UNUSED(frame_seq);
        UNUSED(callbacks);
        struct state_cineform_decompress *s = (struct state_cineform_decompress *) state;
        decompress_status res = DECODER_NO_FRAME;

        CFHD_Error status;

        if(s->out_codec == VIDEO_CODEC_NONE){
                return probe_internal(s, src, src_len, internal_codec);
        }

        if(!prepare(s, src, src_len)){
                return res;
        }

        unsigned char *decode_dst = s->convert ? s->conv_buf.data() : dst;
        int pitch = s->convert ? s->decode_linesize : s->pitch;

        status = CFHD_DecodeSample(s->decoderRef,
                        src,
                        src_len,
                        decode_dst,
                        pitch);

        if(status == CFHD_ERROR_OKAY){
                if(s->convert){
                        s->convert(dst, decode_dst, s->width, s->height, s->pitch);
                }
                res = DECODER_GOT_FRAME;
        } else {
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to decode %i\n", status);
        }

        return res;
}

static int cineform_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_cineform_decompress *s =
                (struct state_cineform_decompress *) state;
        UNUSED(s);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
#ifdef CINEFORM_ACCEPT_CORRUPTED
                                *(int *) val = TRUE;
#else
                                *(int *) val = FALSE;
#endif
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

ADD_TO_PARAM(cfhd_use_12bit, "cfhd-use-12bit",
                "* cfhd-use-12bit\n"
                "  Indicates that we are using decoding to R12L.\n"
                "  With this flag, R12L (12-bit RGB)\n"
                "  will be announced as a supported codec.\n");

static const struct decode_from_to *cineform_decompress_get_decoders() {
        const struct decode_from_to dec_static[] = {
                { CFHD, VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 50 },
                { CFHD, UYVY, UYVY, 500 },
                { CFHD, R12L, R12L, 500 },
                { CFHD, VIDEO_CODEC_NONE, UYVY, 600 },
                { VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 0 },
        };

        static struct decode_from_to ret[sizeof dec_static / sizeof dec_static[0]
                + 1 /* terminating zero */
                + 10 /* place for additional decoders, see below */];

        static std::mutex mutex;

        std::lock_guard<std::mutex> lock(mutex);

        if (ret[0].from == VIDEO_CODEC_NONE) { // not yet initialized
                memcpy(ret, dec_static, sizeof dec_static);
        }

        return ret;
}

static const struct video_decompress_info cineform_info = {
        cineform_decompress_init,
        cineform_decompress_reconfigure,
        cineform_decompress,
        cineform_decompress_get_property,
        cineform_decompress_done,
        cineform_decompress_get_decoders,
};

REGISTER_MODULE(cineform, &cineform_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
