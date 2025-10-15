/**
 * @file   video_decompress/cineform.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2023 CESNET, z. s. p. o.
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

#include <memory>
#include <vector>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "video.h"
#include "video_decompress.h"
#include "utils/macros.h" // to_fourcc

#include "CFHDTypes.h"
#include "CFHDDecoder.h"


struct state_cineform_decompress {
        int pitch = 0;
        int rshift = 0;
        int gshift = 0;
        int bshift = 0;
        video_desc desc = {};
        codec_t in_codec = VIDEO_CODEC_NONE;
        codec_t out_codec = VIDEO_CODEC_NONE;
        CFHD_PixelFormat decode_codec = CFHD_PIXEL_FORMAT_UNKNOWN;
        int decode_linesize = 0;

        using convert_fun_t = void (*)(unsigned char *dst_buffer,
                        unsigned char *src_buffer,
                        int width, int height, int pitch);
        convert_fun_t convert = nullptr;

        std::vector<unsigned char> conv_buf;

        bool prepared_to_decode = false;

        CFHD_DecoderRef decoderRef = nullptr;
        CFHD_MetadataRef metadataRef = nullptr;
};

static void *cineform_decompress_init(){
        auto s = std::make_unique<state_cineform_decompress>();

        CFHD_Error status;
        status = CFHD_OpenDecoder(&s->decoderRef, nullptr);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to open decoder\n");
                return nullptr;
        }
        status = CFHD_OpenMetadata(&s->metadataRef);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to open metadata\n");
                CFHD_CloseDecoder(s->decoderRef);
                return nullptr;
        }

        return s.release();
}

static void cineform_decompress_done(void *state)
{
        auto s = static_cast<struct state_cineform_decompress *>(state);

        CFHD_CloseDecoder(s->decoderRef);
        CFHD_CloseMetadata(s->metadataRef);
        delete s;
}

static void rg48_to_r12l(unsigned char *dst_buffer,
                unsigned char *src_buffer,
                int width, int height, int pitch)
{
        int src_pitch = vc_get_linesize(width, RG48);
        int dst_len = vc_get_linesize(width, R12L);
        decoder_t vc_copylineRG48toR12L = get_decoder_from_to(RG48, R12L);

        for(int i = 0; i < height; i++){
                vc_copylineRG48toR12L(dst_buffer, src_buffer, dst_len, 0, 0, 0);
                src_buffer += src_pitch;
                dst_buffer += pitch;
        }
}

static void abgr_to_rgba(unsigned char *dst_buffer,
                unsigned char *src_buffer,
                int width, int height, int pitch)
{
        int linesize = vc_get_linesize(width, RGBA);

        for(int i = 0; i < height; i++){
                vc_copylineRGBA(dst_buffer, src_buffer, linesize, 16, 8, 0);
                src_buffer += linesize;
                dst_buffer += pitch;
        }
}

static void bgr_to_rgb_invert(unsigned char *dst_buffer,
                unsigned char *src_buffer,
                int width, int height, int pitch)
{
        int linesize = vc_get_linesize(width, RGB);
        src_buffer += linesize * (height - 1);
        decoder_t vc_copylineBGRtoRGB = get_decoder_from_to(BGR, RGB);

        for(int i = 0; i < height; i++){
                vc_copylineBGRtoRGB(dst_buffer, src_buffer, linesize, 0, 0, 0);
                src_buffer -= linesize;
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
        {RG48, CFHD_PIXEL_FORMAT_RG48, nullptr},
        {UYVY, CFHD_PIXEL_FORMAT_2VUY, nullptr},
        {R10k, CFHD_PIXEL_FORMAT_DPX0, nullptr},
        {v210, CFHD_PIXEL_FORMAT_V210, nullptr},
        {RGB, CFHD_PIXEL_FORMAT_RG24, bgr_to_rgb_invert},
        {RGBA, CFHD_PIXEL_FORMAT_BGRa, abgr_to_rgba},
};

static int cineform_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        auto s = static_cast<struct state_cineform_decompress *>(state);

        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        s->in_codec = desc.color_spec;
        s->out_codec = out_codec;
        s->desc = desc;
        s->prepared_to_decode = false;

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
                        s->desc.width,
                        s->desc.height,
                        s->decode_codec,
                        CFHD_DECODED_RESOLUTION_FULL,
                        CFHD_DECODING_FLAGS_NONE,
                        src,
                        src_len,
                        &actualWidth,
                        &actualHeight,
                        &actualFormat
                        );
        assert(actualWidth == (int) s->desc.width);
        assert(actualHeight == (int) s->desc.height);
        assert(actualFormat == s->decode_codec);
        if(s->convert){
                int actualPitch;
                CFHD_GetImagePitch(actualWidth, actualFormat, &actualPitch);
                assert(actualPitch == s->decode_linesize);
                s->conv_buf.resize(s->desc.height * s->decode_linesize);
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

static decompress_status probe_internal_cineform(struct state_cineform_decompress *s,
                                        unsigned char *src,
                                        unsigned src_len,
                                        struct pixfmt_desc *internal_prop)
{
        CFHD_Error status;
        CFHD_PixelFormat fmt_list[64];
        int count = 0;

        status = CFHD_GetOutputFormats(s->decoderRef,
                                       src,
                                       src_len,
                                       fmt_list,
                                       std::size(fmt_list),
                                       &count);
        log_msg(LOG_LEVEL_DEBUG, "[cineform] probing...\n");

        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] probe failed\n");
                return DECODER_NO_FRAME;
        }

        for(int i = 0; i < count; i++){
                for(const auto &codec : decode_codecs){
                        if(codec.cfhd_pixfmt == fmt_list[i]){
                                *internal_prop = get_pixfmt_desc(codec.ug_codec);
                                return DECODER_GOT_CODEC;
                        }
                }
        }

        //Unknown internal format. This should never happen since cineform can
        //decode any internal codec to CFHD_PIXEL_FORMAT_YUY2.
        //Here we just select UYVY and hope for the best.
        *internal_prop = get_pixfmt_desc(UYVY);
        return DECODER_GOT_CODEC;
}

static void write_fcc(char *out, int pixelformat){
        out[4] = '\0';

        for(int i = 0; i < 4; i++){
                out[i] = pixelformat & 0xff;
                pixelformat >>= 8;
        }
}

static decompress_status probe_internal(struct state_cineform_decompress *s,
                                        unsigned char *src,
                                        unsigned src_len,
                                        struct pixfmt_desc *internal_prop)
{
        CFHD_Error status;

        status = CFHD_InitSampleMetadata(s->metadataRef,
                                         METADATATYPE_ORIGINAL,
                                         src,
                                         src_len);
        if(status != CFHD_ERROR_OKAY){
                log_msg(LOG_LEVEL_ERROR, "[cineform] InitSampleMetadata failed\n");
                return DECODER_NO_FRAME;
        }

        CFHD_MetadataTag tag;
        CFHD_MetadataType type;
        void *data;
        CFHD_MetadataSize size;
        char fcc[5];
        while((status = CFHD_ReadMetadata(s->metadataRef, &tag, &type, &data, &size)) == CFHD_ERROR_OKAY){
                write_fcc(fcc, tag);
                log_msg(LOG_LEVEL_DEBUG, "[cineform] Metadata found. tag = %s \n", fcc);
        }

        status = CFHD_FindMetadata(s->metadataRef,
                                   to_fourcc('U', 'G', 'P', 'F'),
                                   &type,
                                   &data,
                                   &size);
        if(status != CFHD_ERROR_OKAY || type != METADATATYPE_UINT32){
                log_msg(LOG_LEVEL_ERROR, "[cineform] UGPF metadata not found or wrong type, "
                                         "falling back to cineform internal format detection.\n");
        } else {
                codec_t pf = *static_cast<codec_t *>(data);
                *internal_prop = get_pixfmt_desc(pf);
                log_msg(LOG_LEVEL_NOTICE, "[cineform] Codec determined from metadata: %s \n", get_codec_name(pf));
                return DECODER_GOT_CODEC;
        }

        return probe_internal_cineform(s, src, src_len, internal_prop);
}

static decompress_status cineform_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks,
                struct pixfmt_desc *internal_prop)
{
        UNUSED(frame_seq);
        UNUSED(callbacks);
        auto s = static_cast<struct state_cineform_decompress *>(state);
        decompress_status res = DECODER_NO_FRAME;

        CFHD_Error status;

        if(s->out_codec == VIDEO_CODEC_NONE){
                return probe_internal(s, src, src_len, internal_prop);
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
                        s->convert(dst, decode_dst, s->desc.width, s->desc.height, s->pitch);
                }
                res = DECODER_GOT_FRAME;
        } else {
                log_msg(LOG_LEVEL_ERROR, "[cineform] Failed to decode %i\n", status);
        }

        return res;
}

static int cineform_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        auto s = static_cast<struct state_cineform_decompress *>(state);
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

static int cineform_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc) {
        UNUSED(internal);
        if (compression != CFHD) {
                return -1;
        }
        switch (ugc) {
                case VIDEO_CODEC_NONE: // probe
                        return 50;
                case UYVY:
                case RGBA:
                case R10k:
                case R12L:
                case RG48:
                case v210:
                        break;
                default:
                        return -1;
        }
        return VDEC_PRIO_PREFERRED;
}

static const struct video_decompress_info cineform_info = {
        cineform_decompress_init,
        cineform_decompress_reconfigure,
        cineform_decompress,
        cineform_decompress_get_property,
        cineform_decompress_done,
        cineform_decompress_get_priority,
};

REGISTER_MODULE(cineform, &cineform_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
