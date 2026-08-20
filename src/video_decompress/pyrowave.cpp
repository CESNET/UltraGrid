/**
 * @file   video_decompress/pyrowave.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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
#include <memory>
#include <vector>

#include "pyrowave_common.hpp"
#include "debug.h"
#include "from_planar.h"
#include "lib_common.h"
#include "video_codec.h"
#include "video_decompress.h"
#include "utils/profile_timer.hpp"

#define MOD_NAME "[Pyrowave dec] "

namespace{

struct pyrowave_decompress_state{
        pyrowave_device_unique device;
        pyrowave_decoder_unique decoder;

        video_desc saved_desc{};
        codec_t out_codec = VIDEO_CODEC_NONE;
        int pitch = 0;

        pyrowave_cpu_frame pyro_frame;

        decode_planar_func_t *from_planar_conv = nullptr;
};

void *pyrowave_decompress_init(){
        auto s = std::make_unique<pyrowave_decompress_state>();

        auto res = pyrowave_create_default_device(out_ptr(s->device));
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create pyro device (%d)\n", res);
                return nullptr;
        }

        return s.release();
}

void pyrowave_done(void *state){
        auto *s = static_cast<pyrowave_decompress_state *>(state);

        delete s;
}

int pyrowave_reconfigure(void *state, video_desc desc, int /*rshift*/, int /*gshift*/, int /*bshift*/, int pitch, codec_t out_codec){
        auto s = static_cast<pyrowave_decompress_state *>(state);
        s->saved_desc = desc;
        s->out_codec = out_codec;
        s->pitch = pitch;

        if(out_codec == VIDEO_CODEC_NONE){
                return true;
        }

        auto pyro_subs = PYROWAVE_CHROMA_SUBSAMPLING_INT_MAX;

        if(out_codec == UYVY){
                pyro_subs = PYROWAVE_CHROMA_SUBSAMPLING_420;
                s->from_planar_conv = yuv420p_to_uyvy;
        } else if(out_codec == VUYA){
                pyro_subs = PYROWAVE_CHROMA_SUBSAMPLING_444;
                s->from_planar_conv = yuv444p_to_vuya;
        } else if(out_codec == RGBA){
                pyro_subs = PYROWAVE_CHROMA_SUBSAMPLING_444;
                s->from_planar_conv = yuv444p_to_vuya;
        } else{
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported out codec (%s)\n", get_codec_name(out_codec));
                return false;
        }


        pyrowave_decoder_create_info decoder_info{};
        decoder_info.chroma = pyro_subs;
        decoder_info.device = s->device.get();
        decoder_info.width = static_cast<int>(desc.width);
        decoder_info.height = static_cast<int>(desc.height);
        decoder_info.fragment_path = false;
        auto res = pyrowave_decoder_create(&decoder_info, out_ptr(s->decoder));

        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create decoder (%d)\n", res);
                return false;
        }

        configure_pyro_frame(s->pyro_frame, desc.width, desc.height, pyro_subs);

        return true;
}

void pyro_to_ug_frame(pyrowave_decompress_state *s, void *dst){
        PROFILE_FUNC;
        from_planar_data conv_data{};
        conv_data.width = s->saved_desc.width;
        conv_data.height = s->saved_desc.height;
        conv_data.out_data = static_cast<unsigned char *>(dst);
        conv_data.out_pitch = s->pitch;
        conv_data.log2_chroma_h = 1;
        for(int i = 0; i < 3; i++){
                conv_data.in_data[i] = static_cast<unsigned char *>(s->pyro_frame.f.data[i]);
                conv_data.in_linesize[i] = s->pyro_frame.f.row_stride_in_bytes[i];
        }

        s->from_planar_conv(conv_data);
}

decompress_status pyrowave_decompress(void *state, unsigned char *dst, unsigned char *buffer,
        unsigned int src_len, int /*frame_seq*/, video_frame_callbacks */*callbacks*/, pixfmt_desc *internal_prop)
{
        PROFILE_FUNC;
        auto s = static_cast<pyrowave_decompress_state *>(state);

        pyrowave_frame_header hdr{};
        assert(src_len >= sizeof(hdr));
        memcpy(&hdr, buffer, sizeof(hdr));

        if(s->out_codec == VIDEO_CODEC_NONE){
                *internal_prop = {
                        .depth = 8,
                        .subsampling = pyro_subsampling_to_ug(hdr.subs),
                        .rgb = codec_is_a_rgb(hdr.internal),
                        .accel_type = HWACCEL_NONE,
                };
                return DECODER_GOT_CODEC;
        }

        auto res = pyrowave_decoder_push_packet(s->decoder.get(), buffer + sizeof(hdr), src_len - sizeof(hdr));
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to push packet (%d)\n", res);
                return DECODER_NO_FRAME;
        }
        if(!pyrowave_decoder_decode_is_ready(s->decoder.get(), true)){
                return DECODER_NO_FRAME;
        }

        res = pyrowave_decoder_decode_cpu_buffer_synchronous(s->decoder.get(), &s->pyro_frame.f);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to decode frame (%d)\n", res);
                return DECODER_NO_FRAME;
        }

        pyro_to_ug_frame(s, dst);

        return DECODER_GOT_FRAME;
}

int pyrowave_decompress_get_property(void */*state*/, int property, void *val, size_t *len) {
        int ret = false;

        switch(property) {
        case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                if(*len >= sizeof(int)) {
                        *(int *) val = false;
                        *len = sizeof(int);
                        ret = false;
                }
                break;
        default:
                ret = false;
        }

        return ret;
}

int pyrowave_get_decompress_priority(codec_t codec, pixfmt_desc internal, codec_t ugc){
        if (codec != PYROWAVE) {
                return VDEC_PRIO_NA;
        }
        if (ugc == VIDEO_CODEC_NONE) {
                return VDEC_PRIO_PROBE_HI;
        }

        auto preferred_codec = internal.subsampling == SUBS_444 ? VUYA : UYVY;
        if(internal.rgb)
                preferred_codec = RGBA;

        return ugc == preferred_codec ? VDEC_PRIO_PREFERRED : VDEC_PRIO_NA;
}

constexpr video_decompress_info info = []{
        video_decompress_info info{};
        info.init = pyrowave_decompress_init;
        info.done = pyrowave_done;
        info.reconfigure = pyrowave_reconfigure;
        info.decompress = pyrowave_decompress;
        info.get_property = pyrowave_decompress_get_property;
        info.get_decompress_priority = pyrowave_get_decompress_priority;
        return info;
}();

REGISTER_MODULE(pyrowave, &info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

}
