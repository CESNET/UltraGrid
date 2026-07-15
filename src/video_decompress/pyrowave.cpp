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
#include <vulkan/vulkan.h>
#include <pyrowave/pyrowave.h>

#include "debug.h"
#include "from_planar.h"
#include "lib_common.h"
#include "video_decompress.h"
#include "utils/misc.h"

#define MOD_NAME "[Pyrowave dec] "

namespace{

using pyrowave_decoder_unique = std::unique_ptr<pyrowave_decoder_opaque, deleter_from_fcn<pyrowave_decoder_destroy>>;
using pyrowave_device_unique = std::unique_ptr<pyrowave_device_opaque, deleter_from_fcn<pyrowave_device_destroy>>;

struct pyro_header{
        size_t packet_size;
};

struct pyrowave_decompress_state{
        pyrowave_device_unique device;
        pyrowave_decoder_unique decoder;

        video_desc saved_desc{};
        codec_t out_codec = VIDEO_CODEC_NONE;
        int pitch = 0;

        std::vector<std::vector<std::byte>> plane_datas;
        pyrowave_cpu_buffer pyro_frame{};
};

void create_cpu_pyro_frame(pyrowave_decompress_state *s, const video_desc &desc){
        s->pyro_frame = {};
        s->pyro_frame.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
        s->pyro_frame.width = static_cast<int>(desc.width);
        s->pyro_frame.height = static_cast<int>(desc.height);

        constexpr size_t alignment = 256;

        s->plane_datas.resize(3);

        size_t luma_stride = ((desc.width + alignment - 1) / alignment) * alignment;
        size_t luma_size = luma_stride * desc.height;
        s->plane_datas[0].resize(luma_size + alignment - 1);
        s->pyro_frame.data[0] = s->plane_datas[0].data();
        s->pyro_frame.row_stride_in_bytes[0] = luma_stride;
        s->pyro_frame.plane_size_in_bytes[0] = s->plane_datas[0].size();
        assert(std::align(alignment, luma_size, s->pyro_frame.data[0], s->pyro_frame.plane_size_in_bytes[0]));

        size_t chroma_stride = ((desc.width / 2 + alignment - 1) / alignment) * alignment;
        size_t chroma_size = chroma_stride * desc.height;

        s->plane_datas[1].resize(chroma_size + alignment - 1);
        s->pyro_frame.data[1] = s->plane_datas[1].data();
        s->pyro_frame.row_stride_in_bytes[1] = chroma_stride;
        s->pyro_frame.plane_size_in_bytes[1] = s->plane_datas[1].size();
        assert(std::align(alignment, chroma_size, s->pyro_frame.data[1], s->pyro_frame.plane_size_in_bytes[1]));

        s->plane_datas[2].resize(chroma_size + alignment - 1);
        s->pyro_frame.data[2] = s->plane_datas[2].data();
        s->pyro_frame.row_stride_in_bytes[2] = chroma_stride;
        s->pyro_frame.plane_size_in_bytes[2] = s->plane_datas[2].size();
        assert(std::align(alignment, chroma_size, s->pyro_frame.data[2], s->pyro_frame.plane_size_in_bytes[2]));
}


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

int pyrowave_reconfigure(void *state, video_desc desc, int rshift, int gshift, int bshift, int pitch, codec_t out_codec){
        auto s = static_cast<pyrowave_decompress_state *>(state);
        s->saved_desc = desc;

        pyrowave_decoder_create_info decoder_info{};
        decoder_info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420; //TODO
        decoder_info.device = s->device.get();
        decoder_info.width = static_cast<int>(desc.width);
        decoder_info.height = static_cast<int>(desc.height);
        decoder_info.fragment_path = false;
        auto res = pyrowave_decoder_create(&decoder_info, out_ptr(s->decoder));

        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create decoder (%d)\n", res);
                return false;
        }

        s->out_codec = out_codec;
        s->pitch = pitch;

        create_cpu_pyro_frame(s, desc);

        return true;
}

void pyro_to_ug_frame(pyrowave_decompress_state *s, void *dst){
        from_planar_data conv_data{};
        conv_data.width = s->saved_desc.width;
        conv_data.height = s->saved_desc.height;
        conv_data.out_data = static_cast<unsigned char *>(dst);
        conv_data.out_pitch = s->pitch;
        conv_data.log2_chroma_h = 1;
        for(int i = 0; i < 3; i++){
                conv_data.in_data[i] = static_cast<unsigned char *>(s->pyro_frame.data[i]);
                conv_data.in_linesize[i] = s->pyro_frame.row_stride_in_bytes[i];
        }

        yuv420p_to_uyvy(conv_data);
}

decompress_status pyrowave_decompress(void *state, unsigned char *dst, unsigned char *buffer,
        unsigned int src_len, int /*frame_seq*/, video_frame_callbacks */*callbacks*/, pixfmt_desc *internal_prop)
{
        auto s = static_cast<pyrowave_decompress_state *>(state);
        if(s->out_codec == VIDEO_CODEC_NONE){
                *internal_prop = {};
                internal_prop->depth = 8;
                internal_prop->rgb = false;
                internal_prop->subsampling = SUBS_420;
                return DECODER_GOT_CODEC;
        }

        pyro_header hdr;
        memcpy(&hdr, buffer, sizeof(hdr));

        log_msg(LOG_LEVEL_INFO, MOD_NAME "packet size (%lu)\n", hdr.packet_size);
        auto res = pyrowave_decoder_push_packet(s->decoder.get(), buffer + sizeof(pyro_header), hdr.packet_size);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to push packet (%d)\n", res);
                return DECODER_NO_FRAME;
        }
        if(!pyrowave_decoder_decode_is_ready(s->decoder.get(), true)){
                return DECODER_NO_FRAME;
        }

        res = pyrowave_decoder_decode_cpu_buffer_synchronous(s->decoder.get(), &s->pyro_frame);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to decode frame (%d)\n", res);
                return DECODER_NO_FRAME;
        }

        pyro_to_ug_frame(s, dst);

        log_msg(LOG_LEVEL_INFO, MOD_NAME "Decoded frame (%d)\n", res);
        return DECODER_GOT_FRAME;
}

int pyrowave_decompress_get_property(void */*state*/, int property, void *val, size_t *len) {
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

int pyrowave_get_decompress_priority(codec_t codec, pixfmt_desc internal, codec_t ugc){
        if (codec != PYROWAVE) {
                return VDEC_PRIO_NA;
        }
        if (ugc == VIDEO_CODEC_NONE) {
                return VDEC_PRIO_PROBE_HI;
        }

        return ugc == UYVY ? VDEC_PRIO_PREFERRED : VDEC_PRIO_NA;
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
