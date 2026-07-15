/**
 * @file   video_compress/pyrowave.cpp
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
#include <vulkan/vulkan.h>
#include <pyrowave/pyrowave.h>

#include "debug.h"
#include "video_compress.h"
#include "lib_common.h"
#include "to_planar.h"
#include "video_frame.h"
#include "utils/misc.h"
#include "utils/video_frame_pool.h"

#define MOD_NAME "[Pyrowave enc] "

namespace{

using pyrowave_encoder_unique = std::unique_ptr<pyrowave_encoder_opaque, deleter_from_fcn<pyrowave_encoder_destroy>>;
using pyrowave_device_unique = std::unique_ptr<pyrowave_device_opaque, deleter_from_fcn<pyrowave_device_destroy>>;

struct pyro_header{
        size_t packet_size;
};

struct pyrowave_compress_state{
        pyrowave_device_unique device;
        pyrowave_encoder_unique encoder;

        video_desc saved_desc{};
        size_t max_frame_size = 0;

        std::vector<std::vector<std::byte>> plane_datas;
        pyrowave_cpu_buffer pyro_frame{};

        video_frame_pool compressed_frame_pool;
        video_desc compressed_desc{};
};

void pyrowave_print_help(){
        //TODO
}

void *pyrowave_compress_init(module */*module*/, const char *arg){
        std::string_view cfg = arg ? arg : "";
        if(cfg == "help"){
                pyrowave_print_help();
                return INIT_NOERR;
        }
        auto s = std::make_unique<pyrowave_compress_state>();

        auto res = pyrowave_create_default_device(out_ptr(s->device));
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create pyro device (%d)\n", res);
                return nullptr;
        }

        return s.release();
}

void pyrowave_compress_done(void *state){
        auto *s = static_cast<pyrowave_compress_state *>(state);
        delete s;
}

bool create_pyro_encoder(pyrowave_compress_state *s, const video_desc& desc){
        s->encoder.reset();

        assert(desc.width % 2 == 0);
        pyrowave_encoder_create_info info{};
        info.device = s->device.get();
        info.width = static_cast<int>(desc.width);
        info.height = static_cast<int>(desc.height);
        info.chroma = PYROWAVE_CHROMA_SUBSAMPLING_420;

        const auto res = pyrowave_encoder_create(&info, out_ptr(s->encoder));
        return res == PYROWAVE_SUCCESS;
}

void create_cpu_pyro_frame(pyrowave_compress_state *s, const video_desc &desc){
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

bool configure_with(pyrowave_compress_state *s, const video_desc& desc){
        assert(desc.color_spec == UYVY); //TODO
        s->saved_desc = desc;
        create_cpu_pyro_frame(s, desc);
        bool res = create_pyro_encoder(s, desc);
        s->compressed_desc = desc;
        s->compressed_desc.color_spec = PYROWAVE;
        s->compressed_frame_pool.reconfigure(s->compressed_desc, s->max_frame_size);
        return res;
}

void ug_to_pyro_frame(pyrowave_compress_state *s, const std::shared_ptr<video_frame>& f){
        to_planar_data conv_data{};
        conv_data.width = f->tiles[0].width;
        conv_data.height = f->tiles[0].height;
        conv_data.in_data = reinterpret_cast<const unsigned char *>(f->tiles[0].data);
        for(int i = 0; i < 3; i++){
                conv_data.out_data[i] = static_cast<unsigned char *>(s->pyro_frame.data[i]);
                conv_data.out_linesize[i] = s->pyro_frame.row_stride_in_bytes[i];
        }

        uyvy_to_i420(conv_data);
}

std::shared_ptr<video_frame> pyrowave_compress_tile(void *state, std::shared_ptr<video_frame> video_frame){
        auto *s = static_cast<pyrowave_compress_state *>(state);

        if(!video_frame){
                return {};
        }

        if(const auto frame_desc = video_desc_from_frame(video_frame.get()); !video_desc_eq(s->saved_desc, frame_desc)){
                configure_with(s, frame_desc);
        }

        ug_to_pyro_frame(s, video_frame);

        pyrowave_rate_control rate_control{};
        rate_control.maximum_bitstream_size = s->max_frame_size;
        auto res= pyrowave_encoder_encode_cpu_synchronous(s->encoder.get(), &s->pyro_frame, &rate_control);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "pyrowave_encoder_encode_cpu_synchronous failed\n");
                return {};
        }

        size_t num_packets = 0;
        int packet_size = s->max_frame_size;
        res = pyrowave_encoder_compute_num_packets(s->encoder.get(), packet_size, &num_packets);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get num_packets (%d)\n", res);
                return {};
        }
        if(num_packets > 1){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Too many packets %d\n", num_packets);
        }

        std::vector<pyrowave_packet> packets(num_packets);
        auto out_frame = s->compressed_frame_pool.get_frame();

        char *bitstream_dst = out_frame->tiles[0].data + sizeof(pyro_header);
        unsigned int bitstream_size = out_frame->tiles[0].data_len - sizeof(pyro_header);
        res = pyrowave_encoder_packetize(s->encoder.get(), packets.data(), packet_size, &num_packets, bitstream_dst, bitstream_size);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get encoded frame (%d)\n", res);
                return {};
        }
        pyro_header hdr{.packet_size = packets[0].size};
        memcpy(out_frame->tiles[0].data, &hdr, sizeof(hdr));
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Packet %lu, %lu\n", packets[0].offset, packets[0].size);

        return out_frame;
}

constexpr video_compress_info pyrowave_info = []{
        video_compress_info info{};
        info.init_func = pyrowave_compress_init;
        info.done = pyrowave_compress_done;
        info.compress_tile_func = pyrowave_compress_tile;
        return info;
}();

REGISTER_MODULE(pyrowave, &pyrowave_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
