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
#include <climits>

#include "pyrowave_common.hpp"
#include "debug.h"
#include "video_compress.h"
#include "lib_common.h"
#include "to_planar.h"
#include "video_codec.h"
#include "video_frame.h"
#include "utils/misc.h"
#include "utils/profile_timer.hpp"
#include "utils/string_view_utils.hpp"
#include "utils/video_frame_pool.h"

#define MOD_NAME "[Pyrowave enc] "

namespace{

struct pyrowave_compress_state{
        pyrowave_device_unique device;
        pyrowave_encoder_unique encoder;

        video_desc saved_desc{};
        size_t max_frame_size = 0;

        pyrowave_chroma_subsampling pyro_subs = PYROWAVE_CHROMA_SUBSAMPLING_INT_MAX;
        pyrowave_cpu_frame pyro_frame;

        video_frame_pool compressed_frame_pool;
        video_desc compressed_desc{};
        uint32_t bitrate = 20'000'000;

        decode_buffer_func_t *to_planar_conv = nullptr;
};

void pyrowave_print_help(){
        color_printf(TBOLD("Pyrowave") " compression usage:\n");
        color_printf("\t" TBOLD(
                TRED("-c pyrowave") "[:bitrate=<br>]") "\n");
        color_printf("\t" TBOLD(TRED("-c pyrowave") ":help") "\n");
}

bool parse_params(pyrowave_compress_state& s, std::string_view cfg){
        while(!cfg.empty()){
                auto tok = tokenize(cfg, ':', '"');
                auto key = tokenize(tok, '=');
                auto val = tokenize(tok, '=');

                if(key == "bitrate"){
                        std::string valstr(val);
                        const char *endptr = nullptr;
                        const auto rate_bps = unit_evaluate(valstr.c_str(), &endptr);
                        if (rate_bps == LLONG_MIN || *endptr != '\0') {
                                MSG(ERROR, "Invalid bitrate value '%s'.\n", valstr.c_str());
                                return false;
                        }
                        s.bitrate = rate_bps;
                }
        }

        return true;
}

void *pyrowave_compress_init(module */*module*/, const char *conf){
        std::string_view cfg = conf ? conf : "";
        if(cfg == "help"){
                pyrowave_print_help();
                return INIT_NOERR;
        }
        auto s = std::make_unique<pyrowave_compress_state>();

        if(!parse_params(*s, cfg)){
                return nullptr;
        }

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
        const pyrowave_encoder_create_info info{
                .device = s->device.get(),
                .width = static_cast<int>(desc.width),
                .height = static_cast<int>(desc.height),
                .chroma = s->pyro_subs,
        };

        const auto res = pyrowave_encoder_create(&info, out_ptr(s->encoder));
        return res == PYROWAVE_SUCCESS;
}

bool configure_with(pyrowave_compress_state *s, const video_desc& desc){
        if(desc.color_spec == UYVY){
                s->pyro_subs = PYROWAVE_CHROMA_SUBSAMPLING_420;
                s->to_planar_conv = uyvy_to_i420;
        } else if(desc.color_spec == VUYA){
                s->pyro_subs = PYROWAVE_CHROMA_SUBSAMPLING_444;
                s->to_planar_conv = vuya_to_i444;
        } else{
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported color spec (%s)\n", get_codec_name(desc.color_spec));
                return false;
        }

        s->saved_desc = desc;
        s->max_frame_size = static_cast<size_t>(s->bitrate / desc.fps);
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Computed max frame size to be %lu\n", s->max_frame_size);
        configure_pyro_frame(s->pyro_frame, desc.width, desc.height, s->pyro_subs);
        bool res = create_pyro_encoder(s, desc);
        s->compressed_desc = desc;
        s->compressed_desc.color_spec = PYROWAVE;
        s->compressed_frame_pool.reconfigure(s->compressed_desc, s->max_frame_size);
        return res;
}

void ug_to_pyro_frame(const pyrowave_compress_state *s, const std::shared_ptr<video_frame>& f){
        PROFILE_FUNC;
        to_planar_data conv_data{};
        conv_data.width = static_cast<int>(f->tiles[0].width);
        conv_data.height = static_cast<int>(f->tiles[0].height);
        conv_data.in_data = reinterpret_cast<const unsigned char *>(f->tiles[0].data);
        for(int i = 0; i < 3; i++){
                conv_data.out_data[i] = static_cast<unsigned char *>(s->pyro_frame.f.data[i]);
                conv_data.out_linesize[i] = s->pyro_frame.f.row_stride_in_bytes[i];
        }

        s->to_planar_conv(conv_data);
}

std::shared_ptr<video_frame> pyrowave_compress_tile(void *state, std::shared_ptr<video_frame> video_frame){
        auto *s = static_cast<pyrowave_compress_state *>(state);

        if(!video_frame){
                return {};
        }
        PROFILE_FUNC;

        if(const auto frame_desc = video_desc_from_frame(video_frame.get()); !video_desc_eq(s->saved_desc, frame_desc)){
                if(!configure_with(s, frame_desc)){
                        return {};
                }
        }

        ug_to_pyro_frame(s, video_frame);

        pyrowave_rate_control rate_control{};
        rate_control.maximum_bitstream_size = s->max_frame_size - sizeof(pyrowave_frame_header);
        auto res= pyrowave_encoder_encode_cpu_synchronous(s->encoder.get(), &s->pyro_frame.f, &rate_control);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "pyrowave_encoder_encode_cpu_synchronous failed\n");
                return {};
        }

        size_t num_packets = 0;
        constexpr size_t packet_size = 1000;
        res = pyrowave_encoder_compute_num_packets(s->encoder.get(), packet_size, &num_packets);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get num_packets (%d)\n", res);
                return {};
        }

        std::vector<pyrowave_packet> packets(num_packets);
        auto out_frame = s->compressed_frame_pool.get_frame();

        char *bitstream_dst = out_frame->tiles[0].data + sizeof(pyrowave_frame_header);
        unsigned int bitstream_size = rate_control.maximum_bitstream_size;
        res = pyrowave_encoder_packetize(s->encoder.get(), packets.data(), packet_size, &num_packets, bitstream_dst, bitstream_size);
        if(res != PYROWAVE_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get encoded frame (%d)\n", res);
                return {};
        }
        size_t total_size = 0;
        for(const auto& packet : packets){
                total_size += packet.size;
        }

        pyrowave_frame_header hdr{
                .subs = s->pyro_subs,
        };
        memcpy(out_frame->tiles[0].data, &hdr, sizeof(hdr));

        assert(total_size <= s->max_frame_size);
        out_frame->tiles[0].data_len = total_size + sizeof(pyrowave_frame_header);

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
