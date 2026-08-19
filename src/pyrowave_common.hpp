/**
 * @file   pyrowave_common.hpp
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

#ifndef PYROWAVE_COMMON_HPP_50FC25A68E8E4E38B52C08E56A7C3311
#define PYROWAVE_COMMON_HPP_50FC25A68E8E4E38B52C08E56A7C3311

#include <memory>
#include <vector>
#include <vulkan/vulkan.h>
#include <pyrowave/pyrowave.h>

#include "types.h"
#include "utils/misc.h"

using pyrowave_device_unique = std::unique_ptr<pyrowave_device_opaque, deleter_from_fcn<pyrowave_device_destroy>>;
using pyrowave_encoder_unique = std::unique_ptr<pyrowave_encoder_opaque, deleter_from_fcn<pyrowave_encoder_destroy>>;
using pyrowave_decoder_unique = std::unique_ptr<pyrowave_decoder_opaque, deleter_from_fcn<pyrowave_decoder_destroy>>;

struct pyrowave_frame_header{
        subsampling subs;
};


struct pyrowave_cpu_frame{
        std::vector<std::vector<std::byte>> plane_datas;
        pyrowave_cpu_buffer f{};
};

void configure_pyro_frame(pyrowave_cpu_frame &f, const video_desc& desc);

constexpr subsampling pyro_subsampling_to_ug(const pyrowave_chroma_subsampling chroma_subsampling){
        switch (chroma_subsampling){
        case PYROWAVE_CHROMA_SUBSAMPLING_420: return SUBS_420;
        case PYROWAVE_CHROMA_SUBSAMPLING_444: return SUBS_444;
        default: return SUBS_UNKNOWN;
        }
}

constexpr pyrowave_chroma_subsampling ug_subsampling_to_pyro(const subsampling subsampling){
        switch (subsampling){
        case SUBS_420: return PYROWAVE_CHROMA_SUBSAMPLING_420;
        case SUBS_444: return PYROWAVE_CHROMA_SUBSAMPLING_444;
        default: return PYROWAVE_CHROMA_SUBSAMPLING_INT_MAX;
        }
}


#endif //PYROWAVE_COMMON_HPP_50FC25A68E8E4E38B52C08E56A7C3311
