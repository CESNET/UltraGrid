/**
 * @file   pyrowave_common.cpp
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

#include "pyrowave_common.hpp"

#include <cassert>
#include <memory>

void configure_pyro_frame(pyrowave_cpu_frame& f, const video_desc &desc){
        f.f = {};
        f.f.format = PYROWAVE_CPU_BUFFER_FORMAT_YUV420P;
        f.f.width = static_cast<int>(desc.width);
        f.f.height = static_cast<int>(desc.height);

        constexpr size_t alignment = 256;

        f.plane_datas.resize(3);

        size_t luma_stride = ((desc.width + alignment - 1) / alignment) * alignment;
        size_t luma_size = luma_stride * desc.height;
        f.plane_datas[0].resize(luma_size + alignment - 1);
        f.f.data[0] = f.plane_datas[0].data();
        f.f.row_stride_in_bytes[0] = luma_stride;
        f.f.plane_size_in_bytes[0] = f.plane_datas[0].size();
        assert(std::align(alignment, luma_size, f.f.data[0], f.f.plane_size_in_bytes[0]));

        size_t chroma_stride = ((desc.width / 2 + alignment - 1) / alignment) * alignment;
        size_t chroma_size = chroma_stride * desc.height;

        f.plane_datas[1].resize(chroma_size + alignment - 1);
        f.f.data[1] = f.plane_datas[1].data();
        f.f.row_stride_in_bytes[1] = chroma_stride;
        f.f.plane_size_in_bytes[1] = f.plane_datas[1].size();
        assert(std::align(alignment, chroma_size, f.f.data[1], f.f.plane_size_in_bytes[1]));

        f.plane_datas[2].resize(chroma_size + alignment - 1);
        f.f.data[2] = f.plane_datas[2].data();
        f.f.row_stride_in_bytes[2] = chroma_stride;
        f.f.plane_size_in_bytes[2] = f.plane_datas[2].size();
        assert(std::align(alignment, chroma_size, f.f.data[2], f.f.plane_size_in_bytes[2]));
}

