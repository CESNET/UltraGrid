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

#include <vulkan/vulkan.h>
#include <pyrowave/pyrowave.h>

#include "video_compress.h"
#include "lib_common.h"
#include "utils/misc.h"

namespace{

using pyrowave_encoder_unique = std::unique_ptr<pyrowave_encoder_opaque, deleter_from_fcn<pyrowave_encoder_destroy>>;

struct pyrowave_compress_state{
        pyrowave_encoder_unique encoder;
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

        return s.release();
}

void pyrowave_compress_done(void *state){
        auto *s = static_cast<pyrowave_compress_state *>(state);
        delete s;
}

constexpr video_compress_info pyrowave_info = []{
        video_compress_info info{};
        info.init_func = pyrowave_compress_init;
        info.done = pyrowave_compress_done;
        return info;
}();

REGISTER_MODULE(pyrowave, &pyrowave_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
