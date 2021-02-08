/**
 * @file   aja_common.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
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

#include <iostream>
#include <map>
#include <ntv2enums.h>

#include "types.h"

#ifdef _MSC_VER
#define log_msg(x, ...) fprintf(stderr, __VA_ARGS__)
#undef LOG
#define LOG(level) if (level > log_level) ; else std::cerr
#endif


// compat
#ifndef NTV2_AUDIOSIZE_MAX
#define NTV2_AUDIOSIZE_MAX      (401 * 1024)
#endif

namespace ultragrid {
namespace aja {
static const std::map<NTV2FrameBufferFormat, codec_t> codec_map = {
        { NTV2_FBF_10BIT_YCBCR, v210 },
        { NTV2_FBF_8BIT_YCBCR, UYVY },
        { NTV2_FBF_ABGR, RGBA },
        { NTV2_FBF_10BIT_DPX, R10k },
        { NTV2_FBF_8BIT_YCBCR_YUY2, YUYV },
        { NTV2_FBF_24BIT_RGB, RGB },
        { NTV2_FBF_24BIT_BGR, BGR },
        { NTV2_FBF_48BIT_RGB, RG48 },
        { NTV2_FBF_12BIT_RGB_PACKED, R12L },
};
} // end of namespace aja
} // end of namespace ultragrid

#ifdef __cplusplus
extern "C" {
#endif

void
vc_copylineR12AtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift);
void
vc_copylineR12LtoR12A(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift);

#ifdef __cplusplus
}
#endif
