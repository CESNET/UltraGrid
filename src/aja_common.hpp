/**
 * @file   aja_common.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2026 CESNET, zájmové sdružení právnických osob
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

#include <ntv2enums.h>

#include "types.h"

class CNTV2Card;

#ifdef _MSC_VER
#include <stdlib.h>
#include <string.h>

#if _MSC_VER <= 1800
#define constexpr
#define noexcept
#define snprintf _snprintf
#endif

#define strtok_r strtok_s
#ifndef strcasecmp
#define strcasecmp  _stricmp
#endif
#ifndef strncasecmp
#define strncasecmp _strnicmp
#endif
#endif

// compat
#ifndef NTV2_AUDIOSIZE_MAX
#define NTV2_AUDIOSIZE_MAX      (401 * 1024)
#endif

#ifdef __cplusplus
extern "C" {
#endif

NTV2FrameBufferFormat get_ntv2_from_ug_pixfmt(codec_t ug_pixfmt);
codec_t               get_ug_from_ntv2_pixfmt(NTV2FrameBufferFormat ntv2_format);
enum { UG_NTV2_FMT_MAX_COUNT = 20 };
unsigned get_ntv2_pixfmts(NTV2FrameBufferFormat *formats);

void
vc_copylineR12AtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift);
void
vc_copylineR12LtoR12A(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift);

void print_aja_device_details(CNTV2Card *device);

#ifdef __cplusplus
}
#endif
