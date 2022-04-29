/**
 * @file   libavcodec/lavc_video.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2022 CESNET, z. s. p. o.
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

#ifndef LIBAVCODEC_LAVC_VIDEO_62BB95A5_236F_4D7C_9DD6_85FD35E6BEB2
#define LIBAVCODEC_LAVC_VIDEO_62BB95A5_236F_4D7C_9DD6_85FD35E6BEB2

#ifdef __cplusplus
#include <cstdio>
#else
#include <stdio.h>
#endif

#include "libavcodec_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void serialize_video_avframe(const void *f, FILE *out);

#ifdef HAVE_SWSCALE
struct SwsContext;
struct SwsContext *getSwsContext(unsigned int SrcW, unsigned int SrcH, enum AVPixelFormat SrcFormat, unsigned int DstW, unsigned int DstH, enum AVPixelFormat DstFormat, int64_t Flags);
#endif // defined HAVE_SWSCALE

#ifdef __cplusplus
}
#endif

#endif // !defined LIBAVCODEC_LAVC_VIDEO_62BB95A5_236F_4D7C_9DD6_85FD35E6BEB2

