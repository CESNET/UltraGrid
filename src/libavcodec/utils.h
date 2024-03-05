/**
 * @file   libavcodec/utils.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2023 CESNET, z. s. p. o.
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
#ifndef LIBAVCODEC_UTILS_H_184d6a99_7712_4002_9938_502CC8A4E9FE
#define LIBAVCODEC_UTILS_H_184d6a99_7712_4002_9938_502CC8A4E9FE

#include <libavutil/pixfmt.h>

#include "types.h"

#ifdef _MSC_VER
#define __attribute__(a)
#endif

#ifdef __cplusplus
extern "C" {
#endif

struct uv_to_av_pixfmt {
        codec_t uv_codec;
        enum AVPixelFormat av_pixfmt;
};
codec_t get_av_to_ug_pixfmt(enum AVPixelFormat av_pixfmt) __attribute__((const));
enum AVPixelFormat get_ug_to_av_pixfmt(codec_t ug_codec) __attribute__((const));
const struct uv_to_av_pixfmt *get_av_to_ug_pixfmts(void) __attribute__((const));

#ifdef __cplusplus
}
#endif

#endif // defined LIBAVCODEC_UTILS_H_184d6a99_7712_4002_9938_502CC8A4E9FE

