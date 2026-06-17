/**
 * @file   openapv/to_openapv_conversions.h
 * @author Juraj Zemančík    <550535@mail.muni.cz>
 */
/*
 * Copyright (c) 2026 CESNET
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

#ifndef TO_OPENAPV_CONVERSIONS_H
#define TO_OPENAPV_CONVERSIONS_H

#include <oapv/oapv.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*uv_to_openapv_conversion_f)(const uint8_t *src, int width, int height, oapv_imgb_t *dst);

struct uv_to_openapv_conversion {
        codec_t src_color_format;
        int     dst_color_format;
        uv_to_openapv_conversion_f convert;
};

const struct uv_to_openapv_conversion* get_uv_to_openapv_conversion(codec_t codec);

#ifdef __cplusplus
}
#endif

#endif // TO_OPENAPV_CONVERSIONS_H
