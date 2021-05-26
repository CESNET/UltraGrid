/**
 * @file   hwaccel_vaapi.h
 * @author Martin Piatka <piatka@cesnet.cz>
 *
 * @brief This file contains functions related to hw acceleration
 */
/*
 * Copyright (c) 2018 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#ifndef HWACCEL_VAAPI_H
#define HWACCEL_VAAPI_H

#ifdef HWACC_VAAPI

#ifdef __cplusplus
extern "C" {
#endif

#include "hwaccel_libav_common.h"
#include <libavcodec/version.h>
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
#include <libavcodec/vaapi.h>
#endif
#include <libavutil/hwcontext_vaapi.h>

struct vaapi_ctx{
        AVBufferRef *device_ref;
        AVHWDeviceContext *device_ctx;
        AVVAAPIDeviceContext *device_vaapi_ctx;

        AVBufferRef *hw_frames_ctx;
        AVHWFramesContext *frame_ctx;

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
        VAProfile va_profile;
        VAEntrypoint va_entrypoint;
        VAConfigID va_config;
        VAContextID va_context;

        struct vaapi_context decoder_context;
#endif
};

void vaapi_uninit(struct hw_accel_state *s);
int vaapi_create_context(struct vaapi_ctx *ctx, AVCodecContext *codec_ctx);
int vaapi_init(struct AVCodecContext *s,
                struct hw_accel_state *state,
                codec_t out_codec);

#ifdef __cplusplus
}
#endif

#endif //HWACC_VAAPI

#endif
