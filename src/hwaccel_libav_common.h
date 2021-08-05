/**
 * @file   hwaccel_libav_common.h
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

#ifndef HWACCEL_LIBAV_COMMON_H
#define HWACCEL_LIBAV_COMMON_H
#ifdef HWACC_COMMON

#include "types.h"
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>

#define DEFAULT_SURFACES 20

#ifdef __cplusplus
extern "C" {
#endif

/**
 * hw_accel_state describes the current state of hw. accleration
 */

struct hw_accel_state {
        enum {
                HWACCEL_NONE,
                HWACCEL_VDPAU,
                HWACCEL_VAAPI,
                HWACCEL_VIDEOTOOLBOX
        } type;

        bool copy; ///< Specifies whether to use the copy mode
        AVFrame *tmp_frame; ///< Temporary frame used when copying

        void (*uninit)(struct hw_accel_state*); ///< Optional uninit function ptr

        void *ctx; ///< Type depends on hwaccel type
};

/**
 * @brief Initializes hw_accel_state structure
 */
void hwaccel_state_init(struct hw_accel_state *hwaccel);

/**
 * @brief Uninitializes existing acceleration and resets hw_accel_state to initial state
 */
void hwaccel_state_reset(struct hw_accel_state *hwaccel);

/**
 * @brief Creates hw. device context
 *
 * @param type Type of device to create
 */
int create_hw_device_ctx(enum AVHWDeviceType type, AVBufferRef **device_ref);

/**
 * @brief Creates hw. frames conext with given parameters
 *
 * @param s Codec context containing information about decoded video
 * @param format Format of hardware surfaces
 * @param sw_format Internal software format
 * @param decode_surfaces Initial pool size
 */
int create_hw_frame_ctx(AVBufferRef *device_ref,
                int width,
                int height,
                enum AVPixelFormat format,
                enum AVPixelFormat sw_format,
                int decode_surfaces,
                AVBufferRef **ctx);

/**
 * @brief Transfers hw frame from hw memory
 *
 * @param frame Contains hw frame. After calling this function it will contain 
 *              a sw frame
 */
void transfer_frame(struct hw_accel_state *s, AVFrame *frame);

#ifdef __cplusplus
}
#endif

#endif //HWACC_COMMON
#endif
