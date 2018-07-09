/**
 * @file   hwaccel_vdpau.h
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

#ifndef HWACCEL_VDPAU_H
#define HWACCEL_VDPAU_H

#ifdef HWACC_VDPAU

#ifdef __cplusplus
extern "C" {
#endif

#include "hwaccel_libav_common.h"
#include <libavutil/hwcontext_vdpau.h>
#include <libavcodec/vdpau.h>

/**
 * @brief Initializes vdpau hw. acceleration
 *
 * @param out_codec Expected output codec. If not HW_VDPAU copy mode is used.
 */
int vdpau_init(struct AVCodecContext *s,
		struct hw_accel_state *state,
		codec_t out_codec);

/**
 * hw_vdpau_ctx Contains VdpDevice handle and pointer to VdpGetProcAddress
 */
typedef struct hw_vdpau_ctx{
        AVBufferRef *device_ref; //Av codec buffer reference

        //These are just pointers to the device_ref buffer
        VdpDevice device;
        VdpGetProcAddress *get_proc_address;
} hw_vdpau_ctx;

/**
 * hw_vdpau_frame represents vdpau hw frames. Contains VdpVideoSurface as well
 *                as the corresponding hw_vdpau_ctx
 */
typedef struct hw_vdpau_frame{
        hw_vdpau_ctx hwctx;
        AVBufferRef *buf[AV_NUM_DATA_POINTERS];

        //These are just pointers to the buffer
        uint8_t *data[AV_NUM_DATA_POINTERS];
        VdpVideoSurface surface; // Same as data[3]
} hw_vdpau_frame;

/**
 * @brief Initializes hw_vdpau_ctx
 */
void hw_vdpau_ctx_init(hw_vdpau_ctx *ctx);

/**
 * @brief Releases reference counted resources in hw_vdpau_ctx
 */
void hw_vdpau_ctx_unref(hw_vdpau_ctx *ctx);

/**
 * @brief Copies hw_vdpau_ctx and makes new references to reference counted resources
 */
hw_vdpau_ctx hw_vdpau_ctx_copy(const hw_vdpau_ctx *ctx);


/**
 * @brief Initializes hw_vdpau_ctx
 */
void hw_vdpau_frame_init(hw_vdpau_frame *frame);

/**
 * @brief Releases reference counted resources in hw_vdpau_frame
 */
void hw_vdpau_frame_unref(hw_vdpau_frame *frame);

/**
 * @brief Calls @ref hw_vdpau_frame_unref
 */
void hw_vdpau_recycle_callback(struct video_frame *frame);

/**
 * @brief Copies hw_vdpau_frame and makes new references to reference counted resources
 */
hw_vdpau_frame hw_vdpau_frame_copy(const hw_vdpau_frame *frame);

void hw_vdpau_copy_callback(struct video_frame *frame);

/**
 * @brief Copies hw_vdpau_frame. Internally calls @ref hw_vdpau_frame_copy
 */
void *hw_vdpau_frame_data_cpy(void *dst, const void *src, size_t n);

/**
 * @brief Creates hw_vdpau_frame from AVFrame
 *
 * @param src Source frame
 */
hw_vdpau_frame *hw_vdpau_frame_from_avframe(hw_vdpau_frame *dst, const AVFrame *src);

/**
 * vdp_funcs contains vdpau function pointers
 */
typedef struct vdp_funcs{
        VdpVideoSurfaceGetParameters *videoSurfaceGetParameters;
        VdpVideoMixerCreate *videoMixerCreate;
        VdpVideoMixerDestroy *videoMixerDestroy;
        VdpVideoMixerRender *videoMixerRender;

        VdpOutputSurfaceCreate *outputSurfaceCreate;
        VdpOutputSurfaceDestroy *outputSurfaceDestroy;
        VdpOutputSurfaceGetParameters *outputSurfaceGetParameters;

        VdpGetErrorString *getErrorString;
} vdp_funcs;

/**
 * @brief Initializes the @ref vdp_funcs structure
 */
void vdp_funcs_init(vdp_funcs *);

/**
 * @brief Loads vdpau function pointers
 */
void vdp_funcs_load(vdp_funcs *, VdpDevice, VdpGetProcAddress *);

#ifdef __cplusplus
}
#endif

#endif //HWACC_VDPAU

#endif
