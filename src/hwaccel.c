/**
 * @file   hwaccel.c
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

#include <assert.h>
#include "debug.h"
#include "hwaccel.h"

void hw_vdpau_ctx_init(hw_vdpau_ctx *ctx){
        ctx->device_ref = NULL;
        ctx->device = 0;
        ctx->get_proc_address = NULL;
}

void hw_vdpau_ctx_unref(hw_vdpau_ctx *ctx){
        av_buffer_unref(&ctx->device_ref);

        hw_vdpau_ctx_init(ctx);
}

hw_vdpau_ctx hw_vdpau_ctx_copy(const hw_vdpau_ctx *ctx){
        hw_vdpau_ctx new_ctx;
        hw_vdpau_ctx_init(&new_ctx);

        new_ctx.device_ref = av_buffer_ref(ctx->device_ref);
        new_ctx.device = ctx->device;
        new_ctx.get_proc_address = ctx->get_proc_address;

        return new_ctx;
}

void hw_vdpau_frame_init(hw_vdpau_frame *frame){
        hw_vdpau_ctx_init(&frame->hwctx);

        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                frame->buf[i] = NULL;
                frame->data[i] = NULL;
        }

        frame->surface = 0;
}

void hw_vdpau_frame_unref(hw_vdpau_frame *frame){
        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                av_buffer_unref(&frame->buf[i]);
        }

        hw_vdpau_ctx_unref(&frame->hwctx);

        hw_vdpau_frame_init(frame);
}

void hw_vdpau_free_extra_data(void *frame){
        hw_vdpau_frame_unref((hw_vdpau_frame *) frame);
}

hw_vdpau_frame hw_vdpau_frame_copy(const hw_vdpau_frame *frame){
        hw_vdpau_frame new_frame;
        hw_vdpau_frame_init(&new_frame);

        new_frame.hwctx = hw_vdpau_ctx_copy(&frame->hwctx);

        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                if(frame->buf[i])
                        new_frame.buf[i] = av_buffer_ref(frame->buf[i]);
                new_frame.data[i] = frame->data[i];
        }

        new_frame.surface = frame->surface;

        return new_frame;
}

void *hw_vdpau_frame_data_cpy(void *dst, const void *src, size_t n){
        assert(n == sizeof(hw_vdpau_frame));

        hw_vdpau_frame *new = (hw_vdpau_frame *) dst;

        *new = hw_vdpau_frame_copy((const hw_vdpau_frame *) src);

        return new;
}

hw_vdpau_frame *hw_vdpau_frame_from_avframe(hw_vdpau_frame *dst, const AVFrame *src){
        hw_vdpau_frame_init(dst);

        AVHWFramesContext *frame_ctx = (AVHWFramesContext *) src->hw_frames_ctx->data;
        AVHWDeviceContext *device_ctx = frame_ctx->device_ctx; 
        AVVDPAUDeviceContext *vdpau_ctx = (AVVDPAUDeviceContext *) device_ctx->hwctx;


        dst->hwctx.device_ref = av_buffer_ref(frame_ctx->device_ref);
        dst->hwctx.device = vdpau_ctx->device;
        dst->hwctx.get_proc_address = vdpau_ctx->get_proc_address;

        for(int i = 0; i < AV_NUM_DATA_POINTERS; i++){
                if(src->buf[i])
                        dst->buf[i] = av_buffer_ref(src->buf[i]);
                dst->data[i] = src->data[i];
        }

        dst->surface = (VdpVideoSurface) dst->data[3];

        return dst;
}

void vdp_funcs_init(vdp_funcs *f){
        memset(f, 0, sizeof(vdp_funcs));
}

static void load_func(void **f, VdpFuncId f_id, VdpDevice dev, VdpGetProcAddress *get_proc_address){
        VdpStatus st;

        st = get_proc_address(dev, f_id, f);

        if(st != VDP_STATUS_OK){
                error_msg("Error loading vdpau function id: %u\n", f_id);
        }
}

void vdp_funcs_load(vdp_funcs *f, VdpDevice device, VdpGetProcAddress *get_proc_address){
#define LOAD(f_point, f_id) (load_func((void **) (f_point), (f_id), device, get_proc_address))
        LOAD(&f->videoSurfaceGetParameters, VDP_FUNC_ID_VIDEO_SURFACE_GET_PARAMETERS);
        LOAD(&f->videoMixerCreate, VDP_FUNC_ID_VIDEO_MIXER_CREATE);
        LOAD(&f->videoMixerDestroy, VDP_FUNC_ID_VIDEO_MIXER_DESTROY);
        LOAD(&f->videoMixerRender, VDP_FUNC_ID_VIDEO_MIXER_RENDER);
        LOAD(&f->outputSurfaceCreate, VDP_FUNC_ID_OUTPUT_SURFACE_CREATE);
        LOAD(&f->outputSurfaceDestroy, VDP_FUNC_ID_OUTPUT_SURFACE_DESTROY);
        LOAD(&f->outputSurfaceGetParameters, VDP_FUNC_ID_OUTPUT_SURFACE_GET_PARAMETERS);
        LOAD(&f->getErrorString, VDP_FUNC_ID_GET_ERROR_STRING);
}
