/**
 * @file   hwaccel_libav_common.c
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "hwaccel_libav_common.h"
#include "libavcodec_common.h"

void hwaccel_state_init(struct hw_accel_state *hwaccel){
        hwaccel->type = HWACCEL_NONE;
        hwaccel->copy = false;
        hwaccel->uninit = NULL;
        hwaccel->tmp_frame = NULL;
        hwaccel->uninit = NULL;
        hwaccel->ctx = NULL;
}

void hwaccel_state_reset(struct hw_accel_state *hwaccel){
        if(hwaccel->ctx){
                hwaccel->uninit(hwaccel);
        }

        if(hwaccel->tmp_frame){
                av_frame_free(&hwaccel->tmp_frame);
        }

        hwaccel_state_init(hwaccel);
}

int create_hw_device_ctx(enum AVHWDeviceType type, AVBufferRef **device_ref){
        const char *device_paths[] = { NULL, "/dev/dri/renderD128" };

        int ret;
        for(size_t i = 0; i < sizeof(device_paths) / sizeof(*device_paths); i++){
                ret = av_hwdevice_ctx_create(device_ref, type, device_paths[i], NULL, 0);
                if(ret == 0)
                        return 0;
        }

        log_msg(LOG_LEVEL_ERROR, "[hw accel] Unable to create hwdevice!!\n");
        return ret;
}

int create_hw_frame_ctx(AVBufferRef *device_ref,
                int width,
                int height,
                enum AVPixelFormat format,
                enum AVPixelFormat sw_format,
                int decode_surfaces,
                AVBufferRef **ctx)
{
        *ctx = av_hwframe_ctx_alloc(device_ref);
        if(!*ctx){
                log_msg(LOG_LEVEL_ERROR, "[hw accel] Failed to allocate hwframe_ctx!!\n");
                return -1;
        }

        AVHWFramesContext *frames_ctx = (AVHWFramesContext *) (*ctx)->data;
        frames_ctx->format    = format;
        frames_ctx->width     = width;
        frames_ctx->height    = height;
        frames_ctx->sw_format = sw_format;
        frames_ctx->initial_pool_size = decode_surfaces;

        int ret = av_hwframe_ctx_init(*ctx);
        if (ret < 0) {
                av_buffer_unref(ctx);
                *ctx = NULL;
                log_msg(LOG_LEVEL_ERROR, "[hw accel] Unable to init hwframe_ctx: %s\n\n",
						av_err2str(ret));
                return ret;
        }

        return 0;
}

void transfer_frame(struct hw_accel_state *s, AVFrame *frame){
        av_hwframe_transfer_data(s->tmp_frame, frame, 0);

        av_frame_copy_props(s->tmp_frame, frame);

        av_frame_unref(frame);
        av_frame_move_ref(frame, s->tmp_frame);
}
