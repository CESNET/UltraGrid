/**
 * @file   hwaccel_rpi4.h
 * @author Martin Piatka <piatka@cesnet.cz>
 *
 * @brief This file contains functions related to hw acceleration
 */
/*
 * Copyright (c) 2021 CESNET z.s.p.o.
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

#ifndef HWACCEL_RPI4_H
#define HWACCEL_RPI4_H

#include "libavcodec_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct av_frame_wrapper{
        AVBufferRef *buf[AV_NUM_DATA_POINTERS];
        char *data[AV_NUM_DATA_POINTERS];
} av_frame_wrapper;

static inline void av_frame_wrapper_recycle(struct video_frame *f){
        for(unsigned i = 0; i < f->tile_count; i++){
                av_frame_wrapper *wrapper = (av_frame_wrapper *)(void *) f->tiles[i].data;

                for(int j = 0; j < AV_NUM_DATA_POINTERS; j++){
                        av_buffer_unref(&wrapper->buf[j]);
                        wrapper->data[j] = NULL;
                }
        }
}

static inline void av_frame_wrapper_copy(struct video_frame *f){
        for(unsigned i = 0; i < f->tile_count; i++){
                av_frame_wrapper *wrapper = (av_frame_wrapper *)(void *) f->tiles[i].data;

                for(int j = 0; j < AV_NUM_DATA_POINTERS; j++){
                        if(wrapper->buf[j])
                                wrapper->buf[j] = av_buffer_ref(wrapper->buf[j]);
                }
        }
}

#ifdef __cplusplus
}
#endif

#endif
