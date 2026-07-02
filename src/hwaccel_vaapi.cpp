/**
 * @file   hwaccel_vaapi.cpp
 * @author Martin Piatka <piatka@cesnet.cz>
 *
 * @brief This file contains functions related to hw acceleration
 */
/*
 * Copyright (c) 2018-2023 CESNET z.s.p.o.
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
#endif // defined HAVE_CONFIG_H

extern "C" {
#include <libavcodec/version.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext_vaapi.h>
}

#include "debug.h"
#include "utils/misc.h"
#include "hwaccel_libav_common.h"
#include "hwaccel_vaapi.h"
#include "libavcodec/lavc_common.h"

#define MOD_NAME "[vaapi] "

namespace{
using AVFrame_uniq = std::unique_ptr<AVFrame, deleter_from_fcn_double_ptr<av_frame_free>>;
using AVBufferRef_uniq = std::unique_ptr<AVBufferRef, deleter_from_fcn_double_ptr<av_buffer_unref>>;
using AVHWFramesConstraints_uniq = std::unique_ptr<AVHWFramesConstraints, deleter_from_fcn_double_ptr<av_hwframe_constraints_free>>;

constexpr int DEFAULT_SURFACES = 20;

/**
 * Returns first SW format from valid_sw_formats. This is usually
 * AV_PIX_FMT_YUV420P or AV_PIX_FMT_NV12.
 *
 * The code borrows heavily from mpv
 * <https://github.com/mpv-player/mpv/blob/master/video/out/hwdec/hwdec_vaapi.c>
 * namely from function try_format_config().
 */
AVPixelFormat get_sw_format(AVBufferRef *device_ref){
        AVPixelFormat ret = AV_PIX_FMT_NONE;

        AVHWFramesConstraints_uniq fc(av_hwdevice_get_hwframe_constraints(device_ref, nullptr));
        if (!fc) {
                MSG(ERROR, "failed to retrieve libavutil frame constraints\n");
                return ret;
        }

        /*
         * We need a hwframe_ctx to be able to get the valid formats, but to
         * initialise it, we need a format, so we get the first format from the
         * hwconfig. We don't care about the other formats in the config because
         * the transfer formats list will already include them.
         */
        AVBufferRef_uniq fref(av_hwframe_ctx_alloc(device_ref));
        if (!fref) {
                MSG(ERROR, "failed to alloc libavutil frame context\n");
                return ret;
        }
        auto fctx = reinterpret_cast<AVHWFramesContext *>(fref->data);
        constexpr int dummy_size = 128; ///< just some valid size
        fctx->format    = AV_PIX_FMT_VAAPI;
        fctx->sw_format = fc->valid_sw_formats[0];
        fctx->width     = dummy_size;
        fctx->height    = dummy_size;
        if (av_hwframe_ctx_init(fref.get()) < 0) {
                MSG(ERROR, "failed to init libavutil frame context\n");
                return ret;
        }

        AVPixelFormat *fmts = nullptr;
        int rc = av_hwframe_transfer_get_formats(
                fref.get(), AV_HWFRAME_TRANSFER_DIRECTION_FROM, &fmts, 0);
        if(rc){
                MSG(ERROR, "failed to get libavutil frame context supported formats\n");
                return ret;
        }
        MSG(DEBUG, "Available HW layouts: %s\n", get_avpixfmts_names(fmts));
        ret = fmts[0];
        av_free(fmts);

        return ret;
}

}

int vaapi_init(AVCodecContext *s, hw_accel_state *state, codec_t /*out_codec*/){
        AVBufferRef_uniq device_ref;
        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_VAAPI, out_ptr(device_ref));
        if(ret < 0){
                return -1;
        }

        int decode_surfaces = DEFAULT_SURFACES;

        if (s->active_thread_type & FF_THREAD_FRAME)
                decode_surfaces += s->thread_count;

        auto sw_format = get_sw_format(device_ref.get());
        if (sw_format == AV_PIX_FMT_NONE) {
                sw_format = s->sw_pix_fmt;
                MSG(WARNING, "Using fallback HW frames layout: %s\n",
                    av_get_pix_fmt_name(s->sw_pix_fmt));
        } else{
                MSG(VERBOSE, "Selected HW frames layout: %s\n", av_get_pix_fmt_name(sw_format));
        }

        AVBufferRef_uniq hw_frames_ctx;
        ret = create_hw_frame_ctx(device_ref.get(),
                        s->coded_width,
                        s->coded_height,
                        AV_PIX_FMT_VAAPI,
                        sw_format,
                        decode_surfaces,
                        out_ptr(hw_frames_ctx));

        if(ret < 0){
                return -1;
        }

        AVFrame_uniq tmp_frame(av_frame_alloc());
        if(!tmp_frame){
                return -1;
        }

        s->hw_frames_ctx = hw_frames_ctx.release();

        state->type = HWACCEL_VAAPI;
        state->copy = true;
        state->ctx = nullptr;
        state->tmp_frame = tmp_frame.release();
        state->uninit = nullptr;

        return 0;
}

