/**
 * @file   hwaccel_vaapi.c
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

#include <libavcodec/version.h>
#include <libavutil/pixdesc.h>
#include <libavutil/hwcontext_vaapi.h>

#include "debug.h"
#include "hwaccel_libav_common.h"
#include "hwaccel_vaapi.h"
#include "libavcodec/lavc_common.h"

#define DEFAULT_SURFACES 20
#define MOD_NAME "[vaapi] "

struct vaapi_ctx {
        AVBufferRef          *device_ref;
        AVHWDeviceContext    *device_ctx;
        AVVAAPIDeviceContext *device_vaapi_ctx;

        AVBufferRef       *hw_frames_ctx;
        AVHWFramesContext *frame_ctx;
};

void vaapi_uninit(struct hw_accel_state *s){

        free(s->ctx);
}

/**
 * Returns first SW format from valid_sw_formats. This is usually
 * AV_PIX_FMT_YUV420P or AV_PIX_FMT_NV12.
 *
 * The code borrows heavily from mpv
 * <https://github.com/mpv-player/mpv/blob/master/video/out/hwdec/hwdec_vaapi.c>
 * namely from function try_format_config().
 */
static enum AVPixelFormat
get_sw_format(VADisplay display, AVBufferRef *device_ref,
              enum AVPixelFormat fallback_fmt)
{
        enum AVPixelFormat     ret       = AV_PIX_FMT_NONE;
        AVVAAPIHWConfig       *hwconfig  = NULL;
        VAConfigID             config_id = 0;
        AVHWFramesConstraints *fc        = NULL;

        VAStatus status = vaCreateConfig(
            display, VAProfileNone, VAEntrypointVideoProc, NULL, 0, &config_id);
        if (status != VA_STATUS_SUCCESS) {
                MSG(ERROR, "cannot create config\n");
                goto fail;
        }
        fc = av_hwdevice_get_hwframe_constraints(device_ref, hwconfig);
        if (!fc) {
                MSG(ERROR, "failed to retrieve libavutil frame constraints\n");
                goto fail;
        }

        /*
         * We need a hwframe_ctx to be able to get the valid formats, but to
         * initialise it, we need a format, so we get the first format from the
         * hwconfig. We don't care about the other formats in the config because
         * the transfer formats list will already include them.
         */
        AVBufferRef *fref = NULL;
        fref              = av_hwframe_ctx_alloc(device_ref);
        if (!fref) {
                MSG(ERROR, "failed to alloc libavutil frame context\n");
                goto fail;
        }
        AVHWFramesContext *fctx = (void *) fref->data;
        enum {
                INIT_SIZE = 128, ///< just some valid size
        };
        fctx->format    = AV_PIX_FMT_VAAPI;
        fctx->sw_format = fc->valid_sw_formats[0];
        fctx->width     = INIT_SIZE;
        fctx->height    = INIT_SIZE;
        if (av_hwframe_ctx_init(fref) < 0) {
                MSG(ERROR, "failed to init libavutil frame context\n");
                goto fail;
        }

        enum AVPixelFormat *fmts = NULL;
        int                 rc   = av_hwframe_transfer_get_formats(
            fref, AV_HWFRAME_TRANSFER_DIRECTION_FROM, &fmts, 0);
        if (rc) {
                MSG(ERROR, "failed to get libavutil frame context supported "
                           "formats\n");
                goto fail;
        }
        MSG(DEBUG, "Available HW layouts: %s\n", get_avpixfmts_names(fmts));
        ret = fmts[0];

fail:
        av_hwframe_constraints_free(&fc);
        av_buffer_unref(&fref);
        if (ret == AV_PIX_FMT_NONE) {
                MSG(WARNING, "Using fallback HW frames layout: %s\n",
                    av_get_pix_fmt_name(ret));
                ret = fallback_fmt;
        }
        MSG(VERBOSE, "Selected HW frames layout: %s\n",
            av_get_pix_fmt_name(ret));
        return ret;
}

int vaapi_init(struct AVCodecContext *s,
                struct hw_accel_state *state,
                codec_t out_codec)
{
        (void)(out_codec);
        struct vaapi_ctx *ctx = calloc(1, sizeof(struct vaapi_ctx));
        if(!ctx){
                return -1;
        }

        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_VAAPI, &ctx->device_ref);
        if(ret < 0)
                goto fail;

        ctx->device_ctx = (AVHWDeviceContext*)(void *)ctx->device_ref->data;
        ctx->device_vaapi_ctx = ctx->device_ctx->hwctx;

        int decode_surfaces = DEFAULT_SURFACES;

        if (s->active_thread_type & FF_THREAD_FRAME)
                decode_surfaces += s->thread_count;

        const enum AVPixelFormat sw_format = get_sw_format(
            ctx->device_vaapi_ctx->display, ctx->device_ref, s->sw_pix_fmt);
        ret = create_hw_frame_ctx(ctx->device_ref,
                        s->coded_width,
                        s->coded_height,
                        AV_PIX_FMT_VAAPI,
                        sw_format,
                        decode_surfaces,
                        &ctx->hw_frames_ctx);
        if(ret < 0)
                goto fail;

        ctx->frame_ctx = (AVHWFramesContext *)(void *) (ctx->hw_frames_ctx->data);

        s->hw_frames_ctx = ctx->hw_frames_ctx;

        state->tmp_frame = av_frame_alloc();
        if(!state->tmp_frame){
                ret = -1;
                goto fail;
        }
        state->type = HWACCEL_VAAPI;
        state->copy = true;
        state->ctx = ctx;
        state->uninit = vaapi_uninit;

        av_buffer_unref(&ctx->device_ref);
        return 0;


fail:
        av_frame_free(&state->tmp_frame);
        av_buffer_unref(&ctx->hw_frames_ctx);
        av_buffer_unref(&ctx->device_ref);
        free(ctx);
        return ret;
}
