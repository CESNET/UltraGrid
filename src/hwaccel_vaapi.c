/**
 * @file   hwaccel_vaapi.c
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

#include "hwaccel_vaapi.h"

#include "debug.h"

void vaapi_uninit(struct hw_accel_state *s){

        free(s->ctx);
}


#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
static const struct {
        enum AVCodecID av_codec_id;
        int codec_profile;
        VAProfile va_profile;
} vaapi_profiles[] = {
        {AV_CODEC_ID_MPEG2VIDEO, FF_PROFILE_MPEG2_SIMPLE, VAProfileMPEG2Simple},
        {AV_CODEC_ID_MPEG2VIDEO, FF_PROFILE_MPEG2_MAIN, VAProfileMPEG2Main},
        {AV_CODEC_ID_H264, FF_PROFILE_H264_CONSTRAINED_BASELINE, VAProfileH264ConstrainedBaseline},
        {AV_CODEC_ID_H264, FF_PROFILE_H264_BASELINE, VAProfileH264Baseline},
        {AV_CODEC_ID_H264, FF_PROFILE_H264_MAIN, VAProfileH264Main},
        {AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH, VAProfileH264High},
#if VA_CHECK_VERSION(0, 37, 0)
        {AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN, VAProfileHEVCMain},
#endif
};

int vaapi_create_context(struct vaapi_ctx *ctx,
                AVCodecContext *codec_ctx)
{
        const AVCodecDescriptor *codec_desc;

        codec_desc = avcodec_descriptor_get(codec_ctx->codec_id);
        if(!codec_desc){
                return -1;
        }

        int profile_count = vaMaxNumProfiles(ctx->device_vaapi_ctx->display);
        log_msg(LOG_LEVEL_VERBOSE, "VAAPI Profile count: %d\n", profile_count);

        VAProfile *list = av_malloc(profile_count * sizeof(VAProfile));
        if(!list){
                return -1;
        }

        VAStatus status = vaQueryConfigProfiles(ctx->device_vaapi_ctx->display,
                        list, &profile_count);
        if(status != VA_STATUS_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Profile query failed: %d (%s)\n", status, vaErrorStr(status));
                av_free(list);
                return -1;
        }

        VAProfile profile = VAProfileNone;
        int match = 0;

        for(unsigned i = 0; i < FF_ARRAY_ELEMS(vaapi_profiles); i++){
                if(vaapi_profiles[i].av_codec_id != codec_ctx->codec_id)
                        continue;

                if(vaapi_profiles[i].codec_profile == codec_ctx->profile){
                        profile = vaapi_profiles[i].va_profile;
                        break;
                }
        }

        for(int i = 0; i < profile_count; i++){
                if(profile == list[i])
                        match = 1;
        }

        av_freep(&list);

        if(!match){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Profile not supported \n");
                return -1;
        }

        ctx->va_profile = profile;
        ctx->va_entrypoint = VAEntrypointVLD;

        status = vaCreateConfig(ctx->device_vaapi_ctx->display, ctx->va_profile,
                        ctx->va_entrypoint, 0, 0, &ctx->va_config);
        if(status != VA_STATUS_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Create config failed: %d (%s)\n", status, vaErrorStr(status));
                return -1;
        }

        AVVAAPIHWConfig *hwconfig = av_hwdevice_hwconfig_alloc(ctx->device_ref);
        if(!hwconfig){
                log_msg(LOG_LEVEL_WARNING, "[lavd] Failed to get constraints. Will try to continue anyways...\n");
                return 0;
        }

        hwconfig->config_id = ctx->va_config;
        AVHWFramesConstraints *constraints = av_hwdevice_get_hwframe_constraints(ctx->device_ref, hwconfig);
        if (!constraints){
                log_msg(LOG_LEVEL_WARNING, "[lavd] Failed to get constraints. Will try to continue anyways...\n");
                av_freep(&hwconfig);
                return 0;
        }

        if (codec_ctx->coded_width  < constraints->min_width  ||
                        codec_ctx->coded_width  > constraints->max_width  ||
                        codec_ctx->coded_height < constraints->min_height ||
                        codec_ctx->coded_height > constraints->max_height)
        {
                log_msg(LOG_LEVEL_WARNING, "[lavd] VAAPI hw does not support the resolution %dx%d\n",
                                codec_ctx->coded_width,
                                codec_ctx->coded_height);
                av_hwframe_constraints_free(&constraints);
                av_freep(&hwconfig);
                return -1;
        }

        av_hwframe_constraints_free(&constraints);
        av_freep(&hwconfig);

        return 0;
}
#endif //LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)

int vaapi_init(struct AVCodecContext *s,
                struct hw_accel_state *state,
                codec_t out_codec)
{
        (void *)(out_codec);
        struct vaapi_ctx *ctx = calloc(1, sizeof(struct vaapi_ctx));
        if(!ctx){
                return -1;
        }

        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_VAAPI, &ctx->device_ref);
        if(ret < 0)
                goto fail;

        ctx->device_ctx = (AVHWDeviceContext*)ctx->device_ref->data;
        ctx->device_vaapi_ctx = ctx->device_ctx->hwctx;

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
        ret = vaapi_create_context(ctx, s);
        if(ret < 0)
                goto fail;
#endif

        int decode_surfaces = DEFAULT_SURFACES;

        if (s->active_thread_type & FF_THREAD_FRAME)
                decode_surfaces += s->thread_count;

        ret = create_hw_frame_ctx(ctx->device_ref,
                        s,
                        AV_PIX_FMT_VAAPI,
                        s->sw_pix_fmt,
                        decode_surfaces,
                        &ctx->hw_frames_ctx);
        if(ret < 0)
                goto fail;

        ctx->frame_ctx = (AVHWFramesContext *) (ctx->hw_frames_ctx->data);

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

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
        AVVAAPIFramesContext *avfc = ctx->frame_ctx->hwctx;
        VAStatus status = vaCreateContext(ctx->device_vaapi_ctx->display,
                        ctx->va_config, s->coded_width, s->coded_height,
                        VA_PROGRESSIVE,
                        avfc->surface_ids,
                        avfc->nb_surfaces,
                        &ctx->va_context);

        if(status != VA_STATUS_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Create config failed: %d (%s)\n", status, vaErrorStr(status));
                ret = -1;
                goto fail;
        }

        ctx->decoder_context.display = ctx->device_vaapi_ctx->display;
        ctx->decoder_context.config_id = ctx->va_config;
        ctx->decoder_context.context_id = ctx->va_context;

        s->hwaccel_context = &ctx->decoder_context;
#endif //LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)

        av_buffer_unref(&ctx->device_ref);
        return 0;


fail:
        av_frame_free(&state->tmp_frame);
        av_buffer_unref(&ctx->hw_frames_ctx);
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
        if(ctx->device_vaapi_ctx)
                vaDestroyConfig(ctx->device_vaapi_ctx->display, ctx->va_config);
#endif
        av_buffer_unref(&ctx->device_ref);
        free(ctx);
        return ret;
}
