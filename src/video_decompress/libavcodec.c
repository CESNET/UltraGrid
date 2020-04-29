/**
 * @file   video_decompress/libavcodec.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2019 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "libavcodec_common.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/resource_manager.h"
#include "video.h"
#include "video_decompress.h"

#ifndef AV_PIX_FMT_FLAG_HWACCEL
#define AV_PIX_FMT_FLAG_HWACCEL PIX_FMT_HWACCEL
#endif

#include "hwaccel_libav_common.h"
#include "hwaccel_vdpau.h"
#include "hwaccel_vaapi.h"

#define MOD_NAME "[lavd] "

struct state_libavcodec_decompress {
        pthread_mutex_t *global_lavcd_lock;
        AVCodecContext  *codec_ctx;
        AVFrame         *frame;
        AVPacket         pkt;

        struct video_desc desc;
        int              pitch;
        int              rgb_shift[3];
        int              max_compressed_len;
        codec_t          internal_codec;
        codec_t          out_codec;
        bool             blacklist_vdpau;

        unsigned         last_frame_seq:22; // This gives last sucessfully decoded frame seq number. It is the buffer number from the packet format header, uses 22 bits.
        bool             last_frame_seq_initialized;

        struct video_desc saved_desc;
        unsigned int     broken_h264_mt_decoding_workaroud_warning_displayed;
        bool             broken_h264_mt_decoding_workaroud_active;

#ifdef HWACC_COMMON
        struct hw_accel_state hwaccel;
#endif
};

static int change_pixfmt(AVFrame *frame, unsigned char *dst, int av_codec,
                codec_t out_codec, int width, int height, int pitch, int rgb_shift[static restrict 3]);
static void error_callback(void *, int, const char *, va_list);
static enum AVPixelFormat get_format_callback(struct AVCodecContext *s, const enum AVPixelFormat *fmt);

static bool broken_h264_mt_decoding = false;

static void deconfigure(struct state_libavcodec_decompress *s)
{
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
        if (s->codec_ctx) {
                int ret;
                ret = avcodec_send_packet(s->codec_ctx, NULL);
                if (ret != 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                        ret);
                }
                do {
                        ret = avcodec_receive_frame(s->codec_ctx, s->frame);
                        if (ret != 0 && ret != AVERROR_EOF) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                                ret);
                                break;
                        }

                } while (ret != AVERROR_EOF);
        }
#endif
        if(s->codec_ctx) {
                pthread_mutex_lock(s->global_lavcd_lock);
                avcodec_close(s->codec_ctx);
                avcodec_free_context(&s->codec_ctx);
                pthread_mutex_unlock(s->global_lavcd_lock);
        }
        av_free(s->frame);
        s->frame = NULL;
        av_packet_unref(&s->pkt);

#ifdef HWACC_COMMON
        hwaccel_state_reset(&s->hwaccel);
#endif
}

static void set_codec_context_params(struct state_libavcodec_decompress *s)
{
        // zero should mean count equal to the number of virtual cores
        if (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) {
                if(!broken_h264_mt_decoding) {
                        s->codec_ctx->thread_count = 0; // == X264_THREADS_AUTO, perhaps same for other codecs
                        s->codec_ctx->thread_type = FF_THREAD_SLICE;
                        s->broken_h264_mt_decoding_workaroud_active = false;
                } else {
                        s->broken_h264_mt_decoding_workaroud_active = true;
                }
        } else {
#if 0
                log_msg(LOG_LEVEL_WARNING, "[lavd] Warning: Codec doesn't support slice-based multithreading.\n");
                if(s->codec->capabilities & CODEC_CAP_FRAME_THREADS) {
                        s->codec_ctx->thread_count = 0;
                        s->codec_ctx->thread_type = FF_THREAD_FRAME;
                } else {
                        fprintf(stderr, "[lavd] Warning: Codec doesn't support frame-based multithreading.\n");
                }
#endif
        }

        s->codec_ctx->flags2 |= AV_CODEC_FLAG2_FAST;

        // set by decoder
        s->codec_ctx->pix_fmt = AV_PIX_FMT_NONE;
        // callback to negotiate pixel format that is supported by UG
        s->codec_ctx->get_format = get_format_callback;

        s->codec_ctx->opaque = s;
}

static void jpeg_callback(void)
{
        log_msg(LOG_LEVEL_WARNING, "[lavd] Warning: JPEG decoder "
                        "will use full-scale YUV.\n");
}

struct decoder_info {
        codec_t ug_codec;
        enum AVCodecID avcodec_id;
        void (*codec_callback)(void);
        // Note:
        // Make sure that if adding hw decoders to prefered_decoders[] that
        // that decoder fails if there is not the HW during init, not while decoding
        // frames (like vdpau does). Otherwise, such a decoder would be initialized
        // but no frame decoded then.
        // Note 2:
        // cuvid decoders cannot be currently used as the default ones because they
        // currently support only 4:2:0 subsampling and fail during decoding if other
        // subsampling is given.
        const char *preferred_decoders[11]; // must be NULL-terminated
};

static const struct decoder_info decoders[] = {
        { H264, AV_CODEC_ID_H264, NULL, { NULL /* "h264_cuvid" */ } },
        { H265, AV_CODEC_ID_HEVC, NULL, { NULL /* "hevc_cuvid" */ } },
        { MJPG, AV_CODEC_ID_MJPEG, jpeg_callback, { NULL } },
        { JPEG, AV_CODEC_ID_MJPEG, jpeg_callback, { NULL } },
        { J2K, AV_CODEC_ID_JPEG2000, NULL, { NULL } },
        { J2KR, AV_CODEC_ID_JPEG2000, NULL, { NULL } },
        { VP8, AV_CODEC_ID_VP8, NULL, { NULL } },
        { VP9, AV_CODEC_ID_VP9, NULL, { NULL } },
        { HFYU, AV_CODEC_ID_HUFFYUV, NULL, { NULL } },
        { FFV1, AV_CODEC_ID_FFV1, NULL, { NULL } },
        { AV1, AV_CODEC_ID_AV1, NULL, { NULL } },
};

ADD_TO_PARAM(force_lavd_decoder, "force-lavd-decoder", "* force-lavd-decoder=<decoder>[:<decoder2>...]\n"
                "  Forces specified Libavcodec decoder. If more need to be specified, use colon as a delimiter\n");

#ifdef HWACC_COMMON
ADD_TO_PARAM(force_hw_accel, "use-hw-accel", "* use-hw-accel\n"
                "  Tries to use hardware acceleration. \n");
#endif
static bool configure_with(struct state_libavcodec_decompress *s,
                struct video_desc desc, void *extradata, int extradata_size)
{
        const struct decoder_info *dec = NULL;

        for (unsigned int i = 0; i < sizeof decoders / sizeof decoders[0]; ++i) {
                if (decoders[i].ug_codec == desc.color_spec) {
                        dec = &decoders[i];
                        break;
                }
        }

        if (dec == NULL) {
                log_msg(LOG_LEVEL_ERROR, "[lavd] Unsupported codec!!!\n");
                return false;
        }

        if (dec->codec_callback) {
                dec->codec_callback();
        }

        // construct priority list of decoders that can be used for the codec
        AVCodec *codecs_available[13]; // max num of preferred decoders (10) + user supplied + default one + NULL
        memset(codecs_available, 0, sizeof codecs_available);
        unsigned int codec_index = 0;
        // first try codec specified from cmdline if any
        if (get_commandline_param("force-lavd-decoder")) {
                const char *param = get_commandline_param("force-lavd-decoder");
                char *val = alloca(strlen(param) + 1);
                strcpy(val, param);
                char *item, *save_ptr;
                while ((item = strtok_r(val, ":", &save_ptr))) {
                        val = NULL;
                        AVCodec *codec = avcodec_find_decoder_by_name(item);
                        if (codec == NULL) {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Decoder not found: %s\n", item);
                        } else {
                                if (codec->id == dec->avcodec_id) {
                                        if (codec_index < (sizeof codecs_available / sizeof codecs_available[0] - 1)) {
                                                codecs_available[codec_index++] = codec;
                                        }
                                } else {
                                        log_msg(LOG_LEVEL_WARNING, "[lavd] Decoder not valid for codec: %s\n", item);
                                }
                        }
                }
        }
        // then try preferred codecs
        const char * const *preferred_decoders_it = dec->preferred_decoders;
        while (*preferred_decoders_it) {
                AVCodec *codec = avcodec_find_decoder_by_name(*preferred_decoders_it);
                if (codec == NULL) {
                        log_msg(LOG_LEVEL_VERBOSE, "[lavd] Decoder not available: %s\n", *preferred_decoders_it);
                        preferred_decoders_it++;
                        continue;
                } else {
                        if (codec_index < (sizeof codecs_available / sizeof codecs_available[0] - 1)) {
                                codecs_available[codec_index++] = codec;
                        }
                }
                preferred_decoders_it++;
        }
        // finally, add a default one if there are no preferred encoders or all fail
        if (codec_index < (sizeof codecs_available / sizeof codecs_available[0]) - 1) {
                AVCodec *default_decoder = avcodec_find_decoder(dec->avcodec_id);
                if (default_decoder == NULL) {
                        log_msg(LOG_LEVEL_WARNING, "[lavd] No decoder found for the input codec (libavcodec perhaps compiled without any)!\n"
                                                "Use \"--param decompress=<d> to select a different decoder than libavcodec if there is any eligibe.\n");
                } else {
                        codecs_available[codec_index++] = default_decoder;
                }
        }

        // initialize the codec - use the first decoder initialization of which succeeds
        AVCodec **codec_it = codecs_available;
        while (*codec_it) {
                log_msg(LOG_LEVEL_VERBOSE, "[lavd] Trying decoder: %s\n", (*codec_it)->name);
                s->codec_ctx = avcodec_alloc_context3(*codec_it);
                if(s->codec_ctx == NULL) {
                        log_msg(LOG_LEVEL_ERROR, "[lavd] Unable to allocate codec context.\n");
                        return false;
                }
                if (extradata) {
                        s->codec_ctx->extradata = malloc(extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
                        memcpy(s->codec_ctx->extradata, extradata, extradata_size);
                        s->codec_ctx->extradata_size = extradata_size;
                }
                s->codec_ctx->width = desc.width;
                s->codec_ctx->height = desc.height;
                set_codec_context_params(s);
                pthread_mutex_lock(s->global_lavcd_lock);
                if (avcodec_open2(s->codec_ctx, *codec_it, NULL) < 0) {
                        avcodec_free_context(&s->codec_ctx);
                        pthread_mutex_unlock(s->global_lavcd_lock);
                        log_msg(LOG_LEVEL_WARNING, "[lavd] Unable to open decoder %s.\n", (*codec_it)->name);
                        codec_it++;
                        continue;
                } else {
                        pthread_mutex_unlock(s->global_lavcd_lock);
                        log_msg(LOG_LEVEL_NOTICE, "[lavd] Using decoder: %s\n", (*codec_it)->name);
                        break;
                }
        }

        if (s->codec_ctx == NULL) {
                log_msg(LOG_LEVEL_ERROR, "[lavd] Decoder could have not been initialized for codec %s.\n",
                                get_codec_name(desc.color_spec));
                return false;
        }

        s->frame = av_frame_alloc();
        if(!s->frame) {
                log_msg(LOG_LEVEL_ERROR, "[lavd] Unable allocate frame.\n");
                return false;
        }

        av_init_packet(&s->pkt);

        s->last_frame_seq_initialized = false;
        s->saved_desc = desc;

        return true;
}

static void * libavcodec_decompress_init(void)
{
        struct state_libavcodec_decompress *s;

        s = (struct state_libavcodec_decompress *)
                calloc(1, sizeof(struct state_libavcodec_decompress));

        s->global_lavcd_lock = rm_acquire_shared_lock(LAVCD_LOCK_NAME);
        if (log_level >= LOG_LEVEL_VERBOSE) {
                av_log_set_level(AV_LOG_VERBOSE);
        }

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        /*   register all the codecs (you can also register only the codec
         *         you wish to have smaller code */
        avcodec_register_all();
#endif

        s->codec_ctx = NULL;
        s->frame = NULL;
        av_init_packet(&s->pkt);
        s->pkt.data = NULL;
        s->pkt.size = 0;

        av_log_set_callback(error_callback);

#ifdef HWACC_COMMON
        hwaccel_state_init(&s->hwaccel);
#endif

        return s;
}

static int libavcodec_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_libavcodec_decompress *s =
                (struct state_libavcodec_decompress *) state;

        s->pitch = pitch;
        s->rgb_shift[R] = rshift;
        s->rgb_shift[G] = gshift;
        s->rgb_shift[B] = bshift;
        s->internal_codec = VIDEO_CODEC_NONE;
        s->blacklist_vdpau = false;
        s->out_codec = out_codec;
        s->desc = desc;

        deconfigure(s);
        if (libav_codec_has_extradata(desc.color_spec)) {
                // for codecs that have metadata we have to defer initialization
                // because we don't have the data right now
                return TRUE;
        } else {
                return configure_with(s, desc, NULL, 0);
        }
}

static bool has_conversion(enum AVPixelFormat pix_fmt, codec_t *ug_pix_fmt) {

        for (const struct av_to_uv_conversion *c = get_av_to_uv_conversions(); c->uv_codec != VIDEO_CODEC_NONE; c++) {
                if (c->av_codec != pix_fmt) { // this conversion is not valid
                        continue;
                }

                if (c->native) {
                        *ug_pix_fmt = c->uv_codec;
                        return true;
                }
        }

        for (const struct av_to_uv_conversion *c = get_av_to_uv_conversions(); c->uv_codec != VIDEO_CODEC_NONE; c++) {
                if (c->av_codec != pix_fmt) { // this conversion is not valid
                        continue;
                }

                *ug_pix_fmt = c->uv_codec;
                return true;
        }
        return false;
}

static enum AVPixelFormat get_format_callback(struct AVCodecContext *s __attribute__((unused)), const enum AVPixelFormat *fmt)
{
        if (log_level >= LOG_LEVEL_DEBUG) {
                char out[1024] = "[lavd] Available output pixel formats:";
                const enum AVPixelFormat *it = fmt;
                while (*it != AV_PIX_FMT_NONE) {
                        strncat(out, " ", sizeof out - strlen(out) - 1);
                        strncat(out, av_get_pix_fmt_name(*it++), sizeof out - strlen(out) - 1);
                }
                log_msg(LOG_LEVEL_DEBUG, "%s\n", out);
        }

        struct state_libavcodec_decompress *state = (struct state_libavcodec_decompress *) s->opaque;
        bool hwaccel = get_commandline_param("use-hw-accel") != NULL;
#ifdef HWACC_COMMON
        hwaccel_state_reset(&state->hwaccel);

        static const struct{
                enum AVPixelFormat pix_fmt;
                int (*init_func)(AVCodecContext *, struct hw_accel_state *, codec_t);
        } accels[] = {
#ifdef HWACC_VDPAU
                {AV_PIX_FMT_VDPAU, vdpau_init},
#endif
#ifdef HWACC_VAAPI
                {AV_PIX_FMT_VAAPI, vaapi_init}
#endif
        };

        if (hwaccel){
                struct state_libavcodec_decompress *state = (struct state_libavcodec_decompress *) s->opaque; 
                for(const enum AVPixelFormat *it = fmt; *it != AV_PIX_FMT_NONE; it++){
                        for(unsigned i = 0; i < sizeof(accels) / sizeof(accels[0]); i++){
                                if(*it == accels[i].pix_fmt){
                                        int ret = accels[i].init_func(s, &state->hwaccel, state->out_codec);
                                        if(ret < 0){
                                                hwaccel_state_reset(&state->hwaccel);
                                                break;
                                        }
                                        if (state->out_codec != VIDEO_CODEC_NONE) { // not probing internal format
                                                return accels[i].pix_fmt;
                                        }
                                }
                        }
                }
                log_msg(LOG_LEVEL_WARNING, "[lavd] Falling back to software decoding!\n");
                if (state->out_codec == HW_VDPAU) {
                        state->blacklist_vdpau = true;
                        return AV_PIX_FMT_NONE;
                }
        }
#endif

        bool use_native[] = { true, false }; // try native first

        for (const bool *use_native_it = use_native; use_native_it !=
                        use_native + sizeof use_native / sizeof use_native[0]; ++use_native_it) {
                for (const enum AVPixelFormat *fmt_it = fmt; *fmt_it != AV_PIX_FMT_NONE; fmt_it++) {
                        //If hwaccel is not enabled skip hw accel pixfmts even if there
                        //are convert functions
                        const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(*fmt_it);
                        if(!hwaccel && fmt_desc && (fmt_desc->flags & AV_PIX_FMT_FLAG_HWACCEL)){
                                continue;
                        }

                        for (const struct av_to_uv_conversion *c = get_av_to_uv_conversions(); c->uv_codec != VIDEO_CODEC_NONE; c++) { // FFMPEG conversion needed
                                if (c->av_codec != *fmt_it) // this conversion is not valid
                                        continue;
                                if (state->out_codec == VIDEO_CODEC_NONE) { // just probing internal format
                                        if (!*use_native_it || c->native) {
                                                state->internal_codec = c->uv_codec;
                                                return AV_PIX_FMT_NONE;
                                        }
                                } else {
                                        if (state->out_codec == c->uv_codec) { // conversion found
                                                state->internal_codec = c->uv_codec; // same as out_codec
                                                return *fmt_it;
                                        }
                                }
                        }
                }
        }

        return AV_PIX_FMT_NONE;
}


/**
 * Changes pixel format from frame to native
 *
 * @todo             figure out color space transformations - eg. JPEG returns full-scale YUV.
 *                   And not in the ITU-T Rec. 701 (eventually Rec. 609) scale.
 * @param  frame     video frame returned from libavcodec decompress
 * @param  dst       destination buffer where data will be stored
 * @param  av_codec  libav pixel format
 * @param  out_codec requested output codec
 * @param  width     frame width
 * @param  height    frame height
 * @retval TRUE      if the transformation was successful
 * @retval FALSE     if transformation failed
 * @see    yuvj422p_to_yuv422
 * @see    yuv420p_to_yuv422
 */
static int change_pixfmt(AVFrame *frame, unsigned char *dst, int av_codec,
                codec_t out_codec, int width, int height, int pitch, int rgb_shift[static restrict 3]) {
        av_to_uv_convert_p convert = NULL;
        for (const struct av_to_uv_conversion *c = get_av_to_uv_conversions(); c->uv_codec != VIDEO_CODEC_NONE; c++) {
                if (c->av_codec == av_codec && c->uv_codec == out_codec) {
                        convert = c->convert;
                }
        }

        if (convert) {
                convert((char *) dst, frame, width, height, pitch, rgb_shift);
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported pixel "
                                "format: %s (id %d)\n",
                                av_get_pix_fmt_name(
                                        av_codec), av_codec);
                return FALSE;
        }

        return TRUE;
}

static void error_callback(void *ptr, int level, const char *fmt, va_list vl) {
        if(strcmp("unset current_picture_ptr on %d. slice\n", fmt) == 0)
                broken_h264_mt_decoding = true;
        av_log_default_callback(ptr, level, fmt, vl);
}

static decompress_status libavcodec_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, codec_t *internal_codec)
{
        struct state_libavcodec_decompress *s = (struct state_libavcodec_decompress *) state;
        int got_frame = 0;
        decompress_status res = DECODER_NO_FRAME;

        if (libav_codec_has_extradata(s->desc.color_spec)) {
                int extradata_size = *(uint32_t *)(void *) src;
                if (s->codec_ctx == NULL) {
                        configure_with(s, s->desc, src + sizeof(uint32_t), extradata_size);
                }
                src += extradata_size + sizeof(uint32_t);
                src_len -= extradata_size + sizeof(uint32_t);
        }

        s->pkt.size = src_len;
        s->pkt.data = src;

        while (s->pkt.size > 0) {
                int len;
                struct timeval t0, t1;
                gettimeofday(&t0, NULL);
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 37, 100)
                len = avcodec_decode_video2(s->codec_ctx, s->frame, &got_frame, &s->pkt);
#else
                if (got_frame) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Decoded frame while compressed data left!\n");
                }
                got_frame = 0;
                int ret = avcodec_send_packet(s->codec_ctx, &s->pkt);
                if (ret == 0 || ret == AVERROR(EAGAIN)) {
                        ret = avcodec_receive_frame(s->codec_ctx, s->frame);
                        if (ret == 0) {
                                got_frame = 1;
                        }
                }
                if (ret != 0) {
                        print_decoder_error(MOD_NAME, ret);
                }
                len = s->pkt.size;
#endif
                gettimeofday(&t1, NULL);

                /*
                 * Hack: Some libavcodec versions (typically found in Libav)
                 * do not correctly support JPEG with more than one reset
                 * segment (GPUJPEG) or more than one slices (compressed with
                 * libavcodec). It returns error although it is actually able
                 * to decompress the frame correctly. So we assume that the
                 * decompression went good even with the reported error.
                 */
                if (len < 0) {
                        if (s->desc.color_spec == JPEG) {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Perhaps JPEG restart interval >0 set? (Not supported by lavd, try '-c JPEG:90:0' on sender).\n");
                        } else if (s->desc.color_spec == MJPG) {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Perhaps old libavcodec without slices support? (Try '-c libavcodec:codec=MJPEG:threads=no' on sender).\n");
#if LIBAVCODEC_VERSION_MAJOR <= 54 // Libav with libavcodec 54 will crash otherwise
                                return DECODER_NO_FRAME;
#endif
                        } else {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Error while decoding frame.\n");
                                return DECODER_NO_FRAME;
                        }
                }

                if(got_frame) {
                        log_msg(LOG_LEVEL_DEBUG, "[lavd] Decompressing %c frame took %f sec.\n", av_get_picture_type_char(s->frame->pict_type), tv_diff(t1, t0));

                        s->frame->opaque = callbacks;
                        /* Skip the frame if this is not an I-frame
                         * and we have missed some of previous frames for VP8 because the
                         * decoder makes ugly artifacts. We rather wait for next I-frame. */
                        if (s->desc.color_spec == VP8 &&
                                        (s->frame->pict_type != AV_PICTURE_TYPE_I &&
                                         (!s->last_frame_seq_initialized || (s->last_frame_seq + 1) % ((1<<22) - 1) != frame_seq))) {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Missing appropriate I-frame "
                                                "(last valid %d, this %u).\n",
                                                s->last_frame_seq_initialized ?
                                                s->last_frame_seq : -1, (unsigned) frame_seq);
                                res = DECODER_NO_FRAME;
                        } else {
#ifdef HWACC_COMMON
                                if(s->hwaccel.copy){
                                        transfer_frame(&s->hwaccel, s->frame);
                                }
#endif
                                if (s->out_codec != VIDEO_CODEC_NONE) {
                                        bool ret = change_pixfmt(s->frame, dst, s->frame->format,
                                                        s->out_codec, s->desc.width, s->desc.height, s->pitch, s->rgb_shift);
                                        if(ret == TRUE) {
                                                s->last_frame_seq_initialized = true;
                                                s->last_frame_seq = frame_seq;
                                                res = DECODER_GOT_FRAME;
                                        } else {
                                                res = DECODER_CANT_DECODE;
                                        }
                                } else {
                                        res = DECODER_GOT_FRAME;
                                }
                        }
                }

                if (len <= 0) {
                        break;
                }

                if(s->pkt.data) {
                        s->pkt.size -= len;
                        s->pkt.data += len;
                }
        }

        if(broken_h264_mt_decoding) {
                if(!s->broken_h264_mt_decoding_workaroud_active) {
                        libavcodec_decompress_reconfigure(s, s->saved_desc,
                                        s->rgb_shift[R], s->rgb_shift[G], s->rgb_shift[B],
                                        s->pitch, s->out_codec);
                }
                if(s->broken_h264_mt_decoding_workaroud_warning_displayed++ % 1000 == 0)
                        av_log(NULL, AV_LOG_WARNING, "Broken multi-threaded decoder detected, "
                                        "switching to a single-threaded one! Consider upgrading your Libavcodec.\n");
        }

        if (s->out_codec == VIDEO_CODEC_NONE && s->internal_codec != VIDEO_CODEC_NONE) {
                *internal_codec = s->internal_codec;
                return DECODER_GOT_CODEC;
        }

        // codec doesn't call get_format_callback (J2K, 10-bit RGB HEVC)
        if (s->out_codec == VIDEO_CODEC_NONE && res == DECODER_GOT_FRAME) {
                if (has_conversion(s->codec_ctx->pix_fmt, internal_codec)) {
                        s->internal_codec = *internal_codec;
                        return DECODER_GOT_CODEC;
                }
                return DECODER_CANT_DECODE;
        }

        if (s->blacklist_vdpau) {
                assert(s->out_codec == HW_VDPAU);
                s->blacklist_vdpau = false;
                return DECODER_CANT_DECODE;
        }

#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
        if (res == DECODER_GOT_FRAME && avcodec_receive_frame(s->codec_ctx, s->frame) != AVERROR(EAGAIN)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Multiple frames decoded at once!\n");
        }
#endif

        return res;
}

ADD_TO_PARAM(lavd_accept_corrputed, "lavd-accept-corrupted",
                "* lavd-accept-corrupted[=no]\n"
                "  Pass corrupted frames to decoder. If decoder isn't error-resilient,\n"
                "  may crash! Use \"no\" to disable even if enabled by default.\n");
static int libavcodec_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_libavcodec_decompress *s =
                (struct state_libavcodec_decompress *) state;
        UNUSED(s);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if (*len < sizeof(int)) {
                                return FALSE;
                        }
                        *(int *) val = FALSE;
                        if (s->codec_ctx && strcmp(s->codec_ctx->codec->name, "h264") == 0) {
                                *(int *) val = TRUE;
                        }
                        if (get_commandline_param("lavd-accept-corrupted")) {
                                *(int *) val =
                                        strcmp(get_commandline_param("lavd-accept-corrupted"), "no") != 0;
                        }

                        *len = sizeof(int);
                        ret = TRUE;
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

static void libavcodec_decompress_done(void *state)
{
        struct state_libavcodec_decompress *s =
                (struct state_libavcodec_decompress *) state;

        deconfigure(s);

        rm_release_shared_lock(LAVCD_LOCK_NAME);

        free(s);
}

static const codec_t supp_codecs[] = { H264, H265, JPEG, MJPG, J2K, J2KR, VP8, VP9,
        HFYU, FFV1, AV1 };
static const struct decode_from_to dec_template[] = {
        { VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, 80 }, // for probe
        { VIDEO_CODEC_NONE, RGB, RGB, 500 },
        { VIDEO_CODEC_NONE, RGB, RGBA, 500 },
        { VIDEO_CODEC_NONE, R10k, R10k, 500 },
        { VIDEO_CODEC_NONE, R10k, RGB, 500 },
        { VIDEO_CODEC_NONE, R10k, RGBA, 500 },
        { VIDEO_CODEC_NONE, R12L, R12L, 500 },
        { VIDEO_CODEC_NONE, R12L, RGB, 500 },
        { VIDEO_CODEC_NONE, R12L, RGBA, 500 },
        { VIDEO_CODEC_NONE, RG48, RGBA, 500 },
        { VIDEO_CODEC_NONE, RG48, R12L, 500 },
        //{ VIDEO_CODEC_NONE, UYVY, RGB, 500 }, // there are conversions but don't enable now
        { VIDEO_CODEC_NONE, UYVY, UYVY, 500 },
        { VIDEO_CODEC_NONE, v210, v210, 500 },
        { VIDEO_CODEC_NONE, v210, UYVY, 500 },
        { VIDEO_CODEC_NONE, VIDEO_CODEC_NONE, UYVY, 900 }, // provide also generic decoder
};
#define SUPP_CODECS_CNT (sizeof supp_codecs / sizeof supp_codecs[0])
#define DEC_TEMPLATE_CNT (sizeof dec_template / sizeof dec_template[0])
/// @todo to remove
ADD_TO_PARAM(lavd_use_10bit, "lavd-use-10bit",
                "* lavd-use-10bit\n"
                "  Do not use, use \"--param decoder-use-codec=v210\" instead.\n");
ADD_TO_PARAM(lavd_use_codec, "lavd-use-codec",
                "* lavd-use-codec=<codec>\n"
                "  Do not use, use \"--param decoder-use-codec=<codec>\" instead.\n");
static const struct decode_from_to *libavcodec_decompress_get_decoders() {

        static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
        static struct decode_from_to ret[SUPP_CODECS_CNT * DEC_TEMPLATE_CNT + 1 /* terminating zero */ + 10 /* place for additional decoders, see below */];

        pthread_mutex_lock(&lock); // prevent concurent initialization
        if (ret[0].from != VIDEO_CODEC_NONE) { // already initialized
                pthread_mutex_unlock(&lock); // prevent concurent initialization
                return ret;
        }

        codec_t force_codec = VIDEO_CODEC_NONE;
        if (get_commandline_param("lavd-use-10bit") || get_commandline_param("lavd-use-codec")) {
                log_msg(LOG_LEVEL_WARNING, "DEPRECATED: Do not use \"--param lavd-use-10bit|lavd-use-codec\", "
                                "use \"--param decoder-use-codec=v210|<codec>\" instead.\n");
                force_codec = get_commandline_param("lavd-use-10bit") ? v210
                        : get_codec_from_name(get_commandline_param("lavd-use-codec"));
        }

        unsigned int ret_idx = 0;
        for (size_t t = 0; t < DEC_TEMPLATE_CNT; ++t) {
                for (size_t c = 0; c < SUPP_CODECS_CNT; ++c) {
                        if (force_codec && force_codec != supp_codecs[c]) {
                                continue;
                        }
                        ret[ret_idx++] = (struct decode_from_to){supp_codecs[c],
                                dec_template[t].internal, dec_template[t].to,
                                dec_template[t].priority};
                }
        }

        if (get_commandline_param("use-hw-accel")) {
                ret[ret_idx++] =
                        (struct decode_from_to) {H264, VIDEO_CODEC_NONE, HW_VDPAU, 200};
        }
        assert(ret_idx < sizeof ret / sizeof ret[0]); // there needs to be at least one zero row

        pthread_mutex_unlock(&lock); // prevent concurent initialization

        return ret;
}

static const struct video_decompress_info libavcodec_info = {
        libavcodec_decompress_init,
        libavcodec_decompress_reconfigure,
        libavcodec_decompress,
        libavcodec_decompress_get_property,
        libavcodec_decompress_done,
        libavcodec_decompress_get_decoders,
};

REGISTER_MODULE(libavcodec, &libavcodec_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

