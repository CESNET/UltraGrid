/**
 * @file   video_decompress/libavcodec.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2023 CESNET, z. s. p. o.
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
#include "libavcodec/lavc_common.h"
#include "libavcodec/lavc_video.h"
#include "libavcodec/from_lavc_vid_conv.h"
#include "lib_common.h"
#include "tv.h"
#include "rtp/rtpdec_h264.h"
#include "rtp/rtpenc_h264.h"
#include "utils/misc.h" // get_cpu_core_count()
#include "utils/worker.h"
#include "video.h"
#include "video_decompress.h"

#ifdef HAVE_SWSCALE
#include <libswscale/swscale.h>
#endif // defined HAVE_SWSCALE
#include <limits.h>

#include "hwaccel_libav_common.h"
#include "hwaccel_vaapi.h"
#include "hwaccel_vdpau.h"
#include "hwaccel_videotoolbox.h"

#define MOD_NAME "[lavd] "

struct state_libavcodec_decompress {
        AVCodecContext  *codec_ctx;
        AVFrame         *frame;
        AVPacket        *pkt;

        struct video_desc desc;
        int              pitch;
        int              rgb_shift[3];
        int              max_compressed_len;
        codec_t          out_codec;
        struct {
                av_to_uv_convert_t convert;
                enum AVPixelFormat convert_in;
        };
        bool             block_accel[HWACCEL_COUNT];
        long long        consecutive_failed_decodes;

        unsigned         last_frame_seq:22; // This gives last sucessfully decoded frame seq number. It is the buffer number from the packet format header, uses 22 bits.
        bool             last_frame_seq_initialized;

        struct state_libavcodec_decompress_sws {
                int width, height;
                enum AVPixelFormat in_codec, out_codec;
                struct SwsContext *ctx;
                AVFrame *frame;
        } sws;

        struct hw_accel_state hwaccel;

        _Bool sps_vps_found; ///< to avoid initial error flood, start decoding after SPS (H.264) or VPS (HEVC) was received
        double mov_avg_comp_duration;
        long mov_avg_frames;
};

static enum AVPixelFormat get_format_callback(struct AVCodecContext *s, const enum AVPixelFormat *fmt);

static void deconfigure(struct state_libavcodec_decompress *s)
{
        if(s->codec_ctx) {
                lavd_flush(s->codec_ctx);
                avcodec_free_context(&s->codec_ctx);
        }
        av_frame_free(&s->frame);
        av_packet_free(&s->pkt);

        hwaccel_state_reset(&s->hwaccel);

        s->convert_in = AV_PIX_FMT_NONE;

#ifdef HAVE_SWSCALE
        sws_freeContext(s->sws.ctx);
        s->sws.ctx = NULL;
        if (s->sws.frame) {
                av_freep(s->sws.frame->data);
        }
        av_frame_free(&s->sws.frame);
#endif // defined HAVE_SWSCALE
}

static int check_av_opt_set(void *state, const char *key, const char *val) {
        int ret = av_opt_set(state, key, val, 0);
        if (ret != 0) {
                printf_libav_error(LOG_LEVEL_WARNING, ret, MOD_NAME "Unable to set %s to %s", key, val);
        }
        return ret;
}

ADD_TO_PARAM("lavd-thread-count", "* lavd-thread-count=<thread_count>[F][S][n][d]\n"
                "  Use <thread_count> decoding threads (0 is usually auto).\n"
                "  Flag 'F' enables frame parallelism (disabled by default), 'S' slice based, can be both (default slice), 'n' for none; 'd' - disable low delay\n");
static void set_codec_context_params(struct state_libavcodec_decompress *s)
{
        int thread_count = 0; ///< decoder may use <cpu_count> frame threads with AV_CODEC_CAP_OTHER_THREADS (latency)
        int req_thread_type = 0;
        bool req_low_delay = true;
        const char *thread_count_opt = get_commandline_param("lavd-thread-count");
        if (thread_count_opt != NULL) {
                char *endptr = NULL;
                errno = 0;
                long val = strtol(thread_count_opt, &endptr, 0);
                if (errno == 0 && thread_count_opt[0] != '\0' && val >= 0 && val <= INT_MAX && (*endptr == '\0' || toupper(*endptr) == 'F' || toupper(*endptr) == 'S')) {
                        thread_count = val;
                        while (*endptr) {
                                switch (toupper(*endptr)) {
                                        case 'F': req_thread_type |= FF_THREAD_FRAME; break;
                                        case 'S': req_thread_type |= FF_THREAD_SLICE; break;
                                        case 'n': req_thread_type = -1; break;
                                        case 'd': req_low_delay = false; break;
                                }
                                endptr++;
                        }
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Wrong value for thread count: %s\n", thread_count_opt);
                }
        }

        s->codec_ctx->thread_count = thread_count; // zero should mean count equal to the number of virtual cores
        s->codec_ctx->thread_type = 0;
        if (req_thread_type == 0) {
                req_thread_type = FF_THREAD_SLICE;
        } if (req_thread_type == -1) {
                req_thread_type = 0;
        }
        if ((req_thread_type & FF_THREAD_FRAME) != 0U) {
                if (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) {
                        s->codec_ctx->thread_type |= FF_THREAD_FRAME;
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: Codec doesn't support frame-based multithreading but requested.\n");
                }
        }
        if ((req_thread_type & FF_THREAD_SLICE) != 0U) {
                if (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) {
                        s->codec_ctx->thread_type |= FF_THREAD_SLICE;
                } else {
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Warning: Codec doesn't support slice-based multithreading.\n");
                }
        }
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Setting thread count to %d, type: %s\n", s->codec_ctx->thread_count, lavc_thread_type_to_str(s->codec_ctx->thread_type));

        s->codec_ctx->flags |= req_low_delay ? AV_CODEC_FLAG_LOW_DELAY : 0;
        s->codec_ctx->flags2 |= AV_CODEC_FLAG2_FAST;
        // set by decoder
        s->codec_ctx->pix_fmt = AV_PIX_FMT_NONE;
        // callback to negotiate pixel format that is supported by UG
        s->codec_ctx->get_format = get_format_callback;
        s->codec_ctx->opaque = s;

        if (strstr(s->codec_ctx->codec->name, "cuvid") != NULL) {
                char gpu[3];
                snprintf(gpu, sizeof gpu, "%u", cuda_devices[0]);
                check_av_opt_set(s->codec_ctx->priv_data, "gpu", gpu);
        }
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
        { PRORES_4444, AV_CODEC_ID_PRORES, NULL, { NULL } },
        { PRORES_4444_XQ, AV_CODEC_ID_PRORES, NULL, { NULL } },
        { PRORES_422_HQ, AV_CODEC_ID_PRORES, NULL, { NULL } },
        { PRORES_422, AV_CODEC_ID_PRORES, NULL, { NULL } },
        { PRORES_422_PROXY, AV_CODEC_ID_PRORES, NULL, { NULL } },
        { PRORES_422_LT, AV_CODEC_ID_PRORES, NULL, { NULL } },
};

ADD_TO_PARAM("force-lavd-decoder", "* force-lavd-decoder=<decoder>[:<decoder2>...]\n"
                "  Forces specified Libavcodec decoder. If more need to be specified, use colon as a delimiter.\n"
                "  Use '-c libavcodec:help' to see available decoders.\n");

#ifdef HWACC_COMMON_IMPL
ADD_TO_PARAM("use-hw-accel", "* use-hw-accel\n"
                "  Try to use hardware accelerated decoding with lavd (NVDEC/VAAPI/VDPAU/VideoToolbox). \n");
#endif
static bool configure_with(struct state_libavcodec_decompress *s,
                struct video_desc desc, void *extradata, int extradata_size)
{
        const struct decoder_info *dec = NULL;

        s->consecutive_failed_decodes = 0;

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
        const AVCodec *codecs_available[13]; // max num of preferred decoders (10) + user supplied + default one + NULL
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
                        const AVCodec *codec = avcodec_find_decoder_by_name(item);
                        if (codec == NULL) {
                                log_msg(LOG_LEVEL_ERROR, "[lavd] Decoder not found: %s\n", item);
                                exit_uv(1);
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
                const AVCodec *codec = avcodec_find_decoder_by_name(*preferred_decoders_it);
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
                const AVCodec *default_decoder = avcodec_find_decoder(dec->avcodec_id);
                if (default_decoder == NULL) {
                        log_msg(LOG_LEVEL_WARNING, "[lavd] No decoder found for the input codec (libavcodec perhaps compiled without any)!\n"
                                                "Use \"--param decompress=<d> to select a different decoder than libavcodec if there is any eligibe.\n");
                } else {
                        codecs_available[codec_index++] = default_decoder;
                }
        }

        // initialize the codec - use the first decoder initialization of which succeeds
        const AVCodec **codec_it = codecs_available;
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
                if (desc.color_spec > PRORES && desc.color_spec <= PRORES_422_LT) {
                        s->codec_ctx->codec_tag = get_fourcc(desc.color_spec);
                }
                set_codec_context_params(s);
                if (avcodec_open2(s->codec_ctx, *codec_it, NULL) < 0) {
                        avcodec_free_context(&s->codec_ctx);
                        log_msg(LOG_LEVEL_WARNING, "[lavd] Unable to open decoder %s.\n", (*codec_it)->name);
                        codec_it++;
                        continue;
                }
                log_msg(LOG_LEVEL_NOTICE, "[lavd] Using decoder: %s\n", (*codec_it)->name);
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Codec %s capabilities: 0x%08X; using thread type %d, count %d\n",
                                (*codec_it)->name, (*codec_it)->capabilities, s->codec_ctx->thread_type, s->codec_ctx->thread_count);
                break;
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

        s->pkt = av_packet_alloc();

        s->last_frame_seq_initialized = false;

        return true;
}

static void * libavcodec_decompress_init(void)
{
        struct state_libavcodec_decompress *s =
                calloc(1, sizeof(struct state_libavcodec_decompress));

        ug_set_av_logging();

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        /*   register all the codecs (you can also register only the codec
         *         you wish to have smaller code */
        avcodec_register_all();
#endif

        hwaccel_state_init(&s->hwaccel);

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
        for(int i = 0; i < HWACCEL_COUNT; i++){
                s->block_accel[i] = get_commandline_param("use-hw-accel") == NULL;
        }
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

#ifdef HWACC_RPI4
static int rpi4_hwacc_init(struct AVCodecContext *s,
                struct hw_accel_state *state,
                codec_t out_codec)
{
        UNUSED(s), UNUSED(out_codec);
        state->type = HWACCEL_RPI4;
        state->copy = false;
        return 0;
}
#endif


#if defined HWACC_COMMON_IMPL && LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 39, 100)
static int vulkan_init(struct AVCodecContext *s,
                struct hw_accel_state *state,
                codec_t out_codec)
{
        UNUSED(s), UNUSED(out_codec);

        AVBufferRef *device_ref = NULL;
        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_VULKAN, &device_ref);
        if(ret < 0)
                return ret;

        state->tmp_frame = av_frame_alloc();
        if(!state->tmp_frame){
                ret = -1;
                goto fail;
        }

        s->hw_device_ctx = device_ref;
        state->type = HWACCEL_VULKAN;
        state->copy = true;
        return 0;

fail:
        av_frame_free(&state->tmp_frame);
        av_buffer_unref(&device_ref);
        return ret;
}
#endif

#ifdef HWACC_COMMON_IMPL
static int hwacc_cuda_init(struct AVCodecContext *s, struct hw_accel_state *state, codec_t out_codec)
{
        UNUSED(out_codec);

        AVBufferRef *device_ref = NULL;
        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_CUDA, &device_ref);
        if(ret < 0)
                return ret;

        state->tmp_frame = av_frame_alloc();
        if(!state->tmp_frame){
                ret = -1;
                goto fail;
        }

        s->hw_device_ctx = device_ref;
        state->type = HWACCEL_CUDA;
        state->copy = true;

        return 0;

fail:
        av_frame_free(&state->tmp_frame);
        av_buffer_unref(&device_ref);
        return ret;
}
#endif

static enum AVPixelFormat get_format_callback(struct AVCodecContext *s, const enum AVPixelFormat *fmt)
{
#define SELECT_PIXFMT(pixfmt) { log_msg(LOG_LEVEL_INFO, MOD_NAME "Selected pixel format: %s\n", av_get_pix_fmt_name(pixfmt)); return pixfmt; }
        if (log_level >= LOG_LEVEL_VERBOSE) {
                char out[1024] = "[lavd] Available output pixel formats:";
                const enum AVPixelFormat *it = fmt;
                while (*it != AV_PIX_FMT_NONE) {
                        strncat(out, " ", sizeof out - strlen(out) - 1);
                        strncat(out, av_get_pix_fmt_name(*it++), sizeof out - strlen(out) - 1);
                }
                log_msg(LOG_LEVEL_VERBOSE, "%s\n", out);
        }

        struct state_libavcodec_decompress *state = (struct state_libavcodec_decompress *) s->opaque;
        const char *hwaccel = get_commandline_param("use-hw-accel");
#ifdef HWACC_COMMON_IMPL
        hwaccel_state_reset(&state->hwaccel);

        static const struct{
                enum AVPixelFormat pix_fmt;
                enum hw_accel_type accel_type;
                int (*init_func)(AVCodecContext *, struct hw_accel_state *, codec_t);
        } accels[] = {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 39, 100)
                {AV_PIX_FMT_VULKAN, HWACCEL_VULKAN, vulkan_init},
#endif
#ifdef HWACC_VDPAU
                {AV_PIX_FMT_VDPAU, HWACCEL_VDPAU, vdpau_init},
#endif
                {AV_PIX_FMT_CUDA, HWACCEL_CUDA, hwacc_cuda_init},
#ifdef HWACC_VAAPI
                {AV_PIX_FMT_VAAPI, HWACCEL_VAAPI, vaapi_init},
#endif
#ifdef HAVE_MACOSX
                {AV_PIX_FMT_VIDEOTOOLBOX, HWACCEL_VIDEOTOOLBOX, videotoolbox_init},
#endif
#ifdef HWACC_RPI4
                {AV_PIX_FMT_RPI4_8, HWACCEL_RPI4, rpi4_hwacc_init},
#endif
                {AV_PIX_FMT_NONE, HWACCEL_NONE, NULL}
        };

        if (hwaccel && state->out_codec != VIDEO_CODEC_NONE) { // not probing internal format
                struct state_libavcodec_decompress *state = (struct state_libavcodec_decompress *) s->opaque; 
                if (!pixfmt_list_has_420_subsampling(fmt)){
                        log_msg(LOG_LEVEL_WARNING, "[lavd] Hw. acceleration requested "
                                        "but incoming video has not 4:2:0 subsampling, "
                                        "which is usually not supported by hw. accelerators\n");
                }
                enum hw_accel_type forced_hwaccel = HWACCEL_NONE;
                if(strlen(hwaccel) > 0){
                        forced_hwaccel = hw_accel_from_str(hwaccel);
                        if(forced_hwaccel == HWACCEL_NONE){
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown accel %s\n", hwaccel);
                                return AV_PIX_FMT_NONE;
                        }
                }
                for(const enum AVPixelFormat *it = fmt; *it != AV_PIX_FMT_NONE; it++){
                        for(unsigned i = 0; i < sizeof(accels) / sizeof(accels[0]); i++){
                                if(*it == accels[i].pix_fmt && !state->block_accel[accels[i].accel_type])
                                {
                                        if(forced_hwaccel != HWACCEL_NONE && accels[i].accel_type != forced_hwaccel){
                                                break;
                                        }
                                        int ret = accels[i].init_func(s, &state->hwaccel, state->out_codec);
                                        if(ret < 0){
                                                hwaccel_state_reset(&state->hwaccel);
                                                break;
                                        }
                                        SELECT_PIXFMT(accels[i].pix_fmt);
                                }
                        }
                }
                if(forced_hwaccel != HWACCEL_NONE){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested hw accel \"%s\" is not available\n", hwaccel);
                        return AV_PIX_FMT_NONE;
                }
                log_msg(LOG_LEVEL_WARNING, "[lavd] Falling back to software decoding!\n");
                if (state->out_codec == HW_VDPAU) {
                        return AV_PIX_FMT_NONE;
                }
        }
#endif

        if (state->out_codec == VIDEO_CODEC_NONE) { // probe
                codec_t c = get_best_ug_codec_to_av(fmt, hwaccel);
                if (c != VIDEO_CODEC_NONE) {
                        enum AVPixelFormat selected_fmt = lavd_get_av_to_ug_codec(fmt, c, hwaccel);
                        return selected_fmt;
                }
        } else {
                enum AVPixelFormat f = lavd_get_av_to_ug_codec(fmt, state->out_codec, hwaccel);
                if (f != AV_PIX_FMT_NONE) {
                        SELECT_PIXFMT(f);
                }
        }

#ifdef HAVE_SWSCALE
        for (const enum AVPixelFormat *fmt_it = fmt; *fmt_it != AV_PIX_FMT_NONE; fmt_it++) {
                const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(*fmt_it);
                if (fmt_desc && (fmt_desc->flags & AV_PIX_FMT_FLAG_HWACCEL) == 0U) {
                        SELECT_PIXFMT(*fmt_it);
                }
        }
#endif

        return AV_PIX_FMT_NONE;
}

#ifdef HAVE_SWSCALE
static bool lavd_sws_convert_reconfigure(struct state_libavcodec_decompress_sws *sws, enum AVPixelFormat sws_in_codec,
                enum AVPixelFormat sws_out_codec, int width, int height)
{
        if (sws->width == width && sws->height == height && sws->in_codec == sws_in_codec && sws->ctx != NULL) {
                return true;
        }
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Using swscale to convert from %s to %s.\n",
                        av_get_pix_fmt_name(sws_in_codec), av_get_pix_fmt_name(sws_out_codec));
        sws_freeContext(sws->ctx);
        if (sws->frame) {
                av_freep(sws->frame->data);
        }
        av_frame_free(&sws->frame);
        sws->ctx = getSwsContext(width, height, sws_in_codec,
                        width, height, sws_out_codec,
                        SWS_POINT);
        if(!sws->ctx){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to init sws context.\n");
                return false;
        }
        sws->frame = av_frame_alloc();
        if (!sws->frame) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not allocate sws frame\n");
                return false;
        }
        sws->frame->width = width;
        sws->frame->height = height;
        sws->frame->format = sws_out_codec;
        int ret = av_image_alloc(sws->frame->data, sws->frame->linesize,
                        sws->frame->width, sws->frame->height,
                        sws_out_codec, 32);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not allocate raw picture buffer for sws\n");
                return false;
        }
        sws->width = width;
        sws->height = height;
        sws->in_codec = sws_in_codec;
        sws->out_codec = sws_out_codec;

        return true;
}

static void lavd_sws_convert(struct state_libavcodec_decompress_sws *sws, AVFrame *in_frame)
{
        sws_scale(sws->ctx,
                        (const uint8_t * const *) in_frame->data,
                        in_frame->linesize,
                        0,
                        in_frame->height,
                        sws->frame->data,
                        sws->frame->linesize);
}

/// @brief Converts directly to out_buffer (instead to sws->frame). This is used for directly mapped
/// UltraGrid pixel formats that can be decoded directly to framebuffer.
static void lavd_sws_convert_to_buffer(struct state_libavcodec_decompress_sws *sws,
                AVFrame *in_frame, char *out_buffer)
{
        struct AVFrame *out = av_frame_alloc();
        codec_t ug_out_pixfmt = get_av_to_ug_pixfmt(sws->frame->format);
        if (codec_is_planar(ug_out_pixfmt)) {
                buf_get_planes(sws->frame->width, sws->frame->height, ug_out_pixfmt, out_buffer, (char **) out->data);
                buf_get_linesizes(sws->frame->width, ug_out_pixfmt, out->linesize);
        } else {
                out->data[0] = (unsigned char *) out_buffer;
                out->linesize[0] = vc_get_linesize(sws->frame->width, ug_out_pixfmt);
        }

        sws_scale(sws->ctx,
                        (const uint8_t * const *) in_frame->data,
                        in_frame->linesize,
                        0,
                        in_frame->height,
                        out->data,
                        out->linesize);
        av_frame_free(&out);
}
#endif

struct convert_task_data {
        const av_to_uv_convert_t *convert;
        unsigned char *out_data;
        AVFrame *in_frame;
        int width;
        int height;
        int pitch;
        const int *rgb_shift;
};

static void *convert_task(void *arg) {
        struct convert_task_data *d = arg;
        av_to_uv_convert(d->convert, (char *) d->out_data, d->in_frame, d->width, d->height, d->pitch, d->rgb_shift);
        return NULL;
}

static void parallel_convert(codec_t out_codec, const av_to_uv_convert_t *convert, char *dst, AVFrame *in, int width, int height, int pitch, int rgb_shift[static restrict 3]) {
        if (codec_is_const_size(out_codec)) { // VAAPI etc
                av_to_uv_convert(convert, dst, in, width, height, pitch, rgb_shift);
                return;
        }

        int cpu_count = get_cpu_core_count();

        struct convert_task_data d[cpu_count];
        AVFrame parts[cpu_count];
        for (int i = 0; i < cpu_count; ++i) {
                int row_height = (height / cpu_count) & ~1; // needs to be even
                unsigned char *part_dst = (unsigned char *) dst + i * row_height * pitch;
                memcpy(parts[i].linesize, in->linesize, sizeof in->linesize);
                const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(in->format);
                for (int plane = 0; plane < AV_NUM_DATA_POINTERS; ++plane) {
                        if (in->data[plane] == NULL) {
                                break;
                        }
                        parts[i].data[plane] = in->data[plane] + ((i * row_height * in->linesize[plane]) >> (plane == 0 ? 0 : fmt_desc->log2_chroma_h));
                }
                if (i == cpu_count - 1) {
                        row_height = height - row_height * (cpu_count - 1);
                }
                d[i] = (struct convert_task_data){convert, part_dst, &parts[i], width, row_height, pitch, rgb_shift};
        }
        task_run_parallel(convert_task, cpu_count, d, sizeof d[0], NULL);
}

static _Bool reconfigure_convert_if_needed(struct state_libavcodec_decompress *s, enum AVPixelFormat av_codec, codec_t out_codec, int width, int height) {
        if (s->convert_in == av_codec) {
                return 1;
        }
        s->convert = get_av_to_uv_conversion(av_codec, out_codec);
        if (s->convert.valid) {
                s->convert_in = av_codec;
                return 1;
        }
#ifdef HAVE_SWSCALE
        if (get_ug_to_av_pixfmt(out_codec) != AV_PIX_FMT_NONE) { // the UG pixfmt can be used directly as dst for sws
                if (!lavd_sws_convert_reconfigure(&s->sws, av_codec, get_ug_to_av_pixfmt(out_codec), width, height)) {
                        return 0;
                }
                s->convert_in = av_codec;
                return 1;
        }

        // else try to find swscale
        enum AVPixelFormat sws_out_codec = pick_av_convertible_to_ug(out_codec, &s->convert);
        if (!sws_out_codec) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported pixel format: %s (id %d)\n",
                                av_get_pix_fmt_name(av_codec), av_codec);
                return 0;
        }
        if (!lavd_sws_convert_reconfigure(&s->sws, av_codec, sws_out_codec, width, height)) {
                return 0;
        }
        s->convert_in = av_codec;
        return 1;
#else
        UNUSED(width), UNUSED(height);
        return 0;
#endif
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
 */
static void change_pixfmt(AVFrame *frame, unsigned char *dst, const av_to_uv_convert_t *convert,
                codec_t out_codec, int width, int height,
                int pitch, int rgb_shift[static restrict 3], struct state_libavcodec_decompress_sws *sws) {
        debug_file_dump("lavd-avframe", serialize_video_avframe, frame);

        if (!sws->ctx) {
                parallel_convert(out_codec, convert, (char *) dst, frame, width, height, pitch, rgb_shift);
                return;
        }

#ifdef HAVE_SWSCALE
        if (get_ug_to_av_pixfmt(out_codec) != AV_PIX_FMT_NONE) { // the UG pixfmt can be used directly as dst for sws
                lavd_sws_convert_to_buffer(sws, frame, (char *) dst);
                return;
        }

        lavd_sws_convert(sws, frame);
        parallel_convert(out_codec, convert, (char *) dst, sws->frame, width, height, pitch, rgb_shift);
#endif // HAVE_SWSCALE
}

/**
 * This function handles beginning of H.264 stream that usually floods terminal
 * output with errors because it usually doesn't start with IDR frame (even if
 * it does, codec probing swallows this). As a workaround, we wait until first
 * SPS NAL unit to avoid initial decoding errors.
 *
 * A drawback may be that it can in theory happen that the SPS NAL unit is not
 * at the beginning of the buffer, but it is not the case of libx264 and
 * hopefully neither other decoders (if so, it needs to be reworked/removed).
 */
static _Bool check_first_sps_vps(struct state_libavcodec_decompress *s, unsigned char *src, unsigned int src_len) {
        if (s->sps_vps_found) {
                return 1;
        }
        _Thread_local static time_ns_t t0;
        if (t0 == 0) {
                t0 = get_time_in_ns();
        }
        if (get_time_in_ns() - t0 > 10 * NS_IN_SEC) { // after 10 seconds surrender and let decoder do the job
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "No SPS found, starting decode, anyway. Please report a bug to " PACKAGE_BUGREPORT " if decoding succeeds from now.\n");
                s->sps_vps_found = 1;
                return 1;
        }

        int nalu_type = 0;
        const unsigned char *nal = src;
        do {
                nal =
                    rtpenc_h264_get_next_nal(nal, src_len - (nal - src), NULL);
                if (!nal) {
                        return 0;
                }
                if (s->desc.color_spec == H264) {
                        nalu_type = NALU_HDR_GET_TYPE(nal[0]);
                } else {
                        nalu_type = nal[0] >> 1;
                }
                debug_msg("Received %s NALU.", get_nalu_name(nalu_type));
        } while (nalu_type == NAL_AUD);

        switch (nalu_type) {
        case NAL_SPS:
        case NAL_HEVC_VPS:
                s->sps_vps_found = 1;
                log_msg(LOG_LEVEL_VERBOSE,
                        MOD_NAME "Received %s NALU, decoding begins.\n",
                        get_nalu_name(nalu_type));
                return 1;
        default:
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "Waiting for first SPS/VPS NALU...\n");
                return 0;
        }
}

/// print hint to improve performance if not making it
static void check_duration(struct state_libavcodec_decompress *s, double duration_total_sec, double duration_pixfmt_change_sec)
{
        const int mov_window = 100;
        if (s->mov_avg_frames >= 10 * mov_window) {
                return;
        }
        s->mov_avg_comp_duration = (s->mov_avg_comp_duration * (mov_window - 1) + duration_total_sec) / mov_window;
        s->mov_avg_frames += 1;
        if (s->mov_avg_frames < 2 * mov_window || s->mov_avg_comp_duration < 1 / s->desc.fps) {
                return;
        }
        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Average decompression time of last %d frames is %f ms but time per frame is only %f ms!\n",
                        mov_window, s->mov_avg_comp_duration * 1000, 1000 / s->desc.fps);
        const char *hint = NULL;
        if ((s->codec_ctx->thread_type & FF_THREAD_SLICE) == 0 && (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) != 0) {
                hint = "\"--param lavd-thread-count=<n>FS\" option with small <n> or 0 (nr of logical cores)";
        } else if (s->codec_ctx->thread_count == 1 && (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_OTHER_THREADS) != 0) {
                hint = "\"--param lavd-thread-count=<n>\" option with small <n> or 0 (nr of logical cores)";
        }
        if (hint) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Consider adding %s to increase throughput at the expense of latency.\n"
                                "You may also try to disable low delay decode using 'd' flag.\n",
                                hint);
        }
        s->mov_avg_frames = LONG_MAX;

        bool in_rgb = av_pix_fmt_desc_get(s->convert_in)->flags & AV_PIX_FMT_FLAG_RGB;
        if (codec_is_a_rgb(s->out_codec) != in_rgb && duration_pixfmt_change_sec > s->mov_avg_comp_duration / 4) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Also pixfmt change of last frame took %f ms.\n"
                        "Consider adding \"--conv-policy cds\" to prevent color space conversion.\n", duration_pixfmt_change_sec / 1000.0);
        }
}

static void handle_lavd_error(struct state_libavcodec_decompress *s, int ret)
{
        print_decoder_error(MOD_NAME, ret);
        if(ret == AVERROR(EIO)){
                s->consecutive_failed_decodes++;
                if(s->consecutive_failed_decodes > 70 && !s->block_accel[s->hwaccel.type]){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Decode failing, "
                                        " blacklisting hw. accelerator...\n");
                        s->block_accel[s->hwaccel.type] = true;
                        deconfigure(s);
                        configure_with(s, s->desc, NULL, 0);
                }
        } else if (ret == AVERROR(EAGAIN) && strcmp(s->codec_ctx->codec->name, "hevc_qsv") == 0) {
                if (s->consecutive_failed_decodes++ == 70) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "hevc_qsv decoder keeps failing, which may be caused by intra refresh period.\n"
                                        "Try disabling intra refresh on encoder: `-c libavcodec:encoder=libx265:disable_intra_refresh`\n");
                }
        }
}

static bool read_forced_pixfmt(codec_t compress, unsigned char *src, unsigned int src_len, struct pixfmt_desc *internal_props) {
        if (compress == H264) {
                char expected_prefix[] = { START_CODE_3B, H264_NAL_SEI_PREFIX, sizeof (unsigned char[]) { UG_ORIG_FORMAT_ISO_IEC_11578_GUID } + 1, UG_ORIG_FORMAT_ISO_IEC_11578_GUID };
                if (src_len < sizeof expected_prefix + 2 || memcmp(src + src_len - sizeof expected_prefix - 2, expected_prefix, sizeof expected_prefix) != 0) {
                        return false;
                }
        } else if (compress == H265) {
                char expected_prefix[] = { START_CODE_3B, HEVC_NAL_SEI_PREFIX, sizeof (unsigned char []) { UG_ORIG_FORMAT_ISO_IEC_11578_GUID } + 1, UG_ORIG_FORMAT_ISO_IEC_11578_GUID };
                if (src_len < sizeof expected_prefix + 2 || memcmp(src + src_len - sizeof expected_prefix - 2, expected_prefix, sizeof expected_prefix) != 0) {
                        return false;
                }
        } else {
                return false;
        }
        unsigned format = src[src_len - 2];
        internal_props->depth = 8 + (format >> 4) * 2;
        int subs_a = ((format >> 2) & 0x3) + 1;
        int subs_b = ((format >> 1) & 0x1) * subs_a;
        internal_props->subsampling = 4000 + subs_a * 100 + subs_b * 10;
        internal_props->rgb = format & 0x1;
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Stream properties read from metadata.\n");
        return true;
}

static decompress_status libavcodec_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, struct pixfmt_desc *internal_props)
{
        struct state_libavcodec_decompress *s = (struct state_libavcodec_decompress *) state;

        if ((s->desc.color_spec == H264 || s->desc.color_spec == H265) && !check_first_sps_vps(s, src, src_len)) {
                return DECODER_NO_FRAME;
        }

        if (libav_codec_has_extradata(s->desc.color_spec)) {
                int extradata_size = *(uint32_t *)(void *) src;
                if (s->codec_ctx == NULL) {
                        configure_with(s, s->desc, src + sizeof(uint32_t), extradata_size);
                }
                src += extradata_size + sizeof(uint32_t);
                src_len -= extradata_size + sizeof(uint32_t);
        }

        s->pkt->size = src_len;
        s->pkt->data = src;

        time_ns_t t0 = get_time_in_ns();

        int ret = avcodec_send_packet(s->codec_ctx, s->pkt);
        if (ret == 0 || ret == AVERROR(EAGAIN)) {
                ret = avcodec_receive_frame(s->codec_ctx, s->frame);
                if (ret == 0) {
                        s->consecutive_failed_decodes = 0;
                }
        }
        if (ret != 0) {
                handle_lavd_error(s, ret);
                return DECODER_NO_FRAME;
        }
        time_ns_t t1 = get_time_in_ns();

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
                        return DECODER_NO_FRAME;
        }
#ifdef HWACC_COMMON_IMPL
        if(s->hwaccel.copy){
                transfer_frame(&s->hwaccel, s->frame);
        }
#endif
        if (s->out_codec != VIDEO_CODEC_NONE) {
                if (!reconfigure_convert_if_needed(s, s->frame->format, s->out_codec, s->desc.width, s->desc.height)) {
                        return DECODER_UNSUPP_PIXFMT;
                }
                change_pixfmt(s->frame, dst, &s->convert, s->out_codec, s->desc.width,
                              s->desc.height, s->pitch, s->rgb_shift, &s->sws);
                s->last_frame_seq_initialized = true;
                s->last_frame_seq = frame_seq;
        }
        time_ns_t t2 = get_time_in_ns();
        log_msg(LOG_LEVEL_DEBUG, MOD_NAME "Decompressing %c frame took %f ms, pixfmt change %f ms.\n", av_get_picture_type_char(s->frame->pict_type),
                (t1 - t0) / NS_IN_MS_DBL, (t2 - t1) / NS_IN_MS_DBL);
        check_duration(s, (t2 - t0) / NS_IN_SEC_DBL, (t2 - t1) / NS_IN_MS_DBL);

        if (s->out_codec == VIDEO_CODEC_NONE) {
                log_msg(LOG_LEVEL_DEBUG, MOD_NAME "Selected output pixel format: %s\n", av_get_pix_fmt_name(s->codec_ctx->pix_fmt));
                if (!read_forced_pixfmt(s->desc.color_spec, src, src_len, internal_props)) {
                        *internal_props = av_pixfmt_get_desc(s->codec_ctx->pix_fmt);
                }
                return DECODER_GOT_CODEC;
        }

        if (avcodec_receive_frame(s->codec_ctx, s->frame) != AVERROR(EAGAIN)) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Multiple frames decoded at once!\n");
        }

        return DECODER_GOT_FRAME;
}

ADD_TO_PARAM("lavd-accept-corrupted",
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

        free(s);
}

/**
 * @todo
 * This should be take into account existing conversions.
 */
static int libavcodec_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc) {
        if (get_commandline_param("use-hw-accel") &&
                        (((compression == H264 || compression == H265) && ugc == HW_VDPAU) ||
                         (compression == H265 && ugc == RPI4_8))) {
                return 200;
        }

        unsigned i = 0;
        for ( ; i < sizeof decoders / sizeof decoders[0]; ++i) {
                if (decoders[i].ug_codec == compression) {
                        break;
                }
        }
        if (i == sizeof decoders / sizeof decoders[0]) { // lavd doesn't handle this compression
                return -1;
        }

        switch (ugc) {
                case VIDEO_CODEC_NONE:
                        return 80; // for probe
                case UYVY:
                case RG48:
                case RGB:
                case RGBA:
                case R10k:
                case R12L:
                case v210:
                case Y416:
                        break;
                default:
                        return -1;
        }
        if (internal.depth == 0) { // unspecified internal format
                return 900;
        }
        return codec_is_a_rgb(ugc) == internal.rgb ? 500 : 800;
}

static const struct video_decompress_info libavcodec_info = {
        libavcodec_decompress_init,
        libavcodec_decompress_reconfigure,
        libavcodec_decompress,
        libavcodec_decompress_get_property,
        libavcodec_decompress_done,
        libavcodec_decompress_get_priority,
};

REGISTER_MODULE(libavcodec, &libavcodec_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

