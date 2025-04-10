/**
 * @file   video_decompress/libavcodec.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2024 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H

#include <assert.h>
#ifdef HAVE_SWSCALE
#include <libswscale/swscale.h>
#endif // defined HAVE_SWSCALE
#include <limits.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "libavcodec/from_lavc_vid_conv.h"
#include "libavcodec/lavc_common.h"
#include "libavcodec/lavc_video.h"
#include "rtp/rtpdec_h264.h"
#include "rtp/rtpenc_h264.h"
#include "tv.h"
#include "utils/debug.h"          // for debug_file_dump
#include "utils/macros.h"
#include "video.h"
#include "video_codec.h"
#include "video_decompress.h"

#include "hwaccel_libav_common.h"
#include "hwaccel_vaapi.h"
#include "hwaccel_vdpau.h"
#include "hwaccel_videotoolbox.h"

#define MOD_NAME "[lavd] "

struct state_libavcodec_decompress {
        AVCodecContext *codec_ctx;
        AVFrame        *frame;
        AVFrame        *tmp_frame;
        AVPacket       *pkt;

        struct video_desc desc;
        int              pitch;
        int              rgb_shift[3];
        int              max_compressed_len;
        codec_t          out_codec;
        struct {
                av_to_uv_convert_t *convert;
                enum AVPixelFormat convert_in;
        };
        bool             block_accel[HWACCEL_COUNT];
        long long        consecutive_failed_decodes;

        struct state_libavcodec_decompress_sws {
                int width, height;
                enum AVPixelFormat in_codec, out_codec;
                struct SwsContext *ctx;
                AVFrame *frame;
        } sws;

        struct hw_accel_state hwaccel;

        _Bool sps_vps_found; ///< to avoid initial error flood, start decoding after SPS (H.264) or VPS (HEVC) was received

        double    mov_avg_comp_duration;
        long long mov_avg_frames;
        time_ns_t duration_warn_last_print;
};

static enum AVPixelFormat get_format_callback(struct AVCodecContext *s, const enum AVPixelFormat *fmt);

static void deconfigure(struct state_libavcodec_decompress *s)
{
        av_to_uv_conversion_destroy(&s->convert);

        if(s->codec_ctx) {
                lavd_flush(s->codec_ctx);
                avcodec_free_context(&s->codec_ctx);
        }
        av_frame_free(&s->frame);
        av_frame_free(&s->tmp_frame);
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

/// @todo 'd' flag is not entirely thread-specific - maybe new param or rename
/// this to "lavd-opts" or so?
ADD_TO_PARAM("lavd-thread-count",
             "* lavd-thread-count=[<thread_count>][F][S][n][d][D]\n"
             "  Use <thread_count> decoding threads (0 is usually auto).\n"
             "  Flag 'F' enables frame parallelism (disabled by default), 'S' "
             "slice based, can be both (default slice), 'n' for none; 'd' - "
             "disable low delay\n"
             "  'D' - don't set anything (keep codec defaults)\n");
static void
set_thread_count(struct state_libavcodec_decompress *s, bool *req_low_delay)
{
        int thread_count = 0; ///< decoder may use <cpu_count> frame threads with AV_CODEC_CAP_OTHER_THREADS (latency)
        int req_thread_type = 0;
        const char *thread_count_opt = get_commandline_param("lavd-thread-count");
        if (thread_count_opt != NULL) {
                char *endptr = NULL;
                errno = 0;
                long val = strtol(thread_count_opt, &endptr, 0);
                if (errno == 0 && val >= 0 && val <= INT_MAX) {
                        thread_count = (int) val;
                }
                while (*endptr) {
                        switch (toupper(*endptr)) {
                                case 'D': thread_count = -1; break;
                                case 'F': req_thread_type |= FF_THREAD_FRAME; break;
                                case 'S': req_thread_type |= FF_THREAD_SLICE; break;
                                case 'n': req_thread_type = -1; break;
                                case 'd': *req_low_delay = false; break;
                                default: errno = EINVAL; break;
                        }
                        endptr++;
                }
                if (errno != 0) {
                        MSG(ERROR, "Wrong value for thread count value: %s\n",
                            thread_count_opt);
                        handle_error(EXIT_FAIL_USAGE);
                }
        }
        if (thread_count == -1) {
                return;
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
}

static void
set_codec_context_params(struct state_libavcodec_decompress *s)
{
        bool req_low_delay = true;
        set_thread_count(s, &req_low_delay);

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

enum {
        MAX_PREFERED = 10,
        MAX_DECODERS = MAX_PREFERED + 1 + 1, ///< + user forced + default
        DEC_LEN      = MAX_DECODERS + 1,     ///< + terminating NULL
};

struct decoder_info {
        codec_t ug_codec;
        enum AVCodecID avcodec_id;
        // Note:
        // Make sure that if adding hw decoders to prefered_decoders[] that
        // that decoder fails if there is not the HW during init, not while decoding
        // frames (like vdpau does). Otherwise, such a decoder would be initialized
        // but no frame decoded then.
        // Note 2:
        // cuvid decoders cannot be currently used as the default ones because they
        // currently support only 4:2:0 subsampling and fail during decoding if other
        // subsampling is given.
        const char *preferred_decoders[MAX_PREFERED + 1]; // must be NULL-terminated
};

static const struct decoder_info decoders[] = {
        { H264,             AV_CODEC_ID_H264,     { NULL /* "h264_cuvid" */ } },
        { H265,             AV_CODEC_ID_HEVC,     { NULL /* "hevc_cuvid" */ } },
        { JPEG,             AV_CODEC_ID_MJPEG,    { NULL }                    },
        { J2K,              AV_CODEC_ID_JPEG2000, { NULL }                    },
        { J2KR,             AV_CODEC_ID_JPEG2000, { NULL }                    },
        { VP8,              AV_CODEC_ID_VP8,      { NULL }                    },
        { VP9,              AV_CODEC_ID_VP9,      { NULL }                    },
        { HFYU,             AV_CODEC_ID_HUFFYUV,  { NULL }                    },
        { FFV1,             AV_CODEC_ID_FFV1,     { NULL }                    },
        { AV1,              AV_CODEC_ID_AV1,      { "libdav1d" }              },
        { PRORES_4444,      AV_CODEC_ID_PRORES,   { NULL }                    },
        { PRORES_4444_XQ,   AV_CODEC_ID_PRORES,   { NULL }                    },
        { PRORES_422_HQ,    AV_CODEC_ID_PRORES,   { NULL }                    },
        { PRORES_422,       AV_CODEC_ID_PRORES,   { NULL }                    },
        { PRORES_422_PROXY, AV_CODEC_ID_PRORES,   { NULL }                    },
        { PRORES_422_LT,    AV_CODEC_ID_PRORES,   { NULL }                    },
        { CFHD,             AV_CODEC_ID_CFHD,     { NULL }                    }
};

static bool
get_usable_decoders(
    enum AVCodecID    avcodec_id,
    const char *const preferred_decoders[static MAX_PREFERED + 1],
    const AVCodec    *usable_decoders[static DEC_LEN])
{
        unsigned int codec_index = 0;
        // first try codec specified from cmdline if any
        const char *param = get_commandline_param("force-lavd-decoder");
        if (param != NULL) {
                char *val = alloca(strlen(param) + 1);
                strcpy(val, param);
                char *item     = NULL;
                char *save_ptr = NULL;
                while ((item = strtok_r(val, ":", &save_ptr))) {
                        val = NULL;
                        const AVCodec *codec =
                            avcodec_find_decoder_by_name(item);
                        if (codec == NULL) {
                                log_msg(LOG_LEVEL_ERROR,
                                        MOD_NAME
                                        "Forced decoder not found: %s\n",
                                        item);
                                exit_uv(1);
                                return false;
                        }
                        if (codec->id != avcodec_id) {
                                log_msg(
                                    LOG_LEVEL_WARNING,
                                    MOD_NAME
                                    "Forced decoder not valid for codec: %s\n",
                                    item);
                                continue;
                        }
                        assert(codec_index < MAX_DECODERS);
                        usable_decoders[codec_index++] = codec;
                }
        }
        // then try preferred codecs
        const char *const *preferred_decoders_it = preferred_decoders;
        while (*preferred_decoders_it) {
                const AVCodec *codec =
                    avcodec_find_decoder_by_name(*preferred_decoders_it);
                if (codec == NULL) {
                        log_msg(LOG_LEVEL_VERBOSE,
                                "[lavd] Decoder not available: %s\n",
                                *preferred_decoders_it);
                        preferred_decoders_it++;
                        continue;
                }
                assert(codec_index < MAX_DECODERS);
                preferred_decoders_it++;
        }
        // finally, add a default one if there are no preferred enc. or all fail
        assert(codec_index < MAX_DECODERS);
        const AVCodec *default_decoder = avcodec_find_decoder(avcodec_id);
        if (default_decoder == NULL) {
                log_msg(LOG_LEVEL_WARNING,
                        "[lavd] No decoder found for the input codec "
                        "(libavcodec perhaps compiled without any)!\n"
                        "Use \"--param force-lavd-decoder=<d> to select a "
                        "different decoder than libavcodec if there is "
                        "any eligibe.\n");
        } else {
                usable_decoders[codec_index++] = default_decoder;
        }
        usable_decoders[codec_index] = NULL;
        return true;
}

ADD_TO_PARAM("force-lavd-decoder", "* force-lavd-decoder=<decoder>[:<decoder2>...]\n"
                "  Forces specified Libavcodec decoder. If more need to be specified, use colon as a delimiter.\n"
                "  Use '-c libavcodec:help' to see available decoders.\n");

ADD_TO_PARAM("use-hw-accel", "* use-hw-accel[=<api>|help]\n"
        "  Try to use hardware accelerated decoding with lavd "
        "(NVDEC/VAAPI/VDPAU/VideoToolbox).\n"
        "  Optionally with enforced API option.\n");
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

        // priority list of decoders that can be used for the codec
        const AVCodec *usable_decoders[DEC_LEN] = { NULL };
        if (!get_usable_decoders(dec->avcodec_id, dec->preferred_decoders,
                                 usable_decoders)) {
                return false;
        }

        // initialize the codec - use the first decoder initialization of which succeeds
        const AVCodec **codec_it = usable_decoders;
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
        s->tmp_frame = av_frame_alloc();
        if (s->frame == NULL || s->tmp_frame == NULL) {
                log_msg(LOG_LEVEL_ERROR, "[lavd] Unable allocate frame.\n");
                return false;
        }

        s->pkt = av_packet_alloc();

        return true;
}

/// @retval false if 1. hwacc not enabled; 2. help; 3. incorect hwacc spec
static bool
validate_hwacc_param(void)
{
        const char *const hwaccel = get_commandline_param("use-hw-accel");
        if (hwaccel == NULL) {
                return true;
        }
#if !defined HWACC_COMMON_IMPL
        MSG(FATAL, "HW acceleration not compiled in!\n");
        exit_uv(1);
        return NULL;
#endif
        if (strlen(hwaccel) == 0 ||
            hw_accel_from_str(hwaccel) != HWACCEL_NONE) {
                return true;
        }
        if (strcmp(hwaccel, "help") == 0) {
                exit_uv(0);
                return false;
        }
        MSG(ERROR, "Wrong HW acceleration specified: %s\n", hwaccel);
        exit_uv(1);
        return false;
}

static void * libavcodec_decompress_init(void)
{
        if (!validate_hwacc_param()) {
                return NULL;
        }

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
        s->rgb_shift[R_SHIFT_IDX] = rshift;
        s->rgb_shift[G_SHIFT_IDX] = gshift;
        s->rgb_shift[B_SHIFT_IDX] = bshift;
        for(int i = 0; i < HWACCEL_COUNT; i++){
                s->block_accel[i] = get_commandline_param("use-hw-accel") == NULL;
        }
        s->out_codec = out_codec;
        s->desc = desc;

        deconfigure(s);
        if (libav_codec_has_extradata(desc.color_spec)) {
                // for codecs that have metadata we have to defer initialization
                // because we don't have the data right now
                return true;
        }
        return configure_with(s, desc, NULL, 0);
}

#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(55, 75, 100)
static int drm_prime_init(struct AVCodecContext *s,
                struct hw_accel_state *state,
                codec_t out_codec)
{
        UNUSED(s), UNUSED(out_codec);
        state->type = HWACCEL_DRM_PRIME;
        AVBufferRef *ref = NULL;
        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_DRM, &ref);
        if(ret < 0)
                return ret;

        s->hw_device_ctx = ref;
        state->copy = out_codec != DRM_PRIME;
        state->tmp_frame = av_frame_alloc();
        if(!state->tmp_frame){
                av_buffer_unref(&ref);
                return -1;
        }
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

#ifdef HWACC_COMMON_IMPL
static void
check_pixfmt_hw_eligibility(enum AVPixelFormat sw_pix_fmt)
{
        const char *fmt_name = av_get_pix_fmt_name(sw_pix_fmt);
        const int   depth    = av_pix_fmt_desc_get(sw_pix_fmt)->comp[0].depth;
        if (depth != 8) {
                MSG(WARNING,
                    "HW acceleration requested "
                    "but incoming video has %d bit depth (%s), which is more "
                    "than "
                    "universally supported 8b.\n",
                    depth, fmt_name);
        }
        if (av_pixfmt_get_subsampling(sw_pix_fmt) != SUBS_420) {
                log_msg(LOG_LEVEL_WARNING,
                        "[lavd] Hw. acceleration requested "
                        "but incoming video has not 4:2:0 subsampling (format "
                        "is %s), "
                        "which is usually not supported by hw. accelerators\n",
                        fmt_name);
        }
}
#endif

static enum AVPixelFormat get_format_callback(struct AVCodecContext *s, const enum AVPixelFormat *fmt)
{
#define SELECT_PIXFMT(pixfmt) { log_msg(LOG_LEVEL_INFO, MOD_NAME "Selected pixel format: %s\n", av_get_pix_fmt_name(pixfmt)); return pixfmt; }
        MSG(VERBOSE, "Available output pixel formats: %s\n",
            get_avpixfmts_names(fmt));

        struct state_libavcodec_decompress *state = (struct state_libavcodec_decompress *) s->opaque;
        const char *hwaccel = get_commandline_param("use-hw-accel");
#ifdef HWACC_COMMON_IMPL
        hwaccel_state_reset(&state->hwaccel);

        static const struct{
                enum AVPixelFormat pix_fmt;
                int (*init_func)(AVCodecContext *, struct hw_accel_state *, codec_t);
        } accels[] = {
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 39, 100)
                {AV_PIX_FMT_VULKAN, vulkan_init},
#endif
#ifdef HWACC_VDPAU
                {AV_PIX_FMT_VDPAU, vdpau_init},
#endif
                {AV_PIX_FMT_CUDA, hwacc_cuda_init},
#ifdef HWACC_VAAPI
                {AV_PIX_FMT_VAAPI, vaapi_init},
#endif
#ifdef __APPLE__
                {AV_PIX_FMT_VIDEOTOOLBOX, videotoolbox_init},
#endif
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(55, 75, 100)
                {AV_PIX_FMT_DRM_PRIME, drm_prime_init},
#endif
                {AV_PIX_FMT_NONE, NULL}
        };

        if (hwaccel != NULL) {
                struct state_libavcodec_decompress *state = (struct state_libavcodec_decompress *) s->opaque; 
                check_pixfmt_hw_eligibility(s->sw_pix_fmt);
                const enum hw_accel_type forced_hwaccel =
                    strlen(hwaccel) > 0 ? hw_accel_from_str(hwaccel)
                                        : HWACCEL_NONE;
                for(const enum AVPixelFormat *it = fmt; *it != AV_PIX_FMT_NONE; it++){
                        for(unsigned i = 0; i < sizeof(accels) / sizeof(accels[0]); i++){
                                if (*it != accels[i].pix_fmt ||
                                    state->block_accel[hw_accel_from_pixfmt(accels[i].pix_fmt)]) {
                                        continue;
                                }
                                if (forced_hwaccel != HWACCEL_NONE &&
                                    hw_accel_from_pixfmt(accels[i].pix_fmt) !=
                                        forced_hwaccel) {
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

static _Bool
reconf_internal(struct state_libavcodec_decompress *s,
                     enum AVPixelFormat av_codec, codec_t out_codec, int width,
                     int height)
{
        av_to_uv_conversion_destroy(&s->convert);
        s->convert = get_av_to_uv_conversion(av_codec, out_codec);
        if (s->convert != NULL) {
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

static bool
reconfigure_convert_if_needed(struct state_libavcodec_decompress *s,
                              enum AVPixelFormat av_codec, codec_t out_codec,
                              int width, int height)
{
        assert(av_codec != AV_PIX_FMT_NONE);
        if (s->convert_in == av_codec) { // no reconf needed
                return true;
        }
        MSG(VERBOSE,
            "Codec characteristics: CS=%s, range=%s, primaries=%s, "
            "transfer=%s, chr_loc=%s\n",
            av_color_space_name(s->codec_ctx->colorspace),
            av_color_range_name(s->codec_ctx->color_range),
            av_color_primaries_name(s->codec_ctx->color_primaries),
            av_color_transfer_name(s->codec_ctx->color_trc),
            av_chroma_location_name(s->codec_ctx->chroma_sample_location));
        return reconf_internal(s, av_codec, out_codec, width, height);
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
static void
change_pixfmt(AVFrame *frame, unsigned char *dst, av_to_uv_convert_t *convert,
              codec_t out_codec, int pitch,
              int rgb_shift[static restrict 3],
              struct state_libavcodec_decompress_sws *sws)
{
        debug_file_dump("lavd-avframe", serialize_video_avframe, frame);

        if (!sws->ctx) {
                av_to_uv_convert(convert, (char *) dst, frame,
                                 pitch, rgb_shift);
                return;
        }

#ifdef HAVE_SWSCALE
        if (get_ug_to_av_pixfmt(out_codec) != AV_PIX_FMT_NONE) { // the UG pixfmt can be used directly as dst for sws
                lavd_sws_convert_to_buffer(sws, frame, (char *) dst);
                return;
        }

        lavd_sws_convert(sws, frame);
        av_to_uv_convert(convert, (char *) dst, sws->frame,
                         pitch, rgb_shift);
#else
        (void) out_codec;
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

        const unsigned char *const nal =
            rtpenc_get_first_nal(src, src_len, s->desc.color_spec == H265);
        if (nal == NULL) {
                return 0;
        }
        const bool hevc      = s->desc.color_spec == H265;
        const int  nalu_type = NALU_HDR_GET_TYPE(nal, hevc);

        if (hevc) {
                if (nalu_type > NAL_HEVC_CODED_SLC_FIRST) {
                        s->sps_vps_found = true;
                }
        } else {
                if (nalu_type == NAL_H264_SPS) {
                        s->sps_vps_found = true;
                }
        }
        if (!s->sps_vps_found)  {
                MSG(WARNING, "Got %s, waiting for first IDR NALU...\n",
                    get_nalu_name(nalu_type, hevc));
        } else {
                MSG(VERBOSE, "Got %s, decode will begin...\n",
                    get_nalu_name(nalu_type, hevc));
        }
        return s->sps_vps_found;
}

/// print hint to improve performance if not making it
static void check_duration(struct state_libavcodec_decompress *s, double duration_total_sec, double duration_pixfmt_change_sec)
{
        enum {
                MOV_WIN_FRM        = 100,
                REPEAT_INT_SEC     = 30,
                TIME_SLOT_PERC_MAX = 66
        };
        double tpf = 1 / s->desc.fps;
        tpf *= TIME_SLOT_PERC_MAX / 100.;
        s->mov_avg_comp_duration =
            (s->mov_avg_comp_duration * (MOV_WIN_FRM - 1) +
             duration_total_sec) /
            MOV_WIN_FRM;
        s->mov_avg_frames += 1;
        if (s->mov_avg_frames < 2 * MOV_WIN_FRM ||
            s->mov_avg_comp_duration < tpf) {
                return;
        }
        const time_ns_t now = get_time_in_ns();
        if (now < s->duration_warn_last_print + NS_IN_SEC * REPEAT_INT_SEC) {
                return;
        }
        s->duration_warn_last_print = now;
        MSG(WARNING,
            "Avg decompress time of last %d frames is %.2f ms which exceeds %.2f "
            "ms (%d%% of TPF)!\n",
            MOV_WIN_FRM, s->mov_avg_comp_duration * MS_IN_SEC, tpf * MS_IN_SEC,
            TIME_SLOT_PERC_MAX);
        const char *hint = NULL;
        if ((s->codec_ctx->thread_type & FF_THREAD_FRAME) == 0 &&
            (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) !=
                0) {
                hint = "\"--param lavd-thread-count=<n>FS\" option with small <n> or 0 (nr of logical cores)";
        } else if (s->codec_ctx->thread_count == 1 && (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_OTHER_THREADS) != 0) {
                hint = "\"--param lavd-thread-count=<n>\" option with small <n> or 0 (nr of logical cores)";
        }
        if (hint) {
                MSG(WARNING,
                    "Consider adding %s to increase throughput at the expense "
                    "of latency.\n",
                    hint);
        }
        if ((s->codec_ctx->flags & AV_CODEC_FLAG_LOW_DELAY) != 0) {
                MSG(WARNING,
                    "Consider %sdisabling low delay decode using 'd' flag.\n",
                    hint != NULL ? "also " : "");
        }

        bool in_rgb = av_pix_fmt_desc_get(s->convert_in)->flags & AV_PIX_FMT_FLAG_RGB;
        if (codec_is_a_rgb(s->out_codec) != in_rgb && duration_pixfmt_change_sec > s->mov_avg_comp_duration / 4) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Also pixfmt change of last frame took %f ms.\n"
                        "Consider adding \"--conv-policy cds\" to prevent color space conversion.\n", duration_pixfmt_change_sec / 1000.0);
        }
}

static void
handle_lavd_error(const char *prefix, struct state_libavcodec_decompress *s,
                  int ret)
{
        print_decoder_error(prefix, ret);
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

static bool
read_forced_pixfmt(codec_t compress, const unsigned char *src,
                   unsigned int src_len, struct pixfmt_desc *internal_props)
{
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

static bool
decode_frame(struct state_libavcodec_decompress *s, unsigned char *src,
             int src_len)
{
        bool frame_decoded = false;
        s->pkt->data       = src;
        s->pkt->size       = src_len;
        int ret            = avcodec_send_packet(s->codec_ctx, s->pkt);
        if (ret != 0 && ret != AVERROR(EAGAIN)) {
                handle_lavd_error(MOD_NAME "send - ", s, ret);
                return false;
        }
        // we output to tmp_frame because even if receive fails,
        // it overrides previous potentially valid frame
        while ((ret = avcodec_receive_frame(s->codec_ctx, s->tmp_frame)) == 0) {
                if (frame_decoded) {
                        log_msg(LOG_LEVEL_WARNING,
                                MOD_NAME "Multiple frames decoded at once!\n");
                }
                frame_decoded = true;
                SWAP_PTR(s->frame, s->tmp_frame);
        }
        if (ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                handle_lavd_error(MOD_NAME "recv - ", s, ret);
        }
        return frame_decoded;
}

static decompress_status libavcodec_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq, struct video_frame_callbacks *callbacks, struct pixfmt_desc *internal_props)
{
        UNUSED(frame_seq);
        struct state_libavcodec_decompress *s = (struct state_libavcodec_decompress *) state;

        if (s->desc.color_spec == H264 || s->desc.color_spec == H265) {
                if (s->out_codec == VIDEO_CODEC_NONE &&
                    read_forced_pixfmt(s->desc.color_spec, src, src_len,
                                        internal_props)) {
                        return DECODER_GOT_CODEC;
                }
                if (!check_first_sps_vps(s, src, src_len)) {
                        return DECODER_NO_FRAME;
                }
        }

        if (libav_codec_has_extradata(s->desc.color_spec)) {
                int extradata_size = *(uint32_t *)(void *) src;
                if (s->codec_ctx == NULL) {
                        if (!configure_with(s, s->desc, src + sizeof(uint32_t),
                                            extradata_size)) {
                                return DECODER_NO_FRAME;
                        }
                }
                src += extradata_size + sizeof(uint32_t);
                src_len -= extradata_size + sizeof(uint32_t);
        }

        time_ns_t t0 = get_time_in_ns();

        if (!decode_frame(s, src, src_len)) {
                log_msg(LOG_LEVEL_DEBUG, MOD_NAME "No frame was decoded!\n");
                return DECODER_NO_FRAME;
        }
        s->consecutive_failed_decodes = 0;

        time_ns_t t1 = get_time_in_ns();

        s->frame->opaque = callbacks;
#ifdef HWACC_COMMON_IMPL
        if(s->hwaccel.copy){
                transfer_frame(&s->hwaccel, s->frame);
        }
#endif
        if (s->out_codec != VIDEO_CODEC_NONE) {
                if (!reconfigure_convert_if_needed(s, s->frame->format, s->out_codec, s->desc.width, s->desc.height)) {
                        return DECODER_UNSUPP_PIXFMT;
                }
                if (s->codec_ctx->codec->id ==
                        AV_CODEC_ID_MJPEG &&s->frame->colorspace ==
                        AVCOL_SPC_BT470BG &&s->frame->color_range ==
                        AVCOL_RANGE_MPEG) {
                        s->frame->colorspace = AVCOL_SPC_BT709;
                }
                change_pixfmt(s->frame, dst, s->convert, s->out_codec,
                              s->pitch,
                              s->rgb_shift, &s->sws);
        }
        time_ns_t t2 = get_time_in_ns();
        log_msg(LOG_LEVEL_DEBUG, MOD_NAME "Decompressing %c frame took %f ms, pixfmt change %f ms.\n", av_get_picture_type_char(s->frame->pict_type),
                NS_TO_MS((double) (t1 - t0)), NS_TO_MS((double) (t2 - t1)));
        check_duration(s, (t2 - t0) / NS_IN_SEC_DBL, (t2 - t1) / NS_IN_SEC_DBL);

        if (s->out_codec == VIDEO_CODEC_NONE) {
                log_msg(LOG_LEVEL_VERBOSE,
                        MOD_NAME "Probed output pixel format: %s (%s)\n",
                        av_get_pix_fmt_name(s->codec_ctx->pix_fmt),
                        av_get_pix_fmt_name(s->codec_ctx->sw_pix_fmt));
                enum AVPixelFormat sw_fmt = s->codec_ctx->sw_pix_fmt == AV_PIX_FMT_NONE ? s->codec_ctx->pix_fmt : s->codec_ctx->sw_pix_fmt;
                *internal_props = av_pixfmt_get_desc(sw_fmt);
                internal_props->accel_type =
                    hw_accel_from_pixfmt(s->codec_ctx->pix_fmt);
                return DECODER_GOT_CODEC;
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
        int ret = false;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if (*len < sizeof(int)) {
                                return false;
                        }
                        *(int *) val = false;
                        if (s->codec_ctx && strcmp(s->codec_ctx->codec->name, "h264") == 0) {
                                *(int *) val = true;
                        }
                        if (get_commandline_param("lavd-accept-corrupted")) {
                                *(int *) val =
                                        strcmp(get_commandline_param("lavd-accept-corrupted"), "no") != 0;
                        }

                        *len = sizeof(int);
                        ret = true;
                        break;
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
        if (internal.accel_type != HWACCEL_NONE &&
            hw_accel_to_ug_pixfmt(internal.accel_type) == ugc) {
                return VDEC_PRIO_PREFERRED;
        }

        unsigned i = 0;
        for ( ; i < sizeof decoders / sizeof decoders[0]; ++i) {
                if (decoders[i].ug_codec == compression) {
                        const AVCodec *const decoder =
                            avcodec_find_decoder(decoders[i].avcodec_id);
                        if (decoder != NULL) {
                                break;
                        }
                        MSG(WARNING,
                            "Codec %s supported by lavd but "
                            "not compiled in FFmpeg build.\n",
                            get_codec_name(compression));
                }
        }
        if (i == sizeof decoders / sizeof decoders[0]) { // lavd doesn't handle this compression
                return VDEC_PRIO_NA;
        }

        switch (ugc) {
                case VIDEO_CODEC_NONE:
                        return VDEC_PRIO_PROBE_LO; // for probe
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
                        return VDEC_PRIO_NA;
        }
        if (internal.depth == 0) { // unspecified internal format
                return VDEC_PRIO_LOW;
        }
        return codec_is_a_rgb(ugc) == internal.rgb ? VDEC_PRIO_NORMAL
                                                   : VDEC_PRIO_NOT_PREFERRED;
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

