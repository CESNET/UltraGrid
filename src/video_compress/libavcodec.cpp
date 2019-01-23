/**
 * @file   video_compress/libavcodec.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2016 CESNET, z. s. p. o.
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

#define __STDC_CONSTANT_MACROS

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "libavcodec_common.h"

#include <cassert>
#include <cmath>
#include <map>
#include <regex>
#include <string>
#include <thread>
#include <unordered_map>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "utils/misc.h"
#include "utils/resource_manager.h"
#include "utils/worker.h"
#include "video.h"
#include "video_compress.h"

#ifdef HWACC_VAAPI
extern "C"
{
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavcodec/vaapi.h>
}
#endif

#ifdef __SSE3__
#include "pmmintrin.h"
// compat with older Clang compiler
#ifndef _mm_bslli_si128
#define _mm_bslli_si128 _mm_slli_si128
#endif
#ifndef _mm_bsrli_si128
#define _mm_bsrli_si128 _mm_srli_si128
#endif
#endif

using namespace std;

static constexpr const codec_t DEFAULT_CODEC = MJPG;
static constexpr double DEFAULT_X264_X265_CRF = 22.0;
static constexpr const int DEFAULT_GOP_SIZE = 20;
static constexpr const char *DEFAULT_THREAD_MODE = "slice";

namespace {

struct setparam_param {
        double fps;
        bool interlaced;
        bool no_periodic_intra;
        int cpu_count;
        string thread_mode;
};

static constexpr const char *DEFAULT_NVENC_PRESET = "llhq";
static constexpr const char *DEFAULT_NVENC_RC = "cbr_hq"; // cbr_ld_hq for equally sized frames
static constexpr const char *DEFAULT_QSV_PRESET = "medium";

typedef struct {
        enum AVCodecID av_codec;
        const char *prefered_encoder; ///< can be nullptr
        double avg_bpp;
        string (*get_preset)(string const & enc_name, int width, int height, double fps);
        void (*set_param)(AVCodecContext *, struct setparam_param *);
} codec_params_t;

static void setparam_default(AVCodecContext *, struct setparam_param *);
static void setparam_jpeg(AVCodecContext *, struct setparam_param *);
static void setparam_h264_h265(AVCodecContext *, struct setparam_param *);
static void setparam_vp8_vp9(AVCodecContext *, struct setparam_param *);
static void libavcodec_check_messages(struct state_video_compress_libav *s);
static void libavcodec_compress_done(struct module *mod);
static string get_h264_h265_preset(string const & enc_name, int width, int height, double fps);

static unordered_map<codec_t, codec_params_t, hash<int>> codec_params = {
        { H264, codec_params_t{
                AV_CODEC_ID_H264,
                "libx264",
                0.07 * 2 /* for H.264: 1 - low motion, 2 - medium motion, 4 - high motion */
                * 2, // take into consideration that our H.264 is less effective due to specific preset/tune
                     // note - not used for libx264, which uses CRF by default
                get_h264_h265_preset,
                setparam_h264_h265
        }},
        { H265, codec_params_t{
                AV_CODEC_ID_HEVC,
                "libx265", //nullptr,
                0.04 * 2 * 2, // note - not used for libx265, which uses CRF by default
                get_h264_h265_preset,
                setparam_h264_h265
        }},
        { MJPG, codec_params_t{
                AV_CODEC_ID_MJPEG,
                nullptr,
                1.2,
                nullptr,
                setparam_jpeg
        }},
        { J2K, codec_params_t{
                AV_CODEC_ID_JPEG2000,
                nullptr,
                1.0,
                nullptr,
                setparam_default
        }},
        { VP8, codec_params_t{
                AV_CODEC_ID_VP8,
                nullptr,
                0.4,
                nullptr,
                setparam_vp8_vp9
        }},
        { VP9, codec_params_t{
                AV_CODEC_ID_VP9,
                nullptr,
                0.4,
                nullptr,
                setparam_vp8_vp9,
        }},
        { HFYU, codec_params_t{
                AV_CODEC_ID_HUFFYUV,
                nullptr,
                0,
                nullptr,
                setparam_default
        }},
        { FFV1, codec_params_t{
                AV_CODEC_ID_FFV1,
                nullptr,
                0,
                nullptr,
                setparam_default
        }},
};

codec_t get_ug_for_av_codec(AVCodecID id) {
        for (auto it = codec_params.begin(); it != codec_params.end(); ++it) {
                if (it->second.av_codec == id) {
                        return it->first;
                }
        }

        return VIDEO_CODEC_NONE;
}

struct state_video_compress_libav {
        struct module       module_data;

        pthread_mutex_t    *lavcd_global_lock;

        struct video_desc   saved_desc;

        AVFrame            *in_frame;
        // for every core - parts of the above
        AVFrame           **in_frame_part;
        AVCodecContext     *codec_ctx;

        unsigned char      *decoded; ///< intermediate representation for codecs
                                     ///< that are not directly supported
        codec_t             decoded_codec;
        decoder_t           decoder;

        codec_t             requested_codec_id;
        long long int       requested_bitrate;
        double              requested_bpp;
        double              requested_crf;
        int                 requested_cqp;
        // may be 422, 420 or 0 (no subsampling explicitly requested
        int                 requested_subsampling;
        // actual value used
        AVPixelFormat       selected_pixfmt;

        codec_t             out_codec;

        struct video_desc compressed_desc;

        struct setparam_param params;
        string              backend;
        int                 requested_gop;

        map<string, string> lavc_opts; ///< user-supplied options from command-line

        bool hwenc;
        AVFrame *hwframe;
};

static void to_yuv420p(AVFrame *out_frame, unsigned char *in_data, int width, int height);
static void to_yuv422p(AVFrame *out_frame, unsigned char *in_data, int width, int height);
static void to_yuv444p(AVFrame *out_frame, unsigned char *in_data, int width, int height);
static void to_nv12(AVFrame *out_frame, unsigned char *in_data, int width, int height);
static void v210_to_yuv420p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height);
static void v210_to_yuv422p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height);
static void v210_to_yuv444p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height);
typedef void (*pixfmt_callback_t)(AVFrame *out_frame, unsigned char *in_data, int width, int height);
static pixfmt_callback_t select_pixfmt_callback(AVPixelFormat fmt, codec_t src);


static void usage(void);
static int parse_fmt(struct state_video_compress_libav *s, char *fmt);
static void cleanup(struct state_video_compress_libav *s);

static void print_codec_info(AVCodecID id, char *buf, size_t buflen)
{
#if LIBAVCODEC_VERSION_INT > AV_VERSION_INT(58, 9, 100)
        assert(buflen > 0);
        buf[0] = '\0';
        const AVCodec *codec = nullptr;
        void *i = 0;
        char *enc = (char *) alloca(buflen);
        char *dec = (char *) alloca(buflen);
        dec[0] = enc[0] = '\0';
        while ((codec = av_codec_iterate(&i))) {
                if (av_codec_is_encoder(codec) && codec->id == id) {
                        strncat(enc, " ", buflen - strlen(enc) - 1);
                        strncat(enc, codec->name, buflen - strlen(enc) - 1);
                }
                if (av_codec_is_decoder(codec) && codec->id == id) {
                        strncat(dec, " ", buflen - strlen(dec) - 1);
                        strncat(dec, codec->name, buflen - strlen(dec) - 1);
                }
        }
        if (strlen(enc) || strlen(dec)) {
                strncat(buf, " (", buflen - strlen(buf) - 1);
                if (strlen(enc)) {
                        strncat(buf, "enc:", buflen - strlen(buf) - 1);
                        strncat(buf, enc, buflen - strlen(buf) - 1);
                }
                if (strlen(dec)) {
                        if (strlen(enc)) {
                                strncat(buf, ", ", buflen - strlen(buf) - 1);
                        }
                        strncat(buf, "dec:", buflen - strlen(buf) - 1);
                        strncat(buf, dec, buflen - strlen(buf) - 1);
                }
                strncat(buf, ")", buflen - strlen(buf) - 1);
        }
#elif LIBAVCODEC_VERSION_MAJOR >= 54
        const AVCodec *codec;
        if ((codec = avcodec_find_encoder(id))) {
                strncpy(buf, " (enc:", buflen - 1);
                buf[buflen - 1] = '\0';
                do {
                        if (av_codec_is_encoder(codec) && codec->id == id) {
                                strncat(buf, " ", buflen - strlen(buf) - 1);
                                strncat(buf, codec->name, buflen - strlen(buf) - 1);
                        }
                } while ((codec = av_codec_next(codec)));
        }

        if ((codec = avcodec_find_decoder(id))) {
                if (avcodec_find_encoder(id)) {
                        strncat(buf, ", ", buflen - strlen(buf) - 1);
                } else {
                        strncat(buf, " (", buflen - strlen(buf) - 1);
                }
                strncat(buf, "dec:", buflen - strlen(buf) - 1);
                do {
                        if (av_codec_is_decoder(codec) && codec->id == id) {
                                strncat(buf, " ", buflen - strlen(buf) - 1);
                                strncat(buf, codec->name, buflen - strlen(buf) - 1);
                        }
                } while ((codec = av_codec_next(codec)));
        }
        if (avcodec_find_encoder(id) || avcodec_find_decoder(id)) {
                strncat(buf, ")", buflen - strlen(buf) - 1);
        }
#else
        UNUSED(id);
        UNUSED(buf);
        UNUSED(buflen);
#endif
}

static void usage() {
        printf("Libavcodec encoder usage:\n");
        printf("\t-c libavcodec[:codec=<codec_name>|:encoder=<encoder>][:bitrate=<bits_per_sec>|:bpp=<bits_per_pixel>][:crf=<crf>|:cqp=<cqp>]"
                        "[:subsampling=<subsampling>][:gop=<gop>]"
                        "[:disable_intra_refresh][:threads=<thr_mode>][:<lavc_opt>=<val>]*\n");
        printf("\t\t<encoder> specifies encoder (eg. nvenc or libx264 for H.264)\n");
        printf("\t\t<codec_name> may be specified codec name (default MJPEG), supported codecs:\n");
        for (auto && param : codec_params) {
                if (param.second.av_codec != AV_CODEC_ID_NONE) {
                        char avail[1024];
                        const AVCodec *codec;
                        if ((codec = avcodec_find_encoder(param.second.av_codec))) {
                                strcpy(avail, "available");
                        } else {
                                strcpy(avail, "not available");
                        }
                        print_codec_info(param.second.av_codec, avail + strlen(avail), sizeof avail - strlen(avail));

                        printf("\t\t\t%s - %s\n", get_codec_name(param.first), avail);
                }

        }
        printf("\t\tdisable_intra_refresh - do not use Periodic Intra Refresh (H.264/H.265)\n");
        printf("\t\t<bits_per_sec> specifies requested bitrate\n");
        printf("\t\t\t0 means codec default (same as when parameter omitted)\n");
        printf("\t\t<crf> specifies CRF factor (only for libx264/libx265)\n");
        printf("\t\t<subsampling> may be one of 444, 422, or 420, default 420 for progresive, 422 for interlaced\n");
        printf("\t\t<thr_mode> can be one of \"no\", \"frame\" or \"slice\"\n");
        printf("\t\t<gop> specifies GOP size\n");
        printf("\t\t<lavc_opt> arbitrary option to be passed directly to libavcodec (eg. preset=veryfast)\n");
        printf("\tLibavcodec version (linked): %s\n", LIBAVCODEC_IDENT);
}

static int parse_fmt(struct state_video_compress_libav *s, char *fmt) {
        char *item, *save_ptr = NULL;
        if(fmt) {
                while((item = strtok_r(fmt, ":", &save_ptr)) != NULL) {
                        if(strncasecmp("help", item, strlen("help")) == 0) {
                                usage();
                                return 1;
                        } else if(strncasecmp("codec=", item, strlen("codec=")) == 0) {
                                char *codec = item + strlen("codec=");
                                s->requested_codec_id = get_codec_from_name(codec);
                                if (s->requested_codec_id == VIDEO_CODEC_NONE) {
                                        log_msg(LOG_LEVEL_ERROR, "[lavc] Unable to find codec: \"%s\"\n", codec);
                                        return -1;
                                }
                        } else if(strncasecmp("bitrate=", item, strlen("bitrate=")) == 0) {
                                char *bitrate_str = item + strlen("bitrate=");
                                s->requested_bitrate = unit_evaluate(bitrate_str);
                                assert(s->requested_bitrate >= 0);
                        } else if(strncasecmp("bpp=", item, strlen("bpp=")) == 0) {
                                char *bpp_str = item + strlen("bpp=");
                                s->requested_bpp = unit_evaluate_dbl(bpp_str);
                                assert(!std::isnan(s->requested_bpp));
                        } else if(strncasecmp("crf=", item, strlen("crf=")) == 0) {
                                char *crf_str = item + strlen("crf=");
                                s->requested_crf = atof(crf_str);
                        } else if(strncasecmp("cqp=", item, strlen("cqp=")) == 0) {
                                char *cqp_str = item + strlen("cqp=");
                                s->requested_cqp = atoi(cqp_str);
                        } else if(strncasecmp("subsampling=", item, strlen("subsampling=")) == 0) {
                                char *subsample_str = item + strlen("subsampling=");
                                s->requested_subsampling = atoi(subsample_str);
                                if (s->requested_subsampling != 444 &&
                                                s->requested_subsampling != 422 &&
                                                s->requested_subsampling != 420) {
                                        log_msg(LOG_LEVEL_ERROR, "[lavc] Supported subsampling is 444, 422, or 420.\n");
                                        return -1;
                                }
                        } else if (strcasecmp("disable_intra_refresh", item) == 0) {
                                s->params.no_periodic_intra = true;
                        } else if(strncasecmp("threads=", item, strlen("threads=")) == 0) {
                                char *threads = item + strlen("threads=");
                                s->params.thread_mode = threads;
                        } else if(strncasecmp("encoder=", item, strlen("encoder=")) == 0) {
                                char *backend = item + strlen("encoder=");
                                s->backend = backend;
                        } else if(strncasecmp("gop=", item, strlen("gop=")) == 0) {
                                char *gop = item + strlen("gop=");
                                s->requested_gop = atoi(gop);
                        } else if (strchr(item, '=')) {
                                string key, val;
                                key = string(item, strchr(item, '='));
                                val = string(strchr(item, '=') + 1);
                                s->lavc_opts[key] = val;
                        } else {
                                log_msg(LOG_LEVEL_ERROR, "[lavc] Error: unknown option %s.\n",
                                                item);
                                return -1;
                        }
                        fmt = NULL;
                }
        }

        return 0;
}

static list<compress_preset> get_libavcodec_presets() {
        list<compress_preset> ret;
#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        avcodec_register_all();
#endif

        pthread_mutex_t *lavcd_global_lock = rm_acquire_shared_lock(LAVCD_LOCK_NAME);
        pthread_mutex_lock(lavcd_global_lock);

        if (avcodec_find_encoder_by_name("libx264")) {
                ret.push_back({"encoder=libx264:bpp=0.096", 20, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 0.096);}, {25, 1.5, 0}, {15, 1, 0}});
                ret.push_back({"encoder=libx264:bpp=0.193", 30, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 0.193);}, {28, 1.5, 0}, {20, 1, 0}});
                ret.push_back({"encoder=libx264:bpp=0.289", 50, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 0.289);}, {30, 1.5, 0}, {25, 1, 0}});
        }
        // NVENC and MJPEG are disabled in order not to be chosen by CoUniverse.
        // Enable if needed (also possible to add H.265 etc).
#if 0
        AVCodec *codec;
        if ((codec = avcodec_find_encoder_by_name("nvenc_h264"))) {
                AVCodecContext *codec_ctx = avcodec_alloc_context3(codec);
                assert(codec_ctx);
                codec_ctx->pix_fmt = AV_PIX_FMT_NV12;
                codec_ctx->width = 1920;
                codec_ctx->height = 1080;
                if (avcodec_open2(codec_ctx, codec, NULL) == 0) {
                        ret.push_back({"encoder=nvenc_h264:bpp=0.096", 20, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 0.096);}, {25, 0, 0.2}, {15, 1, 0}});
                        ret.push_back({"encoder=nvenc_h264:bpp=0.193", 30, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 0.193);}, {28, 0, 0.2}, {20, 1, 0}});
                        ret.push_back({"encoder=nvenc_h264:bitrate=0.289", 50, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 0.289);}, {30, 0, 0.2}, {25, 1, 0}});
                        avcodec_close(codec_ctx);
                }
                av_free(codec_ctx);
        }
#endif
#if 0
        ret.push_back({ "codec=MJPEG", 35, 50*1000*1000, {20, 0.75, 0}, {10, 0.5, 0}});
#endif

        pthread_mutex_unlock(lavcd_global_lock);
        rm_release_shared_lock(LAVCD_LOCK_NAME);

        return ret;
}

struct module * libavcodec_compress_init(struct module *parent, const char *opts)
{
        struct state_video_compress_libav *s;

        s = new state_video_compress_libav();
        s->lavcd_global_lock = rm_acquire_shared_lock(LAVCD_LOCK_NAME);
        if (log_level >= LOG_LEVEL_VERBOSE) {
                av_log_set_level(AV_LOG_VERBOSE);
        }
#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        /*  register all the codecs (you can also register only the codec
         *         you wish to have smaller code */
        avcodec_register_all();
#endif

        s->codec_ctx = NULL;
        s->in_frame = NULL;
        s->requested_codec_id = VIDEO_CODEC_NONE;
        s->requested_subsampling = 0;
        s->params.thread_mode = DEFAULT_THREAD_MODE;
        // both following options take 0 as a valid argument, so we use -1 as an implicit value
        s->requested_crf = -1;
        s->requested_cqp = -1;

        memset(&s->saved_desc, 0, sizeof(s->saved_desc));

        char *fmt = strdup(opts);
        int ret = parse_fmt(s, fmt);
        free(fmt);
        if(ret != 0) {
                delete s;
                if(ret > 0)
                        return &compress_init_noerr;
                else
                        return NULL;
        }

        s->params.cpu_count = thread::hardware_concurrency();
        if(s->params.cpu_count < 1) {
                log_msg(LOG_LEVEL_WARNING, "Warning: Cannot get number of CPU cores!\n");
                s->params.cpu_count = 1;
        }
        s->in_frame_part = (AVFrame **) calloc(s->params.cpu_count, sizeof(AVFrame *));
        for(int i = 0; i < s->params.cpu_count; i++) {
                s->in_frame_part[i] = av_frame_alloc();
        }

        s->decoded = NULL;

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = libavcodec_compress_done;
        module_register(&s->module_data, parent);

        s->hwenc = false;
        s->hwframe = NULL;

        return &s->module_data;
}

#ifdef HWACC_VAAPI
static int create_hw_device_ctx(enum AVHWDeviceType type, AVBufferRef **device_ref){
        int ret;
        ret = av_hwdevice_ctx_create(device_ref, type, NULL, NULL, 0);

        if(ret < 0){
                log_msg(LOG_LEVEL_ERROR, "[lavc] Unable to create hwdevice!!\n");
                return ret;
        }

        return 0;
}

static int create_hw_frame_ctx(AVBufferRef *device_ref,
                AVCodecContext *s,
                enum AVPixelFormat format,
                enum AVPixelFormat sw_format,
                int pool_size,
                AVBufferRef **ctx)
{
        *ctx = av_hwframe_ctx_alloc(device_ref);
        if(!*ctx){
                log_msg(LOG_LEVEL_ERROR, "[lavc] Failed to allocate hwframe_ctx!!\n");
                return -1;
        }

        AVHWFramesContext *frames_ctx = (AVHWFramesContext *) (*ctx)->data;
        frames_ctx->format    = format;
        frames_ctx->width     = s->width;
        frames_ctx->height    = s->height;
        frames_ctx->sw_format = sw_format;
        frames_ctx->initial_pool_size = pool_size;

        int ret = av_hwframe_ctx_init(*ctx);
        if (ret < 0) {
                av_buffer_unref(ctx);
                *ctx = NULL;
                log_msg(LOG_LEVEL_ERROR, "[lavc] Unable to init hwframe_ctx!!\n\n");
                return ret;
        }

        return 0;
}

static int vaapi_init(struct AVCodecContext *s){

        int pool_size = 20; //Default in ffmpeg examples

        AVBufferRef *device_ref = nullptr;
        AVBufferRef *hw_frames_ctx = nullptr;
        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_VAAPI, &device_ref);
        if(ret < 0)
                goto fail;

        if (s->active_thread_type & FF_THREAD_FRAME)
                pool_size += s->thread_count;

        ret = create_hw_frame_ctx(device_ref,
                        s,
                        AV_PIX_FMT_VAAPI,
                        AV_PIX_FMT_NV12,
                        pool_size,
                        &hw_frames_ctx);
        if(ret < 0)
                goto fail;

        s->hw_frames_ctx = hw_frames_ctx;

        av_buffer_unref(&device_ref);
        return 0;

fail:
        av_buffer_unref(&hw_frames_ctx);
        av_buffer_unref(&device_ref);
        return ret;
}
#endif

/**
 * Finds best pixel format
 *
 * Iterates over formats in req_pix_fmts and tries to find the same format in
 * second list, codec_pix_fmts. If found, returns that format. Efectivelly
 * select first match of item from first list in second list.
 */
static enum AVPixelFormat get_first_matching_pix_fmt(const enum AVPixelFormat **req_pix_fmt_it,
                const enum AVPixelFormat *codec_pix_fmts)
{
        assert(req_pix_fmt_it != NULL);
        if(codec_pix_fmts == NULL)
                return AV_PIX_FMT_NONE;

        if (log_level >= LOG_LEVEL_DEBUG) {
                char out[1024] = "[lavc] Available codec pixel formats:";
                const enum AVPixelFormat *it = codec_pix_fmts;
                while (*it != AV_PIX_FMT_NONE) {
                        strncat(out, " ", sizeof out - strlen(out) - 1);
                        strncat(out, av_get_pix_fmt_name(*it++), sizeof out - strlen(out) - 1);
                }
                log_msg(LOG_LEVEL_DEBUG, "%s\n", out);
                out[0] = '\0';
                strncat(out, "[lavd] Requested pixel formats:", sizeof out - strlen(out) - 1);
                it = *req_pix_fmt_it;
                while (*it != AV_PIX_FMT_NONE) {
                        strncat(out, " ", sizeof out - strlen(out) - 1);
                        strncat(out, av_get_pix_fmt_name(*it++), sizeof out - strlen(out) - 1);
                }
                log_msg(LOG_LEVEL_DEBUG, "%s\n", out);
        }

        enum AVPixelFormat req;
        while((req = *(*req_pix_fmt_it)++) != AV_PIX_FMT_NONE) {
                const enum AVPixelFormat *tmp = codec_pix_fmts;
                enum AVPixelFormat fmt;
                while((fmt = *tmp++) != AV_PIX_FMT_NONE) {
                        if(fmt == req)
                                return req;
                }
        }

        return AV_PIX_FMT_NONE;
}

bool set_codec_ctx_params(struct state_video_compress_libav *s, AVPixelFormat pix_fmt, struct video_desc desc, codec_t ug_codec)
{
        bool is_x264_x265 = strcmp(s->codec_ctx->codec->name, "libx264") == 0 ||
                strcmp(s->codec_ctx->codec->name, "libx265") == 0;

        double avg_bpp; // average bit per pixel
        avg_bpp = s->requested_bpp > 0.0 ? s->requested_bpp :
                codec_params[ug_codec].avg_bpp;

        int bitrate = s->requested_bitrate > 0 ? s->requested_bitrate :
                desc.width * desc.height * avg_bpp * desc.fps;

        bool have_preset = s->lavc_opts.find("preset") != s->lavc_opts.end();

        s->codec_ctx->strict_std_compliance = -2;

        // set bitrate
        if ((s->requested_bitrate > 0 || s->requested_bpp > 0.0)
                        || !is_x264_x265) {
                s->codec_ctx->bit_rate = bitrate;
                s->codec_ctx->bit_rate_tolerance = bitrate / desc.fps * 6;
                log_msg(LOG_LEVEL_INFO, "[lavc] Setting bitrate to %d bps.\n", bitrate);
        }

        if (is_x264_x265) {
                // set CRF unless explicitly specified CQP or ABR (bitrate)
                if (s->requested_crf >= 0.0 || (s->requested_bitrate == 0 && s->requested_bpp == 0.0 && s->requested_cqp == -1)) {
                        double crf = s->requested_crf >= 0.0 ? s->requested_crf : DEFAULT_X264_X265_CRF;
                        av_opt_set_double(s->codec_ctx->priv_data, "crf", crf, 0);
                        log_msg(LOG_LEVEL_INFO, "[lavc] Setting CRF to %.2f.\n", crf);
                }
                if (s->requested_cqp >= 0) {
                        av_opt_set_int(s->codec_ctx->priv_data, "qp", s->requested_cqp, 0);
                        log_msg(LOG_LEVEL_INFO, "[lavc] Setting CQP to %d.\n", s->requested_cqp);
                }
        } else {
                if (s->requested_crf >= 0.0) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set CRF! Not supported for this encoder, ignored.\n");
                }
                if (s->requested_cqp >= 0.0) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set CQP! Not supported for this encoder, ignored.\n");
                }
        }

        /* resolution must be a multiple of two */
        s->codec_ctx->width = desc.width;
        s->codec_ctx->height = desc.height;
        /* frames per second */
        s->codec_ctx->time_base = (AVRational){1,(int) desc.fps};
        if (s->requested_gop) {
                s->codec_ctx->gop_size = s->requested_gop;
        } else {
                s->codec_ctx->gop_size = DEFAULT_GOP_SIZE;
        }
        s->codec_ctx->max_b_frames = 0;

        s->codec_ctx->pix_fmt = pix_fmt;

        codec_params[ug_codec].set_param(s->codec_ctx, &s->params);

        if (!have_preset) {
                string preset{};
                if (codec_params[ug_codec].get_preset) {
                        preset = codec_params[ug_codec].get_preset(s->codec_ctx->codec->name, desc.width, desc.height, desc.fps);
                }

                if (!preset.empty()) {
                        if (av_opt_set(s->codec_ctx->priv_data, "preset", preset.c_str(), 0) != 0) {
                                LOG(LOG_LEVEL_WARNING) << "[lavc] Warning: Unable to set preset.\n";
                        } else {
                                LOG(LOG_LEVEL_INFO) << "[lavc] Setting preset to " << preset <<  ".\n";
                        }
                } else {
                        LOG(LOG_LEVEL_WARNING) << "[lavc] Warning: Unable to find suitable preset for encoder " << s->codec_ctx->codec->name << ".\n";
                }
        }

        // set user supplied parameters
        for (auto item : s->lavc_opts) {
                if(av_opt_set(s->codec_ctx->priv_data, item.first.c_str(), item.second.c_str(), 0) != 0) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Error: Unable to set '%s' to '%s'. Check command-line options.\n", item.first.c_str(), item.second.c_str());
                        return false;
                }
        }

        return true;
}

ADD_TO_PARAM(lavc_use_codec, "lavc-use-codec",
                "* lavc-use-codec=<c>\n"
                "  Restrict codec to use user specified pix fmt. Can be used eg. to enforce\n"
                "  AV_PIX_FMT_NV12 (nv12) since some time ago, other codecs were broken\n"
                "  for NVENC encoder.\n"
                "  Another possibility is to use yuv420p10le, yuv422p10le or yuv444p10le\n"
                "  to force 10-bit encoding.\n");

static bool configure_with(struct state_video_compress_libav *s, struct video_desc desc)
{
        int ret;
        codec_t ug_codec = VIDEO_CODEC_NONE;
        AVPixelFormat pix_fmt;
        AVCodec *codec = nullptr;

        s->params.fps = desc.fps;
        s->params.interlaced = desc.interlacing == INTERLACED_MERGED;

        // Open encoder specified by user if given
        if (!s->backend.empty()) {
                codec = avcodec_find_encoder_by_name(s->backend.c_str());
                if (!codec) {
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Warning: requested encoder \"%s\" not found!\n",
                                        s->backend.c_str());
                        return false;
                }
                if (s->requested_codec_id != VIDEO_CODEC_NONE && s->requested_codec_id != get_ug_for_av_codec(codec->id)) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Codec and encoder don't match!\n");
                        return false;

                }
                ug_codec = get_ug_for_av_codec(codec->id);
                if (ug_codec == VIDEO_CODEC_NONE) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Requested encoder not supported in UG!\n");
                        return false;
                }
        }

        if (ug_codec == VIDEO_CODEC_NONE) {
                if (s->requested_codec_id == VIDEO_CODEC_NONE) {
                        ug_codec = DEFAULT_CODEC;
                } else {
                        ug_codec = s->requested_codec_id;
                }
        }
        if (codec_params.find(ug_codec) == codec_params.end()) {
                log_msg(LOG_LEVEL_ERROR, "[lavc] Requested output codec isn't "
                                "currently supported.\n");
                return false;
        }

        // Else, try to open prefered encoder for requested codec
        if (!codec && codec_params[ug_codec].prefered_encoder) {
                const char *prefered_encoder = codec_params[ug_codec].prefered_encoder;
                codec = avcodec_find_encoder_by_name(prefered_encoder);
                if (!codec) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Warning: prefered encoder \"%s\" not found! Trying default encoder.\n",
                                        prefered_encoder);
                }
        }
        // Finally, try to open any encoder for requested codec
        if (!codec) {
                codec = avcodec_find_encoder(codec_params[ug_codec].av_codec);
        }

        if (!codec) {
                log_msg(LOG_LEVEL_ERROR, "Libavcodec doesn't contain encoder for specified codec.\n"
                                "Hint: Check if you have libavcodec-extra package installed.\n");
                return false;
        } else {
                log_msg(LOG_LEVEL_NOTICE, "[lavc] Using codec: %s, encoder: %s\n",
                                get_codec_name(ug_codec), codec->name);
        }

        enum AVPixelFormat requested_pix_fmts[100];
        int total_pix_fmts = 0;

#ifdef HWACC_VAAPI
        if (regex_match(codec->name, regex(".*vaapi.*"))) {
                requested_pix_fmts[total_pix_fmts++] = AV_PIX_FMT_VAAPI;
        }
#endif

        if (s->requested_subsampling == 0) {
                // for interlaced formats, it is better to use either 422 or 444
                if (desc.interlacing == INTERLACED_MERGED) {
                        // 422
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts422_8, sizeof(fmts422_8));
                        total_pix_fmts += sizeof(fmts422_8) / sizeof(enum AVPixelFormat);
                        // 444
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts444_8, sizeof(fmts444_8));
                        total_pix_fmts += sizeof(fmts444_8) / sizeof(enum AVPixelFormat);
                        // 420
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts420_8, sizeof(fmts420_8));
                        total_pix_fmts += sizeof(fmts420_8) / sizeof(enum AVPixelFormat);
                } else {
                        // 420
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts420_8, sizeof(fmts420_8));
                        total_pix_fmts += sizeof(fmts420_8) / sizeof(enum AVPixelFormat);
                        // 422
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts422_8, sizeof(fmts422_8));
                        total_pix_fmts += sizeof(fmts422_8) / sizeof(enum AVPixelFormat);
                        // 444
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts444_8, sizeof(fmts444_8));
                        total_pix_fmts += sizeof(fmts444_8) / sizeof(enum AVPixelFormat);
                }
        } else {
                switch (s->requested_subsampling) {
                case 420:
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts420_8, sizeof(fmts420_8));
                        total_pix_fmts += sizeof(fmts420_8) / sizeof(enum AVPixelFormat);
                        break;
                case 422:
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts422_8, sizeof(fmts422_8));
                        total_pix_fmts += sizeof(fmts422_8) / sizeof(enum AVPixelFormat);
                        break;
                case 444:
                        memcpy(requested_pix_fmts + total_pix_fmts,
                                        fmts444_8, sizeof(fmts444_8));
                        total_pix_fmts += sizeof(fmts444_8) / sizeof(enum AVPixelFormat);
                        break;
                default:
                        abort();
                }
        }

        if (get_commandline_param("lavc-use-codec")) {
                const char *val = get_commandline_param("lavc-use-codec");
                requested_pix_fmts[0] = av_get_pix_fmt(val);
                total_pix_fmts = 1;
        }

        requested_pix_fmts[total_pix_fmts++] = AV_PIX_FMT_NONE;

        // Try to open the codec context
        // It is done in a loop because some pixel formats that are reported
        // by codec can actually fail (typically YUV444 in hevc_nvenc for Maxwell
        // cards).
        const AVPixelFormat *pix_fmt_iterator = requested_pix_fmts;
        while ((pix_fmt = get_first_matching_pix_fmt(&pix_fmt_iterator, codec->pix_fmts)) != AV_PIX_FMT_NONE) {
		// avcodec_alloc_context3 allocates context and sets default value
		s->codec_ctx = avcodec_alloc_context3(codec);
		if (!s->codec_ctx) {
			log_msg(LOG_LEVEL_ERROR, "Could not allocate video codec context\n");
			return false;
		}

		if (!set_codec_ctx_params(s, pix_fmt, desc, ug_codec)) {
			avcodec_free_context(&s->codec_ctx);
                        s->codec_ctx = NULL;
			return false;
		}

                log_msg(LOG_LEVEL_VERBOSE, "[lavc] Trying pixfmt: %s\n", av_get_pix_fmt_name(pix_fmt));
#ifdef HWACC_VAAPI
                if (pix_fmt == AV_PIX_FMT_VAAPI){
                        int ret = vaapi_init(s->codec_ctx);
                        if (ret != 0) {
                                continue;
                        }
                        s->hwenc = true;
                        s->hwframe = av_frame_alloc();
                        av_hwframe_get_buffer(s->codec_ctx->hw_frames_ctx, s->hwframe, 0);
                        pix_fmt = AV_PIX_FMT_NV12;
                }
#endif
                /* open it */
                pthread_mutex_lock(s->lavcd_global_lock);
                if (avcodec_open2(s->codec_ctx, codec, NULL) < 0) {
                        avcodec_free_context(&s->codec_ctx);
                        s->codec_ctx = NULL;
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Could not open codec for pixel format %s\n", av_get_pix_fmt_name(pix_fmt));
                        pthread_mutex_unlock(s->lavcd_global_lock);
                        continue;
                }

                pthread_mutex_unlock(s->lavcd_global_lock);
		break;
	}

        if (pix_fmt == AV_PIX_FMT_NONE) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to find suitable pixel format for: %s.\n", get_codec_name(desc.color_spec));
                if (s->requested_subsampling != 0) {
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Requested subsampling not supported. "
                                        "Try different subsampling, eg. "
                                        "\"subsampling={420,422,444}\".\n");
                }
                return false;
        }

        log_msg(LOG_LEVEL_INFO, "[lavc] Selected pixfmt: %s\n", av_get_pix_fmt_name(pix_fmt));
        s->selected_pixfmt = pix_fmt;

        s->decoded_codec = UYVY; // most compressions use 8-bit YUV formats internally
        switch(desc.color_spec) {
                case UYVY:
                        s->decoder = (decoder_t) memcpy;
                        break;
                case YUYV:
                        s->decoder = (decoder_t) vc_copylineYUYV;
                        break;
                case v210:
                        if (s->selected_pixfmt == AV_PIX_FMT_YUV420P10LE ||
                                        s->selected_pixfmt == AV_PIX_FMT_YUV422P10LE ||
                                        s->selected_pixfmt == AV_PIX_FMT_YUV444P10LE) {
                                s->decoder = (decoder_t) memcpy;
                                s->decoded_codec = v210;
                        } else {
                                s->decoder = (decoder_t) vc_copylinev210;
                        }
                        break;
                case RGB:
                        s->decoder = (decoder_t) vc_copylineRGBtoUYVY;
                        break;
                case BGR:
                        s->decoder = (decoder_t) vc_copylineBGRtoUYVY;
                        break;
                case RGBA:
                        s->decoder = (decoder_t) vc_copylineRGBAtoUYVY;
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "[Libavcodec] Unable to find "
                                        "appropriate pixel format.\n");
                        return false;
        }

        s->decoded = (unsigned char *) malloc(vc_get_linesize(desc.width, s->decoded_codec) * desc.height);

        s->in_frame = av_frame_alloc();
        if (!s->in_frame) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate video frame\n");
                return false;
        }

        AVPixelFormat fmt = (s->hwenc) ? AV_PIX_FMT_NV12 : s->codec_ctx->pix_fmt;
#if LIBAVCODEC_VERSION_MAJOR >= 53
        s->in_frame->format = fmt;
        s->in_frame->width = s->codec_ctx->width;
        s->in_frame->height = s->codec_ctx->height;
#endif

        /* the image can be allocated by any means and av_image_alloc() is
         * just the most convenient way if av_malloc() is to be used */
        ret = av_image_alloc(s->in_frame->data, s->in_frame->linesize,
                        s->codec_ctx->width, s->codec_ctx->height,
                        fmt, 32);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate raw picture buffer\n");
                return false;
        }
        for(int i = 0; i < s->params.cpu_count; ++i) {
                int chunk_size = s->codec_ctx->height / s->params.cpu_count;
                chunk_size = chunk_size / 2 * 2;
                s->in_frame_part[i]->data[0] = s->in_frame->data[0] + s->in_frame->linesize[0] * i *
                        chunk_size;

                if (av_pix_fmt_desc_get(s->selected_pixfmt)->log2_chroma_h == 1) { // eg. 4:2:0
                        chunk_size /= 2;
                }
                s->in_frame_part[i]->data[1] = s->in_frame->data[1] + s->in_frame->linesize[1] * i *
                        chunk_size;
                s->in_frame_part[i]->data[2] = s->in_frame->data[2] + s->in_frame->linesize[2] * i *
                        chunk_size;
                s->in_frame_part[i]->linesize[0] = s->in_frame->linesize[0];
                s->in_frame_part[i]->linesize[1] = s->in_frame->linesize[1];
                s->in_frame_part[i]->linesize[2] = s->in_frame->linesize[2];
        }

        s->saved_desc = desc;
        s->compressed_desc = desc;
        s->compressed_desc.color_spec = ug_codec;
        s->compressed_desc.tile_count = 1;

        s->out_codec = s->compressed_desc.color_spec;

        return true;
}

static void to_yuv420p(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                unsigned char *src = in_data + y * (width * 2);
                /*  every odd row */
                unsigned char *src2 = in_data + (y + 1) * (width * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_y2 = out_frame->data[0] + out_frame->linesize[0] * (y + 1);
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y / 2;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y / 2;
                for(int x = 0; x < width / 2; ++x) {
                        *dst_cb++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                }
        }
}

static void to_yuv422p(AVFrame *out_frame, unsigned char *src, int width, int height)
{
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;
                for(int x = 0; x < width; x += 2) {
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

static void to_yuv444p(AVFrame *out_frame, unsigned char *src, int width, int height)
{
        for(int y = 0; y < height; ++y) {
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;
                for(int x = 0; x < width; x += 2) {
                        *dst_cb++ = *src;
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

static void to_nv12(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                unsigned char *src = in_data + y * (width * 2);
                /*  every odd row */
                unsigned char *src2 = in_data + (y + 1) * (width * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_y2 = out_frame->data[0] + out_frame->linesize[0] * (y + 1);
                unsigned char *dst_cbcr = out_frame->data[1] + out_frame->linesize[1] * y / 2;

                int x = 0;
#ifdef __SSE3__
                __m128i yuv;
                __m128i yuv2;
                __m128i y1;
                __m128i y2;
                __m128i y3;
                __m128i y4;
                __m128i uv;
                __m128i uv2;
                __m128i uv3;
                __m128i uv4;
                __m128i ymask = _mm_set1_epi32(0xFF00FF00);
                __m128i dsty;
                __m128i dsty2;
                __m128i dstuv;

                for (; x < (width - 15); x += 16){
                        yuv = _mm_lddqu_si128((__m128i const*) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*) src2);
                        src += 16;
                        src2 += 16;

                        y1 = _mm_and_si128(ymask, yuv);
                        y1 = _mm_bsrli_si128(y1, 1);
                        y2 = _mm_and_si128(ymask, yuv2);
                        y2 = _mm_bsrli_si128(y2, 1);

                        uv = _mm_andnot_si128(ymask, yuv);
                        uv2 = _mm_andnot_si128(ymask, yuv2);

                        uv = _mm_avg_epu8(uv, uv2);

                        yuv = _mm_lddqu_si128((__m128i const*) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*) src2);
                        src += 16;
                        src2 += 16;

                        y3 = _mm_and_si128(ymask, yuv);
                        y3 = _mm_bsrli_si128(y3, 1);
                        y4 = _mm_and_si128(ymask, yuv2);
                        y4 = _mm_bsrli_si128(y4, 1);

                        uv3 = _mm_andnot_si128(ymask, yuv);
                        uv4 = _mm_andnot_si128(ymask, yuv2);

                        uv3 = _mm_avg_epu8(uv3, uv4);

                        dsty = _mm_packus_epi16(y1, y3);
                        dsty2 = _mm_packus_epi16(y2, y4);
                        dstuv = _mm_packus_epi16(uv, uv3);
                        _mm_storeu_si128((__m128i *) dst_y, dsty);
                        _mm_storeu_si128((__m128i *) dst_y2, dsty2);
                        _mm_storeu_si128((__m128i *) dst_cbcr, dstuv);
                        dst_y += 16;
                        dst_y2 += 16;
                        dst_cbcr += 16;
                }
#endif
                for(; x < width - 1; x += 2) {
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                }
        }
}

static void v210_to_yuv420p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y / 2);
                for(int x = 0; x < width / 6; ++x) {
			//block 1, bits  0 -  9: U0+0
			//block 1, bits 10 - 19: Y0
			//block 1, bits 20 - 29: V0+1
			//block 2, bits  0 -  9: Y1
			//block 2, bits 10 - 19: U2+3
			//block 2, bits 20 - 29: Y2
			//block 3, bits  0 -  9: V2+3
			//block 3, bits 10 - 19: Y3
			//block 3, bits 20 - 29: U4+5
			//block 4, bits  0 -  9: Y4
			//block 4, bits 10 - 19: V4+5
			//block 4, bits 20 - 29: Y5
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;
                        w1_0 = *src2++;
                        w1_1 = *src2++;
                        w1_2 = *src2++;
                        w1_3 = *src2++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_y2++ = (w1_0 >> 10) & 0x3ff;
                        *dst_y2++ = w1_1 & 0x3ff;
                        *dst_y2++ = (w1_1 >> 20) & 0x3ff;
                        *dst_y2++ = (w1_2 >> 10) & 0x3ff;
                        *dst_y2++ = w1_3 & 0x3ff;
                        *dst_y2++ = (w1_3 >> 20) & 0x3ff;

                        *dst_cb++ = ((w0_0 & 0x3ff) + (w1_0 & 0x3ff)) / 2;
                        *dst_cb++ = (((w0_1 >> 10) & 0x3ff) + ((w1_1 >> 10) & 0x3ff)) / 2;
                        *dst_cb++ = (((w0_2 >> 20) & 0x3ff) + ((w1_2 >> 20) & 0x3ff)) / 2;

                        *dst_cr++ = (((w0_0 >> 20) & 0x3ff) + ((w1_0 >> 20) & 0x3ff)) / 2;
                        *dst_cr++ = ((w0_2 & 0x3ff) + (w1_2 & 0x3ff)) / 2;
                        *dst_cr++ = (((w0_3 >> 10) & 0x3ff) + ((w1_3 >> 10) & 0x3ff)) / 2;
                }
        }
}

static void v210_to_yuv422p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);
                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;

                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                }
        }
}

static void v210_to_yuv444p10le(AVFrame *out_frame, unsigned char *in_data, int width, int height)
{
        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);
                for(int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;

                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                }
        }
}

static pixfmt_callback_t select_pixfmt_callback(AVPixelFormat fmt, codec_t src) {

        if (src == v210) {
                if (fmt == AV_PIX_FMT_YUV420P10LE) {
                        return v210_to_yuv420p10le;
                } else if (fmt == AV_PIX_FMT_YUV422P10LE) {
                        return v210_to_yuv422p10le;
                } else if (fmt == AV_PIX_FMT_YUV444P10LE) {
                        return v210_to_yuv444p10le;
                } else {
                        abort();
                }
        }

        if (is422_8(fmt)) {
                return to_yuv422p;
        } else if (is420_8(fmt)) {
                if (fmt == AV_PIX_FMT_NV12)
                        return to_nv12;
                else
                        return to_yuv420p;
        } else if (is444_8(fmt)) {
                return to_yuv444p;
        } else {
                log_msg(LOG_LEVEL_FATAL, "[lavc] Unknown subsampling.\n");
                abort();
        }
}

struct my_task_data {
        void (*callback)(AVFrame *out_frame, unsigned char *in_data, int width, int height);
        AVFrame *out_frame;
        unsigned char *in_data;
        int width;
        int height;
};

void *my_task(void *arg);

void *my_task(void *arg) {
        struct my_task_data *data = (struct my_task_data *) arg;
        data->callback(data->out_frame, data->in_data, data->width, data->height);
        return NULL;
}

static shared_ptr<video_frame> libavcodec_compress_tile(struct module *mod, shared_ptr<video_frame> tx)
{
        struct state_video_compress_libav *s = (struct state_video_compress_libav *) mod->priv_data;
        static int frame_seq = 0;
        int ret;
        unsigned char *decoded;
        shared_ptr<video_frame> out{};

        libavcodec_check_messages(s);

        if(!video_desc_eq_excl_param(video_desc_from_frame(tx.get()),
                                s->saved_desc, PARAM_TILE_COUNT)) {
                cleanup(s);
                int ret = configure_with(s, video_desc_from_frame(tx.get()));
                if(!ret) {
                        return {};
                }
        }

        auto dispose = [](struct video_frame *frame) {
#if LIBAVCODEC_VERSION_MAJOR >= 54 && LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 37, 100)
                AVPacket *pkt = (AVPacket *) frame->callbacks.dispose_udata;
                av_packet_unref(pkt);
                free(pkt);
#else
                free(frame->tiles[0].data);
#endif // LIBAVCODEC_VERSION_MAJOR >= 54
                vf_free(frame);
        };
        out = shared_ptr<video_frame>(vf_alloc_desc(s->compressed_desc), dispose);
#if LIBAVCODEC_VERSION_MAJOR >= 54 && LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 37, 100)
        int got_output;
        AVPacket *pkt;
        pkt = (AVPacket *) malloc(sizeof(AVPacket));
        av_init_packet(pkt);
        pkt->data = NULL;
        pkt->size = 0;
        out->callbacks.dispose_udata = pkt;
#else
        out->tiles[0].data = (char *) malloc(s->compressed_desc.width *
                        s->compressed_desc.height * 4);
#endif // LIBAVCODEC_VERSION_MAJOR >= 54

        s->in_frame->pts = frame_seq++;

        if((void *) s->decoder != (void *) memcpy) {
                unsigned char *line1 = (unsigned char *) tx->tiles[0].data;
                unsigned char *line2 = (unsigned char *) s->decoded;
                int src_linesize = vc_get_linesize(tx->tiles[0].width, tx->color_spec);
                int dst_linesize = tx->tiles[0].width * 2; /* UYVY */
                for (int i = 0; i < (int) tx->tiles[0].height; ++i) {
                        s->decoder(line2, line1, dst_linesize,
                                        0, 8, 16);
                        line1 += src_linesize;
                        line2 += dst_linesize;
                }
                decoded = s->decoded;
        } else {
                decoded = (unsigned char *) tx->tiles[0].data;
        }

        {
                task_result_handle_t handle[s->params.cpu_count];
                struct my_task_data data[s->params.cpu_count];
                for(int i = 0; i < s->params.cpu_count; ++i) {
                        data[i].callback = select_pixfmt_callback(s->selected_pixfmt, s->decoded_codec);
                        data[i].out_frame = s->in_frame_part[i];

                        size_t height = tx->tiles[0].height / s->params.cpu_count;
                        // height needs to be even
                        height = height / 2 * 2;
                        if (i < s->params.cpu_count - 1) {
                                data[i].height = height;
                        } else { // we are last so we need to do the rest
                                data[i].height = tx->tiles[0].height -
                                        height * (s->params.cpu_count - 1);
                        }
                        data[i].width = tx->tiles[0].width;
                        data[i].in_data = decoded + i * height *
                                vc_get_linesize(tx->tiles[0].width, s->decoded_codec);

                        // run !
                        handle[i] = task_run_async(my_task, (void *) &data[i]);
                }

                for(int i = 0; i < s->params.cpu_count; ++i) {
                        wait_task(handle[i]);
                }
        }

        AVFrame *frame = s->in_frame;
#ifdef HWACC_VAAPI
        if(s->hwenc){
                av_hwframe_transfer_data(s->hwframe, s->in_frame, 0);
                frame = s->hwframe;
        }
#endif

        /* encode the image */
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
        out->tiles[0].data_len = 0;
        if (libav_codec_has_extradata(s->out_codec)) { // we need to store extradata for HuffYUV/FFV1 in the beginning
                out->tiles[0].data_len += sizeof(uint32_t) + s->codec_ctx->extradata_size;
                *(uint32_t *) out->tiles[0].data = s->codec_ctx->extradata_size;
                memcpy(out->tiles[0].data + sizeof(uint32_t), s->codec_ctx->extradata, s->codec_ctx->extradata_size);
        }

        ret = avcodec_send_frame(s->codec_ctx, frame);
        if (ret == 0) {
                AVPacket pkt;
                av_init_packet(&pkt);
                ret = avcodec_receive_packet(s->codec_ctx, &pkt);
                while (ret == 0) {
                        assert(pkt.size + out->tiles[0].data_len <= s->compressed_desc.width * s->compressed_desc.height * 4 - out->tiles[0].data_len);
                        memcpy((uint8_t *) out->tiles[0].data + out->tiles[0].data_len,
                                        pkt.data, pkt.size);
                        out->tiles[0].data_len += pkt.size;
                        av_packet_unref(&pkt);
                        ret = avcodec_receive_packet(s->codec_ctx, &pkt);
                }
                if (ret != AVERROR(EAGAIN) && ret != 0) {
                        print_libav_error(LOG_LEVEL_WARNING, "[lavc] Receive packet error", ret);
                }
        } else {
		print_libav_error(LOG_LEVEL_WARNING, "[lavc] Error encoding frame", ret);
                return {};
        }
#elif LIBAVCODEC_VERSION_MAJOR >= 54
        ret = avcodec_encode_video2(s->codec_ctx, pkt,
                        frame, &got_output);
        if (ret < 0) {
                log_msg(LOG_LEVEL_INFO, "Error encoding frame\n");
                return {};
        }

        if (got_output) {
                //printf("Write frame %3d (size=%5d)\n", frame_seq, s->pkt[buffer_idx].size);
                out->tiles[0].data = (char *) pkt->data;
                out->tiles[0].data_len = pkt->size;
        } else {
                return {};
        }
#else
        ret = avcodec_encode_video(s->codec_ctx, (uint8_t *) out->tiles[0].data,
                        out->tiles[0].width * out->tiles[0].height * 4,
                        frame);
        if (ret < 0) {
                log_msg(LOG_LEVEL_INFO, "Error encoding frame\n");
                return {};
        }

        if (ret) {
                //printf("Write frame %3d (size=%5d)\n", frame_seq, s->pkt[buffer_idx].size);
                out->tiles[0].data_len = ret;
        } else {
                return {};
        }
#endif // LIBAVCODEC_VERSION_MAJOR >= 54

        log_msg(LOG_LEVEL_DEBUG, "[lavc] Compressed frame size: %d\n", out->tiles[0].data_len);

        if (out->tiles[0].data_len == 0) { // videotoolbox returns sometimes frames with pkt->size == 0 but got_output == true
                return {};
        }

        return out;
}

static void cleanup(struct state_video_compress_libav *s)
{
        if(s->codec_ctx) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
		int ret;
		ret = avcodec_send_frame(s->codec_ctx, NULL);
		if (ret != 0) {
			log_msg(LOG_LEVEL_WARNING, "[lavc] Unexpected return value %d\n",
					ret);
		}
		do {
			AVPacket pkt;
			av_init_packet(&pkt);
			ret = avcodec_receive_packet(s->codec_ctx, &pkt);
			av_packet_unref(&pkt);
			if (ret != 0 && ret != AVERROR_EOF) {
				log_msg(LOG_LEVEL_WARNING, "[lavc] Unexpected return value %d\n",
						ret);
				break;
			}
		} while (ret != AVERROR_EOF);
#endif
                pthread_mutex_lock(s->lavcd_global_lock);
                avcodec_close(s->codec_ctx);
                avcodec_free_context(&s->codec_ctx);
                pthread_mutex_unlock(s->lavcd_global_lock);
                s->codec_ctx = NULL;
        }
        if(s->in_frame) {
                av_freep(s->in_frame->data);
                av_free(s->in_frame);
                s->in_frame = NULL;
        }
        free(s->decoded);
        s->decoded = NULL;

        if(s->hwframe){
                av_frame_free(&s->hwframe);
        }
}

static void libavcodec_compress_done(struct module *mod)
{
        struct state_video_compress_libav *s = (struct state_video_compress_libav *) mod->priv_data;

        cleanup(s);

        rm_release_shared_lock(LAVCD_LOCK_NAME);
        for(int i = 0; i < s->params.cpu_count; i++) {
                av_free(s->in_frame_part[i]);
        }
        free(s->in_frame_part);
        delete s;
}

static void setparam_default(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        if (codec_ctx->codec->id == AV_CODEC_ID_JPEG2000) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] J2K support is experimental and may be broken!\n");
        }
        if (!param->thread_mode.empty() && param->thread_mode != "no")  {
                if (param->thread_mode == "slice") {
                        // zero should mean count equal to the number of virtual cores
                        if (codec_ctx->codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) {
                                codec_ctx->thread_count = 0;
                                codec_ctx->thread_type = FF_THREAD_SLICE;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, "[lavc] Warning: Codec doesn't support slice-based multithreading.\n");
                        }
                } else if (param->thread_mode == "frame") {
                        if (codec_ctx->codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) {
                                codec_ctx->thread_count = 0;
                                codec_ctx->thread_type = FF_THREAD_FRAME;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, "[lavc] Warning: Codec doesn't support frame-based multithreading.\n");
                        }
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Warning: unknown thread mode: %s.\n", param->thread_mode.c_str());
                }
        }
}

static void setparam_jpeg(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        if (param->thread_mode == "slice") {
                // zero should mean count equal to the number of virtual cores
                codec_ctx->thread_count = 0;
                codec_ctx->thread_type = FF_THREAD_SLICE;
        }

        if (av_opt_set(codec_ctx->priv_data, "huffman", "default", 0) != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Warning: Cannot set default Huffman tables.\n");
        }
}

ADD_TO_PARAM(lavc_h264_interlaced_dct, "lavc-h264-interlaced-dct", "* lavc-h264-interlaced-dct\n"
                 "  Use interlaced DCT for H.264\n");
static void configure_x264_x265(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        const char *tune;
        if (codec_ctx->codec->id == AV_CODEC_ID_H264) {
                tune = "zerolatency,fastdecode";
        } else { // x265 supports only single tune parameter
                tune = "zerolatency";
        }
        int ret = av_opt_set(codec_ctx->priv_data, "tune", tune, 0);
        if (ret != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set tune %s.\n", tune);
        }

        // try to keep frame sizes as even as possible
        codec_ctx->rc_max_rate = codec_ctx->bit_rate;
        //codec_ctx->rc_min_rate = s->codec_ctx->bit_rate / 4 * 3;
        //codec_ctx->rc_buffer_aggressivity = 1.0;
        codec_ctx->rc_buffer_size = codec_ctx->rc_max_rate / param->fps * 8; // "emulate" CBR. Note that less than 8 frame sizes causes encoder buffer overflows and artifacts in stream.
        codec_ctx->qcompress = 0.0f;
        //codec_ctx->qblur = 0.0f;
        //codec_ctx->rc_min_vbv_overflow_use = 1.0f;
        //codec_ctx->rc_max_available_vbv_use = 1.0f;
        codec_ctx->qmin = 0;
        codec_ctx->qmax = 69;
        codec_ctx->max_qdiff = 69;
        //codec_ctx->rc_qsquish = 0;
        //codec_ctx->scenechange_threshold = 100;

        if (get_commandline_param("lavc-h264-interlaced-dct")) {
                // this options increases variance in frame sizes quite a lot
                if (param->interlaced) {
                        codec_ctx->flags |= AV_CODEC_FLAG_INTERLACED_DCT;
                }
        }
}

static void configure_qsv(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        int ret;
        ret = av_opt_set(codec_ctx->priv_data, "look_ahead", "0", 0);
        if (ret != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set unset look-ahead.\n");
        }
        if (!param->no_periodic_intra) {
                ret = av_opt_set(codec_ctx->priv_data, "int_ref_type", "vertical", 0);
                if (ret != 0) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set intra refresh.\n");
                }
#if 0
                ret = av_opt_set(codec_ctx->priv_data, "int_ref_cycle_size", "100", 0);
                if (ret != 0) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set intra refresh size.\n");
                }
#endif
        }
        codec_ctx->rc_max_rate = codec_ctx->bit_rate;
        // no look-ahead and rc_max_rate == bit_rate result in use of CBR for QSV
}

static void configure_nvenc(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        int ret;
        ret = av_opt_set(codec_ctx->priv_data, "rc", DEFAULT_NVENC_RC, 0);
        if (ret != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set RC.\n");
        }
        ret = av_opt_set(codec_ctx->priv_data, "spatial_aq", "0", 0);
        if (ret != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to unset spatial AQ.\n");
        }
        char gpu[3] = "";
        snprintf(gpu, 2, "%d", cuda_devices[0]);
        ret = av_opt_set(codec_ctx->priv_data, "gpu", gpu, 0);
        if (ret != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set GPU.\n");
        }
        ret = av_opt_set(codec_ctx->priv_data, "delay", "0", 0);
        if (ret != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set delay.\n");
        }
        ret = av_opt_set(codec_ctx->priv_data, "zerolatency", "1", 0);
        if (ret != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set zero latency operation (no reordering delay).\n");
        }
        codec_ctx->rc_max_rate = codec_ctx->bit_rate;
        codec_ctx->rc_buffer_size = codec_ctx->rc_max_rate / param->fps;
}


static void setparam_h264_h265(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        if (strcmp(codec_ctx->codec->name, "libx264") == 0 ||
                        strcmp(codec_ctx->codec->name, "libx265") == 0) {
                configure_x264_x265(codec_ctx, param);
        } else if (regex_match(codec_ctx->codec->name, regex(".*nvenc.*"))) {
                configure_nvenc(codec_ctx, param);
        } else if (strcmp(codec_ctx->codec->name, "h264_qsv") == 0 ||
                        strcmp(codec_ctx->codec->name, "hevc_qsv") == 0) {
                configure_qsv(codec_ctx, param);
        } else {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Warning: Unknown encoder %s. Using default configuration values.\n", codec_ctx->codec->name);
        }

        /// turn of periodic intra refresh for libx264/libx265 if requested
        if (!param->no_periodic_intra &&
                        (strcmp(codec_ctx->codec->name, "libx264") == 0 ||
                         strcmp(codec_ctx->codec->name, "libx265") == 0)) {
                codec_ctx->refs = 1;
                int ret = av_opt_set(codec_ctx->priv_data, "intra-refresh", "1", 0);
                if (ret != 0) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set Intra Refresh.\n");
                }
        }
}

static string get_h264_h265_preset(string const & enc_name, int width, int height, double fps)
{
        if (enc_name == "libx264") {
                if (width <= 1920 && height <= 1080 && fps <= 30) {
                        return string("veryfast");
                } else {
                        return string("ultrafast");
                }
        } else if (enc_name == "libx265") {
                return string("ultrafast");
        } else if (regex_match(enc_name, regex(".*nvenc.*"))) { // so far, there are at least nvenc, nvenc_h264 and h264_nvenc variants
                return string(DEFAULT_NVENC_PRESET);
        } else if (enc_name == "h264_qsv") {
                return string(DEFAULT_QSV_PRESET);
        } else {
                return {};
        }
}

static void setparam_vp8_vp9(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        codec_ctx->thread_count = param->cpu_count;
        codec_ctx->rc_buffer_size = codec_ctx->bit_rate / param->fps;
        //codec_ctx->rc_buffer_aggressivity = 0.5;
        if (av_opt_set(codec_ctx->priv_data, "deadline", "realtime", 0) != 0) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set deadline.\n");
        }

        if (av_opt_set(codec_ctx->priv_data, "cpu-used", "8", 0)) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set quality/speed ratio modifier.\n");
        }
}

static void libavcodec_check_messages(struct state_video_compress_libav *s)
{
        struct message *msg;
        while ((msg = check_message(&s->module_data))) {
                struct msg_change_compress_data *data =
                        (struct msg_change_compress_data *) msg;
                struct response *r;
                if (parse_fmt(s, data->config_string) == 0) {
                        log_msg(LOG_LEVEL_NOTICE, "[Libavcodec] Compression successfully changed.\n");
                        r = new_response(RESPONSE_OK, NULL);
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[Libavcodec] Unable to change compression!\n");
                        r = new_response(RESPONSE_INT_SERV_ERR, NULL);
                }
                memset(&s->saved_desc, 0, sizeof(s->saved_desc));
                free_message(msg, r);
        }

}

const struct video_compress_info libavcodec_info = {
        "libavcodec",
        libavcodec_compress_init,
        NULL,
        libavcodec_compress_tile,
        NULL,
        NULL,
        get_libavcodec_presets,
};

REGISTER_MODULE(libavcodec, &libavcodec_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace

