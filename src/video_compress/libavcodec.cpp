/**
 * @file   video_compress/libavcodec.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2022 CESNET, z. s. p. o.
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

#include <array>
#include <cassert>
#include <cmath>
#include <list>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <vector>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "libavcodec/lavc_common.h"
#include "libavcodec/lavc_video.h"
#include "libavcodec/to_lavc_vid_conv.h"
#include "messaging.h"
#include "module.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/text.h" // replace_all
#include "utils/parallel_conv.h"
#include "utils/worker.h"
#include "video.h"
#include "video_compress.h"

#ifdef HWACC_VAAPI
extern "C"
{
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vaapi.h>
}
#include "hwaccel_libav_common.h"
#endif

#ifdef HAVE_SWSCALE
extern "C"{
#include <libswscale/swscale.h>
}
#endif

#define MOD_NAME "[lavc] "

#ifndef AV_CODEC_CAP_OTHER_THREADS // compat - OTHER_THREADS is AUTO_THREADS in older FF
#define AV_CODEC_CAP_OTHER_THREADS AV_CODEC_CAP_AUTO_THREADS
#endif

using namespace std;
using namespace std::string_literals;

static constexpr const codec_t DEFAULT_CODEC = MJPG;
static constexpr double DEFAULT_X264_X265_CRF = 22.0;
static constexpr int DEFAULT_CQP = 21;
static constexpr const int DEFAULT_GOP_SIZE = 20;
static constexpr int DEFAULT_SLICE_COUNT = 32;
static constexpr string_view DONT_SET_PRESET = "dont_set_preset";

#define DEFAULT_X26X_RC_BUF_SIZE_FACTOR 2.5

namespace {

struct setparam_param {
        setparam_param(map<string, string> &lo) : lavc_opts(lo) {}
        struct video_desc desc {};
        bool have_preset = false;
        int periodic_intra = -1; ///< -1 default; 0 disable/not enable; 1 enable
        string thread_mode;
        int slices = -1;
        map<string, string> &lavc_opts; ///< user-supplied options from command-line
};

constexpr const char *DEFAULT_NVENC_PRESET = "p4";
constexpr const char *DEFAULT_NVENC_RC = "cbr";
constexpr const char *DEFAULT_NVENC_TUNE = "ull";
constexpr const char *FALLBACK_NVENC_PRESET = "llhq";

static constexpr const char *DEFAULT_QSV_PRESET = "medium";

typedef struct {
        function<const char*(bool)> get_prefered_encoder; ///< can be nullptr
        double avg_bpp;
        string (*get_preset)(string const & enc_name, int width, int height, double fps);
        void (*set_param)(AVCodecContext *, struct setparam_param *);
        int capabilities_priority;
} codec_params_t;

static string get_h264_h265_preset(string const & enc_name, int width, int height, double fps);
static string get_av1_preset(string const & enc_name, int width, int height, double fps);
static void libavcodec_check_messages(struct state_video_compress_libav *s);
static void libavcodec_compress_done(struct module *mod);
static void setparam_default(AVCodecContext *, struct setparam_param *);
static void setparam_h264_h265_av1(AVCodecContext *, struct setparam_param *);
static void setparam_jpeg(AVCodecContext *, struct setparam_param *);
static void setparam_vp8_vp9(AVCodecContext *, struct setparam_param *);
static void set_codec_thread_mode(AVCodecContext *codec_ctx, struct setparam_param *param);

static pixfmt_callback_t select_pixfmt_callback(AVPixelFormat fmt, codec_t src);
static void show_encoder_help(string const &name);
static void print_codec_supp_pix_fmts(const enum AVPixelFormat *first);
static void usage(void);
static int parse_fmt(struct state_video_compress_libav *s, char *fmt);
static void cleanup(struct state_video_compress_libav *s);

static map<codec_t, codec_params_t> codec_params = {
        { H264, codec_params_t{
                [](bool is_rgb) { return is_rgb ? "libx264rgb" : "libx264"; },
                0.07 * 2 /* for H.264: 1 - low motion, 2 - medium motion, 4 - high motion */
                * 2, // take into consideration that our H.264 is less effective due to specific preset/tune
                     // note - not used for libx264, which uses CRF by default
                get_h264_h265_preset,
                setparam_h264_h265_av1,
                100
        }},
        { H265, codec_params_t{
                [](bool) { return "libx265"; },
                0.04 * 2 * 2, // note - not used for libx265, which uses CRF by default
                get_h264_h265_preset,
                setparam_h264_h265_av1,
                101
        }},
        { MJPG, codec_params_t{
                nullptr,
                1.2,
                nullptr,
                setparam_jpeg,
                102
        }},
        { J2K, codec_params_t{
                nullptr,
                1.0,
                nullptr,
                setparam_default,
                500
        }},
        { VP8, codec_params_t{
                nullptr,
                0.4,
                nullptr,
                setparam_vp8_vp9,
                103
        }},
        { VP9, codec_params_t{
                nullptr,
                0.4,
                nullptr,
                setparam_vp8_vp9,
                104
        }},
        { HFYU, codec_params_t{
                nullptr,
                0,
                nullptr,
                setparam_default,
                501
        }},
        { FFV1, codec_params_t{
                nullptr,
                0,
                nullptr,
                setparam_default,
                502
        }},
        { AV1, codec_params_t{
                [](bool) { return "libsvtav1"; },
                0.1,
                get_av1_preset,
                setparam_h264_h265_av1,
                600
        }},
        { PRORES, codec_params_t{
                nullptr,
                0.5,
                nullptr,
                setparam_default,
                300,
        }},
};

struct state_video_compress_libav {
        state_video_compress_libav(struct module *parent) {
                module_init_default(&module_data);
                module_data.cls = MODULE_CLASS_DATA;
                module_data.priv_data = this;
                module_data.deleter = libavcodec_compress_done;
                module_register(&module_data, parent);
        }
        ~state_video_compress_libav() {
                av_packet_free(&pkt);
        }

        struct module       module_data;

        struct video_desc   saved_desc{};

        AVFrame            *in_frame = nullptr;
        AVPacket           *pkt = av_packet_alloc();
        // for every core - parts of the above
        vector<AVFrame *>   in_frame_part;
        AVCodecContext     *codec_ctx = nullptr;

        unsigned char      *decoded = nullptr; ///< intermediate representation for codecs
                                     ///< that are not directly supported
        codec_t             decoded_codec = VIDEO_CODEC_NONE;
        decoder_t           decoder = nullptr;

        codec_t             requested_codec_id = VIDEO_CODEC_NONE;
        long long int       requested_bitrate = 0;
        double              requested_bpp = 0;
        double              requested_crf = -1;
        int                 requested_cqp = -1;
        int                 requested_q = -1;
        // may be 422, 420 or 0 (no subsampling explicitly requested
        int                 requested_subsampling = 0;
        // contains format that is supplied by UG to the encoder or swscale (if used)
        AVPixelFormat       selected_pixfmt = AV_PIX_FMT_NONE;

        codec_t             out_codec = VIDEO_CODEC_NONE;

        struct video_desc compressed_desc{};

        struct setparam_param params{lavc_opts};
        string              backend;
        int                 requested_gop = DEFAULT_GOP_SIZE;

        map<string, string> lavc_opts; ///< user-supplied options from command-line

        bool hwenc = false;
        AVFrame *hwframe = nullptr;

#ifdef HAVE_SWSCALE
        struct SwsContext *sws_ctx = nullptr;
        AVPixelFormat sws_out_pixfmt = AV_PIX_FMT_NONE;
        AVFrame *sws_frame = nullptr;
#endif

        int conv_thread_count = clamp<unsigned int>(thread::hardware_concurrency(), 1, INT_MAX); ///< number of threads used for UG conversions
        double mov_avg_comp_duration = 0;
        long mov_avg_frames = 0;
};

struct codec_encoders_decoders{
        std::vector<std::string> encoders;
        std::vector<std::string> decoders;
};

static codec_encoders_decoders get_codec_encoders_decoders(AVCodecID id){
        codec_encoders_decoders res;
#if LIBAVCODEC_VERSION_INT > AV_VERSION_INT(58, 9, 100)
        const AVCodec *codec = nullptr;
        void *i = 0;
        while ((codec = av_codec_iterate(&i))) {
                if (av_codec_is_encoder(codec) && codec->id == id) {
                        res.encoders.emplace_back(codec->name);
                }
                if (av_codec_is_decoder(codec) && codec->id == id) {
                        res.decoders.emplace_back(codec->name);
                }
        }
#elif LIBAVCODEC_VERSION_MAJOR >= 54
        const AVCodec *codec = nullptr;
        if ((codec = avcodec_find_encoder(id))) {
                do {
                        if (av_codec_is_encoder(codec) && codec->id == id) {
                                res.encoders.emplace_back(codec->name);
                        }
                } while ((codec = av_codec_next(codec)));
        }

        if ((codec = avcodec_find_decoder(id))) {
                do {
                        if (av_codec_is_decoder(codec) && codec->id == id) {
                                res.decoders.emplace_back(codec->name);
                        }
                } while ((codec = av_codec_next(codec)));
        }
#else
        UNUSED(id);
#endif

        return res;
}

static void print_codec_info(AVCodecID id, char *buf, size_t buflen)
{
        auto info = get_codec_encoders_decoders(id);
        assert(buflen > 0);
        buf[0] = '\0';
        if(info.encoders.empty() && info.decoders.empty())
                return;

        strncat(buf, " (", buflen - strlen(buf) - 1);
        if (!info.encoders.empty()) {
                strncat(buf, "encoders:", buflen - strlen(buf) - 1);
                for(const auto& enc : info.encoders){
                        strncat(buf, " ", buflen - strlen(buf) - 1);
                        strncat(buf, enc.c_str(), buflen - strlen(buf) - 1);
                }
        }
        if (!info.decoders.empty()) {
                if (!info.encoders.empty()) {
                        strncat(buf, ", ", buflen - strlen(buf) - 1);
                }
                strncat(buf, "decoders:", buflen - strlen(buf) - 1);

                for(const auto& dec : info.decoders){
                        strncat(buf, " ", buflen - strlen(buf) - 1);
                        strncat(buf, dec.c_str(), buflen - strlen(buf) - 1);
                }
        }
        strncat(buf, ")", buflen - strlen(buf) - 1);
}

static void usage() {
        printf("Libavcodec encoder usage:\n");
        col() << "\t" SBOLD(SRED("-c libavcodec") << "[:codec=<codec_name>|:encoder=<encoder>][:bitrate=<bits_per_sec>|:bpp=<bits_per_pixel>][:crf=<crf>|:cqp=<cqp>][q=<q>]"
                        "[:subsampling=<subsampling>][:gop=<gop>]"
                        "[:[disable_]intra_refresh][:threads=<threads>][:slices=<slices>][:<lavc_opt>=<val>]*") << "\n";
        col() << "\nwhere\n";
        col() << "\t" << SBOLD("<encoder>") << " specifies encoder (eg. nvenc or libx264 for H.264)\n";
        col() << "\t" << SBOLD("<codec_name>") << " may be specified codec name (default MJPEG), supported codecs:\n";
        for (auto && param : codec_params) {
                enum AVCodecID avID = get_ug_to_av_codec(param.first);
                if (avID == AV_CODEC_ID_NONE) { // old FFMPEG -> codec id is flushed to 0 in compat
                        continue;
                }
                char avail[1024];
                const AVCodec *codec;
                if ((codec = avcodec_find_encoder(avID))) {
                        strcpy(avail, "available");
                } else {
                        strcpy(avail, "not available");
                }
                print_codec_info(avID, avail + strlen(avail), sizeof avail - strlen(avail));

                col() << "\t\t" << SBOLD(get_codec_name(param.first)) << " - " << avail << "\n";

        }
        col() << "\t" << SBOLD("[disable_]intra_refresh") << " - (do not) use Periodic Intra Refresh (H.264/H.265)\n";
        col() << "\t" << SBOLD("<bits_per_sec>") << " specifies requested bitrate\n"
                << "\t\t\t0 means codec default (same as when parameter omitted)\n";
        col() << "\t" << SBOLD("<bits_per_pixel>") << " specifies requested bitrate using compressed bits per pixel\n"
                << "\t\t\tbitrate = frame width * frame height * bits_per_pixel * fps\n";
        col() << "\t" << SBOLD("<cqp>") << " use constant QP value\n";
        col() << "\t" << SBOLD("<crf>") << " specifies CRF factor (only for libx264/libx265)\n";
        col() << "\t" << SBOLD("<q>") << " quality (qmin, qmax) - range usually from 0 (best) to 50-100 (worst)\n";
        col() << "\t" << SBOLD("<subsampling") << "> may be one of 444, 422, or 420, default 420 for progresive, 422 for interlaced\n";
        col() << "\t" << SBOLD("<threads>") << " can be \"no\", or \"<number>[F][S][n]\" where 'F'/'S' indicate if frame/slice thr. should be used, both can be used (default slice), 'n' means none;\n";
        col() << "\t" <<       "         "  << " use a comma to add also number of conversion threads (eg. \"0S,8\"), default: number of logical cores\n";
        col() << "\t" << SBOLD("<slices>") << " number of slices to use (default: " << DEFAULT_SLICE_COUNT << ")\n";
        col() << "\t" << SBOLD("<gop>") << " specifies GOP size\n";
        col() << "\t" << SBOLD("<lavc_opt>") << " arbitrary option to be passed directly to libavcodec (eg. preset=veryfast), eventual colons must be backslash-escaped (eg. for x264opts)\n";
        col() << "\nUse '" << SBOLD("-c libavcodec:encoder=<enc>:help") << "' to display encoder specific options, works on decoders as well (also use keyword \"encoder\").\n";
        col() << "\n";
        col() << "Libavcodec version (linked): " << SBOLD(LIBAVCODEC_IDENT) << "\n";
        const char *swscale = "no";
#ifdef HAVE_SWSCALE
        swscale = "yes";
#endif
        col() << "Libswscale supported: " << SBOLD(swscale) << "\n";
}

static int parse_fmt(struct state_video_compress_libav *s, char *fmt) {
        if (!fmt) {
                return 0;
        }

        bool show_help = false;

        // replace all '\:' with 2xDEL
        replace_all(fmt, ESCAPED_COLON, DELDEL);
        char *item, *save_ptr = NULL;

        while ((item = strtok_r(fmt, ":", &save_ptr)) != NULL) {
                if(strncasecmp("help", item, strlen("help")) == 0) {
                        show_help = true;
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
                        s->requested_bpp = unit_evaluate_dbl(bpp_str, false);
                        if (std::isnan(s->requested_bpp)) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Wrong bitrate: " << bpp_str << "\n";
                                return -1;
                        }
                } else if(strncasecmp("crf=", item, strlen("crf=")) == 0) {
                        char *crf_str = item + strlen("crf=");
                        s->requested_crf = atof(crf_str);
                } else if(strncasecmp("cqp=", item, strlen("cqp=")) == 0) {
                        char *cqp_str = item + strlen("cqp=");
                        s->requested_cqp = atoi(cqp_str);
                } else if(strncasecmp("q=", item, strlen("q=")) == 0) {
                        char *q_str = strchr(item, '=') + 1;
                        s->requested_q = stoi(q_str);
                } else if(strncasecmp("subsampling=", item, strlen("subsampling=")) == 0) {
                        char *subsample_str = item + strlen("subsampling=");
                        s->requested_subsampling = atoi(subsample_str);
                        if (s->requested_subsampling != 444 &&
                                        s->requested_subsampling != 422 &&
                                        s->requested_subsampling != 420) {
                                log_msg(LOG_LEVEL_ERROR, "[lavc] Supported subsampling is 444, 422, or 420.\n");
                                return -1;
                        }
                } else if (strstr(item, "intra_refresh") != nullptr) {
                        s->params.periodic_intra = strstr(item, "disable_") == item ? 0 : 1;
                } else if(strncasecmp("threads=", item, strlen("threads=")) == 0) {
                        char *threads = item + strlen("threads=");
                        if (strchr(threads, ',')) {
                                s->conv_thread_count = stoi(strchr(threads, ',') + 1);
                                *strchr(threads, ',') = '\0';
                        }
                        s->params.thread_mode = threads;
                } else if(strncasecmp("slices=", item, strlen("slices=")) == 0) {
                        char *slices = strchr(item, '=') + 1;
                        s->params.slices = stoi(slices);
                } else if(strncasecmp("encoder=", item, strlen("encoder=")) == 0) {
                        char *backend = item + strlen("encoder=");
                        s->backend = backend;
                } else if(strncasecmp("gop=", item, strlen("gop=")) == 0) {
                        char *gop = item + strlen("gop=");
                        s->requested_gop = atoi(gop);
                } else if (strchr(item, '=')) {
                        char *c_val_dup = strdup(strchr(item, '=') + 1);
                        replace_all(c_val_dup, DELDEL, ":");
                        string key, val;
                        key = string(item, strchr(item, '='));
                        s->lavc_opts[key] = c_val_dup;
                        free(c_val_dup);
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Error: unknown option %s.\n",
                                        item);
                        return -1;
                }
                fmt = NULL;
        }

        if (show_help) {
                if (s->backend.empty()) {
                        usage();
                } else {
                        show_encoder_help(s->backend);
                }
        }

        if ((get_commandline_param("lavc-use-codec") != nullptr && "help"s == get_commandline_param("lavc-use-codec")) ||
                        (show_help && !s->backend.empty())) {
                auto *codec = avcodec_find_encoder_by_name(s->backend.c_str());
                if (codec != nullptr) {
                        cout << "\n";
                        print_codec_supp_pix_fmts(codec->pix_fmts);
                } else {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot open encoder: " << s->backend << "\n";
                }
        }

        if (show_help || (get_commandline_param("lavc-use-codec") != nullptr && "help"s == get_commandline_param("lavc-use-codec"))) {
                return 1;
        }

        return 0;
}

static compress_module_info get_libavcodec_module_info(){
        compress_module_info module_info;
        module_info.name = "libavcodec";
        module_info.opts.emplace_back(module_option{"Bitrate", "Bitrate", "quality", ":bitrate=", false});
        module_info.opts.emplace_back(module_option{"Crf", "specifies CRF factor (only for libx264/libx265)", "crf", ":crf=", false});
        module_info.opts.emplace_back(module_option{"Disable intra refresh",
                        "Do not use Periodic Intra Refresh (H.264/H.265)",
                        "disable_intra_refresh", ":disable_intra_refresh", true});
        module_info.opts.emplace_back(module_option{"Subsampling",
                        "may be one of 444, 422, or 420, default 420 for progresive, 422 for interlaced",
                        "subsampling", ":subsampling=", false});
        module_info.opts.emplace_back(module_option{"Lavc opt",
                        "arbitrary option to be passed directly to libavcodec (eg. preset=veryfast), eventual colons must be backslash-escaped (eg. for x264opts)",
                        "lavc_opt", ":", false});

        for (const auto& param : codec_params) {
                enum AVCodecID avID = get_ug_to_av_codec(param.first);
                if (avID == AV_CODEC_ID_NONE) { // old FFMPEG -> codec id is flushed to 0 in compat
                        continue;
                }
                const AVCodec *i;
                if (!(i = avcodec_find_encoder(avID))) {
                        continue;
                }

                codec codec_info;
                codec_info.name = get_codec_name(param.first);
                codec_info.priority = param.second.capabilities_priority;
                codec_info.encoders.emplace_back(
                                encoder{"default", ":codec=" + codec_info.name});

                auto coders = get_codec_encoders_decoders(avID);
                for(const auto& enc : coders.encoders){
                        codec_info.encoders.emplace_back(
                                        encoder{enc, ":encoder=" + enc});
                }

                module_info.codecs.emplace_back(std::move(codec_info));
        }

        return module_info;
}

struct module * libavcodec_compress_init(struct module *parent, const char *opts)
{
        ug_set_av_logging();
#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        /*  register all the codecs (you can also register only the codec
         *         you wish to have smaller code */
        avcodec_register_all();
#endif

        char *fmt = strdup(opts);
        struct state_video_compress_libav *s = new state_video_compress_libav(parent);
        int ret = parse_fmt(s, fmt);
        free(fmt);
        if (ret != 0) {
                module_done(&s->module_data);
                return ret > 0 ? static_cast<module*>(INIT_NOERR) : NULL;
        }

        s->in_frame_part.resize(s->conv_thread_count);
        for(int i = 0; i < s->conv_thread_count; i++) {
                s->in_frame_part[i] = av_frame_alloc();
        }

        return &s->module_data;
}

#ifdef HWACC_VAAPI
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
                        s->width,
                        s->height,
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

void print_codec_supp_pix_fmts(const enum AVPixelFormat *first) {
        string out;
        if (first == nullptr) {
                out += " (none)";
        }
        const enum AVPixelFormat *it = first;
        while (it != nullptr && *it != AV_PIX_FMT_NONE) {
                out += " "s + av_get_pix_fmt_name(*it++);
        }
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "Codec supported pixel formats:" << SBOLD(out) << "\n";
}

void print_pix_fmts(const list<enum AVPixelFormat>
                &req_pix_fmts, const enum AVPixelFormat *first) {
        print_codec_supp_pix_fmts(first);
        string out;
        for (auto &c : req_pix_fmts) {
                out += " "s + av_get_pix_fmt_name(c);
        }
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "Supported pixel formats:" << SBOLD(out) << "\n";
}

/**
 * Finds best pixel format
 *
 * Iterates over formats in req_pix_fmts and tries to find the same format in
 * second list, codec_pix_fmts. If found, returns that format. Efectivelly
 * select first match of item from first list in second list.
 *
 * @note
 * Unusable pixel formats and a currently selected one are removed from
 * req_pix_fmts.
 */
static enum AVPixelFormat get_first_matching_pix_fmt(list<enum AVPixelFormat>::const_iterator &it,
                list<enum AVPixelFormat>::const_iterator end,
                const enum AVPixelFormat *codec_pix_fmts)
{
        if(codec_pix_fmts == NULL)
                return AV_PIX_FMT_NONE;

        for ( ; it != end; it++) {
                const enum AVPixelFormat *tmp = codec_pix_fmts;
                enum AVPixelFormat fmt;
                while((fmt = *tmp++) != AV_PIX_FMT_NONE) {
                        if (fmt == *it) {
                                enum AVPixelFormat ret = *it++;
                                return ret;
                        }
                }
        }

        return AV_PIX_FMT_NONE;
}

template<typename T>
static inline bool check_av_opt_set(void *priv_data, const char *key, T val, const char *desc = nullptr) {
        int ret = 0;
        string val_str;
        if constexpr (std::is_same_v<T, int>) {
                ret = av_opt_set_int(priv_data, key, val, 0);
                val_str = to_string(val);
        } else if constexpr (std::is_same_v<T, double>) {
                ret = av_opt_set_double(priv_data, key, val, 0);
                val_str = to_string(val);
        } else if constexpr (std::is_same_v<T, const char *>) {
                ret = av_opt_set(priv_data, key, val, 0);
                val_str = val;
        } else {
                static_assert(!std::is_same_v<T, T>, "unsupported type");
        }
        if (ret != 0) {
                string err = string(MOD_NAME) + "Unable to set " + (desc ? desc : key) + " to " + val_str;
                print_libav_error(LOG_LEVEL_WARNING, err.c_str(), ret);
        }
        return ret == 0;
}

bool set_codec_ctx_params(struct state_video_compress_libav *s, AVPixelFormat pix_fmt, struct video_desc desc, codec_t ug_codec)
{
        bool is_x264_x265 = strstr(s->codec_ctx->codec->name, "libx26") == s->codec_ctx->codec->name;
        bool is_vaapi = regex_match(s->codec_ctx->codec->name, regex(".*_vaapi"));

        double avg_bpp; // average bit per pixel
        avg_bpp = s->requested_bpp > 0.0 ? s->requested_bpp :
                codec_params[ug_codec].avg_bpp;

        int_fast64_t bitrate = s->requested_bitrate > 0 ? s->requested_bitrate :
                desc.width * desc.height * avg_bpp * desc.fps;

        s->params.have_preset = s->lavc_opts.find("preset") != s->lavc_opts.end();

        s->codec_ctx->strict_std_compliance = -2;

        // set quality
        if (s->requested_cqp >= 0 || (is_vaapi && s->requested_crf == -1.0 && s->requested_bitrate == 0 && s->requested_bpp == 0.0)) {
                int cqp = s->requested_cqp >= 0 ? s->requested_cqp : DEFAULT_CQP;
                if (check_av_opt_set<int>(s->codec_ctx->priv_data, "qp", cqp, "CQP")) {
                        LOG(LOG_LEVEL_INFO) << MOD_NAME "Setting CQP to " << cqp <<  "\n";
                }
        } else if (s->requested_crf >= 0.0 || (is_x264_x265 && s->requested_bitrate == 0 && s->requested_bpp == 0.0)) {
                double crf = s->requested_crf >= 0.0 ? s->requested_crf : DEFAULT_X264_X265_CRF;
                if (check_av_opt_set<double>(s->codec_ctx->priv_data, "crf", crf)) {
                        log_msg(LOG_LEVEL_INFO, "[lavc] Setting CRF to %.2f.\n", crf);
                }
        } else {
                s->codec_ctx->bit_rate = bitrate;
                s->codec_ctx->bit_rate_tolerance = bitrate / desc.fps * 6;
                LOG(LOG_LEVEL_INFO) << MOD_NAME << "Setting bitrate to " << format_in_si_units(bitrate) << "bps.\n";
        }

        if (s->requested_q != -1) {
                s->codec_ctx->qmin = s->codec_ctx->qmax = s->requested_q;
        }

        /* resolution must be a multiple of two */
        s->codec_ctx->width = desc.width;
        s->codec_ctx->height = desc.height;
        /* frames per second */
        s->codec_ctx->time_base = (AVRational){1,(int) desc.fps};
        s->codec_ctx->gop_size = s->requested_gop;
        s->codec_ctx->max_b_frames = 0;

        s->codec_ctx->pix_fmt = pix_fmt;
        s->codec_ctx->bits_per_raw_sample = min<int>(get_bits_per_component(ug_codec), av_pix_fmt_desc_get(pix_fmt)->comp[0].depth);

        codec_params[ug_codec].set_param(s->codec_ctx, &s->params);
        set_codec_thread_mode(s->codec_ctx, &s->params);
        // currently FFmpeg JPEG encoder produces broken JPEGs if not using encoding threads and slices > 1
        s->codec_ctx->slices = IF_NOT_UNDEF_ELSE(s->params.slices, s->codec_ctx->codec_id == AV_CODEC_ID_MJPEG && s->codec_ctx->thread_count <= 1 ? 1 : DEFAULT_SLICE_COUNT);

        if (!s->params.have_preset) {
                string preset{};
                if (codec_params[ug_codec].get_preset) {
                        preset = codec_params[ug_codec].get_preset(s->codec_ctx->codec->name, desc.width, desc.height, desc.fps);
                }

                if (!preset.empty() && preset != DONT_SET_PRESET) {
                        if (check_av_opt_set<const char *>(s->codec_ctx->priv_data, "preset", preset.c_str())) {
                                LOG(LOG_LEVEL_INFO) << "[lavc] Setting preset to " << preset <<  ".\n";
                        }
                }
                if (preset.empty()) {
                        LOG(LOG_LEVEL_WARNING) << "[lavc] Warning: Unable to find suitable preset for encoder " << s->codec_ctx->codec->name << ".\n";
                }
        }

        // set user supplied parameters
        for (auto const &item : s->lavc_opts) {
                if(av_opt_set(s->codec_ctx->priv_data, item.first.c_str(), item.second.c_str(), 0) != 0) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Error: Unable to set '%s' to '%s'. Check command-line options.\n", item.first.c_str(), item.second.c_str());
                        return false;
                }
        }

        return true;
}

auto get_intermediate_codecs_from_uv_to_av(codec_t in, AVPixelFormat av) {
        set<codec_t> intermediate_codecs; // using set to avoid duplicities
        for (const auto *i = get_av_to_ug_pixfmts(); i->uv_codec != VIDEO_CODEC_NONE; ++i) { // no AV conversion needed - direct mapping
                auto decoder = get_decoder_from_to(in, i->uv_codec);
                if (decoder != nullptr && i->av_pixfmt == av) {
                        intermediate_codecs.insert(i->uv_codec);
                }
        }
        for (const auto *c = get_uv_to_av_conversions(); c->src != VIDEO_CODEC_NONE; ++c) { // AV conversion needed
                auto decoder = get_decoder_from_to(in, c->src);
                if (decoder != nullptr && c->dst == av) {
                        intermediate_codecs.insert(c->src);
                }
        }

        return vector<codec_t>(intermediate_codecs.begin(), intermediate_codecs.end());
}

/**
 * Returns a UltraGrid decoder needed to decode from the UltraGrid codec in
 * to out with respect to conversions in @ref conversions. Therefore it should
 * be feasible to convert in to out and then convert out to av (last step may
 * be omitted if the format is native for both indicated in
 * ug_to_av_pixfmt_map).
 */
decoder_t get_decoder_from_uv_to_uv(codec_t in, AVPixelFormat av, codec_t *out) {
        vector<codec_t> intermediate_codecs = get_intermediate_codecs_from_uv_to_av(in, av);
        if (intermediate_codecs.empty()) {
                return nullptr;
        }

        // select intermediate UG codec same or better in following order of
        // importance: 1) depth, 2) subsampling, 3) color space
        sort(intermediate_codecs.begin(), intermediate_codecs.end(), [&](codec_t a, codec_t b) {
                int depth_in = get_bits_per_component(in);
                int depth_a = get_bits_per_component(a);
                int depth_b = get_bits_per_component(b);
                bool rgb_in = codec_is_a_rgb(in);
                bool rgb_a = codec_is_a_rgb(a);
                bool rgb_b = codec_is_a_rgb(b);
                int subs_in = get_subsampling(in);
                int subs_a = get_subsampling(a);
                int subs_b = get_subsampling(b);
                // check identity first
                if (a == in || b == in) {
                        return a == in;
                }
                // either a or b is narrower than depth_in - sort higher bit depth first
                if (depth_a != depth_b &&
                                (depth_a < depth_in || depth_b < depth_in)) {
                        return depth_a > depth_b;
                }
                if (subs_a != subs_b &&
                                (subs_a < subs_in || subs_b < subs_in)) {
                        return subs_a > subs_b;
                }
                if (rgb_a != rgb_b) {
                        return rgb_a == rgb_in;
                }

                // now all rgb/depth/subs pairs are either the same or both better than in
                assert((depth_a == depth_b || (depth_a >= depth_in && depth_b >= depth_in)) && (subs_a == subs_b || (subs_a >= subs_in && subs_b >= subs_in)));
                if (depth_a != depth_b) {
                        return depth_a < depth_b;
                }
                if (subs_a != subs_b) {
                        return subs_a < subs_b;
                }

                codec_t comp_codecs[] = { a, b, VIDEO_CODEC_NONE };
                codec_t out = VIDEO_CODEC_NONE;
                get_fastest_decoder_from(in, comp_codecs, &out);
                if (out) {
                        return out == a;
                }

                return a < b;
        });

        *out = intermediate_codecs[0];
        return get_decoder_from_to(in, *out);
}

static int get_subsampling(enum AVPixelFormat fmt) {
        const struct AVPixFmtDescriptor *pd = av_pix_fmt_desc_get(fmt);
        if (pd->log2_chroma_w == 0 && pd->log2_chroma_h == 0) {
                return 444;
        }
        if (pd->log2_chroma_w == 1 && pd->log2_chroma_h == 0) {
                return 422;
        }
        if (pd->log2_chroma_w == 1 && pd->log2_chroma_h == 1) {
                return 420;
        }
        return 0; // other (todo)
}

/**
 * Returns list of pix_fmts that UltraGrid can supply to the encoder.
 * The list is ordered according to input description and requested subsampling.
 */
static list<enum AVPixelFormat> get_available_pix_fmts(struct video_desc in_desc,
                int requested_subsampling, codec_t force_conv_to)
{
        list<enum AVPixelFormat> fmts;

#ifdef HWACC_VAAPI
        fmts.push_back(AV_PIX_FMT_VAAPI);
#endif

        // add the format itself if it matches the ultragrid one
        if (get_ug_to_av_pixfmt(in_desc.color_spec) != AV_PIX_FMT_NONE) {
                if (force_conv_to == VIDEO_CODEC_NONE || force_conv_to == in_desc.color_spec) {
                        fmts.push_back(get_ug_to_av_pixfmt(in_desc.color_spec));
                }
        }

        int bits_per_comp = get_bits_per_component(in_desc.color_spec);
        bool is_rgb = codec_is_a_rgb(in_desc.color_spec);
        int subsampling = IF_NOT_NULL_ELSE(requested_subsampling, get_subsampling(in_desc.color_spec) / 10);
        // sort
        auto compare = [bits_per_comp, is_rgb, subsampling](enum AVPixelFormat a, enum AVPixelFormat b) {
                const struct AVPixFmtDescriptor *pda = av_pix_fmt_desc_get(a);
                const struct AVPixFmtDescriptor *pdb = av_pix_fmt_desc_get(b);
#if LIBAVUTIL_VERSION_MAJOR >= 56
                int deptha = pda->comp[0].depth;
                int depthb = pdb->comp[0].depth;
#else
                int deptha = pda->comp[0].depth_minus1 + 1;
                int depthb = pdb->comp[0].depth_minus1 + 1;
#endif
#if defined(AV_PIX_FMT_FLAG_RGB)
                bool rgba = pda->flags & AV_PIX_FMT_FLAG_RGB;
                bool rgbb = pdb->flags & AV_PIX_FMT_FLAG_RGB;
#else
                bool rgba = pda->flags & PIX_FMT_RGB;
                bool rgbb = pdb->flags & PIX_FMT_RGB;
#endif
                int subsa = get_subsampling(a);
                int subsb = get_subsampling(b);

                if (rgba != rgbb) {
                        if (rgba == is_rgb) {
                                return true;
                        }
                        return false;
                }
                if (deptha != depthb) {
                        // either a or b is lower than bits_per_comp - sort higher bit depth first
                        if (deptha < bits_per_comp || depthb < bits_per_comp) {
                                return deptha > depthb;
                        }
                        // both are equal or higher - sort lower bit depth first
                        return deptha < depthb;
                }
                if (subsa != subsb) {
                        if (subsa < subsampling || subsb < subsampling) {
                                return subsa > subsb;
                        }
                        return subsa < subsb;
                }
                return a < b;
        };

        set<enum AVPixelFormat, decltype(compare)> available_formats(compare); // those for that there exitst a conversion and respect requested subsampling (if given)
        for (const auto *i = get_av_to_ug_pixfmts(); i->uv_codec != VIDEO_CODEC_NONE; ++i) { // no conversion needed - direct mapping
                if (get_decoder_from_to(in_desc.color_spec, i->uv_codec)) {
                        int codec_subsampling = get_subsampling(i->av_pixfmt);
                        if ((requested_subsampling == 0 ||
                                        requested_subsampling == codec_subsampling) &&
                                       (force_conv_to == VIDEO_CODEC_NONE || force_conv_to == i->uv_codec)) {
                                available_formats.insert(i->av_pixfmt);
                        }
                }
        }
        for (const auto *c = get_uv_to_av_conversions(); c->src != VIDEO_CODEC_NONE; c++) { // conversion needed
                if (c->src == in_desc.color_spec ||
                                get_decoder_from_to(in_desc.color_spec, c->src)) {
                        int codec_subsampling = get_subsampling(c->dst);
                        if ((requested_subsampling == 0 ||
                                        requested_subsampling == codec_subsampling) &&
                                       (force_conv_to == VIDEO_CODEC_NONE || force_conv_to == c->src)) {
                                available_formats.insert(c->dst);
                        }
                }
        }

        copy(available_formats.begin(), available_formats.end(), back_inserter(fmts));

        return fmts;

}

ADD_TO_PARAM("lavc-use-codec",
                "* lavc-use-codec=<c>\n"
                "  Restrict codec to use user specified pixel fmt. Use either FFmpeg name\n"
                "  (eg. nv12, yuv422p10le or yuv444p10le) or UltraGrid pixel formats names\n"
                "  (v210, R10k, UYVY etc.). See wiki for more info.\n");
/**
 * Returns ordered list of codec preferences for input description and
 * requested_subsampling.
 */
list<enum AVPixelFormat> get_requested_pix_fmts(struct video_desc in_desc,
                int requested_subsampling) {
        codec_t force_conv_to = VIDEO_CODEC_NONE; // if non-zero, use only this codec as a target
                                                  // of UG conversions (before FFMPEG conversion)
                                                  // or (likely) no conversion at all
        if (get_commandline_param("lavc-use-codec")) {
                const char *val = get_commandline_param("lavc-use-codec");
                enum AVPixelFormat fmt = av_get_pix_fmt(val);
                if (fmt != AV_PIX_FMT_NONE) {
                        return { fmt };
                }
                force_conv_to = get_codec_from_name(val);
                if (!force_conv_to) {
                        LOG(LOG_LEVEL_FATAL) << MOD_NAME << "Wrong codec string: " << val << ".\n";
                        exit_uv(1);
                        return {};
                }
        }

        return get_available_pix_fmts(in_desc, requested_subsampling, force_conv_to);
}

void apply_blacklist([[maybe_unused]] list<enum AVPixelFormat> &formats, [[maybe_unused]] const char *encoder_name) {
#if X2RGB10LE_PRESENT
        // blacklist AV_PIX_FMT_X2RGB10LE for NVENC - with current FFmpeg (13d04e3), it produces
        // 10-bit 4:2:0 YUV (FF macro IS_YUV444 and IS_GBRP should contain the codec - if set 1st
        // one, picture is ok, second produces incorrect colors)
        // Incorrect colors are produced also for qsv_hevc
        if (strstr(encoder_name, "nvenc") != nullptr || strstr(encoder_name, "hevc_qsv") != nullptr) {
                if (formats.size() == 1) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Only one codec remaining, not blacklisting!\n";
                        return;
                }
                if (auto it = std::find(formats.begin(), formats.end(), AV_PIX_FMT_X2RGB10LE); it != formats.end()) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Blacklisting x2rgb10le because there has been issues with this pixfmt "
                                "and current encoder (" << encoder_name << ") , use '--param lavc-use-codec=x2rgb10le' to enforce.\n";
                        formats.erase(it);
                }
        }
#endif
}

static bool try_open_codec(struct state_video_compress_libav *s,
                           AVPixelFormat &pix_fmt,
                           struct video_desc desc,
                           codec_t ug_codec,
                           const AVCodec *codec)
{
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
                        avcodec_free_context(&s->codec_ctx);
                        s->codec_ctx = NULL;
                        return false;
                }
                s->hwenc = true;
                s->hwframe = av_frame_alloc();
                av_hwframe_get_buffer(s->codec_ctx->hw_frames_ctx, s->hwframe, 0);
                pix_fmt = AV_PIX_FMT_NV12;
        }
#endif

        if (const AVPixFmtDescriptor * desc = av_pix_fmt_desc_get(pix_fmt)) { // defaults
                s->codec_ctx->colorspace = (desc->flags & AV_PIX_FMT_FLAG_RGB) != 0U ? AVCOL_SPC_RGB : AVCOL_SPC_BT709;
                s->codec_ctx->color_range = (desc->flags & AV_PIX_FMT_FLAG_RGB) != 0U ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
        }
        get_av_pixfmt_details(ug_codec, pix_fmt, &s->codec_ctx->colorspace, &s->codec_ctx->color_range);

        /* open it */
        if (avcodec_open2(s->codec_ctx, codec, NULL) < 0) {
                avcodec_free_context(&s->codec_ctx);
                log_msg(LOG_LEVEL_ERROR, "[lavc] Could not open codec for pixel format %s\n", av_get_pix_fmt_name(pix_fmt));
                return false;
        }

        return true;
}

static bool find_decoder(struct video_desc desc,
                AVPixelFormat pixfmt,
                codec_t *decoded_codec,
                decoder_t *decoder)
{
        if (get_ug_to_av_pixfmt(desc.color_spec) != AV_PIX_FMT_NONE
                        && pixfmt == get_ug_to_av_pixfmt(desc.color_spec)) {
                *decoded_codec = desc.color_spec;
                *decoder = vc_memcpy;
        } else {
                *decoder = get_decoder_from_uv_to_uv(desc.color_spec, pixfmt, decoded_codec);
        }

        return *decoder != nullptr;
}

static bool same_linesizes(codec_t codec, AVFrame *in_frame)
{
        if (codec_is_planar(codec)) {
                assert(get_bits_per_component(codec) == 8);
                int sub[8];
                codec_get_planes_subsampling(codec, sub);
                for (int i = 0; i < 4; ++i) {
                        if (sub[2 * i] == 0) {
                                return true;
                        }
                        if (in_frame->linesize[i] != (in_frame->width + sub[2 * i] - 1) / sub[2 * i]) {
                                return false;
                        }
                }
                return true;
        } else {
                return vc_get_linesize(in_frame->width, codec) == in_frame->linesize[0];
        }
}

const AVCodec *get_av_codec(struct state_video_compress_libav *s, codec_t *ug_codec, bool src_rgb) {
        // Open encoder specified by user if given
        if (!s->backend.empty()) {
                const AVCodec *codec = avcodec_find_encoder_by_name(s->backend.c_str());
                if (!codec) {
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Warning: requested encoder \"%s\" not found!\n",
                                        s->backend.c_str());
                        return nullptr;
                }
                if (s->requested_codec_id != VIDEO_CODEC_NONE && s->requested_codec_id != get_av_to_ug_codec(codec->id)) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME << "Encoder \"" << s->backend << "\" doesn't encode requested codec!\n";
                        return nullptr;

                }
                *ug_codec = get_av_to_ug_codec(codec->id);
                if (*ug_codec == VIDEO_CODEC_NONE) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Requested encoder not supported in UG!\n");
                        return nullptr;
                }
                return codec;
        }

        // Else, try to open prefered encoder for requested codec
        if (codec_params.find(*ug_codec) != codec_params.end() && codec_params[*ug_codec].get_prefered_encoder) {
                const char *prefered_encoder = codec_params[*ug_codec].get_prefered_encoder(
                                src_rgb);
                const AVCodec *codec = avcodec_find_encoder_by_name(prefered_encoder);
                if (!codec) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Warning: prefered encoder \"%s\" not found! Trying default encoder.\n",
                                        prefered_encoder);
                }
                return codec;
        }
        // Finally, try to open any encoder for requested codec
        return avcodec_find_encoder(get_ug_to_av_codec(*ug_codec));
}

static bool configure_swscale(struct state_video_compress_libav *s, struct video_desc desc) {
#ifndef HAVE_SWSCALE
        UNUSED(s), UNUSED(desc);
        return false;
#else
        //get all AVPixelFormats we can convert to and pick the first
        auto fmts = get_available_pix_fmts(desc, s->requested_subsampling, VIDEO_CODEC_NONE);
        s->sws_out_pixfmt = s->selected_pixfmt;
        s->selected_pixfmt = fmts.empty() ? AV_PIX_FMT_UYVY422 : fmts.front();
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Attempting to use swscale to convert from %s to %s.\n", av_get_pix_fmt_name(s->selected_pixfmt), av_get_pix_fmt_name(s->sws_out_pixfmt));
        if(!find_decoder(desc, s->selected_pixfmt, &s->decoded_codec, &s->decoder)){
                //Should not happen as get_available_pix_fmts should only
                //return formats we can decode to
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to find decoder from %s to %s before using swscale.\n", get_codec_name(desc.color_spec), av_get_pix_fmt_name(s->selected_pixfmt));
                return false;
        }

        s->sws_ctx = getSwsContext(desc.width,
                        desc.height,
                        s->selected_pixfmt,
                        desc.width,
                        desc.height,
                        s->sws_out_pixfmt,
                        SWS_POINT);
        if(!s->sws_ctx){
                log_msg(LOG_LEVEL_ERROR, "[lavc] Unable to init sws context.\n");
                return false;
        }

        s->sws_frame = av_frame_alloc();
        if (!s->sws_frame) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate sws frame\n");
                return false;
        }
        s->sws_frame->width = s->codec_ctx->width;
        s->sws_frame->height = s->codec_ctx->height;
        s->sws_frame->format = s->sws_out_pixfmt;
        if (int ret = av_image_alloc(s->sws_frame->data, s->sws_frame->linesize,
                        s->sws_frame->width, s->sws_frame->height,
                        s->sws_out_pixfmt, 32); ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate raw picture buffer for sws\n");
                return false;
        }

        log_msg(LOG_LEVEL_NOTICE, "[lavc] Using swscale to convert %s to %s.\n",
                        av_get_pix_fmt_name(s->selected_pixfmt),
                        av_get_pix_fmt_name(s->sws_out_pixfmt));
        return true;
#endif //HAVE_SWSCALE
}

static bool configure_with(struct state_video_compress_libav *s, struct video_desc desc)
{
        int ret;
        codec_t ug_codec = s->requested_codec_id == VIDEO_CODEC_NONE ? DEFAULT_CODEC : s->requested_codec_id;
        AVPixelFormat pix_fmt;
        const AVCodec *codec = nullptr;
#ifdef HAVE_SWSCALE
        sws_freeContext(s->sws_ctx);
        s->sws_ctx = nullptr;
        av_frame_free(&s->sws_frame);
        s->sws_out_pixfmt = AV_PIX_FMT_NONE;
#endif //HAVE_SWSCALE

        s->params.desc = desc;

        if ((codec = get_av_codec(s, &ug_codec, codec_is_a_rgb(desc.color_spec))) == nullptr) {
                return false;
        }
        log_msg(LOG_LEVEL_NOTICE, "[lavc] Using codec: %s, encoder: %s\n",
                        get_codec_name(ug_codec), codec->name);

        // Try to open the codec context
        // It is done in a loop because some pixel formats that are reported
        // by codec can actually fail (typically YUV444 in hevc_nvenc for Maxwell
        // cards).
        list<enum AVPixelFormat> requested_pix_fmt = get_requested_pix_fmts(desc, s->requested_subsampling);
        apply_blacklist(requested_pix_fmt, codec->name);
        auto requested_pix_fmt_it = requested_pix_fmt.cbegin();
        while ((pix_fmt = get_first_matching_pix_fmt(requested_pix_fmt_it, requested_pix_fmt.cend(), codec->pix_fmts)) != AV_PIX_FMT_NONE) {
                if(try_open_codec(s, pix_fmt, desc, ug_codec, codec)){
                        break;
                }
	}

        if (pix_fmt == AV_PIX_FMT_NONE || log_level >= LOG_LEVEL_VERBOSE) {
                print_pix_fmts(requested_pix_fmt, codec->pix_fmts);
        }

#ifdef HAVE_SWSCALE
        if (pix_fmt == AV_PIX_FMT_NONE) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "No direct decoder format for: " << get_codec_name(desc.color_spec) << ". Trying to convert with swscale instead.\n";
                for (const auto *pix = codec->pix_fmts; *pix != AV_PIX_FMT_NONE; ++pix) {
                        const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(*pix);
                        if (fmt_desc != nullptr && (fmt_desc->flags & AV_PIX_FMT_FLAG_HWACCEL) == 0U) {
                                AVPixelFormat curr_pix_fmt = *pix;
                                if (try_open_codec(s, curr_pix_fmt, desc, ug_codec, codec)) {
                                        pix_fmt = curr_pix_fmt;
                                        break;
                                }
                        }
                }
        }
#endif

        if (pix_fmt == AV_PIX_FMT_NONE) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to find suitable pixel format for: %s.\n", get_codec_name(desc.color_spec));
                if (s->requested_subsampling != 0) {
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Requested subsampling not supported. "
                                        "Try different subsampling, eg. "
                                        "\"subsampling={420,422,444}\".\n");
                }
                return false;
        }

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Codec %s capabilities: 0x%08X using thread type %d, count %d\n", codec->name,
                        codec->capabilities, s->codec_ctx->thread_type, s->codec_ctx->thread_count);
        log_msg(LOG_LEVEL_INFO, "[lavc] Selected pixfmt: %s\n", av_get_pix_fmt_name(pix_fmt));
        s->selected_pixfmt = pix_fmt;
        if (!pixfmt_has_420_subsampling(pix_fmt)) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Selected pixfmt has not 4:2:0 subsampling, "
                                "which is usually not supported by hw. decoders\n");
        }

        if(!find_decoder(desc, s->selected_pixfmt, &s->decoded_codec, &s->decoder)){
                log_msg(LOG_LEVEL_ERROR, "[lavc] Failed to find a way to convert %s to %s\n",
                                get_codec_name(desc.color_spec), av_get_pix_fmt_name(s->selected_pixfmt));
                if (!configure_swscale(s, desc)) {
                        return false;
                }
        }

        s->decoded = (unsigned char *) malloc(vc_get_linesize(desc.width, s->decoded_codec) * desc.height);

        s->in_frame = av_frame_alloc();
        if (!s->in_frame) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate video frame\n");
                return false;
        }
        s->in_frame->pts = -1;

        AVPixelFormat fmt = (s->hwenc) ? AV_PIX_FMT_NV12 : s->selected_pixfmt;
#if LIBAVCODEC_VERSION_MAJOR >= 53
        s->in_frame->format = fmt;
        s->in_frame->width = s->codec_ctx->width;
        s->in_frame->height = s->codec_ctx->height;
#endif

        ret = av_frame_get_buffer(s->in_frame, 0);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate raw picture buffer\n");
                return false;
        }
        // conversion needed
        if (get_ug_to_av_pixfmt(desc.color_spec) == AV_PIX_FMT_NONE
                        || get_ug_to_av_pixfmt(desc.color_spec) != s->selected_pixfmt) {
                for(int i = 0; i < s->conv_thread_count; ++i) {
                        int chunk_size = s->codec_ctx->height / s->conv_thread_count & ~1;
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
        }

        s->saved_desc = desc;
        s->compressed_desc = desc;
        s->compressed_desc.color_spec = ug_codec;
        s->compressed_desc.tile_count = 1;
        s->mov_avg_frames = s->mov_avg_comp_duration = 0;

        s->out_codec = s->compressed_desc.color_spec;

        return true;
}

static pixfmt_callback_t select_pixfmt_callback(AVPixelFormat fmt, codec_t src) {
        // no conversion needed
        if (get_ug_to_av_pixfmt(src) != AV_PIX_FMT_NONE
                        && get_ug_to_av_pixfmt(src) == fmt) {
                return nullptr;
        }

        for (auto c = get_uv_to_av_conversions(); c->src != VIDEO_CODEC_NONE; c++) { // FFMPEG conversion needed
                if (c->src == src && c->dst == fmt) {
                        return c->func;
                }
        }

        log_msg(LOG_LEVEL_FATAL, "[lavc] Cannot find conversion to any of encoder supported pixel format.\n");
        abort();
}

struct pixfmt_conv_task_data {
        pixfmt_callback_t callback;
        AVFrame *out_frame;
        unsigned char *in_data;
        int width;
        int height;
};

static void *pixfmt_conv_task(void *arg) {
        struct pixfmt_conv_task_data *data = (struct pixfmt_conv_task_data *) arg;
        data->callback(data->out_frame, data->in_data, data->width, data->height);
        return NULL;
}

/// print hint to improve performance if not making it
static void check_duration(struct state_video_compress_libav *s, time_ns_t dur_ns)
{
        constexpr int mov_window = 100;
        if (s->mov_avg_frames >= 10 * mov_window) {
                return;
        }
        double duration = dur_ns / NS_IN_SEC_DBL;
        s->mov_avg_comp_duration = (s->mov_avg_comp_duration * (mov_window - 1) + duration) / mov_window;
        s->mov_avg_frames += 1;
        if (s->mov_avg_frames < 2 * mov_window || s->mov_avg_comp_duration < 1 / s->compressed_desc.fps) {
                return;
        }
        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Average compression time of last %d frames is %f ms but time per frame is only %f ms!\n",
                        mov_window, s->mov_avg_comp_duration * 1000, 1000 / s->compressed_desc.fps);
        string hint;
        if (regex_match(s->codec_ctx->codec->name, regex(".*nvenc.*"))) {
                if (s->lavc_opts.find("delay") == s->lavc_opts.end()) {
                        hint = "\"delay=<frames>\" option to NVENC compression (2 suggested)";
                }
        } else if ((s->codec_ctx->thread_type & FF_THREAD_SLICE) == 0 && (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) != 0) {
                hint = "\"threads=<n>FS\" option with small <n> or 0 (nr of logical cores) to compression";
        } else if (s->codec_ctx->thread_count == 1 && (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_OTHER_THREADS) != 0) {
                hint = "\"threads=<n>\" option with small <n> or 0 (nr of logical cores) to compression";
        }
        if (!hint.empty()) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Consider adding " << hint << " to increase throughput at the expense of latency.\n";
        }
        s->mov_avg_frames = LONG_MAX;
}

static shared_ptr<video_frame> libavcodec_compress_tile(struct module *mod, shared_ptr<video_frame> tx)
{
        struct state_video_compress_libav *s = (struct state_video_compress_libav *) mod->priv_data;
        unsigned char *decoded;
        shared_ptr<video_frame> out{};
        list<shared_ptr<void>> cleanup_callbacks; // at function exit handlers

        libavcodec_check_messages(s);

        if(!video_desc_eq_excl_param(video_desc_from_frame(tx.get()),
                                s->saved_desc, PARAM_TILE_COUNT)) {
                cleanup(s);
                int ret = configure_with(s, video_desc_from_frame(tx.get()));
                if(!ret) {
                        return {};
                }
        }

        static auto dispose = [](struct video_frame *frame) {
#if LIBAVCODEC_VERSION_MAJOR >= 54 && LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 37, 100)
                AVPacket *pkt = (AVPacket *) frame->callbacks.dispose_udata;
                av_packet_unref(pkt);
                av_packet_free(&pkt);
#else
                free(frame->tiles[0].data);
#endif // LIBAVCODEC_VERSION_MAJOR >= 54
                vf_free(frame);
        };
        out = shared_ptr<video_frame>(vf_alloc_desc(s->compressed_desc), dispose);
        if (s->compressed_desc.color_spec == PRORES) {
                assert(s->codec_ctx->codec_tag != 0);
                out->color_spec = get_codec_from_fcc(s->codec_ctx->codec_tag);
        }
        vf_copy_metadata(out.get(), tx.get());
#if LIBAVCODEC_VERSION_MAJOR >= 54 && LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 37, 100)
        int got_output;
        AVPacket *pkt = av_packet_alloc();
        pkt->data = NULL;
        pkt->size = 0;
        out->callbacks.dispose_udata = pkt;
#else
        out->tiles[0].data = (char *) malloc(s->compressed_desc.width *
                        s->compressed_desc.height * 4);
#endif // LIBAVCODEC_VERSION_MAJOR >= 54

        if (int ret = av_frame_make_writable(s->in_frame)) {
                print_libav_error(LOG_LEVEL_ERROR, MOD_NAME "Cannot make frame writable", ret);
                return {};
        }
        s->in_frame->pts += 1;

        time_ns_t t_start = get_time_in_ns();
        if (s->decoder != vc_memcpy) {
                int src_linesize = vc_get_linesize(tx->tiles[0].width, tx->color_spec);
                int dst_linesize = vc_get_linesize(tx->tiles[0].width, s->decoded_codec);
                parallel_pix_conv(tx->tiles[0].height, reinterpret_cast<char *>(s->decoded), dst_linesize, tx->tiles[0].data, src_linesize, s->decoder, s->conv_thread_count);
                decoded = s->decoded;
        } else {
                decoded = (unsigned char *) tx->tiles[0].data;
        }

        time_ns_t t0 = get_time_in_ns();
        AVFrame *frame = s->in_frame;
        auto pixfmt_conv_callback = select_pixfmt_callback(s->selected_pixfmt, s->decoded_codec);
        if (pixfmt_conv_callback != nullptr) {
                vector<struct pixfmt_conv_task_data> data(s->conv_thread_count);
                for(int i = 0; i < s->conv_thread_count; ++i) {
                        data[i].callback = pixfmt_conv_callback;
                        data[i].out_frame = s->in_frame_part[i];

                        size_t height = tx->tiles[0].height / s->conv_thread_count & ~1; // height needs to be even
                        if (i < s->conv_thread_count - 1) {
                                data[i].height = height;
                        } else { // we are last so we need to do the rest
                                data[i].height = tx->tiles[0].height -
                                        height * (s->conv_thread_count - 1);
                        }
                        data[i].width = tx->tiles[0].width;
                        data[i].in_data = decoded + i * height *
                                vc_get_linesize(tx->tiles[0].width, s->decoded_codec);
                }
                task_run_parallel(pixfmt_conv_task, s->conv_thread_count, data.data(), sizeof data[0], NULL);
        } else { // no pixel format conversion needed
                if (codec_is_planar(s->decoded_codec) && !same_linesizes(s->decoded_codec, s->in_frame)) {
                        assert(get_bits_per_component(s->decoded_codec) == 8);
                        int sub[8];
                        codec_get_planes_subsampling(s->decoded_codec, sub);
                        unsigned char *in = decoded;
                        for (int i = 0; i < 4; ++i) {
                                if (sub[2 * i] == 0) {
                                        break;
                                }
                                int linesize = (s->in_frame->width + sub[2 * i] - 1) / sub[2 * i];
                                int lines = (s->in_frame->height + sub[2 * i + 1] - 1) / sub[2 * i + 1];
                                for (int y = 0; y < lines; ++y) {
                                        memcpy(s->in_frame->data[i] + y * s->in_frame->linesize[i], in, linesize);
                                        in += linesize;
                                }
                        }
                } else { // just set pointers to input buffer
                        frame = av_frame_alloc();
                        memcpy(frame->linesize, s->in_frame->linesize, sizeof frame->linesize);
                        frame->width = s->in_frame->width;
                        frame->height = s->in_frame->height;
                        frame->format = s->in_frame->format;
                        if (codec_is_planar(s->decoded_codec)) {
                                buf_get_planes(tx->tiles[0].width, tx->tiles[0].height, s->decoded_codec, (char *) decoded, (char **) frame->data);
                        } else {
                                frame->data[0] = (uint8_t *) decoded;
                        }
                        // prevent leaving dangling pointer to the input buffer that may
                        // be freed by cleanup()
                        std::shared_ptr<void> clean_data_ptr((void*)frame,
                                [](void *f) {
                                        auto *frame = (AVFrame *) f;
                                        av_frame_free(&frame);
                                });
                        cleanup_callbacks.push_back(std::move(clean_data_ptr));
                }
        }

        time_ns_t t1 = get_time_in_ns();

        debug_file_dump("lavc-avframe", serialize_video_avframe, s->in_frame);
#ifdef HWACC_VAAPI
        if(s->hwenc){
                av_hwframe_transfer_data(s->hwframe, s->in_frame, 0);
                frame = s->hwframe;
        }
#endif

#ifdef HAVE_SWSCALE
        if(s->sws_ctx){
                sws_scale(s->sws_ctx,
                          frame->data,
                          frame->linesize,
                          0,
                          frame->height,
                          s->sws_frame->data,
                          s->sws_frame->linesize);
                frame = s->sws_frame;
        }
#endif //HAVE_SWSCALE
        time_ns_t t2 = get_time_in_ns();

        /* encode the image */
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
        out->tiles[0].data_len = 0;
        if (libav_codec_has_extradata(s->out_codec)) { // we need to store extradata for HuffYUV/FFV1 in the beginning
                out->tiles[0].data_len += sizeof(uint32_t) + s->codec_ctx->extradata_size;
                *(uint32_t *)(void *) out->tiles[0].data = s->codec_ctx->extradata_size;
                memcpy(out->tiles[0].data + sizeof(uint32_t), s->codec_ctx->extradata, s->codec_ctx->extradata_size);
        }

        if (int ret = avcodec_send_frame(s->codec_ctx, frame)) {
                print_libav_error(LOG_LEVEL_WARNING, "[lavc] Error encoding frame", ret);
                return {};
        }
        int ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
        while (ret == 0) {
                assert(s->pkt->size + out->tiles[0].data_len <= s->compressed_desc.width * s->compressed_desc.height * 4 - out->tiles[0].data_len);
                memcpy((uint8_t *) out->tiles[0].data + out->tiles[0].data_len,
                                s->pkt->data, s->pkt->size);
                out->tiles[0].data_len += s->pkt->size;
                av_packet_unref(s->pkt);
                ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
        }
        if (ret != AVERROR(EAGAIN) && ret != 0) {
                print_libav_error(LOG_LEVEL_WARNING, "[lavc] Receive packet error", ret);
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
        time_ns_t t3 = get_time_in_ns();
        LOG(LOG_LEVEL_DEBUG2) << MOD_NAME << "duration pixfmt change: "
                << (t0 - t_start) / NS_IN_SEC_DBL << "+" << (t1 - t0) / NS_IN_SEC_DBL <<
                " s, dump+swscale " << (t2 - t1) / (double) NS_IN_SEC <<
                " s, compression " << (t3 - t2) / (double) NS_IN_SEC << " s\n";
        check_duration(s, t3 - t0);

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
			AVPacket *pkt = av_packet_alloc();
			ret = avcodec_receive_packet(s->codec_ctx, pkt);
			av_packet_unref(pkt);
			av_packet_free(&pkt);
			if (ret != 0 && ret != AVERROR_EOF) {
				log_msg(LOG_LEVEL_WARNING, "[lavc] Unexpected return value %d\n",
						ret);
				break;
			}
		} while (ret != AVERROR_EOF);
#endif
                avcodec_free_context(&s->codec_ctx);
        }
        if(s->in_frame) {
                av_frame_free(&s->in_frame);
        }
        free(s->decoded);
        s->decoded = NULL;

        av_frame_free(&s->hwframe);

#ifdef HAVE_SWSCALE
        sws_freeContext(s->sws_ctx);
        av_frame_free(&s->sws_frame);
#endif //HAVE_SWSCALE
}

static void libavcodec_compress_done(struct module *mod)
{
        struct state_video_compress_libav *s = (struct state_video_compress_libav *) mod->priv_data;

        cleanup(s);

        for (auto &f : s->in_frame_part) {
                av_free(f);
        }
        delete s;
}

/**
 * 1. sets required thread mode if specified, if not, set slice if available
 * 2. sets required thread count if specified, if not but codec supports other (external) threading
 *    set 0 (auto), otherwise if threading (slice/thread) was set, set it to number of cores
 */
static void set_codec_thread_mode(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        if (param->thread_mode == "no") { // disable threading (which may have been enabled previously
                codec_ctx->thread_type = 0;
                codec_ctx->thread_count = 1;
                return;
        }

        int req_thread_count = -1;
        int req_thread_type = 0;
        size_t endpos = 0;
        try { // just a number
                req_thread_count = stoi(param->thread_mode, &endpos);
        } catch(invalid_argument &) { // not a number
        }
        while (endpos != param->thread_mode.size()) {
                switch (toupper(param->thread_mode[endpos])) {
                        case 'n': req_thread_type = -1; break;
                        case 'F': req_thread_type |= FF_THREAD_FRAME; break;
                        case 'S': req_thread_type |= FF_THREAD_SLICE; break;
                        default: log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown thread mode: '%c'.\n", param->thread_mode[endpos]);
                }
                endpos += 1;
        }

        if (req_thread_type == 0) {
                if ((codec_ctx->codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) != 0) {
                        req_thread_type = FF_THREAD_SLICE;
                } else if ((codec_ctx->codec->capabilities & AV_CODEC_CAP_OTHER_THREADS) == 0 &&
                                (codec_ctx->codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) != 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Slice-based or external multithreading not available, encoding won't be parallel. "
                                        "You may select frame-based paralellism if needed.\n");
                }
        } else if (req_thread_type == -1) {
                req_thread_type = 0;
        }
        if (((req_thread_type & FF_THREAD_SLICE) != 0 && (codec_ctx->codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) == 0) ||
                        ((req_thread_type & FF_THREAD_FRAME) != 0 && (codec_ctx->codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) == 0)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Codec doesn't support specified thread mode.\n");
        } else {
                codec_ctx->thread_type = req_thread_type;
        }

        if (req_thread_count != -1) {
                codec_ctx->thread_count = req_thread_count;
        } else if ((codec_ctx->codec->capabilities & AV_CODEC_CAP_OTHER_THREADS) != 0) {
                // do not enable MT for eg. libx265 - libx265 uses frame threads
                if (strncmp(codec_ctx->codec->name, "libvpx", 6) == 0) {
                        codec_ctx->thread_count = 0;
                }
        } else if (codec_ctx->thread_type != 0) {
                codec_ctx->thread_count = thread::hardware_concurrency();
        }
}

static void setparam_default(AVCodecContext *codec_ctx, struct setparam_param * /* param */)
{
        if (codec_ctx->codec->id == AV_CODEC_ID_JPEG2000) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] J2K support is experimental and may be broken!\n");
        }
}

static void setparam_jpeg(AVCodecContext *codec_ctx, struct setparam_param * /* param */)
{
        check_av_opt_set<const char *>(codec_ctx->priv_data, "huffman", "default", "Huffman tables");
}

static void configure_amf([[maybe_unused]] AVCodecContext *codec_ctx, [[maybe_unused]] struct setparam_param *param) {
        check_av_opt_set<const char *>(codec_ctx->priv_data, "header_insertion_mode", "gop", "header_insertion_mode for AMF");
}

ADD_TO_PARAM("lavc-h264-interlaced-dct", "* lavc-h264-interlaced-dct\n"
                 "  Use interlaced DCT for H.264 (disabled for NVENC)\n");
ADD_TO_PARAM("lavc-h264-no-interlaced-dct", "* lavc-h264-no-interlaced-dct\n"
                 "  Do not use interlaced DCT for H.264 (enabled for x264 and QSV)\n");
ADD_TO_PARAM("lavc-rc-buffer-size-factor", "* lavc-rc-buffer-size-factor=<val>\n"
                 "  Multiplier how much can individual frame overshot average size (default x264/5: " TOSTRING(DEFAULT_X26X_RC_BUF_SIZE_FACTOR) ", nvenc: 1).\n");
static void configure_x264_x265(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        const char *tune = codec_ctx->codec->id == AV_CODEC_ID_H264 ? "zerolatency,fastdecode" : "zerolatency"; // x265 supports only single tune parameter
        check_av_opt_set<const char *>(codec_ctx->priv_data, "tune", tune);

        // try to keep frame sizes as even as possible
        codec_ctx->rc_max_rate = codec_ctx->bit_rate;
        //codec_ctx->rc_min_rate = s->codec_ctx->bit_rate / 4 * 3;
        //codec_ctx->rc_buffer_aggressivity = 1.0;
        double lavc_rc_buffer_size_factor = DEFAULT_X26X_RC_BUF_SIZE_FACTOR;
        if (const char *val = get_commandline_param("lavc-rc-buffer-size-factor")) {
                lavc_rc_buffer_size_factor = stof(val);
        }
        codec_ctx->rc_buffer_size = codec_ctx->rc_max_rate / param->desc.fps * lavc_rc_buffer_size_factor; // "emulate" CBR. Note that factor less than 8 used to cause encoder buffer overflows and artifacts in stream.
        codec_ctx->qcompress = codec_ctx->codec->id == AV_CODEC_ID_H265 ? 0.5F : 0.0F;
        //codec_ctx->qblur = 0.0f;
        //codec_ctx->rc_min_vbv_overflow_use = 1.0f;
        //codec_ctx->rc_max_available_vbv_use = 1.0f;
        codec_ctx->qmin = IF_NOT_UNDEF_ELSE(codec_ctx->qmin, 0);  // qmin,qmax set to -1 by default
        codec_ctx->qmax = IF_NOT_UNDEF_ELSE(codec_ctx->qmax, 69);
        codec_ctx->max_qdiff = 69;
        //codec_ctx->rc_qsquish = 0;
        //codec_ctx->scenechange_threshold = 100;

        if (param->desc.interlacing == INTERLACED_MERGED && get_commandline_param("lavc-h264-no-interlaced-dct") == NULL) {
                codec_ctx->flags |= AV_CODEC_FLAG_INTERLACED_DCT;
        }

        string x265_params;
        if (param->lavc_opts.find("x265-params") != param->lavc_opts.end()) {
                x265_params = param->lavc_opts.at("x265-params");
                param->lavc_opts.erase("x265-params");
        }
        auto x265_params_append = [&](const string &key, const string &val) {
                if (x265_params.find(key) == string::npos) {
                        x265_params += (x265_params.empty() ? "" : ":") + key + "=" + val;
                }
        };
        x265_params_append("keyint", to_string(codec_ctx->gop_size));
        /// turn on periodic intra refresh, unless explicitely disabled
        if (param->periodic_intra != 0) {
                codec_ctx->refs = 1;
                if ("libx264"s == codec_ctx->codec->name || "libx264rgb"s == codec_ctx->codec->name) {
                        check_av_opt_set<const char *>(codec_ctx->priv_data, "intra-refresh", "1");
                } else if ("libx265"s == codec_ctx->codec->name) {
                        x265_params_append("intra-refresh", "1");
                        x265_params_append("constrained-intra", "1");
                        x265_params_append("no-open-gop", "1");
                }
        }
        if ("libx265"s == codec_ctx->codec->name) {
                check_av_opt_set<const char *>(codec_ctx->priv_data, "x265-params", x265_params.c_str());
        }
}

static void configure_qsv(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        if (param->periodic_intra != 0) {
                check_av_opt_set<const char *>(codec_ctx->priv_data, "int_ref_type", "vertical");
#if 0
                ret = av_opt_set(codec_ctx->priv_data, "int_ref_cycle_size", "100", 0);
                if (ret != 0) {
                        log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to set intra refresh size.\n");
                }
#endif
        }
        codec_ctx->rc_max_rate = codec_ctx->bit_rate;
        // no look-ahead and rc_max_rate == bit_rate result in use of CBR for QSV

        if (param->desc.interlacing == INTERLACED_MERGED && get_commandline_param("lavc-h264-no-interlaced-dct") == NULL) {
                codec_ctx->flags |= AV_CODEC_FLAG_INTERLACED_DCT;
        }
}

static void configure_vaapi(AVCodecContext * /* codec_ctx */, struct setparam_param *param) {
        param->thread_mode = "no"; // VA-API doesn't support threads
        // interesting options: "b_depth" (not used - we are not using B-frames), "idr_interval" - set to 0 by default
}

void set_forced_idr(AVCodecContext *codec_ctx, int value)
{
        assert(value <= 9);
        array<char, 2> force_idr_val{};
        force_idr_val[0] = '0' + value;

        if (int ret = av_opt_set(codec_ctx->priv_data, "forced-idr", force_idr_val.data(), 0)) {
                print_libav_error(LOG_LEVEL_WARNING, MOD_NAME "Unable to set Forced IDR", ret);
        }
}

static void configure_nvenc(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        const char *preset = DEFAULT_NVENC_PRESET;

        // important: if "tune" is not supported, then FALLBACK_NVENC_PRESET must be used (it is correlated). If unsupported preset
        // were given, setting would succeed but would cause runtime errors.
        if (!check_av_opt_set<const char *>(codec_ctx->priv_data, "tune", DEFAULT_NVENC_TUNE, "NVENC tune")) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Possibly old libavcodec or compiled with old NVIDIA NVENC headers.\n";
                preset = FALLBACK_NVENC_PRESET;
        }
        if (!param->have_preset) {
                if (check_av_opt_set<const char *>(codec_ctx->priv_data, "preset", preset, "NVENC preset")) {
                        LOG(LOG_LEVEL_INFO) << MOD_NAME "Setting NVENC preset to " << preset << ".\n";
                }
        }

        set_forced_idr(codec_ctx, 1);
#ifdef PATCHED_FF_NVENC_NO_INFINITE_GOP
        const bool patched_ff = true;
#else
        const bool patched_ff = false;
        if (param->periodic_intra != 0) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "FFmpeg not patched, " << (param->periodic_intra != 1 ? "not " : "") << "enabling Intra Refresh.\n";
        }
#endif

        if ((patched_ff && param->periodic_intra != 0) || param->periodic_intra == 1) {
                check_av_opt_set<int>(codec_ctx->priv_data, "intra-refresh", 1);
        }

        check_av_opt_set<const char *>(codec_ctx->priv_data, "rc", DEFAULT_NVENC_RC);
        check_av_opt_set<int>(codec_ctx->priv_data, "spatial_aq", 0);
        check_av_opt_set<int>(codec_ctx->priv_data, "gpu", cuda_devices[0]);
        check_av_opt_set<int>(codec_ctx->priv_data, "delay", 0); // 2'd increase throughput 2x at expense of higher latency
        check_av_opt_set<int>(codec_ctx->priv_data, "zerolatency", 1, "zero latency operation (no reordering delay)");
        check_av_opt_set<const char *>(codec_ctx->priv_data, "b_ref_mode", "disabled", 0);
        codec_ctx->rc_max_rate = codec_ctx->bit_rate;
        codec_ctx->rc_buffer_size = codec_ctx->rc_max_rate / param->desc.fps;
        if (const char *val = get_commandline_param("lavc-rc-buffer-size-factor")) {
                codec_ctx->rc_buffer_size *= stof(val);
        } else {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "To reduce NVENC pulsation, you can try \"--param lavc-rc-buffer-size-factor=0\""
                                       " or a small number. 0 or higher value (than default 1) may cause frame drops on receiver.\n");
        }
        if (param->desc.interlacing == INTERLACED_MERGED && get_commandline_param("lavc-h264-interlaced-dct") != NULL) {
                codec_ctx->flags |= AV_CODEC_FLAG_INTERLACED_DCT;
        }
}

static void configure_svt(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        // see FFMPEG modules' sources for semantics
        set_forced_idr(codec_ctx, strcmp(codec_ctx->codec->name, "libsvt_hevc") == 0 ? 0 : 1);

        if ("libsvt_hevc"s == codec_ctx->codec->name) {
                check_av_opt_set<int>(codec_ctx->priv_data, "la_depth", 0);
                check_av_opt_set<int>(codec_ctx->priv_data, "pred_struct", 0);
                int tile_col_cnt = param->desc.width >= 1024 ? 4 : param->desc.width >= 512 ? 2 : 1;
                int tile_row_cnt = param->desc.height >= 256 ? 4 : param->desc.height >= 128 ? 2 : 1;
                if (tile_col_cnt * tile_row_cnt > 1 && param->desc.width >= 256 && param->desc.height >= 64) {
                        check_av_opt_set<int>(codec_ctx->priv_data, "tile_row_cnt", tile_row_cnt);
                        check_av_opt_set<int>(codec_ctx->priv_data, "tile_col_cnt", tile_col_cnt);
                        check_av_opt_set<int>(codec_ctx->priv_data, "tile_slice_mode", 1);
                        check_av_opt_set<int>(codec_ctx->priv_data, "umv", 0);
                }
        } else if ("libsvtav1"s == codec_ctx->codec->name) {
#if LIBAVCODEC_VERSION_INT > AV_VERSION_INT(59, 21, 100)
                //pred-struct=1 is low-latency mode
                if (int ret = av_opt_set(codec_ctx->priv_data, "svtav1-params", "pred-struct=1:tile-columns=2:tile-rows=2", 0)) {
                        print_libav_error(LOG_LEVEL_WARNING, MOD_NAME "Unable to set svtav1-params for SVT", ret);
                }
#else
                // tile_columns and tile_rows are log2 values
                for (auto const &val : { "tile_columns", "tile_rows" }) {
                        check_av_opt_set<int>(codec_ctx->priv_data, val, 2, "tile dimensions for SVT AV1");
                }
#endif
        }
}

static void setparam_h264_h265_av1(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        if (regex_match(codec_ctx->codec->name, regex(".*_amf"))) {
                configure_amf(codec_ctx, param);
        } if (regex_match(codec_ctx->codec->name, regex(".*_vaapi"))) {
                configure_vaapi(codec_ctx, param);
        } else if (strncmp(codec_ctx->codec->name, "libx264", strlen("libx264")) == 0 || // libx264 and libx264rgb
                        strcmp(codec_ctx->codec->name, "libx265") == 0) {
                configure_x264_x265(codec_ctx, param);
        } else if (regex_match(codec_ctx->codec->name, regex(".*nvenc.*"))) {
                configure_nvenc(codec_ctx, param);
        } else if (strcmp(codec_ctx->codec->name, "h264_qsv") == 0 ||
                        strcmp(codec_ctx->codec->name, "hevc_qsv") == 0) {
                configure_qsv(codec_ctx, param);
        } else if (strstr(codec_ctx->codec->name, "libsvt") == codec_ctx->codec->name) {
                configure_svt(codec_ctx, param);
        } else {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Warning: Unknown encoder %s. Using default configuration values.\n", codec_ctx->codec->name);
        }
}

void show_encoder_help(string const &name) {
        col() << "Options for " << SBOLD(name) << ":\n";
        auto *codec = avcodec_find_encoder_by_name(name.c_str());
        if (codec == nullptr) {
                codec = avcodec_find_decoder_by_name(name.c_str());
        }
        if (codec == nullptr) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Unable to find encoder " << name << "!\n";
                return;
        }
        const auto *opt = codec->priv_class->option;
        if (opt == nullptr) {
                return;
        }
        while (opt->name != nullptr) {
                string default_val;
                if (opt->offset != 0) {
                        if (opt->type == AV_OPT_TYPE_FLOAT || opt->type == AV_OPT_TYPE_DOUBLE) {
                                default_val = to_string(opt->default_val.dbl) + "F";
                        } else if (opt->type == AV_OPT_TYPE_CONST || opt->type == AV_OPT_TYPE_INT64 || opt->type == AV_OPT_TYPE_INT || opt->type == AV_OPT_TYPE_BOOL) {
                                default_val = to_string(opt->default_val.i64);
                        } else if (opt->type == AV_OPT_TYPE_STRING && opt->default_val.str != nullptr) {
                                default_val = string("\"") + opt->default_val.str + "\"";
                        }
                        if (!default_val.empty()) {
                                default_val = ", default " + default_val;
                        }
                }
                col() << (opt->offset == 0 ? "\t\t* " : "\t- ") << SBOLD(opt->name) << (opt->help != nullptr && strlen(opt->help) > 0 ? " - "s + opt->help : ""s) << default_val << "\n";
                opt++;
        }
        if (name == "libx264" || name == "libx265") {
                col() << "(options for " << SBOLD(name.substr(3) << "-params") << " should be actually separated by '\\:', not ':' as indicated above)\n";
        }
}

/// @retval DEFER_PRESET_SETTING - preset will be set individually later (NVENC)
static string get_h264_h265_preset(string const & enc_name, int width, int height, double fps)
{
        if (enc_name == "libx264" || enc_name == "libx264rgb") {
                if (width <= 1920 && height <= 1080 && fps <= 30) {
                        return string("veryfast");
                } else {
                        return string("ultrafast");
                }
        }
        if (enc_name == "libx265") {
                return string("ultrafast");
        }
        if (regex_match(enc_name, regex(".*nvenc.*"))) { // so far, there are at least nvenc, nvenc_h264 and h264_nvenc variants
                return string{DONT_SET_PRESET}; // nvenc preset is handled with configure_nvenc()
        }
        if (regex_match(enc_name, regex(".*_qsv"))) {
                return string(DEFAULT_QSV_PRESET);
        }
        if (regex_match(enc_name, regex(".*_vaapi"))) {
                return string{DONT_SET_PRESET}; // VAAPI doesn't support presets
        }
        return {};
}

static string get_av1_preset(string const & enc_name, int width, int height, double fps)
{
        if (enc_name == "libsvtav1") {
                if (width <= 1920 && height <= 1080 && fps <= 30) {
                        return string("9");
                } else {
                        return string("11");
                }
        }
        return {};
}

static void setparam_vp8_vp9(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        codec_ctx->rc_buffer_size = codec_ctx->bit_rate / param->desc.fps;
        //codec_ctx->rc_buffer_aggressivity = 0.5;
        check_av_opt_set<const char *>(codec_ctx->priv_data, "deadline", "realtime");
        check_av_opt_set<int>(codec_ctx->priv_data, "cpu-used", 8, "quality/speed ration modifier");
        check_av_opt_set<int>(codec_ctx->priv_data, "rc_lookahead", 0);
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
        NULL,
        NULL,
        get_libavcodec_module_info,
};

REGISTER_MODULE(libavcodec, &libavcodec_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace

