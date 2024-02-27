/**
 * @file   video_compress/libavcodec.cpp
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

#define __STDC_CONSTANT_MACROS

#include <array>
#include <cassert>
#include <cmath>
#include <list>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include "config.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "libavcodec/lavc_common.h"
#include "libavcodec/lavc_video.h"
#include "libavcodec/to_lavc_vid_conv.h"
#include "messaging.h"
#include "module.h"
#include "rtp/rtpdec_h264.h"
#include "rtp/rtpenc_h264.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/string.h" // replace_all
#include "utils/text.h"
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

using std::array;
using std::clamp;
using std::cout;
using std::function;
using std::list;
using std::invalid_argument;
using std::map;
using std::min;
using std::regex;
using std::set;
using std::shared_ptr;
using std::stof;
using std::stoi;
using std::string;
using std::thread;
using std::to_string;
using namespace std::string_literals;

// NOLINTNEXTLINE(*-macro-usage): for correct TOSTRING expansion
#define DEFAULT_X26X_RC_BUF_SIZE_FACTOR  2.5
#define DEFAULT_NVENC_RC_BUF_SIZE_FACTOR 1.5 // NOLINT: ditto

namespace {
enum {
        FLW_THRESH = 1920 * 1080 * 30, //< in px/sec
};

constexpr const codec_t DEFAULT_CODEC       = MJPG;
constexpr const int     DEFAULT_GOP_SIZE    = 20;
constexpr int           DEFAULT_SLICE_COUNT = 32;

constexpr const char *DEFAULT_AMF_RC        = "cqp";
constexpr const char *DEFAULT_AMF_USAGE     = "ultralowlatency";
constexpr const char *DEFAULT_AMF_USAGE_AV1 = "lowlatency";
constexpr int         DEFAULT_CQP           = 21;
constexpr int         DEFAULT_CQP_MPJEG     = 7;
constexpr int         DEFAULT_CQP_MJPEG_QSV = 80;
constexpr int         DEFAULT_CQP_QSV       = 5000;
constexpr const char *DEFAULT_NVENC_PRESET  = "p4";
constexpr const char *DEFAULT_NVENC_RC      = "cbr";
constexpr const char *DEFAULT_NVENC_TUNE    = "ull";
constexpr const char *DEFAULT_QSV_PRESET    = "medium";
constexpr const char *DEFAULT_QSV_RC        = "vbr";
constexpr double      DEFAULT_X264_X265_CRF = 22.0;
constexpr const char *FALLBACK_NVENC_PRESET = "llhq";

struct setparam_param {
        setparam_param(map<string, string> &lo, set<string> &bo) : lavc_opts(lo), blacklist_opts(bo) {}
        struct video_desc desc {};
        bool have_preset = false;
        int periodic_intra = -1; ///< -1 default; 0 disable/not enable; 1 enable
        int interlaced_dct = -1; ///< -1 default; 0 disable/not enable; 1 enable
        int header_inserter_req = -1;
        string thread_mode;
        int slices = -1;
        map<string, string> &lavc_opts; ///< user-supplied options from command-line
        set<string>         &blacklist_opts; ///< options that should be blacklisted
};

typedef struct {
        function<const char*(bool)> get_prefered_encoder; ///< can be nullptr
        double avg_bpp;
        void (*set_param)(AVCodecContext *, struct setparam_param *);
        int capabilities_priority;
} codec_params_t;

static void libavcodec_check_messages(struct state_video_compress_libav *s);
static void libavcodec_compress_done(struct module *mod);
static void setparam_default(AVCodecContext *, struct setparam_param *);
static void setparam_h264_h265_av1(AVCodecContext *, struct setparam_param *);
static void setparam_jpeg(AVCodecContext *, struct setparam_param *);
static void setparam_vp8_vp9(AVCodecContext *, struct setparam_param *);
static void set_codec_thread_mode(AVCodecContext *codec_ctx, struct setparam_param *param);

static void show_encoder_help(string const &name);
static void print_codec_supp_pix_fmts(const enum AVPixelFormat *first);
void usage(bool full);
static int parse_fmt(struct state_video_compress_libav *s, char *fmt);
static void cleanup(struct state_video_compress_libav *s);

static map<codec_t, codec_params_t> codec_params = {
        { H264, codec_params_t{
                [](bool is_rgb) { return is_rgb ? "libx264rgb" : "libx264"; },
                0.07 * 2 /* for H.264: 1 - low motion, 2 - medium motion, 4 - high motion */
                * 2, // take into consideration that our H.264 is less effective due to specific preset/tune
                     // note - not used for libx264, which uses CRF by default
                setparam_h264_h265_av1,
                100
        }},
        { H265, codec_params_t{
                [](bool) { return "libx265"; },
                0.04 * 2 * 2, // note - not used for libx265, which uses CRF by default
                setparam_h264_h265_av1,
                101
        }},
        { MJPG, codec_params_t{
                nullptr,
                1.2,
                setparam_jpeg,
                102
        }},
        { J2K, codec_params_t{
                nullptr,
                1.0,
                setparam_default,
                500
        }},
        { VP8, codec_params_t{
                nullptr,
                0.4,
                setparam_vp8_vp9,
                103
        }},
        { VP9, codec_params_t{
                nullptr,
                0.4,
                setparam_vp8_vp9,
                104
        }},
        { HFYU, codec_params_t{
                nullptr,
                0,
                setparam_default,
                501
        }},
        { FFV1, codec_params_t{
                nullptr,
                0,
                setparam_default,
                502
        }},
        { AV1, codec_params_t{
                [](bool) { return is_arm_mac() ? "libaom-av1" : "libsvtav1"; },
                0.1,
                setparam_h264_h265_av1,
                600
        }},
        { PRORES, codec_params_t{
                nullptr,
                0.5,
                setparam_default,
                300,
        }},
};

struct aux_header {
        char   buf[1024]{};
        size_t buf_len = 0;
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
                to_lavc_vid_conv_destroy(&pixfmt_conversion);
        }

        struct module       module_data;

        struct video_desc   saved_desc{};
        struct to_lavc_vid_conv *pixfmt_conversion = nullptr;
        AVPacket           *pkt = av_packet_alloc();
        // for every core - parts of the above
        AVCodecContext     *codec_ctx = nullptr;
        int64_t             cur_pts   = 0;

        codec_t             requested_codec_id = VIDEO_CODEC_NONE;
        long long int       requested_bitrate = 0;
        double              requested_bpp = 0;
        double              requested_crf = -1;
        int                 requested_cqp = -1;
        struct to_lavc_req_prop req_conv_prop{ TO_LAVC_REQ_PROP_INIT };
        bool store_orig_format = false;
        struct aux_header aux_header;

        struct video_desc compressed_desc{};

        struct setparam_param params{lavc_opts, blacklist_opts};
        string              backend;
        int                 requested_gop = DEFAULT_GOP_SIZE;

        map<string, string> lavc_opts; ///< user-supplied options from command-line
        set<string>         blacklist_opts; ///< options that has been processed by setparam handlers and should not be passed to codec

        bool hwenc = false;
        AVFrame *hwframe = nullptr;

#ifdef HAVE_SWSCALE
        struct SwsContext *sws_ctx = nullptr;
        AVFrame *sws_frame = nullptr;
#endif

        int conv_thread_count = clamp<unsigned int>(thread::hardware_concurrency(), 1, INT_MAX); ///< number of threads used for UG conversions

        double    mov_avg_comp_duration = 0;
        long      mov_avg_frames        = 0;
        time_ns_t duration_warn_last_print;

        map<int64_t, char[VF_METADATA_SIZE]> metadata_storage;
};

struct codec_encoders_decoders{
        std::vector<std::string> encoders;
        std::vector<std::string> decoders;
};

enum incomp_feature {
        INCOMP_DEPTH,
        INCOMP_INTERLACED_DCT,
        INCOMP_INTRA_REFRESH,
        INCOMP_SUBSAMPLING,
};
void incomp_feature_warn(enum incomp_feature f, int req_val);

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
#else
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
#endif

        return res;
}

static void get_codec_details(AVCodecID id, char *buf, size_t buflen)
{
        auto info = get_codec_encoders_decoders(id);
        assert(buflen > 0);
        buf[0] = '\0';
        if(info.encoders.empty() && info.decoders.empty())
                return;

        strncat(buf, " (", buflen - strlen(buf) - 1);
        if (!info.encoders.empty()) {
                strncat(buf, TERM_BOLD "encoders:" TERM_RESET, buflen - strlen(buf) - 1);
                for(const auto& enc : info.encoders){
                        strncat(buf, " ", buflen - strlen(buf) - 1);
                        strncat(buf, enc.c_str(), buflen - strlen(buf) - 1);
                }
        }
        if (!info.decoders.empty()) {
                if (!info.encoders.empty()) {
                        strncat(buf, "; ", buflen - strlen(buf) - 1);
                }
                strncat(buf, TERM_BOLD "decoders:" TERM_RESET, buflen - strlen(buf) - 1);

                for(const auto& dec : info.decoders){
                        strncat(buf, " ", buflen - strlen(buf) - 1);
                        strncat(buf, dec.c_str(), buflen - strlen(buf) - 1);
                }
        }
        strncat(buf, ")", buflen - strlen(buf) - 1);
}

void usage(bool full) {
        printf("Libavcodec encoder usage:\n");
        col() << "\t" SBOLD(SRED("-c libavcodec")
                            << "[:codec=<codec_name>|:encoder=<encoder>]\n\t\t["
                               ":bitrate=<bits_per_sec>|:bpp=<bits_per_pixel>|:"
                               "crf=<crf>|:cqp=<cqp>]\n\t\t[:subsampling=<"
                               "subsampling>][:depth=<depth>"
                               "][:rgb|:yuv][:gop=<gop>]\n\t\t"
                               "[:[disable_]intra_refresh][:threads=<threads>]["
                               ":slices=<slices>][safe]\n\t\t[:<lavc_opt>=<val>]*")
              << "\n\t" << SBOLD(SRED("-c libavcodec") << ":[full]help") << "\n";
        col() << "\nwhere\n";
        col() << "\t" << SBOLD("<encoder>") << " specifies encoder (eg. nvenc or libx264 for H.264)\n";
        col() << "\t" << SBOLD("<codec_name>") << " - codec name (default MJPEG) if encoder name is not specified\n";
        col() << "\t" << SBOLD("[disable_]intra_refresh") << ", "
              << SBOLD("[disable_]interlaced_dct")
              << " - (do not) use Periodic Intra Refresh (H.264/H.265), (do "
                 "not) use interlaced DCT for H.264\n";
        col() << "\t" << SBOLD("<bits_per_sec>") << " specifies requested bitrate\n"
                << "\t\t\t0 means codec default (same as when parameter omitted)\n";
        col() << "\t" << SBOLD("<bits_per_pixel>") << " specifies requested bitrate using compressed bits per pixel\n"
                << "\t\t\tbitrate = frame width * frame height * bits_per_pixel * fps\n";
        col() << "\t" << SBOLD("<cqp>") << " use codec-specific constant QP value, for some codecs like MJPEG this is the only quality setting option\n";
        col() << "\t" << SBOLD("<crf>") << " specifies CRF factor (only for libx264/libx265)\n";
        col() << "\t" << SBOLD("<subsampling>") << " may be one of 444, 422, or 420, default 420 for progresive, 422 for interlaced\n";
        col() << "\t" << SBOLD("<depth>") << " enforce specified compression bit depth\n";
        col() << "\t" << SBOLD("rgb|yuv") << " enforce specified color space compreesion\n";
        col() << "\t" << SBOLD("<threads>") << " can be \"no\", or \"<number>[F][S][n]\" where 'F'/'S' indicate if frame/slice thr. should be used, both can be used (default slice), 'n' means none;\n";
        col() << "\t" <<       "         "  << " use a comma to add also number of conversion threads (eg. \"0S,8\"), default: number of logical cores\n";
        col() << "\t" << SBOLD("<slices>") << " number of slices to use (default: " << DEFAULT_SLICE_COUNT << ")\n";
        col() << "\t" << SBOLD("<gop>") << " specifies GOP size\n";
        col() << "\t" << SBOLD("<lavc_opt>") << " arbitrary option to be passed directly to libavcodec (eg. preset=veryfast), eventual colons must be backslash-escaped (eg. for x264opts)\n";
        col() << "\t" << SBOLD("safe") << " use opts for (HW) decode compatibility - 420, no intra refresh and interlacing\n";
        if (full) {
                col() << "\t" << SBOLD("header_inserter[=no]")
                      << " repeat H.264/HEVC VPS/SPS/PPS hdrs (fixes problems "
                         "when not "
                         "contained in the stream)\n";
        }
        col() << "\nSupported codecs:\n";
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
                get_codec_details(avID, avail + strlen(avail), sizeof avail - strlen(avail));
                col() << "\t" << SBOLD(get_codec_name(param.first)) << " - " << avail << "\n";

        }
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

        int show_help = 0;

        // replace all '\:' with 2xDEL
        replace_all(fmt, ESCAPED_COLON, DELDEL);
        char *item, *save_ptr = NULL;

        while ((item = strtok_r(fmt, ":", &save_ptr)) != NULL) {
                if (strcasecmp("help", item) == 0 || strcmp(item, "fullhelp") == 0) {
                        show_help = strcmp(item, "fullhelp") == 0 ? 2 : 1;
                } else if (IS_KEY_PREFIX(item, "codec")) {
                        const char *const codec = strchr(item, '=') + 1;
                        s->requested_codec_id   = get_codec_from_name(codec);
                        if (s->requested_codec_id == VIDEO_CODEC_NONE) {
                                log_msg(LOG_LEVEL_ERROR, "[lavc] Unable to find codec: \"%s\"\n", codec);
                                return -1;
                        }
                } else if (IS_KEY_PREFIX(item, "encoder")) {
                        s->backend = strchr(item, '=') + 1;
                } else if (IS_KEY_PREFIX(item, "bitrate")) {
                        s->requested_bitrate =
                            unit_evaluate(strchr(item, '=') + 1, nullptr);
                        assert(s->requested_bitrate >= 0);
                } else if(strncasecmp("bpp=", item, strlen("bpp=")) == 0) {
                        char *bpp_str = item + strlen("bpp=");
                        s->requested_bpp =
                            unit_evaluate_dbl(bpp_str, false, nullptr);
                        if (std::isnan(s->requested_bpp)) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Wrong bitrate: " << bpp_str << "\n";
                                return -1;
                        }
                } else if(strncasecmp("crf=", item, strlen("crf=")) == 0) {
                        char *crf_str = item + strlen("crf=");
                        s->requested_crf = atof(crf_str);
                } else if (strstr(item, "cqp=") == item || strstr(item, "q=") == item) {
                        if (strstr(item, "q=") == item) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Option \"q=\" is deprecated, use \"cqp=\" instead.\n");
                        }
                        s->requested_cqp = atoi(strchr(item, '=') + 1);
                } else if (IS_KEY_PREFIX(item, "subsampling")) {
                        s->req_conv_prop.subsampling =
                            atoi(strchr(item, '=') + 1);
                        if (s->req_conv_prop.subsampling < 1000) {
                                s->req_conv_prop.subsampling *= 10; // 420->4200
                        }
                        if (s->req_conv_prop.subsampling != 4440 &&
                                        s->req_conv_prop.subsampling != 4220 &&
                                        s->req_conv_prop.subsampling != 4200) {
                                log_msg(LOG_LEVEL_ERROR, "[lavc] Supported subsampling is 444, 422, or 420.\n");
                                return -1;
                        }
                } else if (IS_KEY_PREFIX(item, "depth")) {
                        s->req_conv_prop.depth = atoi(strchr(item, '=') + 1);
                } else if (strcasecmp(item, "rgb") == 0 || strcasecmp(item, "yuv") == 0) {
                        s->req_conv_prop.rgb = strcasecmp(item, "rgb") == 0;
                } else if (strstr(item, "intra_refresh") != nullptr) {
                        s->params.periodic_intra = strstr(item, "disable_") == item ? 0 : 1;
                } else if (strstr(item, "interlaced_dct") != nullptr) {
                        s->params.interlaced_dct = strstr(item, "disable_") == item ? 0 : 1;
                } else if (IS_KEY_PREFIX(item, "threads")) {
                        char *threads = strchr(item, '=') + 1;
                        if (strchr(threads, ',')) {
                                s->conv_thread_count = stoi(strchr(threads, ',') + 1);
                                *strchr(threads, ',') = '\0';
                        }
                        s->params.thread_mode = threads;
                } else if(strncasecmp("slices=", item, strlen("slices=")) == 0) {
                        char *slices = strchr(item, '=') + 1;
                        s->params.slices = stoi(slices);
                } else if(strncasecmp("gop=", item, strlen("gop=")) == 0) {
                        char *gop = item + strlen("gop=");
                        s->requested_gop = atoi(gop);
                } else if (strstr(item, "header_inserter") == item) {
                        s->params.header_inserter_req =
                            strstr(item, "=no") == nullptr ? 1 : 0;
                } else if (strcmp(item, "safe") == 0) {
                        s->params.periodic_intra     = 0;
                        s->params.periodic_intra     = 0;
                        s->params.interlaced_dct     = 0;
                        s->req_conv_prop.depth       = 8;
                        s->req_conv_prop.subsampling = SUBS_420;
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

        if (show_help != 0) {
                if (s->backend.empty()) {
                        usage(show_help == 2);
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

        if (get_commandline_param("keep-pixfmt") != nullptr) {
                s->store_orig_format = true;
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

ADD_TO_PARAM("keep-pixfmt",
                "* keep-pixfmt\n"
                "  Signalize input pixel format to reciever and try\n");
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
        char out[STR_LEN];
        snprintf(out, sizeof out,
                 MOD_NAME "Codec supported pixel formats: " TBOLD("%s"),
                 get_avpixfmts_names(first));
        LOG(LOG_LEVEL_VERBOSE) << wrap_paragraph(out) << TERM_RESET "\n";
}

void print_pix_fmts(const list<enum AVPixelFormat>
                &req_pix_fmts, const enum AVPixelFormat *first) {
        print_codec_supp_pix_fmts(first);
        char out[STR_LEN] = MOD_NAME "Supported pixel formats:" TERM_BOLD;
        for (auto &c : req_pix_fmts) {
                snprintf(out + strlen(out), sizeof out - strlen(out), " %s",
                         av_get_pix_fmt_name(c));
        }
        LOG(LOG_LEVEL_VERBOSE) << wrap_paragraph(out) << TERM_RESET "\n";
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

template<typename T, bool log_err = false>
static inline bool check_av_opt_set(void *priv_data, const char *key, T val, const char *desc = nullptr) {
        string val_str;
        if constexpr (std::is_same_v<T, const char *>) {
                val_str = val;
        } else {
                val_str = to_string(val);
        }
        desc = desc ? desc : key;
        const int ret = av_opt_set(priv_data, key, val_str.c_str(), 0);
        if (ret != 0) {
                string err = string(MOD_NAME) + "Unable to set " + desc + " to " + val_str;
                print_libav_error(log_err ? LOG_LEVEL_ERROR : LOG_LEVEL_WARNING, err.c_str(), ret);
                return false;
        }
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Setting codec parameter %s to %s\n",
                desc, val_str.c_str());
        return true;
}

/// @param requested_cqp requested CQP value if >= 0, autoselect if -1
static void set_cqp(struct AVCodecContext *codec_ctx, int requested_cqp) {
        int cqp = requested_cqp;
        if (requested_cqp == -1) {
                if (strstr(codec_ctx->codec->name, "_qsv") != nullptr) {
                        cqp = strcmp(codec_ctx->codec->name, "mjpeg_qsv") == 0 ? DEFAULT_CQP_MJPEG_QSV : DEFAULT_CQP_QSV;
                } else if (strcmp(codec_ctx->codec->name, "mjpeg") == 0) {
                        cqp = DEFAULT_CQP_MPJEG;
                } else {
                        cqp = DEFAULT_CQP;
                }
        }
        codec_ctx->flags |= AV_CODEC_FLAG_QSCALE;

        if (strcmp(codec_ctx->codec->name, "mjpeg") == 0) {
                codec_ctx->qmin = codec_ctx->qmax = cqp;
                LOG(LOG_LEVEL_INFO) << MOD_NAME "Setting mjpeg qmin/qmax to " << cqp <<  "\n";
        } else if (strstr(codec_ctx->codec->name, "_qsv") != nullptr) {
                codec_ctx->global_quality = cqp;
                LOG(LOG_LEVEL_INFO) << MOD_NAME "Setting QSV global_quality to " << cqp <<  "\n";
        } else {
                if (check_av_opt_set<int>(codec_ctx->priv_data, "qp", cqp, "CQP")) {
                        LOG(LOG_LEVEL_INFO) << MOD_NAME "Setting CQP to " << cqp <<  "\n";
                }
        }
}

bool set_codec_ctx_params(struct state_video_compress_libav *s, AVPixelFormat pix_fmt, struct video_desc desc, codec_t ug_codec)
{
        bool is_x264_x265 = strstr(s->codec_ctx->codec->name, "libx26") == s->codec_ctx->codec->name;
        bool is_vaapi = regex_match(s->codec_ctx->codec->name, regex(".*_vaapi"));
        bool is_mjpeg = strstr(s->codec_ctx->codec->name, "mjpeg") != nullptr;

        double avg_bpp; // average bit per pixel
        avg_bpp = s->requested_bpp > 0.0 ? s->requested_bpp :
                codec_params[ug_codec].avg_bpp;

        bool set_bitrate = false;
        int_fast64_t bitrate = s->requested_bitrate > 0 ? s->requested_bitrate :
                desc.width * desc.height * avg_bpp * desc.fps;

        s->codec_ctx->strict_std_compliance = -2;

        // set quality
        if (s->requested_cqp >= 0 || ((is_vaapi || is_mjpeg) && s->requested_crf == -1.0 && s->requested_bitrate == 0 && s->requested_bpp == 0.0)) {
                set_cqp(s->codec_ctx, s->requested_cqp);
        } else if (s->requested_crf >= 0.0 || (is_x264_x265 && s->requested_bitrate == 0 && s->requested_bpp == 0.0)) {
                double crf = s->requested_crf >= 0.0 ? s->requested_crf : DEFAULT_X264_X265_CRF;
                if (check_av_opt_set<double>(s->codec_ctx->priv_data, "crf", crf)) {
                        log_msg(LOG_LEVEL_INFO, "[lavc] Setting CRF to %.2f.\n", crf);
                }
        } else {
                set_bitrate = true;
        }
        if (set_bitrate || s->requested_bitrate > 0) {
                s->codec_ctx->bit_rate = bitrate;
                s->codec_ctx->bit_rate_tolerance = bitrate / desc.fps * 6;
                LOG(LOG_LEVEL_INFO) << MOD_NAME << "Setting bitrate to " << format_in_si_units(bitrate) << "bps.\n";
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
        s->codec_ctx->slices = IF_NOT_UNDEF_ELSE(s->params.slices, s->codec_ctx->codec_id == AV_CODEC_ID_FFV1 ? 16 : DEFAULT_SLICE_COUNT);

        // set user supplied parameters
        for (auto const &item : s->lavc_opts) {
                if (s->blacklist_opts.count(item.first) == 1) {
                        continue;
                }
                if (!check_av_opt_set<const char *, true>(
                        s->codec_ctx->priv_data, item.first.c_str(), item.second.c_str())) {
                        return false;
                }
        }

        return true;
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
list<enum AVPixelFormat> get_requested_pix_fmts(codec_t in_codec,
                struct to_lavc_req_prop req_conv_prop) {
        if (get_commandline_param("lavc-use-codec")) {
                const char *val = get_commandline_param("lavc-use-codec");
                enum AVPixelFormat fmt = av_get_pix_fmt(val);
                if (fmt != AV_PIX_FMT_NONE) {
                        return { fmt };
                }
                req_conv_prop.force_conv_to = get_codec_from_name(val);
                if (!req_conv_prop.force_conv_to) {
                        LOG(LOG_LEVEL_FATAL) << MOD_NAME << "Wrong codec string: " << val << ".\n";
                        exit_uv(1);
                        return {};
                }
        }

        enum AVPixelFormat pixfmts[AV_PIX_FMT_NB];
        int nb_fmts = get_available_pix_fmts(in_codec, req_conv_prop, pixfmts);
        return { pixfmts, pixfmts + nb_fmts };
}

void apply_blacklist([[maybe_unused]] list<enum AVPixelFormat> &formats, [[maybe_unused]] const char *encoder_name) {
#if X2RGB10LE_PRESENT
        // blacklist AV_PIX_FMT_X2RGB10LE for NVENC - with current FFmpeg (13d04e3), it produces 10-bit 4:2:0 YUV (FF
        // macro IS_YUV444 and IS_GBRP should contain the codec - if set 1st one, picture is ok 444 YUV, second produces
        // incorrect colors). Anyways, even for the case #1 it is perhaps better to keep it blacklisted to allow
        // selection of gbrp16, which doesn't convert to YUV.
        if (strstr(encoder_name, "nvenc") != nullptr ||
            strstr(encoder_name, "qsv") != nullptr) {
                if (formats.size() == 1) {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Only one codec remaining, not blacklisting x2rgb10le!\n";
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
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Using VA-API with sw format %s\n", av_get_pix_fmt_name(pix_fmt));
        }
#endif

        if (const AVPixFmtDescriptor * desc = av_pix_fmt_desc_get(pix_fmt)) { // defaults
                s->codec_ctx->colorspace = (desc->flags & AV_PIX_FMT_FLAG_RGB) != 0U ? AVCOL_SPC_RGB : AVCOL_SPC_BT709;
                s->codec_ctx->color_range = (desc->flags & AV_PIX_FMT_FLAG_RGB) != 0U ? AVCOL_RANGE_JPEG : AVCOL_RANGE_MPEG;
        }
        get_av_pixfmt_details(pix_fmt, &s->codec_ctx->colorspace, &s->codec_ctx->color_range);

        /* open it */
        if (avcodec_open2(s->codec_ctx, codec, NULL) < 0) {
                avcodec_free_context(&s->codec_ctx);
                log_msg(LOG_LEVEL_ERROR, "[lavc] Could not open codec for pixel format %s\n", av_get_pix_fmt_name(pix_fmt));
                return false;
        }

        return true;
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

static bool configure_swscale(struct state_video_compress_libav *s, struct video_desc desc, enum AVPixelFormat sws_out_pixfmt) {
#ifndef HAVE_SWSCALE
        UNUSED(s), UNUSED(desc), UNUSED(sws_out_pixfmt);
        return false;
#else
        //get all AVPixelFormats we can convert to and pick the first
        enum AVPixelFormat pixfmts[AV_PIX_FMT_NB];
        int nb_fmts = get_available_pix_fmts(desc.color_spec, s->req_conv_prop, pixfmts);
        enum AVPixelFormat sws_in_format = nb_fmts == 0 ? AV_PIX_FMT_UYVY422 : pixfmts[0];
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Attempting to use swscale to convert from %s to %s.\n", av_get_pix_fmt_name(sws_in_format), av_get_pix_fmt_name(sws_out_pixfmt));
        if ((s->pixfmt_conversion = to_lavc_vid_conv_init(desc.color_spec, desc.width, desc.height, sws_in_format, s->conv_thread_count)) == nullptr) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get sws input conversion.\n"); // shouldn't happen normally, but user may choose imposible codec
                return false;
        }

        s->sws_ctx = getSwsContext(desc.width,
                        desc.height,
                        sws_in_format,
                        desc.width,
                        desc.height,
                        sws_out_pixfmt,
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
        s->sws_frame->format = sws_out_pixfmt;
        if (int ret = av_image_alloc(s->sws_frame->data, s->sws_frame->linesize,
                        s->sws_frame->width, s->sws_frame->height,
                        sws_out_pixfmt, 32); ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Could not allocate raw picture buffer for sws\n");
                return false;
        }

        log_msg(LOG_LEVEL_NOTICE, "[lavc] Using swscale to convert %s to %s.\n",
                        av_get_pix_fmt_name(sws_in_format),
                        av_get_pix_fmt_name(sws_out_pixfmt));
        return true;
#endif //HAVE_SWSCALE
}

AVPixelFormat
try_open_remaining_pixfmts(state_video_compress_libav *s, video_desc desc,
                           codec_t ug_codec, const AVCodec *codec,
                           set<AVPixelFormat> const &fmts_tried)
{
#ifndef HAVE_SWSCALE
        return AV_PIX_FMT_NONE;
#endif
        unsigned usable_fmt_cnt = 0;
        if (codec->pix_fmts == nullptr) {
                return AV_PIX_FMT_NONE;
        }
        for (const auto *pix = codec->pix_fmts; *pix != AV_PIX_FMT_NONE;
             ++pix) {
                usable_fmt_cnt += 1;
        }
        if (usable_fmt_cnt == fmts_tried.size()) {
                return AV_PIX_FMT_NONE; // no format to try
        }
        LOG(LOG_LEVEL_WARNING) << MOD_NAME "No direct decoder format for: "
                               << get_codec_name(desc.color_spec)
                               << ". Trying to convert with swscale instead.\n";
        for (const auto *pix = codec->pix_fmts; *pix != AV_PIX_FMT_NONE;
             ++pix) {
                const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(*pix);
                if (fmts_tried.count(*pix) == 1 || fmt_desc == nullptr ||
                    (fmt_desc->flags & AV_PIX_FMT_FLAG_HWACCEL) != 0U) {
                        continue;
                }
                AVPixelFormat curr_pix_fmt = *pix;
                if (try_open_codec(s, curr_pix_fmt, desc, ug_codec, codec)) {
                        return curr_pix_fmt;
                }
        }
        return AV_PIX_FMT_NONE;
}

static bool configure_with(struct state_video_compress_libav *s, struct video_desc desc)
{
        s->saved_desc = {};
        codec_t ug_codec = s->requested_codec_id == VIDEO_CODEC_NONE ? DEFAULT_CODEC : s->requested_codec_id;
        AVPixelFormat pix_fmt = AV_PIX_FMT_NONE;
        const AVCodec *codec = nullptr;
#ifdef HAVE_SWSCALE
        sws_freeContext(s->sws_ctx);
        s->sws_ctx = nullptr;
        av_frame_free(&s->sws_frame);
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
        list<enum AVPixelFormat> requested_pix_fmt = get_requested_pix_fmts(desc.color_spec, s->req_conv_prop);
        apply_blacklist(requested_pix_fmt, codec->name);
        auto requested_pix_fmt_it = requested_pix_fmt.cbegin();
        set<AVPixelFormat> fmts_tried;
        while ((pix_fmt = get_first_matching_pix_fmt(requested_pix_fmt_it, requested_pix_fmt.cend(), codec->pix_fmts)) != AV_PIX_FMT_NONE) {
                fmts_tried.insert(pix_fmt);
                if(try_open_codec(s, pix_fmt, desc, ug_codec, codec)){
                        break;
                }
	}

        if (pix_fmt == AV_PIX_FMT_NONE || log_level >= LOG_LEVEL_VERBOSE) {
                print_pix_fmts(requested_pix_fmt, codec->pix_fmts);
        }

        if (pix_fmt == AV_PIX_FMT_NONE && get_commandline_param("lavc-use-codec") == NULL) {
                pix_fmt = try_open_remaining_pixfmts(s, desc, ug_codec, codec,
                                                     fmts_tried);
        }

        if (pix_fmt == AV_PIX_FMT_NONE) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Unable to find suitable pixel format for: %s.\n", get_codec_name(desc.color_spec));
                if (s->req_conv_prop.subsampling != 0 || s->req_conv_prop.depth != 0 || s->req_conv_prop.rgb != -1 || get_commandline_param("lavc-use-codec") != NULL) {
                        log_msg(LOG_LEVEL_ERROR, "[lavc] Requested parameters not supported. %s\n",
                                        get_commandline_param("lavc-use-codec") != NULL ? "Do not enforce encoder codec or use a supported one." :
                                        "Do not enforce subsampling/depth/colospace or use a difffereng configuration that is feasible.");

                }
                return false;
        }

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Codec %s capabilities: 0x%08X using thread type %d, count %d\n", codec->name,
                        codec->capabilities, s->codec_ctx->thread_type, s->codec_ctx->thread_count);
        log_msg(LOG_LEVEL_INFO, "[lavc] Selected pixfmt: %s\n", av_get_pix_fmt_name(pix_fmt));
        incomp_feature_warn(INCOMP_SUBSAMPLING,
                            av_pixfmt_get_subsampling(pix_fmt));
        incomp_feature_warn(INCOMP_DEPTH,
                            av_pix_fmt_desc_get(pix_fmt)->comp[0].depth);

        s->compressed_desc = desc;
        s->compressed_desc.color_spec = ug_codec;
        s->compressed_desc.tile_count = 1;
        s->mov_avg_frames = s->mov_avg_comp_duration = 0;

        to_lavc_vid_conv_destroy(&s->pixfmt_conversion);
        if ((s->pixfmt_conversion = to_lavc_vid_conv_init(desc.color_spec, desc.width, desc.height, pix_fmt, s->conv_thread_count)) == nullptr) {
                if (!configure_swscale(s, desc, pix_fmt)) {
                        return false;
                }
        }

        // we need to store extradata for HuffYUV/FFV1 in the beginning
        if (libav_codec_has_extradata(ug_codec)) {
                s->aux_header.buf_len =
                    sizeof(uint32_t) + s->codec_ctx->extradata_size;
                *(uint32_t *) (void *) s->aux_header.buf =
                    s->codec_ctx->extradata_size;
                memcpy(s->aux_header.buf + sizeof(uint32_t),
                       s->codec_ctx->extradata, s->codec_ctx->extradata_size);
        }

        s->saved_desc = desc;
        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "Successfully reconfigured to " << desc << "\n";

        return true;
}

/// print hint to improve performance if not making it
static void check_duration(struct state_video_compress_libav *s, time_ns_t dur_pixfmt_change_ns, time_ns_t dur_total_ns)
{
        enum { REPEAT_INT_SEC = 30 };
        constexpr int mov_window = 100;
        double duration = dur_total_ns / NS_IN_SEC_DBL;
        s->mov_avg_comp_duration = (s->mov_avg_comp_duration * (mov_window - 1) + duration) / mov_window;
        s->mov_avg_frames += 1;
        if (s->mov_avg_frames < 2 * mov_window || s->mov_avg_comp_duration < 1 / s->compressed_desc.fps) {
                return;
        }
        const time_ns_t now = get_time_in_ns();
        if (now < s->duration_warn_last_print + NS_IN_SEC * REPEAT_INT_SEC) {
                return;
        }
        s->duration_warn_last_print = now;
        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Average compression time of last %d frames is %f ms but time per frame is only %f ms!\n",
                        mov_window, s->mov_avg_comp_duration * 1000, 1000 / s->compressed_desc.fps);
        string hint;
        string quality_hurt = "latency";
        if (regex_match(s->codec_ctx->codec->name, regex(".*nvenc.*"))) {
                if (s->lavc_opts.find("delay") == s->lavc_opts.end()) {
                        hint = "\"delay=<frames>\" option to NVENC compression (2 suggested)";
                }
        } if (strcmp(s->codec_ctx->codec->name, "libaom-av1") == 0) {
                if (s->lavc_opts.find("cpu-used") == s->lavc_opts.end()) {
                        hint = "\"cpu-used=8\" option for quality/speed trade-off to AOM AV1 compression (values 0-8 allowed)";
                        quality_hurt = "quality";
                }
        } else if ((s->codec_ctx->thread_type & FF_THREAD_SLICE) == 0 && (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_FRAME_THREADS) != 0) {
                hint = "\"threads=<n>FS\" option with small <n> or 0 (nr of logical cores) to compression";
        } else if (s->codec_ctx->thread_count == 1 && (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_OTHER_THREADS) != 0) {
                hint = "\"threads=<n>\" option with small <n> or 0 (nr of logical cores) to compression";
        }
        if (!hint.empty()) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Consider adding " << hint << " to increase throughput at the expense of " << quality_hurt << ".\n";
        }

        bool src_rgb = codec_is_a_rgb(s->saved_desc.color_spec);
        bool dst_rgb = av_pix_fmt_desc_get(s->codec_ctx->pix_fmt)->flags & AV_PIX_FMT_FLAG_RGB;
        if (src_rgb != dst_rgb && dur_pixfmt_change_ns / NS_IN_SEC_DBL > s->mov_avg_comp_duration / 4) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME "Also pixfmt change of last frame took " << dur_pixfmt_change_ns / NS_IN_MS_DBL << " ms.\n"
                        "Consider adding \"--conv-policy cds\" to prevent color space conversion.\n";
        }
}

static void write_orig_format(struct video_frame *compressed_frame, codec_t orig_pixfmt) {
        if (compressed_frame == nullptr) {
                return;
        }
        if (compressed_frame->color_spec != H264 && compressed_frame->color_spec != H265) {
                log_msg_once(LOG_LEVEL_ERROR, to_fourcc('L', 'W', 'P', 'T'), MOD_NAME
                                "Currently cannot store input format to different compression stream than H.264/HEVC\n");
                return;
        }
        char *data = compressed_frame->tiles[0].data;
        unsigned int *data_len = &compressed_frame->tiles[0].data_len;
        if (compressed_frame->color_spec == H264) {
                unsigned char sei_nal_prefix[] = { START_CODE_3B, H264_NAL_SEI_PREFIX, sizeof (unsigned char []) { UG_ORIG_FORMAT_ISO_IEC_11578_GUID } + 1, UG_ORIG_FORMAT_ISO_IEC_11578_GUID };
                memcpy(data + *data_len, sei_nal_prefix, sizeof sei_nal_prefix);
                *data_len += sizeof sei_nal_prefix;
        } else {
                unsigned char sei_nal_prefix[] = { START_CODE_3B, HEVC_NAL_SEI_PREFIX, sizeof (unsigned char []) { UG_ORIG_FORMAT_ISO_IEC_11578_GUID } + 1, UG_ORIG_FORMAT_ISO_IEC_11578_GUID };
                memcpy(data + *data_len, sei_nal_prefix, sizeof sei_nal_prefix);
                *data_len += sizeof sei_nal_prefix;
        }

        struct pixfmt_desc desc = get_pixfmt_desc(orig_pixfmt);
        unsigned subs_a = (desc.subsampling / 100 % 10);
        unsigned subs_b = (desc.subsampling / 10 % 10);
        unsigned subs_v = subs_a - 1; // [4 2 1] -> [3 1 0]
        unsigned subs_h = !!subs_b; // 1 - vertically subsampled, 0 - not
        unsigned rgb = desc.rgb;
        uint8_t format = ((desc.depth - 8) / 2) << 4U | subs_v << 2U | subs_h << 1U | rgb;
        memcpy(data + *data_len, &format, sizeof format);
        *data_len += sizeof format;
        uint8_t eob = 0x80; // ? end of bytestream ? (doesn't work without)
        memcpy(data + *data_len, &eob, sizeof eob);
        *data_len += sizeof eob;
}

enum {
        HOUSEKEEP_INTERVAL = 100,
};
void
store_metadata(state_video_compress_libav *s, const struct video_frame *f,
               int64_t pts)
{
        char *metadata_store = s->metadata_storage[pts];
        assert((int) s->metadata_storage.size() <=
               5 * HOUSEKEEP_INTERVAL); // something bad is happening
        vf_store_metadata(f, metadata_store);
}
void
restore_metadata(state_video_compress_libav *s, struct video_frame *out,
                 int64_t pts)
{
        auto it = s->metadata_storage.find(pts);
        if (it != s->metadata_storage.end()) {
                vf_restore_metadata(out, it->second);
        } else {
                debug_msg("Metadata for frame %" PRIu64 " not found!\n", pts);
        }
        // batch remove frames older than HOUSEKEEP_INTERVAL
        if (pts > HOUSEKEEP_INTERVAL && pts % HOUSEKEEP_INTERVAL == 0) {
                s->metadata_storage.erase(
                    s->metadata_storage.begin(),
                    s->metadata_storage.lower_bound(pts - HOUSEKEEP_INTERVAL));
        }
}

void
store_sps_pps_vps(state_video_compress_libav *s, AVPacket *pkt)
{
        if (s->compressed_desc.color_spec != H264 &&
            s->compressed_desc.color_spec != H265) {
                return;
        }
        if (s->aux_header.buf_len > 0) {
                return;
        }
        const bool is_hevc         = s->compressed_desc.color_spec == H265;
        const int  h264_nalu_req[] = { NAL_H264_SPS, NAL_H264_PPS, 0 };
        const int  hevc_nalu_req[] = { NAL_HEVC_VPS, NAL_HEVC_SPS, NAL_HEVC_PPS,
                                       0 };
        const int *nalu_req        = is_hevc ? hevc_nalu_req : h264_nalu_req;
        const unsigned char *const first_nal =
            rtpenc_get_first_nal(s->pkt->data, s->pkt->size, is_hevc);
        const unsigned char *nal    = first_nal;
        int                  i      = 0;
        const unsigned char *endptr = nullptr;
        while (nal != nullptr) {
                if (NALU_HDR_GET_TYPE(nal[0], is_hevc) != nalu_req[i++]) {
                        return;
                }
                if (nalu_req[i] == 0) { // correct seq of NALU
                        break;
                }
                nal = rtpenc_get_next_nal(nal, pkt->size - (nal - pkt->data),
                                          &endptr);
        }
        if (nalu_req[i] != 0) {
                return;
        }
        const char start_code[] = { START_CODE_4B };
        memcpy(s->aux_header.buf, start_code, sizeof start_code);
        memcpy(s->aux_header.buf + sizeof start_code, first_nal,
               endptr - first_nal);
        s->aux_header.buf_len = sizeof start_code + endptr - first_nal;
}

auto out_vf_from_pkt(state_video_compress_libav *s, AVPacket *pkt) {
        shared_ptr<video_frame> out;
        if (pkt->size == 0) { // videotoolbox returns sometimes frames with
                              // pkt->size == 0 but got_output == true
                return out;
        }

        out = shared_ptr<video_frame>(
            vf_alloc_desc(s->compressed_desc), [](struct video_frame *frame) {
                    free(frame->tiles[0].data);
                    vf_free(frame);
            });
        if (s->compressed_desc.color_spec == PRORES) {
                assert(s->codec_ctx->codec_tag != 0);
                out->color_spec = get_codec_from_fcc(s->codec_ctx->codec_tag);
        }

        restore_metadata(s, out.get(), pkt->pts);
        out->tiles[0].data_len = s->aux_header.buf_len + pkt->size;
        out->tiles[0].data     = (char *) malloc(out->tiles[0].data_len);
        memcpy(out->tiles[0].data, s->aux_header.buf, s->aux_header.buf_len);
        memcpy(out->tiles[0].data + s->aux_header.buf_len, pkt->data,
               pkt->size);
        if (s->params.header_inserter_req == 1) {
                store_sps_pps_vps(s, pkt);
        }

        av_packet_unref(s->pkt);

        return out;
}

static shared_ptr<video_frame> libavcodec_compress_tile(struct module *mod, shared_ptr<video_frame> tx)
{
        auto *s = (state_video_compress_libav *) mod->priv_data;
        list<shared_ptr<void>> cleanup_callbacks; // at function exit handlers

        libavcodec_check_messages(s);

        if (tx && !video_desc_eq_excl_param(video_desc_from_frame(tx.get()),
                                            s->saved_desc, PARAM_TILE_COUNT)) {
                cleanup(s);
                if (!configure_with(s, video_desc_from_frame(tx.get()))) {
                        return {};
                }
        }

        if (!tx) { // reading further encoded frames
                const int ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
                if (ret == 0) {
                        return out_vf_from_pkt(s, s->pkt);
                }
                if (ret != AVERROR(EAGAIN)) {
                        print_libav_error(LOG_LEVEL_WARNING,
                                          MOD_NAME "Receive packet error", ret);
                }
                return {};
        }

        time_ns_t t0 = get_time_in_ns();
        struct AVFrame *frame = to_lavc_vid_conv(s->pixfmt_conversion, tx->tiles[0].data);
        if (!frame) {
                return {};
        }
        time_ns_t t1 = get_time_in_ns();

        debug_file_dump("lavc-avframe", serialize_video_avframe, frame);
#ifdef HWACC_VAAPI
        if(s->hwenc){
                av_hwframe_transfer_data(s->hwframe, frame, 0);
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
        frame->pts = s->cur_pts++;
        store_metadata(s, tx.get(), frame->pts);
        if (int ret = avcodec_send_frame(s->codec_ctx, frame)) {
                print_libav_error(LOG_LEVEL_WARNING, "[lavc] Error encoding frame", ret);
                return {};
        }
        int ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
        shared_ptr<video_frame> out{};
        if (ret == 0) {
                out = out_vf_from_pkt(s, s->pkt);
        }
        if (ret != AVERROR(EAGAIN) && ret != 0) {
                print_libav_error(LOG_LEVEL_WARNING, "[lavc] Receive packet error", ret);
        }
        time_ns_t t3 = get_time_in_ns();
        LOG(LOG_LEVEL_DEBUG2) << MOD_NAME << "duration pixfmt change: "
                << (t1 - t0) / NS_IN_SEC_DBL <<
                " s, dump+swscale " << (t2 - t1) / (double) NS_IN_SEC <<
                " s, compression " << (t3 - t2) / (double) NS_IN_SEC << " s\n";
        check_duration(s, t1 - t0, t3 - t0);

        if (s->store_orig_format) {
                write_orig_format(out.get(), tx->color_spec);
        }

        return out;
}

static void cleanup(struct state_video_compress_libav *s)
{
        if(s->codec_ctx) {
		int ret = avcodec_send_frame(s->codec_ctx, NULL);
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
                avcodec_free_context(&s->codec_ctx);
        }

        av_frame_free(&s->hwframe);

#ifdef HAVE_SWSCALE
        sws_freeContext(s->sws_ctx);
        s->sws_ctx = nullptr;
        av_frame_free(&s->sws_frame);
#endif //HAVE_SWSCALE
}

static void libavcodec_compress_done(struct module *mod)
{
        struct state_video_compress_libav *s = (struct state_video_compress_libav *) mod->priv_data;

        cleanup(s);

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
                } else if (strcmp(codec_ctx->codec->name, "libaom-av1") == 0 ||
                                strcmp(codec_ctx->codec->name, "librav1e") == 0) {
                        codec_ctx->thread_count = thread::hardware_concurrency();
                }
        } else if (codec_ctx->thread_type != 0) {
                codec_ctx->thread_count = thread::hardware_concurrency();
        }
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Setting thread count to %d, type: %s\n", codec_ctx->thread_count, lavc_thread_type_to_str(codec_ctx->thread_type));
}

static void setparam_default(AVCodecContext *codec_ctx, struct setparam_param * /* param */)
{
        if (codec_ctx->codec->id == AV_CODEC_ID_JPEG2000) {
                log_msg(LOG_LEVEL_WARNING, "[lavc] J2K support is experimental and may be broken!\n");
        }
}

static void setparam_jpeg(AVCodecContext *codec_ctx, struct setparam_param * /* param */)
{
        if (strcmp(codec_ctx->codec->name, "mjpeg") == 0) {
                check_av_opt_set<const char *>(codec_ctx->priv_data, "huffman", "default", "Huffman tables");
        }
        if (strcmp(codec_ctx->codec->name, "mjpeg_qsv") == 0) {
                check_av_opt_set<int>(codec_ctx->priv_data, "async_depth", 1);
        }
}

static void configure_amf([[maybe_unused]] AVCodecContext *codec_ctx, [[maybe_unused]] struct setparam_param *param) {
        check_av_opt_set(codec_ctx->priv_data, "rc", DEFAULT_AMF_RC);
        const char *const usage = codec_ctx->codec->id == AV_CODEC_ID_AV1
                                      ? DEFAULT_AMF_USAGE_AV1
                                      : DEFAULT_AMF_USAGE;
        check_av_opt_set(codec_ctx->priv_data, "usage", usage);
        if (codec_ctx->codec->id == AV_CODEC_ID_AV1 ||
            codec_ctx->codec->id == AV_CODEC_ID_HEVC) {
                check_av_opt_set<const char *>(codec_ctx->priv_data, "header_insertion_mode", "gop", "header_insertion_mode for AMF");
        } else if ("h264_amf"s == codec_ctx->codec->name) {
                check_av_opt_set<int>(codec_ctx->priv_data, "header_spacing", 1);
                if (param->header_inserter_req == -1) {
                        MSG(INFO, "Auto-enabling header inserter for %s.\n",
                            codec_ctx->codec->name);
                        param->header_inserter_req = 1;
                }
        }
}

void
configure_mf(AVCodecContext                         *codec_ctx,
             [[maybe_unused]] struct setparam_param *param)
{
        check_av_opt_set<const char *>(codec_ctx->priv_data, "rate_control",
                                       "cbr");
        check_av_opt_set<const char *>(codec_ctx->priv_data, "scenario",
                                       "video_conference");
        check_av_opt_set<int>(codec_ctx->priv_data, "hw_encoding", 1);
}

void
incomp_feature_warn(enum incomp_feature f, int req_val)
{
        const char *disable_opt = nullptr;
        switch (f) {
        case INCOMP_DEPTH:
                if (req_val == 8) {
                        return;
                }
                MSG(WARNING,
                    "Selected color depth %d b may not be supported by HW "
                    "decoders.\n",
                    req_val);
                disable_opt = ":depth=8";
                break;
        case INCOMP_INTRA_REFRESH:
                if (req_val != -1) {
                        return;
                }
                MSG(WARNING, "Auto-enabling intra-refresh "
                             "which may not be supported by HW decoders.\n");
                disable_opt = ":disable_intra_refresh";
                break;
        case INCOMP_INTERLACED_DCT:
                if (req_val != -1) {
                        return;
                }
                MSG(WARNING, "Auto-enabling interlaced DCT "
                             "which may not be supported by HW decoders.\n");
                disable_opt = ":disable_interlaced_dct";
                break;
        case INCOMP_SUBSAMPLING:
                if (req_val == SUBS_420) {
                        return;
                }
                MSG(WARNING,
                    "Selected pixfmt has subsampling %d:%d:%d, "
                    "which is usually not supported by hw. decoders\n",
                    (req_val / 1000), ((req_val / 100) % 10),
                    ((req_val / 10) % 10));
                disable_opt = ":subs=420";
                break;
        }
        MSG(INFO, "Use '%s' or ':safe' to disable.\n", disable_opt);
}

ADD_TO_PARAM(
    "lavc-rc-buffer-size-factor",
    "* lavc-rc-buffer-size-factor=<val>\n"
    "  Multiplier how much can individual frame overshot average size (default "
    "x264/5: " TOSTRING(DEFAULT_X26X_RC_BUF_SIZE_FACTOR) ", nvenc: " TOSTRING(
        DEFAULT_NVENC_RC_BUF_SIZE_FACTOR) ").\n");
static void
configure_x264_x265(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        const char *preset = "ultrafast";
        if (strstr(codec_ctx->codec->name, "libx264") ==
                codec_ctx->codec->name &&
            (param->desc.width * param->desc.height * param->desc.fps <=
             FLW_THRESH)) {
                preset = "veryfast";
        }
        check_av_opt_set<const char *>(codec_ctx->priv_data, "preset", preset);

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

        if (param->desc.interlacing == INTERLACED_MERGED &&
            param->interlaced_dct != 0 &&
            codec_ctx->codec_id == AV_CODEC_ID_H264) {
                incomp_feature_warn(INCOMP_INTERLACED_DCT, param->interlaced_dct);
                codec_ctx->flags |= AV_CODEC_FLAG_INTERLACED_DCT;
        }

        string x265_params;
        if (param->lavc_opts.find("x265-params") != param->lavc_opts.end()) {
                x265_params = param->lavc_opts.at("x265-params");
                param->blacklist_opts.insert("x265-params");
        }
        auto x265_params_append = [&](const string &key, const string &val) {
                if (x265_params.find(key) == string::npos) {
                        x265_params += (x265_params.empty() ? "" : ":") + key + "=" + val;
                }
        };
        x265_params_append("keyint", to_string(codec_ctx->gop_size));
        /// turn on periodic intra refresh, unless explicitely disabled
        if (param->periodic_intra != 0) {
                incomp_feature_warn(INCOMP_INTRA_REFRESH, param->periodic_intra);
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

static void configure_qsv_h264_hevc(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        check_av_opt_set<const char *>(codec_ctx->priv_data, "preset",
                                       DEFAULT_QSV_PRESET);
        check_av_opt_set<const char *>(codec_ctx->priv_data, "scenario", "livestreaming");
        check_av_opt_set<int>(codec_ctx->priv_data, "async_depth", 1);

        if (param->periodic_intra != 0) {
                incomp_feature_warn(INCOMP_INTRA_REFRESH, param->periodic_intra);
                check_av_opt_set<const char *>(codec_ctx->priv_data, "int_ref_type", "vertical");
                check_av_opt_set<int>(codec_ctx->priv_data, "int_ref_cycle_size", 20);
        }

        if (param->desc.interlacing == INTERLACED_MERGED && param->interlaced_dct != 0) {
                incomp_feature_warn(INCOMP_INTERLACED_DCT, param->interlaced_dct);
                codec_ctx->flags |= AV_CODEC_FLAG_INTERLACED_DCT;
        }

        // rate control
        const char *rc = DEFAULT_QSV_RC;
        if (auto it = param->lavc_opts.find("rc"); it != param->lavc_opts.end()) {
                rc = it->second.c_str();
                param->blacklist_opts.insert("rc");
        }
        if (strcmp(rc, "help") == 0) {
                col() << "\n\n"
                      << SBOLD("See codec-specific usage for available RCs (``-c "
                               "libavcodec:encoder=<enc>:help`).")
                      << "\n\n\n";
                exit_uv(0);
        } else if (strcasecmp(rc, "cbr") == 0) {
                codec_ctx->rc_max_rate = codec_ctx->bit_rate;
                // no look-ahead and rc_max_rate == bit_rate result in use of CBR for QSV
        } else if (strcasecmp(rc, "cqp") == 0) {
                codec_ctx->flags |= AV_CODEC_FLAG_QSCALE;
        } else if (strcasecmp(rc, "icq") == 0 || strcasecmp(rc, "qvbr") == 0) {
                codec_ctx->global_quality = codec_ctx->global_quality <= 0 ? DEFAULT_CQP : codec_ctx->global_quality;
                codec_ctx->flags &= ~AV_CODEC_FLAG_QSCALE;
                if (strcasecmp(rc, "qvbr") == 0) {
                        assert(codec_ctx->bit_rate > 0);
                        codec_ctx->rc_max_rate = 23 * codec_ctx->bit_rate / 20; // 115%; if rc_max_rate == bit_rate, FF would set CBR
                }
        } else if (strcasecmp(rc, "vbr") == 0) { // no options needed
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown/unsupported RC %s. Please report to %s if you need some mode added.\n",
                                rc, PACKAGE_BUGREPORT);
                exit_uv(1);
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

static void configure_aom_av1(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        auto && usage = get_map_val_or_default<string, string>(param->lavc_opts, "usage", "realtime");
        check_av_opt_set<const char *>(codec_ctx->priv_data, "usage", usage.c_str());
        check_av_opt_set<const char *>(codec_ctx->priv_data, "tiles", "8x8");
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
                incomp_feature_warn(INCOMP_INTRA_REFRESH, param->periodic_intra);
                check_av_opt_set<int>(codec_ctx->priv_data, "intra-refresh", 1);
        }

        check_av_opt_set<const char *>(codec_ctx->priv_data, "rc", DEFAULT_NVENC_RC);
        check_av_opt_set<int>(codec_ctx->priv_data, "spatial_aq", 0);
        check_av_opt_set<int>(codec_ctx->priv_data, "gpu", cuda_devices[0]);
        check_av_opt_set<int>(codec_ctx->priv_data, "delay", 0); // 2'd increase throughput 2x at expense of higher latency
        check_av_opt_set<int>(codec_ctx->priv_data, "zerolatency", 1, "zero latency operation (no reordering delay)");
        check_av_opt_set<const char *>(codec_ctx->priv_data, "b_ref_mode", "disabled", 0);
        codec_ctx->rc_max_rate = codec_ctx->bit_rate;
        double lavc_rc_buffer_size_factor = DEFAULT_NVENC_RC_BUF_SIZE_FACTOR;
        if (const char *val = get_commandline_param("lavc-rc-buffer-size-factor")) {
                lavc_rc_buffer_size_factor = stof(val);
        } else {
                LOG(LOG_LEVEL_WARNING)
                    << MOD_NAME
                    "To reduce NVENC pulsation, you can try \"--param "
                    "lavc-rc-buffer-size-factor=0\""
                    " or a small number. 0 or higher value (than default "
                    << DEFAULT_NVENC_RC_BUF_SIZE_FACTOR
                    << ") may cause frame drops on receiver.\n";
        }
        codec_ctx->rc_buffer_size =
            (int) ((double) codec_ctx->rc_max_rate / param->desc.fps *
                   lavc_rc_buffer_size_factor);
        if (param->desc.interlacing == INTERLACED_MERGED && param->interlaced_dct == 1) {
                codec_ctx->flags |= AV_CODEC_FLAG_INTERLACED_DCT;
        }
}

static void configure_rav1e(AVCodecContext *codec_ctx, struct setparam_param * /*param*/)
{
        check_av_opt_set<const char *>(codec_ctx->priv_data, "rav1e-params", "low_latency=true");
        check_av_opt_set<const char *>(codec_ctx->priv_data, "speed", "10");
        check_av_opt_set<const char *>(codec_ctx->priv_data, "tiles", "64");
}

static void configure_svt(AVCodecContext *codec_ctx, struct setparam_param *param)
{
        // see FFMPEG modules' sources for semantics
        if (codec_ctx->codec_id != AV_CODEC_ID_AV1) {
                set_forced_idr(
                    codec_ctx,
                    strcmp(codec_ctx->codec->name, "libsvt_hevc") == 0 ? 0 : 1);
        }

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
                const char *preset =
                    param->desc.width * param->desc.height * param->desc.fps <=
                            FLW_THRESH
                        ? "9"
                        : "11";
                check_av_opt_set<const char *>(codec_ctx->priv_data, "preset",
                                               preset);
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
        } else if (regex_match(codec_ctx->codec->name, regex(".*_mf"))) {
                configure_mf(codec_ctx, param);
        } else if (regex_match(codec_ctx->codec->name, regex(".*_vaapi"))) {
                configure_vaapi(codec_ctx, param);
        } else if (strncmp(codec_ctx->codec->name, "libx264", strlen("libx264")) == 0 || // libx264 and libx264rgb
                        strcmp(codec_ctx->codec->name, "libx265") == 0) {
                configure_x264_x265(codec_ctx, param);
        } else if (regex_match(codec_ctx->codec->name, regex(".*nvenc.*"))) {
                configure_nvenc(codec_ctx, param);
        } else if (strcmp(codec_ctx->codec->name, "h264_omx") == 0)  {
                if (param->header_inserter_req == -1) {
                        MSG(INFO, "Auto-enabling header inserter for %s.\n",
                            codec_ctx->codec->name);
                        param->header_inserter_req = 1; // (untested)
                }
        } else if (strcmp(codec_ctx->codec->name, "h264_qsv") == 0 ||
                        strcmp(codec_ctx->codec->name, "hevc_qsv") == 0) {
                configure_qsv_h264_hevc(codec_ctx, param);
        } else if (strcmp(codec_ctx->codec->name, "libaom-av1") == 0) {
                configure_aom_av1(codec_ctx, param);
        } else if (strcmp(codec_ctx->codec->name, "librav1e") == 0) {
                configure_rav1e(codec_ctx, param);
        } else if (strstr(codec_ctx->codec->name, "libsvt") == codec_ctx->codec->name) {
                configure_svt(codec_ctx, param);
        } else {
                log_msg(LOG_LEVEL_WARNING, "[lavc] Warning: Unknown encoder %s. Using default configuration values.\n", codec_ctx->codec->name);
        }
}

string
get_opt_default_value(const AVOption *opt)
{
        switch (opt->type) {
        case AV_OPT_TYPE_FLOAT:
        case AV_OPT_TYPE_DOUBLE:
                return to_string(opt->default_val.dbl) + " (float)";
        case AV_OPT_TYPE_CONST:
        case AV_OPT_TYPE_INT64:
        case AV_OPT_TYPE_INT:
        case AV_OPT_TYPE_BOOL:
                return to_string(opt->default_val.i64);
        case AV_OPT_TYPE_STRING:
                return opt->default_val.str == nullptr
                           ? ""
                           : string("\"") + opt->default_val.str + "\"";
        default:
                return "";
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
                const char *indent = "\t\t* ";
                if (opt->offset != 0) {
                        default_val = get_opt_default_value(opt);
                        if (!default_val.empty()) {
                                default_val = "default " + default_val;
                        }
                        indent = "\t- ";
                }
                string help_str;
                if (opt->help != nullptr && strlen(opt->help) > 0) {
                        help_str = " - "s + opt->help;
                        if (!default_val.empty()) {
                                if (strstr(opt->help, "default") == nullptr) {
                                        help_str += ", ";
                                } else {
                                        default_val = "";
                                }
                        }
                }
                col() << indent << SBOLD(opt->name) <<  help_str << default_val << "\n";
                opt++;
        }
        if (name == "libx264" || name == "libx265") {
                col() << "(options for " << SBOLD(name.substr(3) << "-params") << " should be actually separated by '\\:', not ':' as indicated above)\n";
        }
        if (name == "hevc_qsv" || name == "h264_qsv") {
                col() << "\n\t- " << SBOLD("rc") << " - [UltraGrid specific] rate control mode: " << SBOLD("cbr") << ", " << SBOLD("cqp") << ", " << SBOLD("icq") << ", " << SBOLD("qvbr") << " or " << SBOLD("vbr") << "\n";
        }
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

const struct video_compress_info lavc_info = {
        "lavc",
        libavcodec_compress_init,
        nullptr,
        libavcodec_compress_tile,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
};

REGISTER_MODULE(libavcodec, &libavcodec_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
REGISTER_HIDDEN_MODULE(lavc, &lavc_info, LIBRARY_CLASS_VIDEO_COMPRESS,
                       VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace


/* vim: set expandtab sw=8 tw=120: */
