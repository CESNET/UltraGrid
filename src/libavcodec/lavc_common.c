/**
 * @file   lavc_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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
/**
 * @file
 * References:
 * 1. [v210](https://wiki.multimedia.cx/index.php/V210)
 *
 * @todo
 * Some conversions to RGBA ignore RGB-shifts - either fix that or deprecate RGB-shifts
 */

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "debug.h"
#include "host.h"
#include "libavcodec/lavc_common.h"
#include <libavutil/channel_layout.h>
#include "types.h"
#include "utils/macros.h"
#include "video.h"

#define MOD_NAME "[lavc_common] "

//
// UG <-> FFMPEG format translations
//
static const struct {
        enum AVCodecID av;
        codec_t uv;
} av_to_uv_map[] = {
        { AV_CODEC_ID_H264,     H264             },
        { AV_CODEC_ID_HEVC,     H265             },
        { AV_CODEC_ID_MJPEG,    JPEG             },
        { AV_CODEC_ID_JPEG2000, J2K              },
        { AV_CODEC_ID_VP8,      VP8              },
        { AV_CODEC_ID_VP9,      VP9              },
        { AV_CODEC_ID_HUFFYUV,  HFYU             },
        { AV_CODEC_ID_FFV1,     FFV1             },
        { AV_CODEC_ID_AV1,      AV1              },
        { AV_CODEC_ID_PRORES,   PRORES           },
        { AV_CODEC_ID_PRORES,   PRORES_4444      },
        { AV_CODEC_ID_PRORES,   PRORES_4444_XQ   },
        { AV_CODEC_ID_PRORES,   PRORES_422_HQ    },
        { AV_CODEC_ID_PRORES,   PRORES_422       },
        { AV_CODEC_ID_PRORES,   PRORES_422_PROXY },
        { AV_CODEC_ID_PRORES,   PRORES_422_LT    },
        { AV_CODEC_ID_APV,      APV              },
        { AV_CODEC_ID_JPEGXS,   JPEG_XS          },
};

codec_t get_av_to_ug_codec(enum AVCodecID av_codec)
{
        for (unsigned int i = 0; i < sizeof av_to_uv_map / sizeof av_to_uv_map[0]; ++i) {
                if (av_to_uv_map[i].av == av_codec) {
                        return av_to_uv_map[i].uv;
                }
        }
        return VIDEO_CODEC_NONE;
}

enum AVCodecID get_ug_to_av_codec(codec_t ug_codec)
{
        for (unsigned int i = 0; i < sizeof av_to_uv_map / sizeof av_to_uv_map[0]; ++i) {
                if (av_to_uv_map[i].uv == ug_codec) {
                        return av_to_uv_map[i].av;
                }
        }
        return AV_CODEC_ID_NONE;
}

//
// utility functions
//
void print_libav_error(int verbosity, const char *msg, int rc) {
        char errbuf[1024];
        av_strerror(rc, errbuf, sizeof(errbuf));

        log_msg(verbosity, "%s: %s\n", msg, errbuf);
}

void printf_libav_error(int verbosity, int rc, const char *msg, ...) {
        char message[1024];

        va_list ap;
        va_start(ap, msg);
        vsnprintf(message, sizeof message, msg, ap);
        va_end(ap);

        print_libav_error(verbosity, message, rc);
}

bool libav_codec_has_extradata(codec_t codec) {
        return codec == HFYU || codec == FFV1;
}

static inline int av_to_uv_log(int level) {
        enum {
                AV_TO_UV_MULT = 8, // FF uses multiples of 8
        };
        if (level <= AV_LOG_PANIC) {   // AV_LOG_QUIET(-8) -> LOG_LEVEL_QUIET(0)
                level += AV_TO_UV_MULT;// AV_LOG_PANIC(0) -> LOG_LEVEL_FATAL(1)
        }
        // AV_LOG_FATAL(8), ERROR and WARNING map directly to UG counterparts /8
        if (level >= AV_LOG_INFO) {     // AV_LOG_INFO(32) -> LOG_LEVEL_INFO(5)
                level += AV_TO_UV_MULT; // and so on for the rest
        }
        return level / AV_TO_UV_MULT;
}

/**
 * Filters out annoying messages that should not be passed to UltraGrid logger,
 * eg. complains on JPEG APP markers that FFmpeg decoder almost doesn't use.
 * @returns 0 - should be printed; 1 - filtered
 */
static _Bool av_log_filter(const char *ff_module_name, const char *fmt) {
        if (ff_module_name && strcmp(ff_module_name, "mjpeg") == 0 && strstr(fmt, "APP") != NULL) {
                return 1;
        }
        return 0;
}

static void av_log_ug_callback(void *avcl, int av_level, const char *fmt, va_list vl) {
        int level = av_to_uv_log(av_level);
        if (level > log_level) {
                return;
        }
        // avcl handling is taken from av_log_default_callback
        AVClass* avc = avcl ? *(AVClass **) avcl : NULL;
        const char *ff_module_name = avc ? avc->item_name(avcl) : NULL;
        if (av_log_filter(ff_module_name, fmt)) {
                return;
        }
        static _Thread_local char buf[STR_LEN];
        if (strlen(buf) == 0) {
                snprintf(buf + strlen(buf), sizeof buf - strlen(buf), "[lavc");
                if (ff_module_name) {
                        snprintf(buf + strlen(buf), sizeof buf - strlen(buf),
                                 " %s @ %p", ff_module_name, avcl);
                }
                snprintf(buf + strlen(buf), sizeof buf - strlen(buf), "] ");
        }
        vsnprintf(buf + strlen(buf), sizeof buf - strlen(buf), fmt, vl);
        if (buf[strlen(buf) - 1] != '\n' && strlen(buf) < sizeof buf - 1) {
                return;
        }
        if (strlen(buf) == sizeof buf - 1 && buf[strlen(buf) - 1] != '\n') {
                MSG(WARNING, "logger buffer full! flushing output:\n");
        }
        log_msg(level, "%s", buf);
        buf[0] = '\0';
}

ADD_TO_PARAM("lavc-log-level",
                "* lavc-log-level=<num>\n"
                "  Set libavcodec log level (FFmpeg range semantics, bypasses UG logger)\n"
                " - 'D' - use FFmpeg default log handler\n");
/// Sets specified log level either given explicitly or from UG-wide log_level
void ug_set_av_logging() {
        const char *param = get_commandline_param("lavc-log-level");
        if (param == NULL) {
                av_log_set_callback(av_log_ug_callback);
                return;
        }
        char *endptr = NULL;
        const long av_log_level = strtol(param, &endptr, 10);
        if (av_log_level < AV_LOG_QUIET || av_log_level > AV_LOG_TRACE ||
            endptr == param) {
                MSG(ERROR, "Wrong log level for FFmpeg '%s'\n", param);
        }
        av_log_set_level((int) av_log_level);
}

/// @returns subsampling in 'JabA' format (compatible with @ref get_subsamping)
int av_pixfmt_get_subsampling(enum AVPixelFormat fmt) {
        const struct AVPixFmtDescriptor *pd = av_pix_fmt_desc_get(fmt);
        if (pd->log2_chroma_w == 0 && pd->log2_chroma_h == 0) {
                return 4440;
        }
        if (pd->log2_chroma_w == 1 && pd->log2_chroma_h == 0) {
                return 4220;
        }
        if (pd->log2_chroma_w == 1 && pd->log2_chroma_h == 1) {
                return 4200;
        }
        return 0; // other (todo)
}

struct pixfmt_desc av_pixfmt_get_desc(enum AVPixelFormat pixfmt) {
        struct pixfmt_desc ret = { 0 };
        const struct AVPixFmtDescriptor *avd = av_pix_fmt_desc_get(pixfmt);
        ret.depth = avd->comp[0].depth;
        ret.rgb = avd->flags & AV_PIX_FMT_FLAG_RGB;
        ret.subsampling = av_pixfmt_get_subsampling(pixfmt);
        return ret;
}


void lavd_flush(AVCodecContext *codec_ctx) {
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
        int ret = 0;
        char errbuf[AV_ERROR_MAX_STRING_SIZE];
        ret = avcodec_send_packet(codec_ctx, NULL);
        if (ret != 0) {
                av_strerror(ret, errbuf, sizeof errbuf);
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME
                        "lavd_flush send - unexpected return value: %s (%d)\n",
                        errbuf, ret);
        }
        AVFrame *frame = av_frame_alloc();
        do {
                ret = avcodec_receive_frame(codec_ctx, frame);
        } while (ret >= 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN));
        if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
                av_strerror(ret, errbuf, sizeof errbuf);
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME
                        "lavd_flush recv - unexpected return value: %s (%d)\n",
                        errbuf, ret);
        }
        av_frame_free(&frame);
#else
        UNUSED(codec_ctx);
#endif
}
void print_decoder_error(const char *mod_name, int rc) {
        char buf[1024];
        switch (rc) {
        case 0:
                break;
        case EAGAIN:
                log_msg(LOG_LEVEL_VERBOSE,
                        "%sNo frame returned - needs more input data.\n",
                        mod_name);
                break;
        case EINVAL:
                log_msg(LOG_LEVEL_ERROR, "%sDecoder in invalid state!\n",
                        mod_name);
                break;
        default:
                av_strerror(rc, buf, sizeof buf);
                log_msg(LOG_LEVEL_WARNING,
                        "%sError while decoding frame (rc == %d): %s.\n",
                        mod_name, rc, buf);
                break;
        }
}

const char *lavc_thread_type_to_str(int thread_type) {
        static _Thread_local char buf[128];
        memset(buf, 0, sizeof buf);
        if ((thread_type & FF_THREAD_FRAME) != 0) {
                strncpy(buf, "thread", sizeof buf - 1);
        }
        if ((thread_type & FF_THREAD_SLICE) != 0) {
                if (strlen(buf) > 0) {
                        strncpy(buf, ", ", sizeof buf - 1);
                }
                strncpy(buf, "slice", sizeof buf - 1);
        }
        if (strlen(buf) == 0) {
                return "(other)";
        }
        return buf;
}

struct audio_desc
audio_desc_from_av_frame(const AVFrame *frm)
{
        struct audio_desc desc = { 0 };
        desc.bps = av_get_bytes_per_sample(frm->format);
 #if FF_API_NEW_CHANNEL_LAYOUT
        desc.ch_count = frm->ch_layout.nb_channels;
#else
        desc.ch_count = av_get_channel_layout_nb_channels(frm->channel_layout);
#endif
        desc.codec = AC_PCM;
        desc.sample_rate = frm->sample_rate;

        return desc;
}

enum AVSampleFormat
audio_bps_to_av_sample_fmt(int bps, bool planar)
{
        switch (bps) {
        case 1:
                return planar ? AV_SAMPLE_FMT_U8P : AV_SAMPLE_FMT_U8;
        case 2:
                return planar ? AV_SAMPLE_FMT_S16P : AV_SAMPLE_FMT_S16;
                break;
        case 3:
        case 4:
                return planar ? AV_SAMPLE_FMT_S32P : AV_SAMPLE_FMT_S32;
                break;
        default:
                abort();
        }
}

/**
 * Prints space-separated names of AVPixelFormats in AV_PIX_FMT_NONE-terminated
 * pixfmts list to given buf and returns pointer to given buf.
 */
const char *
get_avpixfmts_names(const enum AVPixelFormat *pixfmts)
{
        _Thread_local static char buf[STR_LEN];
        if (pixfmts == NULL || *pixfmts == AV_PIX_FMT_NONE) {
                snprintf(buf, sizeof buf, "(none)");
                return buf;
        }
        buf[0] = '\0';
        const enum AVPixelFormat *it = pixfmts;
        while (*it != AV_PIX_FMT_NONE) {
                snprintf(buf + strlen(buf), sizeof buf - strlen(buf), "%s%s",
                         it != pixfmts ? " " : "", av_get_pix_fmt_name(*it));
                it++;
        }
        return buf;
}

/**
 * @param ctx   may be nullptr if codec is not
 * @param codec may be nullptr if ctx is not
 *
 * If passed ctx, values such as `strict_std_compliance` may affect the result.
 */
#if LIBAVCODEC_VERSION_INT >  AV_VERSION_INT(61, 13, 100)
static const void *
avc_get_supported_config(const AVCodecContext *ctx, const AVCodec *codec,
                         enum AVCodecConfig config)
{
        const void *ret = NULL;
        int         unused_count = 0;
        const int   rc           = avcodec_get_supported_config(
            ctx, codec, config, /*flags*/ 0, &ret,
            &unused_count);
        if (rc != 0) {
                MSG(ERROR, "Cannot get list of supported config %d for %s!\n",
                    (int) config, codec->name);
                return NULL;
        }

        return ret;
}
#endif
///< @copydoc avc_get_supported_config
const enum AVPixelFormat *
avc_get_supported_pix_fmts(const AVCodecContext *ctx, const AVCodec *codec)
{
#if LIBAVCODEC_VERSION_INT > AV_VERSION_INT(61, 13, 100)
        return avc_get_supported_config(ctx, codec, AV_CODEC_CONFIG_PIX_FORMAT);
#else
        if (ctx != NULL) {
          return ctx->codec->pix_fmts;
        }
        return codec->pix_fmts;
#endif
}
///< @copydoc avc_get_supported_config
const enum AVSampleFormat *
avc_get_supported_sample_fmts(const AVCodecContext *ctx, const AVCodec *codec)
{
#if LIBAVCODEC_VERSION_INT > AV_VERSION_INT(61, 13, 100)
        return avc_get_supported_config(ctx, codec, AV_CODEC_CONFIG_SAMPLE_FORMAT);
#else
        if (ctx != NULL) {
          return ctx->codec->sample_fmts;
        }
        return codec->sample_fmts;
#endif
}
///< @copydoc avc_get_supported_config
const int *avc_get_supported_sample_rates(const AVCodecContext *ctx, const AVCodec *codec)
{
#if LIBAVCODEC_VERSION_INT > AV_VERSION_INT(61, 13, 100)
        return avc_get_supported_config(ctx, codec, AV_CODEC_CONFIG_SAMPLE_RATE);
#else
        if (ctx != NULL) {
          return ctx->codec->supported_samplerates;
        }
        return codec->supported_samplerates;
#endif
}
/* vi: set expandtab sw=8: */
