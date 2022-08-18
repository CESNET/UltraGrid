/**
 * @file   lavc_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2013-2021 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <assert.h>
#include <stdbool.h>
#include <stdint.h>

#include "host.h"
#include "libavcodec/lavc_common.h"
#include "video.h"

//
// UG <-> FFMPEG format translations
//
static const struct {
        enum AVCodecID av;
        codec_t uv;
} av_to_uv_map[] = {
        { AV_CODEC_ID_H264, H264 },
        { AV_CODEC_ID_HEVC, H265 },
        { AV_CODEC_ID_MJPEG, MJPG },
        { AV_CODEC_ID_JPEG2000, J2K },
        { AV_CODEC_ID_VP8, VP8 },
        { AV_CODEC_ID_VP9, VP9 },
        { AV_CODEC_ID_HUFFYUV, HFYU },
        { AV_CODEC_ID_FFV1, FFV1 },
        { AV_CODEC_ID_AV1, AV1 },
        { AV_CODEC_ID_PRORES, PRORES },
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

/// known UG<->AV pixfmt conversions, terminate with NULL (returned by
/// get_av_to_ug_pixfmts())
static const struct uv_to_av_pixfmt uv_to_av_pixfmts[] = {
        {RGBA, AV_PIX_FMT_RGBA},
        {UYVY, AV_PIX_FMT_UYVY422},
        {YUYV,AV_PIX_FMT_YUYV422},
        //R10k,
        //v210,
        //DVS10,
        //DXT1,
        //DXT1_YUV,
        //DXT5,
        {RGB, AV_PIX_FMT_RGB24},
        // DPX10,
        //JPEG,
        //RAW,
        //H264,
        //MJPG,
        //VP8,
        {BGR, AV_PIX_FMT_BGR24},
        //J2K,
        {I420, AV_PIX_FMT_YUVJ420P},
        {RG48, AV_PIX_FMT_RGB48LE},
        {0, 0}
};

codec_t get_av_to_ug_pixfmt(enum AVPixelFormat av_pixfmt) {
        for (unsigned int i = 0; uv_to_av_pixfmts[i].uv_codec != VIDEO_CODEC_NONE; ++i) {
                if (uv_to_av_pixfmts[i].av_pixfmt == av_pixfmt) {
                        return uv_to_av_pixfmts[i].uv_codec;
                }
        }
        return VIDEO_CODEC_NONE;
}

enum AVPixelFormat get_ug_to_av_pixfmt(codec_t ug_codec) {
        for (unsigned int i = 0; uv_to_av_pixfmts[i].uv_codec != VIDEO_CODEC_NONE; ++i) {
                if (uv_to_av_pixfmts[i].uv_codec == ug_codec) {
                        return uv_to_av_pixfmts[i].av_pixfmt;
                }
        }
        return AV_PIX_FMT_NONE;
}

/**
 * Returns list all known FFMPEG to UG pixfmt conversions. Terminated with NULL
 * element.
 */
const struct uv_to_av_pixfmt *get_av_to_ug_pixfmts() {
        return uv_to_av_pixfmts;
}

//
// utility functions
//
void print_libav_error(int verbosity, const char *msg, int rc) {
        char errbuf[1024];
        av_strerror(rc, errbuf, sizeof(errbuf));

        log_msg(verbosity, "%s: %s\n", msg, errbuf);
}

bool libav_codec_has_extradata(codec_t codec) {
        return codec == HFYU || codec == FFV1;
}

ADD_TO_PARAM("lavcd-log-level",
                "* lavcd-log-level=<num>\n"
                "  Set libavcodec log level (use UltraGrid range semantics)\n");
/// Sets specified log level either given explicitly or from UG-wide log_level
void ug_set_av_log_level() {
        const char *param = get_commandline_param("lavcd-log-level");
        int av_log_level = param != NULL ? atoi(param) : log_level;
        av_log_set_level((av_log_level - 1) * 8);
}

/* vi: set expandtab sw=8: */
