/**
 * @file   libavcodec/utils.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
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

#include "libavcodec/utils.h"

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
        //JPEG,
        //RAW,
        //H264,
        //MJPG,
        //VP8,
        {BGR, AV_PIX_FMT_BGR24},
        //J2K,
        {I420, AV_PIX_FMT_YUVJ420P},
        {RG48, AV_PIX_FMT_RGB48LE},
#if XV3X_PRESENT
        {Y416, AV_PIX_FMT_XV36},
#endif
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

/* vi: set expandtab sw=8: */
