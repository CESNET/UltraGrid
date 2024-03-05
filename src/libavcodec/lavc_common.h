/**
 * @file   libavcodec/lavc_common.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
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
#ifndef LAVC_COMMON_H_C9D57362_067F_45AD_A491_A8084A39E675
#define LAVC_COMMON_H_C9D57362_067F_45AD_A491_A8084A39E675

#include "audio/types.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/samplefmt.h>
#include <libavutil/version.h>

#ifdef __cplusplus
}
#endif

#include "libavcodec/utils.h"

#ifdef _MSC_VER
#define __attribute__(a)
#endif

// component indices to rgb_shift[] (@ref av_to_uv_convert)
enum {
        R_SHIFT_IDX = 0,
        G_SHIFT_IDX = 1,
        B_SHIFT_IDX = 2,
};

#define LIBAV_ERRBUF_LEN 1024

///
/// compat
///
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 26, 0)
#define AV_CODEC_ID_AV1 AV_CODEC_ID_NONE
#endif

#define Y210_PRESENT LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 42, 100) // FFMPEG commit 1c37cad0
#define X2RGB10LE_PRESENT LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 55, 100) // FFMPEG commit b09fb030
#define P210_PRESENT LIBAVUTIL_VERSION_INT > AV_VERSION_INT(57, 9, 101) // FFMPEG commit b2cd1fb2ec6
#define VUYX_PRESENT LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 34, 100) // FFMPEG commit cc5a5c98604
#define XV3X_PRESENT LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 36, 100) // FFMPEG commit d75c4693fef adds P012, Y212, XV30 and XV36

#if defined FF_API_OLD_CHANNEL_LAYOUT || (LIBAVUTIL_VERSION_MAJOR >= 58)
#define AVCODECCTX_CHANNELS(context) (context)->ch_layout.nb_channels
#define FF_API_NEW_CHANNEL_LAYOUT 1
#else
#define AVCODECCTX_CHANNELS(context) (context)->channels
#endif

#ifndef AV_CODEC_CAP_OTHER_THREADS
#define AV_CODEC_CAP_OTHER_THREADS AV_CODEC_CAP_AUTO_THREADS
#endif


#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(56, 39, 100)
#define AV_PIX_FMT_VULKAN AV_PIX_FMT_NONE
#endif
#ifndef HWACC_RPI4
#define AV_PIX_FMT_RPI4_8 AV_PIX_FMT_NONE
#endif

#ifdef __cplusplus
extern "C" {
#endif

void print_decoder_error(const char *mod_name, int rc);

void print_libav_error(int verbosity, const char *msg, int rc);
void printf_libav_error(int verbosity, int rc, const char *msg, ...) __attribute__((format (printf, 3, 4)));
bool libav_codec_has_extradata(codec_t codec);

codec_t get_av_to_ug_codec(enum AVCodecID av_codec);
enum AVCodecID get_ug_to_av_codec(codec_t ug_codec);

void ug_set_av_logging(void);
int av_pixfmt_get_subsampling(enum AVPixelFormat fmt) __attribute__((const));
struct pixfmt_desc av_pixfmt_get_desc(enum AVPixelFormat pixfmt);
void lavd_flush(AVCodecContext *codec_ctx);
const char *lavc_thread_type_to_str(int thread_type);
struct audio_desc audio_desc_from_av_frame(const AVFrame *frm);
enum AVSampleFormat audio_bps_to_av_sample_fmt(int bps, bool planar);
const char         *get_avpixfmts_names(const enum AVPixelFormat *pixfmts);

#ifdef __cplusplus
}
#endif

#endif // defined LAVC_COMMON_H_C9D57362_067F_45AD_A491_A8084A39E675

