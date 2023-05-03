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

#include "debug.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>
#include <libavutil/version.h>

#ifdef __cplusplus
}
#endif

#include "libavcodec/utils.h"

// component indices to rgb_shift[] (@ref av_to_uv_convert)
#define R 0
#define G 1
#define B 2

#define LIBAV_ERRBUF_LEN 1024

///
/// compat
///
// avcodec
#ifndef AV_CODEC_CAP_FRAME_THREADS
#define AV_CODEC_CAP_FRAME_THREADS CODEC_CAP_FRAME_THREADS
#endif
#ifndef AV_CODEC_CAP_SLICE_THREADS
#define AV_CODEC_CAP_SLICE_THREADS CODEC_CAP_SLICE_THREADS
#endif
#ifndef AV_CODEC_CAP_VARIABLE_FRAME_SIZE
#define AV_CODEC_CAP_VARIABLE_FRAME_SIZE CODEC_CAP_VARIABLE_FRAME_SIZE
#endif
#ifndef AV_CODEC_FLAG2_FAST
#define AV_CODEC_FLAG2_FAST CODEC_FLAG2_FAST
#endif
#ifndef AV_INPUT_BUFFER_PADDING_SIZE
#define AV_INPUT_BUFFER_PADDING_SIZE FF_INPUT_BUFFER_PADDING_SIZE
#endif

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(54, 24, 0)
#define AV_CODEC_ID_H264 CODEC_ID_H264
#define AV_CODEC_ID_JPEG2000 CODEC_ID_JPEG2000
#define AV_CODEC_ID_MJPEG CODEC_ID_MJPEG
#define AV_CODEC_ID_VP8 CODEC_ID_VP8
#endif

// av_frame_* was inbetween moved from lavc to lavu
#if LIBAVCODEC_VERSION_MAJOR < 55
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#define av_frame_unref avcodec_get_frame_defaults
#undef av_frame_free
#define av_frame_free av_free
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 28, 0)
#define AV_CODEC_ID_VP9 AV_CODEC_ID_NONE
#endif

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(55, 35, 100)
#define AV_CODEC_ID_HEVC AV_CODEC_ID_NONE
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 52, 0)
#define avcodec_free_context av_freep
#endif

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(56, 55, 1)
#define AV_CODEC_FLAG_INTERLACED_DCT CODEC_FLAG_INTERLACED_DCT
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 8, 0)
#define av_packet_unref av_free_packet
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 26, 0)
#define AV_CODEC_ID_AV1 AV_CODEC_ID_NONE
#endif

// avutil
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(51, 42, 0) // FFMPEG commit 78071a1420b
#define AV_PIX_FMT_NONE PIX_FMT_NONE
#define AV_PIX_FMT_NV12 PIX_FMT_NV12
#define AV_PIX_FMT_BGR24 PIX_FMT_BGR24
#define AV_PIX_FMT_RGB24 PIX_FMT_RGB24
#define AV_PIX_FMT_RGBA PIX_FMT_RGBA
#define AV_PIX_FMT_UYVY422 PIX_FMT_RGBA
#define AV_PIX_FMT_YUV420P PIX_FMT_YUV420P
#define AV_PIX_FMT_YUV422P PIX_FMT_YUV422P
#define AV_PIX_FMT_YUV444P PIX_FMT_YUV444P
#define AV_PIX_FMT_YUVJ420P PIX_FMT_YUVJ420P
#define AV_PIX_FMT_YUVJ422P PIX_FMT_YUVJ422P
#define AV_PIX_FMT_YUVJ444P PIX_FMT_YUVJ444P
#define AV_PIX_FMT_YUYV422 PIX_FMT_YUYV422
#define AVPixelFormat PixelFormat
#define AVCodecID CodecID
#endif

#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(51, 74, 100)
#define AV_PIX_FMT_GBRP12LE PIX_FMT_GBRP12LE
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

#ifdef __cplusplus
extern "C" {
#endif

void print_decoder_error(const char *mod_name, int rc);
bool pixfmt_has_420_subsampling(enum AVPixelFormat fmt);
/// @retval true if all pixel formats have either 420 subsampling or are HW accelerated
bool pixfmt_list_has_420_subsampling(const enum AVPixelFormat *fmt);

void print_libav_error(int verbosity, const char *msg, int rc);
void printf_libav_error(int verbosity, int rc, const char *msg, ...) __attribute__((format (printf, 3, 4)));
bool libav_codec_has_extradata(codec_t codec);

codec_t get_av_to_ug_codec(enum AVCodecID av_codec);
enum AVCodecID get_ug_to_av_codec(codec_t ug_codec);

void ug_set_av_logging(void);
int av_pixfmt_get_subsampling(enum AVPixelFormat fmt) __attribute__((const));
struct pixfmt_desc av_pixfmt_get_desc(enum AVPixelFormat pixfmt);
void lavd_flush(AVCodecContext *codec_ctx);

#ifdef __cplusplus
}
#endif

#endif // defined LAVC_COMMON_H_C9D57362_067F_45AD_A491_A8084A39E675

