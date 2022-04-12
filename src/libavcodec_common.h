/**
 * @file   libavcodec_common.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
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
#ifndef LIBAVCODEC_COMMON_H_
#define LIBAVCODEC_COMMON_H_

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

#if LIBAVUTIL_VERSION_INT > AV_VERSION_INT(57, 9, 101) // FFMPEG commit b2cd1fb2ec6
#define HAVE_P210 1
#endif

#if defined FF_API_OLD_CHANNEL_LAYOUT || (LIBAVUTIL_VERSION_MAJOR >= 58)
#define AVCODECCTX_CHANNELS(context) (context)->ch_layout.nb_channels
#define FF_API_NEW_CHANNEL_LAYOUT 1
#else
#define AVCODECCTX_CHANNELS(context) (context)->channels
#endif

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(57, 8, 0)
static struct AVPacket *av_packet_alloc(void) ATTRIBUTE(unused);
static struct AVPacket *av_packet_alloc() {
        struct AVPacket *pkt = (struct AVPacket *) calloc(1, sizeof *pkt);
        if (pkt == NULL) {
                return NULL;
        }
        av_init_packet(pkt);
        return pkt;
}

static void av_packet_free(struct AVPacket **pkt) ATTRIBUTE(unused);
static void av_packet_free(struct AVPacket **pkt) {
        if (pkt == NULL || *pkt == NULL) {
                return;
        }
        free(*pkt);
        *pkt = NULL;
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

static void print_decoder_error(const char *mod_name, int rc) ATTRIBUTE(unused);
static void print_decoder_error(const char *mod_name, int rc) {
        char buf[1024];
	switch (rc) {
		case 0:
			break;
		case EAGAIN:
			log_msg(LOG_LEVEL_VERBOSE, "%s No frame returned - needs more input data.\n", mod_name);
			break;
		case EINVAL:
			log_msg(LOG_LEVEL_ERROR, "%s Decoder in invalid state!\n", mod_name);
			break;
		default:
                        av_strerror(rc, buf, 1024);
                        log_msg(LOG_LEVEL_WARNING, "%s Error while decoding frame (rc == %d): %s.\n", mod_name, rc, buf);
			break;
	}
}

inline static bool pixfmt_has_420_subsampling(enum AVPixelFormat fmt){
        const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(fmt);

        return fmt_desc && (fmt_desc->log2_chroma_w == 1 && fmt_desc->log2_chroma_h == 1);
}

/// @retval true if all pixel formats have either 420 subsampling or are HW accelerated
inline static bool pixfmt_list_has_420_subsampling(const enum AVPixelFormat *fmt){
        for(const enum AVPixelFormat *it = fmt; *it != AV_PIX_FMT_NONE; it++){
                const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(*it);
                if (!pixfmt_has_420_subsampling(*it) && !(fmt_desc->flags & AV_PIX_FMT_FLAG_HWACCEL)) {
                        return false;
                }
        }

        return true;
}

void print_libav_error(int verbosity, const char *msg, int rc);
bool libav_codec_has_extradata(codec_t codec);

typedef void uv_to_av_convert(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height);
typedef uv_to_av_convert *pixfmt_callback_t;

/**
 * Conversions from UltraGrid to FFMPEG formats.
 *
 * Currently do not add an "upgrade" conversion (UYVY->10b) because also
 * UltraGrid decoder can be used first and thus conversion v210->UYVY->10b
 * may be used resulting in a precision loss. If needed, put the upgrade
 * conversions below the others.
 */
struct uv_to_av_conversion {
        codec_t src;
        enum AVPixelFormat dst;
        enum AVColorSpace colorspace;  ///< destination colorspace
        enum AVColorRange color_range; ///< destination color range
        pixfmt_callback_t func;        ///< conversion function
};
const struct uv_to_av_conversion *get_uv_to_av_conversions(void);
pixfmt_callback_t get_uv_to_av_conversion(codec_t uv_codec, int av_codec);
/**
 * Returns AV format details for given pair UV,AV codec (must be unique then)
 */
void get_av_pixfmt_details(codec_t uv_codec, int av_codec, enum AVColorSpace *colorspace, enum AVColorRange *color_range);

typedef void av_to_uv_convert(char * __restrict dst_buffer, AVFrame * __restrict in_frame, int width, int height, int pitch, const int * __restrict rgb_shift);
typedef av_to_uv_convert *av_to_uv_convert_p;

struct av_to_uv_conversion {
        int av_codec;
        codec_t uv_codec;
        av_to_uv_convert_p convert;
        bool native; ///< there is a 1:1 mapping between the FFMPEG and UV codec (matching
                     ///< color space, channel count (w/wo alpha), bit-depth,
                     ///< subsampling etc.). Supported out are: RGB, UYVY, v210 (in future
                     ///< also 10,12 bit RGB). Subsampling doesn't need to be respected (we do
                     ///< not have codec for eg. 4:4:4 UYVY).
};

av_to_uv_convert_p get_av_to_uv_conversion(int av_codec, codec_t uv_codec);
const struct av_to_uv_conversion *get_av_to_uv_conversions(void);

codec_t get_av_to_ug_codec(enum AVCodecID av_codec);
enum AVCodecID get_ug_to_av_codec(codec_t ug_codec);

struct uv_to_av_pixfmt {
        codec_t uv_codec;
        enum AVPixelFormat av_pixfmt;
};
codec_t get_av_to_ug_pixfmt(enum AVPixelFormat av_pixfmt) ATTRIBUTE(const);
enum AVPixelFormat get_ug_to_av_pixfmt(codec_t ug_codec) ATTRIBUTE(const);
const struct uv_to_av_pixfmt *get_av_to_ug_pixfmts(void) ATTRIBUTE(const);

void serialize_avframe(const void *f, FILE *out);

#ifdef HAVE_SWSCALE
struct SwsContext;
struct SwsContext *getSwsContext(unsigned int SrcW, unsigned int SrcH, enum AVPixelFormat SrcFormat, unsigned int DstW, unsigned int DstH, enum AVPixelFormat DstFormat, int64_t Flags);
#endif // defined HAVE_SWSCALE

#ifdef __cplusplus
}
#endif

#endif // LIBAVCODEC_COMMON_H_

