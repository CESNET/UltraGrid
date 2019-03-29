#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "types.h"

#ifndef LIBAVCODEC_COMMON_H_
#define LIBAVCODEC_COMMON_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>
#include <libavutil/pixfmt.h>

#ifdef __cplusplus
}
#endif

///
/// compat
///
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

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(58, 26, 0)
#define AV_CODEC_ID_AV1 AV_CODEC_ID_NONE
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 8, 0)
#define av_packet_unref av_free_packet
#endif

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(56, 55, 1)
#define AV_CODEC_FLAG_INTERLACED_DCT CODEC_FLAG_INTERLACED_DCT
#endif

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(55, 35, 100)
#define AV_CODEC_ID_HEVC AV_CODEC_ID_NONE
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 52, 0)
#define avcodec_free_context av_freep
#endif

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(55, 28, 0)
#define AV_CODEC_ID_VP9 AV_CODEC_ID_NONE
#endif

#if LIBAVCODEC_VERSION_MAJOR < 55
#define av_frame_alloc avcodec_alloc_frame
#define av_frame_free avcodec_free_frame
#define av_frame_unref avcodec_get_frame_defaults
#endif

#if LIBAVCODEC_VERSION_MAJOR < 54
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
#define AV_CODEC_ID_NONE CODEC_ID_NONE
#define AV_CODEC_ID_H264 CODEC_ID_H264
#define AV_CODEC_ID_JPEG2000 CODEC_ID_JPEG2000
#define AV_CODEC_ID_MJPEG CODEC_ID_MJPEG
#define AV_CODEC_ID_VP8 CODEC_ID_VP8
#define AVPixelFormat PixelFormat
#define AVCodecID CodecID
#undef av_frame_free
#define av_frame_free av_free
#endif

#define LAVCD_LOCK_NAME "lavcd_lock"

static enum AVPixelFormat fmts444_8[] = { AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUVJ444P };
static enum AVPixelFormat fmts422_8[] = { AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVJ422P };
static enum AVPixelFormat fmts420_8[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_NV12 };
/**
 * @param req_pix_fmts AV_PIX_FMT_NONE-ended priority list of requested pix_fmts
 * @param pix_fmts     AV_PIX_FMT_NONE-ended priority list of codec provided pix fmts
 * */
static bool is444_8(enum AVPixelFormat pix_fmt) __attribute__((unused));
static bool is422_8(enum AVPixelFormat pix_fmt) __attribute__((unused));
static bool is420_8(enum AVPixelFormat pix_fmt) __attribute__((unused));
static void print_decoder_error(const char *mod_name, int rc) __attribute__((unused));
static void print_libav_error(int verbosity, const char *msg, int rc)  __attribute__((unused));
static bool libav_codec_has_extradata(codec_t codec) __attribute__((unused));

static bool is444_8(enum AVPixelFormat pix_fmt) {
        for(unsigned int i = 0; i < sizeof(fmts444_8) / sizeof(enum AVPixelFormat); ++i) {
                if(fmts444_8[i] == pix_fmt)
                        return true;
        }
        return false;
}

static bool is422_8(enum AVPixelFormat pix_fmt) {
        for(unsigned int i = 0; i < sizeof(fmts422_8) / sizeof(enum AVPixelFormat); ++i) {
                if(fmts422_8[i] == pix_fmt)
                        return true;
        }
        return false;
}

static bool is420_8(enum AVPixelFormat pix_fmt) {
        for(unsigned int i = 0; i < sizeof(fmts420_8) / sizeof(enum AVPixelFormat); ++i) {
                if(fmts420_8[i] == pix_fmt)
                        return true;
        }
        return false;
}

#ifdef __cplusplus
#include <unordered_map>
#include "types.h"

static const std::unordered_map<codec_t, enum AVPixelFormat, std::hash<int>> ug_to_av_pixfmt_map = {
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
        {BGR, AV_PIX_FMT_BGR24}
        //J2K,

};

#endif // __cplusplus

static void print_decoder_error(const char *mod_name, int rc) {
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
			log_msg(LOG_LEVEL_WARNING, "%s Error while decoding frame (rc == %d).\n", mod_name, rc);
			break;
	}
}

static void print_libav_error(int verbosity, const char *msg, int rc) {
        char errbuf[1024];
        av_strerror(rc, errbuf, sizeof(errbuf));

        log_msg(verbosity, "%s: %s\n", msg, errbuf);
}

static bool libav_codec_has_extradata(codec_t codec) {
        if (codec == HFYU || codec == FFV1) {
                return true;
        }
        return false;
}

#endif // LIBAVCODEC_COMMON_H_

