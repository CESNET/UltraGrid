#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"

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

static enum AVPixelFormat fmts444[] = { AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUVJ444P };
static enum AVPixelFormat fmts422[] = { AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVJ422P };
static enum AVPixelFormat fmts420[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P, AV_PIX_FMT_NV12 };
/**
 * @param req_pix_fmts AV_PIX_FMT_NONE-emded priority list of requested pix_fmts
 * @param pix_fmts     AV_PIX_FMT_NONE-emded priority list of codec provided pix fmts
 * */
static bool is444(enum AVPixelFormat pix_fmt) __attribute__((unused));
static bool is422(enum AVPixelFormat pix_fmt) __attribute__((unused));
static bool is420(enum AVPixelFormat pix_fmt) __attribute__((unused));
static enum AVPixelFormat get_best_pix_fmt(const enum AVPixelFormat *req_pix_fmts,
                const enum AVPixelFormat *codec_pix_fmts) __attribute__((unused));
static void print_decoder_error(const char *mod_name, int rc) __attribute__((unused));
static void print_libav_error(int verbosity, const char *msg, int rc)  __attribute__((unused));

/**
 * Finds best pixel format
 *
 * Iterates over formats in req_pix_fmts and tries to find the same format in
 * second list, codec_pix_fmts. If found, returns that format. Efectivelly
 * select first match of item from first list in second list.
 */
static enum AVPixelFormat get_best_pix_fmt(const enum AVPixelFormat *req_pix_fmts,
                const enum AVPixelFormat *codec_pix_fmts)
{
        assert(req_pix_fmts != NULL);
        if(codec_pix_fmts == NULL)
                return AV_PIX_FMT_NONE;

        enum AVPixelFormat req;
        while((req = *req_pix_fmts++) != AV_PIX_FMT_NONE) {
                const enum AVPixelFormat *tmp = codec_pix_fmts;
                enum AVPixelFormat fmt;
                while((fmt = *tmp++) != AV_PIX_FMT_NONE) {
                        if(fmt == req)
                                return req;
                }
        }

        return AV_PIX_FMT_NONE;
}

static bool is444(enum AVPixelFormat pix_fmt) {
        for(unsigned int i = 0; i < sizeof(fmts444) / sizeof(enum AVPixelFormat); ++i) {
                if(fmts444[i] == pix_fmt)
                        return true;
        }
        return false;
}

static bool is422(enum AVPixelFormat pix_fmt) {
        for(unsigned int i = 0; i < sizeof(fmts422) / sizeof(enum AVPixelFormat); ++i) {
                if(fmts422[i] == pix_fmt)
                        return true;
        }
        return false;
}

static bool is420(enum AVPixelFormat pix_fmt) {
        for(unsigned int i = 0; i < sizeof(fmts420) / sizeof(enum AVPixelFormat); ++i) {
                if(fmts420[i] == pix_fmt)
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


#endif // LIBAVCODEC_COMMON_H_

