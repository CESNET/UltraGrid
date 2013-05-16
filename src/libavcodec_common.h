#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#ifndef LIBAVCODEC_COMMON_H_
#define LIBAVCODEC_COMMON_H_

#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/mem.h>
#include <libavutil/opt.h>

#ifndef HAVE_AVCODEC_ENCODE_VIDEO2
#define AV_PIX_FMT_NONE PIX_FMT_NONE
#define AV_PIX_FMT_YUV420P PIX_FMT_YUV420P
#define AV_PIX_FMT_YUV422P PIX_FMT_YUV422P
#define AV_PIX_FMT_YUV444P PIX_FMT_YUV444P
#define AV_PIX_FMT_YUVJ420P PIX_FMT_YUVJ420P
#define AV_PIX_FMT_YUVJ422P PIX_FMT_YUVJ422P
#define AV_PIX_FMT_YUVJ444P PIX_FMT_YUVJ444P
#define AV_CODEC_ID_NONE CODEC_ID_NONE
#define AV_CODEC_ID_H264 CODEC_ID_H264
#define AV_CODEC_ID_MJPEG CODEC_ID_MJPEG
#define AV_CODEC_ID_VP8 CODEC_ID_VP8
#define AVPixelFormat PixelFormat
#define AVCodecID CodecID
#endif

#define LAVCD_LOCK_NAME "lavcd_lock"

static enum AVPixelFormat fmts444[] = { AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUVJ444P };
static enum AVPixelFormat fmts422[] = { AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUVJ422P };
static enum AVPixelFormat fmts420[] = { AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUVJ420P };
/**
 * @param req_pix_fmts AV_PIX_FMT_NONE-emded priority list of requested pix_fmts
 * @param pix_fmts     AV_PIX_FMT_NONE-emded priority list of codec provided pix fmts
 * */
static bool is444(enum AVPixelFormat pix_fmt) __attribute__((unused));
static bool is422(enum AVPixelFormat pix_fmt) __attribute__((unused));
static bool is420(enum AVPixelFormat pix_fmt) __attribute__((unused));
static enum AVPixelFormat get_best_pix_fmt(const enum AVPixelFormat *req_pix_fmts,
                const enum AVPixelFormat *codec_pix_fmts) __attribute__((unused));

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

#endif // LIBAVCODEC_COMMON_H_

