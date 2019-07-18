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

#ifdef __cplusplus
}
#endif

#define R 0
#define G 1
#define B 2

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

/**
 * @todo
 * Is this stuff still needed?
 */
#define LAVCD_LOCK_NAME "lavcd_lock"

static void print_decoder_error(const char *mod_name, int rc) __attribute__((unused));
static void print_libav_error(int verbosity, const char *msg, int rc)  __attribute__((unused));
static bool libav_codec_has_extradata(codec_t codec) __attribute__((unused));

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

static void print_libav_error(int verbosity, const char *msg, int rc) {
        char errbuf[1024];
        av_strerror(rc, errbuf, sizeof(errbuf));

        log_msg(verbosity, "%s: %s\n", msg, errbuf);
}

static bool libav_codec_has_extradata(codec_t codec) {
        return codec == HFYU || codec == FFV1;
}

#ifdef __cplusplus
extern "C" {
#endif

typedef void uv_to_av_convert(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height);
typedef uv_to_av_convert *pixfmt_callback_t;

uv_to_av_convert uyvy_to_yuv420p;
uv_to_av_convert uyvy_to_yuv422p;
uv_to_av_convert uyvy_to_yuv444p;
uv_to_av_convert uyvy_to_nv12;
uv_to_av_convert v210_to_yuv420p10le;
uv_to_av_convert v210_to_yuv422p10le;
uv_to_av_convert v210_to_yuv444p10le;
uv_to_av_convert v210_to_p010le;
uv_to_av_convert rgb_to_bgr0;
uv_to_av_convert rgb_to_gbrp;
uv_to_av_convert rgba_to_gbrp;
uv_to_av_convert r10k_to_gbrp10le;
uv_to_av_convert r12l_to_gbrp12le;
uv_to_av_convert r10k_to_yuv422p10le;

/**
 * Conversions from UltraGrid to FFMPEG formats.
 *
 * Currently do not add an "upgrade" conversion (UYVY->10b) because also
 * UltraGrid decoder can be used first and thus conversion v210->UYVY->10b
 * may be used resulting in a precision loss. If needed, put the upgrade
 * conversions below the others.
 */
static const struct {
        codec_t src;
        enum AVPixelFormat dst;
        pixfmt_callback_t func;
} uv_to_av_conversions[] = {
        { v210, AV_PIX_FMT_YUV420P10LE, v210_to_yuv420p10le },
        { v210, AV_PIX_FMT_YUV422P10LE, v210_to_yuv422p10le },
        { v210, AV_PIX_FMT_YUV444P10LE, v210_to_yuv444p10le },
#if LIBAVFORMAT_VERSION_MAJOR > 57 || (LIBAVFORMAT_VERSION_MAJOR == 57 && LIBAVFORMAT_VERSION_MINOR >= 24)
        { v210, AV_PIX_FMT_P010LE, v210_to_p010le },
#endif
        { UYVY, AV_PIX_FMT_YUV422P, uyvy_to_yuv422p },
        { UYVY, AV_PIX_FMT_YUVJ422P, uyvy_to_yuv422p },
        { UYVY, AV_PIX_FMT_YUV420P, uyvy_to_yuv420p },
        { UYVY, AV_PIX_FMT_YUVJ420P, uyvy_to_yuv420p },
        { UYVY, AV_PIX_FMT_NV12, uyvy_to_nv12 },
        { UYVY, AV_PIX_FMT_YUV444P, uyvy_to_yuv444p },
        { UYVY, AV_PIX_FMT_YUVJ444P, uyvy_to_yuv444p },
        { RGB, AV_PIX_FMT_BGR0, rgb_to_bgr0 },
        { RGB, AV_PIX_FMT_GBRP, rgb_to_gbrp },
        { RGBA, AV_PIX_FMT_GBRP, rgba_to_gbrp },
        { R10k, AV_PIX_FMT_GBRP10LE, r10k_to_gbrp10le },
        { R10k, AV_PIX_FMT_YUV422P10LE, r10k_to_yuv422p10le },
#if LIBAVFORMAT_VERSION_MAJOR > 55 || (LIBAVFORMAT_VERSION_MAJOR == 55 && LIBAVFORMAT_VERSION_MINOR >= 24)
        { R12L, AV_PIX_FMT_GBRP12LE, r12l_to_gbrp12le },
#endif
};

typedef void av_to_uv_convert(char * __restrict dst_buffer, AVFrame * __restrict in_frame, int width, int height, int pitch, int * __restrict rgb_shift);
typedef av_to_uv_convert *av_to_uv_convert_p;

av_to_uv_convert nv12_to_uyvy;
av_to_uv_convert rgb24_to_uyvy;
av_to_uv_convert memcpy_data;
av_to_uv_convert gbrp_to_rgb;
av_to_uv_convert gbrp_to_rgba;
av_to_uv_convert gbrp10le_to_r10k;
av_to_uv_convert gbrp10le_to_rgb;
av_to_uv_convert gbrp10le_to_rgba;
av_to_uv_convert gbrp12le_to_r12l;
av_to_uv_convert gbrp12le_to_rgb;
av_to_uv_convert gbrp12le_to_rgba;
av_to_uv_convert rgb48le_to_rgba;
av_to_uv_convert rgb48le_to_r12l;
av_to_uv_convert yuv420p_to_uyvy;
av_to_uv_convert yuv420p_to_v210;
av_to_uv_convert yuv422p_to_uyvy;
av_to_uv_convert yuv422p_to_v210;
av_to_uv_convert yuv444p_to_uyvy;
av_to_uv_convert yuv444p_to_v210;
av_to_uv_convert nv12_to_rgb24;
av_to_uv_convert yuv422p_to_rgb24;
av_to_uv_convert yuv420p_to_rgb24;
av_to_uv_convert yuv444p_to_rgb24;
av_to_uv_convert yuv420p10le_to_v210;
av_to_uv_convert yuv422p10le_to_v210;
av_to_uv_convert yuv444p10le_to_v210;
av_to_uv_convert yuv420p10le_to_uyvy;
av_to_uv_convert yuv422p10le_to_uyvy;
av_to_uv_convert yuv444p10le_to_uyvy;
av_to_uv_convert yuv420p10le_to_rgb24;
av_to_uv_convert yuv422p10le_to_rgb24;
av_to_uv_convert yuv444p10le_to_rgb24;
av_to_uv_convert p010le_to_v210;
av_to_uv_convert p010le_to_uyvy;
#ifdef HWACC_VDPAU
av_to_uv_convert av_vdpau_to_ug_vdpau;
#endif

static const struct {
        int av_codec;
        codec_t uv_codec;
        av_to_uv_convert_p convert;
        bool native; ///< there is a 1:1 mapping between the FFMPEG and UV codec (matching
                     ///< color space, channel count (w/wo alpha), bit-depth,
                     ///< subsampling etc.). Supported out are: RGB, UYVY, v210 (in future
                     ///< also 10,12 bit RGB). Subsampling doesn't need to be respected (we do
                     ///< not have codec for eg. 4:4:4 UYVY).
} av_to_uv_conversions[] = {
        // 10-bit YUV
        {AV_PIX_FMT_YUV420P10LE, v210, yuv420p10le_to_v210, true},
        {AV_PIX_FMT_YUV420P10LE, UYVY, yuv420p10le_to_uyvy, false},
        {AV_PIX_FMT_YUV420P10LE, RGB, yuv420p10le_to_rgb24, false},
        {AV_PIX_FMT_YUV422P10LE, v210, yuv422p10le_to_v210, true},
        {AV_PIX_FMT_YUV422P10LE, UYVY, yuv422p10le_to_uyvy, false},
        {AV_PIX_FMT_YUV422P10LE, RGB, yuv422p10le_to_rgb24, false},
        {AV_PIX_FMT_YUV444P10LE, v210, yuv444p10le_to_v210, true},
        {AV_PIX_FMT_YUV444P10LE, UYVY, yuv444p10le_to_uyvy, false},
        {AV_PIX_FMT_YUV444P10LE, RGB, yuv444p10le_to_rgb24, false},
#if LIBAVFORMAT_VERSION_MAJOR > 57 || (LIBAVFORMAT_VERSION_MAJOR == 57 && LIBAVFORMAT_VERSION_MINOR >= 24)
        {AV_PIX_FMT_P010LE, v210, p010le_to_v210, true},
        {AV_PIX_FMT_P010LE, UYVY, p010le_to_uyvy, true},
#endif
        // 8-bit YUV
        {AV_PIX_FMT_YUV420P, v210, yuv420p_to_v210, false},
        {AV_PIX_FMT_YUV420P, UYVY, yuv420p_to_uyvy, true},
        {AV_PIX_FMT_YUV420P, RGB, yuv420p_to_rgb24, false},
        {AV_PIX_FMT_YUV422P, v210, yuv422p_to_v210, false},
        {AV_PIX_FMT_YUV422P, UYVY, yuv422p_to_uyvy, true},
        {AV_PIX_FMT_YUV422P, RGB, yuv422p_to_rgb24, false},
        {AV_PIX_FMT_YUV444P, v210, yuv444p_to_v210, false},
        {AV_PIX_FMT_YUV444P, UYVY, yuv444p_to_uyvy, true},
        {AV_PIX_FMT_YUV444P, RGB, yuv444p_to_rgb24, false},
        // 8-bit YUV (JPEG color range)
        {AV_PIX_FMT_YUVJ420P, v210, yuv420p_to_v210, false},
        {AV_PIX_FMT_YUVJ420P, UYVY, yuv420p_to_uyvy, true},
        {AV_PIX_FMT_YUVJ420P, RGB, yuv420p_to_rgb24, false},
        {AV_PIX_FMT_YUVJ422P, v210, yuv422p_to_v210, false},
        {AV_PIX_FMT_YUVJ422P, UYVY, yuv422p_to_uyvy, true},
        {AV_PIX_FMT_YUVJ422P, RGB, yuv422p_to_rgb24, false},
        {AV_PIX_FMT_YUVJ444P, v210, yuv444p_to_v210, false},
        {AV_PIX_FMT_YUVJ444P, UYVY, yuv444p_to_uyvy, true},
        {AV_PIX_FMT_YUVJ444P, RGB, yuv444p_to_rgb24, false},
        // 8-bit YUV (NV12)
        {AV_PIX_FMT_NV12, UYVY, nv12_to_uyvy, true},
        {AV_PIX_FMT_NV12, RGB, nv12_to_rgb24, false},
        // RGB
        {AV_PIX_FMT_GBRP, RGB, gbrp_to_rgb, true},
        {AV_PIX_FMT_GBRP, RGBA, gbrp_to_rgba, true},
        {AV_PIX_FMT_RGB24, UYVY, rgb24_to_uyvy, false},
        {AV_PIX_FMT_RGB24, RGB, memcpy_data, true},
        {AV_PIX_FMT_GBRP10LE, R10k, gbrp10le_to_r10k, true},
        {AV_PIX_FMT_GBRP10LE, RGB, gbrp10le_to_rgb, false},
        {AV_PIX_FMT_GBRP10LE, RGBA, gbrp10le_to_rgba, false},
#if LIBAVFORMAT_VERSION_MAJOR > 55 || (LIBAVFORMAT_VERSION_MAJOR == 55 && LIBAVFORMAT_VERSION_MINOR >= 24)
        {AV_PIX_FMT_GBRP12LE, R12L, gbrp12le_to_r12l, true},
        {AV_PIX_FMT_GBRP12LE, RGB, gbrp12le_to_rgb, false},
        {AV_PIX_FMT_GBRP12LE, RGBA, gbrp12le_to_rgba, false},
#endif
        {AV_PIX_FMT_RGB48LE, RG48, memcpy_data, true},
        {AV_PIX_FMT_RGB48LE, R12L, rgb48le_to_r12l, false},
        {AV_PIX_FMT_RGB48LE, RGBA, rgb48le_to_rgba, false},
#ifdef HWACC_VDPAU
        // HW acceleration
        {AV_PIX_FMT_VDPAU, HW_VDPAU, av_vdpau_to_ug_vdpau, false},
#endif
};



#ifdef __cplusplus
}
#endif

#endif // LIBAVCODEC_COMMON_H_

