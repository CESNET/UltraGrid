/**
 * @file   libavcodec_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <445597@mail.muni.cz>
 */
/*
 * Copyright (c) 2013-2020 CESNET, z. s. p. o.
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
 * @todo
 * Some conversions to RGBA ignore RGB-shifts - either fix that or deprecate RGB-shifts
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <stdbool.h>

#include "host.h"
#include "hwaccel_vdpau.h"
#include "libavcodec_common.h"
#include "video.h"

#ifdef __SSE3__
#include "pmmintrin.h"
// compat with older Clang compiler
#ifndef _mm_bslli_si128
#define _mm_bslli_si128 _mm_slli_si128
#endif
#ifndef _mm_bsrli_si128
#define _mm_bsrli_si128 _mm_srli_si128
#endif
#endif

#undef MAX
#undef MIN
#define MAX(a, b)      (((a) > (b))? (a): (b))
#define MIN(a, b)      (((a) < (b))? (a): (b))

#if LIBAVUTIL_VERSION_INT > AV_VERSION_INT(51, 63, 100) // FFMPEG commit e9757066e11
#define HAVE_12_AND_14_PLANAR_COLORSPACES 1
#endif

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

//
// uv_to_av_convert conversions
//
static void uyvy_to_yuv420p(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int y;
        for (y = 0; y < height - 1; y += 2) {
                /*  every even row */
                unsigned char *src = in_data + y * (((width + 1) & ~1) * 2);
                /*  every odd row */
                unsigned char *src2 = in_data + (y + 1) * (((width + 1) & ~1) * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_y2 = out_frame->data[0] + out_frame->linesize[0] * (y + 1);
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * (y / 2);
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * (y / 2);

                int x;
                OPTIMIZED_FOR (x = 0; x < width - 1; x += 2) {
                        *dst_cb++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                }
                if (x < width) {
                        *dst_cb++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cr++ = (*src++ + *src2++) / 2;
                }
        }
        if (y < height) {
                unsigned char *src = in_data + y * (((width + 1) & ~1) * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * (y / 2);
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * (y / 2);
                int x;
                OPTIMIZED_FOR (x = 0; x < width - 1; x += 2) {
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
                if (x < width) {
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src++;
                }
        }
}

static void uyvy_to_yuv422p(AVFrame * __restrict out_frame, unsigned char * __restrict src, int width, int height)
{
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;

                OPTIMIZED_FOR (int x = 0; x < width; x += 2) {
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

static void uyvy_to_yuv444p(AVFrame * __restrict out_frame, unsigned char * __restrict src, int width, int height)
{
        for(int y = 0; y < height; ++y) {
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;

                OPTIMIZED_FOR (int x = 0; x < width; x += 2) {
                        *dst_cb++ = *src;
                        *dst_cb++ = *src++;
                        *dst_y++ = *src++;
                        *dst_cr++ = *src;
                        *dst_cr++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

static void uyvy_to_nv12(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                unsigned char *src = in_data + y * (width * 2);
                /*  every odd row */
                unsigned char *src2 = in_data + (y + 1) * (width * 2);
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_y2 = out_frame->data[0] + out_frame->linesize[0] * (y + 1);
                unsigned char *dst_cbcr = out_frame->data[1] + out_frame->linesize[1] * y / 2;

                int x = 0;
#ifdef __SSE3__
                __m128i yuv;
                __m128i yuv2;
                __m128i y1;
                __m128i y2;
                __m128i y3;
                __m128i y4;
                __m128i uv;
                __m128i uv2;
                __m128i uv3;
                __m128i uv4;
                __m128i ymask = _mm_set1_epi32(0xFF00FF00);
                __m128i dsty;
                __m128i dsty2;
                __m128i dstuv;

                for (; x < (width - 15); x += 16){
                        yuv = _mm_lddqu_si128((__m128i const*) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*) src2);
                        src += 16;
                        src2 += 16;

                        y1 = _mm_and_si128(ymask, yuv);
                        y1 = _mm_bsrli_si128(y1, 1);
                        y2 = _mm_and_si128(ymask, yuv2);
                        y2 = _mm_bsrli_si128(y2, 1);

                        uv = _mm_andnot_si128(ymask, yuv);
                        uv2 = _mm_andnot_si128(ymask, yuv2);

                        uv = _mm_avg_epu8(uv, uv2);

                        yuv = _mm_lddqu_si128((__m128i const*) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*) src2);
                        src += 16;
                        src2 += 16;

                        y3 = _mm_and_si128(ymask, yuv);
                        y3 = _mm_bsrli_si128(y3, 1);
                        y4 = _mm_and_si128(ymask, yuv2);
                        y4 = _mm_bsrli_si128(y4, 1);

                        uv3 = _mm_andnot_si128(ymask, yuv);
                        uv4 = _mm_andnot_si128(ymask, yuv2);

                        uv3 = _mm_avg_epu8(uv3, uv4);

                        dsty = _mm_packus_epi16(y1, y3);
                        dsty2 = _mm_packus_epi16(y2, y4);
                        dstuv = _mm_packus_epi16(uv, uv3);
                        _mm_storeu_si128((__m128i *) dst_y, dsty);
                        _mm_storeu_si128((__m128i *) dst_y2, dsty2);
                        _mm_storeu_si128((__m128i *) dst_cbcr, dstuv);
                        dst_y += 16;
                        dst_y2 += 16;
                        dst_cbcr += 16;
                }
#endif

                OPTIMIZED_FOR (; x < width - 1; x += 2) {
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                        *dst_cbcr++ = (*src++ + *src2++) / 2;
                        *dst_y++ = *src++;
                        *dst_y2++ = *src2++;
                }
        }
}

static void v210_to_yuv420p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y / 2);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
			//block 1, bits  0 -  9: U0+0
			//block 1, bits 10 - 19: Y0
			//block 1, bits 20 - 29: V0+1
			//block 2, bits  0 -  9: Y1
			//block 2, bits 10 - 19: U2+3
			//block 2, bits 20 - 29: Y2
			//block 3, bits  0 -  9: V2+3
			//block 3, bits 10 - 19: Y3
			//block 3, bits 20 - 29: U4+5
			//block 4, bits  0 -  9: Y4
			//block 4, bits 10 - 19: V4+5
			//block 4, bits 20 - 29: Y5
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;
                        w1_0 = *src2++;
                        w1_1 = *src2++;
                        w1_2 = *src2++;
                        w1_3 = *src2++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_y2++ = (w1_0 >> 10) & 0x3ff;
                        *dst_y2++ = w1_1 & 0x3ff;
                        *dst_y2++ = (w1_1 >> 20) & 0x3ff;
                        *dst_y2++ = (w1_2 >> 10) & 0x3ff;
                        *dst_y2++ = w1_3 & 0x3ff;
                        *dst_y2++ = (w1_3 >> 20) & 0x3ff;

                        *dst_cb++ = ((w0_0 & 0x3ff) + (w1_0 & 0x3ff)) / 2;
                        *dst_cb++ = (((w0_1 >> 10) & 0x3ff) + ((w1_1 >> 10) & 0x3ff)) / 2;
                        *dst_cb++ = (((w0_2 >> 20) & 0x3ff) + ((w1_2 >> 20) & 0x3ff)) / 2;

                        *dst_cr++ = (((w0_0 >> 20) & 0x3ff) + ((w1_0 >> 20) & 0x3ff)) / 2;
                        *dst_cr++ = ((w0_2 & 0x3ff) + (w1_2 & 0x3ff)) / 2;
                        *dst_cr++ = (((w0_3 >> 10) & 0x3ff) + ((w1_3 >> 10) & 0x3ff)) / 2;
                }
        }
}

static void v210_to_yuv422p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;

                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                }
        }
}

static void v210_to_yuv444p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;

                        *dst_y++ = (w0_0 >> 10) & 0x3ff;
                        *dst_y++ = w0_1 & 0x3ff;
                        *dst_y++ = (w0_1 >> 20) & 0x3ff;
                        *dst_y++ = (w0_2 >> 10) & 0x3ff;
                        *dst_y++ = w0_3 & 0x3ff;
                        *dst_y++ = (w0_3 >> 20) & 0x3ff;

                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = w0_0 & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_1 >> 10) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;
                        *dst_cb++ = (w0_2 >> 20) & 0x3ff;

                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = (w0_0 >> 20) & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = w0_2 & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                        *dst_cr++ = (w0_3 >> 10) & 0x3ff;
                }
        }
}

static void v210_to_p010le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cbcr = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
			//block 1, bits  0 -  9: U0+0
			//block 1, bits 10 - 19: Y0
			//block 1, bits 20 - 29: V0+1
			//block 2, bits  0 -  9: Y1
			//block 2, bits 10 - 19: U2+3
			//block 2, bits 20 - 29: Y2
			//block 3, bits  0 -  9: V2+3
			//block 3, bits 10 - 19: Y3
			//block 3, bits 20 - 29: U4+5
			//block 4, bits  0 -  9: Y4
			//block 4, bits 10 - 19: V4+5
			//block 4, bits 20 - 29: Y5
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src++;
                        w0_1 = *src++;
                        w0_2 = *src++;
                        w0_3 = *src++;
                        w1_0 = *src2++;
                        w1_1 = *src2++;
                        w1_2 = *src2++;
                        w1_3 = *src2++;

                        *dst_y++ = ((w0_0 >> 10) & 0x3ff) << 6;
                        *dst_y++ = (w0_1 & 0x3ff) << 6;
                        *dst_y++ = ((w0_1 >> 20) & 0x3ff) << 6;
                        *dst_y++ = ((w0_2 >> 10) & 0x3ff) << 6;
                        *dst_y++ = (w0_3 & 0x3ff) << 6;
                        *dst_y++ = ((w0_3 >> 20) & 0x3ff) << 6;

                        *dst_y2++ = ((w1_0 >> 10) & 0x3ff) << 6;
                        *dst_y2++ = (w1_1 & 0x3ff) << 6;
                        *dst_y2++ = ((w1_1 >> 20) & 0x3ff) << 6;
                        *dst_y2++ = ((w1_2 >> 10) & 0x3ff) << 6;
                        *dst_y2++ = (w1_3 & 0x3ff) << 6;
                        *dst_y2++ = ((w1_3 >> 20) & 0x3ff) << 6;

                        *dst_cbcr++ = (((w0_0 & 0x3ff) + (w1_0 & 0x3ff)) / 2) << 6; // Cb
                        *dst_cbcr++ = ((((w0_0 >> 20) & 0x3ff) + ((w1_0 >> 20) & 0x3ff)) / 2) << 6; // Cr
                        *dst_cbcr++ = ((((w0_1 >> 10) & 0x3ff) + ((w1_1 >> 10) & 0x3ff)) / 2) << 6; // Cb
                        *dst_cbcr++ = (((w0_2 & 0x3ff) + (w1_2 & 0x3ff)) / 2) << 6; // Cr
                        *dst_cbcr++ = ((((w0_2 >> 20) & 0x3ff) + ((w1_2 >> 20) & 0x3ff)) / 2) << 6; // Cb
                        *dst_cbcr++ = ((((w0_3 >> 10) & 0x3ff) + ((w1_3 >> 10) & 0x3ff)) / 2) << 6; // Cr
                }
        }
}

static void r10k_to_yuv422p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        const int src_linesize = vc_get_linesize(width, R10k);
        const int32_t y_r = 190893; //0.18205 << 20
        const int32_t y_g = 642179; //0.61243 << 20
        const int32_t y_b = 64833; //0.06183 << 20
        const int32_t cb_r = -122882; //-0.11719 << 20
        const int32_t cb_g = -413380; //-0.39423 << 20
        const int32_t cb_b = 536263; //0.51142 << 20
        const int32_t cr_r = 536263; //0.51142 << 20
        const int32_t cr_g = -487085; //-0.46452 << 20
        const int32_t cr_b = -49168;  //-0.04689 << 20
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);
                unsigned char *src = in_data + y * src_linesize;
                int iterations = width / 2;
                OPTIMIZED_FOR(int x = 0; x < iterations; x++){
                        int32_t r = src[0] << 2 | src[1] >> 6;
                        int32_t g = (src[1] & 0x3f ) << 4 | src[2] >> 4;
                        int32_t b = (src[2] & 0x0f) << 6 | src[3] >> 2;

                        int32_t res_y = ((r * y_r + g * y_g + b * y_b) >> 20) + 64;
                        int32_t res_cb = ((r * cb_r + g * cb_g + b * cb_b) >> 20) + 512;
                        int32_t res_cr = ((r * cr_r + g * cr_g + b * cr_b) >> 20) + 512;

                        res_y = MIN(MAX(res_y, 64), 940);

                        dst_y[x * 2] =  res_y;
                        src += 4;

                        r = src[0] << 2 | src[1] >> 6;
                        g = (src[1] & 0x3f ) << 4 | src[2] >> 4;
                        b = (src[2] & 0x0f) << 6 | src[3] >> 2;

                        res_y = ((r * y_r + g * y_g + b * y_b) >> 20) + 64;
                        res_cb += ((r * cb_r + g * cb_g + b * cb_b) >> 20) + 512;
                        res_cr += ((r * cr_r + g * cr_g + b * cr_b) >> 20) + 512;

                        res_cb /= 2;
                        res_cr /= 2;
                        res_y = MIN(MAX(res_y, 64), 940);
                        res_cb = MIN(MAX(res_cb, 64), 960);
                        res_cr = MIN(MAX(res_cr, 64), 960);

                        dst_y[x * 2 + 1] = res_y;
                        dst_cb[x] = res_cb;
                        dst_cr[x] = res_cr;

                        src += 4;
                }

        }
}

static void rgb_to_bgr0(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, RGB);
        int dst_linesize = vc_get_linesize(width, RGBA);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst = out_frame->data[0] + out_frame->linesize[0] * y;
                vc_copylineRGBtoRGBA(dst, src, dst_linesize, 16, 8, 0);
        }
}

static void r10k_to_bgr0(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R10k);
        int dst_linesize = vc_get_linesize(width, RGBA);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst = out_frame->data[0] + out_frame->linesize[0] * y;
                vc_copyliner10k(dst, src, dst_linesize, 16, 8, 0);
        }
}

static void rgb_rgba_to_gbrp(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, int bpp)
{
        int src_linesize = bpp * width;
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                unsigned char *dst_g = out_frame->data[0] + out_frame->linesize[0] * y;
                unsigned char *dst_b = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_r = out_frame->data[2] + out_frame->linesize[2] * y;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst_r++ = src[0];
                        *dst_g++ = src[1];
                        *dst_b++ = src[2];
                        src += bpp;
                }
        }
}

static void rgb_to_gbrp(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rgb_rgba_to_gbrp(out_frame, in_data, width, height, 3);
}

static void rgba_to_gbrp(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rgb_rgba_to_gbrp(out_frame, in_data, width, height, 4);
}

static void r10k_to_gbrp10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R10k);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_g = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_b = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        unsigned char w0 = *src++;
                        unsigned char w1 = *src++;
                        unsigned char w2 = *src++;
                        unsigned char w3 = *src++;
                        *dst_r++ = w0 << 2 | w1 >> 6;
                        *dst_g++ = (w1 & 0x3f) << 4 | w2 >> 4;
                        *dst_b++ = (w2 & 0xf) << 6 | w3 >> 2;
                }
        }
}

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
static void r12l_to_gbrp12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_g = (uint16_t *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_b = (uint16_t *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
			uint16_t tmp;
			tmp = src[BYTE_SWAP(0)];
			tmp |= (src[BYTE_SWAP(1)] & 0xf) << 8;
			*dst_r++ = tmp; // r0
			*dst_g++ = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // g0
			tmp = src[BYTE_SWAP(3)];
			src += 4;
			tmp |= (src[BYTE_SWAP(0)] & 0xf) << 8;
			*dst_b++ = tmp; // b0
			*dst_r++ = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // r1
			tmp = src[BYTE_SWAP(2)];
			tmp |= (src[BYTE_SWAP(3)] & 0xf) << 8;
			*dst_g++ = tmp; // g1
			tmp = src[BYTE_SWAP(3)] >> 4;
			src += 4;
			*dst_b++ = src[BYTE_SWAP(0)] << 4 | tmp; // b1
			tmp = src[BYTE_SWAP(1)];
			tmp |= (src[BYTE_SWAP(2)] & 0xf) << 8;
			*dst_r++ = tmp; // r2
			*dst_g++ = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // g2
			src += 4;
			tmp = src[BYTE_SWAP(0)];
			tmp |= (src[BYTE_SWAP(1)] & 0xf) << 8;
			*dst_b++ = tmp; // b2
			*dst_r++ = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // r3
			tmp = src[BYTE_SWAP(3)];
			src += 4;
			tmp |= (src[BYTE_SWAP(0)] & 0xf) << 8;
			*dst_g++ = tmp; // g3
			*dst_b++ = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // b3
			tmp = src[BYTE_SWAP(2)];
			tmp |= (src[BYTE_SWAP(3)] & 0xf) << 8;
			*dst_r++ = tmp; // r4
			tmp = src[BYTE_SWAP(3)] >> 4;
			src += 4;
			*dst_g++ = src[BYTE_SWAP(0)] << 4 | tmp; // g4
			tmp = src[BYTE_SWAP(1)];
			tmp |= (src[BYTE_SWAP(2)] & 0xf) << 8;
			*dst_b++ = tmp; // b4
			*dst_r++ = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // r5
			src += 4;
			tmp = src[BYTE_SWAP(0)];
			tmp |= (src[BYTE_SWAP(1)] & 0xf) << 8;
			*dst_g++ = tmp; // g5
			*dst_b++ = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // b5
			tmp = src[BYTE_SWAP(3)];
			src += 4;
			tmp |= (src[BYTE_SWAP(0)] & 0xf) << 8;
			*dst_r++ = tmp; // r6
			*dst_g++ = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // g6
			tmp = src[BYTE_SWAP(2)];
			tmp |= (src[BYTE_SWAP(3)] & 0xf) << 8;
			*dst_b++ = tmp; // b6
			tmp = src[BYTE_SWAP(3)] >> 4;
			src += 4;
			*dst_r++ = src[BYTE_SWAP(0)] << 4 | tmp; // r7
			tmp = src[BYTE_SWAP(1)];
			tmp |= (src[BYTE_SWAP(2)] & 0xf) << 8;
			*dst_g++ = tmp; // g7
			*dst_b++ = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // b7
			src += 4;
                }
        }
}
#endif

//
// av_to_uv_convert conversions
//
static void nv12_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cbcr = (char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                char *dst = dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                }
        }
}

static void rgb24_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoUYVY((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0], vc_get_linesize(width, UYVY), 0, 0, 0);
        }
}

static void memcpy_data(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        UNUSED(width);
        for (int y = 0; y < height; ++y) {
                memcpy(dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                frame->linesize[0]);
        }
}

static void rgb24_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoRGBA((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, RGBA), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void gbrp_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        uint8_t *buf = (uint8_t *) dst_buffer + y * pitch + x * 3;
                        int src_idx = y * frame->linesize[0] + x;
                        buf[0] = frame->data[2][src_idx]; // R
                        buf[1] = frame->data[0][src_idx]; // G
                        buf[2] = frame->data[1][src_idx]; // B
                }
        }
}

static void gbrp_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                uint32_t *line = (uint32_t *) ((uint8_t *) dst_buffer + y * pitch);
                int src_idx = y * frame->linesize[0];

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *line++ = frame->data[2][src_idx] << rgb_shift[R] |
                                frame->data[0][src_idx] << rgb_shift[G] |
                                frame->data[1][src_idx] << rgb_shift[B];
                        src_idx += 1;
                }
        }
}

static void gbrp10le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
		unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
			*dst++ = *src_r >> 2;
			*dst++ = (*src_r++ & 0x3) << 6 | *src_g >> 4;
			*dst++ = (*src_g++ & 0xf) << 4 | *src_b >> 6;
			*dst++ = (*src_b++ & 0x3f) << 2;
                }
        }
}

static void gbrp10le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
		unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
			*dst++ = *src_r++ >> 2;
			*dst++ = *src_g++ >> 2;
			*dst++ = *src_b++ >> 2;
                }
        }
}

static void gbrp10le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
		uint32_t *dst = (uint32_t *) (dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
			*dst++ = (*src_r++ >> 2) << rgb_shift[0] | (*src_g++ >> 2) << rgb_shift[1] |
                                (*src_b++ >> 2) << rgb_shift[2];
                }
        }
}

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
static void gbrp12le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
                        dst[BYTE_SWAP(0)] = *src_r & 0xff;
                        dst[BYTE_SWAP(1)] = (*src_g & 0xf) << 4 | *src_r++ >> 8;
                        dst[BYTE_SWAP(2)] = *src_g++ >> 4;
                        dst[BYTE_SWAP(3)] = *src_b & 0xff;
                        dst[4 + BYTE_SWAP(0)] = (*src_r & 0xf) << 4 | *src_b++ >> 8;
                        dst[4 + BYTE_SWAP(1)] = *src_r++ >> 4;
                        dst[4 + BYTE_SWAP(2)] = *src_g & 0xff;
                        dst[4 + BYTE_SWAP(3)] = (*src_b & 0xf) << 4 | *src_g++ >> 8;
                        dst[8 + BYTE_SWAP(0)] = *src_b++ >> 4;
                        dst[8 + BYTE_SWAP(1)] = *src_r & 0xff;
                        dst[8 + BYTE_SWAP(2)] = (*src_g & 0xf) << 4 | *src_r++ >> 8;
                        dst[8 + BYTE_SWAP(3)] = *src_g++ >> 4;
                        dst[12 + BYTE_SWAP(0)] = *src_b & 0xff;
                        dst[12 + BYTE_SWAP(1)] = (*src_r & 0xf) << 4 | *src_b++ >> 8;
                        dst[12 + BYTE_SWAP(2)] = *src_r++ >> 4;
                        dst[12 + BYTE_SWAP(3)] = *src_g & 0xff;
                        dst[16 + BYTE_SWAP(0)] = (*src_b & 0xf) << 4 | *src_g++ >> 8;
                        dst[16 + BYTE_SWAP(1)] = *src_b++ >> 4;
                        dst[16 + BYTE_SWAP(2)] = *src_r & 0xff;
                        dst[16 + BYTE_SWAP(3)] = (*src_g & 0xf) << 4 | *src_r++ >> 8;
                        dst[20 + BYTE_SWAP(0)] = *src_g++ >> 4;
                        dst[20 + BYTE_SWAP(1)] = *src_b & 0xff;
                        dst[20 + BYTE_SWAP(2)] = (*src_r & 0xf) << 4 | *src_b++ >> 8;
                        dst[20 + BYTE_SWAP(3)] = *src_r++ >> 4;;
                        dst[24 + BYTE_SWAP(0)] = *src_g & 0xff;
                        dst[24 + BYTE_SWAP(1)] = (*src_b & 0xf) << 4 | *src_g++ >> 8;
                        dst[24 + BYTE_SWAP(2)] = *src_b++ >> 4;
                        dst[24 + BYTE_SWAP(3)] = *src_r & 0xff;
                        dst[28 + BYTE_SWAP(0)] = (*src_g & 0xf) << 4 | *src_r++ >> 8;
                        dst[28 + BYTE_SWAP(1)] = *src_g++ >> 4;
                        dst[28 + BYTE_SWAP(2)] = *src_b & 0xff;
                        dst[28 + BYTE_SWAP(3)] = (*src_r & 0xf) << 4 | *src_b++ >> 8;
                        dst[32 + BYTE_SWAP(0)] = *src_r++ >> 4;
                        dst[32 + BYTE_SWAP(1)] = *src_g & 0xff;
                        dst[32 + BYTE_SWAP(2)] = (*src_b & 0xf) << 4 | *src_g++ >> 8;
                        dst[32 + BYTE_SWAP(3)] = *src_b++ >> 4;
                        dst += 36;
                }
        }
}

static void gbrp12le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_r++ >> 4;
                        *dst++ = *src_g++ >> 4;
                        *dst++ = *src_b++ >> 4;
                }
        }
}

static void gbrp12le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *) (frame->data[2] + frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *) (dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
			*dst++ = (*src_r++ >> 4) << rgb_shift[0] | (*src_g++ >> 4) << rgb_shift[1] |
                                (*src_b++ >> 4) << rgb_shift[2];
                }
        }
}
#endif

static void rgb48le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toRGBA((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, RGBA), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void rgb48le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toR12L((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, R12L), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void yuv420p_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (height + 1) / 2; ++y) {
                int scnd_row = y * 2 + 1;
                if (scnd_row == height) {
                        scnd_row = height - 1;
                }
                char *src_y1 = (char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                char *src_y2 = (char *) in_frame->data[0] + in_frame->linesize[0] * scnd_row;
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst1 = dst_buffer + (y * 2) * pitch;
                char *dst2 = dst_buffer + scnd_row * pitch;

                int x = 0;

#ifdef __SSE3__
                __m128i y1;
                __m128i y2;
                __m128i u1;
                __m128i u2;
                __m128i v1;
                __m128i v2;
                __m128i out1l;
                __m128i out1h;
                __m128i out2l;
                __m128i out2h;
                __m128i zero = _mm_set1_epi32(0);

                for (; x < width - 15; x += 16){
                        y1 = _mm_lddqu_si128((__m128i const*) src_y1);
                        y2 = _mm_lddqu_si128((__m128i const*) src_y2);
                        src_y1 += 16;
                        src_y2 += 16;

                        out1l = _mm_unpacklo_epi8(zero, y1);
                        out1h = _mm_unpackhi_epi8(zero, y1);
                        out2l = _mm_unpacklo_epi8(zero, y2);
                        out2h = _mm_unpackhi_epi8(zero, y2);

                        u1 = _mm_lddqu_si128((__m128i const*) src_cb);
                        v1 = _mm_lddqu_si128((__m128i const*) src_cr);
                        src_cb += 8;
                        src_cr += 8;

                        u1 = _mm_unpacklo_epi8(u1, zero);
                        v1 = _mm_unpacklo_epi8(v1, zero);
                        u2 = _mm_unpackhi_epi8(u1, zero);
                        v2 = _mm_unpackhi_epi8(v1, zero);
                        u1 = _mm_unpacklo_epi8(u1, zero);
                        v1 = _mm_unpacklo_epi8(v1, zero);

                        v1 = _mm_bslli_si128(v1, 2);
                        v2 = _mm_bslli_si128(v2, 2);

                        u1 = _mm_or_si128(u1, v1);
                        u2 = _mm_or_si128(u2, v2);

                        out1l = _mm_or_si128(out1l, u1);
                        out1h = _mm_or_si128(out1h, u2);
                        out2l = _mm_or_si128(out2l, u1);
                        out2h = _mm_or_si128(out2h, u2);

                        _mm_storeu_si128((__m128i *) dst1, out1l);
                        dst1 += 16;
                        _mm_storeu_si128((__m128i *) dst1, out1h);
                        dst1 += 16;
                        _mm_storeu_si128((__m128i *) dst2, out2l);
                        dst2 += 16;
                        _mm_storeu_si128((__m128i *) dst2, out2h);
                        dst2 += 16;
                }
#endif


                OPTIMIZED_FOR (; x < width - 1; x += 2) {
                        *dst1++ = *src_cb;
                        *dst1++ = *src_y1++;
                        *dst1++ = *src_cr;
                        *dst1++ = *src_y1++;

                        *dst2++ = *src_cb++;
                        *dst2++ = *src_y2++;
                        *dst2++ = *src_cr++;
                        *dst2++ = *src_y2++;
                }
                if (x < width) {
                        *dst1++ = *src_cb;
                        *dst1++ = *src_y1++;
                        *dst1++ = *src_cr;
                        *dst1++ = 0;

                        *dst2++ = *src_cb++;
                        *dst2++ = *src_y2++;
                        *dst2++ = *src_cr++;
                        *dst2++ = 0;
                }
        }
}

static void yuv420p_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint8_t *src_y1 = (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint8_t *src_y2 = (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src_cb << 2;
                        w1_0 = *src_cb << 2;
                        src_cb++;
                        w0_0 = w0_0 | (*src_y1++ << 2) << 10;
                        w1_0 = w1_0 | (*src_y2++ << 2) << 10;
                        w0_0 = w0_0 | (*src_cr << 2) << 20;
                        w1_0 = w1_0 | (*src_cr << 2) << 20;
                        src_cr++;

                        w0_1 = *src_y1++ << 2;
                        w1_1 = *src_y2++ << 2;
                        w0_1 = w0_1 | (*src_cb << 2) << 10;
                        w1_1 = w1_1 | (*src_cb << 2) << 10;
                        src_cb++;
                        w0_1 = w0_1 | (*src_y1++ << 2) << 20;
                        w1_1 = w1_1 | (*src_y2++ << 2) << 20;

                        w0_2 = *src_cr << 2;
                        w1_2 = *src_cr << 2;
                        src_cr++;
                        w0_2 = w0_2 | (*src_y1++ << 2) << 10;
                        w1_2 = w1_2 | (*src_y2++ << 2) << 10;
                        w0_2 = w0_2 | (*src_cb << 2) << 20;
                        w1_2 = w1_2 | (*src_cb << 2) << 20;
                        src_cb++;

                        w0_3 = *src_y1++;
                        w1_3 = *src_y2++;
                        w0_3 = w0_3 | (*src_cr << 2) << 10;
                        w1_3 = w1_3 | (*src_cr << 2) << 10;
                        src_cr++;
                        w0_3 = w0_3 | (*src_y1++ << 2) << 20;
                        w1_3 = w1_3 | (*src_y2++ << 2) << 20;

                        *dst1++ = w0_0;
                        *dst1++ = w0_1;
                        *dst1++ = w0_2;
                        *dst1++ = w0_3;

                        *dst2++ = w1_0;
                        *dst2++ = w1_1;
                        *dst2++ = w1_2;
                        *dst2++ = w1_3;
                }
        }
}

static void yuv422p_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cr++;
                        *dst++ = *src_y++;
                }
        }
}

static void yuv422p_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src_cb++ << 2;
                        w0_0 = w0_0 | (*src_y++ << 2) << 10;
                        w0_0 = w0_0 | (*src_cr++ << 2) << 20;

                        w0_1 = *src_y++ << 2;
                        w0_1 = w0_1 | (*src_cb++ << 2) << 10;
                        w0_1 = w0_1 | (*src_y++ << 2) << 20;

                        w0_2 = *src_cr++ << 2;
                        w0_2 = w0_2 | (*src_y++ << 2) << 10;
                        w0_2 = w0_2 | (*src_cb++ << 2) << 20;

                        w0_3 = *src_y++ << 2;
                        w0_3 = w0_3 | (*src_cr++ << 2) << 10;
                        w0_3 = w0_3 | (*src_y++ << 2) << 20;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}

static void yuv444p_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = (*src_cb + *(src_cb + 1)) / 2;
                        src_cb += 2;
                        *dst++ = *src_y++;
                        *dst++ = (*src_cr + *(src_cr + 1)) / 2;
                        src_cr += 2;
                        *dst++ = *src_y++;
                }
        }
}

static void yuv444p_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = ((src_cb[0] << 2) + (src_cb[1] << 2)) / 2;
                        w0_0 = w0_0 | (*src_y++ << 2) << 10;
                        w0_0 = w0_0 | ((src_cr[0] << 2) + (src_cr[1] << 2)) / 2 << 20;
                        src_cb += 2;
                        src_cr += 2;

                        w0_1 = *src_y++ << 2;
                        w0_1 = w0_1 | ((src_cb[0] << 2) + (src_cb[1] << 2)) / 2 << 10;
                        w0_1 = w0_1 | (*src_y++ << 2) << 20;
                        src_cb += 2;

                        w0_2 = ((src_cr[0] << 2) + (src_cr[1] << 2)) / 2;
                        w0_2 = w0_2 | (*src_y++ << 2) << 10;
                        w0_2 = w0_2 | ((src_cb[0] << 2) + (src_cb[1] << 2)) / 2 << 20;
                        src_cr += 2;
                        src_cb += 2;

                        w0_3 = *src_y++ << 2;
                        w0_3 = w0_3 | ((src_cr[0] << 2) + (src_cr[1] << 2)) / 2 << 10;
                        w0_3 = w0_3 | (*src_y++ << 2) << 20;
                        src_cr += 2;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}


/**
 * Changes pixel format from planar YUV 422 to packed RGB/A.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static inline void nv12_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift, bool rgba)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cbcr = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        int cb = *src_cbcr++ - 128;
                        int cr = *src_cbcr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst++ = 255;
                        }
                        y = *src_y++ << 16;
                        *dst++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst++ = 255;
                        }
                }
        }
}

static void nv12_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        nv12_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void nv12_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        nv12_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

/**
 * Changes pixel format from planar YUV 422 to packed RGB/A.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static inline void yuv422p_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift, bool rgba)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst++ = 255;
                        }
                        y = *src_y++ << 16;
                        *dst++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst++ = 255;
                        }
                }
        }
}

static void yuv422p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv422p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv422p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv422p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

/**
 * Changes pixel format from planar YUV 420 to packed RGB/A.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static inline void yuv420p_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift, bool rgba)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                unsigned char *src_y1 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                unsigned char *src_y2 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1);
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst1 = (unsigned char *) dst_buffer + pitch * (y * 2);
                unsigned char *dst2 = (unsigned char *) dst_buffer + pitch * (y * 2 + 1);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y1++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst1++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst1++ = 255;
                        }
                        y = *src_y1++ << 16;
                        *dst1++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst1++ = 255;
                        }
                        y = *src_y2++ << 16;
                        *dst2++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst2++ = 255;
                        }
                        y = *src_y2++ << 16;
                        *dst2++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst2++ = 255;
                        }
                }
        }
}

static void yuv420p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv420p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv420p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv420p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

/**
 * Changes pixel format from planar YUV 444 to packed RGB/A.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static inline void yuv444p_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift, bool rgba)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                        if (rgba) {
                                *dst++ = 255;
                        }
                }
        }
}

static void yuv444p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv444p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv444p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv444p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

static void yuv420p10le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src_cb;
                        w1_0 = *src_cb;
                        src_cb++;
                        w0_0 = w0_0 | (*src_y1++) << 10;
                        w1_0 = w1_0 | (*src_y2++) << 10;
                        w0_0 = w0_0 | (*src_cr) << 20;
                        w1_0 = w1_0 | (*src_cr) << 20;
                        src_cr++;

                        w0_1 = *src_y1++;
                        w1_1 = *src_y2++;
                        w0_1 = w0_1 | (*src_cb) << 10;
                        w1_1 = w1_1 | (*src_cb) << 10;
                        src_cb++;
                        w0_1 = w0_1 | (*src_y1++) << 20;
                        w1_1 = w1_1 | (*src_y2++) << 20;

                        w0_2 = *src_cr;
                        w1_2 = *src_cr;
                        src_cr++;
                        w0_2 = w0_2 | (*src_y1++) << 10;
                        w1_2 = w1_2 | (*src_y2++) << 10;
                        w0_2 = w0_2 | (*src_cb) << 20;
                        w1_2 = w1_2 | (*src_cb) << 20;
                        src_cb++;

                        w0_3 = *src_y1++;
                        w1_3 = *src_y2++;
                        w0_3 = w0_3 | (*src_cr) << 10;
                        w1_3 = w1_3 | (*src_cr) << 10;
                        src_cr++;
                        w0_3 = w0_3 | (*src_y1++) << 20;
                        w1_3 = w1_3 | (*src_y2++) << 20;

                        *dst1++ = w0_0;
                        *dst1++ = w0_1;
                        *dst1++ = w0_2;
                        *dst1++ = w0_3;

                        *dst2++ = w1_0;
                        *dst2++ = w1_1;
                        *dst2++ = w1_2;
                        *dst2++ = w1_3;
                }
        }
}

static void yuv422p10le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = *src_cb++;
                        w0_0 = w0_0 | (*src_y++) << 10;
                        w0_0 = w0_0 | (*src_cr++) << 20;

                        w0_1 = *src_y++;
                        w0_1 = w0_1 | (*src_cb++) << 10;
                        w0_1 = w0_1 | (*src_y++) << 20;

                        w0_2 = *src_cr++;
                        w0_2 = w0_2 | (*src_y++) << 10;
                        w0_2 = w0_2 | (*src_cb++) << 20;

                        w0_3 = *src_y++;
                        w0_3 = w0_3 | (*src_cr++) << 10;
                        w0_3 = w0_3 | (*src_y++) << 20;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}

static void yuv444p10le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = (src_cb[0] + src_cb[1]) / 2;
                        w0_0 = w0_0 | (*src_y++) << 10;
                        w0_0 = w0_0 | (src_cr[0] + src_cr[1]) / 2 << 20;
                        src_cb += 2;
                        src_cr += 2;

                        w0_1 = *src_y++;
                        w0_1 = w0_1 | (src_cb[0] + src_cb[1]) / 2 << 10;
                        w0_1 = w0_1 | (*src_y++) << 20;
                        src_cb += 2;

                        w0_2 = (src_cr[0] + src_cr[1]) / 2;
                        w0_2 = w0_2 | (*src_y++) << 10;
                        w0_2 = w0_2 | (src_cb[0] + src_cb[1]) / 2 << 20;
                        src_cr += 2;
                        src_cb += 2;

                        w0_3 = *src_y++;
                        w0_3 = w0_3 | (src_cr[0] + src_cr[1]) / 2 << 10;
                        w0_3 = w0_3 | (*src_y++) << 20;
                        src_cr += 2;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}

static void yuv420p10le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst1 = (uint8_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint8_t *dst2 = (uint8_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        uint8_t tmp;
                        // U
                        tmp = *src_cb++ >> 2;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 2;
                        *dst2++ = *src_y2++ >> 2;
                        // V
                        tmp = *src_cr++ >> 2;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 2;
                        *dst2++ = *src_y2++ >> 2;
                }
        }
}

static void yuv422p10le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++ >> 2;
                        *dst++ = *src_y++ >> 2;
                        *dst++ = *src_cr++ >> 2;
                        *dst++ = *src_y++ >> 2;
                }
        }
}

static void yuv444p10le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = (src_cb[0] + src_cb[1]) / 2 >> 2;
                        *dst++ = *src_y++ >> 2;
                        *dst++ = (src_cr[0] + src_cr[1]) / 2 >> 2;
                        *dst++ = *src_y++ >> 2;
                        src_cb += 2;
                        src_cr += 2;
                }
        }
}

static inline void yuv420p10le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift, bool rgba)
{
        decoder_t decoder = rgba ? vc_copylineUYVYtoRGBA : vc_copylineUYVYtoRGB;
        int linesize = vc_get_linesize(width, rgba ? RGBA : RGB);
        char *tmp = malloc(vc_get_linesize(width, UYVY) * height);
        char *uyvy = tmp;
        yuv420p10le_to_uyvy(uyvy, in_frame, width, height, vc_get_linesize(width, UYVY), rgb_shift);
        for (int i = 0; i < height; i++) {
                decoder((unsigned char *) dst_buffer, (unsigned char *) uyvy, linesize,
                                rgb_shift[R], rgb_shift[G], rgb_shift[B]);
                uyvy += vc_get_linesize(width, UYVY);
                dst_buffer += pitch;
        }
        free(tmp);
}

static inline void yuv420p10le_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv420p10le_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static inline void yuv420p10le_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv420p10le_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

static void yuv422p10le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift, bool rgba)
{
        decoder_t decoder = rgba ? vc_copylineUYVYtoRGBA : vc_copylineUYVYtoRGB;
        int linesize = vc_get_linesize(width, rgba ? RGBA : RGB);
        char *tmp = malloc(vc_get_linesize(width, UYVY) * height);
        char *uyvy = tmp;
        yuv422p10le_to_uyvy(uyvy, in_frame, width, height, vc_get_linesize(width, UYVY), rgb_shift);
        for (int i = 0; i < height; i++) {
                decoder((unsigned char *) dst_buffer, (unsigned char *) uyvy, linesize,
                                rgb_shift[R], rgb_shift[G], rgb_shift[B]);
                uyvy += vc_get_linesize(width, UYVY);
                dst_buffer += pitch;
        }
        free(tmp);
}

static inline void yuv422p10le_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv422p10le_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static inline void yuv422p10le_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        yuv422p10le_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}


static inline void yuv444p10le_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; y++) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        int cb = (*src_cb++ >> 2) - 128;
                        int cr = (*src_cr++ >> 2) - 128;
                        int y = (*src_y++ >> 2) << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = MIN(MAX(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = MIN(MAX(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

static inline void yuv444p10le_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        for (int y = 0; y < height; y++) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        int cb = (*src_cb++ >> 2) - 128;
                        int cr = (*src_cr++ >> 2) - 128;
                        int y = (*src_y++ >> 2) << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
			*dst++ = (MIN(MAX(r + y, 0), (1<<24) - 1) >> 16) << rgb_shift[0] | (MIN(MAX(g + y, 0), (1<<24) - 1) >> 16) << rgb_shift[1] |
                                (MIN(MAX(b + y, 0), (1<<24) - 1) >> 16) << rgb_shift[2];
                }
        }
}

static void p010le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint8_t *src_y1 = (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint8_t *src_y2 = (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint8_t *src_cbcr = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;
                        uint32_t w1_0, w1_1, w1_2, w1_3;

                        w0_0 = *src_cbcr << 2; // Cb0
                        w1_0 = *src_cbcr << 2;
                        src_cbcr++; // Cr0
                        w0_0 = w0_0 | (*src_y1++ << 2) << 10;
                        w1_0 = w1_0 | (*src_y2++ << 2) << 10;
                        w0_0 = w0_0 | (*src_cbcr << 2) << 20;
                        w1_0 = w1_0 | (*src_cbcr << 2) << 20;
                        src_cbcr++; // Cb1

                        w0_1 = *src_y1++ << 2;
                        w1_1 = *src_y2++ << 2;
                        w0_1 = w0_1 | (*src_cbcr << 2) << 10;
                        w1_1 = w1_1 | (*src_cbcr << 2) << 10;
                        src_cbcr++; // Cr1
                        w0_1 = w0_1 | (*src_y1++ << 2) << 20;
                        w1_1 = w1_1 | (*src_y2++ << 2) << 20;

                        w0_2 = *src_cbcr << 2;
                        w1_2 = *src_cbcr << 2;
                        src_cbcr++;
                        w0_2 = w0_2 | (*src_y1++ << 2) << 10;
                        w1_2 = w1_2 | (*src_y2++ << 2) << 10;
                        w0_2 = w0_2 | (*src_cbcr << 2) << 20;
                        w1_2 = w1_2 | (*src_cbcr << 2) << 20;
                        src_cbcr++;

                        w0_3 = *src_y1++;
                        w1_3 = *src_y2++;
                        w0_3 = w0_3 | (*src_cbcr << 2) << 10;
                        w1_3 = w1_3 | (*src_cbcr << 2) << 10;
                        src_cbcr++;
                        w0_3 = w0_3 | (*src_y1++ << 2) << 20;
                        w1_3 = w1_3 | (*src_y2++ << 2) << 20;

                        *dst1++ = w0_0;
                        *dst1++ = w0_1;
                        *dst1++ = w0_2;
                        *dst1++ = w0_3;

                        *dst2++ = w1_0;
                        *dst2++ = w1_1;
                        *dst2++ = w1_2;
                        *dst2++ = w1_3;
                }
        }
}

static void p010le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cbcr = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *dst1 = (uint8_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint8_t *dst2 = (uint8_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        uint8_t tmp;
                        // U
                        tmp = *src_cbcr++ >> 8;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 8;
                        *dst2++ = *src_y2++ >> 8;
                        // V
                        tmp = *src_cbcr++ >> 8;
                        *dst1++ = tmp;
                        *dst2++ = tmp;
                        // Y
                        *dst1++ = *src_y1++ >> 8;
                        *dst2++ = *src_y2++ >> 8;
                }
        }
}

#ifdef HWACC_VDPAU
static void av_vdpau_to_ug_vdpau(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(width);
        UNUSED(height);
        UNUSED(pitch);
        UNUSED(rgb_shift);

        struct video_frame_callbacks *callbacks = in_frame->opaque;

        hw_vdpau_frame *out = (hw_vdpau_frame *) dst_buffer;

        hw_vdpau_frame_init(out);

        hw_vdpau_frame_from_avframe(out, in_frame);

        callbacks->recycle = hw_vdpau_recycle_callback; 
        callbacks->copy = hw_vdpau_copy_callback; 
}
#endif

//
// conversion dispatchers
//
/**
 * @brief returns list of available conversion. Terminated by uv_to_av_conversion::src == VIDEO_CODEC_NONE
 */
const struct uv_to_av_conversion *get_uv_to_av_conversions() {
        /**
         * Conversions from UltraGrid to FFMPEG formats.
         *
         * Currently do not add an "upgrade" conversion (UYVY->10b) because also
         * UltraGrid decoder can be used first and thus conversion v210->UYVY->10b
         * may be used resulting in a precision loss. If needed, put the upgrade
         * conversions below the others.
         */
        static const struct uv_to_av_conversion uv_to_av_conversions[] = {
                { v210, AV_PIX_FMT_YUV420P10LE, v210_to_yuv420p10le },
                { v210, AV_PIX_FMT_YUV422P10LE, v210_to_yuv422p10le },
                { v210, AV_PIX_FMT_YUV444P10LE, v210_to_yuv444p10le },
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(55, 15, 100) // FFMPEG commit c2869b4640f
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
                { R10k, AV_PIX_FMT_BGR0, r10k_to_bgr0 },
                { R10k, AV_PIX_FMT_GBRP10LE, r10k_to_gbrp10le },
                { R10k, AV_PIX_FMT_YUV422P10LE, r10k_to_yuv422p10le },
#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
                { R12L, AV_PIX_FMT_GBRP12LE, r12l_to_gbrp12le },
#endif
                { 0, 0, 0 }
        };
        return uv_to_av_conversions;
}

/**
 * @brief returns list of available conversion. Terminated by uv_to_av_conversion::uv_codec == VIDEO_CODEC_NONE
 */
const struct av_to_uv_conversion *get_av_to_uv_conversions() {
        static const struct av_to_uv_conversion av_to_uv_conversions[] = {
                // 10-bit YUV
                {AV_PIX_FMT_YUV420P10LE, v210, yuv420p10le_to_v210, true},
                {AV_PIX_FMT_YUV420P10LE, UYVY, yuv420p10le_to_uyvy, false},
                {AV_PIX_FMT_YUV420P10LE, RGB, yuv420p10le_to_rgb24, false},
                {AV_PIX_FMT_YUV420P10LE, RGBA, yuv420p10le_to_rgb32, false},
                {AV_PIX_FMT_YUV422P10LE, v210, yuv422p10le_to_v210, true},
                {AV_PIX_FMT_YUV422P10LE, UYVY, yuv422p10le_to_uyvy, false},
                {AV_PIX_FMT_YUV422P10LE, RGB, yuv422p10le_to_rgb24, false},
                {AV_PIX_FMT_YUV422P10LE, RGBA, yuv422p10le_to_rgb32, false},
                {AV_PIX_FMT_YUV444P10LE, v210, yuv444p10le_to_v210, true},
                {AV_PIX_FMT_YUV444P10LE, UYVY, yuv444p10le_to_uyvy, false},
                {AV_PIX_FMT_YUV444P10LE, RGB, yuv444p10le_to_rgb24, false},
                {AV_PIX_FMT_YUV444P10LE, RGBA, yuv444p10le_to_rgb32, false},
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(55, 15, 100) // FFMPEG commit c2869b4640f
                {AV_PIX_FMT_P010LE, v210, p010le_to_v210, true},
                {AV_PIX_FMT_P010LE, UYVY, p010le_to_uyvy, true},
#endif
                // 8-bit YUV
                {AV_PIX_FMT_YUV420P, v210, yuv420p_to_v210, false},
                {AV_PIX_FMT_YUV420P, UYVY, yuv420p_to_uyvy, true},
                {AV_PIX_FMT_YUV420P, RGB, yuv420p_to_rgb24, false},
                {AV_PIX_FMT_YUV420P, RGBA, yuv420p_to_rgb32, false},
                {AV_PIX_FMT_YUV422P, v210, yuv422p_to_v210, false},
                {AV_PIX_FMT_YUV422P, UYVY, yuv422p_to_uyvy, true},
                {AV_PIX_FMT_YUV422P, RGB, yuv422p_to_rgb24, false},
                {AV_PIX_FMT_YUV422P, RGBA, yuv422p_to_rgb32, false},
                {AV_PIX_FMT_YUV444P, v210, yuv444p_to_v210, false},
                {AV_PIX_FMT_YUV444P, UYVY, yuv444p_to_uyvy, true},
                {AV_PIX_FMT_YUV444P, RGB, yuv444p_to_rgb24, false},
                {AV_PIX_FMT_YUV444P, RGBA, yuv444p_to_rgb32, false},
                // 8-bit YUV (JPEG color range)
                {AV_PIX_FMT_YUVJ420P, v210, yuv420p_to_v210, false},
                {AV_PIX_FMT_YUVJ420P, UYVY, yuv420p_to_uyvy, true},
                {AV_PIX_FMT_YUVJ420P, RGB, yuv420p_to_rgb24, false},
                {AV_PIX_FMT_YUVJ420P, RGBA, yuv420p_to_rgb32, false},
                {AV_PIX_FMT_YUVJ422P, v210, yuv422p_to_v210, false},
                {AV_PIX_FMT_YUVJ422P, UYVY, yuv422p_to_uyvy, true},
                {AV_PIX_FMT_YUVJ422P, RGB, yuv422p_to_rgb24, false},
                {AV_PIX_FMT_YUVJ422P, RGBA, yuv422p_to_rgb32, false},
                {AV_PIX_FMT_YUVJ444P, v210, yuv444p_to_v210, false},
                {AV_PIX_FMT_YUVJ444P, UYVY, yuv444p_to_uyvy, true},
                {AV_PIX_FMT_YUVJ444P, RGB, yuv444p_to_rgb24, false},
                {AV_PIX_FMT_YUVJ444P, RGBA, yuv444p_to_rgb32, false},
                // 8-bit YUV (NV12)
                {AV_PIX_FMT_NV12, UYVY, nv12_to_uyvy, true},
                {AV_PIX_FMT_NV12, RGB, nv12_to_rgb24, false},
                {AV_PIX_FMT_NV12, RGBA, nv12_to_rgb32, false},
                // RGB
                {AV_PIX_FMT_GBRP, RGB, gbrp_to_rgb, true},
                {AV_PIX_FMT_GBRP, RGBA, gbrp_to_rgba, true},
                {AV_PIX_FMT_RGB24, UYVY, rgb24_to_uyvy, false},
                {AV_PIX_FMT_RGB24, RGB, memcpy_data, true},
                {AV_PIX_FMT_RGB24, RGBA, rgb24_to_rgb32, false},
                {AV_PIX_FMT_GBRP10LE, R10k, gbrp10le_to_r10k, true},
                {AV_PIX_FMT_GBRP10LE, RGB, gbrp10le_to_rgb, false},
                {AV_PIX_FMT_GBRP10LE, RGBA, gbrp10le_to_rgba, false},
#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
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
                {0, 0, 0, 0}
        };
        return av_to_uv_conversions;
}

