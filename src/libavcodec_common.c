/**
 * @file   libavcodec_common.c
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

#ifdef HAVE_SWSCALE
#include <libswscale/swscale.h>
#endif // defined HAVE_SWSCALE

#include "host.h"
#include "hwaccel_vdpau.h"
#include "hwaccel_rpi4.h"
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

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunknown-pragmas"
#pragma clang diagnostic warning "-Wpass-failed"

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

//
// uv_to_av_convert conversions
//
//
/* @brief Color space coedfficients - RGB full range to YCbCr bt. 709 limited range
 *
 * RGB should use SDI full range [1<<(depth-8)..255<<(depth-8)-1], see [limits]
 *
 * Scaled by 1<<COMP_BASE, footroom 16/255, headroom 235/255 (luma), 240/255 (chroma); limits [2^(depth-8)..255*2^(depth-8)-1]
 * matrix Y = [ 0.182586, 0.614231, 0.062007; -0.100643, -0.338572, 0.4392157; 0.4392157, -0.398942, -0.040274 ]
 * * [coefficients]: https://gist.github.com/yohhoy/dafa5a47dade85d8b40625261af3776a "Rec. 709 coefficients"
 * * [limits]:       https://tech.ebu.ch/docs/r/r103.pdf                             "SDI limits"
 * @todo
 * Use this transformations in all conversions.
 * @{
 */
#define FULL_FOOT(depth) (1<<((depth)-8))
#define FULL_HEAD(depth) ((255<<((depth)-8))-1)
#define CLAMP_FULL(val, depth) MIN(FULL_HEAD(depth), MAX((val), FULL_FOOT(depth)))
typedef int32_t comp_type_t; // int32_t provides much better performance than int_fast32_t
#define COMP_BASE (sizeof(comp_type_t) == 4 ? 14 : 18) // computation will be less precise when comp_type_t is 32 bit
static_assert(sizeof(comp_type_t) * 8 >= COMP_BASE + 18, "comp_type_t not wide enough (we are computing in up to 16 bits!)");
static const comp_type_t y_r = (0.2126*219/255) * (1<<COMP_BASE);
static const comp_type_t y_g = (0.7152*219/255) * (1<<COMP_BASE);
static const comp_type_t y_b = (0.0722*219/255) * (1<<COMP_BASE);
static const comp_type_t cb_r = (-0.2126/1.8556*224/255) * (1<<COMP_BASE);
static const comp_type_t cb_g = (-0.7152/1.8556*224/255) * (1<<COMP_BASE);
static const comp_type_t cb_b = ((1-0.0722)/1.8556*224/255) * (1<<COMP_BASE);
static const comp_type_t cr_r = ((1-0.2126)/1.5748*224/255) * (1<<COMP_BASE);
static const comp_type_t cr_g = (-0.7152/1.5748*224/255) * (1<<COMP_BASE);
static const comp_type_t cr_b = (-0.0722/1.5748*224/255) * (1<<COMP_BASE);
#define RGB_TO_Y_709_SCALED(r, g, b) ((r) * y_r + (g) * y_g + (b) * y_b)
#define RGB_TO_CB_709_SCALED(r, g, b) ((r) * cb_r + (g) * cb_g + (b) * cb_b)
#define RGB_TO_CR_709_SCALED(r, g, b) ((r) * cr_r + (g) * cr_g + (b) * cr_b)

//  matrix Y1^-1 = inv(Y)
static const comp_type_t y_scale = 1.164383 * (1<<COMP_BASE); // precomputed value, Y multiplier is same for all channels
//static const comp_type_t r_y = 1; // during computation already contained in y_scale
//static const comp_type_t r_cb = 0;
static const comp_type_t r_cr = 1.792741 * (1<<COMP_BASE);
//static const comp_type_t g_y = 1;
static const comp_type_t g_cb = -0.213249 * (1<<COMP_BASE);
static const comp_type_t g_cr = -0.532909 * (1<<COMP_BASE);
//static const comp_type_t b_y = 1;
static const comp_type_t b_cb = 2.112402 * (1<<COMP_BASE);
//static const comp_type_t b_cr = 0;
#define YCBCR_TO_R_709_SCALED(y, cb, cr) ((y) /* * r_y */ /* + (cb) * r_cb */ + (cr) * r_cr)
#define YCBCR_TO_G_709_SCALED(y, cb, cr) ((y) /* * g_y */    + (cb) * g_cb    + (cr) * g_cr)
#define YCBCR_TO_B_709_SCALED(y, cb, cr) ((y) /* * b_y */    + (cb) * b_cb /* + (cr) * b_cr */)
/// @}

#define FORMAT_RGBA(r, g, b, depth) (~(0xFFU << (rgb_shift[R]) | 0xFFU << (rgb_shift[G]) | 0xFFU << (rgb_shift[B])) | \
        (CLAMP_FULL((r), (depth)) << rgb_shift[R] | CLAMP_FULL((g), (depth)) << rgb_shift[G] | CLAMP_FULL((b), (depth)) << rgb_shift[B]))

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
                        yuv = _mm_lddqu_si128((__m128i const*)(const void *) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*)(const void *) src2);
                        src += 16;
                        src2 += 16;

                        y1 = _mm_and_si128(ymask, yuv);
                        y1 = _mm_bsrli_si128(y1, 1);
                        y2 = _mm_and_si128(ymask, yuv2);
                        y2 = _mm_bsrli_si128(y2, 1);

                        uv = _mm_andnot_si128(ymask, yuv);
                        uv2 = _mm_andnot_si128(ymask, yuv2);

                        uv = _mm_avg_epu8(uv, uv2);

                        yuv = _mm_lddqu_si128((__m128i const*)(const void *) src);
                        yuv2 = _mm_lddqu_si128((__m128i const*)(const void *) src2);
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
                        _mm_storeu_si128((__m128i *)(void *) dst_y, dsty);
                        _mm_storeu_si128((__m128i *)(void *) dst_y2, dsty2);
                        _mm_storeu_si128((__m128i *)(void *) dst_cbcr, dstuv);
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
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *)(void *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y / 2);

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
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

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
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

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

static void v210_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 1) {
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0 = *src++;
                        uint32_t w0_1 = *src++;
                        uint32_t w0_2 = *src++;
                        uint32_t w0_3 = *src++;

                        *dst_y++ = ((w0_0 >> 10U) & 0x3FFU) << 6U;
                        *dst_y++ = (w0_1 & 0x3FFU) << 6U;
                        *dst_y++ = ((w0_1 >> 20U) & 0x3FFU) << 6U;
                        *dst_y++ = ((w0_2 >> 10U) & 0x3FFU) << 6U;
                        *dst_y++ = (w0_3 & 0x3FFU) << 6U;
                        *dst_y++ = ((w0_3 >> 20U) & 0x3FFU) << 6U;

                        *dst_cb++ = (w0_0 & 0x3FFU) << 6U;
                        *dst_cb++ = (w0_0 & 0x3FFU) << 6U;
                        *dst_cb++ = ((w0_1 >> 10U) & 0x3FFU) << 6U;
                        *dst_cb++ = ((w0_1 >> 10U) & 0x3FFU) << 6U;
                        *dst_cb++ = ((w0_2 >> 20U) & 0x3FFU) << 6U;
                        *dst_cb++ = ((w0_2 >> 20U) & 0x3FFU) << 6U;

                        *dst_cr++ = ((w0_0 >> 20U) & 0x3FFU) << 6U;
                        *dst_cr++ = ((w0_0 >> 20U) & 0x3FFU) << 6U;
                        *dst_cr++ = (w0_2 & 0x3FFU) << 6U;
                        *dst_cr++ = (w0_2 & 0x3FFU) << 6U;
                        *dst_cr++ = ((w0_3 >> 10U) & 0x3FFU) << 6U;
                        *dst_cr++ = ((w0_3 >> 10U) & 0x3FFU) << 6U;
                }
        }
}

static void v210_to_p010le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 4 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        for(int y = 0; y < height; y += 2) {
                /*  every even row */
                uint32_t *src = (uint32_t *)(void *) (in_data + y * vc_get_linesize(width, v210));
                /*  every odd row */
                uint32_t *src2 = (uint32_t *)(void *) (in_data + (y + 1) * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_y2 = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * (y + 1));
                uint16_t *dst_cbcr = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y / 2);

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
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

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
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
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

/**
 * Converts to yuv444p 10/12/14 le
 */
static inline void r10k_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, R10k);
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                unsigned char *src = in_data + y * src_linesize;
                OPTIMIZED_FOR(int x = 0; x < width; x++){
                        comp_type_t r = src[0] << 2 | src[1] >> 6;
                        comp_type_t g = (src[1] & 0x3F ) << 4 | src[2] >> 4;
                        comp_type_t b = (src[2] & 0x0F) << 6 | src[3] >> 2;

                        comp_type_t res_y = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE+10-depth)) + (1<<(depth-4));
                        comp_type_t res_cb = (RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE+10-depth)) + (1<<(depth-1));
                        comp_type_t res_cr = (RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE+10-depth)) + (1<<(depth-1));

                        *dst_y++ = MIN(MAX(res_y, 1<<(depth-4)), 235 * (1<<(depth-8)));
                        *dst_cb++ = MIN(MAX(res_cb, 1<<(depth-4)), 240 * (1<<(depth-8)));
                        *dst_cr++ = MIN(MAX(res_cr, 1<<(depth-4)), 240 * (1<<(depth-8)));
                        src += 4;
                }
        }
}

static void r10k_to_yuv444p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void r10k_to_yuv444p12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void r10k_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

// RGB full range to YCbCr bt. 709 limited range
static inline void r12l_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

#define WRITE_RES \
        res_y = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE+12-depth)) + (1<<(depth-4));\
        res_cb = (RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE+12-depth)) + (1<<(depth-1));\
        res_cr = (RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE+12-depth)) + (1<<(depth-1));\
        *dst_y++ = MIN(MAX(res_y, 1<<(depth-4)), 235 * (1<<(depth-8)));\
        *dst_cb++ = MIN(MAX(res_cb, 1<<(depth-4)), 240 * (1<<(depth-8)));\
        *dst_cr++ = MIN(MAX(res_cr, 1<<(depth-4)), 240 * (1<<(depth-8)));

        const int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
			comp_type_t r = 0;
			comp_type_t g = 0;
			comp_type_t b = 0;
                        comp_type_t res_y = 0;
                        comp_type_t res_cb = 0;
                        comp_type_t res_cr = 0;

			r = src[BYTE_SWAP(0)];
			r |= (src[BYTE_SWAP(1)] & 0xF) << 8;
			g = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // g0
			b = src[BYTE_SWAP(3)];
			src += 4;

			b |= (src[BYTE_SWAP(0)] & 0xF) << 8;
                        WRITE_RES // 0
			r = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // r1
			g = src[BYTE_SWAP(2)];
			g |= (src[BYTE_SWAP(3)] & 0xF) << 8;
			b = src[BYTE_SWAP(3)] >> 4;
			src += 4;

			b |= src[BYTE_SWAP(0)] << 4; // b1
                        WRITE_RES // 1
			r = src[BYTE_SWAP(1)];
			r |= (src[BYTE_SWAP(2)] & 0xF) << 8;
			g = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // g2
			src += 4;

			b = src[BYTE_SWAP(0)];
			b |= (src[BYTE_SWAP(1)] & 0xF) << 8;
                        WRITE_RES // 2
			r = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // r3
			g = src[BYTE_SWAP(3)];
			src += 4;

			g |= (src[BYTE_SWAP(0)] & 0xF) << 8;
			b = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // b3
                        WRITE_RES // 3
			r = src[BYTE_SWAP(2)];
			r |= (src[BYTE_SWAP(3)] & 0xF) << 8;
			g = src[BYTE_SWAP(3)] >> 4;
			src += 4;

			g |= src[BYTE_SWAP(0)] << 4; // g4
			b = src[BYTE_SWAP(1)];
			b |= (src[BYTE_SWAP(2)] & 0xF) << 8;
			WRITE_RES // 4
			r = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // r5
			src += 4;

			g = src[BYTE_SWAP(0)];
			g |= (src[BYTE_SWAP(1)] & 0xF) << 8;
			b = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // b5
                        WRITE_RES // 5
			r = src[BYTE_SWAP(3)];
			src += 4;

			r |= (src[BYTE_SWAP(0)] & 0xF) << 8;
			g = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(0)] >> 4; // g6
			b = src[BYTE_SWAP(2)];
			b |= (src[BYTE_SWAP(3)] & 0xF) << 8;
                        WRITE_RES // 6
			r = src[BYTE_SWAP(3)] >> 4;
			src += 4;

			r |= src[BYTE_SWAP(0)] << 4; // r7
			g = src[BYTE_SWAP(1)];
			g |= (src[BYTE_SWAP(2)] & 0xF) << 8;
			b = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(2)] >> 4; // b7
                        WRITE_RES // 7
			src += 4;
                }
        }
}

static void r12l_to_yuv444p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void r12l_to_yuv444p12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void r12l_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

/// @brief Converts RG48 to yuv444p 10/12/14 le
static inline void rg48_to_yuv444pXXle(int depth, AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, RG48);
        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                uint16_t *src = (uint16_t *)(void *) (in_data + y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < width; x++){
                        comp_type_t r = *src++;
                        comp_type_t g = *src++;
                        comp_type_t b = *src++;

                        comp_type_t res_y = (RGB_TO_Y_709_SCALED(r, g, b) >> (COMP_BASE+16-depth)) + (1<<(depth-4));
                        comp_type_t res_cb = (RGB_TO_CB_709_SCALED(r, g, b) >> (COMP_BASE+16-depth)) + (1<<(depth-1));
                        comp_type_t res_cr = (RGB_TO_CR_709_SCALED(r, g, b) >> (COMP_BASE+16-depth)) + (1<<(depth-1));

                        *dst_y++ = MIN(MAX(res_y, 1<<(depth-4)), 235 * (1<<(depth-8)));
                        *dst_cb++ = MIN(MAX(res_cb, 1<<(depth-4)), 240 * (1<<(depth-8)));
                        *dst_cr++ = MIN(MAX(res_cr, 1<<(depth-4)), 240 * (1<<(depth-8)));
                }
        }
}

static void rg48_to_yuv444p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(10, out_frame, in_data, width, height);
}

static void rg48_to_yuv444p12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(12, out_frame, in_data, width, height);
}

static void rg48_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        rg48_to_yuv444pXXle(16, out_frame, in_data, width, height);
}

static inline void y216_to_yuv422pXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int depth)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, Y216);

        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                uint16_t *src = (uint16_t *)(void *) (in_data + y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < (width + 1) / 2; x++){
                        *dst_y++ = *src++ >> (16U - depth);
                        *dst_cb++ = *src++ >> (16U - depth);
                        *dst_y++ = *src++ >> (16U - depth);
                        *dst_cr++ = *src++ >> (16U - depth);
                }
        }
}

static void y216_to_yuv422p10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        y216_to_yuv422pXXle(out_frame, in_data, width, height, 10);
}

static void y216_to_yuv422p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        y216_to_yuv422pXXle(out_frame, in_data, width, height, 16);
}

static void y216_to_yuv444p16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        const int src_linesize = vc_get_linesize(width, Y216);

        for(int y = 0; y < height; y++) {
                uint16_t *dst_y = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_cb = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_cr = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);
                uint16_t *src = (uint16_t *)(void *) (in_data + y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < (width + 1) / 2; x++){
                        *dst_y++ = *src++;
                        dst_cb[0] = dst_cb[1] = *src++;
                        dst_cb += 2;
                        *dst_y++ = *src++;
                        dst_cr[0] = dst_cr[1] = *src++;
                        dst_cr += 2;
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

static inline void r10k_to_gbrpXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int depth)
{
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        int src_linesize = vc_get_linesize(width, R10k);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_g = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_b = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        unsigned char w0 = *src++;
                        unsigned char w1 = *src++;
                        unsigned char w2 = *src++;
                        unsigned char w3 = *src++;
                        *dst_r++ = (w0 << 2U | w1 >> 6U) << (depth - 10U);
                        *dst_g++ = ((w1 & 0x3FU) << 4U | w2 >> 4U) << (depth - 10U);
                        *dst_b++ = ((w2 & 0xFU) << 6U | w3 >> 2U) << (depth - 10U);
                }
        }
}

static void r10k_to_gbrp10le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_gbrpXXle(out_frame, in_data, width, height, 10U);
}

static void r10k_to_gbrp16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r10k_to_gbrpXXle(out_frame, in_data, width, height, 16U);
}

#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif

/// @note out_depth needs to be at least 12
static inline void r12l_to_gbrpXXle(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height, unsigned int out_depth)
{
        assert(out_depth >= 12);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

#undef S
#define S(x) ((x) << (out_depth - 12U))

        int src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                unsigned char *src = in_data + y * src_linesize;
                uint16_t *dst_g = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_b = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
                        uint16_t tmp = src[BYTE_SWAP(0)];
                        tmp |= (src[BYTE_SWAP(1)] & 0xFU) << 8U;
                        *dst_r++ = S(tmp); // r0
                        *dst_g++ = S(src[BYTE_SWAP(2)] << 4U | src[BYTE_SWAP(1)] >> 4U); // g0
                        tmp = src[BYTE_SWAP(3)];
                        src += 4;
                        tmp |= (src[BYTE_SWAP(0)] & 0xFU) << 8U;
                        *dst_b++ = S(tmp); // b0
                        *dst_r++ = S(src[BYTE_SWAP(1)] << 4U | src[BYTE_SWAP(0)] >> 4U); // r1
                        tmp = src[BYTE_SWAP(2)];
                        tmp |= (src[BYTE_SWAP(3)] & 0xFU) << 8U;
                        *dst_g++ = S(tmp); // g1
                        tmp = src[BYTE_SWAP(3)] >> 4U;
                        src += 4;
                        *dst_b++ = S(src[BYTE_SWAP(0)] << 4U | tmp); // b1
                        tmp = src[BYTE_SWAP(1)];
                        tmp |= (src[BYTE_SWAP(2)] & 0xFU) << 8U;
                        *dst_r++ = S(tmp); // r2
                        *dst_g++ = S(src[BYTE_SWAP(3)] << 4U | src[BYTE_SWAP(2)] >> 4U); // g2
                        src += 4;
                        tmp = src[BYTE_SWAP(0)];
                        tmp |= (src[BYTE_SWAP(1)] & 0xFU) << 8U;
                        *dst_b++ = S(tmp); // b2
                        *dst_r++ = S(src[BYTE_SWAP(2)] << 4U | src[BYTE_SWAP(1)] >> 4U); // r3
                        tmp = src[BYTE_SWAP(3)];
                        src += 4;
                        tmp |= (src[BYTE_SWAP(0)] & 0xFU) << 8U;
                        *dst_g++ = S(tmp); // g3
                        *dst_b++ = S(src[BYTE_SWAP(1)] << 4U | src[BYTE_SWAP(0)] >> 4U); // b3
                        tmp = src[BYTE_SWAP(2)];
                        tmp |= (src[BYTE_SWAP(3)] & 0xFU) << 8U;
                        *dst_r++ = S(tmp); // r4
                        tmp = src[BYTE_SWAP(3)] >> 4U;
                        src += 4;
                        *dst_g++ = S(src[BYTE_SWAP(0)] << 4U | tmp); // g4
                        tmp = src[BYTE_SWAP(1)];
                        tmp |= (src[BYTE_SWAP(2)] & 0xFU) << 8U;
                        *dst_b++ = S(tmp); // b4
                        *dst_r++ = S(src[BYTE_SWAP(3)] << 4U | src[BYTE_SWAP(2)] >> 4U); // r5
                        src += 4;
                        tmp = src[BYTE_SWAP(0)];
                        tmp |= (src[BYTE_SWAP(1)] & 0xFU) << 8U;
                        *dst_g++ = S(tmp); // g5
                        *dst_b++ = S(src[BYTE_SWAP(2)] << 4U | src[BYTE_SWAP(1)] >> 4U); // b5
                        tmp = src[BYTE_SWAP(3)];
                        src += 4;
                        tmp |= (src[BYTE_SWAP(0)] & 0xFU) << 8U;
                        *dst_r++ = S(tmp); // r6
                        *dst_g++ = S(src[BYTE_SWAP(1)] << 4U | src[BYTE_SWAP(0)] >> 4U); // g6
                        tmp = src[BYTE_SWAP(2)];
                        tmp |= (src[BYTE_SWAP(3)] & 0xFU) << 8U;
                        *dst_b++ = S(tmp); // b6
                        tmp = src[BYTE_SWAP(3)] >> 4U;
                        src += 4;
                        *dst_r++ = S(src[BYTE_SWAP(0)] << 4U | tmp); // r7
                        tmp = src[BYTE_SWAP(1)];
                        tmp |= (src[BYTE_SWAP(2)] & 0xFU) << 8U;
                        *dst_g++ = S(tmp); // g7
                        *dst_b++ = S(src[BYTE_SWAP(3)] << 4U | src[BYTE_SWAP(2)] >> 4U); // b7
                        src += 4;
                }
        }
}

static void r12l_to_gbrp16le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_frame, in_data, width, height, 16U);
}

#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
static void r12l_to_gbrp12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        r12l_to_gbrpXXle(out_frame, in_data, width, height, 12U);
}

static void rg48_to_gbrp12le(AVFrame * __restrict out_frame, unsigned char * __restrict in_data, int width, int height)
{
        assert((uintptr_t) in_data % 2 == 0);
        assert((uintptr_t) out_frame->linesize[0] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[1] % 2 == 0);
        assert((uintptr_t) out_frame->linesize[2] % 2 == 0);

        int src_linesize = vc_get_linesize(width, RG48);
        for (int y = 0; y < height; ++y) {
                uint16_t *src = (uint16_t *)(void *) (in_data + y * src_linesize);
                uint16_t *dst_g = (uint16_t *)(void *) (out_frame->data[0] + out_frame->linesize[0] * y);
                uint16_t *dst_b = (uint16_t *)(void *) (out_frame->data[1] + out_frame->linesize[1] * y);
                uint16_t *dst_r = (uint16_t *)(void *) (out_frame->data[2] + out_frame->linesize[2] * y);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst_r++ = *src++ >> 4U;
                        *dst_g++ = *src++ >> 4U;
                        *dst_b++ = *src++ >> 4U;
                }
        }
}
#endif

//
// av_to_uv_convert conversions
//
static void nv12_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoUYVY((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0], vc_get_linesize(width, UYVY), 0, 0, 0);
        }
}

static void memcpy_data(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        UNUSED(width);
        for (int comp = 0; comp < AV_NUM_DATA_POINTERS; ++comp) {
                if (frame->data[comp] == NULL) {
                        break;
                }
                for (int y = 0; y < height; ++y) {
                        memcpy(dst_buffer + y * pitch, frame->data[comp] + y * frame->linesize[comp],
                                        frame->linesize[comp]);
                }
        }
}

static void rgb24_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoRGBA((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, RGBA), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void gbrp_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        assert((uintptr_t) dst_buffer % 4 == 0);

        for (int y = 0; y < height; ++y) {
                uint32_t *line = (uint32_t *)(void *) (dst_buffer + y * pitch);
                int src_idx = y * frame->linesize[0];

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *line++ = frame->data[2][src_idx] << rgb_shift[R] |
                                frame->data[0][src_idx] << rgb_shift[G] |
                                frame->data[1][src_idx] << rgb_shift[B];
                        src_idx += 1;
                }
        }
}

static inline void gbrpXXle_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_r >> (in_depth - 8U);
                        *dst++ = ((*src_r++ >> (in_depth - 10U)) & 0x3U) << 6U | *src_g >> (in_depth - 6U);
                        *dst++ = ((*src_g++ >> (in_depth - 10U)) & 0xFU) << 4U | *src_b >> (in_depth - 4U);
                        *dst++ = ((*src_b++ >> (in_depth - 10U)) & 0x3FU) << 2U;
                }
        }
}

static void gbrp10le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r10k(dst_buffer, frame, width, height, pitch, rgb_shift, 10U);
}

static void gbrp16le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r10k(dst_buffer, frame, width, height, pitch, rgb_shift, 16U);
}

static void yuv444pXXle_to_r10k(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
		unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t y = (y_scale * (*src_y++ - (1<<(depth-4))));
                        comp_type_t cr = *src_cr++ - (1<<(depth-1));
                        comp_type_t cb = *src_cb++ - (1<<(depth-1));

                        comp_type_t r = YCBCR_TO_R_709_SCALED(y, cb, cr) >> (COMP_BASE-10+depth);
                        comp_type_t g = YCBCR_TO_G_709_SCALED(y, cb, cr) >> (COMP_BASE-10+depth);
                        comp_type_t b = YCBCR_TO_B_709_SCALED(y, cb, cr) >> (COMP_BASE-10+depth);
                        // r g b is now on 10 bit scale

                        r = CLAMP_FULL(r, 10);
                        g = CLAMP_FULL(g, 10);
                        b = CLAMP_FULL(b, 10);

			*dst++ = r >> 2;
			*dst++ = (r & 0x3) << 6 | g >> 4;
			*dst++ = (g & 0xF) << 4 | b >> 6;
			*dst++ = (b & 0x3F) << 2;
                }
        }
}

static void yuv444p10le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r10k(10, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p12le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r10k(12, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p16le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r10k(16, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444pXXle_to_r12l(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
                        comp_type_t r[8];
                        comp_type_t g[8];
                        comp_type_t b[8];
                        OPTIMIZED_FOR (int j = 0; j < 8; ++j) {
                                comp_type_t y = (y_scale * (*src_y++ - (1<<(depth-4))));
                                comp_type_t cr = *src_cr++ - (1<<(depth-1));
                                comp_type_t cb = *src_cb++ - (1<<(depth-1));
                                comp_type_t rr = YCBCR_TO_R_709_SCALED(y, cb, cr) >> (COMP_BASE-12+depth);
                                comp_type_t gg = YCBCR_TO_G_709_SCALED(y, cb, cr) >> (COMP_BASE-12+depth);
                                comp_type_t bb = YCBCR_TO_B_709_SCALED(y, cb, cr) >> (COMP_BASE-12+depth);
                                r[j] = CLAMP_FULL(rr, 12);
                                g[j] = CLAMP_FULL(gg, 12);
                                b[j] = CLAMP_FULL(bb, 12);
                        }

                        dst[BYTE_SWAP(0)] = r[0] & 0xff;
                        dst[BYTE_SWAP(1)] = (g[0] & 0xf) << 4 | r[0] >> 8;
                        dst[BYTE_SWAP(2)] = g[0] >> 4;
                        dst[BYTE_SWAP(3)] = b[0] & 0xff;
                        dst[4 + BYTE_SWAP(0)] = (r[1] & 0xf) << 4 | b[0] >> 8;
                        dst[4 + BYTE_SWAP(1)] = r[1] >> 4;
                        dst[4 + BYTE_SWAP(2)] = g[1] & 0xff;
                        dst[4 + BYTE_SWAP(3)] = (b[1] & 0xf) << 4 | g[1] >> 8;
                        dst[8 + BYTE_SWAP(0)] = b[1] >> 4;
                        dst[8 + BYTE_SWAP(1)] = r[2] & 0xff;
                        dst[8 + BYTE_SWAP(2)] = (g[2] & 0xf) << 4 | r[2] >> 8;
                        dst[8 + BYTE_SWAP(3)] = g[2] >> 4;
                        dst[12 + BYTE_SWAP(0)] = b[2] & 0xff;
                        dst[12 + BYTE_SWAP(1)] = (r[3] & 0xf) << 4 | b[2] >> 8;
                        dst[12 + BYTE_SWAP(2)] = r[3] >> 4;
                        dst[12 + BYTE_SWAP(3)] = g[3] & 0xff;
                        dst[16 + BYTE_SWAP(0)] = (b[3] & 0xf) << 4 | g[3] >> 8;
                        dst[16 + BYTE_SWAP(1)] = b[3] >> 4;
                        dst[16 + BYTE_SWAP(2)] = r[4] & 0xff;
                        dst[16 + BYTE_SWAP(3)] = (g[4] & 0xf) << 4 | r[4] >> 8;
                        dst[20 + BYTE_SWAP(0)] = g[4] >> 4;
                        dst[20 + BYTE_SWAP(1)] = b[4] & 0xff;
                        dst[20 + BYTE_SWAP(2)] = (r[5] & 0xf) << 4 | b[4] >> 8;
                        dst[20 + BYTE_SWAP(3)] = r[5] >> 4;;
                        dst[24 + BYTE_SWAP(0)] = g[5] & 0xff;
                        dst[24 + BYTE_SWAP(1)] = (b[5] & 0xf) << 4 | g[5] >> 8;
                        dst[24 + BYTE_SWAP(2)] = b[5] >> 4;
                        dst[24 + BYTE_SWAP(3)] = r[6] & 0xff;
                        dst[28 + BYTE_SWAP(0)] = (g[6] & 0xf) << 4 | r[6] >> 8;
                        dst[28 + BYTE_SWAP(1)] = g[6] >> 4;
                        dst[28 + BYTE_SWAP(2)] = b[6] & 0xff;
                        dst[28 + BYTE_SWAP(3)] = (r[7] & 0xf) << 4 | b[6] >> 8;
                        dst[32 + BYTE_SWAP(0)] = r[7] >> 4;
                        dst[32 + BYTE_SWAP(1)] = g[7] & 0xff;
                        dst[32 + BYTE_SWAP(2)] = (b[7] & 0xf) << 4 | g[7] >> 8;
                        dst[32 + BYTE_SWAP(3)] = b[7] >> 4;
                        dst += 36;
                }
        }
}

static void yuv444p10le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r12l(10, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p12le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r12l(12, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p16le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_r12l(16, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static inline void yuv444pXXle_to_rg48(int depth, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        assert((uintptr_t) dst_buffer % 2 == 0);
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                uint16_t *dst = (uint16_t *)(void *) (dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t y = (y_scale * (*src_y++ - (1<<(depth-4))));
                        comp_type_t cr = *src_cr++ - (1<<(depth-1));
                        comp_type_t cb = *src_cb++ - (1<<(depth-1));

                        comp_type_t r = YCBCR_TO_R_709_SCALED(y, cb, cr) >> (COMP_BASE-16+depth);
                        comp_type_t g = YCBCR_TO_G_709_SCALED(y, cb, cr) >> (COMP_BASE-16+depth);
                        comp_type_t b = YCBCR_TO_B_709_SCALED(y, cb, cr) >> (COMP_BASE-16+depth);
                        // r g b is now on 16 bit scale

                        *dst++ = CLAMP_FULL(r, 16);
                        *dst++ = CLAMP_FULL(g, 16);
                        *dst++ = CLAMP_FULL(b, 16);
                }
        }
}

static void yuv444p10le_to_rg48(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_rg48(10, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p12le_to_rg48(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_rg48(12, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static void yuv444p16le_to_rg48(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444pXXle_to_rg48(16, dst_buffer, frame, width, height, pitch, rgb_shift);
}

static inline void gbrpXXle_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

#undef S
#define S(x) ((x) >> (in_depth - 12))

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; x += 8) {
                        dst[BYTE_SWAP(0)] = S(*src_r) & 0xff;
                        dst[BYTE_SWAP(1)] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[BYTE_SWAP(2)] = S(*src_g++) >> 4;
                        dst[BYTE_SWAP(3)] = S(*src_b) & 0xff;
                        dst[4 + BYTE_SWAP(0)] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[4 + BYTE_SWAP(1)] = S(*src_r++) >> 4;
                        dst[4 + BYTE_SWAP(2)] = S(*src_g) & 0xff;
                        dst[4 + BYTE_SWAP(3)] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[8 + BYTE_SWAP(0)] = S(*src_b++) >> 4;
                        dst[8 + BYTE_SWAP(1)] = S(*src_r) & 0xff;
                        dst[8 + BYTE_SWAP(2)] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[8 + BYTE_SWAP(3)] = S(*src_g++) >> 4;
                        dst[12 + BYTE_SWAP(0)] = S(*src_b) & 0xff;
                        dst[12 + BYTE_SWAP(1)] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[12 + BYTE_SWAP(2)] = S(*src_r++) >> 4;
                        dst[12 + BYTE_SWAP(3)] = S(*src_g) & 0xff;
                        dst[16 + BYTE_SWAP(0)] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[16 + BYTE_SWAP(1)] = S(*src_b++) >> 4;
                        dst[16 + BYTE_SWAP(2)] = S(*src_r) & 0xff;
                        dst[16 + BYTE_SWAP(3)] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[20 + BYTE_SWAP(0)] = S(*src_g++) >> 4;
                        dst[20 + BYTE_SWAP(1)] = S(*src_b) & 0xff;
                        dst[20 + BYTE_SWAP(2)] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[20 + BYTE_SWAP(3)] = S(*src_r++) >> 4;;
                        dst[24 + BYTE_SWAP(0)] = S(*src_g) & 0xff;
                        dst[24 + BYTE_SWAP(1)] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[24 + BYTE_SWAP(2)] = S(*src_b++) >> 4;
                        dst[24 + BYTE_SWAP(3)] = S(*src_r) & 0xff;
                        dst[28 + BYTE_SWAP(0)] = (S(*src_g) & 0xf) << 4 | S(*src_r++) >> 8;
                        dst[28 + BYTE_SWAP(1)] = S(*src_g++) >> 4;
                        dst[28 + BYTE_SWAP(2)] = S(*src_b) & 0xff;
                        dst[28 + BYTE_SWAP(3)] = (S(*src_r) & 0xf) << 4 | S(*src_b++) >> 8;
                        dst[32 + BYTE_SWAP(0)] = S(*src_r++) >> 4;
                        dst[32 + BYTE_SWAP(1)] = S(*src_g) & 0xff;
                        dst[32 + BYTE_SWAP(2)] = (S(*src_b) & 0xf) << 4 | S(*src_g++) >> 8;
                        dst[32 + BYTE_SWAP(3)] = S(*src_b++) >> 4;
                        dst += 36;
                }
        }
}

static inline void gbrpXXle_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
{
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        UNUSED(rgb_shift);
        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                unsigned char *dst = (unsigned char *) dst_buffer + y * pitch;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = *src_r++ >> (in_depth - 8U);
                        *dst++ = *src_g++ >> (in_depth - 8U);
                        *dst++ = *src_b++ >> (in_depth - 8U);
                }
        }
}

static inline void gbrpXXle_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, unsigned int in_depth)
{
        assert((uintptr_t) dst_buffer % 4 == 0);
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        for (int y = 0; y < height; ++y) {
                uint16_t *src_g = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * y);
                uint16_t *src_b = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                uint16_t *src_r = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *) (dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        *dst++ = (*src_r++ >> (in_depth - 8U)) << rgb_shift[0] | (*src_g++ >> (in_depth - 8U)) << rgb_shift[1] |
                                (*src_b++ >> (in_depth - 8U)) << rgb_shift[2];
                }
        }
}

static void gbrp10le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgb(dst_buffer, frame, width, height, pitch, rgb_shift, 10);
}

static void gbrp10le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgba(dst_buffer, frame, width, height, pitch, rgb_shift, 10);
}

#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
static void gbrp12le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r12l(dst_buffer, frame, width, height, pitch, rgb_shift, 12U);
}

static void gbrp12le_to_r10k(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r10k(dst_buffer, frame, width, height, pitch, rgb_shift, 12U);
}

static void gbrp12le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgb(dst_buffer, frame, width, height, pitch, rgb_shift, 12U);
}

static void gbrp12le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgba(dst_buffer, frame, width, height, pitch, rgb_shift, 12U);
}
#endif

static void gbrp16le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_r12l(dst_buffer, frame, width, height, pitch, rgb_shift, 16U);
}

static void gbrp16le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgb(dst_buffer, frame, width, height, pitch, rgb_shift, 16U);
}

static void gbrp16le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        gbrpXXle_to_rgba(dst_buffer, frame, width, height, pitch, rgb_shift, 16U);
}

static void rgb48le_to_rgba(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toRGBA((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, RGBA), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void rgb48le_to_r12l(char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRG48toR12L((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, R12L), rgb_shift[0], rgb_shift[1], rgb_shift[2]);
        }
}

static void yuv420p_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                        y1 = _mm_lddqu_si128((__m128i const*)(const void *) src_y1);
                        y2 = _mm_lddqu_si128((__m128i const*)(const void *) src_y2);
                        src_y1 += 16;
                        src_y2 += 16;

                        out1l = _mm_unpacklo_epi8(zero, y1);
                        out1h = _mm_unpackhi_epi8(zero, y1);
                        out2l = _mm_unpacklo_epi8(zero, y2);
                        out2h = _mm_unpackhi_epi8(zero, y2);

                        u1 = _mm_lddqu_si128((__m128i const*)(const void *) src_cb);
                        v1 = _mm_lddqu_si128((__m128i const*)(const void *) src_cr);
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

                        _mm_storeu_si128((__m128i *)(void *) dst1, out1l);
                        dst1 += 16;
                        _mm_storeu_si128((__m128i *)(void *) dst1, out1h);
                        dst1 += 16;
                        _mm_storeu_si128((__m128i *)(void *) dst2, out2l);
                        dst2 += 16;
                        _mm_storeu_si128((__m128i *)(void *) dst2, out2h);
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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

static void yuv444p16le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y + 1;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y + 1;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y + 1;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        *dst++ = (*src_cb + *(src_cb + 2)) / 2;
                        src_cb += 4;
                        *dst++ = *src_y;
                        src_y += 2;
                        *dst++ = (*src_cr + *(src_cr + 2)) / 2;
                        src_cr += 4;
                        *dst++ = *src_y;
                        src_y += 2;
                }
        }
}

static void yuv444p_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
{
        assert((uintptr_t) dst_buffer % 4 == 0);

        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cbcr = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cb = *src_cbcr++ - 128;
                        comp_type_t cr = *src_cbcr++ - 128;
                        comp_type_t y = *src_y++ * y_scale;
                        comp_type_t r = r_cr * cr;
                        comp_type_t g = g_cb * cb + g_cr * cr;
                        comp_type_t b = b_cb * cb;
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = FORMAT_RGBA((r + y) >> COMP_BASE, (g + y) >> COMP_BASE, (b + y) >> COMP_BASE, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL((r + y) >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL((g + y) >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL((b + y) >> COMP_BASE, 8);
                        }

                        y = *src_y++ * y_scale;
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = FORMAT_RGBA((r + y) >> COMP_BASE, (g + y) >> COMP_BASE, (b + y) >> COMP_BASE, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL((r + y) >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL((g + y) >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL((b + y) >> COMP_BASE, 8);
                        }
                }
        }
}

static void nv12_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        nv12_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void nv12_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        nv12_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

/**
 * Changes pixel format from planar 8-bit YUV to packed RGB/A.
 * Color space is assumed ITU-T Rec. 709 limited range.
 */
static inline void yuv8p_to_rgb(int subsampling, char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
{
        for(int y = 0; y < height / 2; ++y) {
                unsigned char *src_y1 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                unsigned char *src_y2 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1);
                unsigned char *dst1 = (unsigned char *) dst_buffer + pitch * (y * 2);
                unsigned char *dst2 = (unsigned char *) dst_buffer + pitch * (y * 2 + 1);

                unsigned char *src_cb1;
                unsigned char *src_cr1;
                unsigned char *src_cb2;
                unsigned char *src_cr2;
                if (subsampling == 420) {
                        src_cb1 = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                        src_cr1 = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                } else {
                        src_cb1 = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y * 2);
                        src_cr1 = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * (y * 2);
                        src_cb2 = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y * 2 + 1);
                        src_cr2 = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * (y * 2 + 1);
                }

#define WRITE_RES_YUV8P_TO_RGB(DST) if (rgba) {\
                                *((uint32_t *)(void *) DST) = FORMAT_RGBA((r + y) >> COMP_BASE, (g + y) >> COMP_BASE, (b + y) >> COMP_BASE, 8);\
                                DST += 4;\
                        } else {\
                                *DST++ = CLAMP_FULL((r + y) >> COMP_BASE, 8);\
                                *DST++ = CLAMP_FULL((g + y) >> COMP_BASE, 8);\
                                *DST++ = CLAMP_FULL((b + y) >> COMP_BASE, 8);\
                        }\

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cb = *src_cb1++ - 128;
                        comp_type_t cr = *src_cr1++ - 128;
                        comp_type_t y = *src_y1++ * y_scale;
                        comp_type_t r = r_cr * cr;
                        comp_type_t g = g_cb * cb + g_cr * cr;
                        comp_type_t b = b_cb * cb;
                        WRITE_RES_YUV8P_TO_RGB(dst1)

                        y = *src_y1++ * y_scale;
                        WRITE_RES_YUV8P_TO_RGB(dst1)

                        if (subsampling == 422) {
                                cb = *src_cb2++ - 128;
                                cr = *src_cr2++ - 128;
                                r = r_cr * cr;
                                g = g_cb * cb + g_cr * cr;
                                b = b_cb * cb;
                        }
                        y = *src_y2++ * y_scale;
                        WRITE_RES_YUV8P_TO_RGB(dst2)

                        y = *src_y2++ * y_scale;
                        WRITE_RES_YUV8P_TO_RGB(dst2)
                }
        }
}

static void yuv420p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv8p_to_rgb(420, dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv420p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv8p_to_rgb(420, dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

static void yuv422p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv8p_to_rgb(422, dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv422p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv8p_to_rgb(422, dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}


/**
 * Changes pixel format from planar YUV 444 to packed RGB/A.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static inline void yuv444p_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
{
        assert((uintptr_t) dst_buffer % 4 == 0);

        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ << COMP_BASE;
                        int r = r_cr * cr;
                        int g = g_cb * cb + g_cr * cr;
                        int b = b_cb * cb;
                        if (rgba) {
                                *((uint32_t *)(void *) dst) = (MIN(MAX((r + y) >> COMP_BASE, 1), 254) << rgb_shift[R] | MIN(MAX((g + y) >> COMP_BASE, 1), 254) << rgb_shift[G] | MIN(MAX((b + y) >> COMP_BASE, 1), 254) << rgb_shift[B]);
                                dst += 4;
                        } else {
                                *dst++ = MIN(MAX((r + y) >> COMP_BASE, 1), 254);
                                *dst++ = MIN(MAX((g + y) >> COMP_BASE, 1), 254);
                                *dst++ = MIN(MAX((b + y) >> COMP_BASE, 1), 254);
                        }
                }
        }
}

static void yuv444p_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static void yuv444p_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444p_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

static void yuv420p10le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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

static void yuv444p16le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(rgb_shift);
        for(int y = 0; y < height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width / 6; ++x) {
                        uint32_t w0_0, w0_1, w0_2, w0_3;

                        w0_0 = ((src_cb[0] >> 6U) + (src_cb[1] >> 6U)) / 2;
                        w0_0 = w0_0 | (*src_y++ >> 6U) << 10U;
                        w0_0 = w0_0 | ((src_cr[0] >> 6U) + (src_cr[1] >> 6U)) / 2 << 20U;
                        src_cb += 2;
                        src_cr += 2;

                        w0_1 = *src_y++;
                        w0_1 = w0_1 | ((src_cb[0] >> 6U) + (src_cb[1] >> 6U)) / 2 << 10U;
                        w0_1 = w0_1 | (*src_y++ >> 6U) << 20U;
                        src_cb += 2;

                        w0_2 = ((src_cr[0] >> 6U) + (src_cr[1] >> 6U)) / 2;
                        w0_2 = w0_2 | (*src_y++ >> 6U) << 10U;
                        w0_2 = w0_2 | ((src_cb[0] >> 6U) + (src_cb[1] >> 6U)) / 2 << 20U;
                        src_cr += 2;
                        src_cb += 2;

                        w0_3 = *src_y++;
                        w0_3 = w0_3 | ((src_cr[0] >> 6U) + (src_cr[1] >> 6U)) / 2 << 10U;
                        w0_3 = w0_3 | ((*src_y++ >> 6U)) << 20U;
                        src_cr += 2;

                        *dst++ = w0_0;
                        *dst++ = w0_1;
                        *dst++ = w0_2;
                        *dst++ = w0_3;
                }
        }
}

static void yuv420p10le_to_uyvy(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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

static inline void yuvp10le_to_rgb(int subsampling, char * __restrict dst_buffer, AVFrame * __restrict frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, int out_bit_depth)
{
        assert((uintptr_t) dst_buffer % 4 == 0);
        assert((uintptr_t) frame->linesize[0] % 2 == 0);
        assert((uintptr_t) frame->linesize[1] % 2 == 0);
        assert((uintptr_t) frame->linesize[2] % 2 == 0);

        assert(subsampling == 422 || subsampling == 420);
        assert(out_bit_depth == 24 || out_bit_depth == 30 || out_bit_depth == 32);
        const int bpp = out_bit_depth == 30 ? 10 : 8;

        for (int y = 0; y < height / 2; ++y) {
                uint16_t * __restrict src_y1 = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * 2 * y);
                uint16_t * __restrict src_y2 = (uint16_t *)(void *) (frame->data[0] + frame->linesize[0] * (2 * y + 1));
                uint16_t * __restrict src_cb1;
                uint16_t * __restrict src_cr1;
                uint16_t * __restrict src_cb2;
                uint16_t * __restrict src_cr2;
                if (subsampling == 420) {
                        src_cb1 = src_cb2 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * y);
                        src_cr1 = src_cr2 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * y);
                } else {
                        src_cb1 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * (2 * y));
                        src_cb2 = (uint16_t *)(void *) (frame->data[1] + frame->linesize[1] * (2 * y + 1));
                        src_cr1 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * (2 * y));
                        src_cr2 = (uint16_t *)(void *) (frame->data[2] + frame->linesize[2] * (2 * y + 1));
                }
                unsigned char *dst1 = (unsigned char *) dst_buffer + (2 * y) * pitch;
                unsigned char *dst2 = (unsigned char *) dst_buffer + (2 * y + 1) * pitch;

                OPTIMIZED_FOR (int x = 0; x < width / 2; ++x) {
                        comp_type_t cr = *src_cr1++ - (1<<9);
                        comp_type_t cb = *src_cb1++ - (1<<9);
                        comp_type_t rr = YCBCR_TO_R_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        comp_type_t gg = YCBCR_TO_G_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        comp_type_t bb = YCBCR_TO_B_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));

#                       define WRITE_RES_YUV10P_TO_RGB(Y, DST) {\
                                comp_type_t r = Y + rr;\
                                comp_type_t g = Y + gg;\
                                comp_type_t b = Y + bb;\
                                r = CLAMP_FULL(r, bpp);\
                                g = CLAMP_FULL(g, bpp);\
                                b = CLAMP_FULL(b, bpp);\
                                if (out_bit_depth == 32) {\
                                        *((uint32_t *)(void *) DST) = (r << rgb_shift[R] | g << rgb_shift[G] | b << rgb_shift[B]);\
                                        DST += 4;\
                                } else if (out_bit_depth == 24) {\
                                        *DST++ = r;\
                                        *DST++ = g;\
                                        *DST++ = b;\
                                } else {\
                                        *((uint32_t *)(void *) DST) = htonl(r << 22U | g << 12U | b << 2U);\
                                        DST += 4;\
                                }\
                        }

                        comp_type_t y1 = (y_scale * (*src_y1++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y1, dst1)

                        comp_type_t y11 = (y_scale * (*src_y1++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y11, dst1)

                        if (subsampling == 422) {
                                cr = *src_cr2++ - (1<<9);
                                cb = *src_cb2++ - (1<<9);
                                rr = YCBCR_TO_R_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                                gg = YCBCR_TO_G_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                                bb = YCBCR_TO_B_709_SCALED(0, cb, cr) >> (COMP_BASE + (10 - bpp));
                        }

                        comp_type_t y2 = (y_scale * (*src_y2++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y2, dst2)

                        comp_type_t y22 = (y_scale * (*src_y2++ - (1<<6))) >> (COMP_BASE + (10 - bpp));
                        WRITE_RES_YUV10P_TO_RGB(y22, dst2)
                }
        }
}

#define MAKE_YUV_TO_RGB_FUNCTION_NAME(subs, out_bit_depth) yuv ## subs ## p10le_to_rgb ## out_bit_depth

#define MAKE_YUV_TO_RGB_FUNCTION(subs, out_bit_depth) static void MAKE_YUV_TO_RGB_FUNCTION_NAME(subs, out_bit_depth)(char * __restrict dst_buffer, AVFrame * __restrict in_frame,\
                int width, int height, int pitch, const int * __restrict rgb_shift) {\
        yuvp10le_to_rgb(subs, dst_buffer, in_frame, width, height, pitch, rgb_shift, out_bit_depth);\
}

MAKE_YUV_TO_RGB_FUNCTION(420, 24)
MAKE_YUV_TO_RGB_FUNCTION(420, 30)
MAKE_YUV_TO_RGB_FUNCTION(420, 32)
MAKE_YUV_TO_RGB_FUNCTION(422, 24)
MAKE_YUV_TO_RGB_FUNCTION(422, 30)
MAKE_YUV_TO_RGB_FUNCTION(422, 32)

static inline void yuv444p10le_to_rgb(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift, bool rgba)
{
        for (int y = 0; y < height; y++) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                OPTIMIZED_FOR (int x = 0; x < width; ++x) {
                        comp_type_t cb = (*src_cb++ >> 2) - 128;
                        comp_type_t cr = (*src_cr++ >> 2) - 128;
                        comp_type_t y = (*src_y++ >> 2) * y_scale;
                        comp_type_t r = r_cr * cr;
                        comp_type_t g = g_cb * cb + g_cr * cr;
                        comp_type_t b = b_cb * cb;
                        if (rgba) {
                                *(uint32_t *)(void *) dst = FORMAT_RGBA((r + y) >> COMP_BASE, (g + y) >> COMP_BASE, (b + y) >> COMP_BASE, 8);
                                dst += 4;
                        } else {
                                *dst++ = CLAMP_FULL((r + y) >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL((g + y) >> COMP_BASE, 8);
                                *dst++ = CLAMP_FULL((b + y) >> COMP_BASE, 8);
                        }
                }
        }
}

static inline void yuv444p10le_to_rgb24(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444p10le_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, false);
}

static inline void yuv444p10le_to_rgb32(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        yuv444p10le_to_rgb(dst_buffer, in_frame, width, height, pitch, rgb_shift, true);
}

static void p010le_to_v210(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
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
                int width, int height, int pitch, const int * __restrict rgb_shift)
{
        UNUSED(width);
        UNUSED(height);
        UNUSED(pitch);
        UNUSED(rgb_shift);

        struct video_frame_callbacks *callbacks = in_frame->opaque;

        hw_vdpau_frame *out = (hw_vdpau_frame *)(void *) dst_buffer;

        hw_vdpau_frame_init(out);

        hw_vdpau_frame_from_avframe(out, in_frame);

        callbacks->recycle = hw_vdpau_recycle_callback; 
        callbacks->copy = hw_vdpau_copy_callback; 
}
#endif

#ifdef HWACC_RPI4
static void av_rpi4_8_to_ug(char * __restrict dst_buffer, AVFrame * __restrict in_frame,
                int width, int height, int pitch, int * __restrict rgb_shift)
{
        UNUSED(width);
        UNUSED(height);
        UNUSED(pitch);
        UNUSED(rgb_shift);

        av_frame_wrapper *out = (av_frame_wrapper *)(void *) dst_buffer;
        av_frame_ref(out->av_frame, in_frame);
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
                { v210, AV_PIX_FMT_YUV420P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_yuv420p10le },
                { v210, AV_PIX_FMT_YUV422P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_yuv422p10le },
                { v210, AV_PIX_FMT_YUV444P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_yuv444p10le },
                { v210, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_yuv444p16le },
                { R10k, AV_PIX_FMT_YUV444P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv444p10le },
                { R10k, AV_PIX_FMT_YUV444P12LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv444p12le },
                { R10k, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv444p16le },
                { R12L, AV_PIX_FMT_YUV444P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r12l_to_yuv444p10le },
                { R12L, AV_PIX_FMT_YUV444P12LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r12l_to_yuv444p12le },
                { R12L, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r12l_to_yuv444p16le },
                { RG48, AV_PIX_FMT_YUV444P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, rg48_to_yuv444p10le },
                { RG48, AV_PIX_FMT_YUV444P12LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, rg48_to_yuv444p12le },
                { RG48, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, rg48_to_yuv444p16le },
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(55, 15, 100) // FFMPEG commit c2869b4640f
                { v210, AV_PIX_FMT_P010LE,      AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, v210_to_p010le },
#endif
                { UYVY, AV_PIX_FMT_YUV422P,     AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv422p },
                { UYVY, AV_PIX_FMT_YUVJ422P,    AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv422p },
                { UYVY, AV_PIX_FMT_YUV420P,     AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv420p },
                { UYVY, AV_PIX_FMT_YUVJ420P,    AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv420p },
                { UYVY, AV_PIX_FMT_NV12,        AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_nv12 },
                { UYVY, AV_PIX_FMT_YUV444P,     AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv444p },
                { UYVY, AV_PIX_FMT_YUVJ444P,    AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, uyvy_to_yuv444p },
                { Y216, AV_PIX_FMT_YUV422P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, y216_to_yuv422p10le },
                { Y216, AV_PIX_FMT_YUV422P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, y216_to_yuv422p16le },
                { Y216, AV_PIX_FMT_YUV444P16LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, y216_to_yuv444p16le },
                { RGB, AV_PIX_FMT_BGR0,         AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, rgb_to_bgr0 },
                { RGB, AV_PIX_FMT_GBRP,         AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, rgb_to_gbrp },
                { RGBA, AV_PIX_FMT_GBRP,        AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, rgba_to_gbrp },
                { R10k, AV_PIX_FMT_BGR0,        AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r10k_to_bgr0 },
                { R10k, AV_PIX_FMT_GBRP10LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r10k_to_gbrp10le },
                { R10k, AV_PIX_FMT_GBRP16LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r10k_to_gbrp16le },
                { R10k, AV_PIX_FMT_YUV422P10LE, AVCOL_SPC_BT709, AVCOL_RANGE_MPEG, r10k_to_yuv422p10le },
#ifdef HAVE_12_AND_14_PLANAR_COLORSPACES
                { R12L, AV_PIX_FMT_GBRP12LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r12l_to_gbrp12le },
                { R12L, AV_PIX_FMT_GBRP16LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, r12l_to_gbrp16le },
                { RG48, AV_PIX_FMT_GBRP12LE,    AVCOL_SPC_RGB,   AVCOL_RANGE_JPEG, rg48_to_gbrp12le },
#endif
                { 0, 0, 0, 0, 0 }
        };
        return uv_to_av_conversions;
}

pixfmt_callback_t get_uv_to_av_conversion(codec_t uv_codec, int av_codec) {
        for (const struct uv_to_av_conversion *conversions = get_uv_to_av_conversions();
                        conversions->func != 0; conversions++) {
                if (conversions->dst == av_codec &&
                                conversions->src == uv_codec) {
                        return conversions->func;
                }
        }

        return NULL;
}

void get_av_pixfmt_details(codec_t uv_codec, int av_codec, enum AVColorSpace *colorspace, enum AVColorRange *color_range)
{
        for (const struct uv_to_av_conversion *conversions = get_uv_to_av_conversions();
                        conversions->func != 0; conversions++) {
                if (conversions->dst == av_codec &&
                                conversions->src == uv_codec) {
                        *colorspace = conversions->colorspace;
                        *color_range = conversions->color_range;
                        return;
                }
        }
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
                {AV_PIX_FMT_YUV420P10LE, R10k, yuv420p10le_to_rgb30, false},
                {AV_PIX_FMT_YUV422P10LE, v210, yuv422p10le_to_v210, true},
                {AV_PIX_FMT_YUV422P10LE, UYVY, yuv422p10le_to_uyvy, false},
                {AV_PIX_FMT_YUV422P10LE, RGB, yuv422p10le_to_rgb24, false},
                {AV_PIX_FMT_YUV422P10LE, RGBA, yuv422p10le_to_rgb32, false},
                {AV_PIX_FMT_YUV422P10LE, R10k, yuv422p10le_to_rgb30, false},
                {AV_PIX_FMT_YUV444P10LE, v210, yuv444p10le_to_v210, true},
                {AV_PIX_FMT_YUV444P10LE, UYVY, yuv444p10le_to_uyvy, false},
                {AV_PIX_FMT_YUV444P10LE, R10k, yuv444p10le_to_r10k, false},
                {AV_PIX_FMT_YUV444P10LE, RGB, yuv444p10le_to_rgb24, false},
                {AV_PIX_FMT_YUV444P10LE, RGBA, yuv444p10le_to_rgb32, false},
                {AV_PIX_FMT_YUV444P10LE, R12L, yuv444p10le_to_r12l, false},
                {AV_PIX_FMT_YUV444P10LE, RG48, yuv444p10le_to_rg48, false},
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
                // 8-bit YUV - this should be supposedly full range JPEG but lavd decoder doesn't honor
                // GPUJPEG's SPIFF header indicating YUV BT.709 limited range. The YUVJ pixel formats
                // are detected only for GPUJPEG generated JPEGs.
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
                // 12-bit YUV
                {AV_PIX_FMT_YUV444P12LE, R10k, yuv444p12le_to_r10k, false},
                {AV_PIX_FMT_YUV444P12LE, R12L, yuv444p12le_to_r12l, false},
                {AV_PIX_FMT_YUV444P12LE, RG48, yuv444p12le_to_rg48, false},
                // 16-bit YUV
                {AV_PIX_FMT_YUV444P16LE, R10k, yuv444p16le_to_r10k, false},
                {AV_PIX_FMT_YUV444P16LE, R12L, yuv444p16le_to_r12l, false},
                {AV_PIX_FMT_YUV444P16LE, RG48, yuv444p16le_to_rg48, false},
                {AV_PIX_FMT_YUV444P16LE, UYVY, yuv444p16le_to_uyvy, false},
                {AV_PIX_FMT_YUV444P16LE, v210, yuv444p16le_to_v210, false},
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
                {AV_PIX_FMT_GBRP12LE, R10k, gbrp12le_to_r10k, true},
                {AV_PIX_FMT_GBRP12LE, RGB, gbrp12le_to_rgb, false},
                {AV_PIX_FMT_GBRP12LE, RGBA, gbrp12le_to_rgba, false},
#endif
                {AV_PIX_FMT_GBRP16LE, R12L, gbrp16le_to_r12l, true},
                {AV_PIX_FMT_GBRP16LE, R10k, gbrp16le_to_r10k, true},
                {AV_PIX_FMT_GBRP12LE, RGB, gbrp16le_to_rgb, false},
                {AV_PIX_FMT_GBRP12LE, RGBA, gbrp16le_to_rgba, false},
                {AV_PIX_FMT_RGB48LE, RG48, memcpy_data, true},
                {AV_PIX_FMT_RGB48LE, R12L, rgb48le_to_r12l, false},
                {AV_PIX_FMT_RGB48LE, RGBA, rgb48le_to_rgba, false},
#ifdef HWACC_VDPAU
                // HW acceleration
                {AV_PIX_FMT_VDPAU, HW_VDPAU, av_vdpau_to_ug_vdpau, false},
#endif
#ifdef HWACC_RPI4
                {AV_PIX_FMT_RPI4_8, RPI4_8, av_rpi4_8_to_ug, false},
#endif
                {0, 0, 0, 0}
        };
        return av_to_uv_conversions;
}

av_to_uv_convert_p get_av_to_uv_conversion(int av_codec, codec_t uv_codec) {
        for (const struct av_to_uv_conversion *conversions = get_av_to_uv_conversions();
                        conversions->convert != 0; conversions++) {
                if (conversions->av_codec == av_codec &&
                                conversions->uv_codec == uv_codec) {
                        return conversions->convert;
                }
        }

        return NULL;
}


#ifdef HAVE_SWSCALE
/**
* Simplified version of this: https://cpp.hotexamples.com/examples/-/-/av_opt_set_int/cpp-av_opt_set_int-function-examples.html
*
* @todo
* Use more fine-grained color-space characteristics that swscale support (eg. used in the original code).
*/
struct SwsContext *getSwsContext(unsigned int SrcW, unsigned int SrcH, enum AVPixelFormat SrcFormat, unsigned int DstW, unsigned int DstH, enum AVPixelFormat DstFormat, int64_t Flags) {
    struct SwsContext *Context = sws_alloc_context();
    if (!Context) {
            return 0;
    }

    const struct AVPixFmtDescriptor *SrcFormatDesc = av_pix_fmt_desc_get(SrcFormat);
    const struct AVPixFmtDescriptor *DstFormatDesc = av_pix_fmt_desc_get(DstFormat);

    // 0 = limited range, 1 = full range
    int SrcRange = SrcFormatDesc != NULL && (SrcFormatDesc->flags & AV_PIX_FMT_FLAG_RGB) != 0 ? 1 : 0;
    int DstRange = DstFormatDesc != NULL && (DstFormatDesc->flags & AV_PIX_FMT_FLAG_RGB) != 0 ? 1 : 0;

    av_opt_set_int(Context, "sws_flags",  Flags, 0);
    av_opt_set_int(Context, "srcw",       SrcW, 0);
    av_opt_set_int(Context, "srch",       SrcH, 0);
    av_opt_set_int(Context, "dstw",       DstW, 0);
    av_opt_set_int(Context, "dsth",       DstH, 0);
    av_opt_set_int(Context, "src_range",  SrcRange, 0);
    av_opt_set_int(Context, "dst_range",  DstRange, 0);
    av_opt_set_int(Context, "src_format", SrcFormat, 0);
    av_opt_set_int(Context, "dst_format", DstFormat, 0);

    if (sws_init_context(Context, 0, 0) < 0) {
        sws_freeContext(Context);
        return 0;
    }

    return Context;
}
#endif // defined HAVE_SWSCALE

/**
 * Serializes (prints) AVFrame f to output file out
 *
 * @param f   actual (AVFrame *) cast to (void *)
 * @param out file stream to be written to
 */
void serialize_avframe(const void *f, FILE *out) {
        const AVFrame *frame = f;
        const AVPixFmtDescriptor *fmt_desc = av_pix_fmt_desc_get(frame->format);
        for (int comp = 0; comp < AV_NUM_DATA_POINTERS; ++comp) {
                if (frame->data[comp] == NULL) {
                        break;
                }
                for (int y = 0; y < frame->height >> (comp == 0 ? 0 : fmt_desc->log2_chroma_h); ++y) {
                        if (fwrite(frame->data[comp] + y * frame->linesize[comp], frame->linesize[comp], 1, out) != 1) {
                                log_msg(LOG_LEVEL_ERROR, "%s fwrite error\n", __func__);
                        }
                }
        }
}

#pragma GCC diagnostic pop

/* vi: set expandtab sw=8: */
