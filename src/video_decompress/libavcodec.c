/**
 * @file   video_decompress/libavcodec.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2017 CESNET, z. s. p. o.
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

#include "debug.h"
#include "host.h"
#include "libavcodec_common.h"
#include "lib_common.h"
#include "tv.h"
#include "utils/resource_manager.h"
#include "video.h"
#include "video_decompress.h"

#ifdef USE_HWACC
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vdpau.h>
#include <libavutil/hwcontext_vaapi.h>
#include <libavcodec/vdpau.h>
#include <libavcodec/vaapi.h>
#include "hwaccel.h"
#define DEFAULT_SURFACES 20
#endif

#ifdef __cplusplus
#include <algorithm>
using std::max;
using std::min;
#else
#undef max
#undef min
#define max(a, b)      (((a) > (b))? (a): (b))
#define min(a, b)      (((a) < (b))? (a): (b))
#endif

#define MOD_NAME "[lavd] "

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

#ifdef USE_HWACC
struct hw_accel_state {
        enum {
                HWACCEL_NONE,
                HWACCEL_VDPAU,
                HWACCEL_VAAPI
        } type;

        bool copy;
        AVFrame *tmp_frame;

        void (*uninit)(struct hw_accel_state*);

        void *ctx; //Type depends on hwaccel type
};
#endif

struct state_libavcodec_decompress {
        pthread_mutex_t *global_lavcd_lock;
        AVCodecContext  *codec_ctx;
        AVFrame         *frame;
        AVPacket         pkt;

        int              width, height;
        int              pitch;
        int              rshift, gshift, bshift;
        int              max_compressed_len;
        codec_t          in_codec;
        codec_t          out_codec;

        unsigned         last_frame_seq:22; // This gives last sucessfully decoded frame seq number. It is the buffer number from the packet format header, uses 22 bits.
        bool             last_frame_seq_initialized;

        struct video_desc saved_desc;
        unsigned int     broken_h264_mt_decoding_workaroud_warning_displayed;
        bool             broken_h264_mt_decoding_workaroud_active;

#ifdef USE_HWACC
        struct hw_accel_state hwaccel;
#endif
};

static int change_pixfmt(AVFrame *frame, unsigned char *dst, int av_codec,
                codec_t out_codec, int width, int height, int pitch);
static void error_callback(void *, int, const char *, va_list);
static enum AVPixelFormat get_format_callback(struct AVCodecContext *s, const enum AVPixelFormat *fmt);

static bool broken_h264_mt_decoding = false;

#ifdef USE_HWACC
static void hwaccel_state_init(struct hw_accel_state *hwaccel){
        hwaccel->type = HWACCEL_NONE;
        hwaccel->copy = false;
        hwaccel->uninit = NULL;
        hwaccel->tmp_frame = NULL;
        hwaccel->uninit = NULL;
        hwaccel->ctx = NULL;
}

static void hwaccel_state_reset(struct hw_accel_state *hwaccel){
        if(hwaccel->ctx){
                hwaccel->uninit(hwaccel);
        }

        if(hwaccel->tmp_frame){
                av_frame_free(&hwaccel->tmp_frame);
        }

        hwaccel_state_init(hwaccel);
}
#endif

static void deconfigure(struct state_libavcodec_decompress *s)
{
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
        if (s->codec_ctx) {
                int ret;
                ret = avcodec_send_packet(s->codec_ctx, NULL);
                if (ret != 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                        ret);
                }
                do {
                        ret = avcodec_receive_frame(s->codec_ctx, s->frame);
                        if (ret != 0 && ret != AVERROR_EOF) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                                ret);
                                break;
                        }

                } while (ret != AVERROR_EOF);
        }
#endif
        if(s->codec_ctx) {
                pthread_mutex_lock(s->global_lavcd_lock);
                avcodec_close(s->codec_ctx);
                avcodec_free_context(&s->codec_ctx);
                pthread_mutex_unlock(s->global_lavcd_lock);
        }
        av_free(s->frame);
        s->frame = NULL;
        av_packet_unref(&s->pkt);

#ifdef USE_HWACC
        hwaccel_state_reset(&s->hwaccel);
#endif
}

static void set_codec_context_params(struct state_libavcodec_decompress *s)
{
        // zero should mean count equal to the number of virtual cores
        if (s->codec_ctx->codec->capabilities & AV_CODEC_CAP_SLICE_THREADS) {
                if(!broken_h264_mt_decoding) {
                        s->codec_ctx->thread_count = 0; // == X264_THREADS_AUTO, perhaps same for other codecs
                        s->codec_ctx->thread_type = FF_THREAD_SLICE;
                        s->broken_h264_mt_decoding_workaroud_active = false;
                } else {
                        s->broken_h264_mt_decoding_workaroud_active = true;
                }
        } else {
#if 0
                log_msg(LOG_LEVEL_WARNING, "[lavd] Warning: Codec doesn't support slice-based multithreading.\n");
                if(s->codec->capabilities & CODEC_CAP_FRAME_THREADS) {
                        s->codec_ctx->thread_count = 0;
                        s->codec_ctx->thread_type = FF_THREAD_FRAME;
                } else {
                        fprintf(stderr, "[lavd] Warning: Codec doesn't support frame-based multithreading.\n");
                }
#endif
        }

        s->codec_ctx->flags2 |= AV_CODEC_FLAG2_FAST;

        // set by decoder
        s->codec_ctx->pix_fmt = AV_PIX_FMT_NONE;
        // callback to negotiate pixel format that is supported by UG
        s->codec_ctx->get_format = get_format_callback;

        s->codec_ctx->opaque = s;
}

static void jpeg_callback(void)
{
        log_msg(LOG_LEVEL_WARNING, "[lavd] Warning: JPEG decoder "
                        "will use full-scale YUV.\n");
}

struct decoder_info {
        codec_t ug_codec;
        enum AVCodecID avcodec_id;
        void (*codec_callback)(void);
        // Note:
        // Make sure that if adding hw decoders to prefered_decoders[] that
        // that decoder fails if there is not the HW during init, not while decoding
        // frames (like vdpau does). Otherwise, such a decoder would be initialized
        // but no frame decoded then.
        // Note 2:
        // cuvid decoders cannot be currently used as the default ones because they
        // currently support only 4:2:0 subsampling and fail during decoding if other
        // subsampling is given.
        const char *preferred_decoders[11]; // must be NULL-terminated
};

static const struct decoder_info decoders[] = {
        { H264, AV_CODEC_ID_H264, NULL, { NULL /* "h264_cuvid" */ } },
        { H265, AV_CODEC_ID_HEVC, NULL, { NULL /* "hevc_cuvid" */ } },
        { MJPG, AV_CODEC_ID_MJPEG, jpeg_callback, { NULL } },
        { JPEG, AV_CODEC_ID_MJPEG, jpeg_callback, { NULL } },
        { J2K, AV_CODEC_ID_JPEG2000, NULL, { NULL } },
        { VP8, AV_CODEC_ID_VP8, NULL, { NULL } },
        { VP9, AV_CODEC_ID_VP9, NULL, { NULL } },
};

ADD_TO_PARAM(force_lavd_decoder, "force-lavd-decoder", "* force-lavd-decoder=<decoder>[:<decoder2>...]\n"
                "  Forces specified Libavcodec decoder. If more need to be specified, use colon as a delimiter\n");

#ifdef USE_HWACC
ADD_TO_PARAM(force_hw_accel, "use-hw-accel", "* use-hw-accel\n"
                "  Tries to use hardware acceleration. \n");
#endif
static bool configure_with(struct state_libavcodec_decompress *s,
                struct video_desc desc)
{
        const struct decoder_info *dec = NULL;

        for (unsigned int i = 0; i < sizeof decoders / sizeof decoders[0]; ++i) {
                if (decoders[i].ug_codec == desc.color_spec) {
                        dec = &decoders[i];
                        break;
                }
        }

        if (dec == NULL) {
                log_msg(LOG_LEVEL_ERROR, "[lavd] Unsupported codec!!!\n");
                return false;
        }

        if (dec->codec_callback) {
                dec->codec_callback();
        }

        // construct priority list of decoders that can be used for the codec
        AVCodec *codecs_available[13]; // max num of preferred decoders (10) + user supplied + default one + NULL
        memset(codecs_available, 0, sizeof codecs_available);
        unsigned int codec_index = 0;
        // first try codec specified from cmdline if any
        if (get_commandline_param("force-lavd-decoder")) {
                const char *param = get_commandline_param("force-lavd-decoder");
                char *val = alloca(strlen(param) + 1);
                strcpy(val, param);
                char *item, *save_ptr;
                while ((item = strtok_r(val, ":", &save_ptr))) {
                        val = NULL;
                        AVCodec *codec = avcodec_find_decoder_by_name(item);
                        if (codec == NULL) {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Decoder not found: %s\n", item);
                        } else {
                                if (codec->id == dec->avcodec_id) {
                                        if (codec_index < (sizeof codecs_available / sizeof codecs_available[0] - 1)) {
                                                codecs_available[codec_index++] = codec;
                                        }
                                } else {
                                        log_msg(LOG_LEVEL_WARNING, "[lavd] Decoder not valid for codec: %s\n", item);
                                }
                        }
                }
        }
        // then try preferred codecs
        const char * const *preferred_decoders_it = dec->preferred_decoders;
        while (*preferred_decoders_it) {
                AVCodec *codec = avcodec_find_decoder_by_name(*preferred_decoders_it);
                if (codec == NULL) {
                        log_msg(LOG_LEVEL_VERBOSE, "[lavd] Decoder not available: %s\n", *preferred_decoders_it);
                        preferred_decoders_it++;
                        continue;
                } else {
                        if (codec_index < (sizeof codecs_available / sizeof codecs_available[0] - 1)) {
                                codecs_available[codec_index++] = codec;
                        }
                }
                preferred_decoders_it++;
        }
        // finally, add a default one if there are no preferred encoders or all fail
        if (codec_index < (sizeof codecs_available / sizeof codecs_available[0]) - 1) {
                codecs_available[codec_index++] = avcodec_find_decoder(dec->avcodec_id);
        }

        // initialize the codec - use the first decoder initialization of which succeeds
        AVCodec **codec_it = codecs_available;
        while (*codec_it) {
                log_msg(LOG_LEVEL_VERBOSE, "[lavd] Trying decoder: %s\n", (*codec_it)->name);
                s->codec_ctx = avcodec_alloc_context3(*codec_it);
                if(s->codec_ctx == NULL) {
                        log_msg(LOG_LEVEL_ERROR, "[lavd] Unable to allocate codec context.\n");
                        return false;
                }
                set_codec_context_params(s);
                pthread_mutex_lock(s->global_lavcd_lock);
                if (avcodec_open2(s->codec_ctx, *codec_it, NULL) < 0) {
                        avcodec_free_context(&s->codec_ctx);
                        pthread_mutex_unlock(s->global_lavcd_lock);
                        log_msg(LOG_LEVEL_WARNING, "[lavd] Unable to open decoder %s.\n", (*codec_it)->name);
                        codec_it++;
                        continue;
                } else {
                        pthread_mutex_unlock(s->global_lavcd_lock);
                        log_msg(LOG_LEVEL_NOTICE, "[lavd] Using decoder: %s\n", (*codec_it)->name);
                        break;
                }
        }

        if (s->codec_ctx == NULL) {
                log_msg(LOG_LEVEL_ERROR, "[lavd] Decoder could have not been initialized for codec %s.\n",
                                get_codec_name(desc.color_spec));
                return false;
        }

        s->frame = av_frame_alloc();
        if(!s->frame) {
                log_msg(LOG_LEVEL_ERROR, "[lavd] Unable allocate frame.\n");
                return false;
        }

        av_init_packet(&s->pkt);

        s->last_frame_seq_initialized = false;
        s->saved_desc = desc;

        return true;
}

static void * libavcodec_decompress_init(void)
{
        struct state_libavcodec_decompress *s;

        s = (struct state_libavcodec_decompress *)
                calloc(1, sizeof(struct state_libavcodec_decompress));

        s->global_lavcd_lock = rm_acquire_shared_lock(LAVCD_LOCK_NAME);
        if (log_level >= LOG_LEVEL_VERBOSE) {
                av_log_set_level(AV_LOG_VERBOSE);
        }

        /*   register all the codecs (you can also register only the codec
         *         you wish to have smaller code */
        avcodec_register_all();

        s->width = s->height = s->pitch = 0;
        s->codec_ctx = NULL;
        s->frame = NULL;
        av_init_packet(&s->pkt);
        s->pkt.data = NULL;
        s->pkt.size = 0;

        av_log_set_callback(error_callback);

#ifdef USE_HWACC
        hwaccel_state_init(&s->hwaccel);
#endif

        return s;
}

static int libavcodec_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_libavcodec_decompress *s =
                (struct state_libavcodec_decompress *) state;

        s->pitch = pitch;
        assert(out_codec == UYVY ||
                        out_codec == RGB ||
                        out_codec == v210 ||
                        out_codec == HW_VDPAU);

        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        s->in_codec = desc.color_spec;
        s->out_codec = out_codec;
        s->width = desc.width;
        s->height = desc.height;

        deconfigure(s);
        return configure_with(s, desc);
}

static void nv12_to_yuv422(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cbcr = (char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                char *dst = dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cbcr++;
                        *dst++ = *src_y++;
                }
        }
}

static void rgb24_to_uyvy(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch)
{
        for (int y = 0; y < height; ++y) {
                vc_copylineRGBtoUYVY((unsigned char *) dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0], vc_get_linesize(width, UYVY));
        }
}

static void rgb24_to_rgb(char *dst_buffer, AVFrame *frame,
                int width, int height, int pitch)
{
        for (int y = 0; y < height; ++y) {
                memcpy(dst_buffer + y * pitch, frame->data[0] + y * frame->linesize[0],
                                vc_get_linesize(width, RGB));
        }
}

static void yuv420p_to_yuv422(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height / 2; ++y) {
                char *src_y1 = (char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                char *src_y2 = (char *) in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1);
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst1 = dst_buffer + (y * 2) * pitch;
                char *dst2 = dst_buffer + (y * 2 + 1) * pitch;

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

                for(; x < width - 1; x += 2) {
                        *dst1++ = *src_cb;
                        *dst1++ = *src_y1++;
                        *dst1++ = *src_cr;
                        *dst1++ = *src_y1++;

                        *dst2++ = *src_cb++;
                        *dst2++ = *src_y2++;
                        *dst2++ = *src_cr++;
                        *dst2++ = *src_y2++;
                }
        }
}

static void yuv420p_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height / 2; ++y) {
                uint8_t *src_y1 = (in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint8_t *src_y2 = (in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                for(int x = 0; x < width / 6; ++x) {
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

static void yuv422p_to_yuv422(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                char *src_cb = (char *) in_frame->data[1] + in_frame->linesize[1] * y;
                char *src_cr = (char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = *src_cb++;
                        *dst++ = *src_y++;
                        *dst++ = *src_cr++;
                        *dst++ = *src_y++;
                }
        }
}

static void yuv422p_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 6; ++x) {
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


static void yuv444p_to_yuv422(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                char *src_y = (char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                char *dst = dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = (*src_cb + *(src_cb + 1)) / 2;
                        src_cb += 2;
                        *dst++ = *src_y++;
                        *dst++ = (*src_cr + *(src_cr + 1)) / 2;
                        src_cr += 2;
                        *dst++ = *src_y++;
                }
        }
}

static void yuv444p_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                uint8_t *src_y = (in_frame->data[0] + in_frame->linesize[0] * y);
                uint8_t *src_cb = (in_frame->data[1] + in_frame->linesize[1] * y);
                uint8_t *src_cr = (in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 6; ++x) {
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
 * Changes pixel format from planar YUV 422 to packed RGB.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static void nv12_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cbcr = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * (y / 2);
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        int cb = *src_cbcr++ - 128;
                        int cr = *src_cbcr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y++ << 16;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

/**
 * Changes pixel format from planar YUV 422 to packed RGB.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static void yuv422p_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;
                for(int x = 0; x < width / 2; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y++ << 16;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

/**
 * Changes pixel format from planar YUV 422 to packed RGB.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static void yuv420p_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height / 2; ++y) {
                unsigned char *src_y1 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y * 2;
                unsigned char *src_y2 = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1);
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst1 = (unsigned char *) dst_buffer + pitch * (y * 2);
                unsigned char *dst2 = (unsigned char *) dst_buffer + pitch * (y * 2 + 1);
                for(int x = 0; x < width / 2; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y1++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst1++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y1++ << 16;
                        *dst1++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst1++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y2++ << 16;
                        *dst2++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                        y = *src_y2++ << 16;
                        *dst2++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst2++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

/**
 * Changes pixel format from planar YUV 444 to packed RGB.
 * Color space is assumed ITU-T Rec. 609. YUV is expected to be full scale (aka in JPEG).
 */
static void yuv444p_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                unsigned char *src_y = (unsigned char *) in_frame->data[0] + in_frame->linesize[0] * y;
                unsigned char *src_cb = (unsigned char *) in_frame->data[1] + in_frame->linesize[1] * y;
                unsigned char *src_cr = (unsigned char *) in_frame->data[2] + in_frame->linesize[2] * y;
                unsigned char *dst = (unsigned char *) dst_buffer + pitch * y;
                for(int x = 0; x < width; ++x) {
                        int cb = *src_cb++ - 128;
                        int cr = *src_cr++ - 128;
                        int y = *src_y++ << 16;
                        int r = 75700 * cr;
                        int g = -26864 * cb - 38050 * cr;
                        int b = 133176 * cb;
                        *dst++ = min(max(r + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(g + y, 0), (1<<24) - 1) >> 16;
                        *dst++ = min(max(b + y, 0), (1<<24) - 1) >> 16;
                }
        }
}

static void yuv420p10le_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst1 = (uint32_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint32_t *dst2 = (uint32_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                for(int x = 0; x < width / 6; ++x) {
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

static void yuv422p10le_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 6; ++x) {
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

static void yuv444p10le_to_v210(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint32_t *dst = (uint32_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 6; ++x) {
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

static void yuv420p10le_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height / 2; ++y) {
                uint16_t *src_y1 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y * 2);
                uint16_t *src_y2 = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * (y * 2 + 1));
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst1 = (uint8_t *)(void *)(dst_buffer + (y * 2) * pitch);
                uint8_t *dst2 = (uint8_t *)(void *)(dst_buffer + (y * 2 + 1) * pitch);

                for(int x = 0; x < width / 2; ++x) {
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

static void yuv422p10le_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
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

static void yuv444p10le_to_uyvy(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        for(int y = 0; y < (int) height; ++y) {
                uint16_t *src_y = (uint16_t *)(void *)(in_frame->data[0] + in_frame->linesize[0] * y);
                uint16_t *src_cb = (uint16_t *)(void *)(in_frame->data[1] + in_frame->linesize[1] * y);
                uint16_t *src_cr = (uint16_t *)(void *)(in_frame->data[2] + in_frame->linesize[2] * y);
                uint8_t *dst = (uint8_t *)(void *)(dst_buffer + y * pitch);

                for(int x = 0; x < width / 2; ++x) {
                        *dst++ = (src_cb[0] + src_cb[0]) / 2 >> 2;
                        *dst++ = *src_y++ >> 2;
                        *dst++ = (src_cr[0] + src_cr[1]) / 2 >> 2;
                        *dst++ = *src_y++ >> 2;
                        src_cb += 2;
                        src_cr += 2;
                }
        }
}

static void yuv420p10le_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        char *tmp = malloc(vc_get_linesize(UYVY, width) * height);
        char *uyvy = tmp;
        yuv420p10le_to_uyvy(uyvy, in_frame, width, height, vc_get_linesize(UYVY, width));
        for (int i = 0; i < height; i++) {
                vc_copylineUYVYtoRGB((unsigned char *) dst_buffer, (unsigned char *) uyvy, vc_get_linesize(RGB, width));
                uyvy += vc_get_linesize(UYVY, width);
                dst_buffer += pitch;
        }
        free(tmp);
}

static void yuv422p10le_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        char *tmp = malloc(vc_get_linesize(UYVY, width) * height);
        char *uyvy = tmp;
        yuv422p10le_to_uyvy(uyvy, in_frame, width, height, vc_get_linesize(UYVY, width));
        for (int i = 0; i < height; i++) {
                vc_copylineUYVYtoRGB((unsigned char *) dst_buffer, (unsigned char *) uyvy, vc_get_linesize(RGB, width));
                uyvy += vc_get_linesize(UYVY, width);
                dst_buffer += pitch;
        }
        free(tmp);
}

static void yuv444p10le_to_rgb24(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        char *tmp = malloc(vc_get_linesize(UYVY, width) * height);
        char *uyvy = tmp;
        yuv444p10le_to_uyvy(uyvy, in_frame, width, height, vc_get_linesize(UYVY, width));
        for (int i = 0; i < height; i++) {
                vc_copylineUYVYtoRGB((unsigned char *) dst_buffer, (unsigned char *) uyvy, vc_get_linesize(RGB, width));
                uyvy += vc_get_linesize(UYVY, width);
                dst_buffer += pitch;
        }
        free(tmp);
}

static void not_implemented_conv(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        UNUSED(dst_buffer);
        UNUSED(in_frame);
        UNUSED(width);
        UNUSED(height);
        UNUSED(pitch);
        log_msg(LOG_LEVEL_ERROR, "Selected conversion is not implemented!\n");
}

#ifdef USE_HWACC
static void av_vdpau_to_ug_vdpau(char *dst_buffer, AVFrame *in_frame,
                int width, int height, int pitch)
{
        UNUSED(width);
        UNUSED(height);
        UNUSED(pitch);

        hw_vdpau_frame *out = (hw_vdpau_frame *) dst_buffer;

        hw_vdpau_frame_init(out);

        hw_vdpau_frame_from_avframe(out, in_frame);
}
#endif

static const struct {
        int av_codec;
        codec_t uv_codec;
        void (*convert)(char *dst_buffer, AVFrame *in_frame, int width, int height, int pitch);
} convert_funcs[] = {
        // 10-bit YUV
        {AV_PIX_FMT_YUV420P10LE, v210, yuv420p10le_to_v210},
        {AV_PIX_FMT_YUV420P10LE, UYVY, yuv420p10le_to_uyvy},
        {AV_PIX_FMT_YUV420P10LE, RGB, yuv420p10le_to_rgb24},
        {AV_PIX_FMT_YUV422P10LE, v210, yuv422p10le_to_v210},
        {AV_PIX_FMT_YUV422P10LE, UYVY, yuv422p10le_to_uyvy},
        {AV_PIX_FMT_YUV422P10LE, RGB, yuv422p10le_to_rgb24},
        {AV_PIX_FMT_YUV444P10LE, v210, yuv444p10le_to_v210},
        {AV_PIX_FMT_YUV444P10LE, UYVY, yuv444p10le_to_uyvy},
        {AV_PIX_FMT_YUV444P10LE, RGB, yuv444p10le_to_rgb24},
        // 8-bit YUV
        {AV_PIX_FMT_YUV420P, v210, yuv420p_to_v210},
        {AV_PIX_FMT_YUV420P, UYVY, yuv420p_to_yuv422},
        {AV_PIX_FMT_YUV420P, RGB, yuv420p_to_rgb24},
        {AV_PIX_FMT_YUV422P, v210, yuv422p_to_v210},
        {AV_PIX_FMT_YUV422P, UYVY, yuv422p_to_yuv422},
        {AV_PIX_FMT_YUV422P, RGB, yuv422p_to_rgb24},
        {AV_PIX_FMT_YUV444P, v210, yuv444p_to_v210},
        {AV_PIX_FMT_YUV444P, UYVY, yuv444p_to_yuv422},
        {AV_PIX_FMT_YUV444P, RGB, yuv444p_to_rgb24},
        // 8-bit YUV (JPEG color range)
        {AV_PIX_FMT_YUVJ420P, v210, yuv420p_to_v210},
        {AV_PIX_FMT_YUVJ420P, UYVY, yuv420p_to_yuv422},
        {AV_PIX_FMT_YUVJ420P, RGB, yuv420p_to_rgb24},
        {AV_PIX_FMT_YUVJ422P, v210, yuv422p_to_v210},
        {AV_PIX_FMT_YUVJ422P, UYVY, yuv422p_to_yuv422},
        {AV_PIX_FMT_YUVJ422P, RGB, yuv422p_to_rgb24},
        {AV_PIX_FMT_YUVJ444P, v210, yuv444p_to_v210},
        {AV_PIX_FMT_YUVJ444P, UYVY, yuv444p_to_yuv422},
        {AV_PIX_FMT_YUVJ444P, RGB, yuv444p_to_rgb24},
        // 8-bit YUV (NV12)
        {AV_PIX_FMT_NV12, v210, not_implemented_conv},
        {AV_PIX_FMT_NV12, UYVY, nv12_to_yuv422},
        {AV_PIX_FMT_NV12, RGB, nv12_to_rgb24},
        // RGB
        {AV_PIX_FMT_RGB24, v210, not_implemented_conv},
        {AV_PIX_FMT_RGB24, UYVY, rgb24_to_uyvy},
        {AV_PIX_FMT_RGB24, RGB, rgb24_to_rgb},
#ifdef USE_HWACC
        // HW acceleration
        {AV_PIX_FMT_VDPAU, HW_VDPAU, av_vdpau_to_ug_vdpau},
#endif
};

#ifdef USE_HWACC
static int create_hw_device_ctx(enum AVHWDeviceType type, AVBufferRef **device_ref){
        int ret;
        ret = av_hwdevice_ctx_create(device_ref, type, NULL, NULL, 0);

        if(ret < 0){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Unable to create hwdevice!!\n");
                return ret;
        }

        return 0;
}

static int create_hw_frame_ctx(AVBufferRef *device_ref,
                AVCodecContext *s,
                enum AVPixelFormat format,
                enum AVPixelFormat sw_format,
                int decode_surfaces,
                AVBufferRef **ctx)
{
        *ctx = av_hwframe_ctx_alloc(device_ref);
        if(!*ctx){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Failed to allocate hwframe_ctx!!\n");
                return -1;
        }

        AVHWFramesContext *frames_ctx = (AVHWFramesContext *) (*ctx)->data;
        frames_ctx->format    = format;
        frames_ctx->width     = s->coded_width;
        frames_ctx->height    = s->coded_height;
        frames_ctx->sw_format = sw_format;
        frames_ctx->initial_pool_size = decode_surfaces;

        int ret = av_hwframe_ctx_init(*ctx);
        if (ret < 0) {
                av_buffer_unref(ctx);
                *ctx = NULL;
                log_msg(LOG_LEVEL_ERROR, "[lavd] Unable to init hwframe_ctx!!\n\n");
                return ret;
        }

        return 0;
}

static int vdpau_init(struct AVCodecContext *s){

        struct state_libavcodec_decompress *state = s->opaque;

        AVBufferRef *device_ref = NULL;
        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_VDPAU, &device_ref);
        if(ret < 0)
                return ret;

        AVHWDeviceContext *device_ctx = (AVHWDeviceContext*)device_ref->data;
        AVVDPAUDeviceContext *device_vdpau_ctx = device_ctx->hwctx;

        AVBufferRef *hw_frames_ctx = NULL;
        ret = create_hw_frame_ctx(device_ref,
                        s,
                        AV_PIX_FMT_VDPAU,
                        s->sw_pix_fmt,
                        DEFAULT_SURFACES,
                        &hw_frames_ctx);
        if(ret < 0)
                goto fail;

        s->hw_frames_ctx = hw_frames_ctx;

        state->hwaccel.type = HWACCEL_VDPAU;
        state->hwaccel.copy = false;
        state->hwaccel.tmp_frame = av_frame_alloc();
        if(!state->hwaccel.tmp_frame){
                ret = -1;
                goto fail;
        }

        if(av_vdpau_bind_context(s, device_vdpau_ctx->device, device_vdpau_ctx->get_proc_address,
                                AV_HWACCEL_FLAG_ALLOW_HIGH_DEPTH |
                                AV_HWACCEL_FLAG_IGNORE_LEVEL)){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Unable to bind vdpau context!\n");
                ret = -1;
                goto fail;
        }

        av_buffer_unref(&device_ref);
        return 0;

fail:
        av_frame_free(&state->hwaccel.tmp_frame);
        av_buffer_unref(&hw_frames_ctx);
        av_buffer_unref(&device_ref);
        return ret;
}

struct vaapi_ctx{
        AVBufferRef *device_ref;
        AVHWDeviceContext *device_ctx;
        AVVAAPIDeviceContext *device_vaapi_ctx;

        AVBufferRef *hw_frames_ctx;
        AVHWFramesContext *frame_ctx;

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
        VAProfile va_profile;
        VAEntrypoint va_entrypoint;
        VAConfigID va_config;
        VAContextID va_context;

        struct vaapi_context decoder_context;
#endif
};

static void vaapi_uninit(struct hw_accel_state *s){

        free(s->ctx);
}

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
static const struct {
        enum AVCodecID av_codec_id;
        int codec_profile;
        VAProfile va_profile;
} vaapi_profiles[] = {
        {AV_CODEC_ID_MPEG2VIDEO, FF_PROFILE_MPEG2_SIMPLE, VAProfileMPEG2Simple},
        {AV_CODEC_ID_MPEG2VIDEO, FF_PROFILE_MPEG2_MAIN, VAProfileMPEG2Main},
        {AV_CODEC_ID_H264, FF_PROFILE_H264_CONSTRAINED_BASELINE, VAProfileH264ConstrainedBaseline},
        {AV_CODEC_ID_H264, FF_PROFILE_H264_BASELINE, VAProfileH264Baseline},
        {AV_CODEC_ID_H264, FF_PROFILE_H264_MAIN, VAProfileH264Main},
        {AV_CODEC_ID_H264, FF_PROFILE_H264_HIGH, VAProfileH264High},
#if VA_CHECK_VERSION(0, 37, 0)
        {AV_CODEC_ID_HEVC, FF_PROFILE_HEVC_MAIN, VAProfileHEVCMain},
#endif
};

static int vaapi_create_context(struct vaapi_ctx *ctx,
                AVCodecContext *codec_ctx)
{
        const AVCodecDescriptor *codec_desc;

        codec_desc = avcodec_descriptor_get(codec_ctx->codec_id);
        if(!codec_desc){
                return -1;
        }

        int profile_count = vaMaxNumProfiles(ctx->device_vaapi_ctx->display);
        log_msg(LOG_LEVEL_VERBOSE, "VAAPI Profile count: %d\n", profile_count);

        VAProfile *list = av_malloc(profile_count * sizeof(VAProfile));
        if(!list){
                return -1;
        }

        VAStatus status = vaQueryConfigProfiles(ctx->device_vaapi_ctx->display,
                        list, &profile_count);
        if(status != VA_STATUS_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Profile query failed: %d (%s)\n", status, vaErrorStr(status));
                av_free(list);
                return -1;
        }

        VAProfile profile = VAProfileNone;
        int match = 0;

        for(unsigned i = 0; i < FF_ARRAY_ELEMS(vaapi_profiles); i++){
                if(vaapi_profiles[i].av_codec_id != codec_ctx->codec_id)
                        continue;

                if(vaapi_profiles[i].codec_profile == codec_ctx->profile){
                        profile = vaapi_profiles[i].va_profile;
                        break;
                }
        }

        for(int i = 0; i < profile_count; i++){
                if(profile == list[i])
                        match = 1;
        }

        av_freep(&list);

        if(!match){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Profile not supported \n");
                return -1;
        }

        ctx->va_profile = profile;
        ctx->va_entrypoint = VAEntrypointVLD;

        status = vaCreateConfig(ctx->device_vaapi_ctx->display, ctx->va_profile,
                        ctx->va_entrypoint, 0, 0, &ctx->va_config);
        if(status != VA_STATUS_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Create config failed: %d (%s)\n", status, vaErrorStr(status));
                return -1;
        }

        AVVAAPIHWConfig *hwconfig = av_hwdevice_hwconfig_alloc(ctx->device_ref);
        if(!hwconfig){
                log_msg(LOG_LEVEL_WARNING, "[lavd] Failed to get constraints. Will try to continue anyways...\n");
                return 0;
        }

        hwconfig->config_id = ctx->va_config;
        AVHWFramesConstraints *constraints = av_hwdevice_get_hwframe_constraints(ctx->device_ref, hwconfig);
        if (!constraints){
                log_msg(LOG_LEVEL_WARNING, "[lavd] Failed to get constraints. Will try to continue anyways...\n");
                av_freep(&hwconfig);
                return 0;
        }

        if (codec_ctx->coded_width  < constraints->min_width  ||
                        codec_ctx->coded_width  > constraints->max_width  ||
                        codec_ctx->coded_height < constraints->min_height ||
                        codec_ctx->coded_height > constraints->max_height)
        {
                log_msg(LOG_LEVEL_WARNING, "[lavd] VAAPI hw does not support the resolution %dx%d\n",
                                codec_ctx->coded_width,
                                codec_ctx->coded_height);
                av_hwframe_constraints_free(&constraints);
                av_freep(&hwconfig);
                return -1;
        }

        av_hwframe_constraints_free(&constraints);
        av_freep(&hwconfig);

        return 0;
}
#endif

static int vaapi_init(struct AVCodecContext *s){

        struct state_libavcodec_decompress *state = s->opaque;

        struct vaapi_ctx *ctx = calloc(1, sizeof(struct vaapi_ctx));
        if(!ctx){
                return -1;
        }

        int ret = create_hw_device_ctx(AV_HWDEVICE_TYPE_VAAPI, &ctx->device_ref);
        if(ret < 0)
                goto fail;

        ctx->device_ctx = (AVHWDeviceContext*)ctx->device_ref->data;
        ctx->device_vaapi_ctx = ctx->device_ctx->hwctx;

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
        ret = vaapi_create_context(ctx, s);
        if(ret < 0)
                goto fail;
#endif

        int decode_surfaces = DEFAULT_SURFACES;

        if (s->active_thread_type & FF_THREAD_FRAME)
                decode_surfaces += s->thread_count;

        ret = create_hw_frame_ctx(ctx->device_ref,
                        s,
                        AV_PIX_FMT_VAAPI,
                        s->sw_pix_fmt,
                        decode_surfaces,
                        &ctx->hw_frames_ctx);
        if(ret < 0)
                goto fail;

        ctx->frame_ctx = (AVHWFramesContext *) (ctx->hw_frames_ctx->data);

        s->hw_frames_ctx = ctx->hw_frames_ctx;

        state->hwaccel.tmp_frame = av_frame_alloc();
        if(!state->hwaccel.tmp_frame){
                ret = -1;
                goto fail;
        }
        state->hwaccel.type = HWACCEL_VAAPI;
        state->hwaccel.copy = true;
        state->hwaccel.ctx = ctx;
        state->hwaccel.uninit = vaapi_uninit;

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
        AVVAAPIFramesContext *avfc = ctx->frame_ctx->hwctx;
        VAStatus status = vaCreateContext(ctx->device_vaapi_ctx->display,
                        ctx->va_config, s->coded_width, s->coded_height,
                        VA_PROGRESSIVE,
                        avfc->surface_ids,
                        avfc->nb_surfaces,
                        &ctx->va_context);

        if(status != VA_STATUS_SUCCESS){
                log_msg(LOG_LEVEL_ERROR, "[lavd] Create config failed: %d (%s)\n", status, vaErrorStr(status));
                ret = -1;
                goto fail;
        }

        ctx->decoder_context.display = ctx->device_vaapi_ctx->display;
        ctx->decoder_context.config_id = ctx->va_config;
        ctx->decoder_context.context_id = ctx->va_context;

        s->hwaccel_context = &ctx->decoder_context;
#endif

        av_buffer_unref(&ctx->device_ref);
        return 0;


fail:
        av_frame_free(&state->hwaccel.tmp_frame);
        av_buffer_unref(&ctx->hw_frames_ctx);
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 74, 100)
        if(ctx->device_vaapi_ctx)
                vaDestroyConfig(ctx->device_vaapi_ctx->display, ctx->va_config);
#endif
        av_buffer_unref(&ctx->device_ref);
        free(ctx);
        return ret;
}
#endif

static enum AVPixelFormat get_format_callback(struct AVCodecContext *s __attribute__((unused)), const enum AVPixelFormat *fmt)
{
        if (log_level >= LOG_LEVEL_DEBUG) {
                char out[1024] = "[lavd] Available output pixel formats:";
                const enum AVPixelFormat *it = fmt;
                while (*it != AV_PIX_FMT_NONE) {
                        strncat(out, " ", sizeof out - strlen(out) - 1);
                        strncat(out, av_get_pix_fmt_name(*it++), sizeof out - strlen(out) - 1);
                }
                log_msg(LOG_LEVEL_DEBUG, "%s\n", out);
        }


#ifdef USE_HWACC
        struct state_libavcodec_decompress *state = (struct state_libavcodec_decompress *) s->opaque;
        hwaccel_state_reset(&state->hwaccel);
        const char *param = get_commandline_param("use-hw-accel");
        bool hwaccel = param != NULL;

        if(hwaccel){
                for(const enum AVPixelFormat *it = fmt; *it != AV_PIX_FMT_NONE; it++){
                        if (*it == AV_PIX_FMT_VDPAU){
                                int ret = vdpau_init(s);
                                if(ret < 0){
                                        hwaccel_state_reset(&state->hwaccel);
                                        continue;
                                }
                                return AV_PIX_FMT_VDPAU;
                        }
                        if (*it == AV_PIX_FMT_VAAPI){
                                int ret = vaapi_init(s);
                                if(ret < 0){
                                        hwaccel_state_reset(&state->hwaccel);
                                        continue;
                                }
                                return AV_PIX_FMT_VAAPI;
                        }
                }

                log_msg(LOG_LEVEL_WARNING, "[lavd] Falling back to software decoding!\n");
        }

#endif

        while (*fmt != AV_PIX_FMT_NONE) {
                for (unsigned int i = 0; i < sizeof convert_funcs / sizeof convert_funcs[0]; ++i) {
                        if (convert_funcs[i].av_codec == *fmt) {
                                return *fmt;
                        }
                }
                fmt++;
        }

        return AV_PIX_FMT_NONE;
}


/**
 * Changes pixel format from frame to native (currently UYVY).
 *
 * @todo             figure out color space transformations - eg. JPEG returns full-scale YUV.
 *                   And not in the ITU-T Rec. 701 (eventually Rec. 609) scale.
 * @param  frame     video frame returned from libavcodec decompress
 * @param  dst       destination buffer where data will be stored
 * @param  av_codec  libav pixel format
 * @param  out_codec requested output codec
 * @param  width     frame width
 * @param  height    frame height
 * @retval TRUE      if the transformation was successful
 * @retval FALSE     if transformation failed
 * @see    yuvj422p_to_yuv422
 * @see    yuv420p_to_yuv422
 */
static int change_pixfmt(AVFrame *frame, unsigned char *dst, int av_codec,
                codec_t out_codec, int width, int height, int pitch) {
        assert(out_codec == UYVY ||
                        out_codec == RGB ||
                        out_codec == v210 ||
                        out_codec == HW_VDPAU);

        void (*convert)(char *dst_buffer, AVFrame *in_frame, int width, int height, int pitch) = NULL;
        for (unsigned int i = 0; i < sizeof convert_funcs / sizeof convert_funcs[0]; ++i) {
                if (convert_funcs[i].av_codec == av_codec &&
                                convert_funcs[i].uv_codec == out_codec) {
                        convert = convert_funcs[i].convert;
                }
        }

        if (convert) {
                convert((char *) dst, frame, width, height, pitch);
        } else {
                log_msg(LOG_LEVEL_ERROR, "Unsupported pixel "
                                "format: %s (id %d)\n",
                                av_get_pix_fmt_name(
                                        av_codec), av_codec);
                return FALSE;
        }

        return TRUE;
}

static void error_callback(void *ptr, int level, const char *fmt, va_list vl) {
        if(strcmp("unset current_picture_ptr on %d. slice\n", fmt) == 0)
                broken_h264_mt_decoding = true;
        av_log_default_callback(ptr, level, fmt, vl);
}

#ifdef USE_HWACC
static void transfer_frame(struct hw_accel_state *s, AVFrame *frame){
        av_hwframe_transfer_data(s->tmp_frame, frame, 0);

        av_frame_copy_props(s->tmp_frame, frame);

        av_frame_unref(frame);
        av_frame_move_ref(frame, s->tmp_frame);
}
#endif

static int libavcodec_decompress(void *state, unsigned char *dst, unsigned char *src,
                unsigned int src_len, int frame_seq)
{
        struct state_libavcodec_decompress *s = (struct state_libavcodec_decompress *) state;
        int len, got_frame = 0;
        int res = FALSE;

        s->pkt.size = src_len;
        s->pkt.data = src;

        while (s->pkt.size > 0) {
                struct timeval t0, t1;
                gettimeofday(&t0, NULL);
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 37, 100)
                len = avcodec_decode_video2(s->codec_ctx, s->frame, &got_frame, &s->pkt);
#else
                got_frame = 0;
                int ret = avcodec_send_packet(s->codec_ctx, &s->pkt);
                if (ret == 0) {
                        ret = avcodec_receive_frame(s->codec_ctx, s->frame);
                        if (ret == 0) {
                                got_frame = 1;
                        }
                }
                if (ret != 0) {
                        print_decoder_error(MOD_NAME, ret);
                }
                len = s->pkt.size;
#endif
                gettimeofday(&t1, NULL);

                /*
                 * Hack: Some libavcodec versions (typically found in Libav)
                 * do not correctly support JPEG with more than one reset
                 * segment (GPUJPEG) or more than one slices (compressed with
                 * libavcodec). It returns error although it is actually able
                 * to decompress the frame correctly. So we assume that the
                 * decompression went good even with the reported error.
                 */
                if (len < 0) {
                        if (s->in_codec == JPEG) {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Perhaps JPEG restart interval >0 set? (Not supported by lavd, try '-c JPEG:90:0' on sender).\n");
                        } else if (s->in_codec == MJPG) {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Perhaps old libavcodec without slices support? (Try '-c libavcodec:codec=MJPEG:threads=no' on sender).\n");
#if LIBAVCODEC_VERSION_MAJOR <= 54 // Libav with libavcodec 54 will crash otherwise
                                return FALSE;
#endif
                        } else {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Error while decoding frame.\n");
                                return FALSE;
                        }
                }

                if(got_frame) {
                        log_msg(LOG_LEVEL_DEBUG, "[lavd] Decompressing %c frame took %f sec.\n", av_get_picture_type_char(s->frame->pict_type), tv_diff(t1, t0));
                        /* Skip the frame if this is not an I-frame
                         * and we have missed some of previous frames for VP8 because the
                         * decoder makes ugly artifacts. We rather wait for next I-frame. */
                        if (s->in_codec == VP8 &&
                                        (s->frame->pict_type != AV_PICTURE_TYPE_I &&
                                         (!s->last_frame_seq_initialized || (s->last_frame_seq + 1) % ((1<<22) - 1) != frame_seq))) {
                                log_msg(LOG_LEVEL_WARNING, "[lavd] Missing appropriate I-frame "
                                                "(last valid %d, this %u).\n",
                                                s->last_frame_seq_initialized ?
                                                s->last_frame_seq : -1, (unsigned) frame_seq);
                                res = FALSE;
                        } else {
#ifdef USE_HWACC
                                if(s->hwaccel.copy){
                                        transfer_frame(&s->hwaccel, s->frame);
                                }
#endif
                                printf("interlaced: %d", s->frame->interlaced_frame);
                                printf("Display num: %d, coded num: %d, top_first: %d\n", s->frame->display_picture_number, s->frame->coded_picture_number, s->frame->top_field_first);
                                res = change_pixfmt(s->frame, dst, s->frame->format,
                                                s->out_codec, s->width, s->height, s->pitch);
                                if(res == TRUE) {
                                        s->last_frame_seq_initialized = true;
                                        s->last_frame_seq = frame_seq;
                                }
                        }
                }

                if (len <= 0) {
                        break;
                }

                if(s->pkt.data) {
                        s->pkt.size -= len;
                        s->pkt.data += len;
                }
        }

        if(broken_h264_mt_decoding) {
                if(!s->broken_h264_mt_decoding_workaroud_active) {
                        libavcodec_decompress_reconfigure(s, s->saved_desc,
                                        s->rshift, s->gshift, s->bshift, s->pitch, s->out_codec);
                }
                if(s->broken_h264_mt_decoding_workaroud_warning_displayed++ % 1000 == 0)
                        av_log(NULL, AV_LOG_WARNING, "Broken multi-threaded decoder detected, "
                                        "switching to a single-threaded one! Consider upgrading your Libavcodec.\n");
        }

        return res;
}

static int libavcodec_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_libavcodec_decompress *s =
                (struct state_libavcodec_decompress *) state;
        UNUSED(s);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
#ifdef LAVD_ACCEPT_CORRUPTED
                                *(int *) val = TRUE;
#else
                                *(int *) val = FALSE;
#endif
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

static void libavcodec_decompress_done(void *state)
{
        struct state_libavcodec_decompress *s =
                (struct state_libavcodec_decompress *) state;

        deconfigure(s);

        rm_release_shared_lock(LAVCD_LOCK_NAME);

        free(s);
}

ADD_TO_PARAM(lavd_use_10bit, "lavd-use-10bit",
                "* lavd-use-10bit\n"
                "  Indicates that we are using decoding to v210 (currently only H.264/HEVC).\n"
                "  If so, it can be decompressed to v210. With this flag, v210 (10-bit YUV)\n"
                "  will be announced as a supported codec.\n");
static const struct decode_from_to *libavcodec_decompress_get_decoders() {
        const struct decode_from_to dec_static[] = {
                { H264, UYVY, 500 },
                { H265, UYVY, 500 },
                { JPEG, UYVY, 600 },
                { MJPG, UYVY, 500 },
                { J2K, RGB, 500 },
                { VP8, UYVY, 500 },
                { VP9, UYVY, 500 },
                { H264, HW_VDPAU, 200 },
        };

        static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
        static struct decode_from_to ret[sizeof dec_static / sizeof dec_static[0] + 1 /* terminating zero */ + 10 /* place for additional decoders, see below */];

        pthread_mutex_lock(&lock); // prevent concurent initialization
        if (ret[0].from == VIDEO_CODEC_NONE) { // not yet initialized
                memcpy(ret, dec_static, sizeof dec_static);
                // add also decoder from H.264/HEVC to v210 if user explicitly indicated to do so
                if (get_commandline_param("lavd-use-10bit")) {
                        ret[sizeof dec_static / sizeof dec_static[0]] =
                                (struct decode_from_to) {H264, v210, 400};
                        ret[sizeof dec_static / sizeof dec_static[0] + 1] =
                                (struct decode_from_to) {H265, v210, 400};
                }
        }
        pthread_mutex_unlock(&lock); // prevent concurent initialization

        return ret;
}

static const struct video_decompress_info libavcodec_info = {
        libavcodec_decompress_init,
        libavcodec_decompress_reconfigure,
        libavcodec_decompress,
        libavcodec_decompress_get_property,
        libavcodec_decompress_done,
        libavcodec_decompress_get_decoders,
};

REGISTER_MODULE(libavcodec, &libavcodec_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);

