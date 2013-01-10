/*
 * FILE:    video_compress/libavcodec.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2011 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video_compress/libavcodec.h"

#include <assert.h>
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>

#include "debug.h"
#include "host.h"
#include "video.h"
#include "video_codec.h"

#define DEFAULT_CODEC CODEC_ID_H264

struct libav_video_compress {
        struct video_frame *out[2];
        struct video_desc   saved_desc;

        AVFrame            *in_frame;
        AVCodec            *codec;
        AVCodecContext     *codec_ctx;
#ifdef HAVE_AVCODEC_ENCODE_VIDEO2
        AVPacket            pkt[2];
#endif

        unsigned char      *decoded;
        decoder_t           decoder;

        bool                configured;
};

static void to_yuv420(AVFrame *out_frame, unsigned char *in_data);

void * libavcodec_compress_init(char * fmt)
{
        UNUSED(fmt);
        struct libav_video_compress *s;
        
        s = (struct libav_video_compress *) malloc(sizeof(struct libav_video_compress));
        s->out[0] = s->out[1] = NULL;
        s->configured = false;

        s->codec = NULL;
        s->codec_ctx = NULL;
        s->in_frame = NULL;

        s->decoded = NULL;

#ifdef HAVE_AVCODEC_ENCODE_VIDEO2
        for(int i = 0; i < 2; ++i) {
                av_init_packet(&s->pkt[i]);
                s->pkt[i].data = NULL;
                s->pkt[i].size = 0;
        }
#endif

        /*  register all the codecs (you can also register only the codec
         *         you wish to have smaller code */
        avcodec_register_all();

        return s;
}

static bool configure_with(struct libav_video_compress *s, struct video_frame *frame)
{
        int ret;
        int codec_id = DEFAULT_CODEC;
        // implement multiple tiles support if needed
        assert(frame->tile_count == 1);
        s->saved_desc = video_desc_from_frame(frame);

        struct video_desc compressed_desc;
        compressed_desc = video_desc_from_frame(frame);
        switch(codec_id) {
                case CODEC_ID_H264:
                        compressed_desc.color_spec = H264;
                        break;
                default:
                        fprintf(stderr, "[Libavcodec] Unable to match "
                                        "desired codec to UltraGrid internal "
                                        "one.\n");
                        return false;

        }

        for(int i = 0; i < 2; ++i) {
                s->out[i] = vf_alloc_desc(compressed_desc);
#ifndef HAVE_AVCODEC_ENCODE_VIDEO2
                s->out[i]->tiles[0].data = malloc(compressed_desc.width *
                        compressed_desc.height * 4);
#endif // HAVE_AVCODEC_ENCODE_VIDEO2
        }

        /* find the video encoder */
        s->codec = avcodec_find_encoder(codec_id);
        if (!s->codec) {
                fprintf(stderr, "Libavcodec doesn't contain specified codec (H.264).\n"
                                "Hint: Check if you have libavcodec-extra package installed.\n");
                return false;
        }

        // avcodec_alloc_context3 allocates context and sets default value
        s->codec_ctx = avcodec_alloc_context3(s->codec);
        if (!s->codec_ctx) {
                fprintf(stderr, "Could not allocate video codec context\n");
                return false;
        }

        /* put parameters */
        s->codec_ctx->bit_rate = frame->tiles[0].width * frame->tiles[0].height *
                4 * /* for H.264: 1 - low motion, 2 - medium motion, 4 - high motion */
                0.07 * frame->fps;
        /* resolution must be a multiple of two */
        s->codec_ctx->width = frame->tiles[0].width;
        s->codec_ctx->height = frame->tiles[0].height;
        /* frames per second */
        s->codec_ctx->time_base= (AVRational){1,(int) frame->fps};
        s->codec_ctx->gop_size = 20; /* emit one intra frame every ten frames */
        s->codec_ctx->max_b_frames = 0;
        switch(frame->color_spec) {
                case Vuy2:
                case DVS8:
                case UYVY:
                        s->decoder = (decoder_t) memcpy;
                        break;
                case YUYV:
                        s->decoder = (decoder_t) vc_copylineYUYV;
                        break;
                case v210:
                        s->decoder = (decoder_t) vc_copylinev210;
                        break;
                default:
                        fprintf(stderr, "[Libavcodec] Unable to find "
                                        "appropriate pixel format.\n");
                        return false;
        }
#ifdef HAVE_AVCODEC_ENCODE_VIDEO2
        s->codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
#else
        s->codec_ctx->pix_fmt = PIX_FMT_YUV420P;
#endif

        s->decoded = malloc(frame->tiles[0].width * frame->tiles[0].height * 4);

        if(codec_id == CODEC_ID_H264) {
                av_opt_set(s->codec_ctx->priv_data, "preset", "ultrafast", 0);
                //av_opt_set(s->codec_ctx->priv_data, "tune", "fastdecode", 0);
                av_opt_set(s->codec_ctx->priv_data, "tune", "zerolatency", 0);
        }

        /* open it */
        if (avcodec_open2(s->codec_ctx, s->codec, NULL) < 0) {
                fprintf(stderr, "Could not open codec\n");
                return false;
        }

        s->in_frame = avcodec_alloc_frame();
        if (!s->in_frame) {
                fprintf(stderr, "Could not allocate video frame\n");
                return false;
        }
        s->in_frame->format = s->codec_ctx->pix_fmt;
        s->in_frame->width = s->codec_ctx->width;
        s->in_frame->height = s->codec_ctx->height;

        /* the image can be allocated by any means and av_image_alloc() is
         * just the most convenient way if av_malloc() is to be used */
        ret = av_image_alloc(s->in_frame->data, s->in_frame->linesize,
                        s->codec_ctx->width, s->codec_ctx->height,
                        s->codec_ctx->pix_fmt, 32);
        if (ret < 0) {
                fprintf(stderr, "Could not allocate raw picture buffer\n");
                return false;
        }

        return true;
}

static void to_yuv420(AVFrame *out_frame, unsigned char *in_data)
{
        for(int y = 0; y < (int) out_frame->height; ++y) {
                unsigned char *src = in_data + out_frame->width * y * 2;
                unsigned char *dst_y = out_frame->data[0] + out_frame->linesize[0] * y;
                for(int x = 0; x < out_frame->width; ++x) {
                        dst_y[x] = src[x * 2 + 1];
                }
        }

        for(int y = 0; y < (int) out_frame->height / 2; ++y) {
                /*  every even row */
                unsigned char *src1 = in_data + (y * 2) * (out_frame->width * 2);
                /*  every odd row */
                unsigned char *src2 = in_data + (y * 2 + 1) * (out_frame->width * 2);
                unsigned char *dst_cb = out_frame->data[1] + out_frame->linesize[1] * y;
                unsigned char *dst_cr = out_frame->data[2] + out_frame->linesize[2] * y;
                for(int x = 0; x < out_frame->width / 2; ++x) {
                        dst_cb[x] = (src1[x * 4] + src2[x * 4]) / 2;
                        dst_cr[x] = (src1[x * 4 + 2] + src1[x * 4 + 2]) / 2;
                }
        }
}

struct video_frame * libavcodec_compress(void *arg, struct video_frame * tx, int buffer_idx)
{
        struct libav_video_compress *s = (struct libav_video_compress *) arg;
        assert (buffer_idx == 0 || buffer_idx == 1);
        static int frame_seq = 0;
        int ret;
#ifdef HAVE_AVCODEC_ENCODE_VIDEO2
        int got_output;
#endif

        if(!s->configured) {
                int ret = configure_with(s, tx);
                if(!ret) {
                        return NULL;
                }
                s->configured = true;
        } else {
                // reconfiguration not yet implemented
                assert(video_desc_eq(video_desc_from_frame(tx),
                                        s->saved_desc));
        }

        s->in_frame->pts = frame_seq++;
#ifdef HAVE_AVCODEC_ENCODE_VIDEO2
        av_free_packet(&s->pkt[buffer_idx]);
        av_init_packet(&s->pkt[buffer_idx]);
        s->pkt[buffer_idx].data = NULL;
        s->pkt[buffer_idx].size = 0;
#endif

        if((void *) s->decoder != (void *) memcpy) {
                unsigned char *line1 = (unsigned char *) tx->tiles[0].data;
                unsigned char *line2 = (unsigned char *) s->decoded;
                int src_linesize = vc_get_linesize(tx->tiles[0].width, tx->color_spec);
                int dst_linesize = tx->tiles[0].width * 2; /* UYVY */
                for (int i = 0; i < (int) tx->tiles[0].height; ++i) {
                        s->decoder(line2, line1, dst_linesize,
                                        0, 8, 16);
                        line1 += src_linesize;
                        line2 += dst_linesize;
                }
                to_yuv420(s->in_frame, s->decoded);
        } else {
                to_yuv420(s->in_frame, (unsigned char *) tx->tiles[0].data);
        }


#ifdef HAVE_AVCODEC_ENCODE_VIDEO2
        /* encode the image */
        ret = avcodec_encode_video2(s->codec_ctx, &s->pkt[buffer_idx],
                        s->in_frame, &got_output);
        if (ret < 0) {
                fprintf(stderr, "Error encoding frame\n");
                return NULL;
        }

        if (got_output) {
                //printf("Write frame %3d (size=%5d)\n", frame_seq, s->pkt[buffer_idx].size);
                s->out[buffer_idx]->tiles[0].data = (char *) s->pkt[buffer_idx].data;
                s->out[buffer_idx]->tiles[0].data_len = s->pkt[buffer_idx].size;
        } else {
                return NULL;
        }
#else
        /* encode the image */
        ret = avcodec_encode_video(s->codec_ctx, (uint8_t *) s->out[buffer_idx]->tiles[0].data,
                        s->out[buffer_idx]->tiles[0].width * s->out[buffer_idx]->tiles[0].height * 4,
                        s->in_frame);
        if (ret < 0) {
                fprintf(stderr, "Error encoding frame\n");
                return NULL;
        }

        if (ret) {
                //printf("Write frame %3d (size=%5d)\n", frame_seq, s->pkt[buffer_idx].size);
                s->out[buffer_idx]->tiles[0].data_len = ret;
        } else {
                return NULL;
        }
#endif // HAVE_AVCODEC_ENCODE_VIDEO2

        return s->out[buffer_idx];
}

void libavcodec_compress_done(void *arg)
{
        struct libav_video_compress *s = (struct libav_video_compress *) arg;

        for(int i = 0; i < 2; ++i) {
#ifdef HAVE_AVCODEC_ENCODE_VIDEO2
                vf_free(s->out[i]);
                av_free_packet(&s->pkt[i]);
#else
                vf_free_data(s->out[i]);
#endif // HAVE_AVCODEC_ENCODE_VIDEO2
        }

        if(s->codec_ctx)
                avcodec_close(s->codec_ctx);
        if(s->in_frame) {
                av_freep(s->in_frame->data);
                av_free(s->in_frame);
        }
        av_free(s->codec_ctx);
        free(s->decoded);

        free(s);
}

