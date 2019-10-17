/**
 * @file video_capture/file.c
 * @author Martin Pulec <pulec@cesnet.cz>
 *
 * Libavformat demuxer and decompress
 *
 * Inspired with demuxing_decoding.c (but replacing deprecated
 * avcodec_decode_audio/avcodec_decode_video).
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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

#include <assert.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <stdbool.h>
#include <stdint.h>
#include <tv.h>

#include "audio/audio.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "libavcodec_common.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "video.h"
#include "video_capture.h"

#define MAGIC to_fourcc('u', 'g', 'l', 'f')
#define MOD_NAME "[File cap.] "

struct vidcap_state_lavf_decoder {
        uint32_t magic;
        char *src_filename;
        AVFormatContext *fmt_ctx;
        AVCodecContext *aud_ctx, *vid_ctx;
        struct SwsContext *sws_ctx;
        bool use_audio;
        bool no_decode;

        int video_stream_idx, audio_stream_idx;

        struct video_desc video_desc;

        struct video_frame *video_frame;
        struct audio_frame audio_frame[2];
        int cur_aud_idx;

        pthread_t thread_id;
        pthread_mutex_t lock;
        pthread_cond_t new_frame_ready;
        pthread_cond_t frame_consumed;
        struct timeval last_frame;

        bool should_exit;
};

static void vidcap_file_show_help() {
        color_out(0, "Usage:\n");
        color_out(COLOR_OUT_BOLD | COLOR_OUT_RED, "-t file:<name>");
        color_out(COLOR_OUT_BOLD, "[:nodecode]\n");
}

static void vidcap_file_common_cleanup(struct vidcap_state_lavf_decoder *s) {
        if (s->sws_ctx) {
                sws_freeContext(s->sws_ctx);
        }
        if (s->vid_ctx) {
                avcodec_free_context(&s->vid_ctx);
        }
        if (s->aud_ctx) {
                avcodec_free_context(&s->aud_ctx);
        }
        if (s->fmt_ctx) {
                avformat_close_input(&s->fmt_ctx);
        }

        for (int i = 0; i < 2; i++) {
                free(s->audio_frame[i].data);
        }
        VIDEO_FRAME_DISPOSE(s->video_frame);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->new_frame_ready);
        pthread_cond_destroy(&s->frame_consumed);
        free(s->src_filename);
        free(s);
}

static void vidcap_file_write_audio(struct vidcap_state_lavf_decoder *s,
                AVFrame * frame) {
        int plane_count = av_sample_fmt_is_planar(s->aud_ctx->sample_fmt) ? s->aud_ctx->channels : 1;
        // transform from floats
        if (av_get_alt_sample_fmt(s->aud_ctx->sample_fmt, 0) == AV_SAMPLE_FMT_FLT) {
                for (int i = 0; i < plane_count; ++i) {
                        float2int((char *) frame->data[i], (char *) frame->data[i], s->aud_ctx->frame_size * 4);
                }
        } else if (av_get_alt_sample_fmt(s->aud_ctx->sample_fmt, 0) == AV_SAMPLE_FMT_DBL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Doubles not supported!\n");
                return;
        }

        if (av_sample_fmt_is_planar(s->aud_ctx->sample_fmt)) {
                int bps = av_get_bytes_per_sample(s->aud_ctx->sample_fmt);
                if (s->audio_frame[s->cur_aud_idx].data_len + plane_count * bps * s->aud_ctx->frame_size > s->audio_frame[s->cur_aud_idx].max_size) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Audio buffer overflow!\n");
                        return;
                }
                for (int i = 0; i < plane_count; ++i) {
                        mux_channel(s->audio_frame[s->cur_aud_idx].data + s->audio_frame[s->cur_aud_idx].data_len, (char *) frame->data[i], bps, s->aud_ctx->frame_size * bps, plane_count, i, 1.0);
                }
                s->audio_frame[s->cur_aud_idx].data_len += plane_count * bps * s->aud_ctx->frame_size;
        } else {
                int data_size = av_samples_get_buffer_size(NULL, s->audio_frame[s->cur_aud_idx].ch_count,
                                s->aud_ctx->frame_size,
                                s->aud_ctx->sample_fmt, 1);
                append_audio_frame(&s->audio_frame[s->cur_aud_idx], (char *) frame->data[0],
                                data_size);
        }
}

static void *vidcap_file_worker(void *state) {
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        int ret;
        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.size = 0;
        pkt.data = 0;
        while (!s->should_exit && av_read_frame(s->fmt_ctx, &pkt) >= 0) {
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "received %s packet, ID %d, pos %d, size %d\n",
                                av_get_media_type_string(
                                        s->fmt_ctx->streams[pkt.stream_index]->codecpar->codec_type),
                                pkt.stream_index, pkt.pos, pkt.size);

                if (pkt.stream_index == s->audio_stream_idx) {
                        ret = avcodec_send_packet(s->aud_ctx, &pkt);
                        if (ret < 0) {
                                print_decoder_error(MOD_NAME, ret);
                        }
                        AVFrame * frame = av_frame_alloc();
                        while (ret >= 0) {
                                ret = avcodec_receive_frame(s->aud_ctx, frame);
				if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
					av_frame_free(&frame);
					break; // inner loop
                                } else if (ret < 0) {
					print_decoder_error(MOD_NAME, ret);
					av_frame_free(&frame);
					break; // inner loop
				}
				/* if a frame has been decoded, output it */
                                vidcap_file_write_audio(s, frame);
                        }
                        av_frame_free(&frame);
                } else if (pkt.stream_index == s->video_stream_idx) {
                        struct video_frame *out;
                        if (s->no_decode) {
                                out = vf_alloc_desc(s->video_desc);
                                out->callbacks.data_deleter = vf_data_deleter;
                                out->callbacks.dispose = vf_free;
                                out->tiles[0].data_len = pkt.size;
                                out->tiles[0].data = malloc(pkt.size);
                                memcpy(out->tiles[0].data, pkt.data, pkt.size);
                        } else {
                                AVFrame * frame = av_frame_alloc();
                                int got_frame = 0;
                                ret = avcodec_send_packet(s->vid_ctx, &pkt);
                                if (ret == 0 || ret == AVERROR(EAGAIN)) {
                                        ret = avcodec_receive_frame(s->vid_ctx, frame);
                                        if (ret == 0) {
                                                got_frame = 1;
                                        }
                                }
                                if (ret != 0) {
                                        print_decoder_error(MOD_NAME, ret);
                                }

                                if (ret < 0 || !got_frame) {
                                        if (ret < 0) {
                                                fprintf(stderr, "Error decoding video frame (%s)\n", av_err2str(ret));
                                        }
                                        av_frame_free(&frame);
                                        continue;
                                }
                                out = vf_alloc_desc_data(s->video_desc);

                                /* copy decoded frame to destination buffer:
                                 * this is required since rawvideo expects non aligned data */
                                int video_dst_linesize[4] = { vc_get_linesize(out->tiles[0].width, out->color_spec) };
                                uint8_t *dst[4] = { (uint8_t *) out->tiles[0].data };
                                sws_scale(s->sws_ctx, (const uint8_t * const *) frame->data, frame->linesize, 0,
                                                frame->height, dst, video_dst_linesize);
                                out->callbacks.dispose = vf_free;
                        }
                        pthread_mutex_lock(&s->lock);
                        while (!s->should_exit && s->video_frame != NULL) {
                                pthread_cond_wait(&s->frame_consumed, &s->lock);
                        }
                        if (s->should_exit) {
                                VIDEO_FRAME_DISPOSE(out);
                                pthread_mutex_unlock(&s->lock);
                                av_packet_unref(&pkt);
                                return NULL;
                        }
                        s->video_frame = out;
                        pthread_mutex_unlock(&s->lock);
                        pthread_cond_signal(&s->new_frame_ready);
                }
                av_packet_unref(&pkt);
        }

        return NULL;
}

static bool vidcap_file_parse_fmt(struct vidcap_state_lavf_decoder *s, const char *fmt) {
        s->src_filename = strdup(fmt);
        char *tmp = s->src_filename, *item, *saveptr;
        while ((item = strtok_r(tmp, ":", &saveptr)) != NULL) {
                if (tmp != NULL) { // already stored in src_filename
                        tmp = NULL;
                        continue;
                }
                if (strcmp(item, "nodecode") == 0) {
                        s->no_decode = true;
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s\n", item);
                        return false;
                }
        }
        return true;
}

static AVCodecContext *vidcap_file_open_dec_ctx(AVCodec *dec, AVStream *st) {
        AVCodecContext *dec_ctx = avcodec_alloc_context3(dec);
        if (!dec_ctx) {
                return NULL;
        }
        /* Copy codec parameters from input stream to output codec context */
        if (avcodec_parameters_to_context(dec_ctx, st->codecpar) < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to copy parameters\n");
                avcodec_free_context(&dec_ctx);
                return NULL;
        }
        /* Init the decoders, with or without reference counting */
        AVDictionary *opts = NULL;
        av_dict_set(&opts, "refcounted_frames", "0", 0);
        if (avcodec_open2(dec_ctx, dec, &opts) < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to open codec\n");
                avcodec_free_context(&dec_ctx);
                return NULL;
        }
        return dec_ctx;
}

#define CHECK(call) { int ret = call; if (ret != 0) abort(); }
static int vidcap_file_init(struct vidcap_params *params, void **state) {
        if (strlen(vidcap_params_get_fmt(params)) == 0 ||
                        strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                vidcap_file_show_help();
                return strlen(vidcap_params_get_fmt(params)) == 0 ? VIDCAP_INIT_FAIL : VIDCAP_INIT_NOERR;
        }

        struct vidcap_state_lavf_decoder *s = calloc(1, sizeof (struct vidcap_state_lavf_decoder));
        s->magic = MAGIC;
        s->audio_stream_idx = -1;
        s->video_stream_idx = -1;
        CHECK(pthread_mutex_init(&s->lock, NULL));
        CHECK(pthread_cond_init(&s->new_frame_ready, NULL));
        CHECK(pthread_cond_init(&s->frame_consumed, NULL));

        if (!vidcap_file_parse_fmt(s, vidcap_params_get_fmt(params))) {
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        }

        /* open input file, and allocate format context */
        if (avformat_open_input(&s->fmt_ctx, s->src_filename, NULL, NULL) < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not open source file %s\n", s->src_filename);
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        }

        /* retrieve stream information */
        if (avformat_find_stream_info(s->fmt_ctx, NULL) < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not find stream information\n");
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        }

        AVCodec *dec;
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                s->audio_stream_idx = av_find_best_stream(s->fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
                if (s->audio_stream_idx < 0) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not find audio stream!\n");
                        vidcap_file_common_cleanup(s);
                        return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
                } else {
                        s->aud_ctx = vidcap_file_open_dec_ctx(dec,
                                        s->fmt_ctx->streams[s->audio_stream_idx]);

                        if (s->aud_ctx == NULL) {
                                vidcap_file_common_cleanup(s);
                                return VIDCAP_INIT_FAIL;
                        }
                        for (int i = 0; i < 2; i++) {
                                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Input audio sample bps: %s\n",
                                                av_get_sample_fmt_name(s->aud_ctx->sample_fmt));
                                s->audio_frame[i].bps = av_get_bytes_per_sample(s->aud_ctx->sample_fmt);
                                s->audio_frame[i].sample_rate = s->aud_ctx->sample_rate;
                                s->audio_frame[i].ch_count = s->aud_ctx->channels;
                                s->audio_frame[i].max_size = s->audio_frame[i].bps * s->audio_frame[i].ch_count * s->audio_frame[i].sample_rate;
                                s->audio_frame[i].data = malloc(s->audio_frame[i].max_size);
                        }
                }

                s->use_audio = true;
        }

        s->video_stream_idx = av_find_best_stream(s->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &dec, 0);
        if (s->video_stream_idx < 0) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "No video stream found!\n");
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        } else {
                AVStream *st = s->fmt_ctx->streams[s->video_stream_idx];
                s->video_desc.width = st->codecpar->width;
                s->video_desc.height = st->codecpar->height;
                s->video_desc.fps = (double) st->r_frame_rate.num / st->r_frame_rate.den;
                s->video_desc.tile_count = 1;

                if (s->no_decode) {
                        s->video_desc.color_spec =
                                get_av_to_ug_codec(s->fmt_ctx->streams[s->video_stream_idx]->codecpar->codec_id);
                        if (s->video_desc.color_spec == VIDEO_CODEC_NONE) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported codec %s.\n",
                                                avcodec_get_name(s->fmt_ctx->streams[s->video_stream_idx]->codecpar->codec_id));
                                vidcap_file_common_cleanup(s);
                                return VIDCAP_INIT_FAIL;
                        }
                } else {
                        s->video_desc.color_spec = UYVY;
                        s->vid_ctx = vidcap_file_open_dec_ctx(dec, st);
                        if (!s->vid_ctx) {
                                vidcap_file_common_cleanup(s);
                                return VIDCAP_INIT_FAIL;
                        }

                        s->sws_ctx = sws_getContext(s->video_desc.width, s->video_desc.height, s->vid_ctx->pix_fmt,
                                        s->video_desc.width, s->video_desc.height, get_ug_to_av_pixfmt(s->video_desc.color_spec),
                                        0, NULL, NULL, NULL);
                }
                // todo s->video_desc.interlacing
        }

        pthread_create(&s->thread_id, NULL, vidcap_file_worker, s);

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_file_done(void *state) {
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        assert(s->magic == MAGIC);

        pthread_mutex_lock(&s->lock);
        s->should_exit = true;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->frame_consumed);

        pthread_join(s->thread_id, NULL);

        vidcap_file_common_cleanup(s);
}

static struct video_frame *vidcap_file_grab(void *state, struct audio_frame **audio) {
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        struct video_frame *out;

        assert(s->magic == MAGIC);
        *audio = NULL;
        pthread_mutex_lock(&s->lock);
        while (s->video_frame == NULL) {
                pthread_cond_wait(&s->new_frame_ready, &s->lock);
        }
        out = s->video_frame;
        s->video_frame = NULL;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->frame_consumed);

        struct timeval t;
        do {
                gettimeofday(&t, NULL);
        } while (tv_diff(t, s->last_frame) < 1 / s->video_desc.fps);
        s->last_frame = t;

        *audio = &s->audio_frame[s->cur_aud_idx];
        s->cur_aud_idx = (s->cur_aud_idx + 1) % 2;
        s->audio_frame[s->cur_aud_idx].data_len = 0;

        return out;
}

static struct vidcap_type *vidcap_file_probe(bool verbose, void (**deleter)(void *)) {
        UNUSED(verbose);
        *deleter = free;
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name = "file";
                vt->description = "Input file playback";
        }
        return vt;
}

static const struct video_capture_info vidcap_file_info = {
        vidcap_file_probe,
        vidcap_file_init,
        vidcap_file_done,
        vidcap_file_grab,
};

REGISTER_MODULE(file, &vidcap_file_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

