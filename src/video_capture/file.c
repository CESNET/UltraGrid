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
/**
 * @file
 * @todo
 * - selectable pixel format
 * - audio-only input
 * - regularly (every 30 s or so) write position in file (+ duration at the beginning)
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <assert.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>
#include <libavutil/avutil.h>
#include <libavformat/avformat.h>
#include <libavformat/version.h>
#include <libswscale/swscale.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>
#include <tv.h>

#include "audio/audio.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "libavcodec_common.h"
#include "messaging.h"
#include "module.h"
#include "playback.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "utils/time.h"
#include "utils/thread.h"
#include "video.h"
#include "video_capture.h"

#define MAGIC to_fourcc('u', 'g', 'l', 'f')
#define MOD_NAME "[File cap.] "

struct vidcap_state_lavf_decoder {
        struct module mod;
        char *src_filename;
        AVFormatContext *fmt_ctx;
        AVCodecContext *aud_ctx, *vid_ctx;
        struct SwsContext *sws_ctx;
        bool failed;
        bool loop;
        bool new_msg;
        bool no_decode;
        bool paused;
        bool use_audio;

        int video_stream_idx, audio_stream_idx;
        int64_t last_vid_pts;

        struct video_desc video_desc;

        struct video_frame *video_frame;
        struct audio_frame audio_frame;
        pthread_mutex_t audio_frame_lock;

        pthread_t thread_id;
        pthread_mutex_t lock;
        pthread_cond_t new_frame_ready;
        pthread_cond_t frame_consumed;
        pthread_cond_t paused_cv;
        struct timeval last_frame;

        bool should_exit;
};

static void vidcap_file_show_help() {
        color_out(0, "Usage:\n");
        color_out(COLOR_OUT_BOLD | COLOR_OUT_RED, "\t-t file:<name>");
        color_out(COLOR_OUT_BOLD, "[:loop][:nodecode]\n");
        color_out(0, "\t\twhere\n");
        color_out(COLOR_OUT_BOLD, "\tloop\n");
        color_out(0, "\t\tloop the playback\n");
        color_out(COLOR_OUT_BOLD, "\tnodecode\n");
        color_out(0, "\t\tdon't decompress the video (may not work because required data for correct decompess are in container or UG doesn't recognize the codec)\n");
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

        free(s->audio_frame.data);
        VIDEO_FRAME_DISPOSE(s->video_frame);

        pthread_mutex_destroy(&s->audio_frame_lock);
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->frame_consumed);
        pthread_cond_destroy(&s->new_frame_ready);
        pthread_cond_destroy(&s->paused_cv);
        free(s->src_filename);
        module_done(&s->mod);
        free(s);
}

static void vidcap_file_write_audio(struct vidcap_state_lavf_decoder *s, AVFrame * frame) {
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

        pthread_mutex_lock(&s->audio_frame_lock);
        if (av_sample_fmt_is_planar(s->aud_ctx->sample_fmt)) {
                int bps = av_get_bytes_per_sample(s->aud_ctx->sample_fmt);
                if (s->audio_frame.data_len + plane_count * bps * s->aud_ctx->frame_size > s->audio_frame.max_size) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Audio buffer overflow!\n");
                        pthread_mutex_unlock(&s->audio_frame_lock);
                        return;
                }
                for (int i = 0; i < plane_count; ++i) {
                        mux_channel(s->audio_frame.data + s->audio_frame.data_len, (char *) frame->data[i], bps, s->aud_ctx->frame_size * bps, plane_count, i, 1.0);
                }
                s->audio_frame.data_len += plane_count * bps * s->aud_ctx->frame_size;
        } else {
                int data_size = av_samples_get_buffer_size(NULL, s->audio_frame.ch_count,
                                s->aud_ctx->frame_size,
                                s->aud_ctx->sample_fmt, 1);
                append_audio_frame(&s->audio_frame, (char *) frame->data[0],
                                data_size);
        }
        pthread_mutex_unlock(&s->audio_frame_lock);
}

#define CHECK_FF(cmd, action_failed) do { int rc = cmd; if (rc < 0) { char buf[1024]; av_strerror(rc, buf, 1024); log_msg(LOG_LEVEL_ERROR, MOD_NAME #cmd ": %s\n", buf); action_failed} } while(0)
static void vidcap_file_process_messages(struct vidcap_state_lavf_decoder *s) {
        struct msg_universal *msg;
        while ((msg = (struct msg_universal *) check_message(&s->mod)) != NULL) {
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Message: \"%s\"\n", msg->text);
                if (strstr(msg->text, "seek ") != NULL) {
                        const char *count_str = msg->text + strlen("seek ");
                        int sec = atoi(count_str);
                        AVStream *st = s->fmt_ctx->streams[s->video_stream_idx];
                        AVRational tb = st->time_base;
                        CHECK_FF(avformat_seek_file(s->fmt_ctx, s->video_stream_idx, INT64_MIN, st->start_time + s->last_vid_pts + sec * tb.den / tb.num, INT64_MAX, AVSEEK_FLAG_FRAME), {});
                        char position[13], duration[13];
                        format_time_ms(s->last_vid_pts * tb.num * 1000 / tb.den  + sec * 1000, position);
                        format_time_ms(st->duration * tb.num * 1000 / tb.den, duration);
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Seeking to %s / %s\n", position, duration);
                } else if (strcmp(msg->text, "pause") == 0) {
                        s->paused = !s->paused;
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "%s\n", s->paused ? "paused" : "unpaused");
                } else if (strcmp(msg->text, "quit") == 0) {
                        exit_uv(0);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown message: %s\n", msg->text);
                        free_message((struct message *) msg, new_response(RESPONSE_BAD_REQUEST, "unknown message"));
                        continue;
                }

                free_message((struct message *) msg, new_response(RESPONSE_OK, NULL));
        }
}

#define FAIL_WORKER { pthread_mutex_lock(&s->lock); s->failed = true; pthread_mutex_unlock(&s->lock); pthread_cond_signal(&s->new_frame_ready); return NULL; }
static void *vidcap_file_worker(void *state) {
        set_thread_name(__func__);
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.size = 0;
        pkt.data = 0;
        while (!s->should_exit) {
                pthread_mutex_lock(&s->lock);
                if (s->new_msg) {
                        vidcap_file_process_messages(s);
                        s->new_msg = false;
                }
                if (s->paused) {
                        while (!s->should_exit && !s->new_msg) {
                                pthread_cond_wait(&s->paused_cv, &s->lock);
                        }
                        pthread_mutex_unlock(&s->lock);
                        if (s->should_exit) {
                                break;
                        } else { // new_msg -> process in next iteration
                                continue;
                        }
                }
                pthread_mutex_unlock(&s->lock);

                int ret = av_read_frame(s->fmt_ctx, &pkt);
                if (ret == AVERROR_EOF) {
                        if (s->loop) {
                                CHECK_FF(avformat_seek_file(s->fmt_ctx, -1, INT64_MIN, s->fmt_ctx->start_time, INT64_MAX, 0), FAIL_WORKER);
                                continue;
                        } else {
                                s->paused = true;
                                continue;
                        }
                }
                CHECK_FF(ret, FAIL_WORKER); // check the retval of av_read_frame for error other than EOF

                log_msg(LOG_LEVEL_DEBUG, MOD_NAME "received %s packet, ID %d, pos %" PRId64 ", size %d\n",
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
                        s->last_vid_pts = pkt.pts;
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
                                av_frame_free(&frame);
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

static bool vidcap_file_parse_fmt(struct vidcap_state_lavf_decoder *s, const char *fmt,
                bool *opportunistic_audio) {
        s->src_filename = strdup(fmt);
        assert(s->src_filename != NULL);
        char *tmp = s->src_filename, *item, *saveptr;
        while ((item = strtok_r(tmp, ":", &saveptr)) != NULL) {
                if (tmp != NULL) { // already stored in src_filename
                        tmp = NULL;
                        continue;
                }
                if (strcmp(item, "loop") == 0) {
                        s->loop = true;
                } else if (strcmp(item, "nodecode") == 0) {
                        s->no_decode = true;
                } else if (strcmp(item, "opportunistic_audio") == 0) {
                        *opportunistic_audio = true;
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

static void vidcap_file_new_message(struct module *mod) {
        struct vidcap_state_lavf_decoder *s = mod->priv_data;
        pthread_mutex_lock(&s->lock);
        s->new_msg = true;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->paused_cv);
}

static void vidcap_file_should_exit(void *state) {
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        pthread_mutex_lock(&s->lock);
        s->should_exit = true;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->new_frame_ready);
        pthread_cond_signal(&s->frame_consumed);
        pthread_cond_signal(&s->paused_cv);
}

#define CHECK(call) { int ret = call; if (ret != 0) abort(); }
static int vidcap_file_init(struct vidcap_params *params, void **state) {
        bool opportunistic_audio = false; // do not fail if audio requested but not found
        int rc = 0;
        char errbuf[1024] = "";
        if (strlen(vidcap_params_get_fmt(params)) == 0 ||
                        strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                vidcap_file_show_help();
                return strlen(vidcap_params_get_fmt(params)) == 0 ? VIDCAP_INIT_FAIL : VIDCAP_INIT_NOERR;
        }

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 12, 100)
        av_register_all();
#endif
#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        avcodec_register_all();
#endif

        struct vidcap_state_lavf_decoder *s = calloc(1, sizeof (struct vidcap_state_lavf_decoder));
        s->audio_stream_idx = -1;
        s->video_stream_idx = -1;
        CHECK(pthread_mutex_init(&s->audio_frame_lock, NULL));
        CHECK(pthread_mutex_init(&s->lock, NULL));
        CHECK(pthread_cond_init(&s->frame_consumed, NULL));
        CHECK(pthread_cond_init(&s->new_frame_ready, NULL));
        CHECK(pthread_cond_init(&s->paused_cv, NULL));
        module_init_default(&s->mod);
        s->mod.priv_magic = MAGIC;
        s->mod.cls = MODULE_CLASS_DATA;
        s->mod.priv_data = s;
        s->mod.new_message = vidcap_file_new_message;
        module_register(&s->mod, vidcap_params_get_parent(params));

        if (!vidcap_file_parse_fmt(s, vidcap_params_get_fmt(params), &opportunistic_audio)) {
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        }

        /* open input file, and allocate format context */
        if ((rc = avformat_open_input(&s->fmt_ctx, s->src_filename, NULL, NULL)) < 0) {
                snprintf(errbuf, sizeof errbuf, MOD_NAME "Could not open source file %s: ", s->src_filename);
        }

        /* retrieve stream information */
        if (rc >= 0 && (rc = avformat_find_stream_info(s->fmt_ctx, NULL)) < 0) {
                snprintf(errbuf, sizeof errbuf, MOD_NAME "Could not find stream information: \n");
        }

        if (rc < 0) {
                av_strerror(rc, errbuf + strlen(errbuf), sizeof errbuf - strlen(errbuf));
                log_msg(LOG_LEVEL_ERROR, "%s\n", errbuf);
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        }

        AVCodec *dec;
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                s->audio_stream_idx = av_find_best_stream(s->fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, &dec, 0);
                if (s->audio_stream_idx < 0 && !opportunistic_audio) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not find audio stream!\n");
                        vidcap_file_common_cleanup(s);
                        return VIDCAP_INIT_FAIL;
                }
                if (s->audio_stream_idx >= 0) {
                        s->aud_ctx = vidcap_file_open_dec_ctx(dec,
                                        s->fmt_ctx->streams[s->audio_stream_idx]);

                        if (s->aud_ctx == NULL) {
                                vidcap_file_common_cleanup(s);
                                return VIDCAP_INIT_FAIL;
                        }
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Input audio sample bps: %s\n",
                                        av_get_sample_fmt_name(s->aud_ctx->sample_fmt));
                        s->audio_frame.bps = av_get_bytes_per_sample(s->aud_ctx->sample_fmt);
                        s->audio_frame.sample_rate = s->aud_ctx->sample_rate;
                        s->audio_frame.ch_count = s->aud_ctx->channels;
                        s->audio_frame.max_size = s->audio_frame.bps * s->audio_frame.ch_count * s->audio_frame.sample_rate;
                        s->audio_frame.data = malloc(s->audio_frame.max_size);
                        s->use_audio = true;
                }
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
fprintf(stderr, "%d %d\n", s->video_desc.width, s->video_desc.height);
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
                s->video_desc.interlacing = PROGRESSIVE; /// @todo other modes
        }

        s->last_vid_pts = s->fmt_ctx->streams[s->video_stream_idx]->start_time;

        playback_register_keyboard_ctl(&s->mod);
        register_should_exit_callback(&s->mod, vidcap_file_should_exit, s);

        pthread_create(&s->thread_id, NULL, vidcap_file_worker, s);

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_file_done(void *state) {
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        assert(s->mod.priv_magic == MAGIC);

        vidcap_file_should_exit(s);

        pthread_join(s->thread_id, NULL);

        vidcap_file_common_cleanup(s);
}

static void vidcap_file_dispose_audio(struct audio_frame *f) {
        free(f->data);
        free(f);
}


static struct video_frame *vidcap_file_grab(void *state, struct audio_frame **audio) {
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        struct video_frame *out;

        assert(s->mod.priv_magic == MAGIC);
        *audio = NULL;
        pthread_mutex_lock(&s->lock);
        while (s->video_frame == NULL && !s->failed && !s->should_exit) {
                pthread_cond_wait(&s->new_frame_ready, &s->lock);
        }
        if (s->failed || s->should_exit) {
                pthread_mutex_unlock(&s->lock);
                return NULL;
        }
        out = s->video_frame;
        s->video_frame = NULL;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->frame_consumed);

        pthread_mutex_lock(&s->audio_frame_lock);
        *audio = audio_frame_copy(&s->audio_frame, false);
        (*audio)->dispose = vidcap_file_dispose_audio;
        s->audio_frame.data_len = 0;
        pthread_mutex_unlock(&s->audio_frame_lock);

        struct timeval t;
        do {
                gettimeofday(&t, NULL);
        } while (tv_diff(t, s->last_frame) < 1 / s->video_desc.fps);
        s->last_frame = t;

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
        false
};

REGISTER_MODULE(file, &vidcap_file_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

