/**
 * @file video_capture/file.c
 * @author Martin Pulec <pulec@cesnet.cz>
 *
 * Libavformat demuxer and decompress
 */
/*
 * Copyright (c) 2019-2025 CESNET
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
 * Inspired by and rewritten from FFmpeg example demuxing_decoding.c.
 *
 * For testing, it is advisable to use:
 * 1. https://archive.org/download/ElephantsDream/ed_hd.avi
 * 2. a 59.94i file - eg. converted from the above with:
 *   `ffmpeg -i ed_hd.avi -vf fps=30000/1001 -c:v libx264 -flags +ildct ed.mp4`
 *
 * and test things like seek and loop
 *
 * @todo
 * - audio-only input
 */

#include <assert.h>
#include <inttypes.h>
#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>
#include <libavformat/avformat.h>
#include <libavformat/version.h>
#include <libavutil/avutil.h>
#include <libswscale/swscale.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>
#include <tv.h>

#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "libavcodec/from_lavc_vid_conv.h"
#include "libavcodec/lavc_common.h"
#include "messaging.h"
#include "module.h"
#include "pixfmt_conv.h"
#include "playback.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/fs.h"
#include "utils/list.h"
#include "utils/macros.h"
#include "utils/math.h"
#include "utils/ring_buffer.h"
#include "utils/thread.h"
#include "utils/time.h"
#include "video.h"
#include "video_capture.h"

static const double AUDIO_RATIO = 1.05; ///< at this ratio the audio frame can be longer than the video frame
enum {
        AUD_BUF_LEN_SEC = 60,
        FILE_DEFAULT_QUEUE_LEN = 20,
};
#define MAGIC to_fourcc('u', 'g', 'l', 'f')
#define MOD_NAME "[File cap.] "

struct vidcap_state_lavf_decoder {
        uint32_t magic;
        struct module mod;
        char *src_filename;
        AVFormatContext *fmt_ctx;
        AVCodecContext *aud_ctx, *vid_ctx;
        int thread_count;
        int thread_type;

        struct SwsContext *sws_ctx;
        av_to_uv_convert_t *conv_uv;

        bool failed;
        bool loop;
        bool new_msg;
        bool no_decode;
        codec_t convert_to;
        bool paused;
        bool ended;
        int seek_sec;

        int video_stream_idx, audio_stream_idx;
        int64_t last_vid_pts; ///< last played PTS, if PTS == PTS_NO_VALUE, DTS is stored instead

        struct video_desc video_desc;
        struct audio_desc audio_desc;

        struct simple_linked_list *video_frame_queue;
        struct simple_linked_list *vid_frm_noaud; // auxiliary queue for worker
        int max_queue_len;
        struct ring_buffer *audio_data;
        int64_t audio_start_ts;
        int64_t audio_end_ts;
        pthread_mutex_t audio_frame_lock;

        pthread_t thread_id;
        pthread_mutex_t lock;
        pthread_cond_t new_frame_ready;
        pthread_cond_t frame_consumed;
        struct timeval last_frame;
        struct timeval last_stream_stat;

        bool should_exit;

        long long audio_frames;
        long long video_frames;
};

static void flush_captured_data(struct vidcap_state_lavf_decoder *s);

static void vidcap_file_show_help(bool full) {
        color_printf("Usage:\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-t file:<name>" TERM_FG_RESET "[:loop][:nodecode][:codec=<c>][:seek=<sec>]%s\n" TERM_RESET,
                        full ? "[:opportunistic_audio][:queue=<len>][:threads=<n>[FS]]" : "");
        color_printf("where\n");
        color_printf(TERM_BOLD "\tloop\n" TERM_RESET);
        color_printf("\t\tloop the playback\n");
        color_printf(TERM_BOLD "\tnodecode\n" TERM_RESET);
        color_printf("\t\tdon't decompress the video (may not work because required data for correct decompess are in container or UG doesn't recognize the codec)\n");
        color_printf(TERM_BOLD "\tcodec\n" TERM_RESET);
        color_printf("\t\tcodec to decode to\n");
        if (full) {
                color_printf(TERM_BOLD "\topportunistic_audio\n" TERM_RESET);
                color_printf("\t\tgrab audio if not present but do not fail if not\n");
                color_printf(TERM_BOLD "\tqueue\n" TERM_RESET);
                color_printf("\t\tmax queue len (default: %d), increasing may help if video stutters\n", FILE_DEFAULT_QUEUE_LEN);
                color_printf(TERM_BOLD "\tthreads\n" TERM_RESET);
                color_printf("\t\tnumber of threads (0 is default), 'S' and/or 'F' to use slice/frame threads, use at least one flag\n");
        } else {
                color_printf("\n(use \":fullhelp\" to see all available options)\n");
        }
}

// s->lock must be held when calling this function
static void flush_captured_data(struct vidcap_state_lavf_decoder *s) {
        struct video_frame *f = NULL;
        while ((f = simple_linked_list_pop(s->video_frame_queue)) != NULL) {
                VIDEO_FRAME_DISPOSE(f);
        }
        while ((f = simple_linked_list_pop(s->vid_frm_noaud)) != NULL) {
                VIDEO_FRAME_DISPOSE(f);
        }
        if (s->audio_data) {
                ring_buffer_flush(s->audio_data);
        }
        if (s->vid_ctx) {
                avcodec_flush_buffers(s->vid_ctx);
        }
        if (s->aud_ctx) {
                avcodec_flush_buffers(s->aud_ctx);
        }
        s->audio_end_ts = AV_NOPTS_VALUE;
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

        av_to_uv_conversion_destroy(&s->conv_uv);

        flush_captured_data(s);
        ring_buffer_destroy(s->audio_data);

        pthread_mutex_destroy(&s->audio_frame_lock);
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->frame_consumed);
        pthread_cond_destroy(&s->new_frame_ready);
        free(s->src_filename);
        module_done(&s->mod);
        simple_linked_list_destroy(s->video_frame_queue);
        simple_linked_list_destroy(s->vid_frm_noaud);
        free(s);
}

static void vidcap_file_write_audio(struct vidcap_state_lavf_decoder *s, AVFrame * frame) {
        const int plane_count = av_sample_fmt_is_planar(s->aud_ctx->sample_fmt)
                                    ? s->audio_desc.ch_count
                                    : 1;
        // transform from floats
        if (av_get_alt_sample_fmt(s->aud_ctx->sample_fmt, 0) == AV_SAMPLE_FMT_FLT) {
                for (int i = 0; i < plane_count; ++i) {
                        float2int((char *) frame->data[i], (char *) frame->data[i], frame->nb_samples * 4);
                }
        } else if (av_get_alt_sample_fmt(s->aud_ctx->sample_fmt, 0) == AV_SAMPLE_FMT_DBL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Doubles not supported!\n");
                return;
        }

        pthread_mutex_lock(&s->audio_frame_lock);
        if (ring_get_current_size(s->audio_data) == 0) {
                s->audio_start_ts = frame->pts;
        }
        if (av_sample_fmt_is_planar(s->aud_ctx->sample_fmt)) {
                int bps = av_get_bytes_per_sample(s->aud_ctx->sample_fmt);
                char tmp[plane_count * bps * frame->nb_samples];
                for (int i = 0; i < plane_count; ++i) {
                        mux_channel(tmp, (char *)frame->data[i], bps,
                                    frame->nb_samples * bps, plane_count,
                                    i, 1.0);
                }
                ring_buffer_write(s->audio_data, tmp, sizeof tmp);
        } else {
                if (plane_count == s->audio_desc.ch_count) {
                        int data_size = av_samples_get_buffer_size(NULL, s->audio_desc.ch_count,
                                        frame->nb_samples,
                                        s->aud_ctx->sample_fmt, 1);
                        if (data_size < 0) {
                                print_libav_error(LOG_LEVEL_WARNING, MOD_NAME " av_samples_get_buffer_size", data_size);
                        } else {
                                ring_buffer_write(s->audio_data, (char *)frame->data[0],
                                                  data_size);
                        }
                } else {
                        int src_len = s->audio_desc.bps * plane_count;
                        int dst_len =
                            s->audio_desc.bps * s->audio_desc.ch_count;
                        for (int i = 0; i < frame->nb_samples; ++i) {
                                ring_buffer_write(s->audio_data,
                                                  (char *)frame->data[0] +
                                                      i * src_len,
                                                  dst_len);
                        }
                }
        }
        pthread_mutex_unlock(&s->audio_frame_lock);
}

static bool have_audio_for_video(struct vidcap_state_lavf_decoder *s, int64_t vid_frm_ts, int64_t vid_frm_dur) {
        if (simple_linked_list_size(s->vid_frm_noaud) > 300) {
                log_msg(LOG_LEVEL_WARNING, "More than 300 video frames cached, "
                                           "giving up!\n");
                s->audio_end_ts = INT64_MAX / INT_MAX; // allow int mult
        }
        AVRational atb = s->fmt_ctx->streams[s->audio_stream_idx]->time_base;
        AVRational vtb = s->fmt_ctx->streams[s->video_stream_idx]->time_base;
        // Note the 3/4 of the frame was chosen because it is safely more
        // than 1/2 without need to tackle with integer division and/or A/V
        // frame alignment. The remainder (if there will be any) would be
        // dropped with next frame if it was greater than 1/2 (@sa get_audio).
        return s->audio_end_ts * atb.num / atb.den >=
               vid_frm_ts * vtb.num / vtb.den + // at least 3/4 of audio
                   (vid_frm_dur * vtb.num * 3 / vtb.den / 4);
}

static void vidcap_file_process_audio_pkt(struct vidcap_state_lavf_decoder *s,
                                          AVPacket *pkt, AVFrame *frame) {
        int ret = avcodec_send_packet(s->aud_ctx, pkt);
        if (ret < 0) {
                print_decoder_error(MOD_NAME, ret);
                return;
        }
        while (ret >= 0) {
                ret = avcodec_receive_frame(s->aud_ctx, frame);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                }
                if (ret < 0) {
                        print_decoder_error(MOD_NAME, ret);
                        break;
                }
                /* if a frame has been decoded, output it */
                s->audio_end_ts = pkt->pts + pkt->duration;
                vidcap_file_write_audio(s, frame);
        }
        // try to process decoded video frames that didn't have corresponding
        // audio when decompressed
        struct video_frame *vid_frm = simple_linked_list_pop(s->vid_frm_noaud);
        while (vid_frm != NULL) {
                if (!have_audio_for_video(s, vid_frm->seq, vid_frm->duration)) {
                        simple_linked_list_prepend(s->vid_frm_noaud, vid_frm);
                        break;
                }
                pthread_mutex_lock(&s->lock);
                simple_linked_list_append(s->video_frame_queue, vid_frm);
                pthread_mutex_unlock(&s->lock);
                pthread_cond_signal(&s->new_frame_ready);
                vid_frm = simple_linked_list_pop(s->vid_frm_noaud);
        }
}

static const char *get_current_position_str(struct vidcap_state_lavf_decoder *s)
{
        AVStream *st = s->fmt_ctx->streams[s->video_stream_idx];
        AVRational tb = st->time_base;
        static _Thread_local char position[12 * 2 + 3 + 1];
        format_time_ms(s->last_vid_pts * tb.num * 1000 / tb.den, position);
        strncat(position, " / ", sizeof position - strlen(position) - 1);
        format_time_ms(st->duration * tb.num * 1000 / tb.den,
                       position + strlen(position));
        return position;
}

static void print_current_pos(struct vidcap_state_lavf_decoder *s,
                              struct timeval t)
{
        if (tv_diff(t, s->last_stream_stat) < 30) {
                return;
        }
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Current position: %s\n",
                get_current_position_str(s));
        s->last_stream_stat = t;
}

#define CHECK_FF(cmd, action_failed) \
        do { \
                int rc = cmd; \
                if (rc < 0) { \
                        char buf[1024]; \
                        av_strerror(rc, buf, 1024); \
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME #cmd ": %s\n", buf); \
                        action_failed; \
                } \
        } while (0)
// s->lock must be held when calling this function
static void vidcap_file_process_messages(struct vidcap_state_lavf_decoder *s) {
        struct msg_universal *msg;
        while ((msg = (struct msg_universal *) check_message(&s->mod)) != NULL) {
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Message: \"%s\"\n", msg->text);
                if (strstr(msg->text, "seek ") != NULL) {
                        const char *count_str = msg->text + strlen("seek ");
                        char *endptr = NULL;
                        double sec = strtol(count_str, &endptr, 0);
                        if (endptr[0] != 's') {
                                sec /= s->video_desc.fps;
                        }
                        AVStream *st = s->fmt_ctx->streams[s->video_stream_idx];
                        AVRational tb = st->time_base;
                        s->last_vid_pts =
                            MAX(s->last_vid_pts + (sec * tb.den) / tb.num,
                                st->start_time);
                        CHECK_FF(
                            avformat_seek_file(s->fmt_ctx, s->video_stream_idx,
                                               INT64_MIN, s->last_vid_pts,
                                               INT64_MAX, AVSEEK_FLAG_FRAME),
                            {});
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Seeking to %s\n",
                                get_current_position_str(s));
                        flush_captured_data(s);
                        s->ended = false;
                } else if (strcmp(msg->text, "pause") == 0) {
                        s->paused = !s->paused;
                        pthread_cond_signal(&s->new_frame_ready);
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

static struct video_frame *process_video_pkt(struct vidcap_state_lavf_decoder *s,
                              AVPacket *pkt, AVFrame *frame) {
        s->last_vid_pts = pkt->pts == AV_NOPTS_VALUE ? pkt->dts : pkt->pts;
        if (s->no_decode) {
                struct video_frame *out = vf_alloc_desc(s->video_desc);
                out->callbacks.data_deleter = vf_data_deleter;
                out->callbacks.dispose = vf_free;
                out->tiles[0].data_len = pkt->size;
                out->tiles[0].data = malloc(pkt->size);
                memcpy(out->tiles[0].data, pkt->data, pkt->size);
                return out;
        }
        time_ns_t t0 = get_time_in_ns();
        int ret = avcodec_send_packet(s->vid_ctx, pkt);
        if (ret != 0 && ret != AVERROR(EAGAIN)) {
                print_decoder_error(MOD_NAME "send - ", ret);
                return NULL;
        }
        ret = avcodec_receive_frame(s->vid_ctx, frame);
        log_msg(LOG_LEVEL_DEBUG,
                MOD_NAME "Video decompressing %c frame (pts %" PRId64 ") "
                         "duration: %f s\n",
                av_get_picture_type_char(frame->pict_type), frame->pts,
                (get_time_in_ns() - t0) / NS_IN_SEC_DBL);

        if (ret < 0) {
                print_decoder_error(MOD_NAME "recv - ", ret);
                return NULL;
        }
        struct video_frame *out = vf_alloc_desc_data(s->video_desc);
        out->flags |= TIMESTAMP_VALID;

        /* copy decoded frame to destination buffer:
         * this is required since rawvideo expects non aligned data */
        int video_dst_linesize[4] = {
            vc_get_linesize(out->tiles[0].width, out->color_spec)};
        uint8_t *dst[4] = {(uint8_t *)out->tiles[0].data};
        if (s->conv_uv) {
                int rgb_shift[] = DEFAULT_RGB_SHIFT_INIT;
                av_to_uv_convert(s->conv_uv, out->tiles[0].data, frame,
                                 video_dst_linesize[0], rgb_shift);
        } else {
                sws_scale(s->sws_ctx, (const uint8_t *const *)frame->data,
                          frame->linesize, 0, frame->height, dst,
                          video_dst_linesize);
        }
        out->seq = frame->pts < 0 ? UINT32_MAX : MIN(frame->pts, UINT32_MAX);
#if LIBAVUTIL_VERSION_INT >= AV_VERSION_INT(57, 30, 100)
        out->duration = frame->duration;
#else
        out->duration = frame->pkt_duration;
#endif
        out->callbacks.dispose = vf_free;
        return out;
}

static void print_packet_info(const AVPacket *pkt, const AVStream *st) {
        AVRational tb = st->time_base;

        char pts_val[128] = "NO VALUE";
        if (pkt->pts != AV_NOPTS_VALUE) {
                snprintf(pts_val, sizeof pts_val, "%" PRId64, pkt->pts);
        }
        char dts_val[128] = "NO VALUE";
        if (pkt->dts != AV_NOPTS_VALUE) {
                snprintf(dts_val, sizeof dts_val, "%" PRId64, pkt->dts);
        }
        log_msg(LOG_LEVEL_DEBUG,
                MOD_NAME "rcv %s pkt, ID %d, pos %.2f s (pts %s, dts "
                         "%s, dur %" PRId64 ", tb %d/%d), sz %d B\n",
                av_get_media_type_string(st->codecpar->codec_type),
                pkt->stream_index,
                (double)(pkt->pts == AV_NOPTS_VALUE ? pkt->dts : pkt->pts) *
                    tb.num / tb.den,
                pts_val, dts_val, pkt->duration, tb.num, tb.den, pkt->size);
}

static void
rewind_file(struct vidcap_state_lavf_decoder *s)
{
        bool avseek_failed = false;
        CHECK_FF(avformat_seek_file(s->fmt_ctx, -1, INT64_MIN,
                                    s->fmt_ctx->start_time, INT64_MAX, 0),
                 avseek_failed = true);
        const bool mjpeg = s->vid_ctx->codec_id == AV_CODEC_ID_MJPEG &&
                           s->fmt_ctx->ctx_flags & AVFMTCTX_NOHEADER;
        if (avseek_failed || mjpeg) {
                // handle single JPEG loop, inspired by libavformat's
                // seek_frame_generic because img_read_seek
                // (AVInputFormat::read_seek) doesn't do the job - seeking is
                // inmplemeted just in img2dec if VideoDemuxData::loop == 1
                // used also for AnnexB HEVC stream (avformat_seek_file fails)
                CHECK_FF(
                    avio_seek(s->fmt_ctx->pb, s->video_stream_idx, SEEK_SET),
                    {});
        }
        pthread_mutex_lock(&s->lock);
        flush_captured_data(s);
        pthread_mutex_unlock(&s->lock);
}

#define FAIL_WORKER { pthread_mutex_lock(&s->lock); s->failed = true; pthread_mutex_unlock(&s->lock); pthread_cond_signal(&s->new_frame_ready); return NULL; }
static void *vidcap_file_worker(void *state) {
        set_thread_name(__func__);
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        AVPacket *pkt = av_packet_alloc();
        AVFrame *frame = av_frame_alloc();

        pkt->size = 0;
        pkt->data = 0;
        while (true) {
                pthread_mutex_lock(&s->lock);
                while (!s->should_exit && !s->new_msg &&
                       (simple_linked_list_size(s->video_frame_queue) >
                           s->max_queue_len || s->ended)) {
                        pthread_cond_wait(&s->frame_consumed, &s->lock);
                }
                if (s->should_exit) {
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                if (s->new_msg) {
                        vidcap_file_process_messages(s);
                        s->new_msg = false;
                        pthread_mutex_unlock(&s->lock);
                        continue;
                }
                pthread_mutex_unlock(&s->lock);

                av_packet_unref(pkt);
                int ret = av_read_frame(s->fmt_ctx, pkt);
                if (ret == AVERROR_EOF) {
                        if (s->loop) {
                                rewind_file(s);
                                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Rewinding the file.\n");
                                continue;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Playback ended.\n");
                                s->ended = true;
                                continue;
                        }
                }
                CHECK_FF(ret, FAIL_WORKER); // check the retval of av_read_frame for error other than EOF

                if (log_level >= LOG_LEVEL_DEBUG) {
                        print_packet_info(
                            pkt, s->fmt_ctx->streams[pkt->stream_index]);
                }

                if (pkt->stream_index == s->audio_stream_idx) {
                        vidcap_file_process_audio_pkt(s, pkt, frame);
                } else if (pkt->stream_index == s->video_stream_idx) {
                        struct video_frame *out =
                            process_video_pkt(s, pkt, frame);
                        if (!out) {
                                continue;
                        }
                        if (s->audio_stream_idx != -1 && out->seq != UINT32_MAX) {
                                if (!have_audio_for_video(s, out->seq, out->duration)) {
                                        simple_linked_list_append(s->vid_frm_noaud,
                                                                  out);
                                        continue;
                                }
                        }
                        pthread_mutex_lock(&s->lock);
                        simple_linked_list_append(s->video_frame_queue, out);
                        pthread_mutex_unlock(&s->lock);
                        pthread_cond_signal(&s->new_frame_ready);
                }
        }

        av_packet_free(&pkt);
        av_frame_free(&frame);

        return NULL;
}

static bool vidcap_file_parse_fmt(struct vidcap_state_lavf_decoder *s, const char *fmt,
                bool *opportunistic_audio) {
        char fmt_cpy[MAX_PATH_SIZE] = "";
        strncpy(fmt_cpy, fmt, sizeof fmt_cpy - 1);
        char *tmp     = fmt_cpy;
        char *item    = NULL;
        char *saveptr = NULL;
        while ((item = strtok_r(tmp, ":", &saveptr)) != NULL) {
                if (tmp != NULL) { // first item
                        s->src_filename = strdup_path_with_expansion(
                            IS_KEY_PREFIX(item, "name") ? strchr(item, '=') + 1
                                                        : item);
                        tmp = NULL;
                        continue;
                }
                if (strcmp(item, "loop") == 0) {
                        s->loop = true;
                } else if (strcmp(item, "nodecode") == 0) {
                        s->no_decode = true;
                } else if (strcmp(item, "opportunistic_audio") == 0) {
                        *opportunistic_audio = true;
                } else if (strncmp(item, "codec=", strlen("codec=")) == 0) {
                        char *codec_name = item + strlen("codec=");
                        if ((s->convert_to = get_codec_from_name(codec_name)) == VIDEO_CODEC_NONE) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown codec: %s\n", codec_name);
                                return false;
                        }
                } else if (strncmp(item, "queue=", strlen("queue=")) == 0) {
                        s->max_queue_len = atoi(item + strlen("queue="));
                } else if (strncmp(item, "threads=", strlen("threads=")) == 0) {
                        char *endptr = NULL;
                        long count = strtol(item + strlen("threads="), &endptr, 0);
                        s->thread_count = CLAMP(count, 0, INT_MAX);
                        s->thread_type = strchr(endptr, 'F') != NULL ? FF_THREAD_FRAME : 0;
                        s->thread_type |= strchr(endptr, 'S') != NULL ? FF_THREAD_SLICE : 0;
                } else if (strstr(item, "seek=") == item) {
                        s->seek_sec = atoi(strchr(item, '=') + 1);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s\n", item);
                        return false;
                }
        }
        return true;
}

static AVCodecContext *vidcap_file_open_dec_ctx(const AVCodec *dec, AVStream *st, int thread_count, int thread_type) {
        AVCodecContext *dec_ctx = avcodec_alloc_context3(dec);
        if (!dec_ctx) {
                return NULL;
        }
        dec_ctx->thread_count = thread_count;
        dec_ctx->thread_type = thread_type;

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
        pthread_cond_signal(&s->frame_consumed);
}

static void vidcap_file_should_exit(void *state) {
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        pthread_mutex_lock(&s->lock);
        s->should_exit = true;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->new_frame_ready);
        pthread_cond_signal(&s->frame_consumed);
}

static void seek_start(struct vidcap_state_lavf_decoder *s) {
        if (s->seek_sec <= 0) {
                return;
        }
        AVStream *st = s->fmt_ctx->streams[s->video_stream_idx];
        AVRational tb = st->time_base;
        int64_t ts = st->start_time + (int64_t) s->seek_sec * tb.den / tb.num;
        CHECK_FF(avformat_seek_file(s->fmt_ctx, s->video_stream_idx, INT64_MIN,
                                    ts, INT64_MAX, AVSEEK_FLAG_FRAME),
                 {});
}

static enum interlacing_t get_field_order(enum AVFieldOrder av_fo) {
        switch (av_fo) {
        case AV_FIELD_UNKNOWN:
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "Unknown field order, using progressive.\n");
                return PROGRESSIVE;
        case AV_FIELD_PROGRESSIVE:
                return PROGRESSIVE;
        case AV_FIELD_TT:
        case AV_FIELD_BT:
                return INTERLACED_MERGED;
        default:
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "Potentially unsupported field order (lower-"
                                 "field first): %d! Using progressive.\n",
                        (int)av_fo);
                return PROGRESSIVE;
        }
}

static bool setup_video(struct vidcap_state_lavf_decoder *s) {
        AVStream *st = s->fmt_ctx->streams[s->video_stream_idx];
        s->video_desc.width = st->codecpar->width;
        s->video_desc.height = st->codecpar->height;
        s->video_desc.fps =
            (double) st->avg_frame_rate.num / st->avg_frame_rate.den;
        s->video_desc.tile_count = 1;
        if (s->no_decode) {
                s->video_desc.color_spec =
                    get_av_to_ug_codec(st->codecpar->codec_id);
                if (s->video_desc.color_spec == VIDEO_CODEC_NONE) {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "Unsupported codec %s.\n",
                                avcodec_get_name(st->codecpar->codec_id));
                        return false;
                }
                s->video_desc.interlacing = PROGRESSIVE;
                return true;
        }
        const AVCodec *dec = avcodec_find_decoder(st->codecpar->codec_id);
        s->vid_ctx =
            vidcap_file_open_dec_ctx(dec, st, s->thread_count, s->thread_type);
        if (!s->vid_ctx) {
                return false;
        }
        s->video_desc.interlacing = get_field_order(s->vid_ctx->field_order);

        enum AVPixelFormat suggested[] = {s->vid_ctx->pix_fmt, AV_PIX_FMT_NONE};
        s->video_desc.color_spec = IF_NOT_NULL_ELSE(
            s->convert_to, get_best_ug_codec_to_av(suggested, false));
        if (s->video_desc.color_spec == VIDEO_CODEC_NONE) {
                s->video_desc.color_spec =
                    UYVY; // fallback, swscale will perhaps be used
        }
        s->conv_uv = get_av_to_uv_conversion(s->vid_ctx->pix_fmt,
                                             s->video_desc.color_spec);
        if (s->conv_uv) {
                return true;
        }
        // else swscale needed
        enum AVPixelFormat target_pixfmt =
            get_ug_to_av_pixfmt(s->video_desc.color_spec);
        if (target_pixfmt == AV_PIX_FMT_NONE) {
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "Cannot find suitable AVPixelFormat "
                                 "for swscale conversion!\n");
                return false;
        }
        s->sws_ctx = sws_getContext(s->video_desc.width, s->video_desc.height,
                                    s->vid_ctx->pix_fmt, s->video_desc.width,
                                    s->video_desc.height, target_pixfmt, 0,
                                    NULL, NULL, NULL);
        if (s->sws_ctx == NULL) {
                log_msg(LOG_LEVEL_ERROR,
                        MOD_NAME "Cannot find neither UltraGrid nor "
                                 "swscale conversion!\n");
                return false;
        }
        return true;
}

static int get_ach_count(int file_channels) {
        if (audio_capture_channels == 0) {
                return file_channels;
        }
        if ((int)audio_capture_channels > file_channels) {
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "Requested %d channels, file "
                                 "contains only %d!\n",
                        audio_capture_channels, file_channels);
                return file_channels;
        }
        return audio_capture_channels;
}

#define CHECK(call) { int ret = call; if (ret != 0) abort(); }
static int vidcap_file_init(struct vidcap_params *params, void **state) {
        bool opportunistic_audio = false; // do not fail if audio requested but not found
        int rc = 0;
        char errbuf[1024] = "";
        bool fullhelp = strcmp(vidcap_params_get_fmt(params), "fullhelp") == 0;
        if (strlen(vidcap_params_get_fmt(params)) == 0 ||
                        strcmp(vidcap_params_get_fmt(params), "help") == 0 || fullhelp) {
                vidcap_file_show_help(fullhelp);
                return strlen(vidcap_params_get_fmt(params)) == 0 ? VIDCAP_INIT_FAIL : VIDCAP_INIT_NOERR;
        }

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 12, 100)
        av_register_all();
#endif
#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        avcodec_register_all();
#endif

        struct vidcap_state_lavf_decoder *s = calloc(1, sizeof (struct vidcap_state_lavf_decoder));
        s->magic = MAGIC;
        s->video_frame_queue = simple_linked_list_init();
        s->vid_frm_noaud = simple_linked_list_init();
        s->audio_stream_idx = -1;
        s->video_stream_idx = -1;
        s->audio_end_ts = AV_NOPTS_VALUE;
        s->max_queue_len = FILE_DEFAULT_QUEUE_LEN;
        s->thread_count = 0; // means auto for most codecs
        s->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
        CHECK(pthread_mutex_init(&s->audio_frame_lock, NULL));
        CHECK(pthread_mutex_init(&s->lock, NULL));
        CHECK(pthread_cond_init(&s->frame_consumed, NULL));
        CHECK(pthread_cond_init(&s->new_frame_ready, NULL));
        module_init_default(&s->mod);
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
                snprintf(errbuf, sizeof errbuf, MOD_NAME "Could not open source file %s", s->src_filename);
        }

        /* retrieve stream information */
        if (rc >= 0 && (rc = avformat_find_stream_info(s->fmt_ctx, NULL)) < 0) {
                snprintf(errbuf, sizeof errbuf, MOD_NAME "Could not find stream information");
        }

        if (rc < 0) {
                print_libav_error(LOG_LEVEL_ERROR, errbuf, rc);
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        }

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                const AVCodec *dec = NULL;
                s->audio_stream_idx = av_find_best_stream(
                    s->fmt_ctx, AVMEDIA_TYPE_AUDIO, -1, -1, (void *)&dec, 0);
                if (s->audio_stream_idx < 0 && !opportunistic_audio) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not find audio stream!\n");
                        vidcap_file_common_cleanup(s);
                        return VIDCAP_INIT_FAIL;
                }
                if (s->audio_stream_idx >= 0) {
                        s->aud_ctx = vidcap_file_open_dec_ctx(dec,
                                        s->fmt_ctx->streams[s->audio_stream_idx], s->thread_count, s->thread_type);

                        if (s->aud_ctx == NULL) {
                                vidcap_file_common_cleanup(s);
                                return VIDCAP_INIT_FAIL;
                        }
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Input audio sample bps: %s\n",
                                        av_get_sample_fmt_name(s->aud_ctx->sample_fmt));
                        s->audio_desc.bps = av_get_bytes_per_sample(s->aud_ctx->sample_fmt);
                        s->audio_desc.sample_rate = s->aud_ctx->sample_rate;
                        s->audio_desc.ch_count =
                            get_ach_count(AVCODECCTX_CHANNELS(s->aud_ctx));
                        s->audio_data = ring_buffer_init(
                            AUD_BUF_LEN_SEC * s->audio_desc.bps *
                            s->audio_desc.ch_count * s->audio_desc.sample_rate);
                }
        }

        s->video_stream_idx = av_find_best_stream(s->fmt_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, NULL, 0);
        if (s->video_stream_idx < 0) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "No video stream found!\n");
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        }
        if (!setup_video(s)) {
                vidcap_file_common_cleanup(s);
                return VIDCAP_INIT_FAIL;
        }

        if (log_level >= LOG_LEVEL_VERBOSE) {
                av_dump_format(s->fmt_ctx, 0, s->src_filename, 0);
        }
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Audio format: %s\n",
                s->audio_stream_idx >= 0 ? audio_desc_to_cstring(s->audio_desc)
                                         : "(no audio)");
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Video format: %s\n",
                video_desc_to_string(s->video_desc));
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Capturing audio idx %d, video idx %d\n", s->audio_stream_idx, s->video_stream_idx);

        s->last_vid_pts = s->fmt_ctx->streams[s->video_stream_idx]->start_time;
        seek_start(s);

        playback_register_keyboard_ctl(&s->mod);
        register_should_exit_callback(&s->mod, vidcap_file_should_exit, s);

        pthread_create(&s->thread_id, NULL, vidcap_file_worker, s);

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_file_done(void *state) {
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        unregister_should_exit_callback(&s->mod, vidcap_file_should_exit, s);
        assert(s->magic == MAGIC);

        vidcap_file_should_exit(s);

        pthread_join(s->thread_id, NULL);

        vidcap_file_common_cleanup(s);
}

static void vidcap_file_dispose_audio(struct audio_frame *f) {
        free(f->data);
        free(f);
}

static struct audio_frame *get_audio(struct vidcap_state_lavf_decoder *s,
                                     const struct video_frame *vid_frm) {
        if (vid_frm == NULL) {
                return NULL;
        }
        pthread_mutex_lock(&s->audio_frame_lock);
        if (ring_get_current_size(s->audio_data) == 0) {
                pthread_mutex_unlock(&s->audio_frame_lock);
                return NULL;
        }

        struct audio_frame *ret = calloc(1, sizeof *ret);
        audio_frame_write_desc(ret, s->audio_desc);
        ret->flags |= TIMESTAMP_VALID;

        AVRational atb = s->fmt_ctx->streams[s->audio_stream_idx]->time_base;
        long drop_samples = 0;
        if (vid_frm->seq == UINT32_MAX) {
                log_msg_once(LOG_LEVEL_WARNING, 0x292B168B,
                             MOD_NAME "Cannot get video PTS or too high!\n");
                // capture more data to ensure the buffer won't grow - it is
                // capped with actually read data, still. Moreover there
                // number of audio samples per video frame period may not be
                // integer. It shouldn't be much, however, not to confuse
                // adaptible audio buffer.
                ret->max_size = (int)(AUDIO_RATIO * s->audio_desc.sample_rate /
                                      vid_frm->fps) *
                                s->audio_desc.bps * s->audio_desc.ch_count;
        } else {
                AVRational vtb =
                    s->fmt_ctx->streams[s->video_stream_idx]->time_base;
                int64_t apts_start = (int64_t)vid_frm->seq * vtb.num * atb.den /
                                     ((int64_t)vtb.den * atb.num);
                int64_t apts_end =
                    (((int64_t)vid_frm->seq + vid_frm->duration) *
                         (int64_t)vtb.num * atb.den +
                     ((int64_t)vtb.den * atb.num - 1)) /
                    ((int64_t)vtb.den * atb.num);
                const int64_t l = lcm(s->audio_desc.sample_rate, atb.den);
                const int64_t sample_alignment_tb = atb.num * (l / atb.den);
                const int64_t samples_aligned_tb =
                    (apts_end - s->audio_start_ts + sample_alignment_tb + 1) /
                    sample_alignment_tb * sample_alignment_tb;
                const int64_t samples =
                    samples_aligned_tb *
                    ((int64_t)s->audio_desc.sample_rate * atb.num) / atb.den;
                drop_samples = (apts_start - s->audio_start_ts) *
                               ((int64_t)s->audio_desc.sample_rate * atb.num) /
                               atb.den;
                // drop only if >.5 frm time, @sa have_audio_for_video:
                drop_samples = drop_samples < samples / 2 ? 0 : drop_samples;
                ret->max_size =
                    samples * s->audio_desc.bps * s->audio_desc.ch_count;
                debug_msg(MOD_NAME
                          "audio samples: %" PRId64
                          ", drop: %ld, reqB:%d, ring:%d, ast_ts: %" PRIu64
                          ", apts_st: %" PRIu64 ", apts_end: %" PRIu64 "\n",
                          samples, drop_samples, ret->max_size,
                          ring_get_current_size(s->audio_data),
                          s->audio_start_ts, apts_start, apts_end);
                if (ret->max_size <= 0) { // seek - have new audio but old video
                        free(ret);
                        pthread_mutex_unlock(&s->audio_frame_lock);
                        return NULL;
                }
        }
        ret->data = (char *)malloc(ret->max_size);
        ret->data_len =
            ring_buffer_read(s->audio_data, ret->data, ret->max_size);
        int64_t samples_written =
            ret->data_len / (s->audio_desc.bps * s->audio_desc.ch_count);
        s->audio_start_ts += samples_written * atb.den /
                             ((int64_t)s->audio_desc.sample_rate * atb.num);

        long drop_bytes = drop_samples * ret->bps * ret->ch_count;
        if (ret->data_len == 0 || drop_bytes >= ret->data_len) {
                vidcap_file_dispose_audio(ret);
                ret = NULL;
        } else {
                ret->dispose = vidcap_file_dispose_audio;
        }
        if (ret && drop_samples > 0) {
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Dropped %ld audio bytes.\n",
                        drop_bytes);
                memmove(ret->data, ret->data + drop_bytes,
                        ret->data_len - drop_bytes);
                ret->data_len -= drop_bytes;
        }

        pthread_mutex_unlock(&s->audio_frame_lock);
        return ret;
}

static struct audio_frame *
get_timestamped_audio(struct vidcap_state_lavf_decoder *s,
                      const struct video_frame         *vid_frm)
{
        if (s->audio_stream_idx == -1) {
                return NULL;
        }
        struct audio_frame *aud_frm = get_audio(s, vid_frm);
        if (aud_frm == NULL) {
                s->audio_frames = -1; // invalid timestamp
                return NULL;
        }
        if (s->audio_frames == -1) {
                MSG(WARNING, "Resynchronizing audio timestamps.\n");
                s->audio_frames = (long long) (((double) s->video_frames /
                                                s->video_desc.fps) *
                                               aud_frm->sample_rate);
        }
        aud_frm->timestamp =
            ((int64_t) s->audio_frames * kHz90 + aud_frm->sample_rate) /
            aud_frm->sample_rate;
        s->audio_frames +=
            aud_frm->data_len / (aud_frm->ch_count * aud_frm->bps);
        return aud_frm;
}

static struct video_frame *
vidcap_file_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_state_lavf_decoder *s = (struct vidcap_state_lavf_decoder *) state;
        struct video_frame *out;

        assert(s->magic == MAGIC);
        pthread_mutex_lock(&s->lock);
        while ((simple_linked_list_size(s->video_frame_queue) == 0 || s->paused) &&
               !s->failed && !s->should_exit) {
                pthread_cond_wait(&s->new_frame_ready, &s->lock);
        }
        if (s->failed || s->should_exit) {
                pthread_mutex_unlock(&s->lock);
                return NULL;
        }
        out = simple_linked_list_pop(s->video_frame_queue);
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->frame_consumed);

        out->timestamp =
            (uint32_t) ((double) s->video_frames * kHz90 / s->video_desc.fps);

        *audio = get_timestamped_audio(s, out);
        s->video_frames += 1;

        struct timeval t;
        do {
                gettimeofday(&t, NULL);
        } while (tv_diff(t, s->last_frame) < 1 / s->video_desc.fps);
        s->last_frame = t;
        print_current_pos(s, t);

        return out;
}

static void vidcap_file_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        *deleter = free;
        *available_cards = NULL;
        *count = 0;
}

static const struct video_capture_info vidcap_file_info = {
        vidcap_file_probe,
        vidcap_file_init,
        vidcap_file_done,
        vidcap_file_grab,
        MOD_NAME,
};

REGISTER_MODULE(file, &vidcap_file_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

