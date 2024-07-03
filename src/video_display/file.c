/**
 * @file   video_display/file.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET, z. s. p. o.
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
 * Inspired and written according to mux.c FFmpeg example.
 * @note
 * when compression is enabled (output is not NUT), accepted audio format
 * is restricted to 16-bit mono or stereo. We do not have to implement the
 * whole stack of conversions (or utilize libavresample).
 */

#include <assert.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <pthread.h>

#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "libavcodec/lavc_common.h"
#include "libavcodec/to_lavc_vid_conv.h"
#include "libavcodec/utils.h"
#include "tv.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/fs.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/text.h"
#include "video.h"
#include "video_display.h"

#define DEFAULT_FILENAME     "out.mp4"
#define DEFAULT_PIXEL_FORMAT AV_PIX_FMT_YUV420P
#define MOD_NAME             "[File disp.] "

enum {
        DEFAULT_MAX_AV_DIFF_NS = 84 * MS_IN_NS, // 2 24p frames
};

struct output_stream {
        AVStream           *st;
        AVCodecContext     *enc;
        long long int       next_pts;
        time_ns_t           next_frm_time;
        union {
                struct video_frame *vid_frm;
                AVFrame            *aud_frm;
        };
};

struct state_file {
        AVFormatContext     *format_ctx;
        bool                 is_nut; // == use RAW
        struct output_stream audio;
        struct output_stream video;
        struct video_desc    video_desc;
        struct to_lavc_vid_conv *video_conv;
        char                 filename[MAX_PATH_SIZE];
        time_ns_t            max_av_diff_ns; ///< max A/V diff in ns
        pthread_t            thread_id;
        pthread_mutex_t      lock;
        pthread_cond_t       cv;
        bool                 initialized;
        bool                 should_exit;
};

static void *worker(void *arg);

static void
display_file_probe(struct device_info **available_cards, int *count,
                   void (**deleter)(void *))
{
        *deleter         = free;
        *count           = 1;
        *available_cards = calloc(*count, sizeof **available_cards);
        snprintf((*available_cards)[0].name, sizeof(*available_cards)[0].name,
                 "file");
}

static void
display_file_done(void *state)
{
        struct state_file *s = state;
        if (s->should_exit) { // thread started
                pthread_join(s->thread_id, NULL);
                pthread_mutex_destroy(&s->lock);
                pthread_cond_destroy(&s->cv);
        }
        if (s->initialized) {
                av_write_trailer(s->format_ctx);
        }
        avcodec_free_context(&s->video.enc);
        avcodec_free_context(&s->audio.enc);
        if (!(s->format_ctx->oformat->flags & AVFMT_NOFILE)) {
                avio_closep(&s->format_ctx->pb);
        }
        vf_free(s->video.vid_frm);
        av_frame_free(&s->audio.aud_frm);
        to_lavc_vid_conv_destroy(&s->video_conv);
        free(s);
}

static void
usage(bool full)
{
        color_printf("Display " TBOLD("file") " syntax:\n\n");
        color_printf("\t" TBOLD(TRED("file") "[:name=<filename>]") " | " TBOLD(
            "file:[full]help") "\n\n");
        color_printf("where\n\n");
        color_printf("\t" TBOLD("<filename>") " - output file name\n");
        if (full) {
                color_printf(
                    "\t" TBOLD("max_av_diff") " - allowed A/V descync length "
                                              "(in seconds; default %.3f)\n",
                    DEFAULT_MAX_AV_DIFF_NS / NS_IN_SEC_DBL);
        }
        color_printf("\n");
        char codec_note[] = TBOLD(
            "NUT") " files are written uncompressed. For other file "
                   "formats " TBOLD(
                       "FFmpeg") " container default "
                                 "codecs are used.\n\nDefault output file "
                                 "is: " TBOLD(DEFAULT_FILENAME) "\n\n";
        color_printf("%s", wrap_paragraph(codec_note));
}

static bool
parse_fmt(struct state_file *s, char *fmt)
{
        char *end_ptr = NULL;
        char *item = NULL;
        while ((item = strtok_r(fmt, ":", &end_ptr))) {
                fmt = NULL;
                char *val = strchr(item, '=') + 1;
                if (IS_KEY_PREFIX(item, "file") || IS_KEY_PREFIX(item, "name")) {
                        snprintf(s->filename, sizeof s->filename, "%s", val);
                } else if (IS_KEY_PREFIX(item, "max_av_diff")) {
                        s->max_av_diff_ns =
                            (time_ns_t) (strtod(val, NULL) * NS_IN_SEC_DBL);
                        assert(s->max_av_diff_ns >= 0.0);
                } else {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "Unknown option: %s\n", item);
                        return false;
                }
        }
        return true;
}

static void *
display_file_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(parent);
        if (!strcmp(fmt, "help") || !strcmp(fmt, "fullhelp")) {
                usage(!strcmp(fmt, "fullhelp"));
                return INIT_NOERR;
        }

        struct state_file *s = calloc(1, sizeof *s);
        snprintf(s->filename, sizeof s->filename, "%s", DEFAULT_FILENAME);
        s->max_av_diff_ns = DEFAULT_MAX_AV_DIFF_NS;
        char *fmt_c     = strdup(fmt);
        bool  parse_ret = parse_fmt(s, fmt_c);
        free(fmt_c);
        if (!parse_ret) {
                free(s);
                return NULL;
        }

        ug_set_av_logging();
        avformat_alloc_output_context2(&s->format_ctx, NULL, NULL, s->filename);
        if (s->format_ctx == NULL) {
                log_msg(LOG_LEVEL_WARNING, "Could not deduce output format "
                                           "from file extension, using NUT.\n");
                avformat_alloc_output_context2(&s->format_ctx, NULL, "nut",
                                               s->filename);
                assert(s->format_ctx != NULL);
        }
        s->is_nut       = !strcmp(s->format_ctx->oformat->name, "nut");
        s->video.st     = avformat_new_stream(s->format_ctx, NULL);
        s->video.st->id = 0;

        if (!(s->format_ctx->oformat->flags & AVFMT_NOFILE)) {
                int ret =
                    avio_open(&s->format_ctx->pb, s->filename, AVIO_FLAG_WRITE);
                if (ret < 0) {
                        error_msg(MOD_NAME "avio_open %s: %s\n", s->filename,
                                  av_err2str(ret));
                        display_file_done(s);
                        return NULL;
                }
        }

        int ret = pthread_mutex_init(&s->lock, NULL);
        ret |= pthread_cond_init(&s->cv, NULL);
        ret |= pthread_create(&s->thread_id, NULL, worker, s);
        assert(ret == 0);

        if ((flags & DISPLAY_FLAG_AUDIO_ANY) != 0U) {
                s->audio.st     = avformat_new_stream(s->format_ctx, NULL);
                s->audio.st->id = 1;
        }

        return s;
}

static void
delete_frame(struct video_frame *frame)
{
        AVFrame *avfrm = frame->callbacks.dispose_udata;
        av_frame_free(&avfrm);
}

static enum AVPixelFormat
file_get_pix_fmt(bool is_nut, codec_t ug_codec)
{
        if (!is_nut) {
                return DEFAULT_PIXEL_FORMAT;
        }
        if (ug_codec == R10k) {
                return AV_PIX_FMT_GBRP10LE;
        }
        if (ug_codec == R12L) {
                return AV_PIX_FMT_GBRP12LE;
        }
        if (ug_codec == v210) {
                return AV_PIX_FMT_YUV422P10LE;
        }
        return get_ug_to_av_pixfmt(ug_codec);
}

static struct video_frame *
display_file_getf(void *state)
{
        struct state_file  *s   = state;

        if (file_get_pix_fmt(s->is_nut, s->video_desc.color_spec) !=
            get_ug_to_av_pixfmt(s->video_desc.color_spec)) {
                return vf_alloc_desc_data(s->video_desc); // conv needed
        }
        AVFrame *frame = av_frame_alloc();
        frame->format  = get_ug_to_av_pixfmt(s->video_desc.color_spec);
        frame->width  = (int) s->video_desc.width;
        frame->height = (int) s->video_desc.height;
        int ret       = av_frame_get_buffer(frame, 0);
        if (ret < 0) {
                error_msg(MOD_NAME "Could not allocate frame data: %s.\n",
                          av_err2str(ret));
                av_frame_free(&frame);
                return NULL;
        }
        struct video_frame *out      = vf_alloc_desc(s->video_desc);
        out->tiles[0].data           = (char *) frame->data[0];
        out->callbacks.dispose_udata = frame;
        out->callbacks.data_deleter  = delete_frame;
        return out;
}

static bool
display_file_putf(void *state, struct video_frame *frame, long long timeout_ns)
{
        if (timeout_ns == PUTF_DISCARD) {
                vf_free(frame);
                return true;
        }
        struct state_file *s = state;
        pthread_mutex_lock(&s->lock);
        if (frame == NULL) {
                s->should_exit = true;
                pthread_mutex_unlock(&s->lock);
                pthread_cond_signal(&s->cv);
                return true;
        }
        bool ret = true;
        if (s->video.vid_frm != NULL) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Video frame dropped!\n");
                vf_free(s->video.vid_frm);
                ret = false;
        }
        s->video.vid_frm = frame;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->cv);
        return ret;
}

static bool
avp_is_in_set(enum AVPixelFormat needle, int nmembers,
              const enum AVPixelFormat *haystick)
{
        for (int i = 0; i < nmembers; ++i) {
                if (haystick[i] == needle) {
                        return true;
                }
        }
        return false;
}

static bool
display_file_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_file *s = state;

        switch (property) {
        case DISPLAY_PROPERTY_CODECS: {
                codec_t codecs[VIDEO_CODEC_COUNT] = { 0 };
                int     count                     = 0;
                if (s->is_nut) {
                        codecs[count++] = R10k;
                        codecs[count++] = R12L;
                        codecs[count++] = v210;
                }
                for (int i = 0; i < VIDEO_CODEC_COUNT; ++i) {
                        if (s->is_nut) {
                                if (get_ug_to_av_pixfmt(i) != AV_PIX_FMT_NONE) {
                                        codecs[count++] = i;
                                }
                        } else {
                                enum AVPixelFormat      fmts[AV_PIX_FMT_NB];
                                struct to_lavc_req_prop prop = {
                                        TO_LAVC_REQ_PROP_INIT
                                };
                                const int nm =
                                    get_available_pix_fmts(i, prop, fmts);
                                if (avp_is_in_set(DEFAULT_PIXEL_FORMAT, nm,
                                                  fmts)) {
                                        codecs[count++] = i;
                                }
                        }
                }
                const size_t c_len = count * sizeof codecs[0];
                assert(c_len <= *len);
                memcpy(val, codecs, c_len);
                *len = c_len;
                break;
        }
        case DISPLAY_PROPERTY_AUDIO_FORMAT: {
                struct audio_desc *desc = val;
                assert(*len == (int) sizeof *desc);
                desc->codec = AC_PCM;
                if (!s->is_nut) {
                        desc->ch_count = MIN(2, desc->ch_count);
                        desc->bps      = 2;
                }
                break;
        }
        default:
                return false;
        }
        return true;
}

static bool
display_file_reconfigure(void *state, struct video_desc desc)
{
        struct state_file *s = state;

        s->video_desc = desc;
        return true;
}

static void
aud_ctx_set_ch_layout(struct AVCodecContext *ctx, int ch_count, bool raw)
{
#if FF_API_NEW_CHANNEL_LAYOUT
        if (raw) {
                ctx->ch_layout = (AVChannelLayout) AV_CHANNEL_LAYOUT_MASK(
                    ch_count, (1 << ch_count) - 1);
                return;
        }
        av_channel_layout_default(&ctx->ch_layout, ch_count);
#else
        (void) raw;
        ctx->channel_layout = av_get_default_channel_layout(ch_count);
#endif
}

static void
aud_frm_set_ch_layout(AVFrame *frm, int ch_count)
{
#if FF_API_NEW_CHANNEL_LAYOUT
        frm->ch_layout = (AVChannelLayout) AV_CHANNEL_LAYOUT_MASK(
            ch_count, (1 << ch_count) - 1);
#else
        frm->channel_layout = av_get_default_channel_layout(ch_count);
#endif
}

static AVFrame *
alloc_audio_frame(struct audio_desc desc, int nb_samples,
                  enum AVSampleFormat fmt)
{
        AVFrame *av_frm   = av_frame_alloc();
        av_frm->format    = fmt;
        aud_frm_set_ch_layout(av_frm, desc.ch_count);
        av_frm->sample_rate = desc.sample_rate;
        av_frm->nb_samples  = nb_samples;

        if (nb_samples == 0) {
                return av_frm;
        }
        int ret = av_frame_get_buffer(av_frm, 0);
        if (ret < 0) {
                error_msg(MOD_NAME "audio buf alloc: %s\n", av_err2str(ret));
                av_frame_free(&av_frm);
                return NULL;
        }
        return av_frm;
}

static enum AVSampleFormat
select_sample_format(const enum AVSampleFormat *fmts)
{
        enum AVSampleFormat fmt = AV_SAMPLE_FMT_NONE;
        while (*fmts) {
                if (*fmts == AV_SAMPLE_FMT_S16P || *fmts == AV_SAMPLE_FMT_S16) {
                        return *fmts;
                }
                if (*fmts == AV_SAMPLE_FMT_FLTP) { // use fltp only if no s16[p]
                        fmt = *fmts;
                }
                fmts++;
        }
        assert(fmt != AV_SAMPLE_FMT_NONE && MOD_NAME
               "Only S16, S16P and FLTP samples are currently "
               "handled, please report!");
        return fmt;
}

static bool
configure_audio(struct state_file *s, struct audio_desc aud_desc,
                AVFrame **tmp_frame)
{
        enum AVCodecID codec_id = s->format_ctx->oformat->audio_codec;
        if (s->is_nut) {
                switch (aud_desc.bps) {
                case 1:
                        codec_id = AV_CODEC_ID_PCM_U8;
                        break;
                case 2:
                        codec_id = AV_CODEC_ID_PCM_S16LE;
                        break;
                case 3:
                case 4:
                        codec_id = AV_CODEC_ID_PCM_S32LE;
                        break;
                default:
                        abort();
                }
        }
        const AVCodec *codec = avcodec_find_encoder(codec_id);
        if (codec == NULL && !s->is_nut) {
                codec = avcodec_find_encoder(codec_id);
        }
        if (codec == NULL) {
                error_msg(MOD_NAME "Unable to find audio encoder for %s\n",
                          avcodec_get_name(codec_id));
                return false;
        }
        s->audio.enc             = avcodec_alloc_context3(codec);
        s->audio.enc->sample_fmt =
            s->is_nut ? audio_bps_to_av_sample_fmt(aud_desc.bps, false)
                      : select_sample_format(s->audio.enc->codec->sample_fmts);
        aud_ctx_set_ch_layout(s->audio.enc, aud_desc.ch_count, s->is_nut);
        s->audio.enc->sample_rate = aud_desc.sample_rate;
        s->audio.st->time_base    = (AVRational){ 1, aud_desc.sample_rate };
        s->video.enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

        int ret = avcodec_open2(s->audio.enc, codec, NULL);
        if (ret < 0) {
                error_msg(MOD_NAME "audio avcodec_open2: %s\n",
                          av_err2str(ret));
                return false;
        }

        ret = avcodec_parameters_from_context(s->audio.st->codecpar,
                                              s->audio.enc);
        if (ret < 0) {
                error_msg(MOD_NAME
                          "Could not copy audio stream parameters: %s\n",
                          av_err2str(ret));
                return false;
        }

        av_frame_free(tmp_frame);
        *tmp_frame = alloc_audio_frame(
            aud_desc, s->audio.enc->frame_size,
            s->audio.enc->sample_fmt);
        (*tmp_frame)->nb_samples = 0;

        return true;
}

static bool
configure_video(struct state_file *s, struct video_desc vid_desc)
{
        s->video.st->time_base = (AVRational){ get_framerate_d(vid_desc.fps),
                                               get_framerate_n(vid_desc.fps) };
        const enum AVCodecID codec_id =
            s->is_nut ? AV_CODEC_ID_RAWVIDEO
                      : s->format_ctx->oformat->video_codec;
        const AVCodec *codec = avcodec_find_encoder(codec_id);
        if (codec == NULL) {
                error_msg(MOD_NAME "Unable to find video encoder for %s\n",
                          avcodec_get_name(codec_id));
                return false;
        }
        s->video.enc            = avcodec_alloc_context3(codec);
        s->video.enc->width     = (int) vid_desc.width;
        s->video.enc->height    = (int) vid_desc.height;
        s->video.enc->time_base = s->video.st->time_base;
        s->video.enc->pix_fmt =
            file_get_pix_fmt(s->is_nut, vid_desc.color_spec);
        s->video.enc->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
        av_opt_set(s->video.enc->priv_data, "preset", "ultrafast", 0); // x264/5
        av_opt_set(s->video.enc->priv_data, "deadline", "realtime", 0); // vp9
        av_opt_set(s->video.enc->priv_data, "cpu-used", "8", 0);
        int ret = avcodec_open2(s->video.enc, codec, NULL);
        if (ret < 0) {
                error_msg(MOD_NAME "video avcodec_open2: %s\n",
                          av_err2str(ret));
                return false;
        }
        ret = avcodec_parameters_from_context(s->video.st->codecpar,
                                              s->video.enc);
        if (ret < 0) {
                error_msg(MOD_NAME
                          "Could not copy video stream parameters: %s\n",
                          av_err2str(ret));
                return false;
        }
        if (s->video.enc->pix_fmt != get_ug_to_av_pixfmt(vid_desc.color_spec)) {
                s->video_conv = to_lavc_vid_conv_init(
                    vid_desc.color_spec, (int) vid_desc.width,
                    (int) vid_desc.height, s->video.enc->pix_fmt,
                    get_cpu_core_count());
        }
        return true;
}

static bool
initialize(struct state_file *s, struct video_desc *saved_vid_desc,
           const struct video_frame *vid_frm, struct audio_desc *saved_aud_desc,
           const AVFrame *aud_frm, AVFrame **tmp_aud_frame)
{
        if (!vid_frm || (s->audio.st != NULL && !aud_frm)) {
                log_msg(LOG_LEVEL_INFO,
                        MOD_NAME "Waiting for all streams to init.\n");
                return false;
        }

        // video
        const struct video_desc vid_desc = video_desc_from_frame(vid_frm);
        if (!configure_video(s, vid_desc)) {
                exit_uv(1);
                return false;
        }
        *saved_vid_desc = vid_desc;

        // audio
        if (aud_frm != NULL) {
                const struct audio_desc aud_desc =
                    audio_desc_from_av_frame(aud_frm);
                if (!configure_audio(s, aud_desc, tmp_aud_frame)) {
                        exit_uv(1);
                        return false;
                }
                *saved_aud_desc = aud_desc;
        }

        av_dump_format(s->format_ctx, 0, s->filename, 1);

        int ret = avformat_write_header(s->format_ctx, NULL);
        if (ret < 0) {
                error_msg(MOD_NAME
                          "Error occurred when opening output file: %s\n",
                          av_err2str(ret));
                return false;
        }

        s->initialized = true;
        return true;
}

static void
write_frame(AVFormatContext *format_ctx, struct output_stream *ost,
            AVFrame *frame, AVPacket *pkt)
{
        frame->pts = ost->next_pts;
        int ret    = avcodec_send_frame(ost->enc, frame);
        if (ret < 0) {
                error_msg(MOD_NAME "avcodec_send_frame: %s\n",
                          av_err2str(ret));
                return;
        }
        while (ret >= 0) {
                ret = avcodec_receive_packet(ost->enc, pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                }
                if (ret < 0) {
                        error_msg(MOD_NAME "video avcodec_receive_frame: %s\n",
                                  av_err2str(ret));
                        return;
                }
                av_packet_rescale_ts(pkt, ost->enc->time_base,
                                     ost->st->time_base);
                pkt->stream_index = ost->st->index;
                ret = av_interleaved_write_frame(format_ctx, pkt);
                if (ret < 0) {
                        error_msg(MOD_NAME "error writting video packet: %s\n",
                                  av_err2str(ret));
                }
        }
}

static bool
check_reconf(struct video_desc *saved_vid_desc,
           const struct video_frame *vid_frm, struct audio_desc *saved_aud_desc,
           const AVFrame *aud_frm)
{
        if (vid_frm != NULL) {
                const struct video_desc cur_vid_desc =
                    video_desc_from_frame(vid_frm);
                if (!video_desc_eq(*saved_vid_desc, cur_vid_desc)) {
                        return false;
                }
        }
        if (aud_frm) {
                const struct audio_desc cur_aud_desc =
                    audio_desc_from_av_frame(aud_frm);
                if (!audio_desc_eq(*saved_aud_desc, cur_aud_desc)) {
                        return false;
                }
        }
        return true;
}

/// append nb_samples to from src to dst, skipping skip_in_samples from input
static void
file_append_audio_frame(AVFrame *dst, const AVFrame *src, int skip_in_samples,
                        int nb_samples)
{
#if FF_API_NEW_CHANNEL_LAYOUT
        const int ch_count = src->ch_layout.nb_channels;
#else
        const int ch_count =
            av_get_channel_layout_nb_channels(src->channel_layout);
#endif
        const int s_bps    = av_get_bytes_per_sample(src->format);
        const int d_bps    = av_get_bytes_per_sample(dst->format);

        if (av_sample_fmt_is_planar(dst->format)) {
                char *dst_data[ch_count];
                for (int i = 0; i < ch_count; ++i) {
                        dst_data[i] = (char *) dst->data[i] +
                                      (ptrdiff_t) dst->nb_samples * d_bps;
                }
                if (dst->format == AV_SAMPLE_FMT_FLTP) {
                        interleaved2noninterleaved_float(
                            dst_data,
                            (char *) src->data[0] +
                                (ptrdiff_t) skip_in_samples * s_bps * ch_count,
                            s_bps, nb_samples * s_bps * ch_count, ch_count);
                } else {
                        assert(s_bps == d_bps);
                        interleaved2noninterleaved2(
                            dst_data,
                            (char *) src->data[0] +
                                (ptrdiff_t) skip_in_samples * s_bps * ch_count,
                            s_bps, nb_samples * s_bps * ch_count, ch_count);
                }

        } else {
                assert(s_bps == d_bps);
                memcpy(dst->data[0] +
                           (ptrdiff_t) dst->nb_samples * s_bps * ch_count,
                       src->data[0] +
                           (ptrdiff_t) skip_in_samples * s_bps * ch_count,
                       (size_t) nb_samples * s_bps * ch_count);
        }
        dst->nb_samples += nb_samples;
}

static void
write_audio_frame(struct state_file *s, AVFrame *aud_frm, AVFrame *tmp_frm,
                  AVPacket *pkt)
{
        s->audio.next_frm_time =
            get_time_in_ns() +
            (aud_frm->nb_samples * NS_IN_SEC / aud_frm->sample_rate);

        if (s->is_nut) {
                write_frame(s->format_ctx, &s->audio, aud_frm, pkt);
                s->audio.next_pts += aud_frm->nb_samples;
                return;
        }

        // compressed audio
        const int frame_size = s->audio.enc->frame_size;
        int consumed_samples = 0;
        while (tmp_frm->nb_samples + // NOLINT
                   (aud_frm->nb_samples - consumed_samples) >=
               frame_size) {
                const int needed_samples = frame_size - tmp_frm->nb_samples;
                file_append_audio_frame(tmp_frm, aud_frm, consumed_samples,
                                    needed_samples);
                write_frame(s->format_ctx, &s->audio, tmp_frm, pkt);
                s->audio.next_pts += frame_size;
                consumed_samples += needed_samples;
                tmp_frm->nb_samples = 0;
        }

        file_append_audio_frame(tmp_frm, aud_frm, consumed_samples,
                                aud_frm->nb_samples - consumed_samples);
}

static void
write_video_frame(struct state_file *s, struct video_frame *vid_frm,
                  AVPacket *pkt)
{
        const long long vid_frm_time_ns =
            (long long) (NS_IN_SEC / s->video_desc.fps);
        AVFrame *frame =
            s->video_conv == NULL
                ? vid_frm->callbacks.dispose_udata
                : to_lavc_vid_conv(s->video_conv, vid_frm->tiles[0].data);
        bool dup = false;

        // handle AV sync
        if (s->audio.st != NULL) {
                const time_ns_t audio_start =
                    s->audio.next_frm_time -
                    s->audio.next_pts * NS_IN_SEC / s->audio.enc->sample_rate;
                const time_ns_t video_start =
                    s->video.next_frm_time -
                    (long long) ((double) (s->video.next_pts * NS_IN_SEC) /
                                 s->video_desc.fps);
                if (s->video.next_frm_time != 0 &&
                    llabs(audio_start - video_start) > s->max_av_diff_ns) {
                        log_msg(
                            LOG_LEVEL_WARNING,
                            MOD_NAME "A-V desync %f sec, video frame %s...\n",
                            (double) (audio_start - video_start) /
                                NS_IN_SEC_DBL,
                            video_start < audio_start ? "dropped" : "dupped");
                        if (video_start < audio_start) {
                                return; // drop frame
                        }
                        dup = true;
                }
        }

write_frame:
        write_frame(s->format_ctx, &s->video, frame, pkt);
        s->video.next_pts += 1;
        s->video.next_frm_time = get_time_in_ns() + vid_frm_time_ns;

        if (dup) {
                dup = false;
                goto write_frame;
        }
}

static void *
worker(void *arg)
{
        struct state_file *s = arg;

        struct video_desc   saved_vid_desc = { 0 };
        struct audio_desc   saved_aud_desc = { 0 };
        struct video_frame *vid_frm        = NULL;
        AVFrame            *aud_frm        = NULL;
        AVFrame            *tmp_aud_frm    = NULL;
        AVPacket           *pkt            = av_packet_alloc();

        while (!s->should_exit) {
                pthread_mutex_lock(&s->lock);
                while (s->audio.aud_frm == NULL && s->video.vid_frm == NULL &&
                       !s->should_exit) {
                        pthread_cond_wait(&s->cv, &s->lock);
                }
                if (s->should_exit) {
                        break;
                }
                if (s->video.vid_frm) {
                        vf_free(vid_frm);
                        vid_frm = s->video.vid_frm;
                }
                if (s->audio.aud_frm) {
                        av_frame_free(&aud_frm);
                        aud_frm = s->audio.aud_frm;
                }
                s->video.vid_frm = NULL;
                s->audio.aud_frm = NULL;
                pthread_mutex_unlock(&s->lock);

                if (!s->initialized) {
                        if (!initialize(s, &saved_vid_desc, vid_frm,
                                        &saved_aud_desc, aud_frm,
                                        &tmp_aud_frm)) {
                                continue;
                        }
                }

                if (!check_reconf(&saved_vid_desc, vid_frm, &saved_aud_desc,
                                  aud_frm)) {
                        error_msg(MOD_NAME "Reconfiguration not implemented. "
                                           "Let us know if desired.\n");
                        continue;
                }

                if (aud_frm) {
                        write_audio_frame(s, aud_frm, tmp_aud_frm, pkt);
                        av_frame_free(&aud_frm);
                }
                if (vid_frm) {
                        write_video_frame(s, vid_frm, pkt);
                        vf_free(vid_frm);
                        vid_frm = NULL;
                }
        }
        vf_free(vid_frm);
        av_frame_free(&aud_frm);
        if (tmp_aud_frm != NULL && tmp_aud_frm->nb_samples > 0) { // last frame
                write_frame(s->format_ctx, &s->audio, tmp_aud_frm, pkt);
        }
        av_frame_free(&tmp_aud_frm);
        av_packet_free(&pkt);

        pthread_mutex_unlock(&s->lock);
        return NULL;
}

static void
display_file_put_audio_frame(void *state, const struct audio_frame *frame)
{
        struct state_file *s = state;

        // store always as intereleaved integer samples
        AVFrame *av_frm =
            alloc_audio_frame(audio_desc_from_frame(frame),
                              frame->data_len / frame->ch_count / frame->bps,
                              audio_bps_to_av_sample_fmt(frame->bps, false));
        if (av_frm == NULL) {
                return;
        }
        memcpy(av_frm->data[0], frame->data, frame->data_len);
        pthread_mutex_lock(&s->lock);
        if (s->audio.aud_frm != NULL) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Audio frame dropped!\n");
                av_frame_free(&s->audio.aud_frm);
        }
        s->audio.aud_frm = av_frm;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->cv);
}

static bool
display_file_reconfigure_audio(void *state, int quant_samples, int channels,
                              int sample_rate)
{
        UNUSED(state), UNUSED(quant_samples), UNUSED(channels),
            UNUSED(sample_rate);
        return true;
}

static const void *
display_file_info_get()
{
        static const struct video_display_info display_file_info = {
                display_file_probe,
                display_file_init,
                NULL, // _run
                display_file_done,
                display_file_getf,
                display_file_putf,
                display_file_reconfigure,
                display_file_get_property,
                display_file_put_audio_frame,
                display_file_reconfigure_audio,
                MOD_NAME,
        };
        return &display_file_info;
};

REGISTER_MODULE_WITH_FUNC(file, display_file_info_get,
                          LIBRARY_CLASS_VIDEO_DISPLAY,
                          VIDEO_DISPLAY_ABI_VERSION);
