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

#include <assert.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <pthread.h>

#include "debug.h"
#include "lib_common.h"
#include "libavcodec/utils.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/fs.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "video.h"
#include "video_display.h"

#define DEFAULT_FILENAME "out.nut"
#define MOD_NAME "[File disp.] "

struct output_stream {
        AVStream           *st;
        AVCodecContext     *enc;
        AVPacket           *pkt;
        long long int       cur_pts;
        struct video_frame *frame;
};

struct state_file {
        AVFormatContext     *format_ctx;
        struct output_stream video;
        struct video_desc    video_desc;
        char                 filename[MAX_PATH_SIZE];
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
        pthread_join(s->thread_id, NULL);
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->cv);
        if (s->initialized) {
                av_write_trailer(s->format_ctx);
        }
        avcodec_free_context(&s->video.enc);
        if (!(s->format_ctx->oformat->flags & AVFMT_NOFILE)) {
                avio_closep(&s->format_ctx->pb);
        }
        av_packet_free(&s->video.pkt);
        free(s);
}

static void
usage(void)
{
        color_printf("Display " TBOLD("file") " syntax:\n");
        color_printf("\t" TBOLD(TRED("file") "[:file=<name>]") "\n");
}

static void *
display_file_init(struct module *parent, const char *fmt, unsigned int flags)
{
        const char *filename = DEFAULT_FILENAME;
        UNUSED(flags);
        UNUSED(parent);
        if (strlen(fmt) > 0) {
                if (IS_KEY_PREFIX(fmt, "file")) {
                        filename = strchr(fmt, '=') + 1;
                } else {
                        usage();
                        return strcmp(fmt, "help") == 0 ? INIT_NOERR : NULL;
                }
        }
        struct state_file *s = calloc(1, sizeof *s);
        strncat(s->filename, filename, sizeof s->filename - 1);
        avformat_alloc_output_context2(&s->format_ctx, NULL, NULL, filename);
        if (s->format_ctx == NULL) {
                log_msg(LOG_LEVEL_WARNING, "Could not deduce output format "
                                           "from file extension, using NUT.\n");
                avformat_alloc_output_context2(&s->format_ctx, NULL, "nut",
                                               filename);
                assert(s->format_ctx != NULL);
        }
        s->video.st     = avformat_new_stream(s->format_ctx, NULL);
        s->video.st->id = 0;

        if (!(s->format_ctx->oformat->flags & AVFMT_NOFILE)) {
                int ret =
                    avio_open(&s->format_ctx->pb, filename, AVIO_FLAG_WRITE);
                if (ret < 0) {
                        error_msg(MOD_NAME "avio_open: %s\n", av_err2str(ret));
                        display_file_done(s);
                        return NULL;
                }
        }
        s->video.pkt   = av_packet_alloc();

        int ret = pthread_mutex_init(&s->lock, NULL);
        ret |= pthread_cond_init(&s->cv, NULL);
        ret |= pthread_create(&s->thread_id, NULL, worker, s);
        assert(ret == 0);

        return s;
}

static void
delete_frame(struct video_frame *frame)
{
        AVFrame *avfrm = frame->callbacks.dispose_udata;
        av_frame_free(&avfrm);
}

static struct video_frame *
display_file_getf(void *state)
{
        struct state_file  *s   = state;
        AVFrame            *frame = av_frame_alloc();

        frame->format = get_ug_to_av_pixfmt(s->video_desc.color_spec);
        frame->width  = (int) s->video_desc.width;
        frame->height = (int) s->video_desc.height;
        frame->pts    = s->video.cur_pts++;
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
        if (s->video.frame != NULL) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Frame dropped!\n");
                vf_free(frame);
                pthread_mutex_unlock(&s->lock);
                return false;
        }
        s->video.frame = frame;
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->cv);
        return true;
}

static bool
display_file_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);

        codec_t codecs[VIDEO_CODEC_COUNT] = { 0 };
        int     count                     = 0;
        for (int i = 0; i < VIDEO_CODEC_COUNT; ++i) {
                if (get_ug_to_av_pixfmt(i)) {
                        codecs[count++] = i;
                }
        }
        const size_t c_len = count * sizeof codecs[0];
        if (property == DISPLAY_PROPERTY_CODECS) {
                assert(c_len <= *len);
                memcpy(val, codecs, c_len);
                *len = c_len;
                return true;
        }
        return false;
}

static bool
display_file_reconfigure(void *state, struct video_desc desc)
{
        struct state_file *s = state;

        s->video_desc = desc;
        return true;
}

static bool
reconfigure_video(struct state_file *s, struct video_desc desc)
{
        s->video.st->time_base = (AVRational){ get_framerate_d(desc.fps),
                                               get_framerate_n(desc.fps) };
        const AVCodec *codec   = avcodec_find_encoder(AV_CODEC_ID_RAWVIDEO);
        avcodec_free_context(&s->video.enc);
        s->video.enc            = avcodec_alloc_context3(codec);
        s->video.enc->width     = (int) desc.width;
        s->video.enc->height    = (int) desc.height;
        s->video.enc->time_base = s->video.st->time_base;
        s->video.enc->pix_fmt   = get_ug_to_av_pixfmt(desc.color_spec);
        int ret                 = avcodec_open2(s->video.enc, codec, NULL);
        if (ret < 0) {
                error_msg(MOD_NAME "avcodec_open2: %s\n", av_err2str(ret));
                return false;
        }
        ret = avcodec_parameters_from_context(s->video.st->codecpar,
                                              s->video.enc);
        if (ret < 0) {
                error_msg(MOD_NAME "Could not copy the stream parameters: %s\n",
                          av_err2str(ret));
                return false;
        }

        av_dump_format(s->format_ctx, 0, s->filename, 1);

        ret = avformat_write_header(s->format_ctx, NULL);
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
write_video_frame(struct state_file *s, AVFrame *frame)
{
        int ret = avcodec_send_frame(s->video.enc, frame);
        if (ret < 0) {
                error_msg(MOD_NAME "avcodec_send_frame: %s\n", av_err2str(ret));
                return;
        }
        while (ret >= 0) {
                ret = avcodec_receive_packet(s->video.enc, s->video.pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                        break;
                }
                if (ret < 0) {
                        error_msg(MOD_NAME "avcodec_receive_frame: %s\n",
                                  av_err2str(ret));
                        return;
                }
                av_packet_rescale_ts(s->video.pkt, s->video.enc->time_base,
                                     s->video.st->time_base);
                s->video.pkt->stream_index = s->video.st->index;
                ret = av_interleaved_write_frame(s->format_ctx, s->video.pkt);
                if (ret < 0) {
                        error_msg(MOD_NAME "error writting packet: %s\n",
                                  av_err2str(ret));
                }
        }
}

static void *
worker(void *arg)
{
        struct state_file *s = arg;

        struct video_desc saved_vid_desc = { 0 };

        while (!s->should_exit) {
                struct video_frame *frame = NULL;
                pthread_mutex_lock(&s->lock);
                while (s->video.frame == NULL && !s->should_exit) {
                        pthread_cond_wait(&s->cv, &s->lock);
                }
                if (s->should_exit) {
                        break;
                }
                frame = s->video.frame;
                s->video.frame = NULL;
                pthread_mutex_unlock(&s->lock);
                const struct video_desc frame_desc =
                    video_desc_from_frame(frame);
                if (!video_desc_eq(saved_vid_desc, frame_desc)) {
                        if (!reconfigure_video(s, frame_desc)) {
                                exit_uv(1);
                                vf_free(frame);
                                continue;
                        }
                        saved_vid_desc = frame_desc;
                }
                write_video_frame(s, frame->callbacks.dispose_udata);
                vf_free(frame);
        }

        pthread_mutex_unlock(&s->lock);
        return NULL;
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
                NULL,
                NULL,
                MOD_NAME,
        };
        return &display_file_info;
};

REGISTER_MODULE_WITH_FUNC(file, display_file_info_get,
                          LIBRARY_CLASS_VIDEO_DISPLAY,
                          VIDEO_DISPLAY_ABI_VERSION);
