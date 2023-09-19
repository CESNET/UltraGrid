/**
 * @file   audio/codec/libavcodec.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2023 CESNET z.s.p.o.
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

#define __STDC_CONSTANT_MACROS

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "debug.h"
#include "lib_common.h"

#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>

#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"
#include "libavcodec/lavc_common.h"
#include "utils/packet_counter.h"
#include "utils/text.h"

#define MAGIC 0xb135ca11
#define LOW_LATENCY_AUDIOENC_FRAME_DURATION 2.5

enum {
        TMP_DATA_LEN = 1024 * 1024,
};
#define MOD_NAME "[lavcd aud.] "

struct libavcodec_codec_state;

static void *libavcodec_init(audio_codec_t audio_codec, audio_codec_direction_t direction,
                bool silent, int bitrate);
static audio_channel *libavcodec_compress(void *, audio_channel *);
static void libavcodec_done(void *);
static void cleanup_common(struct libavcodec_codec_state *s);

enum codec_flags {
        /// Decoder requires the packets to be passed packetized by the encoder
        /// so that eg. two packet cannot be given in a singe buffer. Some codec
        /// seem to be more tolerant even though it generates packetized output,
        /// eg. AAC or MP3 even though it generates packetized output, eg. AAC
        /// or MP3 even though it generates packetized output, eg. AAC or MP3
        /// even though it generates packetized output, eg. AAC or MP3.
        PACKETIZED_DEC = 1 << 0,
};

struct codec_param {
        enum AVCodecID id;
        const char *preferred_encoder;
        int flags;
};

static const struct codec_param mapping[AC_COUNT] = {
        [AC_ALAW]  = {AV_CODEC_ID_PCM_ALAW,    NULL,         0             },
        [AC_MULAW] = { AV_CODEC_ID_PCM_MULAW,  NULL,         0             },
        [AC_SPEEX] = { AV_CODEC_ID_SPEEX,      NULL,         0             },
        [AC_OPUS]  = { AV_CODEC_ID_OPUS,       NULL,         PACKETIZED_DEC},
        [AC_G722]  = { AV_CODEC_ID_ADPCM_G722, NULL,         0             },
        [AC_FLAC]  = { AV_CODEC_ID_FLAC,       NULL,         0             },
        [AC_MP3]   = { AV_CODEC_ID_MP3,        NULL,         0             },
        [AC_AAC]   = { AV_CODEC_ID_AAC,        "libfdk_aac", 0             },
};

struct libavcodec_codec_state {
        uint32_t magic;
        const struct codec_param *codec_info;
        struct AVPacket *pkt;
        struct AVCodecContext *codec_ctx;
        const struct AVCodec *codec;

        AVFrame            *av_frame;

        struct audio_desc   saved_desc;

        audio_channel       tmp;
        _Alignas(16) char   tmp_data[TMP_DATA_LEN]; ///< tmp.data, but non-const qualified
        audio_channel       output_channel;
        _Alignas(16) char   output_channel_data[TMP_DATA_LEN]; ///< output_channel.data, but non-const qualified

        int                 bitrate;

        bool                context_initialized;
        audio_codec_direction_t direction;

        unsigned char      *tmp_buffer;
        size_t              tmp_buffer_size;
};

///< reallocate tmp_buffer if requested size is greater than currently allocated size
static void resize_tmp_buffer(unsigned char **tmp_buffer, size_t *cur_size, size_t new_size) {
        if (new_size <= *cur_size) {
                return;
        }
        unsigned char *new_buf = realloc(*tmp_buffer, new_size);
        assert(new_buf);
        *tmp_buffer = new_buf;
        *cur_size = new_size;
}

/**
 * @todo
 * Remove and use the global print_libav_error. Dependencies need to be resolved first.
 */
static void print_libav_audio_error(int verbosity, const char *msg, int rc) {
        char errbuf[1024];
        av_strerror(rc, errbuf, sizeof(errbuf));

        log_msg(verbosity, "%s: %s\n", msg, errbuf);
}

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
ADD_TO_PARAM("audioenc-frame-duration", "* audioenc-frame-duration=<ms>\n"
                "  Sets audio encoder frame duration (in ms), default is " STR(LOW_LATENCY_AUDIOENC_FRAME_DURATION) " ms for low-latency-audio\n");
ADD_TO_PARAM("audio-lavc-decoder", "* audio-lavc-decoder=<decoder_name>\n"
                "  Use selected audio lavc decoder\n");
ADD_TO_PARAM("audio-lavc-encoder", "* audio-lavc-encoder=<encoder_name>\n"
                "  Use selected audio lavc encoder\n");
/**
 * Initializates selected audio codec
 * @param audio_codec requested audio codec
 * @param direction   which direction will be used (encoding or decoding)
 * @param silent      if true no error messages will be printed.
 *                    This is intended for checking which codecs are present
 * @retval NULL if initialization failed
 * @retval !=NULL codec state
 */
static void *libavcodec_init(audio_codec_t audio_codec, audio_codec_direction_t direction, bool silent,
                int bitrate)
{
        ug_set_av_logging();

        enum AVCodecID codec_id = AV_CODEC_ID_NONE;

        const struct codec_param *it = &mapping[audio_codec];
        const char *preferred_encoder = NULL;
        
        if (!it->id) {
                if (!silent) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot find mapping for codec \"%s\"!\n",
                                        get_name_to_audio_codec(audio_codec));
                }
                return NULL;
        } else {
                codec_id = it->id;
                preferred_encoder = it->preferred_encoder;
        }

        if (get_commandline_param("audio-lavc-encoder") != NULL) {
                preferred_encoder = get_commandline_param("audio-lavc-encoder");
        }

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        avcodec_register_all();
#endif

        struct libavcodec_codec_state *s = calloc(1, sizeof *s);
        s->magic = MAGIC;
        s->codec_info = it;
        s->direction = direction;
        if(direction == AUDIO_CODER) {
                if (preferred_encoder) {
                        s->codec = avcodec_find_encoder_by_name(preferred_encoder);
                        if (s->codec && s->codec->id != codec_id) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested encoder cannot handle specified codec!\n");
                                free(s);
                                return NULL;
                        }
                }
                if (!s->codec) {
                        s->codec = avcodec_find_encoder(codec_id);
                }
        } else {
                const char *pref_dec = get_commandline_param("audio-lavc-decoder");
                if (pref_dec != NULL) {
                        s->codec = avcodec_find_decoder_by_name(pref_dec);
                        if (!s->codec || s->codec->id != codec_id) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested decoder '%s' %s\n", pref_dec,
                                                s->codec ? "cannot handle received codec" : "not found");
                                handle_error(EXIT_FAIL_USAGE);
                                s->codec = NULL;
                        }
                }
                if (!s->codec) {
                        s->codec = avcodec_find_decoder(codec_id);
                }
        }
        if(!s->codec) {
                if (!silent) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Your Libavcodec build doesn't contain codec \"%s\".\n",
                                get_name_to_audio_codec(audio_codec));
                }
                free(s);
                return NULL;
        }

        if (!silent) {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Using audio %scoder: %s\n",
                        (direction == AUDIO_CODER ? "en" : "de"), s->codec->name);
        }

        s->codec_ctx = avcodec_alloc_context3(s->codec);
        if(!s->codec_ctx) { // not likely :)
                if (!silent) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not allocate audio codec context\n");
                }
                free(s);
                return NULL;
        }
        s->codec_ctx->strict_std_compliance = -2;

        s->bitrate = bitrate;

        s->av_frame = av_frame_alloc();
        s->pkt = av_packet_alloc();

        s->tmp.data = s->tmp_data;
        s->output_channel.data = s->output_channel_data;

        if(direction == AUDIO_CODER) {
                s->output_channel.codec = audio_codec;
        } else {
                s->output_channel.codec = AC_PCM;
        }

        return s;
}

/* check that a given sample format is supported by the encoder */
static int check_sample_fmt(const AVCodec *codec, enum AVSampleFormat sample_fmt)
{
    const enum AVSampleFormat *p = codec->sample_fmts;

    while (*p != AV_SAMPLE_FMT_NONE) {
        if (*p == sample_fmt)
            return 1;
        p++;
    }
    return 0;
}

static bool reinitialize_encoder(struct libavcodec_codec_state *s, struct audio_desc desc)
{
        cleanup_common(s);

        s->codec_ctx = avcodec_alloc_context3(s->codec);
        if (s->codec_ctx == NULL) { // not likely :)
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not allocate audio codec context\n");
                return false;
        }
        s->codec_ctx->strict_std_compliance = -2;

        /*  put sample parameters */
        if (s->bitrate > 0) {
                s->codec_ctx->bit_rate = s->bitrate;
        }
        s->codec_ctx->sample_rate = desc.sample_rate;

        enum AVSampleFormat sample_fmts[AV_SAMPLE_FMT_NB];
        int count = 0;

        switch(desc.bps) {
                case 1:
                        sample_fmts[count++] = AV_SAMPLE_FMT_U8;
                        sample_fmts[count++] = AV_SAMPLE_FMT_U8P;
                        break;
                case 2:
                        sample_fmts[count++] = AV_SAMPLE_FMT_S16;
                        sample_fmts[count++] = AV_SAMPLE_FMT_S16P;
                        break;
                case 3:
                case 4:
                        sample_fmts[count++] = AV_SAMPLE_FMT_S32;
                        sample_fmts[count++] = AV_SAMPLE_FMT_S32P;
                        sample_fmts[count++] = AV_SAMPLE_FMT_FLT;
                        sample_fmts[count++] = AV_SAMPLE_FMT_FLTP;
                        break;
        }

        s->codec_ctx->sample_fmt = AV_SAMPLE_FMT_NONE;

        for (int i = 0; i < count; ++i) {
                if (check_sample_fmt(s->codec, sample_fmts[i])) {
                        s->codec_ctx->sample_fmt = sample_fmts[i];
                        break;
                }
        }

        if (s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_NONE) {
                int i = 0;
                while (s->codec->sample_fmts[i] != AV_SAMPLE_FMT_NONE) {
                        if (s->codec->sample_fmts[i] != AV_SAMPLE_FMT_DBL &&
                                        s->codec->sample_fmts[i] != AV_SAMPLE_FMT_DBLP) {
                                s->codec_ctx->sample_fmt = s->codec->sample_fmts[i];
                                break;
                        }
                        i++;
                }
        }

        if (s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_NONE) {
                log_msg(LOG_LEVEL_ERROR, "[Libavcodec] Unsupported audio sample!\n");
                return false;
        }

        AVCODECCTX_CHANNELS(s->codec_ctx) = 1;
#if FF_API_NEW_CHANNEL_LAYOUT
        s->codec_ctx->ch_layout = (struct AVChannelLayout) AV_CHANNEL_LAYOUT_MONO;
#else
        s->codec_ctx->channel_layout = AV_CH_LAYOUT_MONO;
#endif

        if (strcmp(s->codec->name, "libopus") == 0) {
                int ret = av_opt_set(s->codec_ctx->priv_data, "application", "lowdelay", 0);
                if (ret != 0) {
                        print_libav_audio_error(LOG_LEVEL_WARNING, "Could not set Opus low delay app type", ret);
                }
        } else if (strcmp(s->codec->name, "opus") == 0) {
                char warn[] = MOD_NAME "Native FFmpeg Opus encoder seems to be currently broken "
                                "with UltraGrid. You may be able to use 'libopus' encoder instead. Please let "
                                "us know to " PACKAGE_BUGREPORT " if you either want to use the native encoder "
                                "or it even works for you.\n";
                log_msg(LOG_LEVEL_WARNING, "%s", wrap_paragraph(warn));
                int ret = av_opt_set_double(s->codec_ctx->priv_data, "opus_delay", 5, 0);
                if (ret != 0) {
                        print_libav_audio_error(LOG_LEVEL_WARNING, "Cannot set Opus delay to 5", ret);
                }
        }

        if (s->direction == AUDIO_CODER && (get_commandline_param("low-latency-audio") != NULL
                                || get_commandline_param("audioenc-frame-duration") != NULL)) {
                double frame_duration = get_commandline_param("audioenc-frame-duration") == NULL ?
                        LOW_LATENCY_AUDIOENC_FRAME_DURATION : atof(get_commandline_param("audioenc-frame-duration"));
                if (s->codec->id == AV_CODEC_ID_OPUS) {
                        int ret = av_opt_set_double(s->codec_ctx->priv_data, "frame_duration", frame_duration, 0);
                        if (ret != 0) {
                                print_libav_audio_error(LOG_LEVEL_ERROR, "Could not set Opus frame duration", ret);
                        }
                }
                if (s->codec->id == AV_CODEC_ID_FLAC) {
                        s->codec_ctx->frame_size = desc.sample_rate * frame_duration / MS_IN_SEC;
                }
        }

        /* open it */
        int ret = avcodec_open2(s->codec_ctx, s->codec, NULL);
        if (ret != 0) {
                print_libav_audio_error(LOG_LEVEL_ERROR, "Could not open codec", ret);
                return false;
        }

        if(s->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) {
                s->codec_ctx->frame_size = 1;
        }

        s->av_frame->nb_samples     = s->codec_ctx->frame_size;
        s->av_frame->format         = s->codec_ctx->sample_fmt;
#if FF_API_NEW_CHANNEL_LAYOUT
        s->av_frame->ch_layout = (struct AVChannelLayout) AV_CHANNEL_LAYOUT_MONO;
#else
        s->av_frame->channel_layout = AV_CH_LAYOUT_MONO;
#endif
        s->av_frame->sample_rate    = s->codec_ctx->sample_rate;

        ret = av_frame_get_buffer(s->av_frame, 0);
        if (ret != 0) {
                print_libav_audio_error(LOG_LEVEL_ERROR, "Could not allocate audio data buffers", ret);
                return false;
        }

        s->output_channel.sample_rate = desc.sample_rate;
        s->output_channel.bps = av_get_bytes_per_sample(s->codec_ctx->sample_fmt);
        s->saved_desc = desc;

        s->context_initialized = true;

        return true;
}

static bool reinitialize_decoder(struct libavcodec_codec_state *s, struct audio_desc desc)
{
        cleanup_common(s);

        s->codec_ctx = avcodec_alloc_context3(s->codec);
        if (s->codec_ctx == NULL) { // not likely :)
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not allocate audio codec context\n");
                return false;
        }
        s->codec_ctx->strict_std_compliance = -2;

        AVCODECCTX_CHANNELS(s->codec_ctx) = 1;

        s->codec_ctx->bits_per_coded_sample = 4; // ADPCM
        s->codec_ctx->sample_rate = desc.sample_rate;

        /* open it */
        if (avcodec_open2(s->codec_ctx, s->codec, NULL) < 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Could not open codec\n");
                return false;
        }

        s->saved_desc = desc;

        s->context_initialized = true;

        return true;
}

static audio_channel *libavcodec_compress(void *state, audio_channel * channel)
{
        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *) state;
        assert(s->magic == MAGIC);

        assert(s->codec_ctx->sample_fmt != AV_SAMPLE_FMT_DBL && // not supported yet
                        s->codec_ctx->sample_fmt != AV_SAMPLE_FMT_DBLP);

        if(channel) {
                if(!audio_desc_eq(s->saved_desc, audio_desc_from_audio_channel(channel))) {
                        if(!reinitialize_encoder(s, audio_desc_from_audio_channel(channel))) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to reinitialize audio compress!\n");
                                return NULL;
                        }
                }

                if (s->output_channel.bps != channel->bps || s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLT || s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLTP) {
                        if (s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLT || s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLTP) {
                                if (s->output_channel.bps == channel->bps) {
                                        if (s->tmp.data_len + channel->data_len > TMP_DATA_LEN) {
                                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Auxiliary buffer overflow!\n");
                                        } else {
                                                int2float(s->tmp_data + s->tmp.data_len, channel->data, channel->data_len);
                                                s->tmp.data_len += channel->data_len;
                                        }
                                } else {
                                        size_t data_len = channel->data_len / channel->bps * 4;
                                        resize_tmp_buffer(&s->tmp_buffer, &s->tmp_buffer_size, data_len);
                                        change_bps((char *) s->tmp_buffer, 4, channel->data, channel->bps, channel->data_len);
                                        if (s->tmp.data_len + data_len > TMP_DATA_LEN) {
                                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Auxiliary buffer overflow!\n");
                                        } else {
                                                int2float(s->tmp_data + s->tmp.data_len, (char *) s->tmp_buffer, data_len);
                                                s->tmp.data_len += data_len;
                                        }
                                }
                        } else {
                                change_bps(s->tmp_data + s->tmp.data_len, s->output_channel.bps,
                                                channel->data, s->saved_desc.bps, channel->data_len);
                                s->tmp.data_len += channel->data_len / s->saved_desc.bps * s->output_channel.bps;
                        }
                } else {
                        memcpy(s->tmp_data + s->tmp.data_len, channel->data, channel->data_len);
                        s->tmp.data_len += channel->data_len;
                }
        }

        int bps = s->output_channel.bps;
        int offset = 0;
        s->output_channel.data_len = 0;
        s->output_channel.duration = 0.0;
        int chunk_size = s->codec_ctx->frame_size * bps;
        while(offset + chunk_size <= s->tmp.data_len) {
                if (bps == 1) {
                        signed2unsigned((char *) s->av_frame->data[0], s->tmp.data + offset, chunk_size);
                } else {
                        memcpy(s->av_frame->data[0], s->tmp.data + offset, chunk_size);
                }
		int ret = avcodec_send_frame(s->codec_ctx, s->av_frame);
                if (ret != 0) {
                        print_libav_audio_error(LOG_LEVEL_ERROR, "Error encoding frame", ret);
                        return NULL;
                }
                ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
                while (ret == 0) {
                        if (s->output_channel.data_len + s->pkt->size >
                            TMP_DATA_LEN) {
                                MSG(ERROR, "Output buffer overflow!\n");
                                return NULL;
                        }
                        memcpy(s->output_channel_data +
                                   s->output_channel.data_len,
                               s->pkt->data, s->pkt->size);
                        s->output_channel.data_len += s->pkt->size;
                        av_packet_unref(s->pkt);
                        ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
                        s->output_channel.duration +=
                            s->codec_ctx->frame_size /
                            (double) s->output_channel.sample_rate;
                }
                if (ret != AVERROR(EAGAIN) && ret != 0) {
                        print_libav_audio_error(LOG_LEVEL_WARNING,
                                                "Receive packet error", ret);
                }
                offset += chunk_size;
                // since 2023-04-25, this may not be necessary (for AAC/MP3, it just triggers "Multiple frames in a packet." It seems not
                // working only with native Opus encoder+decoder combination (which doesn't work anyways now). Consider removing this
                // after some decent period (UG prior that date won't decode all frames for AAC when removed).
                if(!(s->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) && s->output_channel.data_len > 0)
                        break;
        }

        s->tmp.data_len -= offset;
        memmove(s->tmp_data, s->tmp.data + offset, s->tmp.data_len);

        ///fprintf(stderr, "%d %d\n", i++% 2, s->output_channel.data_len);
        if(s->output_channel.data_len) {
                return &s->output_channel;
        } else {
                return NULL;
        }
}

static audio_channel *
libavcodec_decompress(void *state, audio_channel *channel,
                      struct packet_iterator *it)
{
        UNUSED(it);
        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *) state;
        assert(channel->data_len > 0);
        assert(s->magic == MAGIC);

        if(!audio_desc_eq(s->saved_desc, audio_desc_from_audio_channel(channel))) {
                if(!reinitialize_decoder(s, audio_desc_from_audio_channel(channel))) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to reinitialize audio decompress!\n");
                        return NULL;
                }
        }

        // FFMPEG buffer needs to be FF_INPUT_BUFFER_PADDING_SIZE longer than data
        resize_tmp_buffer(&s->tmp_buffer, &s->tmp_buffer_size, channel->data_len + AV_INPUT_BUFFER_PADDING_SIZE);
        memcpy(s->tmp_buffer, channel->data, channel->data_len);

        if ((s->codec_info->flags & PACKETIZED_DEC) != 0)  {
                s->pkt->data = s->tmp_buffer + it->offset;
                s->pkt->size = it->len;
        } else {
                s->pkt->data = s->tmp_buffer;
                s->pkt->size = channel->data_len;
        }
        s->output_channel.data_len = 0;

        av_frame_unref(s->av_frame);

        while (1) {
                int ret = avcodec_send_packet(s->codec_ctx, s->pkt);
                if (ret != 0) {
                        print_decoder_error(
                            MOD_NAME "error sending decoded frame -", ret);
                        return NULL;
                }
                /* read all the output frames (in general there may be any
                 * number of them */
                while (ret >= 0) {
                        ret = avcodec_receive_frame(s->codec_ctx, s->av_frame);
                        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                                break;
                        }
                        if (ret < 0) {
                                print_decoder_error(
                                    MOD_NAME "error receiving decoded frame -",
                                    ret);
                                return NULL;
                        }
                        int channels  = 1;
                        /* if a frame has been decoded, output it */
                        int data_size = av_samples_get_buffer_size(
                            NULL, channels, s->av_frame->nb_samples,
                            s->codec_ctx->sample_fmt, 1);
                        memcpy(s->output_channel_data +
                                   s->output_channel.data_len,
                               s->av_frame->data[0], data_size);
                        s->output_channel.data_len += data_size;
                        s->pkt->dts = s->pkt->pts = AV_NOPTS_VALUE;
                }
                if ((s->codec_info->flags & PACKETIZED_DEC) == 0) {
                        break;
                }
                if (!packet_next(it)) {
                        break;
                }
                s->pkt->data = s->tmp_buffer + it->offset;
                s->pkt->size = it->len;
        }

        //
        // perform needed conversions (float->int32, int32->dest_bps)
        //
        assert(s->codec_ctx->sample_fmt != AV_SAMPLE_FMT_DBL && // not supported yet
                        s->codec_ctx->sample_fmt != AV_SAMPLE_FMT_DBLP);

        // convert from float if needed
        if (s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLT ||
                        s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLTP) {
                float2int(s->output_channel_data, s->output_channel.data, s->output_channel.data_len);
        } else if (s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_U8) {
                signed2unsigned(s->output_channel_data, s->output_channel.data, s->output_channel.data_len);
        }

        s->output_channel.bps = av_get_bytes_per_sample(s->codec_ctx->sample_fmt);
        s->output_channel.sample_rate = s->codec_ctx->sample_rate;

        return &s->output_channel;
}

static const int *libavcodec_get_sample_rates(void *state)
{
        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *) state;

        return s->codec->supported_samplerates;
}

static void cleanup_common(struct libavcodec_codec_state *s)
{
        if (s->context_initialized) {
                if (s->direction == AUDIO_DECODER) {
                        lavd_flush(s->codec_ctx);
                } else {
                        int ret;
                        ret = avcodec_send_frame(s->codec_ctx, NULL);
                        if (ret != 0) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                                ret);
                        }
                        do {
                                ret = avcodec_receive_packet(s->codec_ctx, s->pkt);
                                av_packet_unref(s->pkt);
                        } while (ret >= 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN));
                        if (ret < 0 && ret != AVERROR_EOF && ret != AVERROR(EAGAIN)) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                                ret);
                        }
                }
        }

        avcodec_free_context(&s->codec_ctx);

        s->context_initialized = false;
}

static void libavcodec_done(void *state)
{
        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *) state;
        assert(s->magic == MAGIC);

        cleanup_common(s);

        av_frame_free(&s->av_frame);
        av_packet_free(&s->pkt);

        free(s->tmp_buffer);
        free(s);
}

static const audio_codec_t supported_codecs[] = { AC_ALAW, AC_MULAW, AC_SPEEX, AC_OPUS, AC_G722, AC_FLAC, AC_MP3, AC_AAC, AC_NONE };

static const struct audio_compress_info libavcodec_audio_codec = {
        supported_codecs,
        libavcodec_init,
        libavcodec_compress,
        libavcodec_decompress,
        libavcodec_get_sample_rates,
        libavcodec_done
};

REGISTER_MODULE(libavcodec,  &libavcodec_audio_codec, LIBRARY_CLASS_AUDIO_COMPRESS, AUDIO_COMPRESS_ABI_VERSION);

/* vim: set expandtab sw=8 : */
