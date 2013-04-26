/*
 * FILE:    audio/codec/libavcodec.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "audio/codec/libavcodec.h"

#include <libavcodec/avcodec.h>
#include <libavutil/channel_layout.h>
#include <libavutil/mem.h>

#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"
#include "libavcodec_common.h"
#include "utils/resource_manager.h"

#define MAGIC 0xb135ca11

#ifndef HAVE_AVCODEC_ENCODE_VIDEO2
#define AV_CODEC_ID_PCM_ALAW CODEC_ID_PCM_ALAW
#define AV_CODEC_ID_PCM_MULAW CODEC_ID_PCM_MULAW
#define AV_CODEC_ID_ADPCM_IMA_WAV CODEC_ID_ADPCM_IMA_WAV
#define AV_CODEC_ID_SPEEX CODEC_ID_SPEEX
#define AV_CODEC_ID_OPUS CODEC_ID_OPUS
#define AV_CODEC_ID_ADPCM_G722 CODEC_ID_ADPCM_G722
#define AV_CODEC_ID_ADPCM_G726 CODEC_ID_ADPCM_G726
#endif

static void *libavcodec_init(audio_codec_t audio_codec, audio_codec_direction_t direction,
                bool try_init);
static audio_channel *libavcodec_compress(void *, audio_channel *);
static audio_channel *libavcodec_decompress(void *, audio_channel *);
static void libavcodec_done(void *);

static void init(void) __attribute__((constructor));

static void init(void)
{
        register_audio_codec(&libavcodec_audio_codec);
}

typedef struct {
        int codec_id;
} audio_codec_t_to_codec_id_mapping_t;

static const audio_codec_t_to_codec_id_mapping_t mapping[] = 
{
        [AC_ALAW] = { .codec_id = AV_CODEC_ID_PCM_ALAW },
        [AC_MULAW] = { .codec_id = AV_CODEC_ID_PCM_MULAW },
        [AC_ADPCM_IMA_WAV] = { .codec_id = AV_CODEC_ID_ADPCM_IMA_WAV },
        [AC_SPEEX] = { .codec_id = AV_CODEC_ID_SPEEX },
        [AC_OPUS] = { .codec_id = AV_CODEC_ID_OPUS },
        [AC_G722] = { .codec_id = AV_CODEC_ID_ADPCM_G722 },
        [AC_G726] = { .codec_id = AV_CODEC_ID_ADPCM_G726 },
};

struct libavcodec_codec_state {
        uint32_t magic;
        pthread_mutex_t    *libav_global_lock;
        AVCodecContext     *codec_ctx;
        AVCodec            *codec;

        AVPacket            pkt;
        AVFrame            *av_frame;

        struct audio_desc   saved_desc;

        audio_channel       tmp;
        audio_channel       output_channel;

        void               *samples;
        int                 change_bps_to;
};

static void *libavcodec_init(audio_codec_t audio_codec, audio_codec_direction_t direction, bool try_init)
{
        int codec_id = 0;
        
        if(audio_codec <= sizeof(mapping) / sizeof(audio_codec_t_to_codec_id_mapping_t)) {
                codec_id = mapping[audio_codec].codec_id;
        }
        if(codec_id == 0) {
                try_init || fprintf(stderr, "[Libavcodec] Cannot find mapping for codec \"%s\"!\n",
                                get_name_to_audio_codec(audio_codec));
                return NULL;
        }

        avcodec_register_all();

        struct libavcodec_codec_state *s = calloc(1, sizeof(struct libavcodec_codec_state));
        if(direction == AUDIO_CODER) {
                s->codec = avcodec_find_encoder(codec_id);
        } else {
                s->codec = avcodec_find_decoder(codec_id);
        }
        if(!s->codec) {
                try_init || fprintf(stderr, "Your Libavcodec build doesn't contain codec \"%s\".\n",
                                get_name_to_audio_codec(audio_codec));
                return NULL;
        }

        s->magic = MAGIC;
        s->libav_global_lock = rm_acquire_shared_lock(LAVCD_LOCK_NAME);
        s->codec_ctx = avcodec_alloc_context3(s->codec);
        if(!s->codec_ctx) { // not likely :)
                try_init || fprintf(stderr, "Could not allocate audio codec context\n");
                return NULL;
        }

        s->samples = NULL;

        av_init_packet(&s->pkt);
        s->pkt.size = 0;
        s->pkt.data = NULL;

        s->av_frame = avcodec_alloc_frame();

        memset(&s->tmp, 0, sizeof(audio_channel));
        memset(&s->output_channel, 0, sizeof(audio_channel));
        s->tmp.data = malloc(1024*1024);
        s->output_channel.data = malloc(1024*1024);

        if(direction == AUDIO_CODER) {
                s->output_channel.codec = audio_codec;
        } else {
                s->output_channel.codec = AC_PCM;
        }

        return s;
}

/* check that a given sample format is supported by the encoder */
static int check_sample_fmt(AVCodec *codec, enum AVSampleFormat sample_fmt)
{
    const enum AVSampleFormat *p = codec->sample_fmts;

    while (*p != AV_SAMPLE_FMT_NONE) {
        if (*p == sample_fmt)
            return 1;
        p++;
    }
    return 0;
}

static bool reinitialize_coder(struct libavcodec_codec_state *s, struct audio_desc desc)
{
        av_freep(&s->samples);
        pthread_mutex_lock(s->libav_global_lock);
        avcodec_close(s->codec_ctx);
        pthread_mutex_unlock(s->libav_global_lock);

        /*  put sample parameters */
        s->codec_ctx->bit_rate = 64000;
        s->codec_ctx->sample_rate = desc.sample_rate;
        s->change_bps_to = 0;
        switch(desc.bps) {
                case 1:
                        s->codec_ctx->sample_fmt = AV_SAMPLE_FMT_U8;
                        break;
                case 2:
                        s->codec_ctx->sample_fmt = AV_SAMPLE_FMT_S16;
                        break;
                case 3:
                        s->change_bps_to = 4;
                case 4:
                        s->codec_ctx->sample_fmt = AV_SAMPLE_FMT_S32;
                        break;
        }

        if(!check_sample_fmt(s->codec, s->codec_ctx->sample_fmt)) {
                s->codec_ctx->sample_fmt = s->codec->sample_fmts[0];
                s->change_bps_to = av_get_bytes_per_sample(s->codec_ctx->sample_fmt);
        }

        s->codec_ctx->channels = 1;
        s->codec_ctx->channel_layout = AV_CH_LAYOUT_MONO;

        pthread_mutex_lock(s->libav_global_lock);
        /* open it */
        if (avcodec_open2(s->codec_ctx, s->codec, NULL) < 0) {
                fprintf(stderr, "Could not open codec\n");
                pthread_mutex_unlock(s->libav_global_lock);
                return false;
        }
        pthread_mutex_unlock(s->libav_global_lock);

        if(s->codec->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE) {
                s->codec_ctx->frame_size = 1;
        }

        s->av_frame->nb_samples     = s->codec_ctx->frame_size;
        s->av_frame->format         = s->codec_ctx->sample_fmt;
        s->av_frame->channel_layout = AV_CH_LAYOUT_MONO;

        int channels = 1;
        /* the codec gives us the frame size, in samples,
         * we calculate the size of the samples buffer in bytes */
        int buffer_size = av_samples_get_buffer_size(NULL, channels, s->codec_ctx->frame_size,
                        s->codec_ctx->sample_fmt, 0);

        s->samples = av_malloc(buffer_size);
        if (!s->samples) {
                fprintf(stderr, "could not allocate %d bytes for samples buffer\n",
                                buffer_size);
                return false;
        }
        /* setup the data pointers in the AVFrame */
        int ret = avcodec_fill_audio_frame(s->av_frame, channels, s->codec_ctx->sample_fmt,
                        (const uint8_t*)s->samples, buffer_size, 0);
        if (ret < 0) {
                fprintf(stderr, "could not setup audio frame\n");
                return false;
        }

        s->output_channel.sample_rate = desc.sample_rate;
        s->output_channel.bps = desc.bps;

        s->saved_desc = desc;

        return true;
}

static bool reinitialize_decoder(struct libavcodec_codec_state *s, struct audio_desc desc)
{
        pthread_mutex_lock(s->libav_global_lock);
        avcodec_close(s->codec_ctx);
        pthread_mutex_unlock(s->libav_global_lock);

        s->codec_ctx->channels = 1;

        s->codec_ctx->bits_per_coded_sample = 4; // ADPCM
        s->codec_ctx->sample_rate = desc.sample_rate;

        pthread_mutex_lock(s->libav_global_lock);
        /* open it */
        if (avcodec_open2(s->codec_ctx, s->codec, NULL) < 0) {
                fprintf(stderr, "Could not open codec\n");
                pthread_mutex_unlock(s->libav_global_lock);
                return false;
        }
        pthread_mutex_unlock(s->libav_global_lock);

        s->output_channel.sample_rate = desc.sample_rate;
        s->output_channel.bps = desc.bps;
        s->saved_desc = desc;

        s->output_channel.bps = av_get_bytes_per_sample(s->codec_ctx->sample_fmt);

        return true;
}

static audio_channel *libavcodec_compress(void *state, audio_channel * channel)
{
        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *) state;
        assert(s->magic == MAGIC);

        if(channel) {
                if(!audio_desc_eq(s->saved_desc, audio_desc_from_audio_channel(channel))) {
                        if(!reinitialize_coder(s, audio_desc_from_audio_channel(channel))) {
                                fprintf(stderr, "Unable to reinitialize audio compress!\n");
                                return NULL;
                        }
                }

                if(s->change_bps_to) {
                        change_bps(s->tmp.data, s->saved_desc.bps, channel->data,
                                        s->change_bps_to, channel->data_len);
                        s->tmp.data_len += channel->data_len / s->saved_desc.bps * s->change_bps_to;
                } else {
                        memcpy(s->tmp.data + s->tmp.data_len, channel->data, channel->data_len);
                        s->tmp.data_len += channel->data_len;
                }
        }

        int bps = s->output_channel.bps;
        int offset = 0;
        s->output_channel.data_len = 0;
        int chunk_size = s->codec_ctx->frame_size * bps;
        //while(offset + chunk_size <= s->tmp.data_len) {
        while(offset + chunk_size <= s->tmp.data_len) {
                s->pkt.data = (unsigned char *) s->output_channel.data + s->output_channel.data_len;
                s->pkt.size = 1024*1024 - s->output_channel.data_len;
                int got_packet;
                memcpy(s->samples, s->tmp.data + offset, chunk_size);
                int ret = avcodec_encode_audio2(s->codec_ctx, &s->pkt, s->av_frame,
                                &got_packet);
                if(ret) {
                        char errbuf[1024];
                        av_strerror(ret, errbuf, sizeof(errbuf));
                        fprintf(stderr, "Warning: unable to compress audio: %s\n",
                                        errbuf);
                }
                if(got_packet) {
                        s->output_channel.data_len += s->pkt.size;
                }
                offset += chunk_size;
                if(!(s->codec->capabilities & CODEC_CAP_VARIABLE_FRAME_SIZE))
                        break;
        }

        s->tmp.data_len -= offset;
        memmove(s->tmp.data, s->tmp.data + offset, s->tmp.data_len);

        ///fprintf(stderr, "%d %d\n", i++% 2, s->output_channel.data_len);
        if(s->output_channel.data_len) {
                return &s->output_channel;
        } else {
                return NULL;
        }
}

static audio_channel *libavcodec_decompress(void *state, audio_channel * channel)
{
        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *) state;
        assert(s->magic == MAGIC);

        if(!audio_desc_eq(s->saved_desc, audio_desc_from_audio_channel(channel))) {
                if(!reinitialize_decoder(s, audio_desc_from_audio_channel(channel))) {
                        fprintf(stderr, "Unable to reinitialize audio decompress!\n");
                        return NULL;
                }
        }

        int offset = 0;
        s->pkt.data = (unsigned char *) channel->data;
        s->pkt.size = channel->data_len;
        s->output_channel.data_len = 0;
        while (s->pkt.size > 0) {
                int got_frame = 0;

                avcodec_get_frame_defaults(s->av_frame);

                int len = avcodec_decode_audio4(s->codec_ctx, s->av_frame, &got_frame,
                                &s->pkt);
                if (len < 0) {
                        fprintf(stderr, "Error while decoding\n");
                        return NULL;
                }
                if (got_frame) {
                        int channels = 1;
                        /* if a frame has been decoded, output it */
                        int data_size = av_samples_get_buffer_size(NULL, channels,
                                        s->av_frame->nb_samples,
                                        s->codec_ctx->sample_fmt, 1);
                        memcpy(s->output_channel.data + offset, s->av_frame->data[0],
                                        data_size);
                        offset += len;
                        s->output_channel.data_len += data_size;
                }
                s->pkt.size -= len;
                s->pkt.data += len;
#if 0
                if (s->pkt.size < AUDIO_REFILL_THRESH) {
                        /* Refill the input buffer, to avoid trying to decode
                         * incomplete frames. Instead of this, one could also use
                         * a parser, or use a proper container format through
                         * libavformat. */
                        memmove(inbuf, avpkt.data, avpkt.size);
                        avpkt.data = inbuf;
                        len = fread(avpkt.data + avpkt.size, 1,
                                        AUDIO_INBUF_SIZE - avpkt.size, f);
                        if (len > 0)
                                avpkt.size += len;
                }
#endif
        }

        return &s->output_channel;
}

static void libavcodec_done(void *state)
{
        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *) state;
        assert(s->magic == MAGIC);

        pthread_mutex_lock(s->libav_global_lock);
        avcodec_close(s->codec_ctx);
        pthread_mutex_unlock(s->libav_global_lock);

        rm_release_shared_lock(LAVCD_LOCK_NAME);
        free(s->output_channel.data);
        free(s->tmp.data);
        av_free_packet(&s->pkt);
        av_freep(&s->samples);
        avcodec_free_frame(&s->av_frame);

        free(s);
}

struct audio_codec libavcodec_audio_codec = {
        .supported_codecs = (audio_codec_t[]){ AC_ALAW, AC_MULAW, AC_ADPCM_IMA_WAV, AC_SPEEX, AC_OPUS, AC_G722, AC_G726, AC_NONE },
        .supported_bytes_per_second = (int[]){ 2, 0 },
        .init = libavcodec_init,
        .compress = libavcodec_compress,
        .decompress = libavcodec_decompress,
        .done = libavcodec_done
};

