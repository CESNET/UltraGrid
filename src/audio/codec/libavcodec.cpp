/**
 * @file   audio/codec/libavcodec.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2015 CESNET z.s.p.o.
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

#include <memory>

extern "C" {
#include <libavcodec/avcodec.h>
#if LIBAVCODEC_VERSION_MAJOR >= 54
#include <libavutil/channel_layout.h>
#endif
#include <libavutil/mem.h>
}

#include <vector>
#include <unordered_map>
#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"
#include "libavcodec_common.h"
#include "utils/resource_manager.h"

#define MAGIC 0xb135ca11

#if LIBAVCODEC_VERSION_MAJOR < 54
#define AV_CODEC_ID_AAC CODEC_ID_AAC
#define AV_CODEC_ID_PCM_ALAW CODEC_ID_PCM_ALAW
#define AV_CODEC_ID_PCM_MULAW CODEC_ID_PCM_MULAW
#define AV_CODEC_ID_SPEEX CODEC_ID_SPEEX
#define AV_CODEC_ID_OPUS CODEC_ID_OPUS
#define AV_CODEC_ID_ADPCM_G722 CODEC_ID_ADPCM_G722
#define AV_CODEC_ID_FLAC CODEC_ID_FLAC
#define AV_CODEC_ID_MP3 CODEC_ID_MP3
#endif

#define MOD_NAME "[lavd] "

using namespace std;

static void *libavcodec_init(audio_codec_t audio_codec, audio_codec_direction_t direction,
                bool silent, int bitrate);
static audio_channel *libavcodec_compress(void *, audio_channel *);
static audio_channel *libavcodec_decompress(void *, audio_channel *);
static void libavcodec_done(void *);
static void cleanup_common(struct libavcodec_codec_state *s);

struct codec_param {
        AVCodecID id;
        const char *preferred_encoder;
};

static std::unordered_map<audio_codec_t, codec_param, std::hash<int>> mapping {
        { AC_ALAW, {AV_CODEC_ID_PCM_ALAW, NULL} },
        { AC_MULAW, {AV_CODEC_ID_PCM_MULAW, NULL} },
        { AC_SPEEX, {AV_CODEC_ID_SPEEX, NULL} },
#if LIBAVCODEC_VERSION_MAJOR >= 54
        { AC_OPUS, {AV_CODEC_ID_OPUS, NULL} },
#endif
        { AC_G722, {AV_CODEC_ID_ADPCM_G722, NULL} },
        { AC_FLAC, {AV_CODEC_ID_FLAC, NULL} },
        { AC_MP3, {AV_CODEC_ID_MP3, NULL} },
        { AC_AAC, {AV_CODEC_ID_AAC, "libfdk_aac"} },
};

struct libavcodec_codec_state {
        uint32_t magic;
        pthread_mutex_t    *libav_global_lock;
        AVCodecContext     *codec_ctx;
        AVCodec            *codec;

        AVFrame            *av_frame;

        struct audio_desc   saved_desc;

        audio_channel       tmp;
        char               *tmp_data; ///< tmp.data, but non-const qualified
        audio_channel       output_channel;
        char               *output_channel_data; ///< output_channel.data, but non-const qualified

        void               *samples;

        int                 bitrate;

        bool                context_initialized;
        audio_codec_direction_t direction;
};

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
        enum AVCodecID codec_id = AV_CODEC_ID_NONE;

        auto it = mapping.find(audio_codec);
        const char *preferred_encoder = NULL;
        
        if (it == mapping.end()) {
                if (!silent) {
                        fprintf(stderr, "[Libavcodec] Cannot find mapping for codec \"%s\"!\n",
                                        get_name_to_audio_codec(audio_codec));
                }
                return NULL;
        } else {
                codec_id = it->second.id;
                preferred_encoder = it->second.preferred_encoder;
        }

#if LIBAVCODEC_VERSION_INT <= AV_VERSION_INT(58, 9, 100)
        avcodec_register_all();
#endif

        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *)
                calloc(1, sizeof(struct libavcodec_codec_state));
        s->direction = direction;
        if(direction == AUDIO_CODER) {
                if (preferred_encoder) {
                        s->codec = avcodec_find_encoder_by_name(preferred_encoder);
                }
                if (!s->codec) {
                        s->codec = avcodec_find_encoder(codec_id);
                }
        } else {
                s->codec = avcodec_find_decoder(codec_id);
        }
        if(!s->codec) {
                if (!silent) {
                        fprintf(stderr, "Your Libavcodec build doesn't contain codec \"%s\".\n",
                                get_name_to_audio_codec(audio_codec));
                }
                free(s);
                return NULL;
        } else {
                if (!silent) {
                        log_msg(LOG_LEVEL_NOTICE, "[lavc] Using audio %scoder: %s\n", s->codec->name,
                                        direction == AUDIO_CODER ? "en" : "de");
                }
        }

        s->magic = MAGIC;
        s->libav_global_lock = rm_acquire_shared_lock(LAVCD_LOCK_NAME);
        s->codec_ctx = avcodec_alloc_context3(s->codec);
        if(!s->codec_ctx) { // not likely :)
                if (!silent) {
                        fprintf(stderr, "Could not allocate audio codec context\n");
                }
                free(s);
                return NULL;
        }

        s->codec_ctx->strict_std_compliance = -2;

        s->bitrate = bitrate;

        s->samples = NULL;

        s->av_frame = av_frame_alloc();

        memset(&s->tmp, 0, sizeof(audio_channel));
        memset(&s->output_channel, 0, sizeof(audio_channel));
        s->tmp_data = (char *) malloc(1024*1024);
        s->tmp.data = s->tmp_data;
        s->output_channel_data = (char *) malloc(1024*1024);
        s->output_channel.data = s->output_channel_data;

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
        cleanup_common(s);

        av_freep(&s->samples);
        pthread_mutex_lock(s->libav_global_lock);
        avcodec_close(s->codec_ctx);
        pthread_mutex_unlock(s->libav_global_lock);

        /*  put sample parameters */
        if (s->bitrate > 0) {
                s->codec_ctx->bit_rate = s->bitrate;
        }
        s->codec_ctx->sample_rate = desc.sample_rate;

        vector<enum AVSampleFormat> sample_fmts;

        switch(desc.bps) {
                case 1:
                        sample_fmts.push_back(AV_SAMPLE_FMT_U8);
                        sample_fmts.push_back(AV_SAMPLE_FMT_U8P);
                        break;
                case 2:
                        sample_fmts.push_back(AV_SAMPLE_FMT_S16);
                        sample_fmts.push_back(AV_SAMPLE_FMT_S16P);
                        break;
                case 3:
                case 4:
                        sample_fmts.push_back(AV_SAMPLE_FMT_S32);
                        sample_fmts.push_back(AV_SAMPLE_FMT_S32P);
                        sample_fmts.push_back(AV_SAMPLE_FMT_FLT);
                        sample_fmts.push_back(AV_SAMPLE_FMT_FLTP);
                        break;
        }

        s->codec_ctx->sample_fmt = AV_SAMPLE_FMT_NONE;

        for (auto it = sample_fmts.begin(); it != sample_fmts.end(); ++it) {
                if (check_sample_fmt(s->codec, *it)) {
                        s->codec_ctx->sample_fmt = *it;
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

        s->codec_ctx->channels = 1;
#if LIBAVCODEC_VERSION_MAJOR >= 54
        s->codec_ctx->channel_layout = AV_CH_LAYOUT_MONO;
#endif

        pthread_mutex_lock(s->libav_global_lock);
        /* open it */
        if (avcodec_open2(s->codec_ctx, s->codec, NULL) < 0) {
                fprintf(stderr, "Could not open codec\n");
                pthread_mutex_unlock(s->libav_global_lock);
                return false;
        }
        pthread_mutex_unlock(s->libav_global_lock);

        if(s->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE) {
                s->codec_ctx->frame_size = 1;
        }

        s->av_frame->nb_samples     = s->codec_ctx->frame_size;
        s->av_frame->format         = s->codec_ctx->sample_fmt;
#if LIBAVCODEC_VERSION_MAJOR >= 54
        s->av_frame->channel_layout = AV_CH_LAYOUT_MONO;
        s->av_frame->sample_rate    = s->codec_ctx->sample_rate;
#endif

        int channels = 1;
        /* the codec gives us the frame size, in samples,
         * we calculate the size of the samples buffer in bytes */
        int buffer_size = av_samples_get_buffer_size(NULL, channels, s->codec_ctx->frame_size,
                        s->codec_ctx->sample_fmt, 1);

        s->samples = av_malloc(buffer_size);
        if (!s->samples) {
                fprintf(stderr, "could not allocate %d bytes for samples buffer\n",
                                buffer_size);
                return false;
        }
        /* setup the data pointers in the AVFrame */
        int ret = avcodec_fill_audio_frame(s->av_frame, channels, s->codec_ctx->sample_fmt,
                        (const uint8_t*)s->samples, buffer_size, 1);
        if (ret < 0) {
                fprintf(stderr, "could not setup audio frame\n");
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
                        if(!reinitialize_coder(s, audio_desc_from_audio_channel(channel))) {
                                fprintf(stderr, "Unable to reinitialize audio compress!\n");
                                return NULL;
                        }
                }

                if (s->output_channel.bps != channel->bps || s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLT || s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLTP) {
                        if (s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLT || s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLTP) {
                                if (s->output_channel.bps == channel->bps) {
                                        int2float(s->tmp_data + s->tmp.data_len, channel->data, channel->data_len);
                                        s->tmp.data_len += channel->data_len;
                                } else {
                                        size_t data_len = channel->data_len / channel->bps * 4;
                                        unique_ptr<char []> tmp(new char[data_len]);
                                        change_bps((char *) tmp.get(), 4, channel->data, channel->bps, channel->data_len);
                                        int2float(s->tmp_data + s->tmp.data_len, tmp.get(), data_len);
                                        s->tmp.data_len += data_len;
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
        //while(offset + chunk_size <= s->tmp.data_len) {
        while(offset + chunk_size <= s->tmp.data_len) {
		memcpy(s->samples, s->tmp.data + offset, chunk_size);
                AVPacket pkt;
                av_init_packet(&pkt);
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
		int ret = avcodec_send_frame(s->codec_ctx, s->av_frame);
		if (ret == 0) {
			ret = avcodec_receive_packet(s->codec_ctx, &pkt);
			while (ret == 0) {
				//assert(pkt.size + out->tiles[0].data_len <= s->compressed_desc.width * s->compressed_desc.height * 4 - out->tiles[0].data_len);
				memcpy(s->output_channel_data + s->output_channel.data_len,
						pkt.data, pkt.size);
				s->output_channel.data_len += pkt.size;
				av_packet_unref(&pkt);
				ret = avcodec_receive_packet(s->codec_ctx, &pkt);
                                s->output_channel.duration += s->codec_ctx->frame_size / (double) s->output_channel.sample_rate;
			}
			if (ret != AVERROR(EAGAIN) && ret != 0) {
				char errbuf[1024];
				av_strerror(ret, errbuf, sizeof(errbuf));

				log_msg(LOG_LEVEL_WARNING, "Receive packet error: %s %d\n", errbuf, ret);
			}
		} else {
			log_msg(LOG_LEVEL_WARNING, "Error encoding frame. Error = %d\n", ret);
			return {};
		}
#else
                pkt.data = (unsigned char *) s->output_channel.data + s->output_channel.data_len;
                pkt.size = 1024*1024 - s->output_channel.data_len;
                int got_packet = 0;
                int ret = avcodec_encode_audio2(s->codec_ctx, &pkt, s->av_frame,
                                &got_packet);
                if(ret) {
                        char errbuf[1024];
                        av_strerror(ret, errbuf, sizeof(errbuf));
                        fprintf(stderr, "Warning: unable to compress audio: %s\n",
                                        errbuf);
                }
                if(got_packet) {
                        s->output_channel.data_len += pkt.size;
                        s->output_channel.duration += s->codec_ctx->frame_size / (double) s->output_channel.sample_rate;
                }
#endif
                offset += chunk_size;
                if(!(s->codec->capabilities & AV_CODEC_CAP_VARIABLE_FRAME_SIZE))
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
        // FFMPEG buffer needs to be FF_INPUT_BUFFER_PADDING_SIZE longer than data
        unique_ptr<unsigned char []> tmp_buffer(new unsigned char[channel->data_len + AV_INPUT_BUFFER_PADDING_SIZE]);
        memcpy(tmp_buffer.get(), channel->data, channel->data_len);

        AVPacket pkt;
        av_init_packet(&pkt);
        pkt.data = tmp_buffer.get();
        pkt.size = channel->data_len;
        s->output_channel.data_len = 0;
        while (pkt.size > 0) {
                int got_frame = 0;

                av_frame_unref(s->av_frame);

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 37, 100)
                int len = avcodec_decode_audio4(s->codec_ctx, s->av_frame, &got_frame,
                                &pkt);
#else
                got_frame = 0;
                int ret = avcodec_send_packet(s->codec_ctx, &pkt);

                if (ret == 0) {
                        ret = avcodec_receive_frame(s->codec_ctx, s->av_frame);
                        if (ret == 0) {
                                got_frame = 1;
                        }
                }
                if (ret != 0) {
                        print_decoder_error(MOD_NAME, ret);
                }
                int len = pkt.size;
#endif

                if (len <= 0) {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Error while decoding audio\n");
                        return NULL;
                }
                if (got_frame) {
                        int channels = 1;
                        /* if a frame has been decoded, output it */
                        int data_size = av_samples_get_buffer_size(NULL, channels,
                                        s->av_frame->nb_samples,
                                        s->codec_ctx->sample_fmt, 1);
                        memcpy(s->output_channel_data + offset, s->av_frame->data[0],
                                        data_size);
                        offset += len;
                        s->output_channel.data_len += data_size;
                }
                pkt.size -= len;
                pkt.data += len;
                pkt.dts = pkt.pts = AV_NOPTS_VALUE;
#if 0
                if (pkt.size < AUDIO_REFILL_THRESH) {
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

        //
        // perform needed conversions (float->int32, int32->dest_bps)
        //
        assert(s->codec_ctx->sample_fmt != AV_SAMPLE_FMT_DBL && // not supported yet
                        s->codec_ctx->sample_fmt != AV_SAMPLE_FMT_DBLP);

        // convert from float if needed
        if (s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLT ||
                        s->codec_ctx->sample_fmt == AV_SAMPLE_FMT_FLTP) {
                unique_ptr<char []> int32_data(unique_ptr<char []>(new char [s->output_channel.data_len]));
                float2int(int32_data.get(), s->output_channel.data, s->output_channel.data_len);
                memcpy(s->output_channel_data, int32_data.get(), s->output_channel.data_len);
                s->output_channel.bps = 4;
        } else {
                s->output_channel.bps =
                        av_get_bytes_per_sample(s->codec_ctx->sample_fmt);
        }

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
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
                if (s->direction == AUDIO_DECODER) {
                        int ret;
                        ret = avcodec_send_packet(s->codec_ctx, NULL);
                        if (ret != 0) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                                ret);
                        }
                        do {
                                ret = avcodec_receive_frame(s->codec_ctx, s->av_frame);
                                if (ret != 0 && ret != AVERROR_EOF) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                                        ret);
                                        break;
                                }

                        } while (ret != AVERROR_EOF);
                } else {
                        int ret;
                        ret = avcodec_send_frame(s->codec_ctx, NULL);
                        if (ret != 0) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                                ret);
                        }
                        do {
                                AVPacket pkt;
                                av_init_packet(&pkt);
                                ret = avcodec_receive_packet(s->codec_ctx, &pkt);
                                av_packet_unref(&pkt);
                                if (ret != 0 && ret != AVERROR_EOF) {
                                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unexpected return value %d\n",
                                                        ret);
                                        break;
                                }
                        } while (ret != AVERROR_EOF);
                }
#endif
        }

        s->context_initialized = false;
}

static void libavcodec_done(void *state)
{
        struct libavcodec_codec_state *s = (struct libavcodec_codec_state *) state;
        assert(s->magic == MAGIC);

        cleanup_common(s);

        pthread_mutex_lock(s->libav_global_lock);
        avcodec_close(s->codec_ctx);
        avcodec_free_context(&s->codec_ctx);
        pthread_mutex_unlock(s->libav_global_lock);

        rm_release_shared_lock(LAVCD_LOCK_NAME);
        free(s->output_channel_data);
        free(s->tmp_data);
        av_freep(&s->samples);
        av_frame_free(&s->av_frame);

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
