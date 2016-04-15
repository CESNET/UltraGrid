/**
 * @file   rtp/audio_decoders.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2012-2014 CESNET z.s.p.o.
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
#include "lib_common.h"
#include "perf.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/ptime.h"
#include "rtp/pbuf.h"
#include "rtp/audio_decoders.h"
#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"
#include "crypto/crc.h"
#include "crypto/openssl_decrypt.h"

#include "utils/packet_counter.h"
#include "utils/worker.h"

#include <algorithm>
#include <ctype.h>
#include <time.h>
#include <string.h>

#define AUDIO_DECODER_MAGIC 0x12ab332bu

struct scale_data {
        double vol_avg;
        int samples;

        double scale;
};

struct channel_map {
        int **map; // index is source channel, content is output channels
        int *sizes;
        int size;
        int max_output;
};

struct state_audio_decoder {
        uint32_t magic;

        struct timeval t0;

        struct packet_counter *packet_counter;

        unsigned int channel_remapping:1;
        struct channel_map channel_map;

        struct scale_data *scale; ///< contains scaling metadata if we want to perform audio scaling
        bool fixed_scale;

        struct audio_codec_state *audio_decompress;

        struct audio_desc saved_desc; // from network
        uint32_t saved_audio_tag;

        audio_frame2 decoded; ///< buffer that keeps audio samples from last 5 seconds (for statistics)

        const struct openssl_decrypt_info *dec_funcs;
        struct openssl_decrypt *decrypt;

        bool muted;

        audio_frame2_resampler resampler;

        query_supported_format_t query_func;
        void *query_func_data;
};

static int validate_mapping(struct channel_map *map);
static void compute_scale(struct scale_data *scale_data, float vol_avg, int samples, int sample_rate);

static int validate_mapping(struct channel_map *map)
{
        int ret = TRUE;

        for(int i = 0; i < map->size; ++i) {
                for(int j = 0; j < map->sizes[i]; ++j) {
                        if(map->map[i][j] < 0) {
                                log_msg(LOG_LEVEL_ERROR, "Audio channel mapping - negative parameter occured.\n");
                                ret = FALSE;
                                goto return_value;
                        }
                }
        }

return_value:
        return ret;
}

static void compute_scale(struct scale_data *scale_data, float vol_avg, int samples, int sample_rate)
{
        scale_data->vol_avg = scale_data->vol_avg * (scale_data->samples / ((double) scale_data->samples + samples)) +
                vol_avg * (samples / ((double) scale_data->samples + samples));
        scale_data->samples += samples;

        if(scale_data->samples > sample_rate * 6) { // 10 sec
                double ratio = 0.0;

                if(scale_data->vol_avg < 0.01 && scale_data->vol_avg > 0.0001) {
                        ratio = 1.1;
                } else if(scale_data->vol_avg > 0.25) {
                        ratio = 1/1.1;
                } else if(scale_data->vol_avg > 0.05 && scale_data->scale > 1.0) {
                        ratio = 1/1.1;
                } else if(scale_data->vol_avg < 0.20 && scale_data->scale < 1.0) {
                        ratio = 1.1;
                }

                if(ratio != 0.0) {
                        scale_data->scale *= ratio;
                        scale_data->vol_avg *= ratio;
                }

                debug_msg("Audio scale adjusted to: %f (average volume was %f)\n", scale_data->scale, scale_data->vol_avg);

                scale_data->samples = 4 * sample_rate;
        }
}

void *audio_decoder_init(char *audio_channel_map, const char *audio_scale, const char *encryption, query_supported_format_t q, void *q_state)
{
        struct state_audio_decoder *s;
        bool scale_auto = false;
        double scale_factor = 1.0;
        char *tmp = nullptr;

        assert(audio_scale != NULL);

        s = new struct state_audio_decoder();
        s->magic = AUDIO_DECODER_MAGIC;
        s->query_func = q;
        s->query_func_data = q_state;

        gettimeofday(&s->t0, NULL);
        s->packet_counter = NULL;

        s->audio_decompress = NULL;

        if (encryption) {
                s->dec_funcs = static_cast<const struct openssl_decrypt_info *>(load_library("openssl_decrypt",
                                        LIBRARY_CLASS_UNDEFINED, OPENSSL_DECRYPT_ABI_VERSION));
                if (!s->dec_funcs) {
                        log_msg(LOG_LEVEL_ERROR, "This " PACKAGE_NAME " version was build "
                                        "without OpenSSL support!\n");
                        delete s;
                        return NULL;
                }
                if (s->dec_funcs->init(&s->decrypt,
                                                encryption, MODE_AES128_CTR) != 0) {
                        log_msg(LOG_LEVEL_ERROR, "Unable to create decompress!\n");
                        delete s;
                        return NULL;
                }
        }

        if(audio_channel_map) {
                char *save_ptr = NULL;
                char *item;
                char *ptr;
                tmp = ptr = strdup(audio_channel_map);

                s->channel_map.size = 0;
                while((item = strtok_r(ptr, ",", &save_ptr))) {
                        ptr = NULL;
                        // item is in format x1:y1
                        if(isdigit(item[0])) {
                                s->channel_map.size = std::max(s->channel_map.size, atoi(item) + 1);
                        }
                }
                
                s->channel_map.map = (int **) malloc(s->channel_map.size * sizeof(int *));
                s->channel_map.sizes = (int *) malloc(s->channel_map.size * sizeof(int));
                s->channel_map.max_output = -1;

                /* default value, do not process */
                for(int i = 0; i < s->channel_map.size; ++i) {
                        s->channel_map.map[i] = NULL;
                        s->channel_map.sizes[i] = 0;
                }

                free (tmp);
                tmp = ptr = strdup(audio_channel_map);

                while((item = strtok_r(ptr, ",", &save_ptr))) {
                        ptr = NULL;

                        assert(strchr(item, ':') != NULL);
                        int src;
                        if(isdigit(item[0])) {
                                src = atoi(item);
                        } else {
                                src = -1;
                        }
                        if(!isdigit(strchr(item, ':')[1])) {
                                log_msg(LOG_LEVEL_ERROR, "Audio destination channel not entered!\n");
                                goto error;
                        }
                        int dst = atoi(strchr(item, ':') + 1);
                        if(src >= 0) {
                                s->channel_map.sizes[src] += 1;
                                if(s->channel_map.map[src] == NULL) {
                                        s->channel_map.map[src] = (int *) malloc(1 * sizeof(int));
                                } else {
                                        s->channel_map.map[src] = (int *) realloc(s->channel_map.map[src], s->channel_map.sizes[src] * sizeof(int));
                                }
                                s->channel_map.map[src][s->channel_map.sizes[src] - 1] = dst;
                        }
                        s->channel_map.max_output = std::max(dst, s->channel_map.max_output);
                }


                if(!validate_mapping(&s->channel_map)) {
                        log_msg(LOG_LEVEL_ERROR, "Wrong audio mapping.\n");
                        goto error;
                } else {
                        s->channel_remapping = TRUE;
                }

                free (tmp);
                tmp = NULL;
        } else {
                s->channel_remapping = FALSE;
                s->channel_map.map = NULL;
                s->channel_map.sizes = NULL;
                s->channel_map.size = 0;
        } 

        if(strcasecmp(audio_scale, "mixauto") == 0) {
                if(s->channel_remapping) {
                        scale_auto = true;
                } else {
                        scale_auto = false;
                }
        } else if(strcasecmp(audio_scale, "auto") == 0) {
                scale_auto = true;
        } else if(strcasecmp(audio_scale, "none") == 0) {
                scale_auto = false;
                scale_factor = 1.0;
        } else {
                scale_auto = false;
                scale_factor = atof(audio_scale);
                if(scale_factor <= 0.0) {
                        log_msg(LOG_LEVEL_ERROR, "Invalid audio scaling factor!\n");
                        goto error;
                }
        }

        s->fixed_scale = scale_auto ? false : true;

        if(s->fixed_scale) {
                s->scale = (struct scale_data *) malloc(sizeof(struct scale_data));
                s->scale->samples = 0;
                s->scale->vol_avg = 1.0;
                s->scale->scale = scale_factor;
        } else {
                s->scale = NULL; // will allocated by decoder reconfiguration
        }

        return s;

error:
        free(tmp);
        audio_decoder_destroy(s);
        delete s;
        return NULL;
}

void audio_decoder_destroy(void *state)
{
        struct state_audio_decoder *s = (struct state_audio_decoder *) state;

        assert(s != NULL);
        assert(s->magic == AUDIO_DECODER_MAGIC);

        free(s->scale);
        free(s->channel_map.map);
        free(s->channel_map.sizes);
        packet_counter_destroy(s->packet_counter);
        audio_codec_done(s->audio_decompress);

        if (s->dec_funcs) {
                s->dec_funcs->destroy(s->decrypt);
        }

        delete s;
}

bool parse_audio_hdr(uint32_t *hdr, struct audio_desc *desc)
{
        desc->ch_count = ((ntohl(hdr[0]) >> 22) & 0x3ff) + 1;
        desc->sample_rate = ntohl(hdr[3]) & 0x3fffff;
        desc->bps = (ntohl(hdr[3]) >> 26) / 8;

        uint32_t audio_tag = ntohl(hdr[4]);
        desc->codec = get_audio_codec_to_tag(audio_tag);

        return true;
}

struct adec_stats_processing_data {
        audio_frame2 frame;
        double seconds;
        int bytes_received;
        int bytes_expected;
};

static void *adec_compute_and_print_stats(void *arg) {
        auto d = (struct adec_stats_processing_data*) arg;
        log_msg(LOG_LEVEL_INFO, "[Audio decoder] Received %u/%d B, "
                        "decoded %d samples in %.2f sec.\n",
                        d->bytes_received,
                        d->bytes_expected,
                        d->frame.get_sample_count(),
                        d->seconds);
        for (int i = 0; i < d->frame.get_channel_count(); ++i) {
                double rms, peak;
                rms = calculate_rms(&d->frame, i, &peak);
                log_msg(LOG_LEVEL_INFO, "[Audio decoder] Channel %d - volume: %.2f dBFS RMS, %.2f dBFS peak.\n",
                                i, 20 * log(rms) / log(10), 20 * log(peak) / log(10));
        }

        delete d;

        return NULL;
}


int decode_audio_frame(struct coded_data *cdata, void *data, struct pbuf_stats *)
{
        struct pbuf_audio_data *s = (struct pbuf_audio_data *) data;
        struct state_audio_decoder *decoder = s->decoder;

        int input_channels = 0;
        int output_channels = 0;
        int bps, sample_rate, channel;

        if(!cdata) {
                return FALSE;
        }

        if (!cdata->data->m) {
                // skip frame without m-bit, we cannot determine number of channels
                // (it is maximal substream number + 1 in packet with m-bit)
                return FALSE;
        }

        audio_frame2 received_frame;
        received_frame.init(decoder->saved_desc.ch_count,
                        get_audio_codec_to_tag(decoder->saved_audio_tag),
                        decoder->saved_desc.bps,
                        decoder->saved_desc.sample_rate);

        while (cdata != NULL) {
                char *data;
                // for definition see rtp_callbacks.h
                uint32_t *audio_hdr = (uint32_t *)(void *) cdata->data->data;
                const int pt = cdata->data->pt;

                if(pt == PT_ENCRYPT_AUDIO) {
                        if(!decoder->decrypt) {
                                log_msg(LOG_LEVEL_WARNING, "Receiving encrypted audio data but "
                                                "no decryption key entered!\n");
                                return FALSE;
                        }
                } else if(pt == PT_AUDIO) {
                        if(decoder->decrypt) {
                                log_msg(LOG_LEVEL_WARNING, "Receiving unencrypted audio data "
                                                "while expecting encrypted.\n");
                                return FALSE;
                        }
                } else {
                        log_msg(LOG_LEVEL_WARNING, "Unknown audio packet type: %d\n", pt);
                        return FALSE;
                }

                unsigned int length;
                char plaintext[cdata->data->data_len]; // plaintext will be actually shorter
                if(pt == PT_AUDIO) {
                        length = cdata->data->data_len - sizeof(audio_payload_hdr_t);
                        data = cdata->data->data + sizeof(audio_payload_hdr_t);
                } else {
                        assert(pt == PT_ENCRYPT_AUDIO);
                        char *ciphertext = cdata->data->data + sizeof(crypto_payload_hdr_t) +
                                sizeof(audio_payload_hdr_t);
                        int ciphertext_len = cdata->data->data_len - sizeof(audio_payload_hdr_t) -
                                sizeof(crypto_payload_hdr_t);

                        if((length = decoder->dec_funcs->decrypt(decoder->decrypt,
                                        ciphertext, ciphertext_len,
                                        (char *) audio_hdr, sizeof(audio_payload_hdr_t),
                                        plaintext
                                        )) == 0) {
                                log_msg(LOG_LEVEL_VERBOSE, "Warning: Packet dropped AES - wrong CRC!\n");
                                return FALSE;
                        }
                        data = plaintext;
                }

                /* we receive last channel first (with m bit, last packet) */
                /* thus can be set only with m-bit packet */
                if(cdata->data->m) {
                        input_channels = ((ntohl(audio_hdr[0]) >> 22) & 0x3ff) + 1;
                }

                // we have:
                // 1) last packet, then we have just set total channels
                // 2) not last, but the last one was processed at first
                assert(input_channels > 0);

                channel = (ntohl(audio_hdr[0]) >> 22) & 0x3ff;
                int bufnum = ntohl(audio_hdr[0]) & 0x3fffff;
                sample_rate = ntohl(audio_hdr[3]) & 0x3fffff;
                bps = (ntohl(audio_hdr[3]) >> 26) / 8;
                uint32_t audio_tag = ntohl(audio_hdr[4]);
                
                output_channels = decoder->channel_remapping ?
                        decoder->channel_map.max_output + 1: input_channels;

                /*
                 * Reconfiguration
                 */
                if(decoder->saved_desc.ch_count != input_channels ||
                                decoder->saved_desc.bps != bps ||
                                decoder->saved_desc.sample_rate != sample_rate ||
                                decoder->saved_audio_tag != audio_tag) {
                        log_msg(LOG_LEVEL_NOTICE, "New incoming audio format detected: %d Hz, %d channel%s, %d bits per sample, codec %s\n",
                                        sample_rate, input_channels, input_channels == 1 ? "": "s",  bps * 8,
                                        get_name_to_audio_codec(get_audio_codec_to_tag(audio_tag)));

                        audio_desc device_desc = decoder->query_func(decoder->query_func_data,
                                        audio_desc{bps, sample_rate, output_channels, AC_PCM});

                        if (!device_desc) {
                                log_msg(LOG_LEVEL_ERROR, "Unable to query audio desc!\n");
                                return FALSE;
                        }

                        s->buffer.bps = device_desc.bps;
                        s->buffer.ch_count = device_desc.ch_count;
                        s->buffer.sample_rate = device_desc.sample_rate;

                        if(!decoder->fixed_scale) {
                                free(decoder->scale);
                                decoder->scale = (struct scale_data *) malloc(output_channels * sizeof(struct scale_data));

                                for(int i = 0; i < output_channels; ++i) {
                                        decoder->scale[i].samples = 0;
                                        decoder->scale[i].vol_avg = 1.0;
                                        decoder->scale[i].scale = 1.0;
                                }
                        }
                        decoder->saved_desc.ch_count = input_channels;
                        decoder->saved_desc.bps = bps;
                        decoder->saved_desc.sample_rate = sample_rate;
                        decoder->saved_audio_tag = audio_tag;
                        packet_counter_destroy(decoder->packet_counter);
                        decoder->packet_counter = packet_counter_init(input_channels);
                        audio_codec_t audio_codec = get_audio_codec_to_tag(audio_tag);

                        received_frame.init(input_channels, audio_codec, bps, sample_rate); 
                        decoder->decoded.init(input_channels, AC_PCM,
                                        device_desc.bps, device_desc.sample_rate);
                        decoder->decoded.reserve(device_desc.bps * device_desc.sample_rate * 6);

                        decoder->audio_decompress = audio_codec_reconfigure(decoder->audio_decompress, audio_codec, AUDIO_DECODER);
                        if(!decoder->audio_decompress) {
                                log_msg(LOG_LEVEL_FATAL, "Unable to create audio decompress!\n");
                                exit_uv(1);
                                return FALSE;
                        }
                }

                unsigned int offset = ntohl(audio_hdr[1]);
                unsigned int buffer_len = ntohl(audio_hdr[2]);
                //fprintf(stderr, "%d-%d-%d ", length, bufnum, channel);


                received_frame.replace(channel, offset, data, length);

                packet_counter_register_packet(decoder->packet_counter, channel, bufnum, offset, length);

                /* buffer size same for every packet of the frame */
                /// @todo do we really want to scale to expected buffer length even if some frames are missing
                /// at the end of the buffer
                received_frame.resize(channel, buffer_len);
                
                cdata = cdata->nxt;
        }

        audio_frame2 decompressed = audio_codec_decompress(decoder->audio_decompress, &received_frame);
        if (!decompressed) {
                return FALSE;
        }

        if (s->buffer.sample_rate != decompressed.get_sample_rate()) {
                if (decompressed.get_bps() != 2) {
                        decompressed.change_bps(2);
                }
                decompressed.resample(decoder->resampler, s->buffer.sample_rate);
        }

        if (decompressed.get_bps() != s->buffer.bps) {
                decompressed.change_bps(s->buffer.bps);
        }

        size_t new_data_len = s->buffer.data_len + decompressed.get_data_len(0) * s->buffer.ch_count;
        if(s->buffer.max_size < new_data_len) {
                s->buffer.max_size = new_data_len;
                s->buffer.data = (char *) realloc(s->buffer.data, new_data_len);
        }

        memset(s->buffer.data + s->buffer.data_len, 0, new_data_len - s->buffer.data_len);

        if (!decoder->muted) {
                // there is a mapping for channel
                for(int channel = 0; channel < decompressed.get_channel_count(); ++channel) {
                        if(decoder->channel_remapping) {
                                if(channel < decoder->channel_map.size) {
                                        for(int i = 0; i < decoder->channel_map.sizes[channel]; ++i) {
                                                int new_position = decoder->channel_map.map[channel][i];
                                                if (new_position >= s->buffer.ch_count)
                                                        continue;
                                                mux_and_mix_channel(s->buffer.data + s->buffer.data_len,
                                                                decompressed.get_data(channel),
                                                                decompressed.get_bps(), decompressed.get_data_len(channel),
                                                                s->buffer.ch_count, new_position,
                                                                decoder->scale[decoder->fixed_scale ? 0 : new_position].scale);
                                        }
                                }
                        } else {
                                if (channel >= s->buffer.ch_count)
                                        continue;
                                mux_and_mix_channel(s->buffer.data + s->buffer.data_len, decompressed.get_data(channel),
                                                decompressed.get_bps(),
                                                decompressed.get_data_len(channel), s->buffer.ch_count, channel,
                                                decoder->scale[decoder->fixed_scale ? 0 : input_channels].scale);
                        }
                }
        }
        s->buffer.data_len = new_data_len;

        decoder->decoded.append(decompressed);

        double seconds;
        struct timeval t;

        gettimeofday(&t, 0);
        seconds = tv_diff(t, decoder->t0);
        if(seconds > 5.0) {
                auto d = new adec_stats_processing_data;
                d->frame.init(decoder->decoded.get_channel_count(), decoder->decoded.get_codec(), decoder->decoded.get_bps(), decoder->decoded.get_sample_rate());
                /// @todo
                /// this will certainly result in a syscall, maybe use some pool?
                d->frame.reserve(decoder->decoded.get_bps() * decoder->decoded.get_sample_rate() * 6);

                std::swap(d->frame, decoder->decoded);
                d->seconds = seconds;
                d->bytes_received = packet_counter_get_total_bytes(decoder->packet_counter);
                d->bytes_expected = packet_counter_get_all_bytes(decoder->packet_counter);

                task_run_async_detached(adec_compute_and_print_stats, d);

                decoder->t0 = t;
                packet_counter_clear(decoder->packet_counter);
        }

        if(!decoder->fixed_scale) {
                for(int i = 0; i <= decoder->channel_map.max_output; ++i) {
                        double avg = get_avg_volume(s->buffer.data, bps,
                                        s->buffer.data_len / output_channels, output_channels, i);
                        compute_scale(&decoder->scale[i], avg,
                                        s->buffer.data_len / output_channels / bps, sample_rate);
                }
        }
        
        return TRUE;
}
/*
 * Second version that uses external audio configuration,
 * now it uses a struct state_audio_decoder instead an audio_frame2.
 * It does multi-channel handling.
 */
int decode_audio_frame_mulaw(struct coded_data *cdata, void *data, struct pbuf_stats *)
{
    struct pbuf_audio_data *s = (struct pbuf_audio_data *) data;
    struct state_audio_decoder *audio = s->decoder;

    //struct state_audio_decoder *audio = (struct state_audio_decoder *)data;

    if(!cdata) return false;

    audio_frame2 received_frame;
    received_frame.init(audio->saved_desc.ch_count, audio->saved_desc.codec,
                    audio->saved_desc.bps, audio->saved_desc.sample_rate);

    // Check-if-there-is-only-one-channel optimization.
    if (received_frame.get_channel_count() == 1) {
        //char *to = audio->received_frame->data[0];
        while (cdata != NULL) {
            // Get the data to copy into the received_frame.
            char *from = cdata->data->data;

            received_frame.append(0, from, cdata->data->data_len);

            cdata = cdata->nxt;
        }
    } else { // Multi-channel case.

        /*
         * Unoptimized version of the multi-channel handling.
         * TODO: Optimize it! It's a matrix transpose.
         *       Take a look at http://stackoverflow.com/questions/1777901/array-interleaving-problem
         */

        while (cdata != NULL) {
            // Check that the amount of data on cdata->data->data is congruent with 0 modulus audio->received_frame->ch_count.
            if (cdata->data->data_len % received_frame.get_channel_count() != 0) {
                // printf something?
                return false;
            }

                char *from = cdata->data->data;

                // For each group of samples.
                for (int g = 0 ; g < (cdata->data->data_len / received_frame.get_channel_count()) ; g++) {
                    // Iterate throught each channel.
                    for (int ch = 0 ; ch < received_frame.get_channel_count(); ch ++) {

                        received_frame.append(ch, from, cdata->data->data_len);

                        from += received_frame.get_bps();
                    }
                }

            cdata = cdata->nxt;
        }

    }

    return true;
}

void audio_decoder_increase_volume(void *state)
{
    auto s = (struct state_audio_decoder *) state;
    s->scale->scale *= 1.1;
    log_msg(LOG_LEVEL_INFO, "Volume: %f%%\n", s->scale->scale * 100.0);
}

void audio_decoder_decrease_volume(void *state)
{
    auto s = (struct state_audio_decoder *) state;
    s->scale->scale /= 1.1;
    log_msg(LOG_LEVEL_INFO, "Volume: %f%%\n", s->scale->scale * 100.0);
}

void audio_decoder_mute(void *state)
{
    auto s = (struct state_audio_decoder *) state;
    s->muted = !s->muted;
    if (s->muted) {
            log_msg(LOG_LEVEL_INFO, "Muted.\n");
    } else {
            log_msg(LOG_LEVEL_INFO, "Unmuted.\n");
    }
}

