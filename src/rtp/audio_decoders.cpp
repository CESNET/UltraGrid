/**
 * @file   rtp/audio_decoders.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2012-2023 CESNET, z. s. p. o.
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

#include "control_socket.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "module.h"
#include "tv.h"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/ptime.h"
#include "rtp/pbuf.h"
#include "rtp/audio_decoders.h"
#include "audio/audio_playback.h"
#include "audio/codec.h"
#include "audio/resampler.hpp"
#include "audio/types.h"
#include "audio/utils.h"
#include "crypto/crc.h"
#include "crypto/openssl_decrypt.h"
#include "ug_runtime_error.hpp"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/packet_counter.h"
#include "utils/worker.h"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstring>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using std::chrono::duration_cast;
using std::chrono::seconds;
using std::chrono::steady_clock;
using std::fixed;
using std::hex;
using std::map;
using std::ostringstream;
using std::pair;
using std::setprecision;
using std::string;
using std::to_string;
using std::vector;

#define AUDIO_DECODER_MAGIC 0x12ab332bu
#define MOD_NAME "[audio dec.] "

struct scale_data {
        double vol_avg = 1.0;
        int samples = 0;

        double scale = 1.0;
};

struct channel_map {
        ~channel_map() {
                free(sizes);
                for(int i = 0; i < size; ++i) {
                        free(map[i]);
                }
                free(map);
                free(contributors);
        }
        int **map = nullptr; // index is source channel, content is output channels
        int *sizes = nullptr;
        int *contributors = nullptr; // count of contributing channels to output
        int size = 0;
        int max_output = -1;

        bool validate() {
                for(int i = 0; i < size; ++i) {
                        for(int j = 0; j < sizes[i]; ++j) {
                                if(map[i][j] < 0) {
                                        log_msg(LOG_LEVEL_ERROR, "Audio channel mapping - negative parameter occured.\n");
                                        return false;
                                }
                        }
                }

                return true;
        }

        void compute_contributors() {
                for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < sizes[i]; ++j) {
                                max_output = std::max(map[i][j], max_output);
                        }
                }
                contributors = (int *) calloc(max_output + 1, sizeof(int));
                for (int i = 0; i < size; ++i) {
                        for (int j = 0; j < sizes[i]; ++j) {
                                contributors[map[i][j]] += 1;
                        }
                }
        }
};

struct state_audio_decoder_summary {
private:
        unsigned long int last_bufnum = -1;
        int64_t played = 0;
        int64_t missed = 0;

        steady_clock::time_point t_last = steady_clock::now();

        void print() const {
                LOG(LOG_LEVEL_INFO)
                    << SUNDERLINE("Audio dec stats")
                    << " (cumulative): " << SBOLD(played) << " played / "
                    << SBOLD(played + missed) << " total audio frames\n";
        }

public:
        ~state_audio_decoder_summary() {
                print();
        }

        void update(unsigned long int bufnum) {
                if (last_bufnum != static_cast<unsigned long int>(-1)) {
                        if ((last_bufnum + 1) % (1U<<BUFNUM_BITS) == bufnum) {
                                played += 1;
                        } else {
                                unsigned long int diff = (bufnum - last_bufnum + 1 + (1U<<BUFNUM_BITS)) % (1U<<BUFNUM_BITS);
                                if (diff >= (1U<<BUFNUM_BITS) / 2) {
                                        diff -= (1U<<BUFNUM_BITS) / 2;
                                }
                                missed += diff;
                        }
                }
                last_bufnum = bufnum;

                auto now = steady_clock::now();
                if (duration_cast<seconds>(steady_clock::now() - t_last).count() > CUMULATIVE_REPORTS_INTERVAL) {
                        print();
                        t_last = now;
                }
        }
};

struct state_audio_decoder {
        uint32_t magic;
        struct module mod;

        struct timeval t0;

        struct packet_counter *packet_counter;

        unsigned int channel_remapping:1;
        struct channel_map channel_map;

        vector<struct scale_data> scale = vector<scale_data>(1); ///< contains scaling metadata if we want to perform audio scaling
        bool fixed_scale;

        struct audio_codec_state *audio_decompress;

        struct audio_desc saved_desc; // from network
        uint32_t saved_audio_tag;

        audio_frame2 decoded; ///< buffer that keeps audio samples from last 5 seconds (for statistics)

        const struct openssl_decrypt_info *dec_funcs;
        struct openssl_decrypt *decrypt;

        bool muted;

        audio_frame2_resampler resampler;

        audio_playback_ctl_t audio_playback_ctl_func;
        void *audio_playback_state;

        struct control_state *control;
        fec *fec_state;
        fec_desc fec_state_desc;

        struct state_audio_decoder_summary summary;

        audio_frame2 resample_remainder;
        std::atomic_uint64_t req_resample_to{0}; // hi 32 - numerator; lo 32 - denominator
};

constexpr double VOL_UP = 1.1;
constexpr double VOL_DOWN = 1.0/1.1;

static void compute_scale(struct scale_data *scale_data, double vol_avg, int samples, int sample_rate)
{
        scale_data->vol_avg = scale_data->vol_avg * (scale_data->samples / ((double) scale_data->samples + samples)) +
                vol_avg * (samples / ((double) scale_data->samples + samples));
        scale_data->samples += samples;

        if (scale_data->samples > sample_rate * 6) {
                double ratio = 1.0;

                if (scale_data->vol_avg > 0.25 || (scale_data->vol_avg > 0.05 && scale_data->scale > 1.0)) {
                        ratio = VOL_DOWN;
                } else if (scale_data->vol_avg < 0.20 && scale_data->scale < 1.0) {
                        ratio = VOL_UP;
                }

                scale_data->scale *= ratio;
                scale_data->vol_avg *= ratio;

                MSG(VERBOSE,
                    "Audio scale adjusted to: %f (average volume was %f)\n",
                    scale_data->scale, scale_data->vol_avg);

                scale_data->samples = 4 * sample_rate;
        }
}

static void audio_decoder_process_message(struct module *m)
{
        auto *s = static_cast<struct state_audio_decoder *>(m->priv_data);

        while (auto *msg = reinterpret_cast<msg_universal *>(check_message(m))) {
                struct response *r = nullptr;
                if (strstr(msg->text, MSG_UNIVERSAL_TAG_AUDIO_DECODER) == msg->text) {
                        s->req_resample_to = strtoull(msg->text + strlen(MSG_UNIVERSAL_TAG_AUDIO_DECODER), nullptr, 0);
                        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << "Resampling sound to: " << (s->req_resample_to >> ADEC_CH_RATE_SHIFT) << "/" << (s->req_resample_to & ((1LLU << ADEC_CH_RATE_SHIFT) - 1)) << "\n";
                        r = new_response(RESPONSE_OK, nullptr);
                } else {
                        r = new_response(RESPONSE_NOT_FOUND, nullptr);
                }
                free_message(reinterpret_cast<message *>(msg), r);
        }
}

ADD_TO_PARAM("soft-resample", "* soft-resample=<num>/<den>\n"
                "  Resample to specified sampling rate, eg. 12288128/256 for 48000.5 Hz\n");

void *audio_decoder_init(char *audio_channel_map, const char *audio_scale, const char *encryption, audio_playback_ctl_t c, void *p_state, struct module *parent)
{
        struct state_audio_decoder *s = NULL;
        bool scale_auto = false;
        double scale_factor = 1.0;
        char *tmp = nullptr;

        assert(audio_scale != NULL);

        try {
                s = new struct state_audio_decoder();
        } catch (ug_runtime_error &e) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s\n", e.what());
                goto error;
        }

        s->magic = AUDIO_DECODER_MAGIC;
        s->audio_playback_ctl_func = c;
        s->audio_playback_state = p_state;

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_DECODER;
        s->mod.priv_data = s;
        s->mod.new_message = audio_decoder_process_message;
        module_register(&s->mod, parent);

        if (const char *val = get_commandline_param("soft-resample")) {
                assert(strchr(val, '/') != nullptr);
                s->req_resample_to = strtoll(val, NULL, 0) << 32LLU | strtoll(strchr(val, '/') + 1, NULL, 0);
        }

        gettimeofday(&s->t0, NULL);
        s->packet_counter = packet_counter_init(0);

        s->audio_decompress = NULL;

        s->control = (struct control_state *) get_module(get_root_module(parent), "control");

        if (encryption) {
                s->dec_funcs = static_cast<const struct openssl_decrypt_info *>(load_library("openssl_decrypt",
                                        LIBRARY_CLASS_UNDEFINED, OPENSSL_DECRYPT_ABI_VERSION));
                if (!s->dec_funcs) {
                        log_msg(LOG_LEVEL_ERROR, "This " PACKAGE_NAME " version was build "
                                        "without OpenSSL support!\n");
                        delete s;
                        return NULL;
                }
                if (s->dec_funcs->init(&s->decrypt, encryption) != 0) {
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
                }

                if (!s->channel_map.validate()) {
                        log_msg(LOG_LEVEL_ERROR, "Wrong audio mapping.\n");
                        goto error;
                }
                s->channel_remapping = TRUE;
                s->channel_map.compute_contributors();

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
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Channel remapping detected - automatically scaling audio levels. Use \"--audio-scale none\" to disable.\n");
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
        s->scale.at(0).scale = scale_factor;

        return s;

error:
        free(tmp);
        if (s) {
                audio_decoder_destroy(s);
        }
        return NULL;
}

void audio_decoder_destroy(void *state)
{
        struct state_audio_decoder *s = (struct state_audio_decoder *) state;

        assert(s != NULL);
        assert(s->magic == AUDIO_DECODER_MAGIC);

        packet_counter_destroy(s->packet_counter);
        audio_codec_done(s->audio_decompress);

        if (s->dec_funcs) {
                s->dec_funcs->destroy(s->decrypt);
        }

        delete s->fec_state;
        module_done(&s->mod);
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
        long bytes_received;
        long bytes_expected;
        bool muted_receiver;
};

static void *adec_compute_and_print_stats(void *arg) {
        auto d = (struct adec_stats_processing_data*) arg;
        string loss;
        if (d->bytes_received < d->bytes_expected) {
                loss = " (" + to_string(d->bytes_expected - d->bytes_received) + " lost)";
        }
        log_msg(LOG_LEVEL_INFO, "[Audio decoder] Received %ld/%ld B%s, "
                        "decoded %d samples in %.2f sec.\n",
                        d->bytes_received,
                        d->bytes_expected,
                        loss.c_str(),
                        d->frame.get_sample_count(),
                        d->seconds);

        using namespace std::string_literals;
        ostringstream volume;
        volume << fixed << setprecision(2);
        for (int i = 0; i < d->frame.get_channel_count(); ++i) {
                double rms = 0.0;
                double peak = 0.0;
                rms = calculate_rms(&d->frame, i, &peak);
                volume << (i > 0 ? ", ["s : "["s) << i << "] " << SBOLD(SMAGENTA(20.0 * log(rms) / log(10.0) << "/" << 20.0 * log(peak) / log(10.0)));
        }

        LOG(LOG_LEVEL_INFO) << "[Audio decoder] Volume: " << volume.str() << " dBFS RMS/peak" << (d->muted_receiver ? TBOLD(TRED(" (muted)")) : "") << "\n" ;

        delete d;

        return NULL;
}


ADD_TO_PARAM("audio-dec-format", "* audio-dec-format=<fmt>|help\n"
                "  Forces specified format playback format.\n");
/**
 * Compares provided parameters with previous configuration and if it differs, reconfigure
 * the decoder, otherwise the reconfiguration is skipped.
 */
static bool audio_decoder_reconfigure(struct state_audio_decoder *decoder, struct pbuf_audio_data *s, audio_frame2 &received_frame, int input_channels, int bps, int sample_rate, uint32_t audio_tag)
{
        if(decoder->saved_desc.ch_count == input_channels &&
                        decoder->saved_desc.bps == bps &&
                        decoder->saved_desc.sample_rate == sample_rate &&
                        decoder->saved_audio_tag == audio_tag) {
                return true;
        }

        log_msg(LOG_LEVEL_NOTICE, "New incoming audio format detected: %d Hz, %d channel%s, %d bits per sample, codec %s\n",
                        sample_rate, input_channels, input_channels == 1 ? "": "s",  bps * 8,
                        get_name_to_audio_codec(get_audio_codec_to_tag(audio_tag)));

        int output_channels = decoder->channel_remapping ?
                        decoder->channel_map.max_output + 1: input_channels;
        audio_desc device_desc = audio_desc{bps, sample_rate, output_channels, AC_PCM};
        if (const char *fmt = get_commandline_param("audio-dec-format")) {
                if (int ret = parse_audio_format(fmt, &device_desc)) {
                        exit_uv(ret > 0 ? 0 : 1);
                        return false;
                }
        }
        size_t len = sizeof device_desc;
        if (!decoder->audio_playback_ctl_func(decoder->audio_playback_state, AUDIO_PLAYBACK_CTL_QUERY_FORMAT, &device_desc, &len)) {
                log_msg(LOG_LEVEL_ERROR, "Unable to query audio desc!\n");
                return false;
        }

        s->buffer.bps = device_desc.bps;
        s->buffer.ch_count = device_desc.ch_count;
        s->buffer.sample_rate = device_desc.sample_rate;

        if(!decoder->fixed_scale) {
                decoder->scale.clear();
                int scale_count = decoder->channel_remapping ? decoder->channel_map.max_output + 1: decoder->saved_desc.ch_count;
                decoder->scale.resize(scale_count);
        }
        decoder->saved_desc.ch_count = input_channels;
        decoder->saved_desc.bps = bps;
        decoder->saved_desc.sample_rate = sample_rate;
        decoder->saved_audio_tag = audio_tag;
        audio_codec_t audio_codec = get_audio_codec_to_tag(audio_tag);

        received_frame.init(input_channels, audio_codec, bps, sample_rate);
        decoder->decoded.init(input_channels, AC_PCM,
                        device_desc.bps, device_desc.sample_rate);
        decoder->decoded.reserve(device_desc.bps * device_desc.sample_rate * 6);

        decoder->audio_decompress = audio_codec_reconfigure(decoder->audio_decompress, audio_codec, AUDIO_DECODER);
        if(!decoder->audio_decompress) {
                log_msg(LOG_LEVEL_FATAL, "Unable to create audio decompress!\n");
                exit_uv(1);
                return false;
        }

        decoder->resample_remainder = {};
        return true;
}

static bool audio_fec_decode(struct pbuf_audio_data *s, vector<pair<vector<char>, map<int, int>>> &fec_data, uint32_t fec_params, audio_frame2 &received_frame)
{
        struct state_audio_decoder *decoder = s->decoder;
        fec_desc fec_desc { FEC_RS, fec_params >> 19U, (fec_params >> 6U) & 0x1FFFU, fec_params & 0x3F };

        if (decoder->fec_state == NULL || decoder->fec_state_desc.k != fec_desc.k || decoder->fec_state_desc.m != fec_desc.m || decoder->fec_state_desc.c != fec_desc.c) {
                delete decoder->fec_state;
                decoder->fec_state = fec::create_from_desc(fec_desc);
                if (decoder->fec_state == nullptr) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot initialize FEC decoder!\n";
                        return false;
                }
                decoder->fec_state_desc = fec_desc;
        }

        audio_desc desc{};

        int channel = 0;
        for (auto & c : fec_data) {
                char *out = nullptr;
                int out_len = 0;
                if (decoder->fec_state->decode(c.first.data(), c.first.size(), &out, &out_len, c.second)) {
                        if (!desc) {
                                uint32_t quant_sample_rate = 0;
                                uint32_t audio_tag = 0;

                                memcpy(&quant_sample_rate, out + 3 * sizeof(uint32_t), sizeof(uint32_t));
                                memcpy(&audio_tag, out + 4 * sizeof(uint32_t), sizeof(uint32_t));
                                quant_sample_rate = ntohl(quant_sample_rate);
                                audio_tag = ntohl(audio_tag);

                                desc.bps = (quant_sample_rate >> 26) / 8;
                                desc.sample_rate = quant_sample_rate & 0x07FFFFFFU;
                                desc.ch_count = fec_data.size();
                                desc.codec = get_audio_codec_to_tag(audio_tag);
                                if (!desc.codec) {
                                        auto flags = std::clog.flags();
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Wrong AudioTag 0x" << hex << audio_tag << "\n";
                                        std::clog.flags(flags);
                                }

                                if (!audio_decoder_reconfigure(decoder, s, received_frame, desc.ch_count, desc.bps, desc.sample_rate, audio_tag)) {
                                        return FALSE;
                                }
                        }
                        received_frame.replace(channel, 0, out + sizeof(audio_payload_hdr_t), out_len - sizeof(audio_payload_hdr_t));
                }
                channel += 1;
        }

        return true;
}

int decode_audio_frame(struct coded_data *cdata, void *pbuf_data, struct pbuf_stats *)
{
        struct pbuf_audio_data *s = (struct pbuf_audio_data *) pbuf_data;
        struct state_audio_decoder *decoder = s->decoder;

        int input_channels = 0;

        bool first = true;
        int bufnum = 0;

        if(!cdata) {
                return FALSE;
        }

        if (!cdata->data->m) {
                // skip frame without m-bit, we cannot determine number of channels
                // (it is maximal substream number + 1 in packet with m-bit)
                return FALSE;
        }

        DEBUG_TIMER_START(audio_decode);
        audio_frame2 received_frame;
        received_frame.init(decoder->saved_desc.ch_count,
                        get_audio_codec_to_tag(decoder->saved_audio_tag),
                        decoder->saved_desc.bps,
                        decoder->saved_desc.sample_rate);
        received_frame.set_timestamp(cdata->data->ts);
        vector<pair<vector<char>, map<int, int>>> fec_data;
        uint32_t fec_params = 0;

        while (cdata != NULL) {
                char *data;
                // for definition see rtp_callbacks.h
                uint32_t *audio_hdr = (uint32_t *)(void *) cdata->data->data;
                const int pt = cdata->data->pt;
                enum openssl_mode crypto_mode;

                if (PT_AUDIO_IS_ENCRYPTED(pt)) {
                        if(!decoder->decrypt) {
                                log_msg(LOG_LEVEL_WARNING, "Receiving encrypted audio data but "
                                                "no decryption key entered!\n");
                                return FALSE;
                        }
                } else if (PT_IS_AUDIO(pt) && !PT_AUDIO_IS_ENCRYPTED(pt)) {
                        if(decoder->decrypt) {
                                log_msg(LOG_LEVEL_WARNING, "Receiving unencrypted audio data "
                                                "while expecting encrypted.\n");
                                return FALSE;
                        }
                } else {
                        if (pt == PT_Unassign_Type95) {
                                log_msg_once(LOG_LEVEL_WARNING, to_fourcc('U', 'V', 'P', 'T'), MOD_NAME "Unassigned PT 95 received, ignoring.\n");
                        } else {
                                log_msg(LOG_LEVEL_WARNING, "Unknown audio packet type: %d\n", pt);
                        }
                        return FALSE;
                }

                unsigned int length;
                char plaintext[RTP_MAX_PACKET_LEN]; // plaintext will be actually shorter
                size_t main_hdr_len = PT_AUDIO_HAS_FEC(pt) ? sizeof(fec_payload_hdr_t) : sizeof(audio_payload_hdr_t);
                if (PT_AUDIO_IS_ENCRYPTED(pt)) {
                        uint32_t encryption_hdr = ntohl(*(uint32_t *)(void *) (cdata->data->data + main_hdr_len));
                        crypto_mode = (enum openssl_mode) (encryption_hdr >> 24);
                        if (crypto_mode == MODE_AES128_NONE || crypto_mode > MODE_AES128_MAX) {
                                log_msg(LOG_LEVEL_WARNING, "Unknown cipher mode: %d\n", (int) crypto_mode);
                                return FALSE;
                        }
                        char *ciphertext = cdata->data->data + sizeof(crypto_payload_hdr_t) +
                                main_hdr_len;
                        int ciphertext_len = cdata->data->data_len - main_hdr_len -
                                sizeof(crypto_payload_hdr_t);

                        if((length = decoder->dec_funcs->decrypt(decoder->decrypt,
                                        ciphertext, ciphertext_len,
                                        (char *) audio_hdr, sizeof(audio_payload_hdr_t),
                                        plaintext, crypto_mode)) == 0) {
                                return FALSE;
                        }
                        data = plaintext;
                } else {
                        assert(PT_IS_AUDIO(pt));
                        length = cdata->data->data_len - main_hdr_len;
                        data = cdata->data->data + main_hdr_len;
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

                int channel = (ntohl(audio_hdr[0]) >> 22) & 0x3ff;
                bufnum = ntohl(audio_hdr[0]) & ((1U<<BUFNUM_BITS) - 1U);
                int sample_rate = ntohl(audio_hdr[3]) & 0x3fffff;
                unsigned int offset = ntohl(audio_hdr[1]);
                unsigned int buffer_len = ntohl(audio_hdr[2]);
                //fprintf(stderr, "%d-%d-%d ", length, bufnum, channel);

                if (packet_counter_get_channels(decoder->packet_counter) != input_channels) {
                        packet_counter_destroy(decoder->packet_counter);
                        decoder->packet_counter = packet_counter_init(input_channels);
                }

                if (PT_AUDIO_HAS_FEC(pt)) {
                        fec_data.resize(input_channels);
                        fec_data[channel].first.resize(buffer_len);
                        fec_params = ntohl(audio_hdr[3]);
                        fec_data[channel].second[offset] = length;
                        memcpy(fec_data[channel].first.data() + offset, data, length);
                } else {
                        int bps = (ntohl(audio_hdr[3]) >> 26) / 8;
                        uint32_t audio_tag = ntohl(audio_hdr[4]);

                        if (!audio_decoder_reconfigure(decoder, s, received_frame, input_channels, bps, sample_rate, audio_tag)) {
                                return FALSE;
                        }

                        received_frame.replace(channel, offset, data, length);

                        /* buffer size same for every packet of the frame */
                        /// @todo do we really want to scale to expected buffer length even if some frames are missing
                        /// at the end of the buffer
                        received_frame.resize(channel, buffer_len);
                }

                packet_counter_register_packet(decoder->packet_counter, channel, bufnum, offset, length);

                if (first) {
                        memcpy(&s->source, ((char *) cdata->data) + RTP_MAX_PACKET_LEN, sizeof(struct sockaddr_storage));
                        first = false;
                }

                cdata = cdata->nxt;
        }

        decoder->summary.update(bufnum);

        if (fec_params != 0) {
                if (!audio_fec_decode(s, fec_data, fec_params, received_frame)) {
                        return FALSE;
                }
        }

        s->frame_size = received_frame.get_data_len();
        audio_frame2 decompressed = audio_codec_decompress(decoder->audio_decompress, &received_frame);
        if (!decompressed) {
                return FALSE;
        }

        // Perform a variable rate resample if any output device has requested it
        if (decoder->req_resample_to != 0 || s->buffer.sample_rate != decompressed.get_sample_rate()) {
                int resampler_bps = decoder->resampler.align_bps(decompressed.get_bps());
                if (resampler_bps <= 0) {
                        return FALSE;
                }
                if (resampler_bps != decompressed.get_bps()) {
                        decompressed.change_bps(resampler_bps);
                }
                if (decoder->req_resample_to != 0) {
                        auto [ret, remainder] = decompressed.resample_fake(decoder->resampler, decoder->req_resample_to >> ADEC_CH_RATE_SHIFT, decoder->req_resample_to & ((1LLU << ADEC_CH_RATE_SHIFT) - 1));
                        if (!ret) {
                                LOG(LOG_LEVEL_INFO) << MOD_NAME << "You may try to set different sampling on sender.\n";
                                return FALSE;
                        }
                        decoder->resample_remainder = std::move(remainder);
                } else {
                        if (!decompressed.resample(decoder->resampler, s->buffer.sample_rate)) {
                                LOG(LOG_LEVEL_INFO) << MOD_NAME << "You may try to set different sampling on sender.\n";
                                return FALSE;
                        }
                }
        }

        if (decompressed.get_bps() != s->buffer.bps) {
                decompressed.change_bps(s->buffer.bps);
        }

        size_t new_data_len = s->buffer.data_len + decompressed.get_data_len(0) * s->buffer.ch_count;
        if ((size_t) s->buffer.max_size < new_data_len) {
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
                                                                decoder->scale.at(decoder->fixed_scale ? 0 : new_position).scale);
                                        }
                                }
                        } else {
                                if (channel >= s->buffer.ch_count)
                                        continue;
                                mux_and_mix_channel(s->buffer.data + s->buffer.data_len, decompressed.get_data(channel),
                                                decompressed.get_bps(),
                                                decompressed.get_data_len(channel), s->buffer.ch_count, channel,
                                                decoder->scale.at(decoder->fixed_scale ? 0 : input_channels).scale);
                        }
                }
        }
        s->buffer.data_len = new_data_len;
        if (s->buffer.timestamp == -1) {
                s->buffer.timestamp = decompressed.get_timestamp();
        }

        decoder->decoded.append(decompressed);

        if (control_stats_enabled(decoder->control)) {
                std::string report = "ARECV";
                int num_ch = std::min(decompressed.get_channel_count(), control_audio_ch_report_count(decoder->control));
                for(int i = 0; i < num_ch; i++){
                        double rms, peak;
                        rms = calculate_rms(&decompressed, i, &peak);
                        double rms_dbfs0 = 20 * log(rms) / log(10);
                        double peak_dbfs0 = 20 * log(peak) / log(10);
                        report += " volrms" + std::to_string(i) + " " + std::to_string(rms_dbfs0);
                        report += " volpeak" + std::to_string(i) + " " + std::to_string(peak_dbfs0);
                }

                control_report_stats(decoder->control, report);
        }

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
                d->muted_receiver = decoder->muted;

                task_run_async_detached(adec_compute_and_print_stats, d);

                decoder->t0 = t;
                packet_counter_clear(decoder->packet_counter);
        }

        DEBUG_TIMER_START(audio_decode_compute_autoscale);
        if(!decoder->fixed_scale) {
                int output_channels = decoder->channel_remapping ?
                        decoder->channel_map.max_output + 1: input_channels;
                for(int i = 0; i <= decoder->channel_map.max_output; ++i) {
                        if (decoder->channel_map.contributors[i] <= 1) {
                                continue;
                        }
                        double avg = get_avg_volume(s->buffer.data, s->buffer.bps,
                                        s->buffer.data_len / output_channels / s->buffer.bps, output_channels, i);
                        compute_scale(&decoder->scale.at(i), avg,
                                        s->buffer.data_len / output_channels / s->buffer.bps, s->buffer.sample_rate);
                }
        }
        DEBUG_TIMER_STOP(audio_decode_compute_autoscale);
        DEBUG_TIMER_STOP(audio_decode);
        
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

void audio_decoder_set_volume(void *state, double val)
{
    auto s = (struct state_audio_decoder *) state;
    for (auto & i : s->scale) {
            i.scale = val;
    }
    s->muted = val == 0.0;
}

