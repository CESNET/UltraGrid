/**
 * @file   rtp/audio_decoders.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2012-2026 CESNET, zájmové sdružení právnických osob
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
 * consider adding fec/decompress to separate thread
 */

#include "rtp/audio_decoders.h"

#include <cassert>                   // for assert
#include <chrono>                    // for steady_clock, duration_cast, ope...
#include <cstring>                   // for memcpy, memset, strcasecmp...
#include <iostream>                  // for basic_ostream, operator<<, clog
#include <map>                       // for map
#include <sstream>                   // for basic_ostringstream
#include <string>                    // for char_traits, allocator, operator+
#include <utility>                   // for pair, move, swap
#include <vector>                    // for vector

#include "audio/codec.h"             // for get_audio_codec_to_tag, audio_co...
#include "audio/types.h"             // for audio_frame2, audio_desc, AC_PCM
#include "compat/net.h"              // for ntohl, sockaddr_storage
#include "control_socket.h"
#include "crypto/openssl_decrypt.h"  // for openssl_decrypt_info, OPENSSL_DE...
#include "crypto/openssl_encrypt.h"  // for openssl_mode
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "rtp/fec.h"                 // for fec
#include "rtp/pbuf.h"                // for acodec_data, coded_data
#include "rtp/rtp.h"                 // for RTP_MAX_PACKET_LEN
#include "rtp/rtp_types.h"           // for BUFNUM_BITS, audio_payload_hdr_t
#include "types.h"                   // for fec_desc, fec_type
#include "utils/color_out.h"
#include "utils/debug.h"             // for DEBUG_TIMER_*
#include "utils/macros.h"
#include "utils/packet_counter.h"

using std::chrono::duration_cast;
using std::chrono::seconds;
using std::chrono::steady_clock;
using std::hex;
using std::map;
using std::ostringstream;
using std::pair;
using std::string;
using std::vector;

#define AUDIO_DECODER_MAGIC 0x12ab332bu
#define MOD_NAME "[audio dec.] "

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

        struct packet_counter *packet_counter;

        struct audio_codec_state *audio_decompress;

        struct audio_desc saved_desc; // from network
        uint32_t saved_audio_tag;

        const struct openssl_decrypt_info *dec_funcs;
        struct openssl_decrypt *decrypt;

        audio_playback_ctl_t audio_playback_ctl_func;
        void *audio_playback_state;

        struct control_state *control;
        fec *fec_state;
        fec_desc fec_state_desc;

        struct state_audio_decoder_summary summary;

};

void *
audio_decoder_init(const char *encryption, struct module *parent)
{
        auto *s = new struct state_audio_decoder();
        s->magic = AUDIO_DECODER_MAGIC;
        s->audio_decompress = NULL;
        s->packet_counter = packet_counter_init(0);
        s->control = get_control_state(parent);

        if (strlen(encryption) > 0) {
                s->dec_funcs = static_cast<const struct openssl_decrypt_info *>(load_library("openssl_decrypt",
                                        LIBRARY_CLASS_UNDEFINED, OPENSSL_DECRYPT_ABI_VERSION));
                if (!s->dec_funcs) {
                        log_msg(LOG_LEVEL_ERROR, "This UltraGrid version was build "
                                        "without OpenSSL support!\n");
                        goto error;
                }
                if (s->dec_funcs->init(&s->decrypt, encryption) != 0) {
                        log_msg(LOG_LEVEL_ERROR, "Unable to create decompress!\n");
                        goto error;
                }
        }

        return s;

error:
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

/**
 * Compares provided parameters with previous configuration and if it differs, reconfigure
 * the decoder, otherwise the reconfiguration is skipped.
 */
static bool
audio_decoder_reconfigure(struct state_audio_decoder *decoder,
                          audio_frame2 &received_frame, int input_channels,
                          int bps, int sample_rate, uint32_t audio_tag)
{
        if(decoder->saved_desc.ch_count == input_channels &&
                        decoder->saved_desc.bps == bps &&
                        decoder->saved_desc.sample_rate == sample_rate &&
                        decoder->saved_audio_tag == audio_tag) {
                return true;
        }

        log_msg(LOG_LEVEL_NOTICE, "New incoming audio format detected: %d Hz, %d channel%s, %d bits per sample, codec %s\n",
                        sample_rate, input_channels, input_channels == 1 ? "": "s",  bps * 8,
                        get_audio_codec_name(get_audio_codec_to_tag(audio_tag)));

        std::ostringstream oss;
        oss << "new incoming audio fmt: " << sample_rate << "Hz " << input_channels << "ch " << get_audio_codec_name(get_audio_codec_to_tag(audio_tag));
        control_report_stats(decoder->control, oss.str().c_str());

        decoder->saved_desc.ch_count = input_channels;
        decoder->saved_desc.bps = bps;
        decoder->saved_desc.sample_rate = sample_rate;
        decoder->saved_audio_tag = audio_tag;
        audio_codec_t audio_codec = get_audio_codec_to_tag(audio_tag);

        received_frame.init(input_channels, audio_codec, bps, sample_rate);

        decoder->audio_decompress = audio_codec_reconfigure(decoder->audio_decompress, audio_codec, AUDIO_DECODER);
        if(!decoder->audio_decompress) {
                log_msg(LOG_LEVEL_FATAL, "Unable to create audio decompress!\n");
                exit_uv(1);
                return false;
        }

        return true;
}

static bool
audio_fec_decode(struct state_audio_decoder                *decoder,
                 vector<pair<vector<char>, map<int, int>>> &fec_data,
                 uint32_t fec_params, audio_frame2 &received_frame)
{
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
                        assert(out_len >= (int) sizeof(audio_payload_hdr_t));
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

                                if (!audio_decoder_reconfigure(
                                        decoder, received_frame, desc.ch_count,
                                        desc.bps, desc.sample_rate,
                                        audio_tag)) {
                                        return false;
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
        auto *s = (struct acodec_state *) pbuf_data;
        struct state_audio_decoder *decoder = s->decoder;

        int input_channels = 0;

        bool first = true;
        int bufnum = 0;

        if(!cdata) {
                return false;
        }

        if (!cdata->data->m) {
                // skip frame without m-bit, we cannot determine number of channels
                // (it is maximal substream number + 1 in packet with m-bit)
                return false;
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
                                return false;
                        }
                } else if (PT_IS_AUDIO(pt) && !PT_AUDIO_IS_ENCRYPTED(pt)) {
                        if(decoder->decrypt) {
                                log_msg(LOG_LEVEL_WARNING, "Receiving unencrypted audio data "
                                                "while expecting encrypted.\n");
                                return false;
                        }
                } else {
                        if (pt == PT_Unassign_Type95) {
                                log_msg_once(LOG_LEVEL_WARNING, to_fourcc('U', 'V', 'P', 'T'), MOD_NAME "Unassigned PT 95 received, ignoring.\n");
                        } else {
                                log_msg(LOG_LEVEL_WARNING, "Unknown audio packet type: %d\n", pt);
                        }
                        return false;
                }

                unsigned int length;
                char plaintext[RTP_MAX_PACKET_LEN]; // plaintext will be actually shorter
                size_t main_hdr_len = PT_AUDIO_HAS_FEC(pt) ? sizeof(fec_payload_hdr_t) : sizeof(audio_payload_hdr_t);
                if (PT_AUDIO_IS_ENCRYPTED(pt)) {
                        uint32_t encryption_hdr = ntohl(*(uint32_t *)(void *) (cdata->data->data + main_hdr_len));
                        crypto_mode = (enum openssl_mode) (encryption_hdr >> 24);
                        if (crypto_mode == MODE_AES128_NONE || crypto_mode > MODE_AES128_MAX) {
                                log_msg(LOG_LEVEL_WARNING, "Unknown cipher mode: %d\n", (int) crypto_mode);
                                return false;
                        }
                        char *ciphertext = cdata->data->data + sizeof(crypto_payload_hdr_t) +
                                main_hdr_len;
                        int ciphertext_len = cdata->data->data_len - main_hdr_len -
                                sizeof(crypto_payload_hdr_t);

                        if((length = decoder->dec_funcs->decrypt(decoder->decrypt,
                                        ciphertext, ciphertext_len,
                                        (char *) audio_hdr, sizeof(audio_payload_hdr_t),
                                        plaintext, crypto_mode)) == 0) {
                                return false;
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

                        if (!audio_decoder_reconfigure(decoder, received_frame, input_channels, bps, sample_rate, audio_tag)) {
                                return false;
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
        long exp_bytes_unused = 0;
        packet_counter_get_bytes(decoder->packet_counter, &exp_bytes_unused,
                                 &s->received_bytes);
        packet_counter_clear(decoder->packet_counter);

        if (fec_params != 0) {
                if (!audio_fec_decode(decoder, fec_data, fec_params, received_frame)) {
                        return false;
                }
        }

        audio_frame2 decompressed = audio_codec_decompress(decoder->audio_decompress, &received_frame);
        if (!decompressed) {
                return false;
        }

        if (s->decoded == nullptr) {
                s->decoded = new audio_frame2(std::move(decompressed));
        } else {
                s->decoded->append(decompressed);
        }

        DEBUG_TIMER_STOP(audio_decode);
        return true;
}

/**
 * Second version that uses external audio configuration,
 * now it uses a struct state_audio_decoder instead an audio_frame2.
 * It does multi-channel handling.
 * @note
 * This might not ever worked (not before writing this but it doesn't seem that
 * it has broken during the time).
 * @todo
 * This shouldn't perhaps be separate function but decode_audio_frame() should
 * rather dispatch decoders according to packet type number, which is defined in
 * this particular case (PCMU has 0).
 */
int decode_audio_frame_mulaw(struct coded_data *cdata, void *pbuf_data, struct pbuf_stats *)
{
    auto *s = (struct acodec_state *) pbuf_data;
    struct state_audio_decoder *decoder = s->decoder;

    //struct state_audio_decoder *audio = (struct state_audio_decoder *)data;

    if(!cdata) return false;

    if (decoder->audio_decompress == nullptr) {
            decoder->audio_decompress = audio_codec_reconfigure(
                decoder->audio_decompress, AC_MULAW, AUDIO_DECODER);
            assert(decoder->audio_decompress != nullptr);
    }

    audio_frame2 received_frame;
    received_frame.init(1, AC_MULAW, 1, kHz8);

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
                    // Iterate through each channel.
                    for (int ch = 0 ; ch < received_frame.get_channel_count(); ch ++) {

                        received_frame.append(ch, from, cdata->data->data_len);

                        from += received_frame.get_bps();
                    }
                }

            cdata = cdata->nxt;
        }

    }

    audio_frame2 decompressed = audio_codec_decompress(decoder->audio_decompress, &received_frame);
    s->decoded = new audio_frame2(std::move(decompressed));

    return true;
}

