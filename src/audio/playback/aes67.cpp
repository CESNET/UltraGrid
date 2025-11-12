/**
 * @file   audio/playback/aes67.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2025 CESNET z.s.p.o.
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

#include <algorithm>                    // for min
#include <cassert>                      // for assert
#include <cstddef>                      // for byte, size_t
#include <cstdint>                      // for INT32_MAX, uint32_t
#include <cstdlib>                      // for calloc, free
#include <cstring>                      // for strcpy, memcpy
#include <memory>
#include <optional>
#include <chrono>
#include <vector>
#include <thread>
#include <string>                       // for char_traits, basic_string
#include <string_view>                  // for operator==, basic_string_view

#include "audio/audio_playback.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "audio/resampler.hpp"
#include "compat/platform_sched.h"
#include "rtp/net_udp.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/ring_buffer.h"
#include "utils/string_view_utils.hpp"
#include "utils/ptp.hpp"
#include "utils/random.h"

#define MOD_NAME "[aes67 aplay] "

namespace {

struct Rtp_stream{
        const static int fmt_id = 96;
        int sample_rate = 0;
        int ch_count = 0;
        int bps = 0;

        std::string address;
        int port = 0;
};

struct Sap_session{
        uint64_t sess_id = 0; //Numeric session-id from RFC
        uint64_t sess_ver = 0;
        std::string origin_address;
        std::string name;
        std::string description;
        std::string ptp_id;

        /* Only 1ms packet time is guaranteed to be supported by receivers.
         */
        const static int frames_per_pkt = 48;
        Rtp_stream stream;
};

std::string get_fmt_str(int sample_rate, int ch_count, int bps){
        std::string fmt = "L";
        fmt += std::to_string(bps * 8) + "/";
        fmt += std::to_string(sample_rate) + "/";
        fmt += std::to_string(ch_count);
        return fmt;
}

uint64_t get_pkt_time_ns(unsigned frames_per_pkt, uint32_t sample_rate){
        /* Various sources say that 1ms packet time at 44.1kHz should still
         * contain 48 frames. TODO: Figure out how that should work in practice
         */
        return (frames_per_pkt * 1'000'000'000ull) / sample_rate;
}

std::string get_sdp(Sap_session& sap){
        std::string sdp;
        auto& stream = sap.stream;
        sdp += "v=0\r\n";
        sdp += "o=- " + std::to_string(sap.sess_id) + " " +  std::to_string(sap.sess_ver) + " IN IP4 " + sap.origin_address + "\r\n";
        sdp += "s=" + sap.name + "\r\n";
        sdp += "i=" + sap.description + "\r\n";
        sdp += "c=IN IP4 " + stream.address + "/127\r\n"; //TODO
        sdp += "t=0 0\r\n";
        sdp += "a=recvonly\r\n";

        sdp += "m=audio 5004 RTP/AVP " + std::to_string(stream.fmt_id) + "\r\n";
        sdp += "a=rtpmap:" + std::to_string(stream.fmt_id) + " " + get_fmt_str(stream.sample_rate, stream.ch_count, stream.bps) + "\r\n";
        char tmp_buf[32] = {};
        std::to_chars(tmp_buf, tmp_buf + sizeof(tmp_buf), get_pkt_time_ns(sap.frames_per_pkt, sap.stream.sample_rate) / 1000.f, std::chars_format::fixed);
        sdp += "a=ptime:";
        sdp += tmp_buf;
        sdp += "\r\n";

        sdp += "a=ts-refclk:ptp=IEEE1588-2008:" + sap.ptp_id + ":0\r\n";
        sdp += "a=mediaclk:direct=0\r\n";

        return sdp;
}

uint64_t nanoseconds_to_media_ts(uint64_t nanoseconds, uint32_t sample_rate){
        switch (sample_rate) {
        case 48000:
                return (nanoseconds * 3) / 62500;
        case 96000:
                return (nanoseconds * 3) / 31250;
        case 44100:
                return (nanoseconds * 441) / 10'000'000;
        default:
                return (nanoseconds * sample_rate) / 1'000'000'000;
        }
}

struct state_aes67_play{
        std::string network_interface_name;
        std::string sap_address;
        int sap_port = 0;

        std::string session_name = "UltraGrid AES67";
        std::string session_description = "placeholder";

        std::atomic<bool> sdp_should_run = true;
        std::thread sdp_thread;
        socket_udp_uniq sdp_sock;

        std::atomic<bool> rtp_should_run = true;
        std::thread rtp_thread;

        audio_desc in_desc = {};
        Sap_session sap_sess;

        ring_buffer_uniq ring_buf;
        unsigned buf_len_ms = 100;

        int32_t frame_ts_offset = 0;

        std::optional<Ptp_clock> ptpclk;

        audio_frame2_resampler resampler;
};

uint32_t get_rtp_timestamp(state_aes67_play *s){
        uint32_t timestamp = nanoseconds_to_media_ts(s->ptpclk->get_time(), s->sap_sess.stream.sample_rate);
        timestamp += s->frame_ts_offset;

        /* To prevent packets arriving with a timestamp higher than current
         * time on the receiver due to inaccurate clock synchronization we move
         * the timestamp by half a packet time into the past.
         */
        timestamp -= s->sap_sess.frames_per_pkt / 2;
        return timestamp;
}

} //anon namespace


static std::vector<unsigned char> get_sap_pkt(Sap_session& sess){
        std::string sdp = get_sdp(sess);
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "SDP\n%s\n", sdp.c_str());

        uint16_t sap_hash = time(nullptr);
        uint32_t ip_addr = inet_addr(sess.origin_address.c_str());
        std::vector<unsigned char> sap_packet;
        sap_packet.push_back(0x20); //Version and flags
        sap_packet.push_back(0x00); //Authentication headers length
        sap_packet.push_back(sap_hash >> 8);
        sap_packet.push_back(sap_hash & 0xFF);
        sap_packet.push_back((ip_addr >> 0) & 0xFF);
        sap_packet.push_back((ip_addr >> 8) & 0xFF);
        sap_packet.push_back((ip_addr >> 16) & 0xFF);
        sap_packet.push_back((ip_addr >> 24) & 0xFF);
        const char *pt = "application/sdp";
        for(const char *i = pt; *i; i++){
                sap_packet.push_back(*i);
        }
        sap_packet.push_back(0x00);
        sap_packet.insert(sap_packet.end(), sdp.begin(), sdp.end());

        return sap_packet;
}

static void create_sap_sess(state_aes67_play *s){
        Sap_session sess{};
        sess.sess_id = time(nullptr);
        sess.sess_ver = sess.sess_id;
        sess.origin_address = udp_host_addr(s->sdp_sock.get());

        /* Only 48kHz is mandatory in AES67 receivers, so for now, we only
         * support that. TODO: Support other rates as well.
         */
        sess.stream.sample_rate = 48000;
        assert(s->in_desc.sample_rate == 48000);
        sess.stream.ch_count = s->in_desc.ch_count;
        sess.stream.bps = 3;

        in_addr addr{};
        addr.s_addr = (239 << 0) | (69 << 8) | (ug_rand() << 16);
        sess.stream.address = inet_ntoa(addr);
        sess.stream.port = 5004;
        sess.ptp_id = s->ptpclk->get_clock_id_str();

        sess.name = s->session_name;
        sess.description = s->session_description;

        s->sap_sess = std::move(sess);
}

static void rtp_worker(state_aes67_play *s){
        //TODO
        auto& stream = s->sap_sess.stream;
        auto rtp_sock = socket_udp_uniq(udp_init_if(stream.address.c_str(), s->network_interface_name.c_str(), stream.port, 0, 255, 4, false));

        using clk = std::chrono::steady_clock;

        std::vector<unsigned char> rtp_pkt;
        rtp_pkt.push_back(0x80); //Version, no padding, no extension, no cssr
        rtp_pkt.push_back(stream.fmt_id);

        //Seq number
        rtp_pkt.push_back(0x00);
        rtp_pkt.push_back(0x00);

        //timestamp
        rtp_pkt.push_back(0x00);
        rtp_pkt.push_back(0x00);
        rtp_pkt.push_back(0x00);
        rtp_pkt.push_back(0x00);

        //SSRC
        uint32_t ssrc = ug_rand();
        rtp_pkt.push_back(ssrc >> 24);
        rtp_pkt.push_back(ssrc >> 16);
        rtp_pkt.push_back(ssrc >> 8);
        rtp_pkt.push_back(ssrc >> 0);

        auto hdr_size = rtp_pkt.size();

        const unsigned frames_per_packet = s->sap_sess.frames_per_pkt;
        const unsigned in_frame_size = s->in_desc.ch_count * s->in_desc.bps;
        const unsigned out_frame_size = s->sap_sess.stream.ch_count * s->sap_sess.stream.bps;
        int payload_size = out_frame_size * frames_per_packet;
        rtp_pkt.resize(hdr_size + payload_size);

        set_realtime_sched_this_thread();

        uint16_t seq = 0;
        uint32_t rtp_timestamp = get_rtp_timestamp(s);
        auto next_pkt_time = clk::now();
        auto next_pkt_ptp_time = s->ptpclk->get_time();
        do{
#ifdef _WIN32
                while(clk::now() < next_pkt_time) { }
#else
                std::this_thread::sleep_until(next_pkt_time);
#endif
                auto now_ptp_ts = s->ptpclk->get_time();
                next_pkt_ptp_time += get_pkt_time_ns(frames_per_packet, s->sap_sess.stream.sample_rate);

                auto now_next_diff = std::min(next_pkt_ptp_time - now_ptp_ts, now_ptp_ts - next_pkt_ptp_time);
                if (now_next_diff > 1'000'000'000){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Packet timing off by more than 1s\n");
                        rtp_timestamp = get_rtp_timestamp(s);
                        next_pkt_time = clk::now();
                        next_pkt_ptp_time = s->ptpclk->get_time();
                }

                if(next_pkt_ptp_time > now_ptp_ts){
                        next_pkt_time += std::chrono::nanoseconds(next_pkt_ptp_time - now_ptp_ts);
                }

                rtp_pkt[2] = seq >> 8;
                rtp_pkt[3] = seq;
                rtp_pkt[4] = rtp_timestamp >> 24;
                rtp_pkt[5] = rtp_timestamp >> 16;
                rtp_pkt[6] = rtp_timestamp >> 8;
                rtp_pkt[7] = rtp_timestamp;

                char *dst = reinterpret_cast<char *>(rtp_pkt.data() + hdr_size);

                void *src1 = nullptr;
                int size1 = 0;
                void *src2 = nullptr;
                int size2 = 0;

                unsigned frames_written = 0;

                ring_get_read_regions(s->ring_buf.get(), in_frame_size * frames_per_packet, &src1, &size1, &src2, &size2);

                change_bps2(dst, s->sap_sess.stream.bps, static_cast<const char *>(src1), s->in_desc.bps, size1, true);
                frames_written += size1 / in_frame_size;
                if(size2){
                        int offset = (size1 / in_frame_size) * out_frame_size;
                        change_bps2(dst + offset, s->sap_sess.stream.bps, static_cast<const char *>(src2), s->in_desc.bps, size2, true);
                        frames_written += size2 / in_frame_size;
                }
                ring_advance_read_idx(s->ring_buf.get(), size1 + size2);

                swap_endianity(dst, s->sap_sess.stream.bps, frames_written * s->sap_sess.stream.ch_count);

                if(frames_written < frames_per_packet){
                        memset(dst + frames_written * out_frame_size, 0, out_frame_size * (frames_per_packet - frames_written));
                }

                udp_send(rtp_sock.get(), reinterpret_cast<char*>(rtp_pkt.data()), rtp_pkt.size());
                seq += 1;
                rtp_timestamp += frames_per_packet;
        } while(s->rtp_should_run);
}

static void start_rtp_worker(state_aes67_play *s){
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Starting RTP thread\n");
        s->rtp_should_run = true;
        s->rtp_thread = std::thread(rtp_worker, s);
}

static void stop_rtp_worker(state_aes67_play *s){
        s->rtp_should_run = false;

        if(s->rtp_thread.joinable()){
                s->rtp_thread.join();
        }
}

static void sdp_worker(state_aes67_play *s){
        auto sap_packet = get_sap_pkt(s->sap_sess);

        using clk = std::chrono::steady_clock;
        auto announce_period = std::chrono::seconds(10);
        auto next_announce = clk::now();

        do{
                if(!s->ptpclk->is_locked()){
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "PTP clock lost sync, restarting...\n");
                        stop_rtp_worker(s);
                        s->ptpclk->stop();
                        //TODO Make this nice
                        sap_packet[0] |= (1 << 2);
                        udp_send(s->sdp_sock.get(), reinterpret_cast<char*>(sap_packet.data()), sap_packet.size());

                        s->ptpclk.emplace();
                        s->ptpclk->start(s->network_interface_name);
                        s->ptpclk->wait_for_lock();
                        create_sap_sess(s);
                        sap_packet = get_sap_pkt(s->sap_sess);
                        start_rtp_worker(s);
                }

                if(next_announce < clk::now()){
                        udp_send(s->sdp_sock.get(), reinterpret_cast<char*>(sap_packet.data()), sap_packet.size());
                        next_announce += announce_period;
                }

                std::this_thread::sleep_for(std::chrono::seconds(1));
        } while(s->sdp_should_run);

        //TODO Make this nice
        sap_packet[0] |= (1 << 2);
        udp_send(s->sdp_sock.get(), reinterpret_cast<char*>(sap_packet.data()), sap_packet.size());

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "SDP worker stopping\n");
}



static void audio_play_aes67_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = static_cast<device_info *>(calloc(1, sizeof(device_info)));
        strcpy((*available_devices)[0].dev, "");
        strcpy((*available_devices)[0].name, "Default aes67 output");
        *count = 1;
}

static void audio_play_aes67_help(){
        color_printf("AES67 audio output.\n");
        color_printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-r aes67" TERM_FG_RESET "[TODO]\n" TERM_RESET);
        color_printf("\n");
}

static void * audio_play_aes67_init(const struct audio_playback_opts *opts){
        auto s = std::make_unique<state_aes67_play>();
        s->sap_address = "239.255.255.255";
        s->sap_port = 9875;

        std::string_view cfg_sv(opts->cfg);
        while(!cfg_sv.empty()){
                auto tok = tokenize(cfg_sv, ':', '"');
                auto key = tokenize(tok, '=');
                auto val = tokenize(tok, '=');

                if(key == "help"){
                        audio_play_aes67_help();
                        return INIT_NOERR;
                } else if(key == "if"){
                        s->network_interface_name = val;
                } else if (key == "sap_ip"){
                        s->sap_address = val;
                } else if (key == "sap_port"){
                        if(!parse_num(val, s->sap_port)){
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to parse value for option %s\n", std::string(key).c_str());
                                return {};
                        }
                } else if (key == "ts_offset"){
                        if(!parse_num(val, s->frame_ts_offset)){
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to parse value for option %s\n", std::string(key).c_str());
                                return {};
                        }
                } else if(key == "sess_name"){
                        s->session_name = val;
                } else if(key == "sess_desc"){
                        s->session_description = val;
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option %s\n", std::string(key).c_str());
                }
        }

        s->sdp_sock.reset(udp_init_if(s->sap_address.c_str(), s->network_interface_name.c_str(), s->sap_port, 0, 255, 4, false));

        s->ptpclk.emplace();
        s->ptpclk->start(s->network_interface_name);
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Waiting for PTP lock\n");
        s->ptpclk->wait_for_lock();

        return s.release();
}

static void audio_play_aes67_put_frame(void *state, const struct audio_frame *frame){
        auto s = static_cast<state_aes67_play *>(state);

        audio_frame2 frame2(frame);
        frame2.resample(s->resampler, 48000);

        void *ptr1;
        int size1;
        void *ptr2;
        int size2;

        int to_write = frame2.get_data_len();
        ring_get_write_regions(s->ring_buf.get(), to_write, &ptr1, &size1, &ptr2, &size2);

        if(int avail = size1 + size2; to_write > avail){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Got frame of len %d, but ring has only %d free\n", frame->data_len, ring_get_available_write_size(s->ring_buf.get()));
                to_write = avail;
        }

        int frames1 = size1 / frame2.get_bps() / frame2.get_channel_count();
        int frames2 = size2 / frame2.get_bps() / frame2.get_channel_count();

        for(int i = 0; i < frame2.get_channel_count(); i++){
                mux_channel(static_cast<char *>(ptr1), frame2.get_data(i), frame2.get_bps(), frames1 * frame2.get_bps(), s->sap_sess.stream.ch_count, i, 1.0);
        }
        for(int i = 0; i < frame2.get_channel_count(); i++){
                mux_channel(static_cast<char *>(ptr2), frame2.get_data(i) + frames1 * frame2.get_bps(), frame2.get_bps(), frames2 * frame2.get_bps(), s->sap_sess.stream.ch_count, i, 1.0);
        }

        ring_advance_write_idx(s->ring_buf.get(), to_write);
}

static bool is_format_supported(void *data, size_t *len){
        struct audio_desc desc;
        if (*len < sizeof(desc)) {
                return false;
        } else {
                memcpy(&desc, data, sizeof(desc));
        }

        return desc.codec == AC_PCM && desc.bps >= 2 && desc.bps <= 3;
}

static bool audio_play_aes67_ctl(void *state, int request, void *data, size_t *len){
        UNUSED(state);

        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                return is_format_supported(data, len);
        default:
                return false;

        }
}

static void stop_session(state_aes67_play *s){
        s->sdp_should_run = false;
        stop_rtp_worker(s);

        if(s->sdp_thread.joinable()){
                s->sdp_thread.join();
        }
}

static bool audio_play_aes67_reconfigure(void *state, struct audio_desc desc){
        auto s = static_cast<state_aes67_play *>(state);

        stop_session(s);

        desc.sample_rate = 48000; //TODO: Do this more properly
        s->in_desc = desc;
        create_sap_sess(s);

        unsigned ring_size = (s->buf_len_ms * desc.sample_rate / 1000) * desc.ch_count * desc.bps * 2;
        s->ring_buf.reset(ring_buffer_init(ring_size));

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Starting SDP thread\n");
        s->sdp_should_run = true;
        s->sdp_thread = std::thread(sdp_worker, s);

        start_rtp_worker(s);

        return true;
}

static void audio_play_aes67_done(void *state){
        auto s = static_cast<state_aes67_play *>(state);

        s->ptpclk->stop();
        stop_session(s);

        delete s;
}

static const struct audio_playback_info aplay_aes67_info = {
        audio_play_aes67_probe,
        audio_play_aes67_init,
        audio_play_aes67_put_frame,
        audio_play_aes67_ctl,
        audio_play_aes67_reconfigure,
        audio_play_aes67_done
};

REGISTER_MODULE(aes67, &aplay_aes67_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

