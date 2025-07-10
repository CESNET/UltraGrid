/**
 * @file   audio/capture/aes67.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 *
 */
/*
 * Copyright (c) 2025 CESNET, zájmové sdružení právnických osob
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

#include <stdint.h>               // for uint32_t
#include <stdlib.h>               // for free, NULL, malloc
#include <stdio.h>                // for printf
#include <string.h>               // for strcmp

#include <thread>
#include <atomic>
#include <chrono>
#include <string_view>
#include <mutex>
#include <condition_variable>
#include <queue>

#include "audio/audio_capture.h"  // for AUDIO_CAPTURE_ABI_VERSION, audio_ca...
#include "audio/types.h"          // for audio_frame
#include "host.h"                 // for INIT_NOERR
#include "lib_common.h"           // for REGISTER_MODULE, library_class
#include "tv.h"                   // for get_time_in_ns, time_ns_t, NS_TO_US
#include "types.h"                // for kHz48, device_info (ptr only)
#include "debug.h"

#include "rtp/net_udp.h"

#include "utils/color_out.h"
#include "utils/string_view_utils.hpp"
#include "utils/sdp_parser.hpp"
#include "crypto/crc.h"

#define MOD_NAME "[aes67 acap] "

#define MAX_PACKET_LEN 9000

namespace{

struct Rtp_stream{
        std::string name;
        std::string description;
        std::map<uint8_t, audio_desc> fmts;

        std::string address;
        int port;
};

using sess_id_t = uint32_t;

struct Sap_session{
        sess_id_t unique_identifier; //Hash computed from sdp username, session id and unicast address
        uint64_t sess_ver;
        uint16_t sap_hash; //Hash of sap packet that contained sdp for this version of session
        std::string name;
        std::string description;

        std::vector<Rtp_stream> streams;
};

struct Allocated_audio_frame{
        Allocated_audio_frame() = default;
        Allocated_audio_frame(const audio_desc& desc){
                frame.ch_count = desc.ch_count;
                frame.bps = desc.bps;
                frame.sample_rate = desc.sample_rate;

                frame.max_size = desc.bps * desc.ch_count * desc.sample_rate;
                data.reserve(frame.max_size);
                frame.data = data.data();
        }

        bool is_desc_same(const audio_desc& desc){
                return frame.ch_count == desc.ch_count
                        && frame.bps == desc.bps
                        && frame.sample_rate == desc.sample_rate;
        }

        audio_frame frame{};
        std::vector<char> data;
};

} //anon namespace

struct state_aes67_cap {
        std::string network_interface_name;
        std::string sap_address;
        int sap_port;
        std::string requested_sess_hash;
        unsigned req_stream_idx = 0;

        std::atomic<bool> sdp_should_run = true;
        std::thread sdp_thread;
        socket_udp *sdp_sock = nullptr;
        int curr_sap_hash = -1;
        std::map<uint16_t, sess_id_t> sap_hash_to_sess_id_map;
        std::map<sess_id_t, Sap_session> sap_sessions;

        std::atomic<bool> rtp_should_run = true;
        std::thread rtp_thread;

        std::mutex frame_mut;
        std::condition_variable frame_cond;
        Allocated_audio_frame front_frame;
        Allocated_audio_frame back_frame;

};

static audio_desc sdp_fmt_to_audio_desc(std::string_view fmt_sv){
        auto codec_sv = tokenize(fmt_sv, '/');
        auto rate_sv = tokenize(fmt_sv, '/');
        auto ch_n_sv = tokenize(fmt_sv, '/');

        audio_desc fmt{};
        if(codec_sv == "L16"){
                fmt.codec = AC_PCM;
                fmt.bps = 2;
        } else if(codec_sv == "L24"){
                fmt.codec = AC_PCM;
                fmt.bps = 3;
        }
        parse_num(rate_sv, fmt.sample_rate);
        if(!parse_num(ch_n_sv, fmt.ch_count)){
                //channel count is optional
                fmt.ch_count = 1;
        }

        return fmt;
}

static std::string_view sdp_connection_get_address(std::string_view conn){
        [[maybe_unused]] auto nettype = tokenize(conn, ' ');
        [[maybe_unused]] auto addrtype = tokenize(conn, ' ');
        auto addr_with_tll = tokenize(conn, ' ');

        return tokenize(addr_with_tll, '/');
}

static void print_sap_session(const Sap_session& sess){
        color_printf(TBOLD("Session " TGREEN("%x")) " \"%s\" (%s):\n", sess.unique_identifier,
                        std::string(sess.name).c_str(),
                        std::string(sess.description).c_str());
        int i = 0;
        for(const auto& stream : sess.streams){
                color_printf("\t" TBOLD("Stream %d") " \"%s\" (%s:%d)\n", i++, std::string(stream.name).c_str(), std::string(stream.address).c_str(), stream.port);
                color_printf("\t\tFormats:\n");
                for(const auto& fmt : stream.fmts){
                        color_printf("\t\t\t%d: %d bps, %d channels, %d rate\n",
                                        fmt.first,
                                        fmt.second.bps,
                                        fmt.second.ch_count,
                                        fmt.second.sample_rate);
                }
        }
}

static void aes67_rtp_worker(state_aes67_cap *s, Rtp_stream stream){
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "RTP Worker starting (%s:%d)\n", stream.address.c_str(), stream.port);

        auto rtp_sock = udp_init_if(stream.address.c_str(), s->network_interface_name.c_str(), stream.port, 0, 255, 0, false);

        int last_payload_type = 0;
        audio_desc desc{};
        
        using namespace std::literals::chrono_literals;
        while(s->rtp_should_run){
                int buflen = 0;
                uint8_t buffer[MAX_PACKET_LEN];
                timeval timeout {1, 0};
                buflen = udp_recv_timeout(rtp_sock, (char *)buffer, MAX_PACKET_LEN, &timeout);
                                 
                if(!buflen){
                        continue;
                }


                Rtp_pkt_view rtp_pkt = Rtp_pkt_view::from_buffer(buffer, buflen);
                if(!rtp_pkt.isValid()){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid RTP packet\n");
                        continue;
                }

                log_msg(LOG_LEVEL_DEBUG, MOD_NAME "RTP Got packet len %ld, seq %u, timestamp %u, PT %u\n",
                                rtp_pkt.data_len,
                                rtp_pkt.seq,
                                rtp_pkt.timestamp,
                                rtp_pkt.payload_type);

                std::lock_guard<std::mutex> lk(s->frame_mut);

                if(rtp_pkt.payload_type != last_payload_type){
                        auto fmt_it = stream.fmts.find(rtp_pkt.payload_type);
                        if(fmt_it == stream.fmts.end()){
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown payload type\n");
                                continue;
                        }
                        desc = fmt_it->second;
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Reconfigured\n");
                        last_payload_type = rtp_pkt.payload_type;
                }
                if(!s->back_frame.is_desc_same(desc)){
                        s->back_frame = Allocated_audio_frame(desc);
                }

                int sample_count = rtp_pkt.data_len / desc.bps;

                unsigned char *src = static_cast<unsigned char *>(rtp_pkt.data);

                const int bps = desc.bps;
                const int swap_count = bps / 2;
                for(int i = 0; i < sample_count; i++){
                        for(int j = 0; j < swap_count; j++){
                                unsigned char tmp = src[i * bps + j];
                                src[i * bps + j] = src[i * bps + bps - j - 1];
                                src[i * bps + bps - j - 1] = tmp;;
                        }
                }

                size_t to_write = std::min<size_t>(s->back_frame.frame.max_size - s->back_frame.data.size(), sample_count * desc.bps);
                s->back_frame.data.insert(s->back_frame.data.end(), src, src + to_write);
                s->back_frame.frame.data_len = s->back_frame.data.size();
                s->frame_cond.notify_one();
        }

        udp_exit(rtp_sock);

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "RTP Worker stopping\n");
}

static sess_id_t get_unique_sdp_identifier(const Sdp_view& sdp){
        auto hash = crc32buf(sdp.username.data(), sdp.username.size());
        hash = crc32buf_with_oldcrc(reinterpret_cast<const char*>(&sdp.sess_id), sizeof(sdp.sess_id), hash);
        hash = crc32buf_with_oldcrc(sdp.unicast_addr.data(), sdp.unicast_addr.size(), hash);

        return hash;
}

static bool sess_hash_is_prefix(std::string_view req, sess_id_t sess_id){
        char buf[sizeof(sess_id_t) * 2 + 1] = "";
        snprintf(buf, sizeof(buf), "%x", sess_id);

        return sv_is_prefix(buf, req);
}

static void stop_rtp_thread(state_aes67_cap *s){
        s->rtp_should_run = false;
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Joining rtp thread\n");
        s->rtp_thread.join();
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Joined\n");
        s->curr_sap_hash = -1;
}

static void start_rtp_thread(state_aes67_cap *s, const Sap_session& new_sess){
        if(s->curr_sap_hash != -1){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "RTP thread is already running\n");
                stop_rtp_thread(s);
        }

        if(new_sess.streams.size() <= s->req_stream_idx){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested stream index %d but session only has %ld stream(s)!\n", s->req_stream_idx, new_sess.streams.size());
                return;
        }

        s->rtp_should_run = true;
        s->curr_sap_hash = new_sess.sap_hash;
        s->rtp_thread = std::thread(aes67_rtp_worker, s, new_sess.streams[s->req_stream_idx]);
}

static void parse_sap(state_aes67_cap *s, std::string_view sap){
        Sap_packet_view pkt = Sap_packet_view::from_buffer(sap.data(), sap.size());

        if(!pkt.isValid()){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Invalid SDP packet\n");
                return;
        }

        if(pkt.isCompressed() || pkt.isEncrypted()){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Compressed or encrypted SAP packets are not supported\n");
                return;
        }
        if(pkt.isIpv6()){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "IPv6 SAP packets are not supported\n");
                return;
        }

        if(s->sap_hash_to_sess_id_map.find(pkt.hash) != s->sap_hash_to_sess_id_map.end()){
                if(pkt.isDeletion()){
                        uint64_t sess_id = s->sap_hash_to_sess_id_map[pkt.hash];
                        auto& sess = s->sap_sessions[sess_id];
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Removing session %x\n", sess.unique_identifier);
                        if(s->curr_sap_hash == sess.sap_hash){
                                stop_rtp_thread(s);
                        }
                        s->sap_sessions.erase(sess_id);
                } else {
                        log_msg(LOG_LEVEL_INFO, MOD_NAME "SAP with hash %x already known\n", pkt.hash);
                }

                return;
        }

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "New SAP %x\n", pkt.hash);

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Source %u.%u.%u.%u\n",
                        (unsigned char) pkt.source[0],
                        (unsigned char) pkt.source[1],
                        (unsigned char) pkt.source[2],
                        (unsigned char) pkt.source[3]);

        if(pkt.payload_type != "application/sdp"){
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unknown SAP payload type \"%s\"\n", std::string(pkt.payload_type).c_str());
                return;
        }

        Sdp_view sdp = Sdp_view::from_buffer(pkt.payload.data(), pkt.payload.size());

        if(!sdp.isValid()){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to parse SDP\n");
                return;
        }

        Sap_session new_sess{};
        new_sess.unique_identifier = get_unique_sdp_identifier(sdp);
        new_sess.sess_ver = sdp.sess_version;
        new_sess.sap_hash = pkt.hash;
        new_sess.name = sdp.session_name;
        new_sess.description = sdp.session_info;

        std::map<uint8_t, audio_desc> session_fmts;

        for(const auto& sess_attrib : sdp.session_attributes){
                if(sess_attrib.key == "rtpmap"){
                        auto val = sess_attrib.val;
                        auto id_sv = tokenize(val, ' ');
                        auto fmt_sv = tokenize(val, ' ');
                        uint8_t id = 0;
                        parse_num(id_sv, id);

                        auto fmt = sdp_fmt_to_audio_desc(fmt_sv);

                        session_fmts[id] = fmt;
                }
        }

        std::string sess_addr(sdp_connection_get_address(sdp.connection));

        for(const auto& medium : sdp.media){
                Rtp_stream new_stream{};
                new_stream.fmts = session_fmts;
                new_stream.address = sess_addr;
                new_stream.name = medium.title;
                new_stream.description = medium.media_desc;

                auto media_addr = sdp_connection_get_address(medium.connection);
                if(!media_addr.empty()){
                        new_stream.address = media_addr;
                }

                for(const auto& m_attrib : medium.attributes){
                        if(m_attrib.key == "rtpmap"){
                                auto val = m_attrib.val;
                                auto id_sv = tokenize(val, ' ');
                                auto fmt_sv = tokenize(val, ' ');
                                uint8_t id = 0;
                                if(!parse_num(id_sv, id)){
                                        log_msg(LOG_LEVEL_DEBUG, "Failed to parse rtpmap attribute \"%s\"\n", std::string(m_attrib.val).c_str());
                                        continue;
                                }

                                auto fmt = sdp_fmt_to_audio_desc(fmt_sv);

                                new_stream.fmts[id] = fmt;
                        }
                }

                auto m_desc = medium.media_desc;
                [[maybe_unused]] auto m_type = tokenize(m_desc, ' ');
                auto m_port = tokenize(m_desc, ' ');
                if(!parse_num(m_port, new_stream.port)){
                        log_msg(LOG_LEVEL_DEBUG, "Failed to parse stream port from media description \"%s\"\n", std::string(medium.media_desc).c_str());
                        continue;
                }

                new_sess.streams.push_back(std::move(new_stream));
        }

        s->sap_hash_to_sess_id_map[new_sess.sap_hash] = new_sess.unique_identifier;

        if(auto it = s->sap_sessions.find(new_sess.unique_identifier); it != s->sap_sessions.end()){
                if(new_sess.sess_ver > it->second.sess_ver){
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Got session update\n");

                        if(it->second.sap_hash == s->curr_sap_hash){
                                stop_rtp_thread(s);
                                start_rtp_thread(s, new_sess);
                        }

                        s->sap_sessions[new_sess.unique_identifier] = std::move(new_sess);
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Got SAP with lower session version\n");
                }

                return;
        }

        print_sap_session(new_sess);
        if(s->curr_sap_hash == -1 && (sess_hash_is_prefix(s->requested_sess_hash, new_sess.unique_identifier) || s->requested_sess_hash == "any")){
                start_rtp_thread(s, new_sess);
        }
        s->sap_sessions[new_sess.unique_identifier] = std::move(new_sess);
}

static void aes67_sdp_worker(state_aes67_cap *s){
        s->sdp_sock = udp_init_if(s->sap_address.c_str(), s->network_interface_name.c_str(), s->sap_port, 0, 255, 0, false);
        
        while(s->sdp_should_run){
                int buflen = 0;
                uint8_t buffer[MAX_PACKET_LEN];
                timeval timeout {1, 0};
                buflen = udp_recv_timeout(s->sdp_sock, (char *)buffer, MAX_PACKET_LEN, &timeout);

                if(!buflen)
                        continue;

                parse_sap(s, std::string_view((const char *) buffer, buflen));

        }

        udp_exit(s->sdp_sock);

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "SDP worker stopping\n");
}

static void audio_cap_aes67_probe(struct device_info **available_devices,
                int *count,
                void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = nullptr;
        *count = 0;
}

static void audio_cap_aes67_help(state_aes67_cap *s){
        color_printf("AES67 audio capture.\n");
        color_printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-s aes67" TERM_FG_RESET ":if=<network_interfce>[:sess=<hash>][:stream=<index>][:sap_address=<IP>][:sap_port=<port>]\n" TERM_RESET);
        color_printf(TERM_BOLD "\t\tif=<interface>" TERM_RESET " network interface to listen on\n");
        color_printf(TERM_BOLD "\t\tsess=<hash>" TERM_RESET " hash of the session to receive. If not specified first seen session is received\n");
        color_printf(TERM_BOLD "\t\tstream=<index>" TERM_RESET " index of stream in a session to receive. If not specified stream 0 is received\n");
        color_printf(TERM_BOLD "\t\tsap_address=<IP>" TERM_RESET " multicast IP for SAP (default 239.255.255.255)\n");
        color_printf(TERM_BOLD "\t\tsap_port=<port>" TERM_RESET " port for SAP (default 9875)\n");

        color_printf("\n");
        color_printf("Waiting for SAP:\n");

        s->requested_sess_hash = "none";
        s->sdp_thread = std::thread(aes67_sdp_worker, s);
        using namespace std::literals::chrono_literals;
        std::this_thread::sleep_for(31s); //aes67 supposedly announces every 30 seconds
        s->sdp_should_run = false;
        s->sdp_thread.join();
}

static void *audio_cap_aes67_init(struct module *parent, const char *cfg) {
        UNUSED(parent);
        auto s = std::make_unique<state_aes67_cap>();
        s->sap_address = "239.255.255.255";
        s->sap_port = 9875;

        std::string_view cfg_sv(cfg);
        while(!cfg_sv.empty()){
                auto tok = tokenize(cfg_sv, ':', '"');

                auto key = tokenize(tok, '=');
                auto val = tokenize(tok, '=');

                if(key == "if"){
                        s->network_interface_name = val;
                } else if (key == "sess"){
                        s->requested_sess_hash = val;
                } else if (key == "sap_ip"){
                        s->sap_address = val;
                } else if (key == "sap_port"){
                        if(!parse_num(val, s->sap_port)){
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to parse value for option %s\n", std::string(key).c_str());
                                return {};
                        }
                } else if (key == "stream"){
                        if(!parse_num(val, s->req_stream_idx)){
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to parse value for option %s\n", std::string(key).c_str());
                                return {};
                        }
                } else if(key == "help"){
                        audio_cap_aes67_help(s.get());
                        return INIT_NOERR;
                }
        }

        if(s->network_interface_name.empty()){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Network interface must be specified\n");
                return {};
        }

        s->sdp_thread = std::thread(aes67_sdp_worker, s.get());

        return s.release();
}

static struct audio_frame *audio_cap_aes67_read(void *state){
        auto s = static_cast<state_aes67_cap*>(state);

        s->front_frame.data.clear();
        s->front_frame.frame.data_len = 0;

        using namespace std::literals::chrono_literals;

        std::unique_lock<std::mutex> lk(s->frame_mut);

        if(!s->frame_cond.wait_for(lk, 500ms, [&]{ return s->back_frame.frame.data_len > 0; })){
                return nullptr;
        }

        std::swap(s->back_frame, s->front_frame);

        return &s->front_frame.frame;
}

static void audio_cap_aes67_done(void *state)
{
        auto s = std::unique_ptr<state_aes67_cap>(static_cast<state_aes67_cap *>(state));
        s->sdp_should_run = false;
        if(s->sdp_thread.joinable())
                s->sdp_thread.join();

        s->rtp_should_run = false;
        if(s->rtp_thread.joinable())
                s->rtp_thread.join();
        
}

static const struct audio_capture_info acap_aes67_info = {
        audio_cap_aes67_probe,
        audio_cap_aes67_init,
        audio_cap_aes67_read,
        audio_cap_aes67_done
};

REGISTER_MODULE(aes67, &acap_aes67_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);
