/**
 * @file   utils/ptp.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2025-2026 CESNET z.s.p.o.
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

#include "ptp.hpp"
#include <chrono>
#include <cstddef>
#include <algorithm>
#include "rtp/net_udp.h"
#include "utils/thread.h"
#include "compat/platform_sched.h"
#include "debug.h"

#define MOD_NAME "[PTP] "
#define PTP_PORT_EVENT 319
#define PTP_PORT_GENERAL 320
#define MAX_PACKET_LEN 128
#define PTP_ADDRESS "224.0.1.129"

#define PTP_FLAG_TWOSTEP 0x200

#define PTP_MSG_ANNOUNCE 0xb
#define PTP_MSG_SYNC 0x0
#define PTP_MSG_FOLLOWUP 0x8
#define PTP_MSG_DELAY_REQ 0x1
#define PTP_MSG_DELAY_RESP 0x9
#define PTP_MSG_MANAGEMENT 0xd

#define LOCK_THRESH_uS_LOW 200
#define LOCK_THRESH_uS_HIGH 400

using clk = std::chrono::steady_clock;

namespace {

struct Timestamped_pkt{
        uint64_t local_ts;
        uint8_t buf[MAX_PACKET_LEN]; //Event packets should not be larger than 54B
        unsigned buflen = 0;
};

template<typename T, unsigned n>
T read_val(uint8_t *ptr){
        T ret{};
        for(unsigned i = 0; i < n; ++i){
                ret <<= 8;
                ret |= ptr[i];
        }

        return ret;
}

template<unsigned n, typename T>
void write_val(uint8_t *ptr, T val){
        for(unsigned i = 0; i < n; ++i){
                ptr[i] = (val >> ((n - 1 - i) * 8)) & 0xFF;
        }
}


struct Ptp_hdr{
        bool valid = false;

        uint8_t msg_type;
        uint16_t msg_len;
        uint16_t flags;
        uint64_t correction_field;
        uint64_t clock_identity;
        uint16_t port_number;
        uint16_t seq;
        uint8_t log_msg_interval;
};

const char *clock_id_to_str(uint64_t id){
        static char buf[16 + 7 + 1] = {};

        snprintf(buf, sizeof(buf), "%02X-%02X-%02X-%02X-%02X-%02X-%02X-%02X",
                        (int)(id >> 56) & 0xFF,
                        (int)(id >> 48) & 0xFF,
                        (int)(id >> 40) & 0xFF,
                        (int)(id >> 32) & 0xFF,
                        (int)(id >> 24) & 0xFF,
                        (int)(id >> 16) & 0xFF,
                        (int)(id >> 8) & 0xFF,
                        (int)id & 0xFF);
        return buf;
}

Ptp_hdr parse_ptp_header(uint8_t *buf, size_t len){
        const size_t header_length = 34;

        Ptp_hdr ret{};

        if(len < header_length){
                return ret;
        }

        int version = buf[1] & 0x0F;
        if(version != 2){
                return ret;
        }

        ret.msg_type = buf[0] & 0x0F;
        ret.msg_len = read_val<uint16_t, 2>(&buf[2]);
        ret.flags = read_val<uint16_t, 2>(&buf[6]);
        ret.correction_field = read_val<uint64_t, 8>(&buf[8]);
        ret.clock_identity = read_val<uint64_t, 8>(&buf[20]);
        ret.port_number = read_val<uint16_t, 2>(&buf[28]);
        ret.seq = read_val<uint16_t, 2>(&buf[30]);
        ret.log_msg_interval = buf[33];

        if(ret.msg_len >= header_length)
                ret.valid = true;

        return ret;
}

} //anon namespace

void Ptp_clock::update_clock(uint64_t new_local_ts, uint64_t new_ptp_ts){
        if(synth_ptp_ts == 0) synth_ptp_ts = new_ptp_ts;

        auto delta_local = new_local_ts - local_ts;

        if(local_ts == 0){
                ptp_ts = new_ptp_ts;
                local_ts = new_local_ts;
                return;
        }
        ptp_ts = new_ptp_ts;
        local_ts = new_local_ts;

        synth_ptp_ts += delta_local * spa_corr;

        int64_t error_ns = (int64_t) synth_ptp_ts - (int64_t) new_ptp_ts;
        spa_corr = spa_dll_update(&dll, error_ns);

        avg.push(std::abs(error_ns));

        double avg_error_us = avg.get() / 1000;

        if(avg.size() >= 10 && avg_error_us < LOCK_THRESH_uS_LOW){
                if(!locked){
                        std::lock_guard<std::mutex> l(mut);
                        locked = true;
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Clock locked\n");
                        cv.notify_all();
                }
        } else if(avg_error_us > LOCK_THRESH_uS_HIGH && locked){
                std::lock_guard<std::mutex> l(mut);
                locked = false;
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Clock unlocked (avg err: %f)\n", avg_error_us);
                cv.notify_all();
        }

        update_count.fetch_add(1, std::memory_order_seq_cst);
        local_snapshot.store(new_local_ts, std::memory_order_seq_cst);
        ptp_snapshot.store(synth_ptp_ts, std::memory_order_seq_cst);
        corr_snapshot.store(spa_corr, std::memory_order_seq_cst);
        update_count.fetch_add(1, std::memory_order_seq_cst);
}

void Ptp_clock::drop_sync_pkts_older_than(uint16_t seq){
        auto new_end = std::remove_if(sync_pkts.begin(), sync_pkts.end(), [seq](const detail::Sync_pkt_data& pkt){ return pkt.seq <= seq; });
        sync_pkts.erase(new_end, sync_pkts.end());
}

void Ptp_clock::processPtpPkt(uint8_t *buf, size_t len, uint64_t pkt_ts){
        Ptp_hdr header = parse_ptp_header(buf, len);
        if(!header.valid){
                return;
        }

        if(header.clock_identity != clock_identity){
                if(clock_identity == 0){
                        set_clock_identity(header.clock_identity);
                } else {
                        return;
                }
        }

        if(header.msg_type == PTP_MSG_SYNC && (header.flags & PTP_FLAG_TWOSTEP)){
                uint64_t sec = read_val<uint64_t, 6>(&buf[34]);
                uint32_t nsec = read_val<uint32_t, 4>(&buf[40]);

                uint64_t new_ptp_ts = sec * 1'000'000'000 + nsec;

                sync_pkts.push_back({header.seq, new_ptp_ts, pkt_ts});
                return;
        }

        if(header.msg_type == PTP_MSG_FOLLOWUP){
                uint64_t sec = read_val<uint64_t, 6>(&buf[34]);
                uint32_t nsec = read_val<uint32_t, 4>(&buf[40]);

                uint64_t new_ptp_ts = sec * 1'000'000'000 + nsec;

                auto it = std::find_if(sync_pkts.begin(), sync_pkts.end(), [=](const detail::Sync_pkt_data& pkt){ return pkt.seq == header.seq; });

                if(it == sync_pkts.end()){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Sync pkt for followup not found\n");
                        return;
                }

                update_clock(it->local_ts, new_ptp_ts);

                drop_sync_pkts_older_than(header.seq);

                return;
        }

}

void Ptp_clock::wait_for_lock(){
	std::unique_lock<std::mutex> l(mut);

	cv.wait(l, [&]{ return locked; });
}

uint64_t Ptp_clock::get_time(){
        uint32_t seq0;
        uint32_t seq1;

        uint64_t l;
        uint64_t p;
        double c;

        do{
                seq0 = update_count.load(std::memory_order_seq_cst);
                l = local_snapshot.load(std::memory_order_seq_cst);
                p = ptp_snapshot.load(std::memory_order_seq_cst);
                c = corr_snapshot.load(std::memory_order_seq_cst);
                seq1 = update_count.load(std::memory_order_seq_cst);
        }while(seq0 != seq1);

        uint64_t now_ts = std::chrono::nanoseconds(clk::now().time_since_epoch()).count();
        auto delta_local = now_ts - l;

        return p + delta_local * c;
}

void Ptp_clock::ptp_worker_event(){
        set_thread_name(__func__);
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Init sock %s on %s port %d\n", PTP_ADDRESS, network_interface.c_str(), PTP_PORT_EVENT);
        auto ptp_sock = socket_udp_uniq(udp_init_if(PTP_ADDRESS, network_interface.c_str(), PTP_PORT_EVENT, 0, 255, 4, false));

        if(!ptp_sock){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create sock\n");
                return;
        }

        set_realtime_sched_this_thread();

        while(should_run){
                Timestamped_pkt pkt;
                timeval timeout {1, 0};
                pkt.buflen = udp_recv_timeout(ptp_sock.get(), (char *)pkt.buf, MAX_PACKET_LEN, &timeout);
                pkt.local_ts = std::chrono::nanoseconds(clk::now().time_since_epoch()).count();
                if(pkt.buflen == 0)
                        continue;

                Ptp_hdr hdr = parse_ptp_header(pkt.buf, pkt.buflen);
                if(!hdr.valid){
                        continue;
                }

                ring_buffer_write(event_pkt_ring.get(), reinterpret_cast<const char*>(&pkt), sizeof(pkt));

        }
}

void Ptp_clock::ptp_worker_general(){
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Init sock %s on %s port %d\n", PTP_ADDRESS, network_interface.c_str(), PTP_PORT_GENERAL);
        auto ptp_sock = socket_udp_uniq(udp_init_if(PTP_ADDRESS, network_interface.c_str(), PTP_PORT_GENERAL, 0, 255, 4, false));

        if(!ptp_sock){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create sock\n");
                return;
        }

        spa_dll_init(&dll);
        spa_dll_set_bw(&dll, 0.05, 250'000'000, 1'000'000'000); //TODO

		auto last_report = std::chrono::steady_clock::now();

        while(should_run){
                int buflen = 0;
                uint8_t buffer[MAX_PACKET_LEN];
                timeval timeout {0, 33'000'000};
                buflen = udp_recv_timeout(ptp_sock.get(), (char *)buffer, MAX_PACKET_LEN, &timeout);

                auto ring_avail = ring_get_current_size(event_pkt_ring.get());
                while(ring_avail >= (int) sizeof(Timestamped_pkt)){
                        Timestamped_pkt pkt;
                        ring_buffer_read(event_pkt_ring.get(), reinterpret_cast<char *>(&pkt), sizeof(pkt));
                        processPtpPkt(pkt.buf, pkt.buflen, pkt.local_ts);
                        ring_avail -= sizeof(Timestamped_pkt);
                }

                if(buflen == 0)
                        continue;

                processPtpPkt(buffer, buflen, 0);

				if(std::chrono::steady_clock::now() - last_report > std::chrono::seconds(5)){
					last_report = std::chrono::steady_clock::now();

					double avg_err_us = avg.get() / 1000;
					log_msg(LOG_LEVEL_INFO, MOD_NAME "Average absoulute err %.3f usec\n", avg_err_us);
				}
        }
}

const char* Ptp_clock::get_clock_id_str(){
        std::lock_guard<std::mutex> l(mut);
        return clock_id_to_str(clock_identity);
}

void Ptp_clock::set_clock_identity(uint64_t id){
        std::lock_guard<std::mutex> l(mut);
        clock_identity = id;
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Selecting clock %s\n", clock_id_to_str(clock_identity));
}

bool Ptp_clock::is_locked() const{
        std::lock_guard<std::mutex> l(mut);
        return locked;
}

void Ptp_clock::start(std::string_view interface_name){
        network_interface = interface_name;

        constexpr size_t ring_size = sizeof(Timestamped_pkt) * 1000;
        event_pkt_ring.reset(ring_buffer_init(ring_size));

        worker_general = std::thread(&Ptp_clock::ptp_worker_general, this);
        worker_event = std::thread(&Ptp_clock::ptp_worker_event, this);
}

void Ptp_clock::stop(){
        should_run = false;

        if(worker_event.joinable()){
                worker_event.join();
        }

        if(worker_general.joinable()){
                worker_general.join();
        }
}
