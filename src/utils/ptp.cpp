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
#include <algorithm>
#include "rtp/net_udp.h"
#include "debug.h"

#define MOD_NAME "[PTP] "
#define PTP_PORT_CRITICAL 319
#define PTP_PORT_GENERAL 320
#define MAX_PACKET_LEN 1024
#define PTP_ADDRESS "224.0.1.129"

#define PTP_MSG_ANNOUNCE 0xb
#define PTP_MSG_SYNC 0x0
#define PTP_MSG_FOLLOWUP 0x8

using clk = std::chrono::steady_clock;

namespace {


} //anon namespace

void Ptp_clock::processPtpPkt(uint8_t *buf, size_t len){
        assert(len >= 34);

        int version = buf[1] & 0x0F;
        if(version != 2)
                return;

        uint64_t pkt_ts = std::chrono::nanoseconds(clk::now().time_since_epoch()).count();

        int msgType = buf[0] & 0x0F;

        uint16_t flags = (buf[6] << 8) | buf[7];
        uint16_t seq = (buf[30] << 8) | buf[31];

        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Msg type %x, flags %x, seq = %u\n", msgType, flags, seq);

        if(msgType == PTP_MSG_FOLLOWUP || msgType == PTP_MSG_SYNC){
                uint64_t sec = (buf[34] << 40) | (buf[35] << 32) | (buf[36] << 24) | (buf[37] << 16) | (buf[38] << 8) | (buf[39]);
                uint32_t nsec = (buf[40] << 24) | (buf[41] << 16) | (buf[42] << 8) | (buf[43]);

                uint64_t new_ptp_ts = sec * 1'000'000'000 + nsec;
                uint64_t new_local_ts = pkt_ts;
                if(synth_ptp_ts == 0) synth_ptp_ts = new_ptp_ts;

                if(local_ts != 0){
                        auto delta_local = new_local_ts - local_ts;
                        auto delta_ptp = new_ptp_ts - ptp_ts;

                        synth_ptp_ts += delta_local * spa_corr;

                        spa_corr = spa_dll_update(&dll, (int64_t) synth_ptp_ts - (int64_t) new_ptp_ts);
                }

                auto new_offset = new_ptp_ts - new_local_ts;
                if(offset == 0)
                        offset = new_offset;
                else
                        offset = offset / 2 + new_offset / 2;

                ptp_ts = new_ptp_ts;
                local_ts = new_local_ts;
        }

}

uint64_t Ptp_clock::get_time(){
        uint64_t now_ts = std::chrono::nanoseconds(clk::now().time_since_epoch()).count();

        return now_ts + offset;
}

void Ptp_clock::ptp_worker_general(){
        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Init sock %s on %s port %d\n", PTP_ADDRESS, network_interface.c_str(), PTP_PORT_GENERAL);
        auto ptp_sock = socket_udp_uniq(udp_init_if(PTP_ADDRESS, network_interface.c_str(), PTP_PORT_CRITICAL, 0, 255, 4, false));

        if(!ptp_sock){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create sock\n");
                return;
        }

        if(true){
                printf("Setting realtime...\t");
                struct sched_param p;
                memset(&p, 0, sizeof(p));
                p.sched_priority = 99;
                int ret = sched_setscheduler(0, SCHED_RR|SCHED_RESET_ON_FORK, &p);
                printf(ret < 0 ? "failed\n" : "success\n");
        }

        spa_dll_init(&dll);
        spa_dll_set_bw(&dll, 0.05, 250'000'000, 1'000'000'000);

        while(should_run){
                int buflen = 0;
                uint8_t buffer[MAX_PACKET_LEN];
                timeval timeout {1, 0};
                buflen = udp_recv_timeout(ptp_sock.get(), (char *)buffer, MAX_PACKET_LEN, &timeout);
                if(buflen == 0)
                        continue;

                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Got msg len %d\n", buflen);

                processPtpPkt(buffer, buflen);

        }
}

void Ptp_clock::start(std::string_view interface){
        network_interface = interface;

        worker_general = std::thread(&Ptp_clock::ptp_worker_general, this);
}

void Ptp_clock::stop(){
        should_run = false;

        if(worker_general.joinable()){
                worker_general.join();
        }
}
