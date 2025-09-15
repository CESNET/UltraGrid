/**
 * @file   utils/ptp.hpp
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

#ifndef PTP_HPP_df6ffbd91502
#define PTP_HPP_df6ffbd91502

#include <atomic>
#include <thread>
#include <string>
#include "utils/spa_dll.h"

class Ptp_clock{
public:
        void start(std::string_view interface);
        void stop();

        uint64_t get_time();

private:
        std::string network_interface;

        std::atomic<bool> should_run = true;
        std::thread worker_critical;
        std::thread worker_general;

        uint64_t ptp_ts = 0;
        uint64_t local_ts = 0;
        uint64_t synth_ptp_ts = 0;
        double spa_corr = 1.0;
        spa_dll dll;

        std::atomic<uint32_t> update_count = 0;
        std::atomic<uint64_t> local_snapshot = 0;
        std::atomic<uint64_t> ptp_snapshot = 0;
        std::atomic<double> corr_snapshot = 1.0;


        void ptp_worker_general();
        void processPtpPkt(uint8_t *buf, size_t len);
};


#endif
