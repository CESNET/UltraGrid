/**
 * @file   utils/timed_message.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2015 CESNET, z. s. p. o.
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

#ifndef TIMED_MESSAGE_H_
#define TIMED_MESSAGE_H_

#include <chrono>

#include "debug.h"

/**
 * Simple tool to display a warning message at most in some amount of seconds
 * (default 5).
 */
template<int log_level = LOG_LEVEL_INFO>
struct timed_message {
        inline timed_message(double sec = 5) : m_last_displayed(std::chrono::steady_clock::now() - std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::duration<double>(sec))), m_sec(sec) {}
        inline void print(const char *str) {
                auto t = std::chrono::steady_clock::now();

                if (std::chrono::duration_cast<std::chrono::duration<double>>(t - m_last_displayed).count() > m_sec) {
                        LOG(log_level) << str;
                        m_last_displayed = t;
                }

        }
private:
        std::chrono::steady_clock::time_point m_last_displayed;
        double m_sec;
};

#endif // TIMED_MESSAGE_H_

