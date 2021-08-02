/*
 * FILE:    debug.h
 * PROGRAM: RAT
 * AUTHOR:  Isidor Kouvelas + Colin Perkins + Mark Handley + Orion Hodson
 * 
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 * Copyright (c) 1995-2000 University College London
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#ifndef _RAT_DEBUG_H
#define _RAT_DEBUG_H

#ifndef __cplusplus
#include <stdbool.h>
#endif // ! defined __cplusplus

#define UNUSED(x)	(x=x)

#define LOG_LEVEL_QUIET   0 ///< suppress all logging
#define LOG_LEVEL_FATAL   1 ///< errors that prevent UG run
#define LOG_LEVEL_ERROR   2 ///< general errors
#define LOG_LEVEL_WARNING 3 ///< less severe errors
#define LOG_LEVEL_NOTICE  4 ///< information that may be interesting
#define LOG_LEVEL_INFO    5 ///< normal reporting
#define LOG_LEVEL_VERBOSE 6 ///< display more messages but no more than
                            ///< one of a kind every ~1 sec
#define LOG_LEVEL_DEBUG   7 ///< like LOG_LEVEL_VERBOSE, freq. approx.
                            ///< 1 message every video frame
#define LOG_LEVEL_DEBUG2  8 ///< even more verbose - eg. every packet can
                            ///< be logged
#define LOG_LEVEL_MAX LOG_LEVEL_DEBUG2
extern volatile int log_level;


#ifdef __cplusplus
extern "C" {
#endif

void debug_dump(void*lp, int len);

#ifndef ATTRIBUTE
#define ATTRIBUTE(a)
#endif

#define error_msg(...) log_msg(LOG_LEVEL_ERROR, __VA_ARGS__)
#define verbose_msg(...) log_msg(LOG_LEVEL_VERBOSE, __VA_ARGS__)
///#define debug_msg(...) log_msg(LOG_LEVEL_DEBUG, "[pid/%d +%d %s] ", getpid(), __LINE__, __FILE__), log_msg(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define debug_msg(...) log_msg(LOG_LEVEL_DEBUG, __VA_ARGS__)
void log_msg(int log_level, const char *format, ...) ATTRIBUTE(format (printf, 2, 3));
void log_perror(int log_level, const char *msg);

bool set_log_level(const char *optarg, bool *logger_repeat_msgs, int *show_timestamps);

#ifdef __cplusplus
}
#endif

#define CUMULATIVE_REPORTS_INTERVAL 30

#ifdef __cplusplus
#include <atomic>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include "compat/platform_time.h"
#include "rang.hpp"

class keyboard_control; // friend

// Log, version 0.1: a simple logging class
class Logger
{
public:
        static void preinit(bool skip_repeated, int show_timetamps);
        inline Logger(int l) : level(l) {}
        inline ~Logger() {
                rang::fg color = rang::fg::reset;
                rang::style style = rang::style::reset;

                std::string msg = oss.str();

                if (skip_repeated && rang::rang_implementation::isTerminal(std::clog.rdbuf())) {
                        auto last = last_msg.exchange(nullptr);
                        if (last != nullptr && last->msg == msg) {
                                int count = last->count += 1;
                                auto current = last_msg.exchange(last);
                                delete current;
                                std::clog << "    Last message repeated " << count << " times\r";
                                return;
                        }
                        if (last != nullptr) {
                                if (last->count > 0) {
                                        std::clog << "\n";
                                }
                                delete last;
                        }
                }

                switch (level) {
                case LOG_LEVEL_FATAL:   color = rang::fg::red; style = rang::style::bold; break;
                case LOG_LEVEL_ERROR:   color = rang::fg::red; break;
                case LOG_LEVEL_WARNING: color = rang::fg::yellow; break;
                case LOG_LEVEL_NOTICE:  color = rang::fg::green; break;
                }

                std::ostringstream timestamp;
                if (show_timestamps == 1 || (show_timestamps == -1 && log_level >= LOG_LEVEL_VERBOSE)) {
                        auto time_ms = time_since_epoch_in_ms();
                        timestamp << "[" << std::fixed << std::setprecision(3) << time_ms / 1000.0  << "] ";
                }

                std::clog << style << color << timestamp.str() << msg << rang::style::reset << rang::fg::reset;

                auto *lmsg = new last_message{std::move(msg)};
                auto current = last_msg.exchange(lmsg);
                delete current;
        }
        inline std::ostream& Get() {
                return oss;
        }
private:
        int level;
        std::ostringstream oss;

        static std::atomic<bool> skip_repeated;
        static int show_timestamps;
        struct last_message {
                std::string msg;
                int count{0};
        };
        static std::atomic<last_message *> last_msg; // leaks last message upon exit

        friend class keyboard_control;
};

#define LOG(level) \
if (level <= log_level) Logger(level).Get()

#endif

#ifdef DEBUG
#define DEBUG_TIMER_EVENT(name) struct timeval name = { 0, 0 }; gettimeofday(&name, NULL)
#define DEBUG_TIMER_START(name) DEBUG_TIMER_EVENT(name##_start);
#define DEBUG_TIMER_STOP(name) DEBUG_TIMER_EVENT(name##_stop); log_msg(LOG_LEVEL_DEBUG2, "%s duration: %lf s\n", #name, tv_diff(name##_stop, name##_start)) // NOLINT(cppcoreguidelines-pro-type-vararg, hicpp-vararg)
#else
#define DEBUG_TIMER_START(name)
#define DEBUG_TIMER_STOP(name)
#endif

#endif
