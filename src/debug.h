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

#define error_msg(...) log_msg(LOG_LEVEL_ERROR, __VA_ARGS__)
#define verbose_msg(...) log_msg(LOG_LEVEL_VERBOSE, __VA_ARGS__)
///#define debug_msg(...) log_msg(LOG_LEVEL_DEBUG, "[pid/%d +%d %s] ", getpid(), __LINE__, __FILE__), log_msg(LOG_LEVEL_DEBUG, __VA_ARGS__)
#define debug_msg(...) log_msg(LOG_LEVEL_DEBUG, __VA_ARGS__)
void log_msg(int log_level, const char *format, ...) ATTRIBUTE(format (printf, 2, 3));;

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <iomanip>
#include <iostream>
#include "compat/platform_time.h"
#include "rang.hpp"

// Log, version 0.1: a simple logging class
class Logger
{
public:
        inline Logger(int l) : level(l) {}
        inline ~Logger() {
                std::cerr << rang::style::reset << rang::fg::reset;
        }
        inline std::ostream& Get() {
                rang::fg color = rang::fg::reset;
                rang::style style = rang::style::reset;

                switch (level) {
                case LOG_LEVEL_FATAL:   color = rang::fg::red; style = rang::style::bold; break;
                case LOG_LEVEL_ERROR:   color = rang::fg::red; break;
                case LOG_LEVEL_WARNING: color = rang::fg::yellow; break;
                case LOG_LEVEL_NOTICE:  color = rang::fg::green; break;
                }
                std::cerr << style << color;
                if (log_level >= LOG_LEVEL_VERBOSE) {
                        unsigned long long time_ms = time_since_epoch_in_ms();
                        auto flags = std::cerr.flags();
                        auto precision = std::cerr.precision();
                        std::cerr << "[" << std::fixed << std::setprecision(3) << time_ms / 1000.0  << "] ";
                        std::cerr.precision(precision);
                        std::cerr.flags(flags);
                }

                return std::cerr;
        }
private:
        int level;
};

#define LOG(level) \
if (level > log_level) ; \
else Logger(level).Get()

#endif

#endif
