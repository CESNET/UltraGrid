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

#ifdef __cplusplus
extern "C" {
#endif

void debug_dump(void*lp, int len);

#include "host.h"

#define error_msg(...) log_msg(LOG_LEVEL_ERROR, __VA_ARGS__)
#define verbose_msg(...) log_msg(LOG_LEVEL_VERBOSE, __VA_ARGS__)
#define debug_msg(...) log_msg(LOG_LEVEL_DEBUG, "[pid/%d +%d %s] ", getpid(), __LINE__, __FILE__), log_msg(LOG_LEVEL_DEBUG, __VA_ARGS__)
void log_msg(int log_level, const char *format, ...);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <sstream>
// Log, version 0.1: a simple logging class
class Logger
{
public:
        inline Logger(int l) : level(l) {}
        inline virtual ~Logger() {
                log_msg(level, os.str().c_str());
        }
        inline std::ostringstream& Get() {
                return os;
        }
protected:
        std::ostringstream os;
private:
        int level;
};

#define LOG(level) \
if (level > log_level) ; \
else Logger(level).Get()

#endif

#endif
