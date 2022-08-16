/**
 * @file   utils/color_out.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2022 CESNET, z. s. p. o.
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

#ifndef COLOR_OUT_H_
#define COLOR_OUT_H_

#define TERM_RESET      "\033[0m"
#define TERM_BOLD       "\033[1m"
#define TERM_FG_RED     "\033[31m"
#define TERM_FG_GREEN   "\033[32m"
#define TERM_FG_YELLOW  "\033[33m"
#define TERM_FG_BLUE    "\033[34m"
#define TERM_FG_MAGENTA "\033[35m"
#define TERM_FG_BRIGHT_GREEN "\033[92m"
#define TERM_FG_RESET   "\033[39m"
#define TERM_BG_BLACK   "\033[40m"
#define TERM_BG_RESET   "\033[49m"

#define TBOLD(x) TERM_BOLD x TERM_RESET
#define TGREEN(x) TERM_FG_GREEN x TERM_FG_RESET
#define TRED(x) TERM_FG_RED x TERM_FG_RESET

#ifdef __cplusplus
extern "C" {
#endif

bool color_output_init(void);
int color_printf(const char *format, ...) ATTRIBUTE(format (printf, 1, 2));
// utils
bool isMsysPty(int fd);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
#include <iostream>
#include <string>

std::string prune_ansi_sequences_str(const char *in);

/**
 * Class wrapping color output to terminal. Sample usage:
 *
 *     col() << TERM_RED << "Red message: " << TERM_RESET << "normal font.\n"
 */
class col
{
public:
        template<class T>
        col &operator<< (T val) {
                oss << val;
                return *this;
        }

        inline ~col() {
                color_printf("%s", oss.str().c_str());
        }
private:
        std::ostringstream oss;
};
#endif

#endif // defined COLOR_OUT_H_

