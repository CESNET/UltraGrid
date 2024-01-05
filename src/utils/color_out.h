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

#ifndef __cplusplus
#include <stdbool.h>
#endif

#define TERM_RESET      "\033[0m"
#define TERM_BOLD       "\033[1m"
#define TERM_UNDERLINE  "\033[4m"
#define TERM_FG_RED     "\033[31m"
#define TERM_FG_GREEN   "\033[32m"
#define TERM_FG_YELLOW  "\033[33m"
#define TERM_FG_BLUE    "\033[34m"
#define TERM_FG_MAGENTA "\033[35m"
#define TERM_FG_BRIGHT_BLACK "\033[90m"
#define TERM_FG_BRIGHT_GREEN "\033[92m"
#define TERM_FG_BRIGHT_BLUE "\033[94m"
#define TERM_FG_DARK_YELLOW "\033[38;5;220m"
#define TERM_FG_RESET   "\033[39m"
#define TERM_BG_BLACK   "\033[40m"
#define TERM_BG_RESET   "\033[49m"
// 256 color palette
#define T_PEACH_FUZZ 209
#define T256_FG(col, x) "\033[38;5;" #col "m" x TERM_FG_RESET
#define S256_FG(col, x) T256_FG(col, << x <<)

#define TBOLD(x) TERM_BOLD x TERM_RESET
#define TUNDERLINE(x) TERM_UNDERLINE x TERM_RESET
#define TGREEN(x) TERM_FG_GREEN x TERM_FG_RESET
#define TYELLOW(x) TERM_FG_YELLOW x TERM_FG_RESET
#define TMAGENTA(x) TERM_FG_MAGENTA x TERM_FG_RESET
#define TRED(x) TERM_FG_RED x TERM_FG_RESET
#define TBRIGHT_BLUE(x) TERM_FG_BRIGHT_BLUE x TERM_FG_RESET
#define TDARK_YELLOW(x) TERM_FG_DARK_YELLOW x TERM_FG_RESET

// macros intended for C++ streams - enclosed arg doesn't need to be C string literals
#define SBOLD(x) TBOLD(<< x <<)
#define SUNDERLINE(x) TUNDERLINE(<< x <<)
#define SGREEN(x) TGREEN(<< x <<)
#define SYELLOW(x) TYELLOW(<< x <<)
#define SMAGENTA(x) TMAGENTA(<< x <<)
#define SRED(x) TRED(<< x <<)
#define SBRIGHT_BLUE(x) TBRIGHT_BLUE(<< x <<)
#define SDARK_YELLOW(x) TDARK_YELLOW(<< x <<)

#ifdef __cplusplus
extern "C" {
#endif

bool color_output_init(void);
int color_printf(const char *format, ...) __attribute__((format (printf, 1, 2)));

char *prune_ansi_sequences_inplace_cstr(char *cstr);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
#include <iostream>
#include <sstream>
#include <string>

std::string prune_ansi_sequences_str(const char *in);
void prune_ansi_sequences_inplace(std::string& str);

/**
 * Class wrapping color output to terminal. Sample usage:
 *
 *     col() << SBOLD("Red message: ") << "normal font.\n"
 *
 * TERM_* or T<PROP>() macros can be used as well.
 */
class col
{
public:
        template<class T>
        col &operator<< (const T& val) {
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

