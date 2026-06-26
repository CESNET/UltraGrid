/**
 * @file   utils/color_out.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2026 CESNET, zájmové sdružení právnických osob
 * All rigahts reserved.
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

#ifdef _WIN32
#include <windows.h>
#endif

#include <stdarg.h>
#include <stdio.h>   // for vsnprintf, fileno, stdout
#include <stdlib.h>  // for getenv
#include <string.h>  // for strcmp, strlen
#include <unistd.h> // for isatty

#include "compat/c23.h"       // IWYU pragma: keep
#include "debug.h"
#include "host.h"
#include "utils/color_out.h"

#define MOD_NAME "[color_out] "

static bool color_stdout;

#ifdef _WIN32
#include <wchar.h>

/// Taken from [rang](https://github.com/agauniyal/rang)
static bool setWinTermAnsiColors(DWORD stream) {
        HANDLE h = GetStdHandle(stream);
        if (h == INVALID_HANDLE_VALUE) {
                return false;
        }
        DWORD dwMode = 0;
        if (!GetConsoleMode(h, &dwMode)) {
                return false;
        }
        dwMode |= ENABLE_VIRTUAL_TERMINAL_PROCESSING;
        if (!SetConsoleMode(h, dwMode)) {
                return false;
        }
        return true;
}
#endif // defined _WIN32

ADD_TO_PARAM("log-color", "* log-color[=no]\n"
                 "  Force enable/disable ANSI text formatting.\n");
ADD_TO_PARAM("log-nocolor", "* log-nocolor\n"
                 "  Force disable ANSI text formatting.\n");
/**
 * @returns whether stdout can process ANSI escape sequences
 */
static bool
is_output_color()
{
        const char *const param_val = get_commandline_param("log-color");
        if (param_val != nullptr) {
                return strcmp(param_val, "no") != 0;
        }
        if (get_commandline_param("log-nocolor") != nullptr) {
                fprintf(stderr,
                        MOD_NAME "The param log-nocolor is deprecated, use "
                                 "'log-color=no' instead\n");
                return false;
        }
        const char *env_val = getenv("ULTRAGRID_COLOR_OUT");
        if (env_val != nullptr && strlen(env_val) > 0) {
                return strcmp(env_val, "0") != 0;
        }
#ifdef _WIN32
        return (_isatty(fileno(stdout)) &&
                setWinTermAnsiColors(STD_OUTPUT_HANDLE));
#else
        return isatty(fileno(stdout));
#endif
}
/// sets internal variable color_stdout and returns its contents
bool
color_output_init()
{
        color_stdout = is_output_color();
        return color_stdout;
}

int
prune_ansi_sequences(char *str)
{
        const char *in = str;
        char *out = str;
        char c = *in;
        bool in_control = false;
        int written = 0;
        while (c != '\0') {
                switch (c) {
                        case '\033':
                                in_control = true;
                                break;
                        case 'm':
                                   if (in_control) {
                                           in_control = false;
                                           break;
                                   }
                                   // fall through
                        default:
                                   if (!in_control) {
                                           *out++ = c;
                                           written++;
                                   }
                }
                c = *++in;
        }
        *out = '\0';
        return written;
}

int color_printf(const char *format, ...) {
        va_list ap;

        // format the string
        char buf[10001];
        va_start(ap, format);
        vsnprintf(buf, sizeof buf, format, ap);
        va_end(ap);

        if (!color_stdout) {
                prune_ansi_sequences(buf);
        }

        return log_puts(buf);
}

