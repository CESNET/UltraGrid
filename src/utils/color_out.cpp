/**
 * @file   utils/color_out.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2025 CESNET
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

#ifdef _WIN32
#include <windows.h>
#endif

#include <cstdarg>
#include <cstdio>   // for vsnprintf, fileno, stdout
#include <cstdlib>  // for getenv
#include <cstring>  // for strcmp, strlen
#include <iterator> // for back_insert_iterator, back_inserter
#include <memory>   // for unique_ptr
#include <unistd.h> // for isatty

#include "debug.h"
#include "host.h"
#include "utils/color_out.h"

#define MOD_NAME "[color_out] "

using std::string;

static bool color_stdout;

#ifdef _WIN32
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

/// Taken from [rang](https://github.com/agauniyal/rang)
static bool isMsysPty(int fd) {
        // Dynamic load for binary compatibility with old Windows
        const auto ptrGetFileInformationByHandleEx
                = reinterpret_cast<decltype(&GetFileInformationByHandleEx)>(reinterpret_cast<void *>(
                                        GetProcAddress(GetModuleHandle(TEXT("kernel32.dll")),
                                                "GetFileInformationByHandleEx")));
        if (!ptrGetFileInformationByHandleEx) {
                return false;
        }

        HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        if (h == INVALID_HANDLE_VALUE) {
                return false;
        }

        // Check that it's a pipe:
        if (GetFileType(h) != FILE_TYPE_PIPE) {
                return false;
        }

        // POD type is binary compatible with FILE_NAME_INFO from WinBase.h
        // It have the same alignment and used to avoid UB in caller code
        struct MY_FILE_NAME_INFO {
                DWORD FileNameLength;
                WCHAR FileName[MAX_PATH];
        };

        auto pNameInfo = std::unique_ptr<MY_FILE_NAME_INFO>(
                        new (std::nothrow) MY_FILE_NAME_INFO());
        if (!pNameInfo) {
                return false;
        }

        // Check pipe name is template of
        // {"cygwin-","msys-"}XXXXXXXXXXXXXXX-ptyX-XX
        if (!ptrGetFileInformationByHandleEx(h, FileNameInfo, pNameInfo.get(),
                                sizeof(MY_FILE_NAME_INFO))) {
                return false;
        }
        std::wstring name(pNameInfo->FileName, pNameInfo->FileNameLength / sizeof(WCHAR));
        if ((name.find(L"msys-") == std::wstring::npos
                                && name.find(L"cygwin-") == std::wstring::npos)
                        || name.find(L"-pty") == std::wstring::npos) {
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
        return isMsysPty(fileno(stdout)) ||
               (_isatty(fileno(stdout)) &&
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

template<class OutputIt>
static int prune_ansi_sequences(const char *in, OutputIt out) {
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

string prune_ansi_sequences_str(const char *in) {
        string out;
        prune_ansi_sequences(in, std::back_inserter(out));
        return out;
}

void prune_ansi_sequences_inplace(std::string& str){
        auto c_ptr = str.data();
        auto len = prune_ansi_sequences(c_ptr, c_ptr);
        str.resize(len);
}

/// @returns cstr
char *prune_ansi_sequences_inplace_cstr(char *cstr) {
        prune_ansi_sequences(cstr, cstr);
        return cstr;
}

int color_printf(const char *format, ...) {
        va_list ap;
        va_start(ap, format);
        int size = vsnprintf(NULL, 0, format, ap);
        va_end(ap);

        // format the string
        auto buf = get_log_output().get_buffer();
        buf.append(size + 1, '\0');
        va_start(ap, format);
        vsnprintf(buf.data(), size + 1, format, ap);
        va_end(ap);

        if (!color_stdout) {
                int len = prune_ansi_sequences(buf.data(), buf.data());
                buf.get().resize(len);
        }

        buf.submit_raw();
        return size;
}

