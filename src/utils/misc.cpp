/**
 * @file   utils/misc.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2021 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <unistd.h>

#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <sstream>
#include <string>

#include "debug.h"
#include "utils/misc.h"
#include "utils/color_out.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#define STRERROR_BUF_LEN 1024

using std::invalid_argument;
using std::out_of_range;
using std::stoll;

int clampi(long long val, int lo, int hi) {
        if (val < lo) {
                return lo;
        }
        if (val > hi) {
                return hi;
        }
        return val;
}

/**
 * Converts units in format <val>[.<val>][kMG] to integral representation.
 *
 * @param   str string to be parsed
 * @returns     positive integral representation of the string
 * @returns     LLONG_MIN if error
 */
long long unit_evaluate(const char *str) {
        double ret = unit_evaluate_dbl(str, false);

        if (ret == NAN || ret >= nexttoward((double) LLONG_MAX, LLONG_MAX)) {
                return LLONG_MIN;
        }

        return ret;
}

/**
 * Converts units in format <val>[.<val>][kMG] to floating point representation.
 *
 * @param    str            string to be parsed, suffix following SI suffix is ignored (as in 1ms or 100MB)
 * @param    case_sensitive should 'm' be considered as mega
 * @returns                 positive floating point representation of the string
 * @returns                 NAN if error
 */
double unit_evaluate_dbl(const char *str, bool case_sensitive) {
        char *end_ptr;
        char unit_prefix;
        errno = 0;
        double ret = strtod(str, &end_ptr);
        if (errno != 0) {
                perror("strtod");
                return NAN;
        }
        if (end_ptr == str) {
                log_msg(LOG_LEVEL_ERROR, "'%s' is not a number\n", str);
                return NAN;
        }
        unit_prefix = case_sensitive ? *end_ptr : toupper(*end_ptr);
        switch(unit_prefix) {
                case 'n':
                case 'N':
                        ret /= 1000'000'000;
                        break;
                case 'u':
                case 'U':
                        ret /= 1000'000;
                        break;
                case 'm':
                        ret /= 1000;
                        break;
                case 'k':
                case 'K':
                        ret *= 1000;
                        break;
                case 'M':
                        ret *= 1000'000LL;
                        break;
                case 'g':
                case 'G':
                        ret *= 1000'000'000LL;
                        break;
        }

        return ret;
}

/**
 * Formats number in format "ABCD.E [S]" where 'S' is an SI unit prefix.
 */
const char *format_in_si_units(unsigned long long int val) {
    const char *si_prefixes[] = { "", "k", "M", "G", "T" };
    int prefix_idx = 0;
    int reminder = 0;
    while (val > 10000) {
        reminder = val % 1000;
        val /= 1000;
        prefix_idx += 1;
        if (prefix_idx == sizeof si_prefixes / sizeof si_prefixes[0] - 1) {
            break;
        }
    }
    thread_local char buf[128];
    snprintf(buf, sizeof buf, "%lld.%d %s", val, reminder / 100, si_prefixes[prefix_idx]);
    return buf;
}


bool is_wine() {
#ifdef WIN32
        HMODULE hntdll = GetModuleHandle("ntdll.dll");
        if(!hntdll) {
                return false;
        }

        return GetProcAddress(hntdll, "wine_get_version");
#endif
        return false;
}

int get_framerate_n(double fps) {
        int denominator = get_framerate_d(fps);
        return round(fps * denominator / 100.0) * 100.0; // rounding to 100s to fix inaccuracy errors
                                                         // eg. 23.98 * 1001 = 24003.98
}

int get_framerate_d(double fps) {
        fps = fps - 0.00001; // we want to round halves down -> base for 10.5 could be 10 rather than 11
        int fps_rounded_x1000 = round(fps) * 1000;
        if (fabs(fps * 1001 - fps_rounded_x1000) <
                        fabs(fps * 1000 - fps_rounded_x1000)
                        && fps * 1000 < fps_rounded_x1000) {
                return 1001;
        } else {
                return 1000;
        }
}

const char *ug_strerror(int errnum)
{
        static thread_local char strerror_buf[STRERROR_BUF_LEN];
        const char *errstring = strerror_buf;
#ifdef _WIN32
        strerror_s(strerror_buf, sizeof strerror_buf, errnum); // C11 Annex K (bounds-checking interfaces)
#elif ! defined _POSIX_C_SOURCE || (_POSIX_C_SOURCE >= 200112L && !  _GNU_SOURCE)
        strerror_r(errnum, strerror_buf, sizeof strerror_buf); // XSI version
#else // GNU strerror_r version
        errstring = strerror_r(errnum, strerror_buf, sizeof strerror_buf);
#endif

        return errstring;
}

void ug_perror(const char *s) {
        log_msg(LOG_LEVEL_ERROR, "%s: %s\n", s, ug_strerror(errno));
}

/// @retval number of usable CPU cores or 1 if unknown
int get_cpu_core_count(void)
{
#ifdef _WIN32
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        return MIN(sysinfo.dwNumberOfProcessors, INT_MAX);
#else
        long numCPU = sysconf(_SC_NPROCESSORS_ONLN);
        if (numCPU == -1) {
                perror("sysconf(_SC_NPROCESSORS_ONLN)");
                return 1;
        }
        return MIN(numCPU, INT_MAX);
#endif
}

/**
 * Prints module usage in unified format.
 *
 * @param module_name       module name _including_ command-line parameter (eg. "-t syphon")
 * @param options           accepted options (may be NULL if no options are accepted)
 * @param options_full      options that are shown only if fullhelp is requested (NULL if none)
 * @param printf_full_help  whether to print options_full
 */
void print_module_usage(const char *module_name, const struct key_val *options, const struct key_val *options_full, bool print_full_help) {
        UNUSED(options_full), UNUSED(print_full_help);
        struct key_val nullopts[] = { { nullptr, nullptr } };
        if (options == nullptr) {
                options = nullopts;
        }
        std::ostringstream oss;
        oss << TERM_BOLD << TERM_FG_RED << module_name << TERM_FG_RESET;
        int max_key_len = 0;
        auto desc_key = [](const char *key) {
                if (const char *k = strchr(key, '='); k != nullptr) {
                        if (k[-1] != '[') {
                                key = k + 1;
                        }
                }
                return key;
        };
        bool first = true;
        auto add_opt = [&](const struct key_val *it) {
                if (first) {
                        oss << "[";
                        first = false;
                } else {
                        oss << "|";
                }
                max_key_len = MAX(max_key_len, (int) strlen(desc_key(it->key)));
                oss << ":" << it->key;
        };
        for (const auto *it = options; it->key != nullptr; ++it) {
                add_opt(it);
        }
        if (print_full_help && options_full) {
                for (const auto *it = options_full; it->key != nullptr; ++it) {
                        add_opt(it);
                }
        }
        if (!first) { // == has at least one option
                oss << "] | " << module_name << (options_full ? ":[full]help" : ":help") << TERM_RESET;
        }

        color_printf("Usage:\n\t%s\n", oss.str().c_str());
        if (first) { // no opts
                return;
        }
        color_printf("where\n");
        for (const auto *it = options; it->key != nullptr; ++it) {
                color_printf("\t" TBOLD("%*s") " - %s\n", max_key_len, desc_key(it->key), it->val);
        }
        if (print_full_help && options_full) {
                for (const auto *it = options_full; it->key != nullptr; ++it) {
                        color_printf("\t" TBOLD("%*s") " - %s\n", max_key_len, desc_key(it->key), it->val);
                }
        }
}

uint32_t parse_uint32(const char *value_str) noexcept(false)
{
        size_t pos = 0;
        long long val = stoll(value_str, &pos);
        if (val < 0) {
                throw out_of_range("negative number");
        }
        if (val > UINT32_MAX) {
                throw out_of_range("higher value than range of uint32");
        }
        if (pos == 0) {
                throw invalid_argument("no conversion was performed");
        }
        return val;
}
