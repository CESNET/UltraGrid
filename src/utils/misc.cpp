/**
 * @file   utils/misc.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2024 CESNET
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

#include "utils/misc.h"

#ifdef _WIN32
#include <windows.h>          // for GetModuleHandle, GetProcAddress
#else
#include <unistd.h>           // for sysconf, _SC_NPROCESSORS_ONLN
#endif
#include <cassert>
#include <cctype>             // for toupper
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>             // for perror, snprintf
#include <cstdlib>            // for strtod
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>

#include "compat/strings.h" // sterror_s
#include "debug.h"
#include "utils/macros.h"
#include "utils/color_out.h"

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#endif

using std::invalid_argument;
using std::out_of_range;
using std::stoll;

/**
 * Converts units in format <val>[.<val>][kMG] to integral representation.
 *
 * @param   str    string to be parsed
 * @param   endptr if not NULL, point to suffix after parse
 * @returns     positive integral representation of the string
 * @returns     LLONG_MIN if error
 */
long long
unit_evaluate(const char *str, const char **endptr)
{
        double ret = unit_evaluate_dbl(str, false, endptr);

        if (ret == NAN || ret >= nexttoward((double) LLONG_MAX, LLONG_MAX)) {
                return LLONG_MIN;
        }

        return ret;
}

/**
 * Converts units in format <val>[.<val>][kMG] to floating point representation.
 *
 * @param    str            string to be parsed, suffix following SI suffix is ignored (as in 1ms or 100MB)
 * @param    case_sensitive if true 'm' will be considered as milli, otherwise mega
 * @param    endptr         if not NULL, point to suffix after parse
 * @returns                 positive floating point representation of the string
 * @returns                 NAN if error
 */
double
unit_evaluate_dbl(const char *str, bool case_sensitive, const char **endptr)
{
        if (endptr != nullptr) {
                *endptr = str;
        }

        char *endptr_tmp = nullptr;
        errno = 0;
        double ret = strtod(str, &endptr_tmp);
        if (errno != 0) {
                perror("strtod");
                return NAN;
        }
        if (endptr_tmp == str) {
                log_msg(LOG_LEVEL_ERROR, "'%s' is not a number\n", str);
                return NAN;
        }
        char unit_prefix = case_sensitive ? *endptr_tmp : toupper(*endptr_tmp);
        endptr_tmp += 1;
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
                default:
                        endptr_tmp -= 1;
        }

        if (endptr != nullptr) {
                *endptr = endptr_tmp;
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
#ifdef _WIN32
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

/// @note consider using rather strerror_s() directly
const char *ug_strerror(int errnum)
{
        enum {
                STRERROR_BUF_LEN = 1024,
        };
        static thread_local char strerror_buf[STRERROR_BUF_LEN];
        strerror_s(strerror_buf, sizeof strerror_buf, errnum); // C11 Annex K (bounds-checking interfaces)
        return strerror_buf;
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
                // print only <val> for key=<val> if right side is "simple" (mandatory and without further opts like "key=<val>:opt=<o>")
                if (const char *k = strchr(key, '='); k != nullptr && strpbrk(key, "[:") == nullptr) {
                        key = k + 1;
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

/**
 * Checks whether host machine is an ARM Mac (native or running x86_64 code with Rosetta2)
*/
bool is_arm_mac() {
#ifdef __APPLE__
        char machine[256];
        size_t len = sizeof machine;
        if (sysctlbyname("hw.machine", &machine, &len, NULL, 0) != 0) {
                perror("sysctlbyname hw.machine");
                return false;
        }
        int proc_translated = -1; // Rosetta2 - 0 off, 1 on; sysctl ENOENT - x86_64 mac
        len = sizeof proc_translated;
        if (int ret = sysctlbyname("sysctl.proc_translated", &proc_translated, &len, NULL, 0); ret != 0 && errno != ENOENT) {
                perror("sysctlbyname sysctl.proc_translated");
        }
        return strstr(machine, "arm") != NULL || proc_translated == 1;
#else
        return false;
#endif
}

/**
 * @param what  value returned by std::logic_error::what()
 * @returns if given std::invalid_argument message belongs to a stoi/stof
 * converison
 */

bool invalid_arg_is_numeric(const char *what) {
        if (what == nullptr) {
                return false;
        }
        return strncmp(what, "stoi", 4) == 0 || strncmp(what, "stod", 4) == 0;
}

/**
 * get warning color for status printout
 *
 * If ratio is <= 0.98 or >= 1.02, return "warning" color (different for 2% and
 * 5% outside the range). Note that the caller must also output TERM_RESET or
 * TERM_FG_RESET to reset the terminal FG color.
 *
 * @returns the color (or "")
 */
const char *
get_stat_color(double ratio)
{
        const double diff = fabs(1 - ratio);
        if (diff < 0.02) {
                return "";
        }
        return diff < 0.05 ? T256_FG_SYM(T_ARCTIC_LIME)
                           : T256_FG_SYM(T_SADDLE_BROWN);
}

char *
format_number_with_delim(size_t num, char *buf, size_t buflen)
{
        assert(buflen >= 1);
        buf[buflen - 1] = '\0';
        char *ptr       = buf + buflen - 1;
        int   grp_count = 0;
        do {
                if (ptr == buf || (grp_count == 3 && ptr == buf + 1)) {
                        snprintf(buf, buflen, "%s", "ERR");
                        return buf;
                }
                if (grp_count++ == 3) {
                        grp_count = 1;
                        *--ptr    = ',';
                }
                *--ptr = (char) ('0' + (num % 10));
                num /= 10;
        } while (num != 0);

        return ptr;
}

/// @returns INT_MIN on error, otherwise converted number
int
parse_number(const char *str, int min, int base)
{
        assert(min != INT_MIN);
        if (base == 10 && str[0] == '0' && str[1] != '\0') {
                log_msg(LOG_LEVEL_WARNING,
                        "Input number %s will be interpreted in base 10, not "
                        "as octal/hexadecimal!\n",
                        str);
        }
        char      *endptr = nullptr;
        const long ret    = strtol(str, &endptr, base);
        if (*endptr != '\0') {
                log_msg(LOG_LEVEL_ERROR,
                        "Input number %s contains non-numeric symbols!\n", str);
                return INT_MIN;
        }
        if (endptr == str) {
                log_msg(LOG_LEVEL_ERROR,
                        "Empty string passed where expected a number!\n");
                return INT_MIN;
        }
        if (ret < min) {
                log_msg(LOG_LEVEL_ERROR,
                        "Passed number %s where number >= %d expected!\n", str,
                        min);
                return INT_MIN;
        }
        if (ret > INT_MAX) {
                log_msg(LOG_LEVEL_ERROR, "The number %s would overflow int!\n",
                        str);
                return INT_MIN;
        }
        return (int) ret;
}
