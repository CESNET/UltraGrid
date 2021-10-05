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

#include <limits.h>
#include <math.h>

#include "debug.h"
#include "utils/misc.h"

#define STRERROR_BUF_LEN 1024

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
        char *end_ptr;
        char unit_prefix_u;
        errno = 0;
        double ret = strtod(str, &end_ptr);
        if (errno != 0) {
                perror("strtod");
                return LLONG_MIN;
        }
        unit_prefix_u = toupper(*end_ptr);
        switch(unit_prefix_u) {
                case 'G':
                        ret *= 1000;
                        /* fall through */
                case 'M':
                        ret *= 1000;
                        /* fall through */
                case 'K':
                        ret *= 1000;
                        break;
                case '\0':
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "Error: unknown unit suffix %c.\n", *end_ptr);
                        return LLONG_MIN;
        }

        if (ret >= nexttoward((double) LLONG_MAX, LLONG_MAX) || strlen(end_ptr) > 1) {
                return LLONG_MIN;
        } else {
                return ret;
        }
}

/**
 * Converts units in format <val>[.<val>][kMG] to floating point representation.
 *
 * @param   str string to be parsed
 * @returns     positive floating point representation of the string
 * @returns     NAN if error
 */
double unit_evaluate_dbl(const char *str) {
        char *end_ptr;
        char unit_prefix_u;
        errno = 0;
        double ret = strtod(str, &end_ptr);
        if (errno != 0) {
                perror("strtod");
                return NAN;
        }
        unit_prefix_u = toupper(*end_ptr);
        switch(unit_prefix_u) {
                case 'G':
                        ret *= 1000;
                        /* fall through */
                case 'M':
                        ret *= 1000;
                        /* fall through */
                case 'K':
                        ret *= 1000;
                        break;
                case '\0':
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "Error: unknown unit suffix %c.\n", *end_ptr);
                        return NAN;
        }

        return ret;
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

/**
 * @brief Replaces all occurencies of 'from' to 'to' in string 'in'
 *
 * Typical use case is to process escaped colon in arguments:
 * ~~~~~~~~~~~~~~~{.c}
 * // replace all '\:' with 2xDEL
 * replace_all(fmt, ESCAPED_COLON, DELDEL);
 * while ((item = strtok())) {
 *         char *item_dup = strdup(item);
 *         replace_all(item_dup, DELDEL, ":");
 *         free(item_dup);
 * }
 * ~~~~~~~~~~~~~~~
 *
 * @note
 * Replacing pattern must not be longer than the replaced one (because then
 * we need to extend the string)
 */
void replace_all(char *in, const char *from, const char *to) {
        assert(strlen(from) >= strlen(to) && "Longer dst pattern than src!");
        char *tmp = in;
        while ((tmp = strstr(tmp, from)) != NULL) {
                memcpy(tmp, to, strlen(to));
                if (strlen(to) < strlen(from)) { // move the rest
                        size_t len = strlen(tmp + strlen(from));
                        char *src = tmp + strlen(from);
                        char *dst = tmp + strlen(to);
                        memmove(dst, src, len);
                        dst[len] = '\0';
                }
                tmp += strlen(from);
        }
}

int urlencode_html5_eval(int c)
{
        return isalnum(c) || c == '*' || c == '-' || c == '.' || c == '_';
}

int urlencode_rfc3986_eval(int c)
{
        return isalnum(c) || c == '~' || c == '-' || c == '.' || c == '_';
}

/**
 * Replaces all occurences where eval() evaluates to true with %-encoding
 * @param in        input
 * @param out       output array
 * @param max_len   maximal lenght to be written (including terminating NUL)
 * @param eval_pass predictor if an input character should be kept (functions
 *                  from ctype.h may be used)
 * @param space_plus_replace replace spaces (' ') with ASCII plus sign -
 *                  should be true for HTML5 URL encoding, false for RFC 3986
 * @returns bytes written to out
 *
 * @note
 * Symbol ' ' is not treated specially (unlike in classic URL encoding which
 * translates it to '+'.
 * @todo
 * There may be a LUT as in https://rosettacode.org/wiki/URL_encoding#C
 */
size_t urlencode(char *out, size_t max_len, const char *in, int (*eval_pass)(int c),
                bool space_plus_replace)
{
        if (max_len == 0 || max_len >= INT_MAX) { // prevent overflow
                return 0;
        }
        size_t len = 0;
        while (*in && len < max_len - 1) {
                if (*in == ' ' && space_plus_replace) {
                        *out++ = '+';
                        in++;
                } else if (eval_pass(*in) != 0) {
                        *out++ = *in++;
                        len++;
                } else {
                        if ((int) len < (int) max_len - 3 - 1) {
                                int ret = sprintf(out, "%%%02X", *in++);
                                out += ret;
                                len += ret;
                        } else {
                                break;
                        }
                }
        }
        *out = '\0';
        len++;

        return len;
}

static inline int ishex(int x)
{
	return	(x >= '0' && x <= '9')	||
		(x >= 'a' && x <= 'f')	||
		(x >= 'A' && x <= 'F');
}

/**
 * URL decodes input string (replaces all "%XX" sequences with ASCII representation of 0xXX)
 * @param in      input
 * @param out     output array
 * @param max_len maximal lenght to be written (including terminating NUL)
 * @returns bytes written, 0 on error
 *
 * @note
 * Symbol '+' is not treated specially (unlike in classic URL decoding which
 * translates it to ' '.
 */
size_t urldecode(char *out, size_t max_len, const char *in)
{
        if (max_len == 0) { // avoid (uint) -1 cast
                return 0;
        }
        size_t len = 0;
        while (*in && len < max_len - 1) {
                if (*in == '+') {
                        *out++ = ' ';
                        in++;
                } else if (*in != '%') {
                        *out++ = *in++;
                } else {
                        in++; // skip '%'
                        if (!ishex(in[0]) || !ishex(in[1])) {
                                return 0;
                        }
                        unsigned int c = 0;
                        if (sscanf(in, "%2x", &c) != 1) {
                                return 0;
                        }
                        *out++ = c;
                        in += 2;
                }
                len++;
        }
        *out = '\0';
        len++;

        return len;
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

std::string_view tokenize(std::string_view& str, char delim){
        if(str.empty())
                return {};

        auto token_begin = str.begin();
        while(token_begin != str.end() && *token_begin == delim){
                token_begin++;
        }

        auto token_end = token_begin;
        while(token_end != str.end() && *token_end != delim){
                token_end++;
        }

        str = std::string_view(token_end, str.end() - token_end);

        return std::string_view(token_begin, token_end - token_begin);
}
