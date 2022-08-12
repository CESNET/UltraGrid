/**
 * @file   utils/misc.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2022 CESNET z.s.p.o.
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

#ifndef UTILS_MISC_H_
#define UTILS_MISC_H_

#ifdef __cplusplus
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

int clampi(long long val, int lo, int hi);

bool is_prefix_of(const char *haystack, const char *needle);
bool is_wine(void);
long long unit_evaluate(const char *str);
const char *format_in_si_units(unsigned long long int val);
double unit_evaluate_dbl(const char *str);
int get_framerate_n(double framerate);
int get_framerate_d(double framerate);
#define DELDEL "\177\177"
#define ESCAPED_COLON "\\:"
void replace_all(char *in, const char *from, const char *to);

int urlencode_html5_eval(int c);
int urlencode_rfc3986_eval(int c);
size_t urlencode(char *out, size_t max_len, const char *in, int (*eval_pass)(int c), bool space_plus_replace);
size_t urldecode(char *out, size_t max_len, const char *in);

const char *ug_strerror(int errnum);
int get_cpu_core_count(void);

unsigned char *base64_decode(const char *in, unsigned int *length);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <optional>
#include <string_view>

/**
 * @brief Tokenizer for string_view
 *
 * Useful for non-destructive tokenization of strings. Skips empty tokens.
 * str is modified to view the not yet processed remainder.
 *
 * Typical usage (prints lines "Hello", "World", "!"):
 * ~~~~~~~~~~~~~~~~~~~{.cpp}
 * std::string_view sv = ":::Hello:World::!::";
 * while(!sv.empty()){
 *     cout << tokenize(sv, ':') << "\n";
 * }
 * ~~~~~~~~~~~~~~~~~~~
 *
 * The 'quot' param allows for a basic quotation based deilimiter escape,
 * however the whole token must be escaped.
 * i.e. sync:"opts=a=1:b=2":fs is fine, but sync:opts="a=1:b=2":fs is not.
 */
std::string_view tokenize(std::string_view& str, char delim, char quot = '\0');

template<typename T> struct ref_count_init_once {
        std::optional<T> operator()(T (*init)(void), int &i) {
                if (i++ == 0) {
                        return std::optional<T>(init());
                }
                return std::nullopt;
        }
};

struct ref_count_terminate_last {
        void operator()(void (*terminate)(void), int &i) {
                if (--i == 0) {
                        terminate();
                }
        }
};
#endif //__cplusplus

#endif// UTILS_MISC_H_

