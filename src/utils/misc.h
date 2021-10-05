/**
 * @file   utils/misc.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2021 CESNET z.s.p.o.
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

#define STRINGIFY(A) #A
#define TOSTRING(A) STRINGIFY(A) // https://stackoverflow.com/questions/240353/convert-a-preprocessor-token-to-a-string
#define IF_NOT_NULL_ELSE(cond, alt_val) (cond) ? (cond) : (alt_val)

#ifdef __cplusplus
extern "C" {
#endif

int clampi(long long val, int lo, int hi);

bool is_wine(void);
long long unit_evaluate(const char *str);
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

/**
 * @brief Creates FourCC word
 *
 * The main idea of FourCC is that it can be anytime read by human (by hexa editor, gdb, tcpdump).
 * Therefore, this is stored as a big endian even on little-endian architectures - first byte
 * of FourCC is in the memory on the lowest address.
 */
#ifdef WORDS_BIGENDIAN
#define to_fourcc(a,b,c,d)     (((uint32_t)(d)) | ((uint32_t)(c)<<8U) | ((uint32_t)(b)<<16U) | ((uint32_t)(a)<<24U))
#else
#define to_fourcc(a,b,c,d)     (((uint32_t)(a)) | ((uint32_t)(b)<<8U) | ((uint32_t)(c)<<16U) | ((uint32_t)(d)<<24U))
#endif

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
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
 */
std::string_view tokenize(std::string_view& str, char delim);
#endif //__cplusplus

#endif// UTILS_MISC_H_

