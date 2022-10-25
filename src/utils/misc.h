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

bool is_wine(void);
long long unit_evaluate(const char *str);
double unit_evaluate_dbl(const char *str, bool case_sensitive);
const char *format_in_si_units(unsigned long long int val);
int get_framerate_n(double framerate);
int get_framerate_d(double framerate);

const char *ug_strerror(int errnum);
void ug_perror(const char *s);
int get_cpu_core_count(void);

struct key_val {
        const char *key;
        const char *val;
};
void print_module_usage(const char *module_name, const struct key_val *options, const struct key_val *options_full, bool print_full_help);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
uint32_t parse_uint32(const char *value_str) noexcept(false);

#include <optional>

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

