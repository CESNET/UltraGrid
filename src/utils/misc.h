/**
 * @file   utils/misc.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2023 CESNET z.s.p.o.
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
#include <cstdint>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif


bool is_wine(void);
long long   unit_evaluate(const char *str, const char **endptr);
double      unit_evaluate_dbl(const char *str, bool case_sensitive,
                              const char **endptr);
const char *format_in_si_units(unsigned long long int val);
int get_framerate_n(double framerate);
int get_framerate_d(double framerate);

const char *ug_strerror(int errnum);
void ug_perror(const char *s);
int get_cpu_core_count(void);
bool is_arm_mac(void);

struct key_val {
        const char *key;
        const char *val;
};
void print_module_usage(const char *module_name, const struct key_val *options, const struct key_val *options_full, bool print_full_help);

bool invalid_arg_is_numeric(const char *what);

const char *get_stat_color(double ratio);

enum { FORMAT_NUM_MAX_SZ = 27, /*20 dec num + 6 delim + '\0' */ };
char *format_number_with_delim(size_t num, char *buf, size_t buflen);
#if !defined __cplusplus
#define fmt_number_with_delim(num) \
        format_number_with_delim(num, (char[FORMAT_NUM_MAX_SZ]) { 0 }, \
                                 FORMAT_NUM_MAX_SZ)
#endif

int parse_number(const char *str, int min, int base);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
uint32_t parse_uint32(const char *value_str) noexcept(false);

#include <map>
template<typename key, typename T>
inline T get_map_val_or_default(std::map<key, T> const& map, key const& k, T const& def) {
        if (auto && it = map.find(k); it != map.end()) {
                return it->second;
        }
        return def;
}

/* Like std::out_ptr from C++23 */
template<class Smart, class Pointer = typename Smart::pointer>
class out_ptr{
public:
        out_ptr(Smart& out) noexcept: smart_ptr(out) {  }
        ~out_ptr(){
                smart_ptr.reset(ptr);
        }

        out_ptr(out_ptr&&) = delete;
        out_ptr& operator=(out_ptr&&) = delete;

        operator Pointer*() noexcept { return &ptr; }
        operator void**() noexcept { return &ptr; }
private:
        Smart& smart_ptr;
        Pointer ptr = nullptr;
};
#endif //__cplusplus

#endif// UTILS_MISC_H_

