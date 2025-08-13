/**
 * @file   utils/string.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2025 CESNET
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

#ifndef UTILS_STRING_H_09AB88E2_E93F_443B_BF01_EA8F6D90B643
#define UTILS_STRING_H_09AB88E2_E93F_443B_BF01_EA8F6D90B643

#ifdef __cplusplus
#include <cstddef>
#include <cstdint>    // for uintmax_t
extern "C" {
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>   // for uintmax_t
#endif

// functions documented at definition
#define DELDEL "\177\177"
#define ESCAPED_COLON "\\:"
bool ends_with(const char *haystick, const char *needle);
void replace_all(char *in, const char *from, const char *to);
bool is_prefix_of(const char *haystack, const char *needle);
/// same as strpbrk but finds in a reverse order (last occurrence returned)
char *strrpbrk(char *s, const char *accept);
void strappend(char **ptr, const char *ptr_end, const char *src);
void append_number(char **ptr, const char *ptr_end, uintmax_t num);
void append_sig_desc(char **ptr, const char *ptr_end, int signum);
void write_all(int fd, size_t len, const char *msg);
const char *pretty_print_fourcc(const void *fcc);

#ifdef __cplusplus
}
#endif

#endif // defined UTILS_STRING_H_09AB88E2_E93F_443B_BF01_EA8F6D90B643

