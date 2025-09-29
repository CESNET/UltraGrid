/**
 * @file   utils/text.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2025 CESNET, zájmové sdružení právnických osob
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

#ifndef UTILS_TEXT_H_AFEA0012_0A4B_4DC5_95FC_4B070B9D79CD
#define UTILS_TEXT_H_AFEA0012_0A4B_4DC5_95FC_4B070B9D79CD

#ifdef __cplusplus
#include <cstddef>    // for size_t
#include <cstdint>    // for uint32_t
#else
#include <stdbool.h>  // for bool
#include <stddef.h>   // for size_t
#include <stdint.h>   // for uint32_t
#endif

#ifdef __cplusplus
extern "C" {
#endif

unsigned char *base64_decode(const char *in, unsigned int *length);
char *wrap_paragraph(char *text);
int urlencode_html5_eval(int c);
int urlencode_rfc3986_eval(int c);
size_t urlencode(char *out, size_t max_len, const char *in, int (*eval_pass)(int c), bool space_plus_replace);
size_t urldecode(char *out, size_t max_len, const char *in);

bool draw_line(char *buf, int pitch, const char *text, uint32_t color, bool solid);
bool draw_line_scaled(char *buf, int pitch, const char *text, uint32_t fg,
                      uint32_t bg, unsigned scale);

const char *const *get_font_candidates(void);

#ifdef __cplusplus
}
#endif

#endif // defined UTILS_TEXT_H_AFEA0012_0A4B_4DC5_95FC_4B070B9D79CD

