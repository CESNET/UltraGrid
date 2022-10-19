/**
 * @file   utils/text.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2022 CESNET, z. s. p. o.
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

#include <string.h>

#include "utils/color_out.h" // prune_ansi_sequences_inplace_cstr
#include "utils/text.h"

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
        assert(strlen(from) > 0 && "From pattern should be non-empty!");
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
                tmp += strlen(to);
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

/**
 * Checks if needle is prefix in haystack, case _insensitive_.
 */
bool is_prefix_of(const char *haystack, const char *needle) {
        return strncasecmp(haystack, needle, strlen(needle)) == 0;
}

/**
 * C-adapted version of https://stackoverflow.com/a/34571089
 *
 * As the output is a generic binary string, it is not NULL-terminated.
 *
 * Caller is obliged to free the returned string.
 */
unsigned char *base64_decode(const char *in, unsigned int *length) {
    unsigned int allocated = 128;
    unsigned char *out = (unsigned char *) malloc(allocated);
    *length = 0;

    int T[256];
    for (unsigned int i = 0; i < sizeof T / sizeof T[0]; i++) {
        T[i] = -1;
    }
    for (int i=0; i<64; i++) T[(int) "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"[i]] = i;

    int val=0, valb=-8;
    unsigned char c = 0;
    while ((c = *in++) != '\0') {
        if (T[c] == -1) break;
        val = (val << 6) + T[c];
        valb += 6;
        if (valb >= 0) {
            if (allocated == *length) {
                allocated *= 2;
                out = (unsigned char *) realloc(out, allocated);
                assert(out != NULL);
            }
            out[(*length)++] = (val>>valb)&0xFF;
            valb -= 8;
        }
    }
    return out;
}

/**
 * Indents paragraph (possibly with ANSI colors) to (currently only) the width
 * of 80. Inplace (just replaces spaces with newlines).
 * @returns     text
 */
char *indent_paragraph(char *text) {
        char *pos = text;
        char *last_space = NULL;
        int line_len = 0;

        char *next = NULL;
        while ((next = strpbrk(pos, " \n")) != NULL) {
                if (next[0] == '\n') {
                        line_len = 0;
                        pos = next + 1;
                        continue;
                }

                int len_raw = next - pos;
                char str_in[len_raw + 1];
                memcpy(str_in, pos, len_raw);
                str_in[len_raw] = '\0';
                int len_net = strlen(prune_ansi_sequences_inplace_cstr(str_in));
                if (line_len + len_net > 80) {
                        if (line_len == 0) { // |word|>80 starting a line
                                *next = '\n';
                                line_len = 0;
                        } else {
                                *last_space = '\n';
                                line_len = len_net;
                        }
                        pos = next + 1;
                        continue;
                }
                last_space = next;
                line_len += len_net + 1;
                pos += len_raw + 1;
        }
        return text;
}

