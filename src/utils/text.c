/**
 * @file   utils/text.c
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

#include "utils/text.h"

#include <assert.h>             // for assert
#include <ctype.h>              // for isalnum
#include <limits.h>             // for INT_MAX, PATH_MAX
#include <stdio.h>              // for fclose, fwrite, rewind, sprintf, sscanf
#include <stdlib.h>             // for getenv, malloc, free, realloc
#include <string.h>             // for memcpy, strlen, strpbrk
#include <unistd.h>             // for unlink

#include "debug.h"
#include "utils/bitmap_font.h"
#include "utils/color_out.h" // prune_ansi_sequences_inplace_cstr
#include "utils/fs.h"
#include "utils/macros.h"    // for snprintf_ch
#include "utils/pam.h"

int urlencode_html5_eval(int c)
{
        return isalnum(c) || c == '*' || c == '-' || c == '.' || c == '_';
}

int urlencode_rfc3986_eval(int c)
{
        return isalnum(c) || c == '~' || c == '-' || c == '.' || c == '_';
}

/**
 * Replaces all occurrences where eval() evaluates to true with %-encoding
 * @param in        input
 * @param out       output array
 * @param max_len   maximal length to be written (including terminating NUL)
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
 * @param max_len maximal length to be written (including terminating NUL)
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
char *
wrap_paragraph(char *text)
{
        char *pos = text;
        char *last_space = NULL;

        char *next = NULL;
        int line_len = 0;
        while ((next = strpbrk(pos, " \n")) != NULL) {
                int len_raw = next - pos;
                char str_in[len_raw + 1];
                memcpy(str_in, pos, len_raw);
                str_in[len_raw] = '\0';
                int len_net = strlen(prune_ansi_sequences_inplace_cstr(str_in));
                if (line_len + len_net > 80) {
                        if (line_len == 0) { // |word|>80 starting a line
                                *next = '\n';
                                pos = next + 1;
                        } else {
                                *last_space = '\n';
                                pos = last_space + 1;
                        }
                        line_len = 0;
                        continue;
                }
                if (next[0] == '\n') {
                        pos = next + 1;
                        line_len = 0;
                        continue;
                }
                last_space = next;
                pos += len_raw + 1;
                line_len += len_net + 1;
        }
        return text;
}


static unsigned char font_data[(FONT_W * FONT_COUNT + 7) / 8 * FONT_H];
static bool          font_data_initialized = false;

// since the data are already in memory, it would be also possible to use
// pointer directly to font array (skipping or deleting PBM header)
static bool draw_line_init(unsigned char *out) {
        const char *filename = NULL;
        FILE *f = get_temp_file(&filename);
        if (!f) {
                log_msg(LOG_LEVEL_ERROR, "Cannot get temporary file!\n");
                return false;
        }
        fwrite(font, sizeof font, 1, f);
        rewind(f);
        struct pam_metadata info;
        unsigned char *font_data = NULL;

        bool ret = pam_read(filename, &info, &font_data, malloc);
        fclose(f);
        unlink(filename);
        if (!ret) {
                log_msg(LOG_LEVEL_ERROR, "Cannot read PAM!\n");
                return false;
        }
        assert(info.width == FONT_W * FONT_COUNT && info.height == FONT_H && info.maxval == 1 && info.bitmap_pbm);

        memcpy(out, font_data, (info.width + 7) / 8 * info.height);
        free(font_data);
        font_data_initialized = true;
        return true;
}

/// draw the letter as RGBA pixmap
static void
draw_letter(char c, uint32_t fg, uint32_t bg, unsigned char *buf,
            const unsigned char *font_data, int pitch)
{
        if (c < ' ' || c > '~') {
                c = '?';
        }
        c -= ' ';
        const unsigned char fg0 = fg & 0xFFU;
        const unsigned char fg1 = (fg >> 8U) & 0xFFU;
        const unsigned char fg2 = (fg >> 16U) & 0xFFU;
        const unsigned char fg3 = (fg >> 24U) & 0xFFU;
        const unsigned char bg0 = bg & 0xFFU;
        const unsigned char bg1 = (bg >> 8U) & 0xFFU;
        const unsigned char bg2 = (bg >> 16U) & 0xFFU;
        const unsigned char bg3 = (bg >> 24U) & 0xFFU;
        for (int j = 0; j < FONT_H; ++j) {
                for (int i = 0; i < FONT_W; ++i) {
                        int pos_x  = (FONT_W * c + i) / 8;
                        int mask   = 1 << (FONT_W - ((FONT_W * c + i) % 8));
                        int offset = (FONT_W * FONT_COUNT + 7) / 8 * j;
                        if (font_data[offset + pos_x] & mask) {
                                // clang-format off
                                buf[(j * pitch) + (4 * i)]     = fg0;
                                buf[(j * pitch) + (4 * i) + 1] = fg1;
                                buf[(j * pitch) + (4 * i) + 2] = fg2;
                                buf[(j * pitch) + (4 * i) + 3] = fg3;
                                // clang-format on
                        } else if (bg != 0) {
                                buf[(j * pitch) + (4 * i)]     = bg0;
                                buf[(j * pitch) + (4 * i) + 1] = bg1;
                                buf[(j * pitch) + (4 * i) + 2] = bg2;
                                buf[(j * pitch) + (4 * i) + 3] = bg3;
                        }
                }
                if (bg) { // fill space between characters
                        buf[(j * pitch) + (4 * (FONT_W_SPACE - 1))]     = bg0;
                        buf[(j * pitch) + (4 * (FONT_W_SPACE - 1)) + 1] = bg1;
                        buf[(j * pitch) + (4 * (FONT_W_SPACE - 1)) + 2] = bg2;
                        buf[(j * pitch) + (4 * (FONT_W_SPACE - 1)) + 3] = bg3;
                }
        }
}

/**
 * draws a line with built-in bitmap 12x7 bitmap font separated by 1 px space, RGBA
 */
bool draw_line(char *buf, int pitch, const char *text, uint32_t color, bool solid) {
        if (!font_data_initialized && !draw_line_init(font_data)) {
                return false;
        }
        int idx = 0;
        const uint32_t bg = solid ? 0xFF000000U : 0;
        while (*text) {
                char c = *text++;
                draw_letter(c, color, bg,
                            (unsigned char *) buf +
                                (size_t) (4 * idx * FONT_W_SPACE),
                            font_data, pitch);
                if ((++idx + 1) * FONT_W_SPACE * 4 > pitch) {
                        return true;
                }
        }

        return true;
}

/**
 * similar to draw_line() but integer upscaling can be applied
 *
 * transparency not supported/implemented
 *
 * @param bg  must not be 0, use 0xFF<<24 (alpha set to 0xFF)
 */
bool
draw_line_scaled(char *buf, int pitch, const char *text, uint32_t fg,
                 uint32_t bg, unsigned scale)
{
        (void) buf;
        assert(scale > 0);
        assert(bg != 0); // draw letter would than assume transparency which
                         // doesn't work with the _scaled version
        if (!font_data_initialized && !draw_line_init(font_data)) {
                return false;
        }
        int idx = 0;
        while (*text) {
                char c = *text++;
                unsigned char letter[4 * FONT_W_SPACE * FONT_H];
                draw_letter(c, fg, bg, letter, font_data, 4 * FONT_W_SPACE);
                // resize
                for (unsigned y = 0; y < FONT_H; ++y) {
                        for (unsigned x = 0; x < FONT_W_SPACE; ++x) {
                                for (unsigned sv = 0; sv < scale; ++sv) {
                                        for (unsigned sh = 0; sh < scale; ++sh) {
                                                memcpy(buf + ((scale * y + sv) * pitch + (((idx * scale * FONT_W_SPACE)
                                                           + (x * scale + sh)) * 4)), letter + (4 * (y * FONT_W_SPACE + x)), 4);
                                        }
                                }
                        }
                }
                if ((++idx + 1) * FONT_W_SPACE * 4 * scale > (unsigned) pitch) {
                        return true;
                }
        }

        return true;
}

/// @returns null-terminated list of TTF font candidates
const char *const *
get_font_candidates()
{
#ifdef _WIN32
#define DEFAULT_FONT_DIR "C:\\windows\\fonts"
        static const char *const font_candidates[] = {
                "cour.ttf",
        };
#elif defined __APPLE__
#define DEFAULT_FONT_DIR "/System/Library/Fonts"
        static const char *const font_candidates[] = {
                "Monaco.ttf",
                "Monaco.dfont",
                "Geneva.ttf",
                "Keyboard.ttf",
        };
#else
#define DEFAULT_FONT_DIR "/usr/share/fonts"
        static const char *const font_candidates[] = {
                "DejaVuSansMono.ttf", // bundled in AppImage
                "truetype/freefont/FreeMonoBold.ttf",
                "truetype/dejavu/DejaVuSansMono.ttf", // Ubuntu
                "TTF/DejaVuSansMono.ttf",
                "liberation/LiberationMono-Regular.ttf",      // Arch
                "liberation-mono/LiberationMono-Regular.ttf", // Fedora
        };
#endif

        static const char *const *ret = NULL;
        if (ret != NULL) {
                return ret;
        }

        static char tmp[sizeof font_candidates / sizeof font_candidates[0]][PATH_MAX] = { 0 };
        static char *ptrs[(sizeof tmp / sizeof tmp[0]) + 1] = { 0 };

        const char *font_dir = IF_NOT_NULL_ELSE(getenv("UG_FONT_DIR"), DEFAULT_FONT_DIR);
        for (unsigned i = 0; i < sizeof font_candidates / sizeof font_candidates[0]; ++i) {
                snprintf_ch(tmp[i], "%s/%s", font_dir, font_candidates[i]);
                ptrs[i] = tmp[i];
        }

        ret = (const char *const *) ptrs;
        return ret;
}
