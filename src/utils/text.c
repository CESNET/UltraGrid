/**
 * @file   utils/text.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET, z. s. p. o.
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
                        } else {
                                *last_space = '\n';
                        }
                        pos = next + 1;
                        line_len = 0;
                        continue;
                }
                last_space = next;
                line_len += len_net + 1;
                pos += len_raw + 1;
        }
        return text;
}

