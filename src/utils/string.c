/**
 * @file   utils/string.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2023 CESNET, z. s. p. o.
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

#ifndef __APPLE__
#include <signal.h>
#endif
#include <string.h>

#include "debug.h"
#include "utils/string.h"

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

/**
 * Checks if needle is prefix in haystack, case _insensitive_.
 */
bool is_prefix_of(const char *haystack, const char *needle) {
        return strncasecmp(haystack, needle, strlen(needle)) == 0;
}

char *strrpbrk(char *s, const char *accept) {
        char *end = s + strlen(s) - 1;
        while (end >= s) {
                const char *accept_it = accept;
                while (*accept_it != '\0') {
                        if (*accept_it++ == *end) {
                                return end;
                        }
                }
                end--;
        }
        return NULL;
}

bool ends_with(const char *haystick, const char *needle) {
        if (strlen(haystick) < strlen(needle)) {
                return false;
        }
        return strcmp(haystick + strlen(haystick) - strlen(needle), needle) == 0;
}

/**
 * Copies string at src to pointer *ptr and increases the pointner.
 * @todo
 * Functions defined in string.h should signal safe in POSIX.1-2008 - check if in glibc, if so, use strcat.
 * @note
 * Strings are not NULL-terminated.
 */
void strappend(char **ptr, const char *ptr_end, const char *src) {
        while (*src != '\0') {
                if (*ptr == ptr_end) {
                        return;
                }
                *(*ptr)++ = *src++;
        }
}

void write_all(size_t len, const char *msg) {
        const char *ptr = msg;
        do {
                ssize_t written = write(STDERR_FILENO, ptr, len);
                if (written < 0) {
                        break;
                }
                len -= written;
                ptr += written;
        } while (len > 0);
}

/**
 * Appends signal number description to ptr and moves ptr to end of the
 * appended string. The string is not NULL-terminated.
 *
 * This function is async-signal-safe.
 */
void
append_sig_desc(char **ptr, const char *ptr_end, int signum)
{
#ifdef _WIN32
        (void) ptr, (void) ptr_end, (void) signum;
#else
        strappend(ptr, ptr_end, " (");
#if __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 32)
        strappend(ptr, ptr_end, sigabbrev_np(signum));
        strappend(ptr, ptr_end, " - ");
        strappend(ptr, ptr_end, sigdescr_np(signum));
#else
        strappend(ptr, ptr_end, sys_siglist[signum]);
#endif
        strappend(ptr, ptr_end, ")");
#endif // !defined _WIN32
}
