/**
 * @file   utils/string.c
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

#include <assert.h>
#include <ctype.h>
#include <stdint.h>
#include <signal.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

#include "compat/strings.h"
#include "debug.h"
#include "utils/macros.h"    // for MIN
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
 * Typical use case:
 * ````
 * consexpr bool nul_terminate = true; // false if not needed
 * char buf[STR_LEN];
 * char *ptr = buf;
 * const char *const end = buf + sizeof buf - null_terminate;
 * strappend(&ptr, end, "this");
 * strappend(&ptr, end, " and that");
 * if (null_terminate) {
 *         *ptr = '\0';
 * }
 *
 * @todo
 * Functions defined in string.h should signal safe in POSIX.1-2008 - check if in glibc, if so, use strcat.
 * @note
 * Output string is not NULL-terminated but src must be!
 */
void strappend(char **ptr, const char *ptr_end, const char *src) {
        while (*src != '\0') {
                if (*ptr == ptr_end) {
                        return;
                }
                *(*ptr)++ = *src++;
        }
}

void write_all(int fd, size_t len, const char *msg) {
        const char *ptr = msg;
        do {
                ssize_t written = write(fd, ptr, len);
                if (written < 0) {
                        break;
                }
                len -= written;
                ptr += written;
        } while (len > 0);
}

void
append_number(char **ptr, const char *ptr_end, uintmax_t num)
{
        char num_buf[100];
        int  idx       = sizeof num_buf;
        num_buf[--idx] = '0' + num % 10; // write '0' if num=0
        while ((num /= 10) != 0) {
                num_buf[--idx] = '0' + num % 10;
        }
        const size_t buflen = ptr_end - *ptr;
        const size_t len = MIN(buflen, sizeof num_buf - idx);
        strncpy(*ptr, num_buf + idx, len);
        *ptr += len;
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
        strappend(ptr, ptr_end, " (");
#ifdef _WIN32
#define SIGERR(x) [x] = #x
        // from MinGW signal.h, unsure if other signals occur natively in MSW
        const char *signames[NSIG] = {
                SIGERR(SIGINT),   SIGERR(SIGILL),  SIGERR(SIGABRT_COMPAT),
                SIGERR(SIGFPE),   SIGERR(SIGSEGV), SIGERR(SIGTERM),
                SIGERR(SIGBREAK), SIGERR(SIGABRT), // SIGERR(SIGABRT2),
        };
#undef SIGERR
        if (signames[signum] == NULL) {
                strappend(ptr, ptr_end, "UNKNOWN");
        } else {
                strappend(ptr, ptr_end, signames[signum]);
        }
#elif __GLIBC__ > 2 || (__GLIBC__ == 2 && __GLIBC_MINOR__ >= 32)
        strappend(ptr, ptr_end, sigabbrev_np(signum));
        strappend(ptr, ptr_end, " - ");
        strappend(ptr, ptr_end, sigdescr_np(signum));
#elif defined(__APPLE__)
        strappend(ptr, ptr_end, sys_siglist[signum]);
#else
        // not async-signal-safe in general
        const char *strdesc = strsignal(signum);
        strappend(ptr, ptr_end, strdesc);
#endif
        strappend(ptr, ptr_end, ")");
}

/**
 * Returns string representation of input FourCC in a fashion FFmpeg does -
 * non-printable character value is printed as a number in '[]'.
 *
 * @returns NUL-terminated FourCC, which will be longer than 4B + '\0' if
 *           non-printable characters are present
 */

const char *
pretty_print_fourcc(const void *fcc)
{
        enum {
                CHAR_LEN = 5, // at worst [XXX]
                CHARS = sizeof(uint32_t),
                MAX_LEN = CHARS * CHAR_LEN + 1 /* '\0' */,
        };
        _Thread_local static char out[MAX_LEN];
        char *out_ptr = out;
        const unsigned char *fourcc = fcc;

        for (int i = 0; i < CHARS; ++i) {
                if (isprint(fourcc[i])) {
                        *out_ptr++ = (char) fourcc[i];
                } else {
                        const int written =
                            snprintf(out_ptr, CHAR_LEN + 1, "[%hhu]", fourcc[i]);
                        out_ptr+= written;
                }
        }
        *out_ptr = '\0';
        return out;
}

const char *
ug_strcasestr(const char *haystick, const char *needle)
{
        while (strlen(haystick) >= strlen(needle)) {
                const char *cur_haystick = haystick;
                const char *cur_needle   = needle;
                while (*cur_needle != '\0') {
                        if (tolower(*cur_haystick) != tolower(*cur_needle)) {
                                break;
                        }
                        cur_haystick += 1;
                        cur_needle += 1;
                }
                if (*cur_needle == '\0') {
                        return haystick;
                }

                haystick += 1;
        }
        return NULL;
}
