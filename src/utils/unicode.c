// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

/**
 * @file
 * @note
 * char32_t implementation with c32rtomb() could be more compatible since it
 * should be supported since C11/C++11 but its implementation is unfortuntately
 * missing from macOS (up to including 26) and the pass-through workaround won't
 * work with UTF-32 data.
 */

#include "utils/unicode.h"

#ifdef HAVE_CONFIG_H
#include "config.h" // for HAVE_C8RTOMB
#endif

#include <assert.h> // for assert
#include <limits.h> // for INT_MAX, PATH_MAX
#include <locale.h> // for setlocale
#include <string.h> // for memcpy, strlen, strpbrk
#ifdef HAVE_C8RTOMB
#include <uchar.h> // for c8rtomb
#include <wchar.h> // for mbstate_t
#endif

#include "compat/c23.h" // IWYU pragma: keep
#include "debug.h"

#define MOD_NAME "[utils/unicode] "

static bool utf8_terminal = false;
void
u8_out_init(bool is_win_utf8_terminal)
{
#ifdef _WIN32
        utf8_terminal = is_win_utf8_terminal;
#else
        const char *lc_ctype = setlocale(LC_CTYPE, "");
        if (lc_ctype == nullptr) {
                MSG(WARNING, "Cannot set locale.");
        } else {
                MSG(DEBUG, "LC_CTYPE set to: %s\n", lc_ctype);
                utf8_terminal = strstr(lc_ctype, ".UTF-8") != nullptr;
        }
        (void) is_win_utf8_terminal;
#endif
}

/**
 * Tries to convert utf-8 string to locale-specific multibyte string. If
 * c8rtomb not present and UTF-8 terminal detected, copies the UTF-8 string.
 * Otherwise, out_fallback is kept untouched (should contain fallback text).
 *
 * @param buflen  out_fallback buffer length long enough to hold the converted
                  string with some headroom (MB_LEN_MAX-1)
 * @param[in,out] out_fallback NUL-terminated fallback string. If conversion
 *                succeeds, it is rewritten by the u8_str converted to MBS
 * @returns out_fallback with converted data if we have c8rtomb
 * @returns u8_str if it is safe to pass-through
 * @returns out_fallback unchanged if u8_str not convertible and terminal not in
 UTF-8
 *
 * u8_to_mb_init() must be called otherwise fallback is always ret
 */
const char *
u8s_to_mbs_buf(const unsigned char *u8_str, size_t buflen, char *out_fallback)
{
#if defined HAVE_C8RTOMB && !defined _WIN32
        mbstate_t      ps     = { 0 };
        const char8_t *in_ptr = u8_str;
        // check convertibility first
        do {
                char   discard[MB_LEN_MAX];
                size_t ret = c8rtomb(discard, *in_ptr, &ps);
                if (ret == (size_t) -1) { // not convertible
                        return out_fallback;
                }
        } while (*in_ptr++ != '\0');
        // actual conversion
        in_ptr        = u8_str;
        char *out_ptr = out_fallback;
        do {
                if (buflen < MB_LEN_MAX || buflen == 1) {
                        MSG(WARNING, "utf-8 string truncated.\n");
                        assert(buflen >= 1);
                        *out_ptr = '\0';
                        break;
                }
                size_t ret = c8rtomb(out_ptr, *in_ptr, &ps);
                assert(ret != (size_t) -1);
                out_ptr += ret;
                buflen -= ret;
        } while (*in_ptr++ != '\0');
        return out_fallback;
#else
        (void) buflen;
        if (!utf8_terminal) {
                return out_fallback;
        }
        return (const char *) u8_str;
#endif
}
