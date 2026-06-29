// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

/**
 * @file
 * helper function around wcsrtombs() to handle conversion failures
 * (fallback) or where is the MBS output may be unsupported (eg. in
 * Windows if neither in Terminal nor in MSYS2 window).
 */

#include "utils/unicode.h"

#include <locale.h> // for setlocale
#include <string.h> // for memcpy, strlen, strpbrk, strstr
#include <wchar.h> // for mbstate_t

#include "compat/c23.h" // IWYU pragma: keep
#include "debug.h"

#define MOD_NAME "[utils/unicode] "

#ifdef _WIN32
static bool win_utf8_terminal = false;
#endif

void
u8_out_init(bool is_win_utf8_terminal)
{
#ifdef _WIN32
        win_utf8_terminal = is_win_utf8_terminal;
        // this is necessary for wcsrtombs work with 2 wchar_t character
        // conversion (eg. W_LARGE_RED_CIRCLE)
        const char *lc_ctype = setlocale(LC_CTYPE, ".UTF8");
#else
        (void) is_win_utf8_terminal;
        const char *lc_ctype = setlocale(LC_CTYPE, "");
#endif
        if (lc_ctype == nullptr) {
                MSG(WARNING, "Cannot set locale.");
        } else {
                MSG(DEBUG, "LC_CTYPE set to: %s\n", lc_ctype);
        }
}

/**
 * Tries to convert wide string to locale-specific multibyte string. If
 * conversion fails out_fallback is returned untouched (should contain fallback
 * text).
 *
 * @param buflen  out_fallback buffer length long enough to hold the converted
                  string
 * @param[in,out] out_fallback NUL-terminated fallback string. If conversion
 *                succeeds, it is rewritten by the u8_str converted to MBS
 * @returns out_fallback with converted data if conversion suceeds
 * @returns out_fallback unchanged if wstr not convertible
 *
 * u8_out_init() must be called otherwise fallback is always ret
 */
const char *
wcs_to_mbs_buf(const wchar_t *wstr, size_t buflen, char *out_fallback)
{
#ifdef _WIN32
        if (!win_utf8_terminal) {
                return out_fallback;
        }
#endif

        mbstate_t ps = { 0 };

        // check convertibility first
        size_t ret = wcsrtombs(nullptr, &wstr, 0, &ps);
        if (ret == (size_t) -1) { // not convertible
                return out_fallback;
        }

        // actual conversion
        (void) wcsrtombs(out_fallback, &wstr, buflen, &ps);
        if (wstr != nullptr) {
                MSG(WARNING, "%s: wide string truncated.\n", __func__);
        }
        return out_fallback;
}
