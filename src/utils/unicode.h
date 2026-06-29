// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#ifndef UTILS_UNICODE_H_21A466F2_6B6F_4EF2_83CE_AE364FEFDEAE
#define UTILS_UNICODE_H_21A466F2_6B6F_4EF2_83CE_AE364FEFDEAE

#ifdef __cplusplus
#include <cstddef> // for size_t
#else
#include <stddef.h>  // for size_t
#endif

#include "compat/c23.h" // IWYU pragma: keep for bool

#ifdef __cplusplus
extern "C" {
#endif

#define U8_REGISTERED_SIGN  u8"\xC2\xAE"
#define U8_DEGREE_SIGN      u8"\xC2\xB0"
#define U8_SUPERSCRIPT_ONE  u8"\xC2\xB9"
#define U8_LARGE_RED_CIRCLE u8"\xF0\x9F\x94\xB4"

void        u8_out_init(bool is_win_utf8_terminal);
const char *u8s_to_mbs_buf(const unsigned char *u8_str, size_t buflen,
                           char *out_fallback);
// convenience macro casting char8_t ptr to (const unsigned char *) + len from
// buf
#define u8s_to_mbs(u8_str, out_fallback)                                       \
        u8s_to_mbs_buf((const unsigned char *) (u8_str), sizeof(out_fallback), \
                       (out_fallback))

#ifdef __cplusplus
}
#endif

#endif // defined UTILS_UNICODE_H_21A466F2_6B6F_4EF2_83CE_AE364FEFDEAE
