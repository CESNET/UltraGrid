/**
 * @file   utils/color_out.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Simple C wrapper around the rang header and utility macros for the
 * rang.hpp (C++ only)
 */
/*
 * Copyright (c) 2018-2019 CESNET, z. s. p. o.
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

#ifndef COLOR_OUT_H_
#define COLOR_OUT_H_

#ifdef __cplusplus
#include "rang.hpp"
#define BOLD(x) rang::style::bold << x << rang::style::reset
#define GREEN(x) rang::fg::green << x << rang::fg::reset
#define RED(x) rang::fg::red << x << rang::fg::reset
#include <cstdio>
#else
#include <stdbool.h>
#include <stdio.h>
#endif

#define COLOR_OUT_BOLD      1u
#define COLOR_OUT_DIM       2u
#define COLOR_OUT_ITALIC    3u
#define COLOR_OUT_UNDERLINE 5u
#define COLOR_OUT_BLINK     6u
#define COLOR_OUT_RBLINK    7u
#define COLOR_OUT_REVERSED  8u
#define COLOR_OUT_CONCEAL   9u
#define COLOR_OUT_CROSSED  10u

#define COLOR_OUT_FG_SHIFT 4u
#define COLOR_BITS 7
#define COLOR_OUT_BLACK   (1u<<4u)
#define COLOR_OUT_RED     (2u<<4u)
#define COLOR_OUT_GREEN   (3u<<4u)
#define COLOR_OUT_YELLOW  (4u<<4u)
#define COLOR_OUT_BLUE    (5u<<4u)
#define COLOR_OUT_MAGENTA (6u<<4u)
#define COLOR_OUT_CYAN    (7u<<4u)
#define COLOR_OUT_GRAY    (8u<<4u)
#define COLOR_OUT_BRIGHT_BLACK (61u<<4u)
#define COLOR_OUT_BRIGHT_RED     (62u<<4u)
#define COLOR_OUT_BRIGHT_GREEN   (63u<<4u)
#define COLOR_OUT_BRIGHT_YELLOW  (64u<<4u)
#define COLOR_OUT_BRIGHT_BLUE    (65u<<4u)
#define COLOR_OUT_BRIGHT_MAGENTA (66u<<4u)
#define COLOR_OUT_BRIGHT_CYAN    (67u<<4u)
#define COLOR_OUT_BRIGHT_GRAY    (68u<<4u)
#define COLOR_OUT_BG_SHIFT (COLOR_OUT_FG_SHIFT+COLOR_BITS)

#define TERM_RESET "\e[0m"
#define TERM_BOLD "\e[1m"
#define TERM_FG_RED "\e[31m"
#define TERM_FG_RESET "\e[39m"

#ifdef __cplusplus
extern "C" {
#endif

// old API
void color_out(uint32_t modificators, const char *format, ...) ATTRIBUTE(format (printf, 2, 3));

// new API
void color_output_init(void);
int color_printf(const char *format, ...) ATTRIBUTE(format (printf, 1, 2));
// utils
bool isMsysPty(int fd);

#ifdef __cplusplus
} // extern "C"
#endif


#endif // defined COLOR_OUT_H_

