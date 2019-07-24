/**
 * @file   utils/misc.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET z.s.p.o.
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

#include <limits.h>

#include "debug.h"
#include "utils/misc.h"

/**
 * Converts units in format <val>[.<val>][kMG] to integral representation.
 *
 * @param   str string to be parsed
 * @returns     positive integral representation of the string
 * @returns     -1 if error
 */
long long unit_evaluate(const char *str) {
        char *end_ptr;
        char unit_prefix_u;
        double ret = strtod(str, &end_ptr);
        unit_prefix_u = toupper(*end_ptr);
        switch(unit_prefix_u) {
                case 'G':
                        ret *= 1000;
                        /* fall through */
                case 'M':
                        ret *= 1000;
                        /* fall through */
                case 'K':
                        ret *= 1000;
                        break;
                case '\0':
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "Error: unknown unit suffix %c.\n", *end_ptr);
                        return -1;
        }

        if (ret < 0.0 || ret >= LLONG_MAX || strlen(end_ptr) > 1) {
                return -1;
        } else {
                return ret;
        }
}

/**
 * Converts units in format <val>[.<val>][kMG] to floating point representation.
 *
 * @param   str string to be parsed
 * @returns     positive floating point representation of the string
 * @returns     NAN if error
 */
double unit_evaluate_dbl(const char *str) {
        char *end_ptr;
        char unit_prefix_u;
        double ret = strtod(str, &end_ptr);
        unit_prefix_u = toupper(*end_ptr);
        switch(unit_prefix_u) {
                case 'G':
                        ret *= 1000;
                        /* fall through */
                case 'M':
                        ret *= 1000;
                        /* fall through */
                case 'K':
                        ret *= 1000;
                        break;
                case '\0':
                        break;
                default:
                        log_msg(LOG_LEVEL_ERROR, "Error: unknown unit suffix %c.\n", *end_ptr);
                        return NAN;
        }

        if (ret < 0.0) {
                return NAN;
        } else {
                return ret;
        }
}

bool is_wine() {
#ifdef WIN32
        HMODULE hntdll = GetModuleHandle("ntdll.dll");
        if(!hntdll) {
                return false;
        }

        return GetProcAddress(hntdll, "wine_get_version");
#endif
        return false;
}

