/**
 * @file   debug.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2021-2024 CESNET
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

#ifndef UTILS_DEBUG_H_36CF2E79_AF28_4308_BA8D_56D403BDCC44
#define UTILS_DEBUG_H_36CF2E79_AF28_4308_BA8D_56D403BDCC44

#ifdef HAVE_CONFIG_H
#include "config.h"      // for DEBUG
#endif

#include "../debug.h"    // for log_msg
#include "tv.h"          // for get_time_in_ns

#ifdef __cplusplus
#include <cstdio>        // for FILE
#define EXTERNC extern "C"
#else
#include <stdio.h>
#define EXTERNC          // for FILE
#endif

#ifdef DEBUG

EXTERNC void debug_file_dump(const char *key,
                             void (*serialize)(const void *data, FILE *),
                             void *data);
#define DEBUG_TIMER_EVENT(name) time_ns_t name = get_time_in_ns()
#define DEBUG_TIMER_START(name) DEBUG_TIMER_EVENT(name##_start);
#define DEBUG_TIMER_STOP(name) \
        DEBUG_TIMER_EVENT(name##_stop); \
        log_msg(LOG_LEVEL_DEBUG2, "%s duration: %lf s\n", #name, \
                (name##_stop - name##_start) / NS_IN_SEC_DBL) \
                // NOLINT(cppcoreguidelines-pro-type-vararg, hicpp-vararg)

#else

#define debug_file_dump(key, serialize, data) \
        (void) (key), (void) (serialize), (void) (data)
#define DEBUG_TIMER_START(name)
#define DEBUG_TIMER_STOP(name)

#endif // ! defined DEBUG

#endif // ! defined UTILS_DEBUG_H_36CF2E79_AF28_4308_BA8D_56D403BDCC44
