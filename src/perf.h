/**
 * @file   perf.
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011 CESNET, z. s. p. o.
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

#ifndef _PERF_H
#define _PERF_H


#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

/* Public API */
#define UVP_INIT 1
#define UVP_GETFRAME 2
#define UVP_PUTFRAME 3
#define UVP_DECODEFRAME 4
#define UVP_SEND 5
#define UVP_CREATEPBUF 6

#if (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 1)) && defined PERF

typedef uint32_t _uvp_event_t;
typedef int64_t _uvp_arg_t;

void perf_init(void);

/* Private section */
struct _uvp_entry {
        long int tv_sec;
        long int tv_usec;
        _uvp_event_t event;
        _uvp_arg_t arg;
};

#define ENTRY_LEN sizeof(struct _uvp_entry)
#define BUFFER_LEN (ENTRY_LEN * 512)

static key_t _uvp_key = 5043;

void perf_record(_uvp_event_t event, _uvp_arg_t arg);

#else

#define perf_record(x,y) {}
#define perf_init() {}

#endif /* __GNUC__ && PERF */

#endif /* _PERF_H */

