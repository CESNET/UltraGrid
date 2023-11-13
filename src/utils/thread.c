/**
 * @file   utils/thread.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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
#endif

#include <libgen.h>
#ifndef _WIN32
#include <pthread.h>
#endif
#ifdef HAVE_SETTHREADDESCRIPTION
#include <processthreadsapi.h>
#endif
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "host.h"
#include "utils/thread.h"

#if ! defined  WIN32 || defined HAVE_SETTHREADDESCRIPTION
static inline char *get_argv_program_name(void) {
#ifdef HAVE_CONFIG_H
        if (uv_argv != NULL && uv_argv[0] != NULL) {
                char *prog_name = (char *) malloc(strlen(uv_argv[0]) + 2);
                strcpy(prog_name, uv_argv[0]);
                char *name = basename(prog_name);
                memmove(prog_name, name, strlen(name) + 1);
                strcat(prog_name, "-");
                return prog_name;
        }
#endif // defined HAVE_CONFIG_H
        return strdup("uv-");
}
#endif

void set_thread_name(const char *name) {
#ifdef __linux__
// thread name can have at most 16 chars (including terminating null char)
        char *prog_name = get_argv_program_name();
        char tmp[16];
        tmp[sizeof tmp - 1] = '\0';
        strncpy(tmp, prog_name, sizeof tmp - 1);
        free(prog_name);
        strncat(tmp, name,  sizeof tmp - strlen(tmp) - 1);
        pthread_setname_np(pthread_self(), tmp);
#elif defined __APPLE__
        char *prog_name = get_argv_program_name();
        char *tmp = (char *) alloca(strlen(prog_name) + strlen(name) + 1);
        strcpy(tmp, prog_name);
        free(prog_name);
        strcat(tmp, name);
        pthread_setname_np(tmp);
#elif defined _WIN32
// supported from Windows 10, not yet in headers
#ifdef HAVE_SETTHREADDESCRIPTION
        const char *prog_name = get_argv_program_name();
        size_t dst_len = (mbstowcs(NULL, prog_name, 0) + mbstowcs(NULL, name, 0) + 1) * sizeof(wchar_t);
        wchar_t *tmp = (wchar_t *) alloca(dst_len);
        mbstowcs(tmp, prog_name, dst_len / sizeof(wchar_t));
        mbstowcs(tmp + wcslen(tmp), name, dst_len / sizeof(wchar_t) - wcslen(tmp));
        SetThreadDescription(GetCurrentThread(), tmp);
#else
	UNUSED(name);
#endif
#endif
}

