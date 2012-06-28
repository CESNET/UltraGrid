/*
 * FILE:    perf.c
 * AUTHORS: Isidor Kouvelas 
 *          Colin Perkins 
 *          Mark Handley 
 *          Orion Hodson
 *          Jerry Isdale
 * 
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 * Copyright (c) 1995-2000 University College London
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

#if defined PERF && (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 1))

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>

static volatile size_t _uvp_offset;
static int _uvp_id;
static char *_uvp_shm;

void perf_init()
{
        _uvp_id = shmget(_uvp_key, BUFFER_LEN, IPC_CREAT | 0666);
        _uvp_shm = shmat(_uvp_id, NULL, 0);
        _uvp_offset = 0;
}

void perf_record(_uvp_event_t event, _uvp_arg_t arg)
{
        size_t current;
        struct timeval tv;
        char *ptr;

        gettimeofday(&tv, NULL);
        current = __sync_fetch_and_add(&_uvp_offset, ENTRY_LEN);
        if(current == BUFFER_LEN)
        {
                _uvp_offset = 0;
                current = 0;
        } else if (current > BUFFER_LEN) {
                while((current = __sync_fetch_and_add(&_uvp_offset, ENTRY_LEN)) > BUFFER_LEN)
                        ;
        }
        ptr = _uvp_shm + current;
        ((struct _uvp_entry *) ptr)->tv_sec = tv.tv_sec;
        ((struct _uvp_entry *) ptr)->tv_usec = tv.tv_usec;
        ((struct _uvp_entry *) ptr)->event = event;
        ((struct _uvp_entry *) ptr)->arg = arg;
}

#endif /* PERF && __GNUC__ */
