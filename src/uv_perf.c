/*
 * FILE:    uv_perf.c
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

struct message {
        const char * msg;
        const char * argtype;
};

static const struct message messages[] = {
        [UVP_INIT] = { "Profiling tool initialized", "" },
        [UVP_GETFRAME] = { "Frame get", "display address" },
        [UVP_PUTFRAME] = { "Frame put", "frame address (UNUSED!)" },
        [UVP_DECODEFRAME] = { "Decode frame", "frame address" },
        [UVP_SEND] = { "TX Send", "timestamp" },
        [UVP_CREATEPBUF] = { "RX Received (first packet)", "timestamp" },
};

int main(int argc, char *argv[])
{
        int id;
        volatile struct _uvp_entry *shm;
        volatile struct _uvp_entry *tmp;
        struct timeval tv_old;
        struct timeval tv_cur;

        tv_old.tv_sec = 0;
        tv_old.tv_usec = 0;

        id = shmget(_uvp_key, BUFFER_LEN, IPC_CREAT | 0666);
        shm = shmat(id, NULL, 0);
        tmp = shm;

        memset((char *) shm, '\0', BUFFER_LEN);

        while(1)
        {
                if(tmp == shm + BUFFER_LEN/ENTRY_LEN) {
                        tmp = shm;
                }
                tv_cur.tv_sec = tmp->tv_sec;
                tv_cur.tv_usec = tmp->tv_usec;
                if(tv_gt(tv_old, tv_cur) && tv_diff_usec(tv_old, tv_cur) > 1000)  /* 0.001 sec out of order is OK */
                                //tmp->tv_usec < tv_old.tv_usec) 
                                {
                        usleep(100000);
                        continue;
                }
                tv_old.tv_sec = tmp->tv_sec;
                tv_old.tv_usec = tmp->tv_usec;

                if(tmp->tv_sec != 0) {
                        printf("%10ld %10ld%30s:\t%20lX (%s)\n", tmp->tv_sec, tmp->tv_usec, messages[tmp->event].msg, tmp->arg, messages[tmp->event].argtype);
                }
                tmp++;
        }
        shmdt(shm);

        return 0;
}

