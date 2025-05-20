/*
 * FILE:    platform_semaphore.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2012 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 *
 */

#include "compat/platform_semaphore.h"

#include <errno.h>      // for EINTR, errno
#include <stdio.h>      // for perror
#include <stdlib.h>     // for abort

#ifdef __APPLE__
#include <mach/semaphore.h>
#include <mach/task.h>
#include <mach/mach.h>
#else
#include <semaphore.h>
#endif                          /* __APPLE__ */

void platform_sem_init(void *semStructure, int pshared, int initialValue)
{
#ifdef __APPLE__
        (void) pshared;
        semaphore_create(mach_task_self(), (semaphore_t *) semStructure,
                         SYNC_POLICY_FIFO, initialValue);
#else
        if (sem_init((sem_t *) semStructure, pshared, initialValue) != 0) {
                perror("sem_init");
                abort();
        }
#endif                          /* __APPLE__ */
}

void platform_sem_post(void *semStructure)
{
#ifdef __APPLE__
        semaphore_signal(*((semaphore_t *) semStructure));
#else
        sem_post((sem_t *) semStructure);
#endif                          /* __APPLE__ */
}

void platform_sem_wait(void *semStructure)
{
#ifdef __APPLE__
        semaphore_wait(*((semaphore_t *) semStructure));
#else
        int ret = 0;
        while ((ret = sem_wait((sem_t *) semStructure)) == -1 && errno == EINTR) {  }
        if (ret == -1) {
                perror("sem_wait");
        }
#endif                          /* __APPLE__ */
}

void platform_sem_destroy(void *semStructure)
{
#ifdef __APPLE__
        semaphore_destroy(mach_task_self(), *(semaphore_t *) semStructure);
#else
        sem_destroy((sem_t *) semStructure);
#endif                          /* __APPLE__ */
}

