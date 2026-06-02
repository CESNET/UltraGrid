/**
 * @file   utils/wait_obj.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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

#include "utils/wait_obj.h"

#include <pthread.h>
#include <stdlib.h>

#include "compat/c23.h" // IWYU pragma: keep
#include "utils/pthread.h"

#define MOD_NAME "[wait_obj] "

struct wait_obj {
        pthread_mutex_t lock;
        pthread_cond_t  cv;
        bool            val;
};

struct wait_obj *
wait_obj_init()
{
        struct wait_obj *wait_obj = calloc(1, sizeof *wait_obj);
        ug_pthread_mutex_init(&wait_obj->lock);
        pthread_cond_init(&wait_obj->cv, nullptr);
        return wait_obj;
}

void
wait_obj_reset(struct wait_obj *wait_obj)
{
        CHK_PTHR(pthread_mutex_lock(&wait_obj->lock));
        wait_obj->val = false;
        CHK_PTHR(pthread_mutex_unlock(&wait_obj->lock));
}

void
wait_obj_wait(struct wait_obj *wait_obj)
{
        CHK_PTHR(pthread_mutex_lock(&wait_obj->lock));
        while (!wait_obj->val) {
                CHK_PTHR(pthread_cond_wait(&wait_obj->cv, &wait_obj->lock));
        }
        CHK_PTHR(pthread_mutex_unlock(&wait_obj->lock));
}

void
wait_obj_notify(struct wait_obj *wait_obj)
{
        CHK_PTHR(pthread_mutex_lock(&wait_obj->lock));
        wait_obj->val = true;
        CHK_PTHR(pthread_mutex_unlock(&wait_obj->lock));
        CHK_PTHR(pthread_cond_signal(&wait_obj->cv));
}

void
wait_obj_done(struct wait_obj *wait_obj)
{
        CHK_PTHR(pthread_cond_destroy(&wait_obj->cv));
        CHK_PTHR(pthread_mutex_destroy(&wait_obj->lock));
        free(wait_obj);
}
