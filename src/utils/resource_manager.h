/**
 * @file   utils/resource_manager.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013 CESNET, z. s. p. o.
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

#ifndef RESOURCE_MANAGER_H_
#define RESOURCE_MANAGER_H_

#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif
/*
 * Don't bother with the fact that the prototypes are actually function pointers.
 * It is so as it is callable also from libraries. Just call it as an ordinary
 * function.
 */

/**
 * @param name is used to uniquely (!) identify lock. So when requested from 2 locations
 * lock with same name, the same lock handle will be returned
 */
pthread_mutex_t *rm_acquire_shared_lock(const char *name);
void rm_release_shared_lock(const char *name);

void rm_lock();
void rm_unlock();
void *rm_get_shm(const char *name, int size);
typedef void *(*singleton_initializer_t)(void *);
typedef void (*singleton_deleter_t)(void *);
void *rm_singleton(const char *name, singleton_initializer_t initializer, void *initializer_data,
                singleton_deleter_t deleter);

#ifdef __cplusplus
}
#endif

#endif /* RESOURCE_MANAGER_H_ */

