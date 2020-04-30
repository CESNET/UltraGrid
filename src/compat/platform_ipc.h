/**
 * @file   compat/platform_ipc.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file is a part of UltraGrid.
 */
/*
 * Copyright (c) 2020 CESNET, z. s. p. o.
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

#ifndef _PLATFORM_IPC
#define _PLATFORM_IPC

#ifndef __cplusplus
#include <stdbool.h>
#endif // defined __cplusplus

#ifdef _WIN32
typedef HANDLE platform_ipc_shm_t;
typedef HANDLE platform_ipc_sem_t;
#else // Linux
typedef int platform_ipc_shm_t;
typedef int platform_ipc_sem_t;
#define PLATFORM_IPC_ERR (-1)
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @param index   index of SHM for unique ID (see ftok(), param proj_id),
 *                must be nonzero
 */
platform_ipc_shm_t platform_ipc_shm_create(const char *id, size_t size);
platform_ipc_shm_t platform_ipc_shm_open(const char *id, size_t size);
void *platform_ipc_shm_attach(platform_ipc_shm_t handle);
void platform_ipc_shm_detach(void *ptr);
void platform_ipc_shm_destroy(platform_ipc_shm_t handle);

/**
 * @param index   index of semaphore for unique ID (see ftok(), param proj_id),
 *                must be nonzero
 */
platform_ipc_sem_t platform_ipc_sem_create(const char *id, int index);
platform_ipc_sem_t platform_ipc_sem_open(const char *id, int index);
bool platform_ipc_sem_post(platform_ipc_sem_t handle);
bool platform_ipc_sem_wait(platform_ipc_sem_t handle);
void platform_ipc_sem_destroy(platform_ipc_sem_t handle);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // defined _PLATFORM_IPC
