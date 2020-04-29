/**
 * @file   compat/platform_ipc.c
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <unistd.h>

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "platform_ipc.h"

#define MOD_NAME "[platform IPC] "

static key_t get_key(const char *id, int proj_id) {
        char key_path[1024];
        snprintf(key_path, sizeof key_path, "/tmp/%s-%" PRIdMAX, id, (intmax_t) getuid());

        int fd = open(key_path, O_WRONLY | O_CREAT, 0600);
        if (fd == -1) {
                perror("create key file");
                fprintf(stderr, MOD_NAME "Cannot create %s\n", key_path);
                return -1;
        }
        close(fd);

        return ftok(key_path, proj_id);
}

//
// SHM
//
static platform_ipc_shm_t platform_ipc_shm_open_common(const char *id, size_t size, int shmflg)
{
        key_t key = get_key(id, 1);
        if (key == -1) {
                perror("ftok");
                return -1;
        }

        int handle;

        if ((handle = shmget(key, size, shmflg)) == -1) {
                if (errno != EEXIST) {
                        perror("shmget");
                        if (errno == EINVAL) {
                                fprintf(stderr, MOD_NAME "Try to remove it with \"ipcrm\" (see \"ipcs\"), "
                                                "key %lld:\n ipcrm -M %lld", (long long) key, (long long) key);
                        }
                }
                return -1;
        }

        return handle;
}

platform_ipc_shm_t platform_ipc_shm_create(const char *id, size_t size)
{
        return platform_ipc_shm_open_common(id, size, IPC_CREAT | 0666);
}

platform_ipc_shm_t platform_ipc_shm_open(const char *id, size_t size)
{
        return platform_ipc_shm_open_common(id, size, 0);
}

void *platform_ipc_shm_attach(platform_ipc_shm_t handle)
{
        void *ret = shmat(handle, NULL, 0);
        if (ret == (void *) -1) {
                perror("shmat");
        }
        return ret;
}

void platform_ipc_shm_detach(void *ptr)
{
        if (shmdt(ptr) == -1) {
                perror("shmdt");
        }
}

void platform_ipc_shm_done(platform_ipc_shm_t handle)
{
        if (shmctl(handle, IPC_RMID , 0) == -1) {
		perror("shmctl");
	}
}

//
// Semaphores
//

static platform_ipc_sem_t platform_ipc_sem_open_common(const char *id, int index, int semflg)
{
        key_t key = get_key(id, index);
        if (key == -1) {
                perror("ftok");
                return -1;
        }

        int handle = semget(key, 1, semflg);
        if (handle == -1) {
                if (errno != EEXIST) {
                        perror("semget");
                        if (errno == EINVAL) {
                                fprintf(stderr, MOD_NAME "Try to remove it with \"ipcrm\" (see \"ipcs\"), "
                                                "key %lld:\n ipcrm -S %lld", (long long) key, (long long) key);
                        }
                }
                return -1;
        }
        return handle;
}

platform_ipc_sem_t platform_ipc_sem_create(const char *id, int index)
{
        return platform_ipc_sem_open_common(id, index, IPC_CREAT | 0666);
}

platform_ipc_sem_t platform_ipc_sem_open(const char *id, int index)
{
        return platform_ipc_sem_open_common(id, index, 0);
}

bool platform_ipc_sem_post(platform_ipc_sem_t handle)
{
        struct sembuf op;
        op.sem_num = 0;
        op.sem_op = 1;
        op.sem_flg = 0;
        if (semop(handle, &op, 1) < 0) {
                perror("semop");
                return false;
        }
        return true;
}

bool platform_ipc_sem_wait(platform_ipc_sem_t handle)
{
        struct sembuf op;
        op.sem_num = 0;
        op.sem_op = -1;
        op.sem_flg = 0;
        if (semop(handle, &op, 1) < 0) {
                perror("semop");
                return false;
        }
        return true;
}

void platform_ipc_sem_done(platform_ipc_sem_t handle)
{
	if (semctl(handle, IPC_RMID , 0) == -1) {
		perror("semctl");
	}
}

