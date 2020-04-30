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

#ifdef _WIN32
#include <windows.h>
#include <tchar.h>
#else
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/sem.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <sys/types.h>
#endif

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>

#include "platform_ipc.h"

#define MOD_NAME "[platform IPC] "

#ifdef __gnu_linux__
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
#endif

#ifdef _WIN32
void PrintError(void)
{
    // Retrieve the system error message for the last-error code

    LPVOID lpMsgBuf;
    DWORD dw = GetLastError();

    FormatMessage(
        FORMAT_MESSAGE_ALLOCATE_BUFFER |
        FORMAT_MESSAGE_FROM_SYSTEM |
        FORMAT_MESSAGE_IGNORE_INSERTS,
        NULL,
        dw,
        MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
        (LPTSTR) &lpMsgBuf,
        0, NULL );

    // Display the error message and exit the process

    _tprintf(TEXT("%s"), (LPCTSTR) lpMsgBuf);

    LocalFree(lpMsgBuf);
}
#endif

//
// SHM
//
#ifdef __gnu_linux__
static platform_ipc_shm_t platform_ipc_shm_open_common(const char *id, size_t size, int shmflg)
{
        key_t key = get_key(id, 1);
        if (key == -1) {
                perror("ftok");
                return -1;
        }

        int handle;

        if ((handle = shmget(key, size, shmflg)) == -1) {
                perror("shmget");
                if (errno == EEXIST) {
                        fprintf(stderr, MOD_NAME "Try to remove it with \"ipcrm\" (see \"ipcs\"), "
                                        "key %" PRIdMAX ":\n\n\tipcrm -M %" PRIdMAX "\n",
                                        (intmax_t) key, (intmax_t) key);
                }
                return -1;
        }

        return handle;
}
#endif // defined __gnu_linux__

platform_ipc_shm_t platform_ipc_shm_create(const char *id, size_t size)
{
#ifdef _WIN32
        HANDLE hMapFile = CreateFileMapping(
                        INVALID_HANDLE_VALUE,    // use paging file
                        NULL,                    // default security
                        PAGE_READWRITE,          // read/write access
                        0,                       // maximum object size (high-order DWORD)
                        size,                   // maximum object size (low-order DWORD)
                        id);                     // name of mapping object
        if (hMapFile == NULL) {
                _tprintf(TEXT("Could not create file mapping object (%d).\n"),
                                GetLastError());
                PrintError();
                return NULL;
        }
        return hMapFile;
#else
        return platform_ipc_shm_open_common(id, size, IPC_CREAT | IPC_EXCL | 0666);
#endif
}

platform_ipc_shm_t platform_ipc_shm_open(const char *id, size_t size)
{
#ifdef _WIN32
        HANDLE hMapFile = OpenFileMapping(
                        FILE_MAP_ALL_ACCESS,   // read/write access
                        FALSE,                 // do not inherit the name
                        id);               // name of mapping object

        if (hMapFile == NULL)
        {
                _tprintf(TEXT("Could not open file mapping object (%d).\n"),
                                GetLastError());
                PrintError();
                return NULL;
        }
        return hMapFile;
#else
        return platform_ipc_shm_open_common(id, size, 0);
#endif
}

void *platform_ipc_shm_attach(platform_ipc_shm_t handle, size_t size)
{
#ifdef _WIN32
        LPVOID pBuf = (LPTSTR) MapViewOfFile(handle, // handle to map object
                        FILE_MAP_ALL_ACCESS,  // read/write permission
                        0,
                        0,
                        size);

        if (pBuf == NULL)
        {
                _tprintf(TEXT("Could not map view of file (%d).\n"),
                                GetLastError());
                PrintError();
                return NULL;
        }
        return pBuf;
#else
        (void) size;

        void *ret = shmat(handle, NULL, 0);
        if (ret == (void *) -1) {
                perror("shmat");
        }
        return ret;
#endif
}

void platform_ipc_shm_detach(void *ptr)
{
#ifdef _WIN32
        UnmapViewOfFile(ptr);
#else
        if (shmdt(ptr) == -1) {
                perror("shmdt");
        }
#endif
}

void platform_ipc_shm_done(platform_ipc_shm_t handle, bool destroy)
{
#ifdef _WIN32
        (void) destroy;
        CloseHandle(handle);
#else
        if (destroy) {
                if (shmctl(handle, IPC_RMID , 0) == -1) {
                        perror("shmctl");
                }
        }
#endif
}

//
// Semaphores
//
#ifdef __gnu_linux__
static platform_ipc_sem_t platform_ipc_sem_open_common(const char *id, int index, int semflg)
{
        key_t key = get_key(id, index);
        if (key == -1) {
                perror("ftok");
                return -1;
        }

        int handle = semget(key, 1, semflg);
        if (handle == -1) {
                perror("semget");
                if (errno == EEXIST) {
                        fprintf(stderr, MOD_NAME "Try to remove it with \"ipcrm\" (see \"ipcs\"), "
                                        "key %" PRIdMAX ":\n\n\tipcrm -S %" PRIdMAX "\n",
                                        (intmax_t) key, (intmax_t) key);
                }
                return -1;
        }
        return handle;
}
#endif // defined __gnu_linux__

platform_ipc_sem_t platform_ipc_sem_create(const char *id, int index)
{
#ifdef _WIN32
        char name[strlen(id) + 21 + 1];
        sprintf(name, "%s-%d", id, index);

        HANDLE ghSemaphore = CreateSemaphore(
                        NULL,           // default security attributes
                        0,              // initial count
                        (1<<15) - 1,    // maximum count
                        name);          // unnamed semaphore

        if (ghSemaphore == NULL) {
                printf("CreateSemaphore error: %d\n", GetLastError());
                PrintError();
                return NULL;
        }

        return ghSemaphore;
#else
        return platform_ipc_sem_open_common(id, index, IPC_CREAT | IPC_EXCL | 0666);
#endif
}

platform_ipc_sem_t platform_ipc_sem_open(const char *id, int index)
{
#ifdef _WIN32
        char name[strlen(id) + 21 + 1];
        sprintf(name, "%s-%d", id, index);

        HANDLE ghSemaphore = OpenSemaphore(
                        SEMAPHORE_ALL_ACCESS,  // default security attributes
                        FALSE,          // inherit
                        name);          // named semaphore

        if (ghSemaphore == NULL) {
                printf("CreateSemaphore error: %d\n", GetLastError());
                PrintError();
                return NULL;
        }

        return ghSemaphore;
#else
        return platform_ipc_sem_open_common(id, index, 0);
#endif
}

bool platform_ipc_sem_post(platform_ipc_sem_t handle)
{
#ifdef _WIN32
        if (!ReleaseSemaphore(
                                handle,       // handle to semaphore
                                1,            // increase count by one
                                NULL) )       // not interested in previous count
        {
                printf("ReleaseSemaphore error: %d\n", GetLastError());
                PrintError();
                return false;
        }
#else
        struct sembuf op;
        op.sem_num = 0;
        op.sem_op = 1;
        op.sem_flg = 0;
        if (semop(handle, &op, 1) < 0) {
                perror("semop");
                return false;
        }
#endif
        return true;
}

bool platform_ipc_sem_wait(platform_ipc_sem_t handle)
{
#ifdef _WIN32
        DWORD dwWaitResult;

        dwWaitResult = WaitForSingleObject(
            handle,        // handle to semaphore
            INFINITE);     // infinite time-out interval
        switch (dwWaitResult)
        {
        // The semaphore object was signaled.
        case WAIT_OBJECT_0:
                return true;

                // The semaphore was nonsignaled, so a time-out occurred.
        case WAIT_TIMEOUT:
                printf("Thread %d: wait timed out\n", GetCurrentThreadId());
                return false;
        default:
                abort();
        }

#else
        struct sembuf op;
        op.sem_num = 0;
        op.sem_op = -1;
        op.sem_flg = 0;
        if (semop(handle, &op, 1) < 0) {
                perror("semop");
                return false;
        }
        return true;
#endif
}

void platform_ipc_sem_done(platform_ipc_sem_t handle, bool destroy)
{
#ifdef _WIN32
        (void) destroy;
        CloseHandle(handle);
#else
        if (destroy) {
                if (semctl(handle, IPC_RMID , 0) == -1) {
                        perror("semctl");
                }
        }
#endif
}

