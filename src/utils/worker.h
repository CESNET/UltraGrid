/**
 * @file   utils/worker.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2023 CESNET, z. s. p. o.
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

 
#ifndef WORKER_H
#define WORKER_H

#ifndef __cplusplus
#include <stddef.h>
#else
#include <cstddef>
extern "C" {
#endif

typedef void *task_result_handle_t;
typedef void *(*runnable_t)(void *);

// fuctions documented at definition
task_result_handle_t task_run_async(runnable_t task, void *data);
void task_run_async_detached(runnable_t task, void *data);
void *wait_task(task_result_handle_t handle);
void task_run_parallel(runnable_t task, int worker_count, void *data, size_t data_size, void **res);

/**
 * @param data_len   in/out processed block length in bytes (multpile of respawn_parallel's size param)
 */
typedef void (*respawn_parallel_callback_t)(void *in, void *out, size_t data_len, void *udata);
void respawn_parallel(void *in, void *out, size_t nmemb, size_t size, respawn_parallel_callback_t c, void *udata);

#ifdef __cplusplus
}
#endif

#endif // defined WORKER_H

