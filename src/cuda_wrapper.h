/**
 * @file   cuda_wrapper.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013 CESNET z.s.p.o.
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

#ifndef CUDA_WRAPPER_H_
#define CUDA_WRAPPER_H_

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined _MSC_VER || defined __MINGW32__
#ifdef EXPORT_DLL_SYMBOLS
#define CUDA_DLL_API __declspec(dllexport)
#else
#define CUDA_DLL_API __declspec(dllimport)
#endif
#else // other platforms
#define CUDA_DLL_API
#endif

/// @{
#define CUDA_WRAPPER_SUCCESS 0
/// @}

/// @{
#define CUDA_WRAPPER_MEMCPY_HOST_TO_DEVICE 0
#define CUDA_WRAPPER_MEMCPY_DEVICE_TO_HOST 1
/// @}

typedef void *cuda_wrapper_stream_t;

CUDA_DLL_API int cuda_wrapper_free(void *buffer);
CUDA_DLL_API int cuda_wrapper_free_host(void *buffer);
CUDA_DLL_API int cuda_wrapper_host_alloc(void **pHost, size_t size, unsigned int flags);
CUDA_DLL_API int cuda_wrapper_malloc(void **buffer, size_t data_len);
CUDA_DLL_API int cuda_wrapper_malloc_host(void **buffer, size_t data_len);
CUDA_DLL_API int cuda_wrapper_memcpy(void *dst, const void *src,
                size_t count, int kind);
CUDA_DLL_API const char *cuda_wrapper_last_error_string(void);
CUDA_DLL_API int cuda_wrapper_set_device(int index);
CUDA_DLL_API int cuda_wrapper_get_last_error(void);
CUDA_DLL_API const char * cuda_wrapper_get_error_string(int error);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // CUDA_WRAPPER_H_

