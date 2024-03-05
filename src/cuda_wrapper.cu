/**
 * @file   cuda_wrapper.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This file contais wrapper around CUDA functions.
 *
 * This file is needed for Windows to wrap CUDA calls that need to be compiled
 * with nvcc/MSVC and not gcc/clang as a glue with the rest of UltraGrid
 * CUDA-related code (using cudaMemcpy etc., obviously not possible to compile
 * kernels etc.)
 */
/*
 * Copyright (c) 2013-2024 CESNET z.s.p.o.
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

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "cuda_wrapper.h"

typedef void *cuda_wrapper_stream_t;

static inline int map_cuda_error(cudaError_t cuda_error) {
        return (int) cuda_error;
};

static inline const char * map_error_string(int error) {

        return cudaGetErrorString((cudaError_t) error);
};

static inline enum cudaMemcpyKind map_cuda_memcpy_kind(int our_kind) {
        struct kind_mapping {
                enum cudaMemcpyKind kind;
                int our_kind;
        };
        struct kind_mapping mapping[] = {
                { cudaMemcpyHostToDevice, CUDA_WRAPPER_MEMCPY_HOST_TO_DEVICE },
                { cudaMemcpyDeviceToHost, CUDA_WRAPPER_MEMCPY_DEVICE_TO_HOST },
        };

        int i;
        for (i = 0; i < sizeof(mapping)/sizeof(struct kind_mapping); ++i) {
                if (our_kind == mapping[i].our_kind) {
                        return mapping[i].kind;
                }
        }

        abort(); // should not reach here
};

int cuda_wrapper_free(void *buffer)
{
        return map_cuda_error(cudaFree(buffer));
}

int cuda_wrapper_free_host(void *buffer)
{
        return map_cuda_error(cudaFreeHost(buffer));
}

int cuda_wrapper_host_alloc(void **pHost, size_t size, unsigned int flags)
{
        return map_cuda_error(cudaHostAlloc(pHost, size, flags));
}

int cuda_wrapper_malloc(void **buffer, size_t data_len)
{
        return map_cuda_error(cudaMalloc(buffer, data_len));
}

int cuda_wrapper_malloc_host(void **buffer, size_t data_len)
{
        return map_cuda_error(cudaMallocHost(buffer, data_len));
}

int cuda_wrapper_memcpy(void *dst, const void *src,
                size_t count, int kind)
{
        return map_cuda_error(
                        cudaMemcpy(dst, src, count,
                                map_cuda_memcpy_kind(kind)));
}

const char *cuda_wrapper_last_error_string(void)
{
        return cudaGetErrorString(cudaGetLastError());
}

int cuda_wrapper_get_last_error(void)
{
        return map_cuda_error(cudaGetLastError());
}

const char *cuda_wrapper_get_error_string(int error)
{
        return map_error_string(error);
}

int cuda_wrapper_set_device(int index)
{
        return map_cuda_error(
                        cudaSetDevice(index));
}

/// adapted from gpujpeg_print_devices_info()
void cuda_wrapper_print_devices_info(void)
{
        int device_count = 0;
        if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
                fprintf(stderr, "Cannot get number of CUDA devices: %s\n", cudaGetErrorString(cudaGetLastError()));
                return;
        }
        if (device_count == 0) {
                fprintf(stderr, "There is no device supporting CUDA.\n");
        } else {
                printf("There %s %d devices supporting CUDA:\n", device_count == 1 ? "is" : "are", device_count);
        }

        for ( int device_id = 0; device_id < device_count; device_id++ ) {
                struct cudaDeviceProp device_properties;
                if (cudaGetDeviceProperties(&device_properties, device_id) != cudaSuccess) {
                        fprintf(stderr, "Cannot get number of CUDA device #%d properties: %s\n", device_id, cudaGetErrorString(cudaGetLastError()));
                        continue;
                }

                printf("\nDevice #%d: \"%s\"\n", device_id, device_properties.name);
                printf("  Compute capability: %d.%d\n", device_properties.major, device_properties.minor);
                printf("  Total amount of global memory: %zu kB\n", device_properties.totalGlobalMem / 1024);
                printf("  Total amount of constant memory: %zu kB\n", device_properties.totalGlobalMem / 1024);
                printf("  Total amount of shared memory per block: %zu kB\n", device_properties.sharedMemPerBlock / 1024);
                printf("  Total number of registers available per block: %d\n", device_properties.regsPerBlock);
                printf("  Multiprocessors: %d\n", device_properties.multiProcessorCount);
    }
}

