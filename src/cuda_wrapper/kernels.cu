/**
 * @file   cuda_wrapper/kernels.cu
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * This file hosts various CUDA kernels. Currently there are only kernels
 * for cmpto_j2k compression and decompression.
 */
/*
 * Copyright (c) 2024 CESNET
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

#include "kernels.hpp"

#include <cstdint>
#include <cstdio>
#include <cuda_runtime_api.h>

extern volatile int log_level;
#define LOG_LEVEL_DEBUG 7

#define MEASURE_KERNEL_DURATION_START(stream) \
        cudaEvent_t t0, t1; \
        cudaEventCreate(&t0); \
        cudaEventCreate(&t1); \
        cudaEventRecord(t0, stream);
#define MEASURE_KERNEL_DURATION_STOP(stream) \
        cudaEventRecord(t1, stream); \
        cudaEventSynchronize(t1); \
        float elapsedTime = NAN; \
        cudaEventElapsedTime(&elapsedTime, t0, t1); \
        if (log_level >= LOG_LEVEL_DEBUG) { \
                printf("%s elapsed time: %f ms\n", __func__, elapsedTime); \
        } \
        if (elapsedTime > 10.0) { \
                fprintf( \
                    stderr, \
                    "Kernel in func %s duration %f > 10 ms, please report!\n", \
                    __func__, elapsedTime); \
        }

//    ___   _____  ____  ___      __     ___   ___   ___    __
//   / _ \ / ___/ / / / ( _ ) ____\ \   / _ \ <  /  |_  |  / /
//  / , _// (_ / /_  _// _  |/___/ > > / , _/ / /  / __/  / /__
// /_/|_| \___/   /_/  \___/      /_/ /_/|_| /_/  /____/ /____/

/**
 * modified @ref vc_copylineRG48toR12L
 */
template <typename load_t>
__device__ static void
rt48_to_r12l_compute_blk(const uint8_t *in, uint8_t *out)
{
         // load the data from in to src_u32
         auto    *in_t = (load_t *) in;
         uint32_t src_u32[12];
         for (unsigned i = 0; i < sizeof src_u32 / sizeof src_u32[0]; ++i) {
                 static_assert(sizeof(load_t) == 2 || sizeof(load_t) == 4,
                               "Just uint{16,32} supported!");
                 if (sizeof(load_t) == 4) {
                         src_u32[i] = in_t[i];
                 } else {
                         src_u32[i] = in_t[2 * i] | in_t[2 * i + 1] << 16;
                 }
         }

        uint32_t dst_u32[9];
        auto *dst = (uint8_t *) dst_u32;
        auto *src = (uint8_t *) src_u32;

        // 0
        dst[0] = src[0] >> 4;
        dst[0] |= src[1] << 4;
        dst[1] = src[1] >> 4;
        src += 2;

        dst[1] |= src[0] & 0xF0;
        dst[2] = src[1];
        src += 2;

        dst[3] = src[0] >> 4;
        dst[3] |= src[1] << 4;
        dst[4 + 0] = src[1] >> 4;
        src += 2;

        // 1
        dst[4 + 0] |= src[0] & 0xF0;
        dst[4 + 1] = src[1];
        src += 2;

        dst[4 + 2] = src[0] >> 4;
        dst[4 + 2] |= src[1] << 4;
        dst[4 + 3] = src[1] >> 4;
        src += 2;

        dst[4 + 3] |= src[0] & 0xF0;
        dst[8 + 0] = src[1];
        src += 2;

        // 2
        dst[8 + 1] = src[0] >> 4;
        dst[8 + 1] |= src[1] << 4;
        dst[8 + 2] = src[1] >> 4;
        src += 2;

        dst[8 + 2] |= src[0] & 0xF0;
        dst[8 + 3] = src[1];
        src += 2;

        dst[12 + 0] = src[0] >> 4;
        dst[12 + 0] |= src[1] << 4;
        dst[12 + 1] = src[1] >> 4;
        src += 2;

        // 3
        dst[12 + 1] |= src[0] & 0xF0;
        dst[12 + 2] = src[1];
        src += 2;

        dst[12 + 3] = src[0] >> 4;
        dst[12 + 3] |= src[1] << 4;
        dst[16 + 0] = src[1] >> 4;
        src += 2;

        dst[16 + 0] |= src[0] & 0xF0;
        dst[16 + 1] = src[1];
        src += 2;

        // 4
        dst[16 + 2] = src[0] >> 4;
        dst[16 + 2] |= src[1] << 4;
        dst[16 + 3] = src[1] >> 4;
        src += 2;

        dst[16 + 3] |= src[0] & 0xF0;
        dst[20 + 0] = src[1];
        src += 2;

        dst[20 + 1] = src[0] >> 4;
        dst[20 + 1] |= src[1] << 4;
        dst[20 + 2] = src[1] >> 4;
        src += 2;

        // 5
        dst[20 + 2] |= src[0] & 0xF0;
        dst[20 + 3] = src[1];
        src += 2;

        dst[24 + 0] = src[0] >> 4;
        dst[24 + 0] |= src[1] << 4;
        dst[24 + 1] = src[1] >> 4;
        src += 2;

        dst[24 + 1] |= src[0] & 0xF0;
        dst[24 + 2] = src[1];
        src += 2;

        // 6
        dst[24 + 3] = src[0] >> 4;
        dst[24 + 3] |= src[1] << 4;
        dst[28 + 0] = src[1] >> 4;
        src += 2;

        dst[28 + 0] |= src[0] & 0xF0;
        dst[28 + 1] = src[1];
        src += 2;

        dst[28 + 2] = src[0] >> 4;
        dst[28 + 2] |= src[1] << 4;
        dst[28 + 3] = src[1] >> 4;
        src += 2;

        // 7
        dst[28 + 3] |= src[0] & 0xF0;
        dst[32 + 0] = src[1];
        src += 2;

        dst[32 + 1] = src[0] >> 4;
        dst[32 + 1] |= src[1] << 4;
        dst[32 + 2] = src[1] >> 4;
        src += 2;

        dst[32 + 2] |= src[0] & 0xF0;
        dst[32 + 3] = src[1];
        src += 2;

        // store the result
        auto *out_u32 = (uint32_t *) out;
        for (unsigned i = 0; i < sizeof dst_u32 / sizeof dst_u32[0]; ++i) {
                out_u32[i] = dst_u32[i];
        }
}

template <typename load_t>
__device__ static void
rt48_to_r12l_compute_last_blk(uint8_t *src, uint8_t *dst, unsigned width)
{
        alignas(uint32_t) uint8_t tmp[48];
        for (unsigned i = 0; i < width * 6; ++i) {
                tmp[i] = src[i];
        }
        rt48_to_r12l_compute_blk<load_t>(tmp, dst);
}

/**
 * @todo fix the last block for widths not divisible by 8
 */
template <typename load_t>
__global__ static void
kernel_rg48_to_r12l(uint8_t *in, uint8_t *out, unsigned size_x)
{
        unsigned position_x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned position_y = threadIdx.y + blockIdx.y * blockDim.y;
        if (position_x >= (size_x + 7) / 8) {
                return;
        }
        uint8_t *src = in + 2 * (position_y * 3 * size_x + position_x * 3 * 8);
        uint8_t *dst =
            out + (position_y * ((size_x + 7) / 8) + position_x) * 36;

        // handle incomplete blocks
        if (position_x == size_x / 8) {
                rt48_to_r12l_compute_last_blk<load_t>(src, dst,
                                                      size_x - position_x * 8);
                return;
        }
        rt48_to_r12l_compute_blk<load_t>(src, dst);
}

/**
 * @sa cmpto_j2k_dec_postprocessor_run_callback_cuda
 */
int postprocess_rg48_to_r12l(
    void * /* postprocessor */,
    void * /* img_custom_data*/,
    size_t /* img_custom_data_size */,
    int size_x,
    int size_y,
    struct cmpto_j2k_dec_comp_format * /* comp_formats */,
    int /* comp_count */,
    void *input_samples,
    size_t /* input_samples_size */,
    void * /* temp_buffer */,
    size_t /* temp_buffer_size */,
    void * output_buffer,
    size_t /* output_buffer_size */,
    void * vstream
) {
        cudaStream_t stream = (cudaStream_t) vstream;
        dim3 threads_per_block(256);
        dim3 blocks((((size_x + 7) / 8) + 255) / 256, size_y);

        MEASURE_KERNEL_DURATION_START(stream)

        if (size_x % 2 == 0) {
                kernel_rg48_to_r12l<uint32_t>
                    <<<blocks, threads_per_block, 0, (cudaStream_t) stream>>>(
                        (uint8_t *) input_samples, (uint8_t *) output_buffer,
                        size_x);
        } else {
                thread_local bool warn_print;
                if (!warn_print) {
                        fprintf(stderr,
                                "%s: Odd width %d px will use slower kernel!\n",
                                __func__, size_x);
                        warn_print = true;
                }
                kernel_rg48_to_r12l<uint16_t>
                    <<<blocks, threads_per_block, 0, (cudaStream_t) stream>>>(
                        (uint8_t *) input_samples, (uint8_t *) output_buffer,
                        size_x);
        }

        MEASURE_KERNEL_DURATION_STOP(stream)

        return cudaGetLastError();
}

//     ___   ___   ___    __      __     ___   _____  ____  ___
//    / _ \ <  /  |_  |  / /  ____\ \   / _ \ / ___/ / / / ( _ )
//   / , _/ / /  / __/  / /__/___/ > > / , _// (_ / /_  _// _  |
//  /_/|_| /_/  /____/ /____/     /_/ /_/|_| \___/   /_/  \___/

template <typename store_t>
__device__ static void r12l_to_rg48_compute_blk(const uint8_t *src,
                                                uint8_t       *dst);

template <typename store_t>
__global__ static void
kernel_r12l_to_rg48(uint8_t *in, uint8_t *out, unsigned size_x)
{
        unsigned position_x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned position_y = threadIdx.y + blockIdx.y * blockDim.y;
        if (position_x >= (size_x + 7) / 8) {
                return;
        }
        uint8_t *dst = out + 2 * (position_y * 3 * size_x + position_x * 3 * 8);
        uint8_t *src =
            in + (position_y * ((size_x + 7) / 8) + position_x) * 36;

        if (position_x == size_x / 8) {
                // compute the last incomplete block
                alignas(uint32_t) uint8_t tmp[48];
                r12l_to_rg48_compute_blk<store_t>(src, tmp);
                for (unsigned i = 0; i < (size_x - position_x * 8) * 6; ++i) {
                        dst[i] = tmp[i];
                }
                return;
        }
        r12l_to_rg48_compute_blk<store_t>(src, dst);
}

/// adapted variant of @ref vc_copylineR12LtoRG48
template <typename store_t>
__device__ static void
r12l_to_rg48_compute_blk(const uint8_t *in, uint8_t *out)
{
        // load the data from in to src_u32
        auto *in_u32 = (uint32_t *) in;
        uint32_t src_u32[9];
        for (unsigned i = 0; i < sizeof src_u32 / sizeof src_u32[0]; ++i) {
                src_u32[i] = in_u32[i];
        }

        uint32_t dst_u32[12];
        uint8_t *dst = (uint8_t *) dst_u32;
        uint8_t *src = (uint8_t *) src_u32;

        // 0
        // R
        *dst++ = src[0] << 4;
        *dst++ = (src[1] << 4) | (src[0] >> 4);
        // G
        *dst++ = src[1] & 0xF0;
        *dst++ = src[2];
        // B
        *dst++ = src[3] << 4;
        *dst++ = (src[4 + 0] << 4) | (src[3] >> 4);

        // 1
        *dst++ = src[4 + 0] & 0xF0;
        *dst++ = src[4 + 1];

        *dst++ = src[4 + 2] << 4;
        *dst++ = (src[4 + 3] << 4) | (src[4 + 2] >> 4);

        *dst++ = src[4 + 3] & 0xF0;
        *dst++ = src[8 + 0];

        // 2
        *dst++ = src[8 + 1] << 4;
        *dst++ = (src[8 + 2] << 4) | (src[8 + 1] >> 4);

        *dst++ = src[8 + 2] & 0xF0;
        *dst++ = src[8 + 3];

        *dst++ = src[12 + 0] << 4;
        *dst++ = (src[12 + 1] << 4) | (src[12 + 0] >> 4);

        // 3
        *dst++ = src[12 + 1] & 0xF0;
        *dst++ = src[12 + 2];

        *dst++ = src[12 + 3] << 4;
        *dst++ = (src[16 + 0] << 4) | (src[12 + 3] >> 4);

        *dst++ = src[16 + 0] & 0xF0;
        *dst++ = src[16 + 1];

        // 4
        *dst++ = src[16 + 2] << 4;
        *dst++ = (src[16 + 3] << 4) | (src[16 + 2] >> 4);

        *dst++ = src[16 + 3] & 0xF0;
        *dst++ = src[20 + 0];

        *dst++ = src[20 + 1] << 4;
        *dst++ = (src[20 + 2] << 4) | (src[20 + 1] >> 4);

        // 5
        *dst++ = src[20 + 2] & 0xF0;
        *dst++ = src[20 + 3];

        *dst++ = src[24 + 0] << 4;
        *dst++ = (src[24 + 1] << 4) | (src[24 + 0] >> 4);

        *dst++ = src[24 + 1] & 0xF0;
        *dst++ = src[24 + 2];

        // 6
        *dst++ = src[24 + 3] << 4;
        *dst++ = (src[28 + 0] << 4) | (src[24 + 3] >> 4);

        *dst++ = src[28 + 0] & 0xF0;
        *dst++ = src[28 + 1];

        *dst++ = src[28 + 2] << 4;
        *dst++ = (src[28 + 3] << 4) | (src[28 + 2] >> 4);

        // 7
        *dst++ = src[28 + 3] & 0xF0;
        *dst++ = src[32 + 0];

        *dst++ = src[32 + 1] << 4;
        *dst++ = (src[32 + 2] << 4) | (src[32 + 1] >> 4);

        *dst++ = src[32 + 2] & 0xF0;
        *dst++ = src[32 + 3];

        // store the result
        auto *out_t = (store_t *) out;
        for (unsigned i = 0; i < sizeof dst_u32 / sizeof dst_u32[0]; ++i) {
                static_assert(sizeof(store_t) == 2 || sizeof(store_t) == 4,
                              "Just uint{16,32} supported!");
                if (sizeof(store_t) == 4) {
                        out_t[i] = dst_u32[i];
                } else {
                        out_t[2 * i] = dst_u32[i] & 0xFFFFU;
                        out_t[2 * i + 1] = dst_u32[i] >> 16;
                }
        }
}

int
preprocess_r12l_to_rg48(void *preprocessor, void *img_custom_data,
                        size_t img_custom_data_size, int size_x, int size_y,
                        struct cmpto_j2k_enc_comp_format *comp_formats,
                        int comp_count, void *input_samples,
                        size_t input_samples_size, void *output_samples,
                        size_t output_samples_size, void *vstream)
{
        (void) preprocessor, (void) img_custom_data,
            (void) img_custom_data_size, (void) comp_formats, (void) comp_count,
            (void) input_samples_size, (void) output_samples_size;

        cudaStream_t stream = (cudaStream_t) vstream;
        dim3 threads_per_block(256);
        dim3 blocks((((size_x+ 7) / 8) + 255) / 256, size_y);

        MEASURE_KERNEL_DURATION_START(stream)
        if (size_x % 2 == 0) {
                kernel_r12l_to_rg48<uint32_t>
                    <<<blocks, threads_per_block, 0, stream>>>(
                        (uint8_t *) input_samples, (uint8_t *) output_samples,
                        size_x);
        } else {
                thread_local bool warn_print;
                if (!warn_print) {
                        fprintf(stderr,
                                "%s: Odd width %d px will use slower kernel!\n",
                                __func__, size_x);
                        warn_print = true;
                }
                kernel_r12l_to_rg48<uint16_t>
                    <<<blocks, threads_per_block, 0, stream>>>(
                        (uint8_t *) input_samples, (uint8_t *) output_samples,
                        size_x);
        }
        MEASURE_KERNEL_DURATION_STOP(stream)

        return cudaGetLastError();
}
