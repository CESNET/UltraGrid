/**
 * @file   cuda_wrapper/kernels.cu
 * @author Martin Pulec     <pulec@cesnet.cz>
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

#include <cuda_runtime_api.h>
#include <stdint.h>

/// modified vc_copylineRG48toR12L
__global__ void
kernel_rg48_to_r12l(uint8_t *in, uint8_t *out, unsigned size_x, unsigned size_y)
{
        unsigned position_x = threadIdx.x + blockIdx.x * blockDim.x;
        unsigned position_y = threadIdx.y + blockIdx.y * blockDim.y;
        if (position_x > (size_x + 7) / 8) {
                return;
        }
        // drop last block if not aligned (prevent OOB read from input)
        if (position_y == size_y - 1 && position_x > size_x / 8) {
                return;
        }
        uint8_t *src = in + 2 * (position_y * 3 * size_x + position_x * 3 * 8);
        uint8_t *dst =
            out + (position_y * ((size_x + 7) / 8) + position_x) * 36;

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
    void * stream
) {
        dim3 threads_per_block(256);
        dim3 blocks((((size_x + 7) / 8) + 255) / 256, size_y);

        kernel_rg48_to_r12l<<<blocks, threads_per_block, 0,
                              (cudaStream_t) stream>>>(
            (uint8_t *) input_samples, (uint8_t *) output_buffer, size_x,
            size_y);
        return 0;
}
