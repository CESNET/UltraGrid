/**
 * Copyright (c) 2011, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_util.h"

/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/** Fast integer multiplication */
#define FMUL(x,y)   (__mul24(x,y))
//#define FMUL(x,y)   ((x)*(y))

// X block count which will be processed by one thread block
#define GPUJPEG_DCT_BLOCK_COUNT_X       4
// Y block count which will be processed by one thread block
#define GPUJPEG_DCT_BLOCK_COUNT_Y       4

// Thread block width
#define GPUJPEG_DCT_THREAD_BLOCK_WIDTH  (GPUJPEG_BLOCK_SIZE * GPUJPEG_DCT_BLOCK_COUNT_X)
// Thread block height
#define GPUJPEG_DCT_THREAD_BLOCK_HEIGHT (GPUJPEG_BLOCK_SIZE * GPUJPEG_DCT_BLOCK_COUNT_Y)

// Stride of shared memory buffer (short kernel)
#define GPUJPEG_DCT_THREAD_BLOCK_STRIDE (GPUJPEG_DCT_THREAD_BLOCK_WIDTH + 4)

#define IMAD(a, b, c) ( ((a) * (b)) + (c) )
#define IMUL(a, b) ((a) * (b))

#define SIN_1_4     0x5A82
#define COS_1_4     0x5A82
#define SIN_1_8     0x30FC
#define COS_1_8     0x7642

#define OSIN_1_16   0x063E
#define OSIN_3_16   0x11C7
#define OSIN_5_16   0x1A9B
#define OSIN_7_16   0x1F63

#define OCOS_1_16   0x1F63
#define OCOS_3_16   0x1A9B
#define OCOS_5_16   0x11C7
#define OCOS_7_16   0x063E

/**
 * Package of 2 shorts into 1 int - designed to perform i/o by integers to avoid bank conflicts
 */
union PackedInteger
{
    struct __align__(8)
    {
        int16_t hShort1;
        int16_t hShort2;
    };
    int32_t hInt;
};

/**
 * Converts fixed point value to short value
 */
__device__ inline int16_t
unfixh(int x)
{
    return (int16_t)((x + 0x8000) >> 16);
}

/**
 * Converts fixed point value to short value
 */
__device__ inline int
unfixo(int x)
{
    return (x + 0x1000) >> 13;
}

/**
 * Performs in-place DCT of vector of 8 elements (used to access columns in shared memory).
 *
 * @param SrcDst [IN/OUT] - Pointer to the first element of vector
 * @param Stride [IN] - Value to add to ptr to access other elements
 * @return None
 */
__device__ void
gpujpeg_dct_gpu_kernel_inplace(int16_t* SrcDst, int Stride)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp14, tmp15, tmp16, tmp17;
    int tmp25, tmp26;

    int DoubleStride = Stride << 1;

    int16_t* DstPtr = SrcDst;
    in0 = *DstPtr;
    DstPtr += Stride;
    in1 = *DstPtr;
    DstPtr += Stride;
    in2 = *DstPtr;
    DstPtr += Stride;
    in3 = *DstPtr;
    DstPtr += Stride;
    in4 = *DstPtr;
    DstPtr += Stride;
    in5 = *DstPtr;
    DstPtr += Stride;
    in6 = *DstPtr;
    DstPtr += Stride;
    in7 = *DstPtr;

    tmp0 = in7 + in0;
    tmp1 = in6 + in1;
    tmp2 = in5 + in2;
    tmp3 = in4 + in3;
    tmp4 = in3 - in4;
    tmp5 = in2 - in5;
    tmp6 = in1 - in6;
    tmp7 = in0 - in7;

    tmp10 = tmp3 + tmp0;
    tmp11 = tmp2 + tmp1;
    tmp12 = tmp1 - tmp2;
    tmp13 = tmp0 - tmp3;

    tmp16 = unfixo(FMUL(tmp6 + tmp5, SIN_1_4));
    tmp15 = unfixo(FMUL(tmp6 - tmp5, COS_1_4));

    tmp4 <<= 2;
    tmp7 <<= 2;

    tmp14 = tmp4 + tmp15;
    tmp25 = tmp4 - tmp15;
    tmp26 = tmp7 - tmp16;
    tmp17 = tmp7 + tmp16;

    DstPtr = SrcDst;
    *DstPtr = unfixh(FMUL(tmp10 + tmp11, SIN_1_4));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp13, COS_1_8) + FMUL(tmp12, SIN_1_8));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp10 - tmp11, COS_1_4));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp13, SIN_1_8) - FMUL(tmp12, COS_1_8));

    DstPtr = SrcDst + Stride;
    *DstPtr = unfixh(FMUL(tmp17, OCOS_1_16) + FMUL(tmp14, OSIN_1_16));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp26, OCOS_3_16) - FMUL(tmp25, OSIN_3_16));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp26, OCOS_5_16) + FMUL(tmp25, OSIN_5_16));
    DstPtr += DoubleStride;
    *DstPtr = unfixh(FMUL(tmp17, OCOS_7_16) - FMUL(tmp14, OSIN_7_16));
}

/**
 * Performs in-place DCT of vector of 8 elements (used to access rows in shared memory).
 *
 * @param V8 [IN/OUT] - Pointer to the first two elements of vector
 * @return None
 */
__device__ void
gpujpeg_dct_gpu_kernel_inplace(uint32_t* V8)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp0, tmp1, tmp2, tmp3, tmp4, tmp5, tmp6, tmp7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp14, tmp15, tmp16, tmp17;
    int tmp25, tmp26;
    PackedInteger sh0, sh1, sh2, sh3;

    sh0.hInt = V8[0];
    sh1.hInt = V8[1];
    sh2.hInt = V8[2];
    sh3.hInt = V8[3];
    in0 = sh0.hShort1;
    in1 = sh0.hShort2;
    in2 = sh1.hShort1;
    in3 = sh1.hShort2;
    in4 = sh2.hShort1;
    in5 = sh2.hShort2;
    in6 = sh3.hShort1;
    in7 = sh3.hShort2;

    tmp0 = in7 + in0;
    tmp1 = in6 + in1;
    tmp2 = in5 + in2;
    tmp3 = in4 + in3;
    tmp4 = in3 - in4;
    tmp5 = in2 - in5;
    tmp6 = in1 - in6;
    tmp7 = in0 - in7;

    tmp10 = tmp3 + tmp0;
    tmp11 = tmp2 + tmp1;
    tmp12 = tmp1 - tmp2;
    tmp13 = tmp0 - tmp3;

    sh0.hShort1 = unfixh(FMUL(tmp10 + tmp11, SIN_1_4));
    sh2.hShort1 = unfixh(FMUL(tmp10 - tmp11, COS_1_4));

    sh1.hShort1 = unfixh(FMUL(tmp13, COS_1_8) + FMUL(tmp12, SIN_1_8));
    sh3.hShort1 = unfixh(FMUL(tmp13, SIN_1_8) - FMUL(tmp12, COS_1_8));

    tmp16 = unfixo(FMUL(tmp6 + tmp5, SIN_1_4));
    tmp15 = unfixo(FMUL(tmp6 - tmp5, COS_1_4));

    tmp4 <<= 2;
    tmp7 <<= 2;

    tmp14 = tmp4 + tmp15;
    tmp25 = tmp4 - tmp15;
    tmp26 = tmp7 - tmp16;
    tmp17 = tmp7 + tmp16;

    sh0.hShort2 = unfixh(FMUL(tmp17, OCOS_1_16) + FMUL(tmp14, OSIN_1_16));
    sh3.hShort2 = unfixh(FMUL(tmp17, OCOS_7_16) - FMUL(tmp14, OSIN_7_16));
    sh2.hShort2 = unfixh(FMUL(tmp26, OCOS_5_16) + FMUL(tmp25, OSIN_5_16));
    sh1.hShort2 = unfixh(FMUL(tmp26, OCOS_3_16) - FMUL(tmp25, OSIN_3_16));

    V8[0] = sh0.hInt;
    V8[1] = sh1.hInt;
    V8[2] = sh2.hInt;
    V8[3] = sh3.hInt;
}

/**
 * Performs in-place IDCT of vector of 8 elements (used to access columns in shared memory).
 *
 * @param SrcDst [IN/OUT] - Pointer to the first element of vector
 * @param Stride [IN] - Value to add to ptr to access other elements
 * @return None
 */
__device__ void
gpujpeg_idct_gpu_kernel_inplace(int16_t* SrcDst, int Stride)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp20, tmp21, tmp22, tmp23;
    int tmp30, tmp31;
    int tmp40, tmp41, tmp42, tmp43;
    int tmp50, tmp51, tmp52, tmp53;

    int16_t *DstPtr = SrcDst;
    in0 = *DstPtr;
    DstPtr += Stride;
    in1 = *DstPtr;
    DstPtr += Stride;
    in2 = *DstPtr;
    DstPtr += Stride;
    in3 = *DstPtr;
    DstPtr += Stride;
    in4 = *DstPtr;
    DstPtr += Stride;
    in5 = *DstPtr;
    DstPtr += Stride;
    in6 = *DstPtr;
    DstPtr += Stride;
    in7 = *DstPtr;

    tmp10 = FMUL(in0 + in4, COS_1_4);
    tmp11 = FMUL(in0 - in4, COS_1_4);
    tmp12 = FMUL(in2, SIN_1_8) - FMUL(in6, COS_1_8);
    tmp13 = FMUL(in6, SIN_1_8) + FMUL(in2, COS_1_8);

    tmp20 = tmp10 + tmp13;
    tmp21 = tmp11 + tmp12;
    tmp22 = tmp11 - tmp12;
    tmp23 = tmp10 - tmp13;

    tmp30 = unfixo(FMUL(in3 + in5, COS_1_4));
    tmp31 = unfixo(FMUL(in3 - in5, COS_1_4));

    in1 <<= 2;
    in7 <<= 2;

    tmp40 = in1 + tmp30;
    tmp41 = in7 + tmp31;
    tmp42 = in1 - tmp30;
    tmp43 = in7 - tmp31;

    tmp50 = FMUL(tmp40, OCOS_1_16) + FMUL(tmp41, OSIN_1_16);
    tmp51 = FMUL(tmp40, OSIN_1_16) - FMUL(tmp41, OCOS_1_16);
    tmp52 = FMUL(tmp42, OCOS_5_16) + FMUL(tmp43, OSIN_5_16);
    tmp53 = FMUL(tmp42, OSIN_5_16) - FMUL(tmp43, OCOS_5_16);

    DstPtr = SrcDst;
    *DstPtr = unfixh(tmp20 + tmp50);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp21 + tmp53);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp22 + tmp52);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp23 + tmp51);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp23 - tmp51);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp22 - tmp52);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp21 - tmp53);
    DstPtr += Stride;
    *DstPtr = unfixh(tmp20 - tmp50);
}

/**
 * Performs in-place IDCT of vector of 8 elements (used to access rows in shared memory).
 *
 * @param V8 [IN/OUT] - Pointer to the first two elements of vector
 * @return None
 */
__device__ void
gpujpeg_idct_gpu_kernel_inplace(uint32_t* V8)
{
    int in0, in1, in2, in3, in4, in5, in6, in7;
    int tmp10, tmp11, tmp12, tmp13;
    int tmp20, tmp21, tmp22, tmp23;
    int tmp30, tmp31;
    int tmp40, tmp41, tmp42, tmp43;
    int tmp50, tmp51, tmp52, tmp53;
    PackedInteger sh0, sh1, sh2, sh3;

    sh0.hInt = V8[0];
    sh1.hInt = V8[1];
    sh2.hInt = V8[2];
    sh3.hInt = V8[3];
    in0 = sh0.hShort1;
    in1 = sh0.hShort2;
    in2 = sh1.hShort1;
    in3 = sh1.hShort2;
    in4 = sh2.hShort1;
    in5 = sh2.hShort2;
    in6 = sh3.hShort1;
    in7 = sh3.hShort2;

    tmp10 = FMUL(in0 + in4, COS_1_4);
    tmp11 = FMUL(in0 - in4, COS_1_4);
    tmp12 = FMUL(in2, SIN_1_8) - FMUL(in6, COS_1_8);
    tmp13 = FMUL(in6, SIN_1_8) + FMUL(in2, COS_1_8);

    tmp20 = tmp10 + tmp13;
    tmp21 = tmp11 + tmp12;
    tmp22 = tmp11 - tmp12;
    tmp23 = tmp10 - tmp13;

    tmp30 = unfixo(FMUL(in3 + in5, COS_1_4));
    tmp31 = unfixo(FMUL(in3 - in5, COS_1_4));

    in1 <<= 2;
    in7 <<= 2;

    tmp40 = in1 + tmp30;
    tmp41 = in7 + tmp31;
    tmp42 = in1 - tmp30;
    tmp43 = in7 - tmp31;

    tmp50 = FMUL(tmp40, OCOS_1_16) + FMUL(tmp41, OSIN_1_16);
    tmp51 = FMUL(tmp40, OSIN_1_16) - FMUL(tmp41, OCOS_1_16);
    tmp52 = FMUL(tmp42, OCOS_5_16) + FMUL(tmp43, OSIN_5_16);
    tmp53 = FMUL(tmp42, OSIN_5_16) - FMUL(tmp43, OCOS_5_16);

    sh0.hShort1 = unfixh(tmp20 + tmp50);
    sh0.hShort2 = unfixh(tmp21 + tmp53);
    sh1.hShort1 = unfixh(tmp22 + tmp52);
    sh1.hShort2 = unfixh(tmp23 + tmp51);
    sh2.hShort1 = unfixh(tmp23 - tmp51);
    sh2.hShort2 = unfixh(tmp22 - tmp52);
    sh3.hShort1 = unfixh(tmp21 - tmp53);
    sh3.hShort2 = unfixh(tmp20 - tmp50);

    V8[0] = sh0.hInt;
    V8[1] = sh1.hInt;
    V8[2] = sh2.hInt;
    V8[3] = sh3.hInt;
}

/** Quantization table */
__constant__ uint16_t gpujpeg_dct_gpu_quantization_table[64];

/**
 * Performs 8x8 block-wise Forward Discrete Cosine Transform of the given
 * image plane and outputs result to the array of coefficients. Short implementation.
 * This kernel is designed to process image by blocks of blocks8x8 that
 * utilize maximum warps capacity, assuming that it is enough of 8 threads
 * per block8x8.
 *
 * @param source        [IN]  - Source coefficients
 * @param source_stride [IN]  - Stride of source
 * @param output        [OUT] - Source coefficients
 * @param output_stride [OUT] - Stride of source
 * @param table         [IN]  - Quantization table
 * @return None
 */
__global__ void
gpujpeg_dct_gpu_kernel(int block_count_x, int block_count_y, uint8_t* source, int source_stride,
                       int16_t* output, int output_stride, uint16_t* quantization_table)
{
// For pre-fermi GPUs, quantization table in constant memory is faster
#if __CUDA_ARCH__ < 200
    quantization_table = gpujpeg_dct_gpu_quantization_table;
#endif
    
    // Shared data
    __shared__ int16_t block[GPUJPEG_DCT_THREAD_BLOCK_HEIGHT * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];

    // Block position
    int block_x = IMAD(blockIdx.x, GPUJPEG_DCT_BLOCK_COUNT_X, threadIdx.y);
    int block_y = IMAD(blockIdx.y, GPUJPEG_DCT_BLOCK_COUNT_Y, threadIdx.z);

    // Thread position in thread block
    int thread_x = IMAD(threadIdx.y, GPUJPEG_BLOCK_SIZE, threadIdx.x);
    int thread_y = IMUL(threadIdx.z, GPUJPEG_BLOCK_SIZE);
    int thread_x_permutated = (thread_x & 0xFFFFFFE0) | (((thread_x << 1) | ((thread_x >> 4) & 0x1)) & 0x1F);

    // Determine position into shared buffer
    int16_t* block_ptr = block + IMAD(thread_y, GPUJPEG_DCT_THREAD_BLOCK_STRIDE, thread_x);

    // Determine position in source buffer and apply it
    int source_x = IMAD(block_x, GPUJPEG_BLOCK_SIZE, threadIdx.x);
    int source_y = IMUL(block_y, GPUJPEG_BLOCK_SIZE);
    source += IMAD(source_y, source_stride, source_x);

    // Load data to shared memory memory
    if ( block_x < block_count_x && block_y < block_count_y ) {
        
// For pre-fermi GPUs, loading from global memory by 4 bytes is faster
#if __CUDA_ARCH__ < 200
        __shared__ uint8_t block_byte[GPUJPEG_DCT_THREAD_BLOCK_HEIGHT * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];
        uint8_t* block_byte_ptr = block_byte + IMAD(thread_y, GPUJPEG_DCT_THREAD_BLOCK_STRIDE, thread_x);
        if ( threadIdx.x % 4 == 0 ) {
            #pragma unroll
            for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++)
                ((uint32_t*)block_byte_ptr)[i * (GPUJPEG_DCT_THREAD_BLOCK_STRIDE / 4)] = ((uint32_t*)source)[i * (source_stride / 4)];
        }
        source = block_byte_ptr;
        source_stride = GPUJPEG_DCT_THREAD_BLOCK_STRIDE;
#endif
    
        #pragma unroll
        for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++) {
            int16_t coefficient = (int16_t)(source[i * source_stride]);
            coefficient -= 128;
            block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE] = coefficient;
        }
    }

    // Perform DCT
    __syncthreads();
    gpujpeg_dct_gpu_kernel_inplace(block + thread_y * GPUJPEG_DCT_THREAD_BLOCK_STRIDE + thread_x_permutated, GPUJPEG_DCT_THREAD_BLOCK_STRIDE);
    __syncthreads();
    gpujpeg_dct_gpu_kernel_inplace((uint32_t*)(block + (thread_y + threadIdx.x) * GPUJPEG_DCT_THREAD_BLOCK_STRIDE + threadIdx.y * GPUJPEG_BLOCK_SIZE));
    __syncthreads();

    // Quantization
    for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++) {
        uint16_t quantization = quantization_table[i * GPUJPEG_BLOCK_SIZE + threadIdx.x];
        int coefficient = block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];

        if ( coefficient < 0 ) {
            coefficient = -coefficient;
            coefficient = (coefficient * quantization + 16384) / 32767;
            coefficient = -coefficient;
        } else {
            coefficient = (coefficient * quantization + 16384) / 32767;
        }

        block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE] = coefficient;
    }
    __syncthreads();

    // Determine position in output buffer and apply it
    int output_x = IMAD(IMAD(blockIdx.x, GPUJPEG_DCT_BLOCK_COUNT_X, threadIdx.y), GPUJPEG_BLOCK_SQUARED_SIZE, threadIdx.x * 2);
    int output_y = IMAD(blockIdx.y, GPUJPEG_DCT_BLOCK_COUNT_Y, threadIdx.z);
    output += IMAD(output_y, output_stride, output_x);

    // Store data to global memory, only half of threads in each cell performs data moving (each thread moves 2 shorts)
    int16_t* block_store_ptr = block_ptr + threadIdx.x; // Shortcut for "IMAD(..., threadIdx.x * 2)"
    if ( threadIdx.x < (GPUJPEG_BLOCK_SIZE / 2) && block_x < block_count_x && block_y < block_count_y ) {
        #pragma unroll
        for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++)
            ((int*)output)[i * (GPUJPEG_BLOCK_SIZE / 2)] = ((int*)block_store_ptr)[i * (GPUJPEG_DCT_THREAD_BLOCK_STRIDE / 2)];
    }
}

/** Quantization table */
__constant__ uint16_t gpujpeg_idct_gpu_quantization_table[64];

/**
 * Performs 8x8 block-wise Inverse Discrete Cosine Transform of the given
 * image plane and outputs result to the array of coefficients. Short implementation.
 * This kernel is designed to process image by blocks of blocks8x8 that
 * utilize maximum warps capacity, assuming that it is enough of 8 threads
 * per block8x8.
 *
 * @param source        [IN]  - Source coefficients
 * @param source_stride [IN]  - Stride of source
 * @param output        [OUT] - Source coefficients
 * @param output_stride [OUT] - Stride of source
 * @param table         [IN]  - Quantization table
 * @return None
 */
__global__ void
gpujpeg_idct_gpu_kernel(int block_count_x, int block_count_y, int16_t* source, int source_stride,
                        uint8_t* output, int output_stride, uint16_t* quantization_table)
{
// For pre-fermi GPUs, quantization table in constant memory is faster
#if __CUDA_ARCH__ < 200
    quantization_table = gpujpeg_idct_gpu_quantization_table;
#endif
    
    // Shared data
    __shared__ int16_t block[GPUJPEG_DCT_THREAD_BLOCK_HEIGHT * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];

    // Block position
    int block_x = IMAD(blockIdx.x, GPUJPEG_DCT_BLOCK_COUNT_X, threadIdx.y);
    int block_y = IMAD(blockIdx.y, GPUJPEG_DCT_BLOCK_COUNT_Y, threadIdx.z);

    // Thread position in thread block
    int thread_x = IMAD(threadIdx.y, GPUJPEG_BLOCK_SIZE, threadIdx.x);
    int thread_y = IMUL(threadIdx.z, GPUJPEG_BLOCK_SIZE);
    int thread_x_permutated = (thread_x & 0xFFFFFFE0) | (((thread_x << 1) | ((thread_x >> 4) & 0x1)) & 0x1F);

    // Determine position into shared buffer
    int16_t* block_ptr = block + IMAD(thread_y, GPUJPEG_DCT_THREAD_BLOCK_STRIDE, thread_x);

    // Determine position in source buffer and apply it    
    int source_x = IMAD(block_x, GPUJPEG_BLOCK_SQUARED_SIZE, threadIdx.x * 2);
    int source_y = block_y;
    source += IMAD(source_y, source_stride, source_x);

    // Load data to shared memory, only half of threads in each cell performs data moving (each thread moves 2 shorts)
    if ( block_x < block_count_x && block_y < block_count_y ) {
        int16_t* block_load_ptr = block_ptr + threadIdx.x; // Shortcut for "IMAD(..., threadIdx.x * 2)"
        if ( threadIdx.x < (GPUJPEG_BLOCK_SIZE / 2) ) {
            #pragma unroll
            for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++)
                ((int*)block_load_ptr)[i * (GPUJPEG_DCT_THREAD_BLOCK_STRIDE / 2)] = ((int*)source)[i * (GPUJPEG_BLOCK_SIZE / 2)];
        }
    }
    __syncthreads();

    // Quantization
    for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++) {
        int16_t quantization = quantization_table[i * GPUJPEG_BLOCK_SIZE + threadIdx.x];
        int16_t coefficient = block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];

        coefficient = coefficient * quantization;

        block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE] = coefficient;
    }

    // Perform IDCT
    __syncthreads();
    gpujpeg_idct_gpu_kernel_inplace(block + thread_y * GPUJPEG_DCT_THREAD_BLOCK_STRIDE + thread_x_permutated, GPUJPEG_DCT_THREAD_BLOCK_STRIDE);
    __syncthreads();
    gpujpeg_idct_gpu_kernel_inplace((uint32_t*)(block + (thread_y + threadIdx.x) * GPUJPEG_DCT_THREAD_BLOCK_STRIDE + threadIdx.y * GPUJPEG_BLOCK_SIZE));
    __syncthreads();

     // Determine position in output buffer and apply it
    int output_x = IMAD(blockIdx.x, GPUJPEG_DCT_THREAD_BLOCK_WIDTH, thread_x);
    int output_y = IMAD(blockIdx.y, GPUJPEG_DCT_THREAD_BLOCK_HEIGHT, thread_y);
    output += IMAD(output_y, output_stride, output_x);

// For pre-fermi GPUs, storing to global memory by 4 bytes is faster
#if __CUDA_ARCH__ < 200
    __shared__ uint8_t block_byte[GPUJPEG_DCT_THREAD_BLOCK_HEIGHT * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];
    uint8_t* block_byte_ptr = block_byte + IMAD(thread_y, GPUJPEG_DCT_THREAD_BLOCK_STRIDE, thread_x);
    uint8_t* __output = output;
    int __output_stride = output_stride;
    output = block_byte_ptr;
    output_stride = GPUJPEG_DCT_THREAD_BLOCK_STRIDE;
#endif

    // Store data to global memory
    if ( block_x < block_count_x && block_y < block_count_y ) {
        #pragma unroll
        for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++) {
            int16_t coefficient = block_ptr[i * GPUJPEG_DCT_THREAD_BLOCK_STRIDE];
            coefficient += 128;
            if ( coefficient > 255 )
                coefficient = 255;
            if ( coefficient < 0 )
                coefficient = 0;
            output[i * output_stride] = (uint8_t)coefficient;
        }
        
// For pre-fermi GPUs, storing to global memory by 4 bytes is faster
#if __CUDA_ARCH__ < 200
        if ( threadIdx.x % 4 == 0 ) {
            #pragma unroll
            for(int i = 0; i < GPUJPEG_BLOCK_SIZE; i++)
                ((uint32_t*)__output)[i * (__output_stride / 4)] = ((uint32_t*)block_byte_ptr)[i * (GPUJPEG_DCT_THREAD_BLOCK_STRIDE / 4)];
        }
#endif
    }
}

/** Documented at declaration */
void
gpujpeg_dct_gpu(struct gpujpeg_encoder* encoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;

    // Encode each component
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        // Determine table type
        enum gpujpeg_component_type type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;

        int roi_width = component->data_width;
        int roi_height = component->data_height;
        assert(GPUJPEG_BLOCK_SIZE == 8);

        int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
        int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;
        
        // Get quantization table
        uint16_t* d_quantization_table = encoder->table_quantization[type].d_table;
        
        // Copy quantization table to constant memory
        cudaMemcpyToSymbol(
            (const char*)gpujpeg_dct_gpu_quantization_table,
            d_quantization_table, 
            64 * sizeof(uint16_t),
            0,
            cudaMemcpyDeviceToDevice
        );
        gpujpeg_cuda_check_error("Copy DCT quantization table to constant memory");

        // Perform block-wise DCT processing
        dim3 dct_grid(
            gpujpeg_div_and_round_up(block_count_x, GPUJPEG_DCT_BLOCK_COUNT_X),
            gpujpeg_div_and_round_up(block_count_y, GPUJPEG_DCT_BLOCK_COUNT_Y),
            1
        );
        dim3 dct_block(
            GPUJPEG_BLOCK_SIZE,
            GPUJPEG_DCT_BLOCK_COUNT_X,
            GPUJPEG_DCT_BLOCK_COUNT_Y
        );
        gpujpeg_dct_gpu_kernel<<<dct_grid, dct_block>>>(
            block_count_x,
            block_count_y,
            component->d_data,
            component->data_width,
            component->d_data_quantized,
            component->data_width * GPUJPEG_BLOCK_SIZE,
            d_quantization_table
        );
        cudaThreadSynchronize();
        gpujpeg_cuda_check_error("Forward Integer DCT failed");
    }
}

/** Documented at declaration */
void
gpujpeg_idct_gpu(struct gpujpeg_decoder* decoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;

    // Encode each component
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];

        // Determine table type
        enum gpujpeg_component_type type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;

        int roi_width = component->data_width;
        int roi_height = component->data_height;
        assert(GPUJPEG_BLOCK_SIZE == 8);

        int block_count_x = roi_width / GPUJPEG_BLOCK_SIZE;
        int block_count_y = roi_height / GPUJPEG_BLOCK_SIZE;
        
        // Get quantization table
        uint16_t* d_quantization_table = decoder->table_quantization[type].d_table;
        
        // Copy quantization table to constant memory
        cudaMemcpyToSymbol(
            (const char*)gpujpeg_idct_gpu_quantization_table,
            d_quantization_table, 
            64 * sizeof(uint16_t),
            0,
            cudaMemcpyDeviceToDevice
        );
        gpujpeg_cuda_check_error("Copy IDCT quantization table to constant memory");

        // Perform block-wise IDCT processing
        dim3 dct_grid(
            gpujpeg_div_and_round_up(block_count_x, GPUJPEG_DCT_BLOCK_COUNT_X),
            gpujpeg_div_and_round_up(block_count_y, GPUJPEG_DCT_BLOCK_COUNT_Y),
            1
        );
        dim3 dct_block(
            GPUJPEG_BLOCK_SIZE,
            GPUJPEG_DCT_BLOCK_COUNT_X,
            GPUJPEG_DCT_BLOCK_COUNT_Y
        );
        gpujpeg_idct_gpu_kernel<<<dct_grid, dct_block>>>(
            block_count_x,
            block_count_y,
            component->d_data_quantized,
            component->data_width * GPUJPEG_BLOCK_SIZE,
            component->d_data,
            component->data_width,
            d_quantization_table
        );
        cudaThreadSynchronize();
        gpujpeg_cuda_check_error("Inverse Integer DCT failed");
    }
}
