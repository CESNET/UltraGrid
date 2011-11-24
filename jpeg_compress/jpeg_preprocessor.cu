/**
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
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
 
#include "jpeg_preprocessor.h"
#include "jpeg_util.h"

/**
 * Color space transformation
 *
 * @param color_space_from
 * @param color_space_to
 */
template<enum jpeg_color_space color_space_from, enum jpeg_color_space color_space_to>
struct jpeg_color_transform
{
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        assert(false);
    }
};

/** Specialization [color_space_from = color_space_to] */
template<enum jpeg_color_space color_space>
struct jpeg_color_transform<color_space, color_space> {
    /** None transform */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        // Same color space so do nothing 
    }
};

/** Specialization [color_space_from = JPEG_RGB, color_space_to = JPEG_YCBCR] */
template<>
struct jpeg_color_transform<JPEG_RGB, JPEG_YCBCR> {
    /** RGB -> YCbCr transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        float r1 = 0.299f * c1 + 0.587f * c2 + 0.114f * c3;
        float r2 = -0.1687f * c1 - 0.3313f * c2 + 0.5f * c3 + 128.0f;
        float r3 = 0.5f * c1 - 0.4187f * c2 - 0.0813f * c3 + 128.0f;
        c1 = r1;
        c2 = r2;
        c3 = r3;
    }
};

/** Specialization [color_space_from = JPEG_YUV, color_space_to = JPEG_YCBCR] */
template<>
struct jpeg_color_transform<JPEG_YUV, JPEG_YCBCR> {
    /** YUV -> YCbCr transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        // Do nothing
    }
};

/** Specialization [color_space_from = JPEG_YCBCR, color_space_to = JPEG_RGB] */
template<>
struct jpeg_color_transform<JPEG_YCBCR, JPEG_RGB> {
    /** YCbCr -> RGB transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        // Update values
        float r1 = c1 - 0.0f;
        float r2 = c2 - 128.0f;
        float r3 = c3 - 128.0f;
        // Perfomr YCbCr -> RGB conversion
        c1 = (1.0f * r1 + 0.0f * r2 + 1.402f * r3);
        c2 = (1.0f * r1 - 0.344136f * r2 - 0.714136f * r3);
        c3 = (1.0f * r1 + 1.772f * r2 + 0.0f * r3);
        // Check minimum value 0
        c1 = (c1 >= 0.0f) ? c1 : 0.0f;
        c2 = (c2 >= 0.0f) ? c2 : 0.0f;
        c3 = (c3 >= 0.0f) ? c3 : 0.0f;
        // Check maximum value 255
        c1 = (c1 <= 255.0) ? c1 : 255.0f;
        c2 = (c2 <= 255.0) ? c2 : 255.0f;
        c3 = (c3 <= 255.0) ? c3 : 255.0f;    
    }
};

/** Specialization [color_space_from = JPEG_YCBCR, color_space_to = JPEG_YUV] */
template<>
struct jpeg_color_transform<JPEG_YCBCR, JPEG_YUV> {
    /** YCbCr -> YUV transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        // Do nothing
    }
};

#define RGB_8BIT_THREADS 256

/**
 * Kernel - Copy raw image source data into three separated component buffers
 *
 * @param d_c1  First component buffer
 * @param d_c2  Second component buffer
 * @param d_c3  Third component buffer
 * @param d_source  Image source data
 * @param pixel_count  Number of pixels to copy
 * @return void
 */
typedef void (*jpeg_preprocessor_encode_kernel)(uint8_t* d_c1, uint8_t* d_c2, uint8_t* d_c3, const uint8_t* d_source, int pixel_count);
 
/** Specialization [sampling factor is 4:4:4] */
template<enum jpeg_color_space color_space>
__global__ void 
jpeg_preprocessor_raw_to_comp_kernel_4_4_4(uint8_t* d_c1, uint8_t* d_c2, uint8_t* d_c3, const uint8_t* d_source, int pixel_count)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
        
    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 3];
    if ( (x * 4) < RGB_8BIT_THREADS * 3 ) {
        int* s = (int*)d_source;
        int* d = (int*)s_data;
        d[x] = s[((gX * 3) >> 2) + x];
    }
    __syncthreads();

    // Load
    int offset = x * 3;
    float r1 = (float)(s_data[offset]);
    float r2 = (float)(s_data[offset + 1]);
    float r3 = (float)(s_data[offset + 2]);
    // Color transform
    jpeg_color_transform<color_space, JPEG_YCBCR>::perform(r1, r2, r3);
    // Store
    int globalOutputPosition = gX + x;
    if ( globalOutputPosition < pixel_count ) {
        d_c1[globalOutputPosition] = (uint8_t)r1;
        d_c2[globalOutputPosition] = (uint8_t)r2;
        d_c3[globalOutputPosition] = (uint8_t)r3;
    }
}

/** Specialization [sampling factor is 4:2:2] */
template<enum jpeg_color_space color_space>
__global__ void 
jpeg_preprocessor_raw_to_comp_kernel_4_2_2(uint8_t* d_c1, uint8_t* d_c2, uint8_t* d_c3, const uint8_t* d_source, int pixel_count)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
        
    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 2];
    if ( (x * 4) < RGB_8BIT_THREADS * 2 ) {
        int* s = (int*)d_source;
        int* d = (int*)s_data;
        d[x] = s[((gX * 2) >> 2) + x];
    }
    __syncthreads();

    // Load
    int offset = x * 2;
    float r1 = (float)(s_data[offset + 1]);
    float r2;
    float r3;
    if ( (gX + x) % 2 == 0 ) {
        r2 = (float)(s_data[offset]);
        r3 = (float)(s_data[offset + 2]);
    } else {
        r2 = (float)(s_data[offset - 2]);
        r3 = (float)(s_data[offset]);
    }
    // Color transform
    jpeg_color_transform<color_space, JPEG_YCBCR>::perform(r1, r2, r3);
    // Store
    int globalOutputPosition = gX + x;
    if ( globalOutputPosition < pixel_count ) {
        d_c1[globalOutputPosition] = (uint8_t)r1;
        d_c2[globalOutputPosition] = (uint8_t)r2;
        d_c3[globalOutputPosition] = (uint8_t)r3;
    }
}

/**
 * Select preprocessor encode kernel
 * 
 * @param encoder
 * @return kernel
 */
jpeg_preprocessor_encode_kernel
jpeg_preprocessor_select_encode_kernel(struct jpeg_encoder* encoder)
{
    // RGB color space
    if ( encoder->param_image.color_space == JPEG_RGB ) {
        assert(encoder->param_image.sampling_factor == JPEG_4_4_4);
        return &jpeg_preprocessor_raw_to_comp_kernel_4_4_4<JPEG_RGB>;
    } 
    // YUV color space
    else if ( encoder->param_image.color_space == JPEG_YUV ) {
        if ( encoder->param_image.sampling_factor == JPEG_4_4_4 ) {
            return &jpeg_preprocessor_raw_to_comp_kernel_4_4_4<JPEG_YUV>;
        } else if ( encoder->param_image.sampling_factor == JPEG_4_2_2 ) {
            return &jpeg_preprocessor_raw_to_comp_kernel_4_2_2<JPEG_YUV>;
        } else {
            assert(false);
        }
    }
    // Unknown color space
    else {
        assert(false);
    }
    return NULL;
}

/** Documented at declaration */
int
jpeg_preprocessor_encode(struct jpeg_encoder* encoder)
{        
    int pixel_count = encoder->param_image.width * encoder->param_image.height;
    int alignedSize = (pixel_count / RGB_8BIT_THREADS + 1) * RGB_8BIT_THREADS * 3;

    // Select kernel
    jpeg_preprocessor_encode_kernel kernel = jpeg_preprocessor_select_encode_kernel(encoder);
    
    // Prepare kernel
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * 3));
    assert(alignedSize % (RGB_8BIT_THREADS * 3) == 0);

    // Run kernel
    uint8_t* d_c1 = &encoder->d_data[0 * pixel_count];
    uint8_t* d_c2 = &encoder->d_data[1 * pixel_count];
    uint8_t* d_c3 = &encoder->d_data[2 * pixel_count];
    kernel<<<grid, threads>>>(d_c1, d_c2, d_c3, encoder->d_data_source, pixel_count);
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Preprocessor encoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
        
    return 0;
}

/**
 * Kernel - Copy three separated component buffers into target image data
 *
 * @param d_c1  First component buffer
 * @param d_c2  Second component buffer
 * @param d_c3  Third component buffer
 * @param d_target  Image target data
 * @param pixel_count  Number of pixels to copy
 * @return void
 */
typedef void (*jpeg_preprocessor_decode_kernel)(const uint8_t* d_c1, const uint8_t* d_c2, const uint8_t* d_c3, uint8_t* d_target, int pixel_count);

/** Specialization [sampling factor is 4:4:4] */
template<enum jpeg_color_space color_space>
__global__ void
jpeg_preprocessor_comp_to_raw_kernel_4_4_4(const uint8_t* d_c1, const uint8_t* d_c2, const uint8_t* d_c3, uint8_t* d_target, int pixel_count)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
    int globalInputPosition = gX + x;
    if ( globalInputPosition >= pixel_count )
        return;
    int globalOutputPosition = (gX + x) * 3;
    
    // Load
    float r1 = (float)(d_c1[globalInputPosition]);
    float r2 = (float)(d_c2[globalInputPosition]);
    float r3 = (float)(d_c3[globalInputPosition]);
    // Color transform
    jpeg_color_transform<JPEG_YCBCR, color_space>::perform(r1, r2, r3);
    // Save
    d_target[globalOutputPosition + 0] = (uint8_t)r1;
    d_target[globalOutputPosition + 1] = (uint8_t)r2;
    d_target[globalOutputPosition + 2] = (uint8_t)r3;
}

/**
 * Select preprocessor decode kernel
 * 
 * @param decoder
 * @return kernel
 */
jpeg_preprocessor_decode_kernel
jpeg_preprocessor_select_decode_kernel(struct jpeg_decoder* decoder)
{
    // RGB color space
    if ( decoder->param_image.color_space == JPEG_RGB ) {
        assert(decoder->param_image.sampling_factor == JPEG_4_4_4);
        return &jpeg_preprocessor_comp_to_raw_kernel_4_4_4<JPEG_RGB>;
    } 
    // YUV color space
    else if ( decoder->param_image.color_space == JPEG_YUV ) {
        assert(decoder->param_image.sampling_factor == JPEG_4_4_4);
        return &jpeg_preprocessor_comp_to_raw_kernel_4_4_4<JPEG_YUV>;
    }
    // Unknown color space
    else {
        assert(false);
    }
    return NULL;
}

/** Documented at declaration */
int
jpeg_preprocessor_decode(struct jpeg_decoder* decoder)
{
    int pixel_count = decoder->param_image.width * decoder->param_image.height;
    int alignedSize = (pixel_count / RGB_8BIT_THREADS + 1) * RGB_8BIT_THREADS * 3;
        
    // Select kernel
    jpeg_preprocessor_decode_kernel kernel = jpeg_preprocessor_select_decode_kernel(decoder);
    
    // Prepare kernel
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * 3));
    assert(alignedSize % (RGB_8BIT_THREADS * 3) == 0);

    // Run kernel
    uint8_t* d_c1 = &decoder->d_data[0 * pixel_count];
    uint8_t* d_c2 = &decoder->d_data[1 * pixel_count];
    uint8_t* d_c3 = &decoder->d_data[2 * pixel_count];
    kernel<<<grid, threads>>>(d_c1, d_c2, d_c3, decoder->d_data_target, pixel_count);
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Preprocessing decoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
    
    return 0;
}