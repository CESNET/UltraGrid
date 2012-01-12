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
 
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_util.h"

/**
 * Color space transformation
 *
 * @param color_space_from
 * @param color_space_to
 */
template<enum gpujpeg_color_space color_space_from, enum gpujpeg_color_space color_space_to>
struct gpujpeg_color_transform
{
    static __device__ void
    perform(float & c1, float & c2, float & c3) {
        assert(false);
    }
};

/** Specialization [color_space_from = color_space_to] */
template<enum gpujpeg_color_space color_space>
struct gpujpeg_color_transform<color_space, color_space> {
    /** None transform */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        // Same color space so do nothing 
    }
};

/** Specialization [color_space_from = GPUJPEG_RGB, color_space_to = GPUJPEG_YCBCR_JPEG] */
template<>
struct gpujpeg_color_transform<GPUJPEG_RGB, GPUJPEG_YCBCR_JPEG> {
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

/** Specialization [color_space_from = GPUJPEG_YCBCR_ITU_R, color_space_to = GPUJPEG_YCBCR_JPEG] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_ITU_R, GPUJPEG_YCBCR_JPEG> {
    /** YUV -> YCbCr transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        c1 -= 16;
        // Check minimum value 0
        c1 = (c1 >= 0.0f) ? c1 : 0.0f;
    }
};

/** Specialization [color_space_from = GPUJPEG_YCBCR_JPEG, color_space_to = GPUJPEG_RGB] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_JPEG, GPUJPEG_RGB> {
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

/** Specialization [color_space_from = GPUJPEG_YCBCR_JPEG, color_space_to = GPUJPEG_YCBCR_ITU_R] */
template<>
struct gpujpeg_color_transform<GPUJPEG_YCBCR_JPEG, GPUJPEG_YCBCR_ITU_R> {
    /** YCbCr -> YUV transform (8 bit) */
    static __device__ void 
    perform(float & c1, float & c2, float & c3) {
        c1 += 16;
        // Check maximum value 255
        c1 = (c1 <= 255.0) ? c1 : 255.0f;
    }
};

#define RGB_8BIT_THREADS 256

/**
 * Preprocessor data for component
 */
struct gpujpeg_preprocessor_data_component
{
    uint8_t* d_data;
    int data_width;
    int data_height;
    struct gpujpeg_component_sampling_factor sampling_factor;
};

/**
 * Preprocessor data
 */
struct gpujpeg_preprocessor_data
{
    struct gpujpeg_preprocessor_data_component comp[3];
};

/**
 * Store value to component data buffer in specified position by buffer size and subsampling
 * 
 * @param value
 * @param position_x
 * @param position_y
 * @param comp
 */
__device__ void
gpujpeg_preprocessor_raw_to_comp_store(uint8_t value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
{
    if ( (position_x % comp.sampling_factor.horizontal) != 0 && (position_x % comp.sampling_factor.vertical) != 0 )
        return;
    position_x = position_x / comp.sampling_factor.horizontal;
    position_y = position_y / comp.sampling_factor.vertical;
    
    int data_position = position_y * comp.data_width + position_x;
    comp.d_data[data_position] = value;
}

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
typedef void (*gpujpeg_preprocessor_encode_kernel)(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, int image_width, int image_height);
 
/** Specialization [sampling factor is 4:4:4] */
template<enum gpujpeg_color_space color_space>
__global__ void 
gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, int image_width, int image_height)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
        
    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 3];
    if ( (x * 4) < RGB_8BIT_THREADS * 3 ) {
        int* s = (int*)d_data_raw;
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
    gpujpeg_color_transform<color_space, GPUJPEG_YCBCR_JPEG>::perform(r1, r2, r3);
    // Store
    int image_position = gX + x;
    if ( image_position < (image_width * image_height) ) {
        int image_position_x = image_position % image_width;
        int image_position_y = image_position / image_width;
        gpujpeg_preprocessor_raw_to_comp_store((uint8_t)r1, image_position_x, image_position_y, data.comp[0]);
        gpujpeg_preprocessor_raw_to_comp_store((uint8_t)r2, image_position_x, image_position_y, data.comp[1]);
        gpujpeg_preprocessor_raw_to_comp_store((uint8_t)r3, image_position_x, image_position_y, data.comp[2]);
    }
}

/** Specialization [sampling factor is 4:2:2] */
template<enum gpujpeg_color_space color_space>
__global__ void 
gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2(struct gpujpeg_preprocessor_data data, const uint8_t* d_data_raw, int image_width, int image_height)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
        
    // Load to shared
    __shared__ unsigned char s_data[RGB_8BIT_THREADS * 2];
    if ( (x * 4) < RGB_8BIT_THREADS * 2 ) {
        int* s = (int*)d_data_raw;
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
    gpujpeg_color_transform<color_space, GPUJPEG_YCBCR_JPEG>::perform(r1, r2, r3);
    // Store
    int image_position = gX + x;
    if ( image_position < (image_width * image_height) ) {
        int image_position_x = image_position % image_width;
        int image_position_y = image_position / image_width;
        gpujpeg_preprocessor_raw_to_comp_store((uint8_t)r1, image_position_x, image_position_y, data.comp[0]);
        gpujpeg_preprocessor_raw_to_comp_store((uint8_t)r2, image_position_x, image_position_y, data.comp[1]);
        gpujpeg_preprocessor_raw_to_comp_store((uint8_t)r3, image_position_x, image_position_y, data.comp[2]);
    }
}

/**
 * Select preprocessor encode kernel
 * 
 * @param encoder
 * @return kernel
 */
gpujpeg_preprocessor_encode_kernel
gpujpeg_preprocessor_select_encode_kernel(struct gpujpeg_encoder* encoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    
    // RGB color space
    if ( coder->param_image.color_space == GPUJPEG_RGB ) {
        assert(coder->param_image.sampling_factor == GPUJPEG_4_4_4);
        return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<GPUJPEG_RGB>;
    } 
    // YCbCr ITU-R color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_ITU_R ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<GPUJPEG_YCBCR_ITU_R>;
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2<GPUJPEG_YCBCR_ITU_R>;
        } else {
            assert(false);
        }
    } 
    // YCbCr JPEG color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_JPEG ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_4_4<GPUJPEG_YCBCR_JPEG>;
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            return &gpujpeg_preprocessor_raw_to_comp_kernel_4_2_2<GPUJPEG_YCBCR_JPEG>;
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
gpujpeg_preprocessor_encode(struct gpujpeg_encoder* encoder)
{    
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    
    cudaMemset(coder->d_data, 0, coder->data_size * sizeof(uint8_t));

    // Select kernel
    gpujpeg_preprocessor_encode_kernel kernel = gpujpeg_preprocessor_select_encode_kernel(encoder);
    
    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;
    
    // When loading 4:2:2 data of odd width, the data in fact has even width, so round it
    // (at least imagemagick convert tool generates data stream in this way)
    if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 )
        image_width = gpujpeg_div_and_round_up(coder->param_image.width, 2) * 2;
        
    // Prepare unit size
    assert(coder->param_image.sampling_factor == GPUJPEG_4_4_4 || coder->param_image.sampling_factor == GPUJPEG_4_2_2);
    int unitSize = coder->param_image.sampling_factor == GPUJPEG_4_4_4 ? 3 : 2;
    
    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);

    // Run kernel
    struct gpujpeg_preprocessor_data data;
    for ( int comp = 0; comp < 3; comp++ ) {
        assert(coder->sampling_factor.horizontal % coder->component[comp].sampling_factor.horizontal == 0);
        assert(coder->sampling_factor.vertical % coder->component[comp].sampling_factor.vertical == 0);
        data.comp[comp].d_data = coder->component[comp].d_data;
        data.comp[comp].sampling_factor.horizontal = coder->sampling_factor.horizontal / coder->component[comp].sampling_factor.horizontal;
        data.comp[comp].sampling_factor.vertical = coder->sampling_factor.vertical / coder->component[comp].sampling_factor.vertical;
        data.comp[comp].data_width = coder->component[comp].data_width;
        data.comp[comp].data_height = coder->component[comp].data_height;
    }
    kernel<<<grid, threads>>>(
        data,
        coder->d_data_raw, 
        image_width,
        image_height
    );
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Preprocessor encoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
        
    return 0;
}

/**
 * Store value to component data buffer in specified position by buffer size and subsampling
 * 
 * @param value
 * @param position_x
 * @param position_y
 * @param comp
 */
__device__ void
gpujpeg_preprocessor_comp_to_raw_load(float & value, int position_x, int position_y, struct gpujpeg_preprocessor_data_component & comp)
{
    position_x = position_x / comp.sampling_factor.horizontal;
    position_y = position_y / comp.sampling_factor.vertical;
    
    int data_position = position_y * comp.data_width + position_x;
    value = (float)comp.d_data[data_position];
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
typedef void (*gpujpeg_preprocessor_decode_kernel)(struct gpujpeg_preprocessor_data data, uint8_t* d_data_raw, int image_width, int image_height);

/** Specialization [sampling factor is 4:4:4] */
template<enum gpujpeg_color_space color_space>
__global__ void
gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4(struct gpujpeg_preprocessor_data data, uint8_t* d_data_raw, int image_width, int image_height)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
    int image_position = gX + x;
    if ( image_position >= (image_width * image_height) )
        return;
    int image_position_x = image_position % image_width;
    int image_position_y = image_position / image_width;
        
    // Load
    float r1;
    float r2;
    float r3;
    gpujpeg_preprocessor_comp_to_raw_load(r1, image_position_x, image_position_y, data.comp[0]);
    gpujpeg_preprocessor_comp_to_raw_load(r2, image_position_x, image_position_y, data.comp[1]);
    gpujpeg_preprocessor_comp_to_raw_load(r3, image_position_x, image_position_y, data.comp[2]);
    
    // Color transform
    gpujpeg_color_transform<GPUJPEG_YCBCR_JPEG, color_space>::perform(r1, r2, r3);
    
    // Save
    image_position = image_position * 3;
    d_data_raw[image_position + 0] = (uint8_t)r1;
    d_data_raw[image_position + 1] = (uint8_t)r2;
    d_data_raw[image_position + 2] = (uint8_t)r3;
}

/** Specialization [sampling factor is 4:2:2] */
template<enum gpujpeg_color_space color_space>
__global__ void
gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2(struct gpujpeg_preprocessor_data data, uint8_t* d_data_raw, int image_width, int image_height)
{
    int x  = threadIdx.x;
    int gX = blockDim.x * blockIdx.x;
    int image_position = gX + x;
    if ( image_position >= (image_width * image_height) )
        return;
    int image_position_x = image_position % image_width;
    int image_position_y = image_position / image_width;
        
    // Load
    float r1;
    float r2;
    float r3;
    gpujpeg_preprocessor_comp_to_raw_load(r1, image_position_x, image_position_y, data.comp[0]);
    gpujpeg_preprocessor_comp_to_raw_load(r2, image_position_x, image_position_y, data.comp[1]);
    gpujpeg_preprocessor_comp_to_raw_load(r3, image_position_x, image_position_y, data.comp[2]);    
    
    // Color transform
    gpujpeg_color_transform<GPUJPEG_YCBCR_JPEG, color_space>::perform(r1, r2, r3);
    
    // Save
    image_position = image_position * 2;
    d_data_raw[image_position + 1] = (uint8_t)r1;
    if ( (image_position_x % 2) == 0 )
        d_data_raw[image_position + 0] = (uint8_t)r2;
    else
        d_data_raw[image_position + 0] = (uint8_t)r3;
}

/**
 * Select preprocessor decode kernel
 * 
 * @param decoder
 * @return kernel
 */
gpujpeg_preprocessor_decode_kernel
gpujpeg_preprocessor_select_decode_kernel(struct gpujpeg_decoder* decoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    // RGB color space
    if ( coder->param_image.color_space == GPUJPEG_RGB ) {
        assert(coder->param_image.sampling_factor == GPUJPEG_4_4_4);
        return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<GPUJPEG_RGB>;
    } 
    // YCbCr ITU-R color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_ITU_R ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<GPUJPEG_YCBCR_ITU_R>;
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2<GPUJPEG_YCBCR_ITU_R>;
        } else {
            assert(false);
        }
    }
    // YCbCr JPEG color space
    else if ( coder->param_image.color_space == GPUJPEG_YCBCR_JPEG ) {
        if ( coder->param_image.sampling_factor == GPUJPEG_4_4_4 ) {
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_4_4<GPUJPEG_YCBCR_JPEG>;
        } else if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 ) {
            return &gpujpeg_preprocessor_comp_to_raw_kernel_4_2_2<GPUJPEG_YCBCR_JPEG>;
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
gpujpeg_preprocessor_decode(struct gpujpeg_decoder* decoder)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    cudaMemset(coder->d_data_raw, 0, coder->data_raw_size * sizeof(uint8_t));
    
    // Select kernel
    gpujpeg_preprocessor_decode_kernel kernel = gpujpeg_preprocessor_select_decode_kernel(decoder);
    
    int image_width = coder->param_image.width;
    int image_height = coder->param_image.height;
    
    // When saving 4:2:2 data of odd width, the data should have even width, so round it
    if ( coder->param_image.sampling_factor == GPUJPEG_4_2_2 )
        image_width = gpujpeg_div_and_round_up(coder->param_image.width, 2) * 2;
        
    // Prepare unit size
    assert(coder->param_image.sampling_factor == GPUJPEG_4_4_4 || coder->param_image.sampling_factor == GPUJPEG_4_2_2);
    int unitSize = coder->param_image.sampling_factor == GPUJPEG_4_4_4 ? 3 : 2;
    
    // Prepare kernel
    int alignedSize = gpujpeg_div_and_round_up(image_width * image_height, RGB_8BIT_THREADS) * RGB_8BIT_THREADS * unitSize;
    dim3 threads (RGB_8BIT_THREADS);
    dim3 grid (alignedSize / (RGB_8BIT_THREADS * unitSize));
    assert(alignedSize % (RGB_8BIT_THREADS * unitSize) == 0);

    // Run kernel
    struct gpujpeg_preprocessor_data data;
    for ( int comp = 0; comp < 3; comp++ ) {
        assert(coder->sampling_factor.horizontal % coder->component[comp].sampling_factor.horizontal == 0);
        assert(coder->sampling_factor.vertical % coder->component[comp].sampling_factor.vertical == 0);
        data.comp[comp].d_data = coder->component[comp].d_data;
        data.comp[comp].sampling_factor.horizontal = coder->sampling_factor.horizontal / coder->component[comp].sampling_factor.horizontal;
        data.comp[comp].sampling_factor.vertical = coder->sampling_factor.vertical / coder->component[comp].sampling_factor.vertical;
        data.comp[comp].data_width = coder->component[comp].data_width;
        data.comp[comp].data_height = coder->component[comp].data_height;
    }
    kernel<<<grid, threads>>>(
        data,
        coder->d_data_raw, 
        image_width,
        image_height
    );
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Preprocessing decoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
    
    return 0;
}
