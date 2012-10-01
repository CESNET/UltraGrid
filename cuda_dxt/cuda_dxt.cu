///
///  @file    cuda_dxt.cu   
///  @author  Martin Jirman <jirman@cesnet.cz>
///  @brief   CUDA implementation of DXT compression
///

#include <stdio.h>
#include "cuda_dxt.h"


typedef unsigned int u32;


/// Encodes color palette endpoint into 565 code and adjusts input values.
__device__ static u32 encode_endpoint(float & r, float & g, float & b) {
    // clamp to range [0,1] and use full output range for each component
    r = rintf(__saturatef(r) * 31.0f);
    g = rintf(__saturatef(g) * 63.0f);  // 6 bits for green sample
    b = rintf(__saturatef(b) * 31.0f);
    
    // compose output 16bit code representing the endpoint color
    const u32 code = ((u32)r << 11) + ((u32)g << 5) + (u32)b;

    // convert all 3 endpoint component samples back to unit range
    r *= 0.0322580645161f;  // divide by 31
    g *= 0.015873015873f;   // divide by 63
    b *= 0.0322580645161f;  // divide by 31
    
    // return output 16bit code for the endpoint
    return code;
}


/// Transform YUV to RGB.
__device__ static void yuv_to_rgb(float & r, float & g, float & b) {
    const float y = 1.1643f * (r - 0.0625f);  // TODO: convert to FFMA
    const float u = g - 0.5f;
    const float v = b - 0.5f;
    r = y + 1.7926f * v;
    g = y - 0.2132f * u - 0.5328f * v;
    b = y + 2.1124f * u;
}


/// Swaps two referenced values.
template <typename T>
__device__ static void swap(T & a, T & b) {
    const T temp = a;
    a = b;
    b = temp;
}


/// DXT compression - each thread compresses one 4x4 DXT block.
/// Alpha-color palette mode is not used (always emmits 4color palette code).
template <bool YUV_TO_RGB>
__global__ static void dxt_kernel(const void * src, void * out, int size_x, int size_y) {
    // coordinates of this thread's 4x4 block
    const int block_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    const int block_idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    
    // coordinates of block's top-left pixel
    const int block_x = block_idx_x * 4;
    const int block_y = block_idx_y * 4;
    
    // skip if out of bounds
    if(block_y >= size_y || block_x >= size_x) {
        return;
    }

    // samples of 16 pixels
    float r[16];
    float g[16];
    float b[16];

    // load RGB samples for all 16 input pixels
    const int src_stride = (size_x >> 2) * 3;
    for(int y = 0; y < 4; y++) {
        // offset of loaded pixels in the buffer
        const int load_offset = y * 4;
        
        // pointer to source of this input row
        const uchar4 * const row_src = (uchar4*)src
                                     + src_stride * (block_y + y)
                                     + block_idx_x * 3;
        
        // load all 4 3component pixels of the row
        const uchar4 p0 = row_src[0];
        const uchar4 p1 = row_src[1];
        const uchar4 p2 = row_src[2];
        
        // pixel #0
        r[load_offset + 0] = p0.x * 0.00392156862745f;
        g[load_offset + 0] = p0.y * 0.00392156862745f;
        b[load_offset + 0] = p0.z * 0.00392156862745f;
        
        // pixel #1
        r[load_offset + 1] = p0.w * 0.00392156862745f;
        g[load_offset + 1] = p1.x * 0.00392156862745f;
        b[load_offset + 1] = p1.y * 0.00392156862745f;
        
        // pixel #2
        r[load_offset + 2] = p1.z * 0.00392156862745f;
        g[load_offset + 2] = p1.w * 0.00392156862745f;
        b[load_offset + 2] = p2.x * 0.00392156862745f;
        
        // pixel #3
        r[load_offset + 3] = p2.y * 0.00392156862745f;
        g[load_offset + 3] = p2.z * 0.00392156862745f;
        b[load_offset + 3] = p2.w * 0.00392156862745f;
    }
    
    // transform colors from YUV to RGB if required
    if(YUV_TO_RGB) {
        for(int i = 0; i < 16; i++) {
            yuv_to_rgb(r[i], g[i], b[i]);
        }
    }
    
    // find min and max sample values for each component
    float mincol_r = r[0];
    float mincol_g = g[0];
    float mincol_b = b[0];
    float maxcol_r = r[0];
    float maxcol_g = g[0];
    float maxcol_b = b[0];
    for(int i = 1; i < 16; i++) {
        mincol_r = min(mincol_r, r[i]);
        mincol_g = min(mincol_g, g[i]);
        mincol_b = min(mincol_b, b[i]);
        maxcol_r = max(maxcol_r, r[i]);
        maxcol_g = max(maxcol_g, g[i]);
        maxcol_b = max(maxcol_b, b[i]);
    }

    // inset the bounding box
    const float inset_r = (maxcol_r - mincol_r) * 0.0625f;
    const float inset_g = (maxcol_g - mincol_g) * 0.0625f;
    const float inset_b = (maxcol_b - mincol_b) * 0.0625f;
    mincol_r += inset_r;
    mincol_g += inset_g;
    mincol_b += inset_b;
    maxcol_r -= inset_r;
    maxcol_g -= inset_g;
    maxcol_b -= inset_b;

    // select diagonal
    const float center_r = (mincol_r + maxcol_r) * 0.5f;
    const float center_g = (mincol_g + maxcol_g) * 0.5f;
    const float center_b = (mincol_b + maxcol_b) * 0.5f;
    float cov_x = 0.0f;
    float cov_y = 0.0f;
    for(int i = 0; i < 16; i++) {
        const float dir_r = r[i] - center_r;
        const float dir_g = g[i] - center_g;
        const float dir_b = b[i] - center_b;
        cov_x += dir_r * dir_b;
        cov_y += dir_g * dir_b;
    }
    if(cov_x < 0.0f) {
        swap(maxcol_r, mincol_r);
    }
    if(cov_y < 0.0f) {
        swap(maxcol_g, mincol_g);
    }

    // encode both endpoints into 565 color format
    const u32 max_code = encode_endpoint(maxcol_r, maxcol_g, maxcol_b);
    const u32 min_code = encode_endpoint(mincol_r, mincol_g, mincol_b);

    // swap palette end colors if 'max' code is less than 'min' color code
    // (Palette color #3 would otherwise be interpreted as 'transparent'.)
    const bool swap_end_colors = max_code < min_code;
    
    // encode the palette into 32 bits (Only 2 end colors are stored.)
    const u32 palette_code = swap_end_colors ?
            min_code + (max_code << 16): max_code + (min_code << 16);
    
    // pack palette color indices (if both endpoint colors are not equal)
    u32 indices = 0;
    if(max_code != min_code) {
        // project each color to line maxcol-mincol, represent it as
        // "mincol + t * (maxcol - mincol)" and then use 't' to find closest 
        // palette color index.
        const float dir_r = mincol_r - maxcol_r;
        const float dir_g = mincol_g - maxcol_g;
        const float dir_b = mincol_b - maxcol_b;
        const float dir_sqr_len = dir_r * dir_r + dir_g * dir_g + dir_b * dir_b;
        const float dir_inv_sqr_len = __fdividef(1.0f, dir_sqr_len);
        const float t_r = dir_r * dir_inv_sqr_len;
        const float t_g = dir_g * dir_inv_sqr_len;
        const float t_b = dir_b * dir_inv_sqr_len;
        const float t_bias = t_r * maxcol_r + t_g * maxcol_g + t_b * maxcol_b;
        
        // for each pixel color:
        for(int i = 0; i < 16; i++) {
            // get 't' for the color
            const float col_t = r[i] * t_r + g[i] * t_g + b[i] * t_b - t_bias;
            
            // scale the range of the 't' to [0..3] and convert to integer
            // to get the index of palette color
            const u32 col_idx = (u32)(3.0f * __saturatef(col_t) + 0.5f);
            
            // pack the color palette index with others
            indices += col_idx << (i * 2);
        }
    }
    
    // possibly invert indices if end colors must be swapped
    if(swap_end_colors) {
        indices = ~indices;
    }
    
    // substitute all packed indices (each index is packed into two bits)
    // 00 -> 00, 01 -> 10, 10 -> 11 and 11 -> 01
    const u32 lsbs = indices & 0x55555555;
    const u32 msbs = indices & 0xaaaaaaaa;
    indices = msbs ^ (2 * lsbs + (msbs >> 1));

    // compose and save output
    const int out_idx = block_idx_x + (size_x >> 2) * block_idx_y;
    ((uint2*)out)[out_idx] = make_uint2(palette_code, indices);
}


/// Compute grid size and launch DXT kernel.
template <bool YUV_TO_RGB>
static int dxt_launch(const void * src, void * out, int sx, int sy, cudaStream_t str) {
    // check image size and alignment
    if((sx & 3) || (sy & 3) || (15 & (size_t)src) || (7 & (size_t)out)) {
        return -1;
    }
    
    // grid and threadblock sizes
    const dim3 tsiz(16, 16);
    const dim3 gsiz((sx + tsiz.x - 1) / tsiz.x, (sy + tsiz.y - 1) / tsiz.y);
    
    // launch kernel, sync and check the result
    dxt_kernel<YUV_TO_RGB><<<gsiz, tsiz, 0, str>>>(src, out, sx, sy);
    return cudaSuccess != cudaStreamSynchronize(str) ? -3 : 0;
}


/// CUDA DXT1 compression (only RGB without alpha).
/// @param src  Pointer to top-left source pixel in device-memory buffer. 
///             8bit RGB samples are expected (no alpha and no padding).
///             (Pointer must be aligned to multiples of 16 bytes.)
/// @param out  Pointer to output buffer in device memory.
///             (Must be aligned to multiples of 8 bytes.)
/// @param size_x  Width of the input image (must be divisible by 4).
/// @param size_y  Height of the input image (must be divisible by 4).
/// @param stream  CUDA stream to run in, or 0 for default stream.
/// @return 0 if OK, nonzero if failed.
int cuda_rgb_to_dxt1(const void * src, void * out, int size_x, int size_y, cudaStream_t stream) {
    return dxt_launch<false>(src, out, size_x, size_y, stream);
}


/// CUDA DXT1 compression (only RGB without alpha).
/// Converts input from YUV to RGB color space.
/// @param src  Pointer to top-left source pixel in device-memory buffer. 
///             8bit RGB samples are expected (no alpha and no padding).
///             (Pointer must be aligned to multiples of 16 bytes.)
/// @param out  Pointer to output buffer in device memory.
///             (Must be aligned to multiples of 8 bytes.)
/// @param size_x  Width of the input image (must be divisible by 4).
/// @param size_y  Height of the input image (must be divisible by 4).
/// @param stream  CUDA stream to run in, or 0 for default stream.
/// @return 0 if OK, nonzero if failed.
int cuda_yuv_to_dxt1(const void * src, void * out, int size_x, int size_y, cudaStream_t stream) {
    return dxt_launch<true>(src, out, size_x, size_y, stream);
}

