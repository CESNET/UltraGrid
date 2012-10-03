/**
 *   @file    cuda_dxt.h 
 *   @author  Martin Jirman <jirman@cesnet.cz> 
 *            Based on GLSL code by Martin Srom
 *   @brief   Interface of DXT compression CUDA implementation
 */

#ifndef CUDA_DXT_H
#define CUDA_DXT_H

#ifdef __cplusplus
extern "C" {
#endif

#include <cuda_runtime_api.h>


/**
 * CUDA DXT1 compression (only RGB without alpha).
 * @param src  Pointer to top-left source pixel in device-memory buffer. 
 *             8bit RGB samples are expected (no alpha and no padding).
 *             (Pointer must be aligned to multiples of 16 bytes.)
 * @param out  Pointer to output buffer in device memory.
 *             (Must be aligned to multiples of 8 bytes.)
 * @param size_x  Width of the input image (must be divisible by 4).
 * @param size_y  Height of the input image (must be divisible by 4).
 *                (Input is read bottom up if negative)
 * @param stream  CUDA stream to run in, or 0 for default stream.
 * @return 0 if OK, nonzero if failed.
 */
int cuda_rgb_to_dxt1
(
    const void * src, 
    void * out, 
    int size_x, 
    int size_y, 
    cudaStream_t stream
);


/**
 * CUDA DXT1 compression (only RGB without alpha). 
 * Converts input from YUV to RGB color space.
 * @param src  Pointer to top-left source pixel in device-memory buffer. 
 *             8bit RGB samples are expected (no alpha and no padding).
 *             (Pointer must be aligned to multiples of 16 bytes.)
 * @param out  Pointer to output buffer in device memory.
 *             (Must be aligned to multiples of 8 bytes.)
 * @param size_x  Width of the input image (must be divisible by 4).
 * @param size_y  Height of the input image (must be divisible by 4).
 *                (Input is read bottom up if negative)
 * @param stream  CUDA stream to run in, or 0 for default stream.
 * @return 0 if OK, nonzero if failed.
 */
int cuda_yuv_to_dxt1
(
    const void * src,
    void * out,
    int size_x,
    int size_y,
    cudaStream_t stream
);


/**
 * CUDA DXT6 (DXT5-YcOcG) compression (only RGB without alpha).
 * @param src  Pointer to top-left source pixel in device-memory buffer. 
 *             8bit RGB samples are expected (no alpha and no padding).
 *             (Pointer must be aligned to multiples of 16 bytes.)
 * @param out  Pointer to output buffer in device memory.
 *             (Must be aligned to multiples of 8 bytes.)
 * @param size_x  Width of the input image (must be divisible by 4).
 * @param size_y  Height of the input image (must be divisible by 4).
 *                (Input is read bottom up if negative)
 * @param stream  CUDA stream to run in, or 0 for default stream.
 * @return 0 if OK, nonzero if failed.
 */
int cuda_rgb_to_dxt6
(
    const void * src, 
    void * out, 
    int size_x, 
    int size_y, 
    cudaStream_t stream
);


#ifdef __cplusplus
} /* end of extern "C" */
#endif

#endif /* CUDA_DXT_H */
