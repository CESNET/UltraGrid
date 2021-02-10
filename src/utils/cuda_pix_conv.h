#include <cuda_runtime.h>

#ifndef CUDA_RGB_RGBA_H
#define CUDA_RGB_RGBA_H

void cuda_RGB_to_RGBA(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height,
                struct CUstream_st *stream);

void cuda_RGBA_to_RGB(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height,
                struct CUstream_st *stream);

void cuda_RGBA_to_UYVY(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height,
                struct CUstream_st *stream);

void cuda_UYVY_to_RGBA(unsigned char *dst,
                size_t dstPitch,
                unsigned char *src,
                size_t srcPitch,
                size_t width,
                size_t height,
                struct CUstream_st *stream);

#endif
