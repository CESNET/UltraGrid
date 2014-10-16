#define cudaError_t int
#define cudaFreeHost cuda_wrapper_free_host
#define cudaMalloc cuda_wrapper_malloc
#define cudaMemcpy cuda_wrapper_memcpy
#define cudaGetErrorString cuda_wrapper_get_error_string
#define cudaGetLastError cuda_wrapper_get_last_error
#define cudaSuccess CUDA_WRAPPER_SUCCESS
#define cudaFree cuda_wrapper_free
#define cudaHostAllocDefault 0
#define cudaHostAlloc cuda_wrapper_host_alloc
#define cudaMemcpyHostToDevice CUDA_WRAPPER_MEMCPY_HOST_TO_DEVICE

#include "cuda_wrapper.h"

