#include <stdio.h>
#include <cuda_runtime.h>

// CUDA check error
#define cuda_check_error(msg) \
    { \
        cudaError_t err = cudaGetLastError(); \
        if( cudaSuccess != err) { \
            fprintf(stderr, "[GPUJPEG] [Error] %s (line %i): %s: %s.\n", \
                __FILE__, __LINE__, msg, cudaGetErrorString( err) ); \
            exit(-1); \
        } \
    } \

__global__
void get_value(int* index, int* value)
{
    int x[3];
    for ( int i = 0; i < 3; i++ )
        x[i] = 55;

    *value = x[*index];
}

int main()
{
    int* d_index;
    int* d_value;
    cudaMalloc((void**)&d_index, sizeof(int));
    cudaMalloc((void**)&d_value, sizeof(int));
    cuda_check_error("Alloc failed");

    int index = 0;
    int value = 0;
    cudaMemcpy(d_index, &index, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_value, &value, sizeof(int), cudaMemcpyHostToDevice);
    cuda_check_error("Init failed");

    get_value<<<1, 1>>>(d_index, d_value);
    cudaThreadSynchronize();
    cuda_check_error("Kernel failed");

    cudaMemcpy(&index, d_index,  sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&value, d_value,  sizeof(int), cudaMemcpyDeviceToHost);
    cuda_check_error("Copy failed");
    printf("index = %d\n", index);
    printf("value = %d\n", value);

    return 0;
}
