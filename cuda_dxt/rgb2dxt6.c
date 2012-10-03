#include <stdio.h>
#include <stdlib.h>
#include "cuda_dxt.h"


static void usage(const char * const name, const char * const message) {
    printf("ERROR: %s\n", message);
    printf("Usage: %s width height input.rgb output.yog\n", name);
}


int main(int argc, char** argv) {
    FILE *in_file, *out_file;
    size_t in_size, out_size;
    int size_x, size_y;
    void *in_host = 0, *out_host = 0, *out_gpu = 0, *in_gpu = 0;
    unsigned int * header;
    cudaEvent_t event_begin, event_end;
    float kernel_time_ms;
    
    /* check arguments */
    if(5 != argc) {
        usage(argv[0], "Incorrect argument count.");
        return -1;
    }
    
    /* get image size */
    size_x = atoi(argv[1]);
    size_y = atoi(argv[2]);
    if(size_x <= 0 || (size_x & 3) || size_y == 0 || (abs(size_y) & 3)) {
        usage(argv[0], "Image sizes must be positive numbers divisible by 4.");
        return -1;
    }
    
    /* Check CUDA device */
    if(cudaSuccess != cudaSetDevice(0)) {
        usage(argv[0], "Cannot set CUDA device #0.");
        return -1;
    }
    
    /* allocate buffers (both GPU and host buffers) */
    out_size = size_x * abs(size_y);
    in_size = out_size * 3;
    cudaMallocHost(&in_host, in_size);
    cudaMallocHost(&out_host, out_size);
    cudaMalloc(&in_gpu, in_size);
    cudaMalloc(&out_gpu, out_size);
    if(!in_host || !out_host || !out_gpu || !in_gpu) {
        usage(argv[0], "Cannot allocate buffers.\n");
        return -1;
    }
    
    /* open input file */
    in_file = fopen(argv[3], "r");
    if(!in_file) {
        usage(argv[0], "Could not open input file.");
        return -1;
    }
    
#if 0
    /* check file size */
    fseek(in_file, 0, SEEK_END);
    if(ftell(in_file) != (long int)in_size) {
        usage(argv[0], "Input file size does not match.");
        return -1;
    }
    fseek(in_file, 0, SEEK_SET);
#endif
    
    /* load data */
    if(1 != fread(in_host, in_size, 1, in_file)) {
        usage(argv[0], "Could not read from input file.");
        return -1;
    }
    fclose(in_file);
    
    /* copy data into GPU buffer */
    if(cudaSuccess != cudaMemcpy(in_gpu, in_host, in_size, cudaMemcpyHostToDevice)) {
        usage(argv[0], "Could not copy input to GPU.");
        return -1;
    }
    
    /* prepare envents for time emasurement and begin */
    cudaEventCreate(&event_end);
    cudaEventCreate(&event_begin);
    cudaEventRecord(event_begin, 0);
    
    /* compress */
    if(cuda_rgb_to_dxt6(in_gpu, out_gpu, size_x, size_y, 0)) {
        usage(argv[0], "DXT Encoder error.");
        return -1;
    }
    
    /* measure kernel time */
    cudaEventRecord(event_end, 0);
    
    /* check kernel call */
    if(cudaSuccess != cudaDeviceSynchronize()) {
        usage(argv[0], cudaGetErrorString(cudaGetLastError()));
        return -1;
    }
    
    /* print the time */
    cudaEventElapsedTime(&kernel_time_ms, event_begin, event_end);
    printf("DXT6 compression time: %.3f ms.\n", kernel_time_ms);
    
    /* copy back to host memory */
    if(cudaSuccess != cudaMemcpy((char*)out_host, out_gpu, out_size, cudaMemcpyDeviceToHost)) {
        usage(argv[0], "Could not copy output to host memory.");
        return -1;
    }
    
    /* open output file */
    out_file = fopen(argv[4], "w");
    if(!out_file) {
        usage(argv[0], "Could not open output file for writing.");
        return -1;
    }
    
    /* write output into the file */
    if(1 != fwrite(out_host, out_size, 1, out_file)) {
        usage(argv[0], "Could not write into output file.");
        return -1;
    }
    fclose(out_file);
    
    /* free buffers */
    cudaFreeHost(in_host);
    cudaFreeHost(out_host);
    cudaFree(in_gpu);
    cudaFree(out_gpu);
    
    /* indicate success */
    return 0;
}

