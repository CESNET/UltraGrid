/**
 * Copyright (c) 2011, CESNET z.s.p.o
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

#include "image.h"
#include "util.h"

/** Documented at declaration */
struct image*
image_create(int width, int height)
{
    struct image* image = (struct image*)malloc(sizeof(struct image));
    if ( image == NULL )
        return NULL;
    image->width = width;
    image->height = height;
    
    cudaMallocHost((void**)&image->data, image->width * image->height * 3 * sizeof(uint8_t));
    cudaCheckError();
    
    cudaMalloc((void**)&image->d_data, image->width * image->height * 3 * sizeof(uint8_t));
    cudaCheckError();
    
    return image;
}

/** Documented at declaration */
void
image_destroy(struct image* image)
{
    cudaFreeHost(image->data);
    cudaFree(image->d_data);
    free(image);
}

/**
 * CUDA kernel that fills image data by gradient
 * 
 * @param data
 * @param width
 * @param height
 * @param max
 */
__global__ void
image_render_kernel(uint8_t* data, int width, int height, int max)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if ( x >= width || y >= height )
        return;
        
    int index = (y * width + x) * 3;
    
    data[index + 0] = 0;
    data[index + 1] = max * y / height;
    data[index + 2] = max * x / width;
}

/** Documented at declaration */
void
image_render(struct image* image, int max)
{        
    dim3 block(8, 8);
    dim3 grid(image->width / block.x + 1, image->height / block.y + 1);
    image_render_kernel<<<grid, block>>>(image->d_data, image->width, image->height, max);
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Kernel failed: %s!\n", cudaGetErrorString(cuerr));
        return;
    }
}
