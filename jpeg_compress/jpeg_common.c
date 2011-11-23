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
 
#include "jpeg_common.h"
#include "jpeg_util.h"

/** Documented at declaration */
void
jpeg_image_set_default_parameters(struct jpeg_image_parameters* param)
{
    param->width = 0;
    param->height = 0;
    param->comp_count = 3;
    param->color_space = JPEG_RGB;
    param->sampling_factor = JPEG_4_4_4;
}

/** Documented at declaration */
int
jpeg_init_device(int device_id, int verbose)
{
    int dev_count;

    cudaGetDeviceCount(&dev_count);

    if ( dev_count == 0 ) {
        printf("No CUDA enabled device\n");
        return -1;
    }

    if ( device_id < 0 || device_id >= dev_count ) {
        printf("Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
               device_id, 0, dev_count - 1);
        return -1;
    }

    struct cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device_id);

    if ( devProp.major < 1 ) {
        printf("Device %d does not support CUDA\n", device_id);
        return -1;
    }

    if ( verbose == 1 )
        printf("Setting device %d: %s (c.c. %d.%d)\n", device_id, devProp.name, devProp.major, devProp.minor);
    cudaSetDevice(device_id);

    return 0;
}

/** Documented at declaration */
enum jpeg_image_file_format
jpeg_image_get_file_format(const char* filename)
{
    static const char *extension[] = { "raw", "rgb", "yuv", "jpg" };
    static const enum jpeg_image_file_format format[] = { IMAGE_FILE_RAW, IMAGE_FILE_RGB, IMAGE_FILE_YUV, IMAGE_FILE_JPEG };
        
    char * ext = strrchr(filename, '.');
    if ( ext == NULL )
        return -1;
    ext++;
    for ( int i = 0; i < sizeof(format) / sizeof(*format); i++ ) {
        if ( strncasecmp(ext, extension[i], 3) == 0 ) {
            return format[i];
        }
    }
    return IMAGE_FILE_UNKNOWN;
}

/** Documented at declaration */
int
jpeg_image_load_from_file(const char* filename, uint8_t** image, int* image_size)
{
    FILE* file;
	file = fopen(filename, "rb");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for reading!\n", filename);
		return -1;
	}

    if ( *image_size == 0 ) {
        fseek(file, 0, SEEK_END);
        *image_size = ftell(file);
        rewind(file);
    }
    
    uint8_t* data = (uint8_t*)malloc(*image_size * sizeof(uint8_t));
    if ( *image_size != fread(data, sizeof(uint8_t), *image_size, file) ) {
        fprintf(stderr, "Failed to load image data [%d bytes] from file %s!\n", *image_size, filename);
        return -1;
    }
    fclose(file);
    
    *image = data;
    
    return 0;
}

/** Documented at declaration */
int
jpeg_image_save_to_file(const char* filename, uint8_t* image, int image_size)
{
    FILE* file;
	file = fopen(filename, "wb");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for writing!\n", filename);
		return -1;
	}
    
    if ( image_size != fwrite(image, sizeof(uint8_t), image_size, file) ) {
        fprintf(stderr, "Failed to write image data [%d bytes] to file %s!\n", image_size, filename);
        return -1;
    }
    fclose(file);
    
    return 0;
}

/** Documented at declaration */
int
jpeg_image_destroy(uint8_t* image)
{
    free(image);
}
