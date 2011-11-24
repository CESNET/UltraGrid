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
 
#ifndef JPEG_COMMON_H
#define JPEG_COMMON_H

#include <stdint.h>

#include "jpeg_type.h"

/** Image file formats */
enum jpeg_image_file_format {
    // Unknown image file format
    IMAGE_FILE_UNKNOWN = 0,
    // Raw file format
    IMAGE_FILE_RAW = 1,
    // JPEG file format
    IMAGE_FILE_JPEG = 2,
    // RGB file format, simple data format without header [R G B] [R G B] ...
    IMAGE_FILE_RGB = 1 | 4,
    // YUV file format, simple data format without header [Y U V] [Y U V] ...
    IMAGE_FILE_YUV = 1 | 8
};

/** Image parameters */
struct jpeg_image_parameters {
    // Image data width
    int width;
    // Image data height
    int height;
    // Image data component count
    int comp_count;
    // Image data color space
    enum jpeg_color_space color_space;
    // Image data sampling factor
    enum jpeg_sampling_factor sampling_factor;
};

/**
 * Set default parameters for JPEG image
 * 
 * @param param  Parameters for image
 * @return void
 */
void
jpeg_image_set_default_parameters(struct jpeg_image_parameters* param);

/**
 * Init CUDA device
 * 
 * @param device_id  CUDA device id (starting at 0)
 * @param verbose  Flag if device info should be printed out (0 or 1)
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_init_device(int device_id, int verbose);

/**
 * Get image file format from filename
 *
 * @param filename  Filename of image file
 * @return image_file_format or IMAGE_FILE_UNKNOWN if type cannot be determined
 */
enum jpeg_image_file_format
jpeg_image_get_file_format(const char* filename);

/**
 * Load RGB image from file
 * 
 * @param filaname  Image filename
 * @param image  Image data buffer
 * @param image_size  Image data buffer size (can be specified for verification or 0 for retrieval)
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_image_load_from_file(const char* filename, uint8_t** image, int* image_size);

/**
 * Save RGB image to file
 * 
 * @param filaname  Image filename
 * @param image  Image data buffer
 * @param image_size  Image data buffer size
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_image_save_to_file(const char* filename, uint8_t* image, int image_size);

/**
 * Destroy DXT image
 * 
 * @param image  Image data buffer
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_image_destroy(uint8_t* image);

#endif // JPEG_COMMON_H