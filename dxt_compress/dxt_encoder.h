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
 
#ifndef DXT_ENCODER_H
#define DXT_ENCODER_H

#include "dxt_common.h"

/**
 * DXT encoder structure
 */
struct dxt_encoder;

/**
 * Create DXT encoder
 * 
 * @param type
 * @param width
 * @param height
 * @param format
 * @return encoder structure or zero if fails
 */
struct dxt_encoder*
dxt_encoder_create(enum dxt_type type, int width, int height, enum dxt_format format, int legacy);

/**
 * Allocate buffer for compressed image by encoder
 * 
 * @param encoder Encoder structure
 * @param image_compressed Pointer to variable where buffer pointer will be placed
 * @param image_compressed_size Pointer to variable where compressed image data size will be set
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_encoder_buffer_allocate(struct dxt_encoder* encoder, unsigned char** image_compressed, int* image_compressed_size);

/**
 * Compress image by DXT encoder
 * 
 * @param encoder Encoder structure
 * @param image Image data
 * @param image_compressed Pointer to buffer where compressed image data will be placed
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_encoder_compress(struct dxt_encoder* encoder, DXT_IMAGE_TYPE* image, unsigned char* image_compressed);

/**
 * Compress image by DXT encoder
 * 
 * @param encoder Encoder structure
 * @param tex texture index
 * @param image_compressed Pointer to buffer where compressed image data will be placed
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_encoder_compress_texture(struct dxt_encoder* encoder, int texture, unsigned char* image_compressed);

/**
 * Free buffer for compressed image
 * 
 * @param image_compressed Pointer to buffer where compressed image data are stored
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_encoder_buffer_free(unsigned char* image_compressed);

/**
 * Destroy DXT encoder
 * 
 * @param encoder Encoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_encoder_destroy(struct dxt_encoder* encoder);

#endif // DXT_ENCODER_H
