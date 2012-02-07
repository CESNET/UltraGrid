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
 
#ifndef DXT_DECODER_H
#define DXT_DECODER_H

#include "dxt_common.h"

/**
 * DXT decoder structure
 */
struct dxt_decoder;

/**
 * Create DXT decoder
 * 
 * @param width
 * @param height
 * @return decoder structure or zero if fails
 */
struct dxt_decoder*
dxt_decoder_create(enum dxt_type type, int width, int height, enum dxt_format out_format, int legacy);

/**
 * Allocate buffer for decompressed image by decoder
 * 
 * @param decoder Decoder structure
 * @param image Pointer to variable where buffer pointer will be placed
 * @param image_size Pointer to variable where image data size will be set
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_decoder_buffer_allocate(struct dxt_decoder* decoder, unsigned char** image, int* image_size);

/**
 * Decompress image by DXT decoder
 * 
 * @param Decoder structure
 * @param image_compressed Compressed image data
 * @param image_compressed_size Compressed image data size
 * @param image Pointer to buffer where image data will be placed
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_decoder_decompress(struct dxt_decoder* encoder, unsigned char* image_compressed, DXT_IMAGE_TYPE* image);

/**
 * Free buffer for decompressed image
 * 
 * @param image Pointer to buffer where decompressed image data are stored
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_decoder_buffer_free(unsigned char* image);

/**
 * Destroy DXT decoder
 * 
 * @param decoder Decoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_decoder_destroy(struct dxt_decoder* decoder);

#endif // DXT_DECODER_H
