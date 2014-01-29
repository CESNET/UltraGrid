/**
 * Copyright (c) 2011, CESNET z.s.p.o
 * Copyright (c) 2011, Silicon Genome, LLC.
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

#ifndef GPUJPEG_DECODER_H
#define GPUJPEG_DECODER_H

#include <stdint.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_type.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined _MSC_VER || defined __MINGW32__
#ifdef GPUJPEG_EXPORTS
#define GPUJPEG_API __declspec(dllexport)
#else
#define GPUJPEG_API __declspec(dllimport)
#endif
#else // other platforms
#define GPUJPEG_API
#endif

struct gpujpeg_decoder;

/**
 * Decoder output type
 */
enum gpujpeg_decoder_output_type {
    // Decoder will use it's internal output buffer
    GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER,
    // Decoder will use custom output buffer
    GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER,
    // Decoder will use OpenGL Texture PBO Resource as output buffer
    GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE,
    // Decoder will use CUDA buffer as output buffer
    GPUJPEG_DECODER_OUTPUT_CUDA_BUFFER,
};

/**
 * Decoder output structure
 */
struct gpujpeg_decoder_output
{
    // Output type
    enum gpujpeg_decoder_output_type type;
    
    // Compressed data
    uint8_t* data;
    
    // Compressed data size
    int data_size;
    
    // OpenGL texture
    struct gpujpeg_opengl_texture* texture;
};

/**
 * Set default parameters to decoder output structure
 * 
 * @param output  Decoder output structure
 * @return void
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_default(struct gpujpeg_decoder_output* output);

/**
 * Setup decoder output to custom buffer
 *
 * @param output        Decoder output structure
 * @param custom_buffer Custom buffer
 * @return void
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_custom(struct gpujpeg_decoder_output* output, uint8_t* custom_buffer);

/**
 * Set decoder output to OpenGL texture
 *
 * @param output  Decoder output structure
 * @return void
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_texture(struct gpujpeg_decoder_output* output, struct gpujpeg_opengl_texture* texture);

/**
 * Sets output to CUDA buffer
 * 
 * @param output  Decoder output structure
 */
GPUJPEG_API void
gpujpeg_decoder_output_set_cuda_buffer(struct gpujpeg_decoder_output* output);

/**
 * Create JPEG decoder
 * 
 * @param param  Parameters for coder
 * @param param_image  Parameters for image data
 * @return decoder structure if succeeds, otherwise NULL
 */
GPUJPEG_API struct gpujpeg_decoder*
gpujpeg_decoder_create();

/**
 * Init JPEG decoder for specific image size
 * 
 * @param decoder  Decoder structure
 * @param param  Parameters for coder
 * @param param_image  Parameters for image data
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image);

/**
 * Decompress image by decoder
 * 
 * @param decoder  Decoder structure
 * @param image  Source image data
 * @param image_size  Source image data size
 * @param image_decompressed  Pointer to variable where decompressed image data buffer will be placed
 * @param image_decompressed_size  Pointer to variable where decompressed image size will be placed
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size, struct gpujpeg_decoder_output* output);

/**
 * Destory JPEG decoder
 * 
 * @param decoder  Decoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder);

/**
 * Sets output format
 *
 * @param decoder         Decoder structure
 * @param color_space     Requested output color space
 * @param sampling_factor Requestd color sampling factor
 */
GPUJPEG_API void
gpujpeg_decoder_set_output_format(struct gpujpeg_decoder* decoder,
                enum gpujpeg_color_space color_space,
                enum gpujpeg_sampling_factor sampling_factor);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_DECODER_H
