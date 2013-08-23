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

#ifndef GPUJPEG_ENCODER_H
#define GPUJPEG_ENCODER_H

#include <libgpujpeg/gpujpeg_common.h>
#include <stdint.h>

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

struct gpujpeg_encoder;

/**
 * Encoder input type
 */
enum gpujpeg_encoder_input_type {
    // Encoder will use custom input buffer
    GPUJPEG_ENCODER_INPUT_IMAGE,
    // Encoder will use OpenGL Texture PBO Resource as input buffer
    GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE,
};

/**
 * Encoder input structure
 */
struct gpujpeg_encoder_input
{
    // Output type
    enum gpujpeg_encoder_input_type type;

    // Image data
    uint8_t* image;

    // Registered OpenGL Texture
    struct gpujpeg_opengl_texture* texture;
};

/**
 * Set encoder input to image data
 *
 * @param encoder_input  Encoder input structure
 * @param image  Input image data
 * @return void
 */
GPUJPEG_API void
gpujpeg_encoder_input_set_image(struct gpujpeg_encoder_input* input, uint8_t* image);

/**
 * Set encoder input to OpenGL texture
 *
 * @param encoder_input  Encoder input structure
 * @param texture_id  OpenGL texture id
 * @return void
 */
GPUJPEG_API void
gpujpeg_encoder_input_set_texture(struct gpujpeg_encoder_input* input, struct gpujpeg_opengl_texture* texture);

/**
 * Create JPEG encoder
 * 
 * @param param  Parameters for coder
 * @param param_image  Parameters for image data
 * @return encoder structure if succeeds, otherwise NULL
 */
GPUJPEG_API struct gpujpeg_encoder*
gpujpeg_encoder_create(struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image);

/**
 * Compress image by encoder
 * 
 * @param encoder  Encoder structure
 * @param image  Source image data
 * @param image_compressed  Pointer to variable where compressed image data buffer will be placed
 * @param image_compressed_size  Pointer to variable where compressed image size will be placed
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_encoder_encode(struct gpujpeg_encoder* encoder, struct gpujpeg_encoder_input* input, uint8_t** image_compressed, int* image_compressed_size);

/**
 * Destory JPEG encoder
 * 
 * @param encoder  Encoder structure
 * @return 0 if succeeds, otherwise nonzero
 */
GPUJPEG_API int
gpujpeg_encoder_destroy(struct gpujpeg_encoder* encoder);

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_ENCODER_H
