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

#include "dxt_common.h"
#include "dxt_util.h"

/** Documented at declaration */
int
dxt_init()
{
    int argc = 0;
    glutInit(&argc, NULL);
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
    glutCreateWindow("DXT"); 
    glutHideWindow();
    glewInit();
	
    int result = 0;
    
#define CHECK_EXTENSION(extension) \
    if ( !glewIsSupported(extension) ) { \
        fprintf(stderr, "OpenGL extension '%s' is missing!\n", extension); \
        result = -1; \
    } \
    
    CHECK_EXTENSION("GL_VERSION_2_0");
    CHECK_EXTENSION("GL_ARB_vertex_program");
    CHECK_EXTENSION("GL_ARB_fragment_program");
    CHECK_EXTENSION("GL_EXT_gpu_shader4");
    CHECK_EXTENSION("GL_EXT_framebuffer_object");
    CHECK_EXTENSION("GL_ARB_texture_compression");
    CHECK_EXTENSION("GL_EXT_texture_compression_s3tc");
    CHECK_EXTENSION("GL_EXT_texture_integer");
    
    printf("Graphic Card:   %s\n", glGetString(GL_RENDERER));
    printf("OpenGL version: %s\n", glGetString(GL_VERSION));
    printf("GLSL version:   %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    return result;
}

/** Documented at declaration */
int
dxt_image_load_from_file(const char* filename, int width, int height, DXT_IMAGE_TYPE** image)
{
    FILE* file;
	file = fopen(filename, "rb");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for reading!\n", filename);
		return -1;
	}

    int data_size = width * height * 3;
    unsigned char* data = (unsigned char*)malloc(data_size * sizeof(unsigned char));
    if ( (size_t) data_size != fread(data, sizeof(unsigned char), data_size, file) ) {
        fprintf(stderr, "Failed to load image data [%d bytes] from file %s!\n", data_size, filename);
        return -1;
    }
    fclose(file);
    
    *image = (DXT_IMAGE_TYPE*)malloc(width * height * 4 * sizeof(DXT_IMAGE_TYPE));
    for ( int y = 0; y < height; y++ ) {
        for ( int x = 0; x < width; x++ ) {
            int index = 4 * ((y * width) + x);
            int index_data = 3 * ((y * width) + x);
            (*image)[index + 0] = (DXT_IMAGE_TYPE)data[index_data + 0];
            (*image)[index + 1] = (DXT_IMAGE_TYPE)data[index_data + 1];
            (*image)[index + 2] = (DXT_IMAGE_TYPE)data[index_data + 2];
            (*image)[index + 3] = 1.0;
        }
    }

    free(data);
    
#ifdef DEBUG
    printf("Load Image [file: %s, size: %d bytes, resolution: %dx%d]\n", filename, data_size, width, height);
#endif
    
    return 0;
}

/** Documented at declaration */
int
dxt_image_save_to_file(const char* filename, DXT_IMAGE_TYPE* image, int width, int height)
{
    FILE* file;
	file = fopen(filename, "wb");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for writing!\n", filename);
		return -1;
	}
    
    int data_size = width * height * 3;
    unsigned char* data = (unsigned char*)malloc(data_size * sizeof(unsigned char));
    for ( int y = 0; y < height; y++ ) {
        for ( int x = 0; x < width; x++ ) {
            int index = 4 * ((y * width) + x);
            int index_data = 3 * ((y * width) + x);
            data[index_data + 0] = (unsigned char)(image[index + 0]);
            data[index_data + 1] = (unsigned char)(image[index + 1]);
            data[index_data + 2] = (unsigned char)(image[index + 2]);
        }
    }
    
    if ( (size_t) data_size != fwrite(data, sizeof(unsigned char), data_size, file) ) {
        fprintf(stderr, "Failed to write image data [%d bytes] to file %s!\n", width * height * 3, filename);
        free(data);
        return -1;
    }
    free(data);
    fclose(file);

#ifdef DEBUG
    printf("Save Image [file: %s, size: %d bytes, resolution: %dx%d]\n", filename, data_size, width, height);
#endif
    
    return 0;
}

/** Documented at declaration */
int
dxt_image_compressed_load_from_file(const char* filename, unsigned char** data, int* data_size)
{
    FILE* file;
	file = fopen(filename, "rb");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for reading!\n", filename);
		return -1;
	}

    fseek(file, 0, SEEK_END);
    *data_size = ftell(file);
    rewind(file);
    *data = (unsigned char*)malloc(*data_size * sizeof(unsigned char));
    if ( (size_t) *data_size != fread(*data, sizeof(unsigned char), *data_size, file) ) {
        fprintf(stderr, "Failed to load image data [%d bytes] from file %s!\n", *data_size, filename);
        return -1;
    }
    fclose(file);
    
#ifdef DEBUG
    printf("Load Compressed Image [file: %s, size: %d bytes]\n", filename, *data_size);
#endif
    
    return 0;
}

/** Documented at declaration */
int
dxt_image_compressed_save_to_file(const char* filename, unsigned char* image, int image_size)
{
    FILE* file;
	file = fopen(filename, "wb");
	if ( !file ) {
		fprintf(stderr, "Failed open %s for writing!\n", filename);
		return -1;
	}
    
    if ( (size_t) image_size != fwrite(image, sizeof(unsigned char), image_size, file) ) {
        fprintf(stderr, "Failed to write image data [%d bytes] to file %s!\n", image_size, filename);
        return -1;
    }
    fclose(file);
    
#ifdef DEBUG
    printf("Save Compressed Image [file: %s, size: %d bytes]\n", filename, image_size);
#endif
    
    return 0;
}

/** Documented at declaration */
int
dxt_image_destroy(DXT_IMAGE_TYPE* image)
{
    free(image);

    return 0;
}

/** Documented at declaration */
int
dxt_image_compressed_destroy(unsigned char* image_compressed)
{
    free(image_compressed);

    return 0;
}
