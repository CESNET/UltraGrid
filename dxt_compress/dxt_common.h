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
 
#ifndef DXT_COMMON_H
#define DXT_COMMON_H

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#else
#define DXT_COMPRESS_STANDALONE
#endif

#ifdef DXT_COMPRESS_STANDALONE
#include <GL/glew.h>
#include <GL/glut.h>
#endif


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <sys/time.h>

/**
 * Timer
 */
#define TIMER_INIT() \
    struct timeval __start, __stop; \
    long __seconds, __useconds; \
    float __elapsedTime;
#define TIMER_START() \
    gettimeofday(&__start, NULL);
#define TIMER_STOP() \
    gettimeofday(&__stop, NULL); \
    __seconds  = __stop.tv_sec  - __start.tv_sec; \
    __useconds = __stop.tv_usec - __start.tv_usec; \
    __elapsedTime = ((__seconds) * 1000.0 + __useconds/1000.0) + 0.5;
#define TIMER_DURATION() __elapsedTime
#define TIMER_STOP_PRINT(text) \
    TIMER_STOP(); \
    printf("%s %f ms\n", text, __elapsedTime)

/**
 * Image type
 */
#define DXT_IMAGE_TYPE      unsigned char
#define DXT_IMAGE_GL_TYPE   GL_UNSIGNED_INT_8_8_8_8_REV
#define DXT_IMAGE_GL_FORMAT GL_RGBA

/**
 * DXT format
 */
enum dxt_format {
    DXT_FORMAT_RGB,
    DXT_FORMAT_RGBA,
    DXT_FORMAT_YUV,
    DXT_FORMAT_YUV422
};

/**
 * DXT type
 */
enum dxt_type {
    DXT_TYPE_DXT1,
    DXT_TYPE_DXT1_YUV,
    DXT_TYPE_DXT5_YCOCG
};

/**
 * Initialize DXT
 * 
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_init();

/**
 * Load RGB image from file
 * 
 * @param filaname Image filename
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @return image data buffer or zero if fails
 */
int
dxt_image_load_from_file(const char* filename, int width, int height, DXT_IMAGE_TYPE** image);

/**
 * Load RGB image from file
 * 
 * @param filaname Image filename
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @return image data buffer or zero if fails
 */
int
dxt_image_save_to_file(const char* filename, DXT_IMAGE_TYPE* image, int width, int height);

/**
 * Load compressed image from file
 * 
 * @param filaname Image filename
 * @param width Image width in pixels
 * @param height Image height in pixels
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_image_compressed_load_from_file(const char* filename, unsigned char** data, int* data_size);

/**
 * Save compressed image to file
 * 
 * @param filename Image filename
 * @param image Image data
 * @param image_size Image data size
 * @return 0 if succeeds, otherwise nonzero
 */
int
dxt_image_compressed_save_to_file(const char* filename, unsigned char* image, int image_size);

/**
 * Destroy DXT image
 * 
 * @param image  Image data buffer
 */
int
dxt_image_destroy(DXT_IMAGE_TYPE* image);

/**
 * Destroy DXT compressed image
 * 
 * @param image_compressed  Compressed image data buffer
 */
int
dxt_image_compressed_destroy(unsigned char* image_compressed);

#endif // DXT_COMMON_H
