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

#ifndef TEST_OPENGL_INTEROP_UTIL_H
#define TEST_OPENGL_INTEROP_UTIL_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <unistd.h>
#include <GL/gl.h>
#include <GL/glut.h>
#include <GL/glx.h>

/**
 * Check CUDA error
 */
#define cudaCheckError() { \
    enum cudaError err = cudaGetLastError(); \
    if ( cudaSuccess != err) { \
        fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n", \
                __FILE__, __LINE__, cudaGetErrorString(err) ); \
        exit(-1); \
    } \
} \

/**
 * Check OpenGL error
 */
#define glCheckError() { \
    GLenum error = glGetError(); \
    if ( error != GL_NO_ERROR ) { \
        fprintf(stderr, "OpenGL error in file '%s' in line %i: %s.\n", \
                __FILE__, __LINE__, gluErrorString(error)); \
        exit(-1); \
    } \
} \

/**
 * Timer
 */
#define TIMER_INIT() \
    cudaEvent_t __start, __stop; \
    cudaEventCreate(&__start); \
    cudaEventCreate(&__stop); \
    float __elapsedTime;
#define TIMER_START() \
    cudaEventRecord(__start,0)
#define TIMER_STOP() \
    cudaEventRecord(__stop,0); \
    cudaEventSynchronize(__stop); \
    cudaEventElapsedTime(&__elapsedTime, __start, __stop)
#define TIMER_DURATION() __elapsedTime
#define TIMER_STOP_PRINT(text) \
    TIMER_STOP(); \
    printf("%s %f ms\n", text, __elapsedTime)

#endif // TEST_OPENGL_INTEROP_UTIL_H
