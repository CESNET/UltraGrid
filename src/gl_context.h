/**
 * @file   gl_context.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2023 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef _GL_CONTEXT_H_
#define _GL_CONTEXT_H_

#if defined __linux__ || defined _WIN32
#include <GL/glew.h>
#else
#include <OpenGL/GL.h>
#endif /*  HAVE_LINUX */

#ifdef __APPLE__
#include <Availability.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if defined __APPLE__ && ! defined __MAC_10_8
#define glGenFramebuffers glGenFramebuffersEXT
#define glBindFramebuffer glBindFramebufferEXT
#define GL_FRAMEBUFFER GL_FRAMEBUFFER_EXT
#define glFramebufferTexture2D glFramebufferTexture2DEXT
#define glDeleteFramebuffers glDeleteFramebuffersEXT
#define GL_FRAMEBUFFER_COMPLETE GL_FRAMEBUFFER_COMPLETE_EXT
#define glCheckFramebufferStatus glCheckFramebufferStatusEXT
#endif

struct gl_context {
        unsigned legacy:1;
        void *context;
        int gl_major;
        int gl_minor;
};

#define GL_CONTEXT_ANY 0x00
#define GL_CONTEXT_LEGACY 0x01

/**
 * @param which GL_CONTEXT_ANY or GL_CONTEXT_LEGACY
 */
bool init_gl_context(struct gl_context *context, int which);
void gl_context_make_current(struct gl_context *context);
void destroy_gl_context(struct gl_context *context);

#define gl_check_error() do { \
        GLenum msg = glGetError(); \
        int flag = 1; \
        switch(msg){ \
                case GL_NO_ERROR: \
                        flag = 0; \
                        break; \
                case GL_INVALID_ENUM: \
                        fprintf(stderr, "GL_INVALID_ENUM\n"); \
                        break; \
                case GL_INVALID_VALUE: \
                        fprintf(stderr, "GL_INVALID_VALUE\n"); \
                        break; \
                case GL_INVALID_OPERATION: \
                        fprintf(stderr, "GL_INVALID_OPERATION\n"); \
                        break; \
                case GL_STACK_OVERFLOW: \
                        fprintf(stderr, "GL_STACK_OVERFLOW\n"); \
                        break; \
                case GL_STACK_UNDERFLOW: \
                        fprintf(stderr, "GL_STACK_UNDERFLOW\n"); \
                         break; \
                case GL_OUT_OF_MEMORY: \
                        fprintf(stderr, "GL_OUT_OF_MEMORY\n"); \
                        break; \
                case 1286: \
                        fprintf(stderr, "INVALID_FRAMEBUFFER_OPERATION_EXT\n"); \
                        break; \
                default: \
                        fprintf(stderr, "wft mate? Unknown GL ERROR: %d\n", msg); \
                break; \
        } \
        if(flag) { \
                fprintf(stderr, "\tat %s:%d: %s\n", __FILE__, __LINE__, __func__); \
                abort(); \
        } \
} while(0)

/**
 * compiles and links specified program shaders
 * @returns shader ID
 * @retval 0 on error
 */
GLuint glsl_compile_link(const char *vprogram, const char *fprogram);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* _GL_CONTEXT_H_ */

