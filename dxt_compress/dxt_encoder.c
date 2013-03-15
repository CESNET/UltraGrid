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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif
#include "dxt_encoder.h"
#include "dxt_util.h"
#include "dxt_glsl.h"
#if defined HAVE_MACOSX && OS_VERSION_MAJOR >= 11
#include <OpenGL/gl3.h>
#endif
#ifndef HAVE_MACOSX /* Linux */
#include <GL/glew.h>
#endif

#if defined HAVE_MACOSX && OS_VERSION_MAJOR < 11
#define glGenFramebuffers glGenFramebuffersEXT
#define glBindFramebuffer glBindFramebufferEXT
#define GL_FRAMEBUFFER GL_FRAMEBUFFER_EXT
#define glFramebufferTexture2D glFramebufferTexture2DEXT
#define glDeleteFramebuffers glDeleteFramebuffersEXT
#define GL_FRAMEBUFFER_COMPLETE GL_FRAMEBUFFER_COMPLETE_EXT
#define glCheckFramebufferStatus glCheckFramebufferStatusEXT
#endif

#ifdef HAVE_GPUPERFAPI
#include "GPUPerfAPI.h"
static void WriteSession( gpa_uint32 currentWaitSessionID, const char* filename );
#endif

#include <string.h>

static GLfloat points[] = { -1.0f, -1.0f, 0.0f, 1.0f,
    1.0f, -1.0f, 0.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 1.0f,
    /* second triangle */
    1.0f, -1.0f, 0.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f, 1.0f
};

const int FORMAT_RGB = 0;
const int FORMAT_YUV = 1;

/** Documented at declaration */ 
struct dxt_encoder
{
    unsigned int legacy:1;

    // DXT type
    enum dxt_type type;

    // Width in pixels
    int width;

    // Height in pixels
    int height;

    // Format
    enum dxt_format format;

    // Texture id
    GLuint texture_id;

#ifdef USE_PBO_DXT_ENCODER
    // PBO
    GLuint pbo_in;
    GLuint pbo_out;
#endif

    // Compressed texture
    GLuint texture_compressed_id;
    
    GLuint texture_yuv422;
    
    // Framebuffer
    GLuint fbo_id;
  
    // Program and shader handles
    GLuint program_compress;
    GLuint shader_fragment_compress;
    GLuint shader_vertex_compress;

    GLuint yuv422_to_444_program;
    GLuint yuv422_to_444_fp;
    GLuint yuv422_to_444_vp;
    GLuint fbo444_id;

    /**
     * The VAO for the vertices etc.
     */
    GLuint g_vao;
    GLuint g_vao_422;

    /**
     * The VBO for the vertices.
     */
    GLuint g_vertices;

#ifdef HAVE_GPUPERFAPI
    gpa_uint32 numRequiredPasses;
    gpa_uint32 sessionID;
    gpa_uint32 currentWaitSessionID;
#endif
#ifdef RTDXT_DEBUG
    GLuint queries[4];
#endif
};

static int dxt_prepare_yuv422_shader(struct dxt_encoder *encoder);

static int dxt_prepare_yuv422_shader(struct dxt_encoder *encoder) {
        encoder->yuv422_to_444_fp = 0;    
        if(encoder->legacy) {
            encoder->yuv422_to_444_fp = dxt_shader_create_from_source(fp_yuv422_to_yuv_444_legacy, GL_FRAGMENT_SHADER);
        } else {
            encoder->yuv422_to_444_fp = dxt_shader_create_from_source(fp_yuv422_to_yuv_444, GL_FRAGMENT_SHADER);
        }

        if ( encoder->yuv422_to_444_fp == 0) {
                fprintf(stderr, "Failed to compile YUV422->YUV444 fragment program!\n");
                return 0;
        }

        if(encoder->legacy) {
            encoder->yuv422_to_444_vp = dxt_shader_create_from_source(vp_compress_legacy, GL_VERTEX_SHADER);
        } else {
            encoder->yuv422_to_444_vp = dxt_shader_create_from_source(vp_compress, GL_VERTEX_SHADER);
        }
        
        encoder->yuv422_to_444_program = glCreateProgram();
        glAttachShader(encoder->yuv422_to_444_program, encoder->yuv422_to_444_fp);
        glAttachShader(encoder->yuv422_to_444_program, encoder->yuv422_to_444_vp);
        glLinkProgram(encoder->yuv422_to_444_program);

        if(!encoder->legacy) {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
            GLint g_vertexLocation;
            GLuint g_vertices;

            // ToDo:
            glGenVertexArrays(1, &encoder->g_vao_422);

            // ToDo:
            glBindVertexArray(encoder->g_vao_422);

            // http://www.opengl.org/sdk/docs/man/xhtml/glGetAttribLocation.xml
            g_vertexLocation = glGetAttribLocation(encoder->yuv422_to_444_program, "position");

            // http://www.opengl.org/sdk/docs/man/xhtml/glGenBuffers.xml
            glGenBuffers(1, &g_vertices);

            // http://www.opengl.org/sdk/docs/man/xhtml/glBindBuffer.xml
            glBindBuffer(GL_ARRAY_BUFFER, g_vertices);

            // http://www.opengl.org/sdk/docs/man/xhtml/glBufferData.xml
            //glBufferData(GL_ARRAY_BUFFER, 3 * 4 * sizeof(GLfloat), (GLfloat*) points, GL_STATIC_DRAW);
            glBufferData(GL_ARRAY_BUFFER, 8 * 4 * sizeof(GLfloat), (GLfloat*) points, GL_STATIC_DRAW);

            // http://www.opengl.org/sdk/docs/man/xhtml/glUseProgram.xml
            glUseProgram(encoder->yuv422_to_444_program);

            // http://www.opengl.org/sdk/docs/man/xhtml/glVertexAttribPointer.xml
            glVertexAttribPointer(g_vertexLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

            // http://www.opengl.org/sdk/docs/man/xhtml/glEnableVertexAttribArray.xml
            glEnableVertexAttribArray(g_vertexLocation);

            glBindFragDataLocation(encoder->yuv422_to_444_program, 0, "colorOut");

            glUseProgram(0);
#endif
        }
        
        char log[32768];
        glGetProgramInfoLog(encoder->yuv422_to_444_program, 32768, NULL, (GLchar*)log);
        if ( strlen(log) > 0 )
                printf("Link Log: %s\n", log);
        
        glGenTextures(1, &encoder->texture_yuv422);
        glBindTexture(GL_TEXTURE_2D, encoder->texture_yuv422);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, encoder->width / 2, encoder->height,
                        0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, NULL);
        
        glUseProgram(encoder->yuv422_to_444_program);
        glUniform1i(glGetUniformLocation(encoder->yuv422_to_444_program, "image"), 0);
        glUniform1f(glGetUniformLocation(encoder->yuv422_to_444_program, "imageWidth"),
                        (GLfloat) encoder->width);

        // Create fbo    
        glGenFramebuffers(1, &encoder->fbo444_id);
        return 1;
}


/** Documented at declaration */
struct dxt_encoder*
dxt_encoder_create(enum dxt_type type, int width, int height, enum dxt_format format, int legacy)
{
    struct dxt_encoder* encoder = (struct dxt_encoder*)malloc(sizeof(struct dxt_encoder));

#ifndef HAVE_MACOSX
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    assert(err == GLEW_OK);
#endif

    if ( encoder == NULL )
        return NULL;
    encoder->type = type;
    encoder->width = width;
    encoder->height = height;
    encoder->format = format;
    encoder->legacy = legacy;

    if(legacy) {
            glEnable(GL_TEXTURE_2D);
    }

    //glEnable(GL_TEXTURE_2D);
#ifdef HAVE_GPUPERFAPI
    GPA_EnableAllCounters();
#endif

    glGenFramebuffers(1, &encoder->fbo_id);
    glBindFramebuffer(GL_FRAMEBUFFER, encoder->fbo_id);

#ifdef USE_PBO_DXT_ENCODER
    int bpp;
    if(format == DXT_FORMAT_RGB) {
        bpp = 3;
    } else {
        bpp = 4;
    }

    glGenBuffersARB(1, &encoder->pbo_in); //Allocate PBO
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, encoder->pbo_in);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB,width*height*bpp,0,GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
        
    glGenBuffersARB(1, &encoder->pbo_out); //Allocate PBO
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, encoder->pbo_out);
    glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, ((width + 3) / 4 * 4) * ((height + 3) / 4 * 4)  / (encoder->type == DXT_TYPE_DXT5_YCOCG ? 1 : 2), 0, GL_STREAM_READ_ARB);
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
    
#endif
    
    GLuint fbo_tex;
    glGenTextures(1, &fbo_tex); 
    glBindTexture(GL_TEXTURE_2D, fbo_tex); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA32UI_EXT, (encoder->width + 3) / 4 * 4, (encoder->height + 3) / 4, 0, GL_RGBA_INTEGER_EXT, GL_INT, 0); 
    else
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA16UI_EXT, (encoder->width + 3) / 4 * 4, (encoder->height + 3) / 4, 0, GL_RGBA_INTEGER_EXT, GL_INT, 0); 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, fbo_tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    // Create program [display] and its shader
    encoder->program_compress = glCreateProgram();
    // Create fragment shader from file
    encoder->shader_fragment_compress = 0;
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG ) {
        if(encoder->legacy) {
            if(format == DXT_FORMAT_YUV422 || format == DXT_FORMAT_YUV) {
                encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt5ycocg_yuv_legacy, GL_FRAGMENT_SHADER);
            } else {
                encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt5ycocg_legacy, GL_FRAGMENT_SHADER);
            }
        } else {
            if(format == DXT_FORMAT_YUV422 || format == DXT_FORMAT_YUV) {
                encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt5ycocg_yuv, GL_FRAGMENT_SHADER);
            } else {
                encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt5ycocg, GL_FRAGMENT_SHADER);
            }
        }
    } else {
        if(encoder->legacy) {
            if(format == DXT_FORMAT_YUV422 || format == DXT_FORMAT_YUV) {
                encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt1_yuv_legacy, GL_FRAGMENT_SHADER);
            } else {
                encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt1_legacy, GL_FRAGMENT_SHADER);
            }
        } else {
            if(format == DXT_FORMAT_YUV422 || format == DXT_FORMAT_YUV) {
                encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt1_yuv, GL_FRAGMENT_SHADER);
            } else {
                encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt1, GL_FRAGMENT_SHADER);
            }
        }
    }
    if ( encoder->shader_fragment_compress == 0 ) {
        fprintf(stderr, "Failed to compile fragment compress program!\n");
        return NULL;
    }
    // Create vertex shader from file
    encoder->shader_vertex_compress = 0;
    if(encoder->legacy) {
        encoder->shader_vertex_compress = dxt_shader_create_from_source(vp_compress_legacy, GL_VERTEX_SHADER);
    } else {
        encoder->shader_vertex_compress = dxt_shader_create_from_source(vp_compress, GL_VERTEX_SHADER);
    }
    if ( encoder->shader_vertex_compress == 0 ) {
        fprintf(stderr, "Failed to compile vertex compress program!\n");
        return NULL;
    }
    // Attach shader to program and link the program
    glAttachShader(encoder->program_compress, encoder->shader_fragment_compress);
    glAttachShader(encoder->program_compress, encoder->shader_vertex_compress);
    glLinkProgram(encoder->program_compress);
    
    char log[32768];
    glGetProgramInfoLog(encoder->program_compress, 32768, NULL, (GLchar*)log);
    if ( strlen(log) > 0 )
        printf("Link Log: %s\n", log);
    

    glGenTextures(1, &encoder->texture_id);
    glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if(format == DXT_FORMAT_RGB) {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, encoder->width, encoder->height, 0, GL_RGB, GL_BYTE, NULL);
    } else {
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, encoder->width, encoder->height, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    }
    
    //glActiveTexture(GL_TEXTURE0);
    
    if(format == DXT_FORMAT_YUV422) {
            if(!dxt_prepare_yuv422_shader(encoder))
                return NULL;
    }

    glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
    //glClear(GL_COLOR_BUFFER_BIT);
 
    glViewport(0, 0, (encoder->width + 3) / 4, encoder->height / 4);
    glDisable(GL_DEPTH_TEST);
    
    // User compress program and set image size parameters
    glUseProgram(encoder->program_compress);
//glBindFragDataLocation(encoder->program_compress, 0, "colorInt");
    glUniform1i(glGetUniformLocation(encoder->program_compress, "image"), 0);
    
    if(format == DXT_FORMAT_YUV422 || format == DXT_FORMAT_YUV) {
            glUniform1i(glGetUniformLocation(encoder->program_compress, "imageFormat"), FORMAT_YUV);
    } else {
            glUniform1i(glGetUniformLocation(encoder->program_compress, "imageFormat"), FORMAT_RGB); 
    }
    glUniform2f(glGetUniformLocation(encoder->program_compress, "imageSize"), encoder->width, (encoder->height + 3) / 4 * 4); 
    glUniform1f(glGetUniformLocation(encoder->program_compress, "textureWidth"), (encoder->width + 3) / 4 * 4); 

    if(!encoder->legacy) {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
        GLint g_vertexLocation;

        // ToDo:
        glGenVertexArrays(1, &encoder->g_vao);

        // ToDo:
        glBindVertexArray(encoder->g_vao);

        // http://www.opengl.org/sdk/docs/man/xhtml/glGetAttribLocation.xml
        g_vertexLocation = glGetAttribLocation(encoder->program_compress, "position");

        // http://www.opengl.org/sdk/docs/man/xhtml/glGenBuffers.xml
        glGenBuffers(1, &encoder->g_vertices);

        // http://www.opengl.org/sdk/docs/man/xhtml/glBindBuffer.xml
        glBindBuffer(GL_ARRAY_BUFFER, encoder->g_vertices);

        // http://www.opengl.org/sdk/docs/man/xhtml/glBufferData.xml
        //glBufferData(GL_ARRAY_BUFFER, 3 * 4 * sizeof(GLfloat), (GLfloat*) points, GL_STATIC_DRAW);
        glBufferData(GL_ARRAY_BUFFER, 8 * 4 * sizeof(GLfloat), (GLfloat*) points, GL_STATIC_DRAW);

        // http://www.opengl.org/sdk/docs/man/xhtml/glUseProgram.xml
        //glUseProgram(g_program.program);

        // http://www.opengl.org/sdk/docs/man/xhtml/glVertexAttribPointer.xml
        glVertexAttribPointer(g_vertexLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

        // http://www.opengl.org/sdk/docs/man/xhtml/glEnableVertexAttribArray.xml
        glEnableVertexAttribArray(g_vertexLocation);

        glBindFragDataLocation(encoder->program_compress, 0, "colorOut");
#endif
    }

    // Render to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, encoder->fbo_id);

#ifdef HAVE_GPUPERFAPI
    GPA_BeginSession( &encoder->sessionID );
    GPA_GetPassCount( &encoder->numRequiredPasses );
    encoder->currentWaitSessionID = 0;
#endif
#ifdef RTDXT_DEBUG
    glGenQueries(4, encoder->queries);
#endif

    return encoder;
}

/** Documented at declaration */
int
dxt_encoder_buffer_allocate(struct dxt_encoder* encoder, unsigned char** image_compressed, int* image_compressed_size)
{
    int size = 0;
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )
        size = (encoder->width / 4) * (encoder->height / 4) * 4 * sizeof(int);
    else if ( encoder->type == DXT_TYPE_DXT1 )
        size = (encoder->width / 4) * (encoder->height / 4) * 2 * sizeof(int);
    else
        assert(0);
        
    *image_compressed = (unsigned char*)malloc(size);
    if ( *image_compressed == NULL )
        return -1;
        
    if ( image_compressed_size != NULL )
        *image_compressed_size = size;
        
    return 0;
}

/** Documented at declaration */
int
dxt_encoder_compress(struct dxt_encoder* encoder, DXT_IMAGE_TYPE* image, unsigned char* image_compressed)
{        
#ifdef RTDXT_DEBUG
    glBeginQuery(GL_TIME_ELAPSED_EXT, encoder->queries[0]);
#endif
#ifdef RTDXT_DEBUG_HOST
    TIMER_INIT();

    TIMER_START();
#endif
#ifdef HAVE_GPUPERFAPI
    GPA_BeginPass();
    GPA_BeginSample(0);
#endif
#ifdef USE_PBO_DXT_ENCODER
    GLubyte *ptr;
#endif

    int data_size = encoder->width * encoder->height;
    switch(encoder->format) {
            case DXT_FORMAT_YUV422:
                data_size *= 2;
                break;
            case DXT_FORMAT_RGB:
                data_size *= 3;
                break;
            case DXT_FORMAT_RGBA:
            case DXT_FORMAT_YUV:
                data_size *= 4;
                break;
    }

    switch(encoder->format) {
            case DXT_FORMAT_YUV422:
                        glBindFramebuffer(GL_FRAMEBUFFER, encoder->fbo444_id);
                        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, encoder->texture_id, 0);
                        glBindTexture(GL_TEXTURE_2D, encoder->texture_yuv422);
                        
                        glPushAttrib(GL_VIEWPORT_BIT);
                        glViewport( 0, 0, encoder->width, encoder->height);
                
#ifdef USE_PBO_DXT_ENCODER
                        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, encoder->pbo_in); // current pbo
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width / 2, encoder->height,  GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, 0);
                        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, data_size, 0, GL_STREAM_DRAW_ARB);
                        ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
                        if(ptr)
                        {
                            // update data directly on the mapped buffer
                            memcpy(ptr, image, data_size); 
                            glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release pointer to mapping buffer
                        }
#else
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width / 2, encoder->height,  GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, image);
#endif
                        glUseProgram(encoder->yuv422_to_444_program);
#ifdef RTDXT_DEBUG
    glEndQuery(GL_TIME_ELAPSED_EXT);
    glBeginQuery(GL_TIME_ELAPSED_EXT, encoder->queries[1]);
#endif
#ifdef HAVE_GPUPERFAPI
    GPA_EndSample();
    GPA_BeginSample(1);
#endif
                        
                        if(encoder->legacy) {
                            glBegin(GL_QUADS);
                            glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                            glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                            glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                            glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
                            glEnd();
                        } else {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
                            // Compress
                            glBindVertexArray(encoder->g_vao_422);
                            //glDrawElements(GL_TRIANGLE_STRIP, sizeof(m_quad.indices) / sizeof(m_quad.indices[0]), GL_UNSIGNED_SHORT, BUFFER_OFFSET(0));
                            glDrawArrays(GL_TRIANGLES, 0, 6);
                            glBindVertexArray(0);
#endif
                        }
                        
                        glPopAttrib();
                        /* there is some problem with restoring viewport state (Mac OS Lion, OpenGL 3.2) */
                        glViewport( 0, 0, (encoder->width + 3) / 4, encoder->height / 4);
                        
                        //glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
                        
                        glUseProgram(encoder->program_compress);
                        glBindFramebuffer(GL_FRAMEBUFFER, encoder->fbo_id);
                        glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
                
                        //gl_check_error();
                        break;
                case DXT_FORMAT_YUV:
                case DXT_FORMAT_RGBA:
                        glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
#ifdef USE_PBO_DXT_ENCODER
                        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, encoder->pbo_in); // current pbo
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width, encoder->height, GL_RGBA, DXT_IMAGE_GL_TYPE, 0);
                        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, data_size, 0, GL_STREAM_DRAW_ARB);
                        ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
                        if(ptr)
                        {
                            // update data directly on the mapped buffer
                            memcpy(ptr, image, data_size); 
                            glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release pointer to mapping buffer
                        }
#else
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width, encoder->height, GL_RGBA, DXT_IMAGE_GL_TYPE, image);
#endif

                        break;
                case DXT_FORMAT_RGB:
                        glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
#ifdef USE_PBO_DXT_ENCODER
                        glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, encoder->pbo_in); // current pbo
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width, encoder->height, GL_RGB, GL_UNSIGNED_BYTE, 0);
                        glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, data_size, 0, GL_STREAM_DRAW_ARB);
                        ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
                        if(ptr)
                        {
                            // update data directly on the mapped buffer
                            memcpy(ptr, image, data_size); 
                            glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release pointer to mapping buffer
                        }
#else
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width, encoder->height, GL_RGB, GL_UNSIGNED_BYTE, image);
#endif
                        break;
    }
#ifdef RTDXT_DEBUG
    glEndQuery(GL_TIME_ELAPSED_EXT);
    glBeginQuery(GL_TIME_ELAPSED_EXT, encoder->queries[2]);
#endif
#ifdef RTDXT_DEBUG_HOST
    glFinish();
    TIMER_STOP_PRINT("Tex Load (+->444): ");
    TIMER_START();
#endif

#ifdef HAVE_GPUPERFAPI
    GPA_EndSample();
    GPA_BeginSample(2);
#endif

    return dxt_encoder_compress_texture(encoder, encoder->texture_id, image_compressed);
}


int
dxt_encoder_compress_texture(struct dxt_encoder* encoder, int texture, unsigned char* image_compressed)
{
#ifdef USE_PBO_DXT_ENCODER
    GLubyte *ptr;
#endif

    glBindTexture(GL_TEXTURE_2D, texture);

    glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); 
    assert(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER));

    glClearColor(1,0,0,1);
    glClear(GL_COLOR_BUFFER_BIT);

    if(encoder->legacy) {
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();
    } else {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
        // Compress
        glBindVertexArray(encoder->g_vao);
        //glDrawElements(GL_TRIANGLE_STRIP, sizeof(m_quad.indices) / sizeof(m_quad.indices[0]), GL_UNSIGNED_SHORT, BUFFER_OFFSET(0));
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
#endif
    }
#ifdef HAVE_GPUPERFAPI
    GPA_EndSample();
    GPA_BeginSample(3);
#endif
#ifdef RTDXT_DEBUG
    glEndQuery(GL_TIME_ELAPSED_EXT);
    glBeginQuery(GL_TIME_ELAPSED_EXT, encoder->queries[3]);
#endif
#ifdef RTDXT_DEBUG_HOST
    glFinish();
    TIMER_STOP_PRINT("Texture Compress:  ");
    TIMER_START();
#endif


    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);

#ifdef USE_PBO_DXT_ENCODER
    // Read back
    // read pixels from framebuffer to PBO
    // glReadPixels() should return immediately.
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, encoder->pbo_out);
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )
        glReadPixels(0, 0, (encoder->width + 3) / 4, (encoder->height + 3) / 4, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, 0);
    else
        glReadPixels(0, 0, (encoder->width + 3) / 4, (encoder->height + 3) / 4 , GL_RGBA_INTEGER_EXT, GL_UNSIGNED_SHORT, 0);

    // map the PBO to process its data by CPU
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, encoder->pbo_out);
    ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB,
                                            GL_READ_ONLY_ARB);
    if(ptr)
    {
        memcpy(image_compressed, ptr, ((encoder->width + 3) / 4 * 4) * ((encoder->height + 3) / 4 * 4) / (encoder->type == DXT_TYPE_DXT5_YCOCG ? 1 : 2));
        glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);
    }

    // back to conventional pixel operation
    glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
#else
        glReadPixels(0, 0, (encoder->width + 3) / 4, (encoder->height + 3) / 4, GL_RGBA_INTEGER_EXT,
                        encoder->type == DXT_TYPE_DXT5_YCOCG ? GL_UNSIGNED_INT : GL_UNSIGNED_SHORT, image_compressed);
#endif

        
#ifdef RTDXT_DEBUG_HOST
    glFinish();
    TIMER_STOP_PRINT("Texture Save:      ");
#endif
#ifdef RTDXT_DEBUG
    glEndQuery(GL_TIME_ELAPSED_EXT);
    {
        GLint available = 0;
        GLuint64 load = 0,
                 convert = 0,
                 compress = 0,
                 store = 0;
        while (!available) {
            glGetQueryObjectiv(encoder->queries[3], GL_QUERY_RESULT_AVAILABLE, &available);
        }
        glGetQueryObjectui64vEXT(encoder->queries[0], GL_QUERY_RESULT, &load);
        glGetQueryObjectui64vEXT(encoder->queries[1], GL_QUERY_RESULT, &convert);
        glGetQueryObjectui64vEXT(encoder->queries[2], GL_QUERY_RESULT, &compress);
        glGetQueryObjectui64vEXT(encoder->queries[3], GL_QUERY_RESULT, &store);
        printf("Load: %8lu; YUV444->YUV422: %8lu; compress: %8lu; store: %8lu\n",
                load, convert, compress, store);
    }
#endif
#ifdef HAVE_GPUPERFAPI
    GPA_EndSample();
    GPA_EndPass();
#endif
    
    return 0;
}

/** Documented at declaration */
int
dxt_encoder_buffer_free(unsigned char* image_compressed)
{
    free(image_compressed);
    return 0;
}

/** Documented at declaration */
int
dxt_encoder_destroy(struct dxt_encoder* encoder)
{
#ifdef HAVE_GPUPERFAPI
    GPA_EndSession();
    bool readyResult = FALSE;
    if ( encoder->sessionID != encoder->currentWaitSessionID )
    {
        GPA_Status sessionStatus;
        sessionStatus = GPA_IsSessionReady( &readyResult,
                encoder->currentWaitSessionID );
        while ( sessionStatus == GPA_STATUS_ERROR_SESSION_NOT_FOUND )
        {
            // skipping a session which got overwritten
            encoder->currentWaitSessionID++;
            sessionStatus = GPA_IsSessionReady( &readyResult,
                    encoder->currentWaitSessionID );
        }
    }
    if ( readyResult )
    {
        WriteSession( encoder->currentWaitSessionID,
                "GPUPerfAPI-RTDXT-Results.csv" );
        encoder->currentWaitSessionID++;
    }

#endif
    glDeleteShader(encoder->shader_fragment_compress);
    glDeleteShader(encoder->shader_vertex_compress);
    glDeleteProgram(encoder->program_compress);
    free(encoder);
    return 0;
}

#ifdef HAVE_GPUPERFAPI
/// Given a sessionID, query the counter values and save them to a file
static void WriteSession( gpa_uint32 currentWaitSessionID,
        const char* filename )
{
    static bool doneHeadings = FALSE;
    gpa_uint32 count;
    GPA_GetEnabledCount( &count );
    FILE* f;
    if ( !doneHeadings )
    {
        const char* name;
        f = fopen( filename, "w" );
        assert( f );
        fprintf( f, "Identifier, " );
        for (gpa_uint32 counter = 0 ; counter < count ; counter++ )
        {
            gpa_uint32 enabledCounterIndex;
            GPA_GetEnabledIndex( counter, &enabledCounterIndex );
            GPA_GetCounterName( enabledCounterIndex, &name );
            fprintf( f, "%s, ", name );
        }
        fprintf( f, "\n" );
        fclose( f );
        doneHeadings = TRUE;
    }
    f = fopen( filename, "a+" );
    assert( f );
    gpa_uint32 sampleCount;
    GPA_GetSampleCount( currentWaitSessionID, &sampleCount );
    for (gpa_uint32 sample = 0 ; sample < sampleCount ; sample++ )
    {
        fprintf( f, "session: %d; sample: %d, ", currentWaitSessionID,
                sample );
        for (gpa_uint32 counter = 0 ; counter < count ; counter++ )
        {
            gpa_uint32 enabledCounterIndex;
            GPA_GetEnabledIndex( counter, &enabledCounterIndex );
            GPA_Type type;
            GPA_GetCounterDataType( enabledCounterIndex, &type );
            if ( type == GPA_TYPE_UINT32 )
            {
                gpa_uint32 value;
                GPA_GetSampleUInt32( currentWaitSessionID,
                        sample, enabledCounterIndex, &value );
                fprintf( f, "%u,", value );
            }
            else if ( type == GPA_TYPE_UINT64 )
            {
                gpa_uint64 value;
                GPA_GetSampleUInt64( currentWaitSessionID,
                        sample, enabledCounterIndex, &value );
                fprintf( f, "%lu,", value );
            }
            else if ( type == GPA_TYPE_FLOAT32 )
            {
                gpa_float32 value;
                GPA_GetSampleFloat32( currentWaitSessionID,
                        sample, enabledCounterIndex, &value );
                fprintf( f, "%f,", (double) value );
            }
            else if ( type == GPA_TYPE_FLOAT64 )
            {
                gpa_float64 value;
                GPA_GetSampleFloat64( currentWaitSessionID,
                        sample, enabledCounterIndex, &value );
                fprintf( f, "%f,", value );
            }
            else
            {
                assert(FALSE);
            }
        }
        fprintf( f, "\n" );
    }
    fclose( f );
}
#endif

/* vim: set expandtab: sw=4 */

