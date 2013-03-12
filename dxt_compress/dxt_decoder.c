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
#endif
#include "dxt_common.h"
#include "dxt_decoder.h"
#include "dxt_util.h"
#include "dxt_glsl.h"
#if defined HAVE_MACOSX && OS_VERSION_MAJOR >= 11
#include <OpenGL/gl3.h>
#endif

#if defined HAVE_MACOSX && OS_VERSION_MAJOR < 11
#define glGenFramebuffers glGenFramebuffersEXT
#define glBindFramebuffer glBindFramebufferEXT
#define GL_FRAMEBUFFER GL_FRAMEBUFFER_EXT
#define glFramebufferTexture2D glFramebufferTexture2DEXT
#define glDeleteFramebuffers glDeleteFramebuffersEXT
#endif

static GLfloat points[] = { -1.0f, -1.0f, 0.0f, 1.0f,
    1.0f, -1.0f, 0.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 1.0f,
    /* second triangle */
    1.0f, -1.0f, 0.0f, 1.0f,
    -1.0f, 1.0f, 0.0f, 1.0f,
    1.0f, 1.0f, 0.0f, 1.0f
};

/** Documented at declaration */ 
struct dxt_decoder
{
    // DXT type
    enum dxt_type type;
    enum dxt_format out_format;
    
    // Width in pixels
    int width;
    
    // Height in pixels
    int height;
    
    // Texture id
    GLuint compressed_tex; /* compressed */
    GLuint uncompressed_rgba_tex; /* decompressed */
    GLuint uncompressed_yuv422_tex;
    
    // Framebuffer
    GLuint fbo_rgba;
    GLuint fbo_yuv422;
    
    // Program and shader handles
    GLuint program_display;
    GLuint shader_fragment_display;
    GLuint shader_vertex_display;
    
    GLuint program_rgba_to_yuv422;
    GLuint shader_fragment_rgba_to_yuv422;
    GLuint shader_vertex_rgba_to_yuv422;

    /**
     * The VAO for the vertices etc.
     */
    GLuint g_vao;
    GLuint g_vao_422;

#ifdef USE_PBO_DXT_DECODER
    GLuint pbo_in;
    GLuint pbo_out;
#endif

    unsigned int legacy:1;
};

/** Documented at declaration */
struct dxt_decoder*
dxt_decoder_create(enum dxt_type type, int width, int height, enum dxt_format out_format, int legacy)
{
    struct dxt_decoder* decoder = (struct dxt_decoder*)malloc(sizeof(struct dxt_decoder));

#ifndef HAVE_MACOSX
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    assert(err == GLEW_OK);
#endif
    
    assert(out_format == DXT_FORMAT_RGBA || out_format == DXT_FORMAT_YUV422);
    assert(out_format == DXT_FORMAT_RGBA ||
        (out_format == DXT_FORMAT_YUV422 && width % 2 == 0)); /* width % 2 for YUV422 */

    if ( decoder == NULL )
        return NULL;
    decoder->type = type;
    decoder->width = width;
    decoder->height = height;
    decoder->out_format = out_format;
    decoder->legacy = legacy;
    //glutReshapeWindow(1, 1);
    
    if(legacy) {
            glEnable(GL_TEXTURE_2D);
    }
    
    // Create empty texture
    glGenTextures(1, &decoder->compressed_tex);
    glBindTexture(GL_TEXTURE_2D, decoder->compressed_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    
    GLenum internalformat;
    size_t size;

    if ( decoder->type == DXT_TYPE_DXT5_YCOCG ) {
            internalformat = GL_COMPRESSED_RGBA_S3TC_DXT5_EXT;
            size = ((decoder->width + 3) / 4 * 4) * ((decoder->height + 3) / 4 * 4);
    } else {
            internalformat = GL_COMPRESSED_RGB_S3TC_DXT1_EXT;
            size = ((decoder->width + 3) / 4 * 4) * ((decoder->height + 3) / 4 * 4) / 2;
    }

    GLvoid *tmp_data = malloc(size); // data pointer to glCompressedTexImage2D shouldn't be 0
    glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, internalformat, decoder->width, decoder->height, 0,
                    size, tmp_data);
    free(tmp_data);

    // Create fbo    
    glGenFramebuffers(1, &decoder->fbo_rgba);
    glBindFramebuffer(GL_FRAMEBUFFER, decoder->fbo_rgba);
    
    glGenTextures(1, &decoder->uncompressed_rgba_tex); 
    glBindTexture(GL_TEXTURE_2D, decoder->uncompressed_rgba_tex); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
    glTexImage2D(GL_TEXTURE_2D, 0 , DXT_IMAGE_GL_FORMAT, decoder->width, decoder->height, 0, GL_RGBA, DXT_IMAGE_GL_TYPE, 0); 
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, decoder->uncompressed_rgba_tex, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    
    // Create program [display] and its shader
    decoder->program_display = glCreateProgram();  
    // Create shader from file
    decoder->shader_fragment_display = 0;
    switch(decoder->type) 
    {
        case DXT_TYPE_DXT5_YCOCG:
                if(legacy)
                    decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display_dxt5ycocg_legacy, GL_FRAGMENT_SHADER);
                else
                    decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display_dxt5ycocg, GL_FRAGMENT_SHADER);
                break;
        case DXT_TYPE_DXT1:
                if(legacy)
                    decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display_legacy, GL_FRAGMENT_SHADER);
                else
                    decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display, GL_FRAGMENT_SHADER);
                break;
        case DXT_TYPE_DXT1_YUV:
                if(legacy)
                    decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display_dxt1_yuv_legacy, GL_FRAGMENT_SHADER);
                else
                    decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display_dxt1_yuv, GL_FRAGMENT_SHADER);
                break;
    }
    if(legacy) {
        decoder->shader_vertex_display = dxt_shader_create_from_source(vp_compress_legacy, GL_VERTEX_SHADER);
    } else {
        decoder->shader_vertex_display = dxt_shader_create_from_source(vp_compress, GL_VERTEX_SHADER);
    }
    if ( decoder->shader_fragment_display == 0 || decoder->shader_vertex_display == 0 )
        return NULL;
    // Attach shader to program and link the program
    glAttachShader(decoder->program_display, decoder->shader_fragment_display);
    glAttachShader(decoder->program_display, decoder->shader_vertex_display);
    glLinkProgram(decoder->program_display);
    
    if(out_format == DXT_FORMAT_YUV422) {
            glGenTextures(1, &decoder->uncompressed_yuv422_tex);
            glBindTexture(GL_TEXTURE_2D, decoder->uncompressed_yuv422_tex); 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); 
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
            glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA, decoder->width / 2, decoder->height, 0, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, 0); 
            
            glGenFramebuffers(1, &decoder->fbo_yuv422);
            
            if(legacy) {
                decoder->shader_fragment_rgba_to_yuv422 = dxt_shader_create_from_source(fp_display_rgba_to_yuv422_legacy, GL_FRAGMENT_SHADER);
                decoder->shader_vertex_rgba_to_yuv422 = dxt_shader_create_from_source(vp_compress_legacy, GL_VERTEX_SHADER);
            } else {
                decoder->shader_fragment_rgba_to_yuv422 = dxt_shader_create_from_source(fp_display_rgba_to_yuv422, GL_FRAGMENT_SHADER);
                decoder->shader_vertex_rgba_to_yuv422 = dxt_shader_create_from_source(vp_compress, GL_VERTEX_SHADER);
            }
            
            decoder->program_rgba_to_yuv422 = glCreateProgram();
            glAttachShader(decoder->program_rgba_to_yuv422, decoder->shader_fragment_rgba_to_yuv422);
            glAttachShader(decoder->program_rgba_to_yuv422, decoder->shader_vertex_rgba_to_yuv422);
            glLinkProgram(decoder->program_rgba_to_yuv422);
            
            glUseProgram(decoder->program_rgba_to_yuv422);
            glUniform1i(glGetUniformLocation(decoder->program_rgba_to_yuv422, "image"), 0);
            glUniform1f(glGetUniformLocation(decoder->program_rgba_to_yuv422, "imageWidth"),
                        (GLfloat) decoder->width);

            if(!decoder->legacy) {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
                GLint g_vertexLocation;
                GLuint g_vertices;

                // ToDo:
                glGenVertexArrays(1, &decoder->g_vao_422);

                // ToDo:
                glBindVertexArray(decoder->g_vao_422);

                // http://www.opengl.org/sdk/docs/man/xhtml/glGetAttribLocation.xml
                g_vertexLocation = glGetAttribLocation(decoder->program_rgba_to_yuv422, "position");

                // http://www.opengl.org/sdk/docs/man/xhtml/glGenBuffers.xml
                glGenBuffers(1, &g_vertices);

                // http://www.opengl.org/sdk/docs/man/xhtml/glBindBuffer.xml
                glBindBuffer(GL_ARRAY_BUFFER, g_vertices);

                // http://www.opengl.org/sdk/docs/man/xhtml/glBufferData.xml
                //glBufferData(GL_ARRAY_BUFFER, 3 * 4 * sizeof(GLfloat), (GLfloat*) points, GL_STATIC_DRAW);
                glBufferData(GL_ARRAY_BUFFER, 8 * 4 * sizeof(GLfloat), (GLfloat*) points, GL_STATIC_DRAW);

                // http://www.opengl.org/sdk/docs/man/xhtml/glUseProgram.xml
                glUseProgram(decoder->program_rgba_to_yuv422);

                // http://www.opengl.org/sdk/docs/man/xhtml/glVertexAttribPointer.xml
                glVertexAttribPointer(g_vertexLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

                // http://www.opengl.org/sdk/docs/man/xhtml/glEnableVertexAttribArray.xml
                glEnableVertexAttribArray(g_vertexLocation);

                glBindFragDataLocation(decoder->program_rgba_to_yuv422, 0, "colorOut");
#endif
            }
                        
            glUseProgram(0);
    }
    if(!decoder->legacy) {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
        GLint g_vertexLocation;
        GLuint g_vertices;

        // ToDo:
        glGenVertexArrays(1, &decoder->g_vao);

        // ToDo:
        glBindVertexArray(decoder->g_vao);

        // http://www.opengl.org/sdk/docs/man/xhtml/glGetAttribLocation.xml
        g_vertexLocation = glGetAttribLocation(decoder->program_display, "position");

        // http://www.opengl.org/sdk/docs/man/xhtml/glGenBuffers.xml
        glGenBuffers(1, &g_vertices);

        // http://www.opengl.org/sdk/docs/man/xhtml/glBindBuffer.xml
        glBindBuffer(GL_ARRAY_BUFFER, g_vertices);

        // http://www.opengl.org/sdk/docs/man/xhtml/glBufferData.xml
        //glBufferData(GL_ARRAY_BUFFER, 3 * 4 * sizeof(GLfloat), (GLfloat*) points, GL_STATIC_DRAW);
        glBufferData(GL_ARRAY_BUFFER, 8 * 4 * sizeof(GLfloat), (GLfloat*) points, GL_STATIC_DRAW);

        // http://www.opengl.org/sdk/docs/man/xhtml/glUseProgram.xml
        glUseProgram(decoder->program_display);

        // http://www.opengl.org/sdk/docs/man/xhtml/glVertexAttribPointer.xml
        glVertexAttribPointer(g_vertexLocation, 4, GL_FLOAT, GL_FALSE, 0, 0);

        // http://www.opengl.org/sdk/docs/man/xhtml/glEnableVertexAttribArray.xml
        glEnableVertexAttribArray(g_vertexLocation);

        glBindFragDataLocation(decoder->program_display, 0, "colorOut");

        glUseProgram(0);
#endif


#ifdef USE_PBO_DXT_DECODER
            glGenBuffersARB(1, &decoder->pbo_in); //Allocate PBO
            glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, decoder->pbo_in);
            glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB,
                            ((width + 3) / 4 * 4) * ((height + 3) / 4 * 4)  / (decoder->type == DXT_TYPE_DXT5_YCOCG ? 1 : 2),
                            0,
                            GL_STREAM_DRAW_ARB);
            glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

            glGenBuffersARB(1, &decoder->pbo_out); //Allocate PBO
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, decoder->pbo_out);
            glBufferDataARB(GL_PIXEL_PACK_BUFFER_ARB, ((width + 3) / 4 * 4) * ((height + 3) / 4 * 4)  * (out_format == DXT_FORMAT_YUV422 ? 2 : 4), 0, GL_STREAM_READ_ARB);
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
#endif

    }
    
    return decoder;
}

/** Documented at declaration */
int
dxt_decoder_buffer_allocate(struct dxt_decoder* decoder, unsigned char** image, int* image_size)
{
    *image_size = decoder->width * decoder->height * 4 * sizeof(DXT_IMAGE_TYPE);
    *image = (DXT_IMAGE_TYPE*)malloc(*image_size);
    if ( *image == NULL )
        return -1;
    return 0;
}

/** Documented at declaration */
int
dxt_decoder_decompress(struct dxt_decoder* decoder, unsigned char* image_compressed, DXT_IMAGE_TYPE* image)
{    
#ifdef USE_PBO_DXT_DECODER
        GLubyte *ptr;
#endif

#ifdef RTDXT_DEBUG
    TIMER_INIT();
    
    TIMER_START();
#endif
    glBindTexture(GL_TEXTURE_2D, decoder->compressed_tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        int data_size = ((decoder->width + 3) / 4 * 4) * ((decoder->height + 3) / 4 * 4) /
                               (decoder->type == DXT_TYPE_DXT5_YCOCG ? 1 : 2);
#ifdef USE_PBO_DXT_DECODER
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, decoder->pbo_in); // current pbo
    glCompressedTexSubImage2DARB (GL_TEXTURE_2D, 0, 0, 0,
                    (decoder->width + 3) / 4 * 4,
                    (decoder->height + 3) / 4 * 4,
                    decoder->type == DXT_TYPE_DXT5_YCOCG ? GL_COMPRESSED_RGBA_S3TC_DXT5_EXT : GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
                    data_size,
                    0);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, data_size, 0, GL_STREAM_DRAW_ARB);
    ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB);
    if(ptr)
    {
            // update data directly on the mapped buffer
            memcpy(ptr, image_compressed, data_size); 
            glUnmapBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB); // release pointer to mapping buffer
    }
#else
    glCompressedTexSubImage2DARB (GL_TEXTURE_2D, 0, 0, 0,
                    (decoder->width + 3) / 4 * 4,
                    (decoder->height + 3) / 4 * 4,
                    decoder->type == DXT_TYPE_DXT5_YCOCG ? GL_COMPRESSED_RGBA_S3TC_DXT5_EXT : GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
                    data_size,
                    image_compressed);
#endif


#ifdef RTDXT_DEBUG
    glFinish();
    TIMER_STOP_PRINT("Texture Load:      ");
    
    TIMER_START();
#endif
    
    // Render to framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, decoder->fbo_rgba);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, decoder->width, decoder->height);

    glUseProgram(decoder->program_display);
    
    glBindTexture(GL_TEXTURE_2D, decoder->compressed_tex);
    
    if(decoder->legacy) {
        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
        glEnd();
    } else {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
        glBindVertexArray(decoder->g_vao);
        glDrawArrays(GL_TRIANGLES, 0, 6);
        glBindVertexArray(0);
#endif
    }
    
    glUseProgram(0);
#ifdef RTDXT_DEBUG
    glFinish();
    TIMER_STOP_PRINT("Texture Decompress:");
    
    TIMER_START();
#endif
    if(decoder->out_format != DXT_FORMAT_YUV422) { /* so RGBA */
            // Disable framebuffer

#ifdef USE_PBO_DXT_DECODER
            // glReadPixels() should return immediately.
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, decoder->pbo_out);
            glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glReadPixels(0, 0, decoder->width, decoder->height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, 0);

            // map the PBO to process its data by CPU
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, decoder->pbo_out);
            ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB,
                            GL_READ_ONLY_ARB);
            if(ptr)
            {
                    memcpy(image, ptr, decoder->width * decoder->height * 4);
                    glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);
            }

            // back to conventional pixel operation
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
#else
            glReadPixels(0, 0, decoder->width, decoder->height, GL_RGBA, DXT_IMAGE_GL_TYPE, image);
#endif

            glBindFramebuffer(GL_FRAMEBUFFER, 0);
    } else {

            glBindFramebuffer(GL_FRAMEBUFFER, decoder->fbo_yuv422);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, decoder->uncompressed_yuv422_tex, 0);
            
            glBindTexture(GL_TEXTURE_2D, decoder->uncompressed_rgba_tex); /* to texturing unit 0 */

            glPushAttrib(GL_VIEWPORT_BIT);
            glViewport( 0, 0, decoder->width / 2, decoder->height);

            glUseProgram(decoder->program_rgba_to_yuv422);
            
            if(decoder->legacy) {
                glBegin(GL_QUADS);
                glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
                glEnd();
            } else {
#if ! defined HAVE_MACOSX || OS_VERSION_MAJOR >= 11
                glBindVertexArray(decoder->g_vao_422);
                glDrawArrays(GL_TRIANGLES, 0, 6);
                glBindVertexArray(0);
#endif
            }
            
            glPopAttrib();

#ifdef USE_PBO_DXT_DECODER
            // glReadPixels() should return immediately.
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, decoder->pbo_out);
            glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
            glReadPixels(0, 0, decoder->width / 2, decoder->height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, 0);

            // map the PBO to process its data by CPU
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, decoder->pbo_out);
            ptr = (GLubyte*)glMapBufferARB(GL_PIXEL_PACK_BUFFER_ARB,
                            GL_READ_ONLY_ARB);
            if(ptr)
            {
                    memcpy(image, ptr, decoder->width  * decoder->height * (decoder->out_format == DXT_FORMAT_YUV422 ? 2 : 4));
                    glUnmapBufferARB(GL_PIXEL_PACK_BUFFER_ARB);
            }

            // back to conventional pixel operation
            glBindBufferARB(GL_PIXEL_PACK_BUFFER_ARB, 0);
#else
            glReadPixels(0, 0, decoder->width / 2, decoder->height, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, image);
#endif

            //glReadPixels(0, 0, decoder->width / 2, decoder->height, GL_RGBA, DXT_IMAGE_GL_TYPE, image);
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glUseProgram(0);
    }
    
#ifdef RTDXT_DEBUG    
    glFinish();
    TIMER_STOP_PRINT("Texture Save:      ");
#endif
    
    return 0;
}

/** Documented at declaration */
int
dxt_decoder_buffer_free(unsigned char* image)
{
    free(image);
    return 0;
}

/** Documented at declaration */
int
dxt_decoder_destroy(struct dxt_decoder* decoder)
{
    glDeleteShader(decoder->shader_fragment_display);
    glDeleteProgram(decoder->program_display);
    glDeleteTextures(1, &decoder->compressed_tex);
    glDeleteTextures(1, &decoder->uncompressed_rgba_tex);
    glDeleteFramebuffers(1,  &decoder->fbo_rgba);
    
    if(decoder->out_format == DXT_FORMAT_YUV422) {
        glDeleteTextures(1, &decoder->uncompressed_yuv422_tex);
        glDeleteFramebuffers(1,  &decoder->fbo_yuv422);
    }
    free(decoder);
    return 0;
}
 
