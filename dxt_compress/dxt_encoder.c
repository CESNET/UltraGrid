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

#include "dxt_encoder.h"
#include "dxt_util.h"
#include "dxt_glsl.h"

#include <string.h>


const int FORMAT_RGB = 0;
const int FORMAT_YUV = 1;

/** Documented at declaration */ 
struct dxt_encoder
{
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
    
    // Compressed texture
    GLuint texture_compressed_id;
    
    GLuint texture_yuv422;
    
    // Framebuffer
    GLuint fbo_id;
  
    // Program and shader handles
    GLhandleARB program_compress;
    GLhandleARB shader_fragment_compress;
    GLhandleARB shader_vertex_compress;

    GLhandleARB yuv422_to_444_program;
    GLhandleARB yuv422_to_444_fp;
    GLuint fbo444_id;
};

int dxt_prepare_yuv422_shader(struct dxt_encoder *encoder);

int dxt_prepare_yuv422_shader(struct dxt_encoder *encoder) {
        encoder->yuv422_to_444_fp = 0;    
        encoder->yuv422_to_444_fp = dxt_shader_create_from_source(fp_yuv422_to_yuv_444, GL_FRAGMENT_SHADER_ARB);
        if ( encoder->yuv422_to_444_fp == 0) {
                printf("Failed to compile YUV422->YUV444 fragment program!\n");
                return 0;
        }
        
        encoder->yuv422_to_444_program = glCreateProgramObjectARB();
        glAttachObjectARB(encoder->yuv422_to_444_program, encoder->shader_vertex_compress);
        glAttachObjectARB(encoder->yuv422_to_444_program, encoder->yuv422_to_444_fp);
        glLinkProgramARB(encoder->yuv422_to_444_program);
        
        char log[32768];
        glGetInfoLogARB(encoder->yuv422_to_444_program, 32768, NULL, (GLchar*)log);
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
        
        glUseProgramObjectARB(encoder->yuv422_to_444_program);
        glUniform1i(glGetUniformLocation(encoder->yuv422_to_444_program, "image"), 0);
        glUniform1f(glGetUniformLocation(encoder->yuv422_to_444_program, "imageWidth"),
                        (GLfloat) encoder->width);

        // Create fbo    
        glGenFramebuffersEXT(1, &encoder->fbo444_id);
        return 1;
}


/** Documented at declaration */
struct dxt_encoder*
dxt_encoder_create(enum dxt_type type, int width, int height, enum dxt_format format)
{
    struct dxt_encoder* encoder = (struct dxt_encoder*)malloc(sizeof(struct dxt_encoder));
    if ( encoder == NULL )
        return NULL;
    encoder->type = type;
    encoder->width = width;
    encoder->height = height;
    encoder->format = format;
    
    // Create empty data
    /*GLubyte * data = NULL;
    int data_size = 0;
    dxt_encoder_buffer_allocate(encoder, &data, &data_size);
    // Create empty compressed texture
    glGenTextures(1, &encoder->texture_compressed_id);
    glBindTexture(GL_TEXTURE_2D, encoder->texture_compressed_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); 
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )   
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, encoder->width, encoder->height, 0, data_size, data);
    else
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, encoder->width, encoder->height, 0, data_size, data);
    // Free empty data
    dxt_encoder_buffer_free(data);*/
    
    // Create fbo    
    glGenFramebuffersEXT(1, &encoder->fbo_id);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, encoder->fbo_id);
    
    GLuint fbo_tex;
    glGenTextures(1, &fbo_tex); 
    glBindTexture(GL_TEXTURE_2D, fbo_tex); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_RGBA32UI_EXT, encoder->width / 4, encoder->height / 4, 0, GL_RGBA_INTEGER_EXT, GL_INT, 0); 
    else
        glTexImage2D(GL_TEXTURE_2D, 0 , GL_LUMINANCE_ALPHA32UI_EXT, encoder->width / 4, encoder->height / 4, 0, GL_LUMINANCE_ALPHA_INTEGER_EXT, GL_INT, 0); 
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, fbo_tex, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    
    // Create program [display] and its shader
    encoder->program_compress = glCreateProgramObjectARB();
    // Create fragment shader from file
    encoder->shader_fragment_compress = 0;
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )
        encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt5ycocg, GL_FRAGMENT_SHADER_ARB);
    else
        encoder->shader_fragment_compress = dxt_shader_create_from_source(fp_compress_dxt1, GL_FRAGMENT_SHADER_ARB);
    if ( encoder->shader_fragment_compress == 0 )
        return NULL;
    // Create vertex shader from file
    encoder->shader_vertex_compress = 0;
    encoder->shader_vertex_compress = dxt_shader_create_from_source(vp_compress, GL_VERTEX_SHADER_ARB);
    if ( encoder->shader_vertex_compress == 0 ) {
        printf("Failed to compile vertex compress program!\n");
        return NULL;
    }
    // Attach shader to program and link the program
    glAttachObjectARB(encoder->program_compress, encoder->shader_fragment_compress);
    glAttachObjectARB(encoder->program_compress, encoder->shader_vertex_compress);
    glLinkProgramARB(encoder->program_compress);
    
    char log[32768];
    glGetInfoLogARB(encoder->program_compress, 32768, NULL, (GLchar*)log);
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
    glEnable(GL_TEXTURE_2D);
    
    if(format == DXT_FORMAT_YUV422) {
            if(!dxt_prepare_yuv422_shader(encoder))
                return NULL;
    }

    glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
    glClear(GL_COLOR_BUFFER_BIT);
 
    glViewport(0, 0, encoder->width / 4, encoder->height / 4);
    glDisable(GL_DEPTH_TEST);
    
    // User compress program and set image size parameters
    glUseProgramObjectARB(encoder->program_compress);
    glUniform1i(glGetUniformLocation(encoder->program_compress, "image"), 0);
    
    if(format == DXT_FORMAT_YUV422 || format == DXT_FORMAT_YUV) {
            glUniform1i(glGetUniformLocation(encoder->program_compress, "imageFormat"), FORMAT_YUV);
    } else {
            glUniform1i(glGetUniformLocation(encoder->program_compress, "imageFormat"), FORMAT_RGB); 
    }
    glUniform2f(glGetUniformLocation(encoder->program_compress, "imageSize"), encoder->width, encoder->height); 

    // Render to framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, encoder->fbo_id);

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
    TIMER_INIT();
 
    TIMER_START();
    switch(encoder->format) {
            case DXT_FORMAT_YUV422:
                        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, encoder->fbo444_id);
                        glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, encoder->texture_id, 0);
                        //assert(GL_FRAMEBUFFER_COMPLETE_EXT == glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT));
                        glBindTexture(GL_TEXTURE_2D, encoder->texture_yuv422);
                        
                        glPushAttrib(GL_VIEWPORT_BIT);
                        glViewport( 0, 0, encoder->width, encoder->height);
                
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width / 2, encoder->height,  GL_RGBA, GL_UNSIGNED_INT_8_8_8_8_REV, image);
                        glUseProgramObjectARB(encoder->yuv422_to_444_program);
                        
                        //glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); 
                        
                        glBegin(GL_QUADS);
                        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
                        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
                        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
                        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
                        glEnd();
                        
                        glPopAttrib();
                        
                        //glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
                        
                        glUseProgramObjectARB(encoder->program_compress);
                        glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, encoder->fbo_id);
                        glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
                
                        //gl_check_error();
                        break;
                case DXT_FORMAT_YUV:
                case DXT_FORMAT_RGBA:
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width, encoder->height, GL_RGBA, DXT_IMAGE_GL_TYPE, image);
                        break;
                case DXT_FORMAT_RGB:
                        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, encoder->width, encoder->height, GL_RGB, GL_UNSIGNED_BYTE, image);
                        break;
    }
                        
#ifdef DEBUG
    glFinish();
#endif
    TIMER_STOP_PRINT("Texture Load:      ");
    
    TIMER_START();
    
    // Compress
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
    glEnd();
        
#ifdef DEBUG
    glFinish();
#endif
    TIMER_STOP_PRINT("Texture Compress:  ");
            
    TIMER_START();
    // Read back
    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )
        glReadPixels(0, 0, encoder->width / 4, encoder->height / 4, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, image_compressed);
    else
        glReadPixels(0, 0, encoder->width / 4, encoder->height / 4, GL_LUMINANCE_ALPHA_INTEGER_EXT, GL_UNSIGNED_INT, image_compressed);
        
#ifdef DEBUG
    glFinish();
#endif
    TIMER_STOP_PRINT("Texture Save:      ");
    
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
    glDeleteShader(encoder->shader_fragment_compress);
    glDeleteShader(encoder->shader_vertex_compress);
    glDeleteProgram(encoder->program_compress);
    free(encoder);
    return 0;
}
