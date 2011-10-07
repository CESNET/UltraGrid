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
    
    // Framebuffer
    GLuint fbo_id;
  
    // Program and shader handles
    GLhandleARB program_compress;
    GLhandleARB shader_fragment_compress;
    GLhandleARB shader_vertex_compress;    
};

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
    
    // Create empty compressed texture
    glGenTextures(1, &encoder->texture_compressed_id);
    glBindTexture(GL_TEXTURE_2D, encoder->texture_compressed_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, encoder->width, encoder->height, 0, 0, 0);

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
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, DXT_IMAGE_GL_FORMAT, encoder->width, encoder->height, 0, GL_RGBA, DXT_IMAGE_GL_TYPE, NULL);
    
    return encoder;
}

/** Documented at declaration */
int
dxt_encoder_buffer_allocate(struct dxt_encoder* encoder, unsigned char** image_compressed, int* image_compressed_size)
{
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )
        *image_compressed_size = (encoder->width / 4) * (encoder->height / 4) * 4 * sizeof(int);
    else if ( encoder->type == DXT_TYPE_DXT1 )
        *image_compressed_size = (encoder->width / 4) * (encoder->height / 4) * 2 * sizeof(int);
    else
        assert(0);
        
    *image_compressed = (unsigned char*)malloc(*image_compressed_size);
    if ( *image_compressed == NULL )
        return -1;
        
    return 0;
}

/** Documented at declaration */
int
dxt_encoder_compress(struct dxt_encoder* encoder, DXT_IMAGE_TYPE* image, unsigned char* image_compressed)
{        
    TIMER_INIT();
    
    TIMER_START();
    glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
    // TODO: Zkusi udelat nasledujic zmenu navhovanou Martinem Pulcem:
    // jo prvne jsem si bindl data s glTexImage2D a pak uz pro kazdy frame glTexSubImage2D, to by mohlo byt rychlejsi
    glTexImage2D(GL_TEXTURE_2D, 0, DXT_IMAGE_GL_FORMAT, encoder->width, encoder->height, 0, GL_RGBA, DXT_IMAGE_GL_TYPE, image);
    TIMER_STOP_PRINT("Texture Load:      ");
    
    TIMER_START();
    
    glClear(GL_COLOR_BUFFER_BIT);
    
    // Render to framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, encoder->fbo_id);
    
    // Compress
    glViewport(0, 0, encoder->width / 4, encoder->height / 4);
    glDisable(GL_DEPTH_TEST);
    
    // User compress program and set image size parameters
    glUseProgramObjectARB(encoder->program_compress);
    glUniform1i(glGetUniformLocation(encoder->program_compress, "imageFormat"), encoder->format); 
    glUniform2f(glGetUniformLocation(encoder->program_compress, "imageSize"), encoder->width, encoder->height); 
        
    glBindTexture(GL_TEXTURE_2D, encoder->texture_id);
        
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
    glEnd();
        
    glUseProgramObjectARB(0);
    TIMER_STOP_PRINT("Texture Compress:  ");
            
    TIMER_START();
    // Read back
    glReadBuffer(GL_COLOR_ATTACHMENT0_EXT);
    if ( encoder->type == DXT_TYPE_DXT5_YCOCG )
        glReadPixels(0, 0, encoder->width / 4, encoder->height / 4, GL_RGBA_INTEGER_EXT, GL_UNSIGNED_INT, image_compressed);
    else
        glReadPixels(0, 0, encoder->width / 4, encoder->height / 4, GL_LUMINANCE_ALPHA_INTEGER_EXT, GL_UNSIGNED_INT, image_compressed);
        
    // Disable framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
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
