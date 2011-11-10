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
#include "dxt_decoder.h"
#include "dxt_util.h"
#include "dxt_glsl.h"

/** Documented at declaration */ 
struct dxt_decoder
{
    // DXT type
    enum dxt_type type;
    
    // Width in pixels
    int width;
    
    // Height in pixels
    int height;
    
    // Texture id
    GLuint texture_id;
    
    // Framebuffer
    GLuint fbo_id;
    
    // Program and shader handles
    GLhandleARB program_display;
    GLhandleARB shader_fragment_display;
};

/** Documented at declaration */
struct dxt_decoder*
dxt_decoder_create(enum dxt_type type, int width, int height)
{
    struct dxt_decoder* decoder = (struct dxt_decoder*)malloc(sizeof(struct dxt_decoder));
    if ( decoder == NULL )
        return NULL;
    decoder->type = type;
    decoder->width = width;
    decoder->height = height;
    //glutReshapeWindow(1, 1);
    
    // Create empty texture
    glGenTextures(1, &decoder->texture_id);
    glBindTexture(GL_TEXTURE_2D, decoder->texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if ( decoder->type == DXT_TYPE_DXT5_YCOCG )
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, decoder->width, decoder->height, 0, 0, 0);
    else
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, decoder->width, decoder->height, 0, 0, 0);
        
    // Create fbo    
    glGenFramebuffersEXT(1, &decoder->fbo_id);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, decoder->fbo_id);
    
    GLuint fbo_tex;
    glGenTextures(1, &fbo_tex); 
    glBindTexture(GL_TEXTURE_2D, fbo_tex); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP); 
    glTexImage2D(GL_TEXTURE_2D, 0 , DXT_IMAGE_GL_FORMAT, decoder->width, decoder->height, 0, GL_RGBA, DXT_IMAGE_GL_TYPE, 0); 
    glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, fbo_tex, 0);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    
    // Create program [display] and its shader
    decoder->program_display = glCreateProgramObjectARB();  
    // Create shader from file
    decoder->shader_fragment_display = 0;
    switch(decoder->type) 
    {
        case DXT_TYPE_DXT5_YCOCG:
                decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display_dxt5ycocg, GL_FRAGMENT_SHADER_ARB);
                break;
        case DXT_TYPE_DXT1:
                decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display, GL_FRAGMENT_SHADER_ARB);
                break;
        case DXT_TYPE_DXT1_YUV:
                decoder->shader_fragment_display = dxt_shader_create_from_source(fp_display_dxt1_yuv, GL_FRAGMENT_SHADER_ARB);
                break;
    }
    if ( decoder->shader_fragment_display == 0 )
        return NULL;
    // Attach shader to program and link the program
    glAttachObjectARB(decoder->program_display, decoder->shader_fragment_display);
    glLinkProgramARB(decoder->program_display);
    
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
    TIMER_INIT();
    
    TIMER_START();
    glBindTexture(GL_TEXTURE_2D, decoder->texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if ( decoder->type == DXT_TYPE_DXT5_YCOCG )
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
                        decoder->width, decoder->height, 0, decoder->width * decoder->height, image_compressed);
    else
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
                        decoder->width, decoder->height, 0, decoder->width * decoder->height / 2, image_compressed);
    glFinish();
    TIMER_STOP_PRINT("Texture Load:      ");
    
    TIMER_START();
    
    // Render to framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, decoder->fbo_id);
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0, 0, decoder->width, decoder->height);

    glUseProgramObjectARB(decoder->program_display);
    
    glBindTexture(GL_TEXTURE_2D, decoder->texture_id);
    
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
    glEnd();
    
    glUseProgramObjectARB(0);
    glFinish();
    TIMER_STOP_PRINT("Texture Decompress:");
    
    TIMER_START();
    glReadPixels(0, 0, decoder->width, decoder->height, GL_RGBA, DXT_IMAGE_GL_TYPE, image);
    
    // Disable framebuffer
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
    glFinish();
    TIMER_STOP_PRINT("Texture Save:      ");
    
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
    free(decoder);
    return 0;
}
 
