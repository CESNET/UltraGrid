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

/** Documented at declaration */ 
struct dxt_display
{
    // DXT type
    enum dxt_type type;
    
    // Width in pixels
    int width;
    
    // Height in pixels
    int height;
    
    // Texture id
    GLuint texture_id;
    
    // Comperessed texture id
    GLuint texture_compressed_id;
    
    // Current texture id
    GLuint texture_current_id;

    // CG context, profiles, programs
    CGcontext context;
    CGprofile profile_fragment;
    CGprogram program_fragment_display;
    
    // Show
    int show;
};

/**
 * Current DXT decoder
 */
struct dxt_display g_display;

/**
 * OpenGL render
 * 
 * @return void
 */
void
dxt_display_render(void)
{            
    glClear(GL_COLOR_BUFFER_BIT);
    glViewport(0, 0, glutGet(GLUT_WINDOW_WIDTH), glutGet(GLUT_WINDOW_HEIGHT));

    cgGLBindProgram(g_display.program_fragment_display);
    cgGLEnableProfile(g_display.profile_fragment);
    
    if ( g_display.texture_id != 0 && (g_display.texture_current_id == 0 || 
         g_display.texture_current_id == g_display.texture_id ) ) {
        glBindTexture(GL_TEXTURE_2D, g_display.texture_id);
        CGparameter parameter = cgGetNamedParameter(g_display.program_fragment_display, "reconstructColor");
        cgGLSetParameter1f(parameter, 0);
        if ( g_display.texture_current_id != 0)
            glutSetWindowTitle("Display Image");
    }
    if ( g_display.texture_compressed_id != 0 && (g_display.texture_current_id == 0 || 
         g_display.texture_current_id == g_display.texture_compressed_id ) ) {
        glBindTexture(GL_TEXTURE_2D, g_display.texture_compressed_id);
        CGparameter parameter = cgGetNamedParameter(g_display.program_fragment_display, "reconstructColor");
        if ( g_display.type == DXT5_YCOCG )
            cgGLSetParameter1f(parameter, 1);
        else
            cgGLSetParameter1f(parameter, 0);
        if ( g_display.texture_current_id != 0)
            glutSetWindowTitle("Display Compressed Image");
    }
    
    glBegin(GL_QUADS);
    glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, -1.0);
    glTexCoord2f(1.0, 0.0); glVertex2f(1.0, -1.0);
    glTexCoord2f(1.0, 1.0); glVertex2f(1.0, 1.0);
    glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, 1.0);
    glEnd();
    
    cgGLDisableProfile(g_display.profile_fragment);
    
	glutSwapBuffers();
}

/**
 * OpenGL handle keyboard
 * 
 * @return void
 */
void
dxt_display_keyboard(unsigned char key, int x, int y)
{
    switch ( key ) {
    case 9: // [Tab] Change texture
        if ( g_display.texture_current_id != 0 ) {
            if ( g_display.texture_current_id == g_display.texture_id && g_display.texture_compressed_id != 0 )
                g_display.texture_current_id = g_display.texture_compressed_id;
            else if ( g_display.texture_current_id == g_display.texture_compressed_id && g_display.texture_id != 0 )
                g_display.texture_current_id = g_display.texture_id;
            glutPostRedisplay();
        }
        break;
    case 27: // [ESC] Quit
        g_display.show = 0;
        return;
    }
}

/**
 * OpenGL init
 * 
 * @return void
 */
void
dxt_display_init(const char* title, enum dxt_type type, int width, int height)
{
    g_display.type = type;
    g_display.width = width;
    g_display.height = height;
    g_display.texture_id = 0;
    g_display.texture_compressed_id = 0;
    g_display.texture_current_id = 0;
    
    int argc = 0;
    if ( title != NULL )
        glutSetWindowTitle(title);
    glutReshapeWindow(width, height);
    glutDisplayFunc(dxt_display_render); 
    glutKeyboardFunc(dxt_display_keyboard);
    glutShowWindow();
    
    // Init CG
    g_display.context = cgCreateContext();
    cgCheckError(g_display.context, "creating context");
    g_display.profile_fragment = cgGLGetLatestProfile(CG_GL_FRAGMENT); 
    cgGLSetOptimalOptions(g_display.profile_fragment); 
    cgCheckError(g_display.context, "selecting fragment profile"); 

    g_display.program_fragment_display = cgCreateProgramFromFile(g_display.context, CG_SOURCE, "dxt.cg", g_display.profile_fragment, "display_fp", NULL);
    cgCheckError(g_display.context, "creating fragment program from file"); 
    cgGLLoadProgram(g_display.program_fragment_display); 
    cgCheckError(g_display.context, "loading fragment program");
}

/**
 * OpenGL run loop
 * 
 * @return void
 */
void
dxt_display_run()
{
    g_display.show = 1;
    glutPostRedisplay();
    while( g_display.show ) {
        glutMainLoopEvent();
    }
    glutHideWindow();
}

/** Documented at declaration */
int
dxt_display_image(const char* title, DXT_IMAGE_TYPE* image, int width, int height)
{
    printf("Display Image [resolution: %dx%d]\n", width, height);
    
    dxt_display_init(title, DXT1, width, height);
    
    glGenTextures(1, &g_display.texture_id);
    glBindTexture(GL_TEXTURE_2D, g_display.texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, DXT_IMAGE_GL_FORMAT, width, height, 0, GL_RGBA, DXT_IMAGE_GL_TYPE, image);
    
    dxt_display_run();
}

/** Documented at declaration */
int
dxt_display_image_compressed(const char* title, unsigned char* image_compressed, int image_compressed_size, enum dxt_type type, int width, int height)
{
    printf("Display Compressed Image [size: %d bytes, resolution: %dx%d]\n", image_compressed_size, width, height);
    
    dxt_display_init(title, type, width, height);
    
    glGenTextures(1, &g_display.texture_compressed_id);
    glBindTexture(GL_TEXTURE_2D, g_display.texture_compressed_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if ( type == DXT5_YCOCG )
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, width, height, 0, image_compressed_size, image_compressed);
    else
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, width, height, 0, image_compressed_size, image_compressed);
    
    dxt_display_run();
}
 
/** Documented at declaration */
int
dxt_display_image_comparison(DXT_IMAGE_TYPE* image, unsigned char* image_compressed, int image_compressed_size, enum dxt_type type, int width, int height)
{
    printf("Display Image Comparison [size: %d bytes, resolution: %dx%d]\n", image_compressed_size, width, height);
    
    dxt_display_init("Display Comparison", type, width, height);
    
    glGenTextures(1, &g_display.texture_id);
    glBindTexture(GL_TEXTURE_2D, g_display.texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, DXT_IMAGE_GL_FORMAT, width, height, 0, GL_RGBA, DXT_IMAGE_GL_TYPE, image);
    
    glGenTextures(1, &g_display.texture_compressed_id);
    glBindTexture(GL_TEXTURE_2D, g_display.texture_compressed_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    if ( g_display.type == DXT5_YCOCG )
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGBA_S3TC_DXT5_EXT, width, height, 0, image_compressed_size, image_compressed);
    else
        glCompressedTexImage2DARB(GL_TEXTURE_2D, 0, GL_COMPRESSED_RGB_S3TC_DXT1_EXT, width, height, 0, image_compressed_size, image_compressed);
    
    g_display.texture_current_id = g_display.texture_id;
    
    dxt_display_run();
    
    return 0;
}
