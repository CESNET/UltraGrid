/******************************************************************************
 * a DXT viewing utility
 *
 * Author : Robert Kooima
 *
 * Copyright (C) 2007 Electronic Visualization Laboratory,
 * University of Illinois at Chicago
 *
 * This library is free software; you can redistribute it and/or modify it
 * under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either Version 2.1 of the License, or
 * (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
 * or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public
 * License for more details.
 *
 * You should have received a copy of the GNU Lesser Public License along
 * with this library; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 *****************************************************************************/


#include <SDL/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>

#if !defined(WIN32)
#include <unistd.h>
#include <sys/mman.h>
#endif

#define MAXSTR 256

#include "dxt.h"
#include "glsl.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#endif

#ifdef __linux__
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>
#define glGetProcAddress(n) glXGetProcAddressARB((GLubyte *) n)
#endif

#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#include "glext.h"
#define glGetProcAddress(n) wglGetProcAddress(n)
#endif


/*---------------------------------------------------------------------------*/

#ifndef GLSL_YCOCG
#ifndef __APPLE__
static PFNGLGETCOMPRESSEDTEXIMAGEARBPROC  glGetCompressedTexImage;
static PFNGLCOMPRESSEDTEXIMAGE2DARBPROC   glCompressedTexImage2D;
#endif
#endif

static void init_gl(void)
{
#ifndef __APPLE__
    glGetCompressedTexImage = (PFNGLGETCOMPRESSEDTEXIMAGEARBPROC)
               glGetProcAddress("glGetCompressedTexImageARB");
    glCompressedTexImage2D  = (PFNGLCOMPRESSEDTEXIMAGE2DARBPROC)
               glGetProcAddress("glCompressedTexImage2DARB");
#endif

    glEnable(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

/*---------------------------------------------------------------------------*/


static void in_dxt(int w, int h, int format, const char *name)
{
    FILE *fin = NULL;
    static int first = 1;

    if ((fin = fopen(name, "rb")))
      {
        GLubyte *p;
	GLsizei sz;
	
	if (format == 5 || format == 6)
	  sz = 4 * w * h / 4;
	else
	  sz = 8 * (w / 4) * (h / 4);
	
        if ((p = (GLubyte *) malloc(sz)))
	  {
	    int ww, hh;
	    fread( &ww, sizeof( int ), 1, fin );
	    fread( &hh, sizeof( int ), 1, fin );
	    
            fread(p, 1, sz, fin);
	    
		    if (format == 5 || format == 6)
		      glCompressedTexImage2D(GL_TEXTURE_2D, 0,
					     GL_COMPRESSED_RGBA_S3TC_DXT5_EXT,
					     w, h, 0, sz, p);
		    else
		      glCompressedTexImage2D(GL_TEXTURE_2D, 0,
					     GL_COMPRESSED_RGB_S3TC_DXT1_EXT,
					     w, h, 0, sz, p);
            free(p);
	  }
        fclose(fin);
      }
    else perror("fopen");
}

/*---------------------------------------------------------------------------*/

static void display(int w, int h)
{
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(0, w, 0, h, 0, 1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glClearColor(0,1,0,1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_POLYGON);
    {
        glTexCoord2i(0, 1); glVertex2i(-1, -1);
        glTexCoord2i(1, 1); glVertex2i(+1, -1);
        glTexCoord2i(1, 0); glVertex2i(+1, +1);
        glTexCoord2i(0, 0); glVertex2i(-1, +1);
    }
    glEnd();

    glFinish();

    SDL_GL_SwapBuffers();
}

/*---------------------------------------------------------------------------*/

int main(int argc, char *argv[])
{
    int w = 1024;
    int h = 512;
    int format = 1; // DXT1 or DXT5 or DXT5YCoCg (6)
    int out = 0;
    int argi;

    /* Process arguments. */

    for (argi = 1; argi < argc; ++argi)
        if (strcmp(argv[argi], "-5") == 0) format = 5;
        else if (strcmp(argv[argi], "-6") == 0) format = 6;
        else if (strcmp(argv[argi], "-1") == 0) format = 1;
        else if (strcmp(argv[argi], "-w") == 0) w = atoi(argv[++argi]);
        else if (strcmp(argv[argi], "-h") == 0) h = atoi(argv[++argi]);
        else     break;

    /* Use SDL to get an OpenGL context for use as compressor/decompressor. */

    if (SDL_Init(SDL_INIT_VIDEO) == 0)
    {
        //SDL_GL_SetAttribute(SDL_GL_RED_SIZE,     8);
        //SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,   8);
        //SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,    8);
        //SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,  16);
        //SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

        if (SDL_SetVideoMode(w, h, 0, SDL_OPENGL))
        {
            SDL_Event e;

            init_gl();

            /* Convert and display named files. */

#if defined(GLSL_YCOCG)
	    // Initialize the "OpenGL Extension Wrangler" library
	    glewInit();
	    
	    if (!glewIsSupported("GL_VERSION_2_0 "
				 "GL_ARB_vertex_program "
				 "GL_ARB_fragment_program "
				 "GL_ARB_texture_compression "
				 "GL_EXT_texture_compression_s3tc "
				 )
		) {
	      fprintf(stderr, "GLSL_YCOCG> Unable to load required OpenGL extension\n");
	      exit(-1);
	    }

	    // Load the shaders	    
	    GLchar *FragmentShaderSource;
	    GLchar *VertexShaderSource;
	    GLSLreadShaderSource("ycocg", &VertexShaderSource, &FragmentShaderSource);
	    PHandle = GLSLinstallShaders(VertexShaderSource, FragmentShaderSource);
	    
	    /* Finally, use the program. */
	    glUseProgramObjectARB(PHandle);
	    
	    glUseProgramObjectARB(0);
#endif
	    
#if defined(GLSL_YCOCG)
	    if (format == 6)
	    {
			glUseProgramObjectARB(PHandle);
			glActiveTexture(GL_TEXTURE0);
			int h=glGetUniformLocationARB(PHandle,"yuvtex");
			glUniform1iARB(h,0);  /* Bind yuvtex to texture unit 0 */
	    }
#endif
	    

		for (; argi < argc; ++argi)
		{
			in_dxt(w, h, format, argv[argi]);
			display(w, h);

			while (SDL_PollEvent(&e))
			    if (e.type == SDL_QUIT || e.type == SDL_KEYDOWN)
				break;
		}

#if defined(GLSL_YCOCG)
	    if (format == 6)
	      glUseProgramObjectARB(0);
#endif

		while (SDL_WaitEvent(&e))
		    if (e.type == SDL_QUIT || e.type == SDL_KEYDOWN)
			break;

        }
        else fprintf(stderr, "SDL_SetVideoMode: %s\n", SDL_GetError());

        SDL_Quit();
    }
    else fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());

    return 0;
}

/*---------------------------------------------------------------------------*/
