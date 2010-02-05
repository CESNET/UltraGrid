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

#if defined(WIN32)
#define strdup(x) _strdup(x)
#else
#include <unistd.h>
#include <sys/mman.h>
#endif

#define MAXSTR 256

#include "dxt.h"
#include "glsl.h"

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <OpenGL/glext.h>
#include <Carbon/Carbon.h>
#define glGetProcAddress(n) aglGetProcAddress(n)

/*
 *
 * imported from http://developer.apple.com/qa/qa2001/qa1188.html
 *
 */

CFBundleRef gBundleRefOpenGL = NULL;

// -------------------------

OSStatus aglInitEntryPoints (void)
{
    OSStatus err = noErr;
    const Str255 frameworkName = "\nOpenGL.framework";
    FSRefParam fileRefParam;
    FSRef fileRef;
    CFURLRef bundleURLOpenGL;

    memset(&fileRefParam, 0, sizeof(fileRefParam));
    memset(&fileRef, 0, sizeof(fileRef));

    fileRefParam.ioNamePtr  = frameworkName;
    fileRefParam.newRef = &fileRef;

    // Frameworks directory/folder
    err = FindFolder (kSystemDomain, kFrameworksFolderType, false,
                      &fileRefParam.ioVRefNum, &fileRefParam.ioDirID);
    if (noErr != err) {
        //DebugStr ("\nCould not find frameworks folder");
        return err;
    }
    err = PBGetCatalogInfoSync (&fileRefParam); // make FSRef for folder
    if (noErr != err) {
        //DebugStr ("\nCould make FSref to frameworks folder");
        return err;
    }
    // create URL to folder
    bundleURLOpenGL = CFURLCreateFromFSRef (kCFAllocatorDefault,
                                            &fileRef);
    if (!bundleURLOpenGL) {
        //DebugStr ("\nCould create OpenGL Framework bundle URL");
        return paramErr;
    }
    // create ref to GL's bundle
    gBundleRefOpenGL = CFBundleCreate (kCFAllocatorDefault,
                                       bundleURLOpenGL);
    if (!gBundleRefOpenGL) {
        //DebugStr ("\nCould not create OpenGL Framework bundle");
        return paramErr;
    }
    CFRelease (bundleURLOpenGL); // release created bundle
    // if the code was successfully loaded, look for our function.
    if (!CFBundleLoadExecutable (gBundleRefOpenGL)) {
        //DebugStr ("\nCould not load MachO executable");
        return paramErr;
    }
    return err;
}

// -------------------------

void aglDellocEntryPoints (void)
{
    if (gBundleRefOpenGL != NULL) {
        // unload the bundle's code.
        CFBundleUnloadExecutable (gBundleRefOpenGL);
        CFRelease (gBundleRefOpenGL);
        gBundleRefOpenGL = NULL;
    }
}

// -------------------------

void * aglGetProcAddress (char * pszProc)
{
    return CFBundleGetFunctionPointerForName (gBundleRefOpenGL,
                CFStringCreateWithCStringNoCopy (NULL,
                     pszProc, CFStringGetSystemEncoding (), NULL));
}

#endif /* __APPLE__ */

#ifdef __linux__
#include <GL/gl.h>
#include <GL/glx.h>
#include <GL/glext.h>
#define glGetProcAddress(n) glXGetProcAddressARB((GLubyte *) n)
#endif /* __linux__ */

#ifdef _WIN32
#include <windows.h>
#include <GL/gl.h>
#include "glext.h"
#define glGetProcAddress(n) wglGetProcAddress(n)
#endif /* _WIN32 */


/*---------------------------------------------------------------------------*/

byte *in_data;
byte *dxt_data;
byte *ref_data;


/*---------------------------------------------------------------------------*/

#ifndef GLSL_YCOCG
#ifdef __APPLE__
static PFNGLGETCOMPRESSEDTEXIMAGEARBPROC  glGetCompressedTexImage;
static PFNGLCOMPRESSEDTEXIMAGE2DARBPROC   glCompressedTexImage2D;
#endif /* __APPLE__ */
#endif /* GLSL_YCOCG */

static void init_gl(void)
{
#ifdef __APPLE__
    glGetCompressedTexImage = (PFNGLGETCOMPRESSEDTEXIMAGEARBPROC)
               glGetProcAddress("glGetCompressedTexImageARB");
    glCompressedTexImage2D  = (PFNGLCOMPRESSEDTEXIMAGE2DARBPROC)
               glGetProcAddress("glCompressedTexImage2DARB");
#endif /* __APPLE__ */

    glEnable(GL_TEXTURE_2D);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
}

/*---------------------------------------------------------------------------*/

static void out_rgb(int w, int h, const char *name)
{
    double e;
    FILE *fout = NULL;

    if ((fout = fopen(name, "wb")))
    {
        GLsizei sz = 3 * w * h;
        GLubyte *p;

        if ((p = (GLubyte *) malloc(sz)))
        {
	    glReadPixels( 0,0, w, h, GL_RGB, GL_UNSIGNED_BYTE, p );

	    dxt_data = p;

            fwrite(p, 1, sz, fout);

            //free(p);

	    // Compute error
		e = ComputeError(ref_data, dxt_data, w, h);
	    fprintf(stdout, "RMS Error %.4f\n", e);
        }
        fclose(fout);
    }
    else perror("fopen");
}


static void in_dxt(int w, int h, int format, const char *name)
{
    FILE *fin = NULL;

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
            //free(p);
	    in_data = p;
	  }
        fclose(fin);
      }
    else perror("fopen");
}

/*---------------------------------------------------------------------------*/

static void in_rgb(int w, int h, const char *name)
{
    FILE *fin = NULL;

    if ((fin = fopen(name, "rb")))
      {
        GLubyte *p;
	GLsizei sz;
	
	sz = 4 * w * h;
	
        if ((p = (GLubyte *) malloc(sz)))
	  {
	    memset(p, 0, sz);

            fread(p, 1, sz, fin);
	    
	    ref_data = p;
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

    glClearColor(0,1,0,0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glBegin(GL_POLYGON);
    {
        glTexCoord2i(0, 1); glVertex2i(-1, -1);
        glTexCoord2i(1, 1); glVertex2i(+1, -1);
        glTexCoord2i(1, 0); glVertex2i(+1, +1);
        glTexCoord2i(0, 0); glVertex2i(-1, +1);
    }
    glEnd();

    SDL_GL_SwapBuffers();
}

/*---------------------------------------------------------------------------*/
#ifdef __APPLE__
#undef main
#endif /* __APPLE__ */

int main(int argc, char *argv[])
{
    int w = 1024;
    int h = 512;
    int p = 1;
    int format = 1; // DXT1 or DXT5 or DXT5YCoCg (6)
    int out = 0;
    int argi;
    char *rgbfile;

    /* Process arguments. */

    for (argi = 1; argi < argc; ++argi)
        if      (strcmp(argv[argi], "-p") == 0) p = 0;
        else if (strcmp(argv[argi], "-5") == 0) format = 5;
        else if (strcmp(argv[argi], "-6") == 0) format = 6;
        else if (strcmp(argv[argi], "-1") == 0) format = 1;
        else if (strcmp(argv[argi], "-o") == 0) out = 1;
        else if (strcmp(argv[argi], "-r") == 0) rgbfile = strdup(argv[++argi]);
        else if (strcmp(argv[argi], "-w") == 0) w = atoi(argv[++argi]);
        else if (strcmp(argv[argi], "-h") == 0) h = atoi(argv[++argi]);
        else     break;

    /* Use SDL to get an OpenGL context for use as compressor/decompressor. */

    if (SDL_Init(SDL_INIT_VIDEO) == 0)
    {
        SDL_GL_SetAttribute(SDL_GL_RED_SIZE,     8);
        SDL_GL_SetAttribute(SDL_GL_GREEN_SIZE,   8);
        SDL_GL_SetAttribute(SDL_GL_BLUE_SIZE,    8);
        SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE,  16);
        SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);

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
	    
	    in_dxt(w, h, format, argv[argi]);
	    
#if defined(GLSL_YCOCG)
	    if (format == 6)
	      {
		glUseProgramObjectARB(PHandle);
		glActiveTexture(GL_TEXTURE0);
		int h=glGetUniformLocationARB(PHandle,"yuvtex");
		glUniform1iARB(h,0);  /* Bind yuvtex to texture unit 0 */
	      }
#endif
	    
	    display(w, h);
	    
#if defined(GLSL_YCOCG)
	    if (format == 6)
	      glUseProgramObjectARB(0);
#endif

	    if (out)
	      {
		in_rgb(w, h, rgbfile);      // ref_data
		out_rgb(w, h, "out.rgb");   // dxt_data
	      }

            /* Pause until the user closes the window. */

            if (p)
            {
                while (SDL_WaitEvent(&e))
                    if (e.type == SDL_QUIT || e.type == SDL_KEYDOWN)
                        break;
                    else {
		      //display(w, h);
		    }
            }
        }
        else fprintf(stderr, "SDL_SetVideoMode: %s\n", SDL_GetError());

        SDL_Quit();
    }
    else fprintf(stderr, "SDL_Init: %s\n", SDL_GetError());

    return 0;
}

/*---------------------------------------------------------------------------*/
