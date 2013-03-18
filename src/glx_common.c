/*
 * FILE:    glx_common.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif
#include "debug.h"

#include <GL/glew.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/glx.h>
#include "x11_common.h"
#include "glx_common.h"

#ifdef HAVE_GPUPERFAPI
#include "GPUPerfAPI.h"
#endif

#define GLX_MAGIC 0x665f43ffu

#define VERSION_GET_MAJOR(version) (version >> 8u)
#define VERSION_GET_MINOR(version) (version & 0xFF)
#define VERSION_IS_UNSPECIFIED(version) (!version)

/*
 * GLX context creation overtaken from:
 * http://www.opengl.org/wiki/Tutorial:_OpenGL_3.0_Context_Creation_%28GLX%29
 */
 
struct state_glx
{
        GLXContext ctx;
        Window win;
        Colormap cmap;
        uint32_t magic;
};

static int isExtensionSupported(const char *extList, const char *extension);
static int ctxErrorHandler( Display *dpy, XErrorEvent *ev );

#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
 
void glx_validate(void *arg)
{
        debug_msg("Validating GLX context %p.\n", arg);
        struct state_glx *context = (struct state_glx *) arg;

        assert(context->magic == GLX_MAGIC);
        debug_msg("GLX context %p validated.\n", context);
}
 
void glx_free(void *arg)
{
        struct state_glx *context = (struct state_glx *) arg;
        Display *display = NULL;

        assert(context->magic == GLX_MAGIC);
        
        x11_lock();

#ifdef HAVE_GPUPERFAPI
        GPA_CloseContext();
#endif

        display = x11_get_display();
        if(!display) {
                fprintf(stderr, "Unable to get shared display for GLX!\n");
                x11_unlock();
        }

        
        glXMakeCurrent( display, context->win, context->ctx );
        
        glXDestroyContext( display, context->ctx );
        fprintf(stderr, "GLX context destroyed\n");
#ifdef HAVE_GPUPERFAPI
        GPA_Destroy();
#endif
        
        XDestroyWindow( display, context->win );
        XFreeColormap( display, context->cmap );
        
        x11_release_display();
        free(context);
        
        x11_unlock();
}

// Helper to check for extension string presence.  Adapted from:
//   http://www.opengl.org/resources/features/OGLextensions/
static int isExtensionSupported(const char *extList, const char *extension)
{
  const char *start;
  const char *where, *terminator;
 
  /* Extension names should not have spaces. */
  where = strchr(extension, ' ');
  if ( where || *extension == '\0' )
    return FALSE;
 
  /* It takes a bit of care to be fool-proof about parsing the
     OpenGL extensions string. Don't be fooled by sub-strings,
     etc. */
  for ( start = extList; ; ) {
    where = strstr( start, extension );
 
    if ( !where )
      break;
 
    terminator = where + strlen( extension );
 
    if ( where == start || *(where - 1) == ' ' )
      if ( *terminator == ' ' || *terminator == '\0' )
        return TRUE;
 
    start = terminator;
  }
 
  return FALSE;
}
 
static int ctxErrorOccurred = FALSE;
static int ctxErrorHandler( Display *dpy, XErrorEvent *ev )
{
        UNUSED(dpy);
        UNUSED(ev);
    ctxErrorOccurred = TRUE;
    return 0;
}

void *glx_init(glx_opengl_version_t version)
{
        Display *display;
        struct state_glx *context;

        x11_enter_thread();
        
        context = (struct state_glx *) malloc(sizeof(struct state_glx));
        context->magic = GLX_MAGIC;
        x11_lock();

        display = x11_acquire_display();

        if(!display) {
                free(context);
                x11_unlock();
                return NULL;
        }

#ifdef HAVE_GPUPERFAPI
        GPA_Status gpa_status;

        gpa_status = GPA_Initialize();
        assert(gpa_status == GPA_STATUS_OK);
#endif
 
  
 
  // Get a matching FB config
  static int visual_attribs[] =
    {
      GLX_X_RENDERABLE    , True,
      GLX_DRAWABLE_TYPE   , GLX_WINDOW_BIT,
      GLX_RENDER_TYPE     , GLX_RGBA_BIT,
      GLX_X_VISUAL_TYPE   , GLX_TRUE_COLOR,
      GLX_RED_SIZE        , 8,
      GLX_GREEN_SIZE      , 8,
      GLX_BLUE_SIZE       , 8,
      GLX_ALPHA_SIZE      , 8,
      GLX_DEPTH_SIZE      , 24,
      GLX_STENCIL_SIZE    , 8,
      GLX_DOUBLEBUFFER    , True,
      //GLX_SAMPLE_BUFFERS  , 1,
      //GLX_SAMPLES         , 4,
      None
    };
 
  int glx_major, glx_minor;

  int fbcount;
  int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
  GLXFBConfig *fbc;
  GLXFBConfig bestFbc;
  XVisualInfo *vi;
  const char *glxExts;
  glXCreateContextAttribsARBProc glXCreateContextAttribsARB;
  int (*oldHandler)(Display*, XErrorEvent*);
 
  // FBConfigs were added in GLX version 1.3.
  if ( !glXQueryVersion( display, &glx_major, &glx_minor ) || 
       ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) )
  {
    printf( "Invalid GLX version" );
    goto error;
  }
 
  printf( "Getting matching framebuffer configs\n" );
  fbc = glXChooseFBConfig( display, DefaultScreen( display ), 
                                        visual_attribs, &fbcount );
  if ( !fbc )
  {
    printf( "Failed to retrieve a framebuffer config\n" );
    goto error;
  }
  printf( "Found %d matching FB configs.\n", fbcount );
 
  // Pick the FB config/visual with the most samples per pixel
  printf( "Getting XVisualInfos\n" );
 
  for ( int i = 0; i < fbcount; i++ )
  {
    XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
    if ( vi )
    {
      int samp_buf, samples;
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES       , &samples  );
 
      printf( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d,"
              " SAMPLES = %d\n", 
              i, (unsigned int) vi -> visualid, samp_buf, samples );
 
      if ( best_fbc < 0 || (samp_buf && samples > best_num_samp ))
        best_fbc = i, best_num_samp = samples;
      if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
        worst_fbc = i, worst_num_samp = samples;
    }
    XFree( vi );
  }
 
  bestFbc = fbc[ best_fbc ];
 
  // Be sure to free the FBConfig list allocated by glXChooseFBConfig()
  XFree( fbc );
 
  // Get a visual
  vi = glXGetVisualFromFBConfig( display, bestFbc );
  printf( "Chosen visual ID = 0x%x\n", (unsigned int) vi->visualid );
 
  printf( "Creating colormap\n" );
  XSetWindowAttributes swa;
  swa.colormap = context->cmap = XCreateColormap( display,
                                         RootWindow( display, vi->screen ), 
                                         vi->visual, AllocNone );
  swa.background_pixmap = None ;
  swa.border_pixel      = 0;
  swa.event_mask        = StructureNotifyMask;
 
  printf( "Creating window\n" );
  context->win = XCreateWindow( display, RootWindow( display, vi->screen ), 
                              0, 0, 1920, 1080, 0, vi->depth, InputOutput, 
                              vi->visual, 
                              CWBorderPixel|CWColormap|CWEventMask, &swa );
  XMoveWindow(display, context->win, 0, 0);

  if ( !context->win )
  {
    printf( "Failed to create window.\n" );
    goto error;
  }
 
  // Done with the visual info data
  XFree( vi );
 
 /* We don't need this for UG */
  /*XStoreName( display, win, "GL 3.0 Window" );
 
  printf( "Mapping window\n" );
  XMapWindow( display, win );*/
 
  // Get the default screen's GLX extension list
  glxExts = glXQueryExtensionsString( display,
                                                  DefaultScreen( display ) );
 
  // NOTE: It is not necessary to create or make current to a context before
  // calling glXGetProcAddressARB
  glXCreateContextAttribsARB = 0;
  glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
           glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );
 
  context->ctx = 0;
 
  // Install an X error handler so the application won't exit if GL 3.0
  // context allocation fails.
  //
  // Note this error handler is global.  All display connections in all threads
  // of a process use the same error handler, so be sure to guard against other
  // threads issuing X commands while this code is running.
  ctxErrorOccurred = FALSE;
  oldHandler = XSetErrorHandler(&ctxErrorHandler);
 
  // Check for the GLX_ARB_create_context extension string and the function.
  // If either is not present, use GLX 1.3 context creation method.
  if ( !isExtensionSupported( glxExts, "GLX_ARB_create_context" ) ||
       !glXCreateContextAttribsARB )
  {
    printf( "glXCreateContextAttribsARB() not found"
            " ... using old-style GLX context\n" );
    context->ctx = glXCreateNewContext( display, bestFbc, GLX_RGBA_TYPE, 0, True );
    if(!VERSION_IS_UNSPECIFIED(version)) {
            return NULL;
    }
  }
 
  // If it does, try to get a GL 3.0 context!
  else
  {
    int context_attribs[] =
      {
        GLX_CONTEXT_MAJOR_VERSION_ARB, 3,
        GLX_CONTEXT_MINOR_VERSION_ARB, 0,
        //GLX_CONTEXT_FLAGS_ARB        , GLX_CONTEXT_FORWARD_COMPATIBLE_BIT_ARB,
        None
      };

    if(!VERSION_IS_UNSPECIFIED(version)) {
            context_attribs[1] = VERSION_GET_MAJOR(version);
            context_attribs[3] = VERSION_GET_MINOR(version);
    }
 
    printf( "Creating context\n" );
    context->ctx = glXCreateContextAttribsARB( display, bestFbc, 0,
                                      True, context_attribs );
 
    // Sync to ensure any errors generated are processed.
    XSync( display, False );
    if ( !ctxErrorOccurred && context->ctx )
      printf( "Created GL %d.%d context\n", context_attribs[1], context_attribs[3]);
    else
    {
      if(VERSION_IS_UNSPECIFIED(version)) {
        // Couldn't create GL 3.0 context.  Fall back to old-style 2.x context.
        // When a context version below 3.0 is requested, implementations will
        // return the newest context version compatible with OpenGL versions less
        // than version 3.0.
        // GLX_CONTEXT_MAJOR_VERSION_ARB = 1
        context_attribs[1] = 1;
        // GLX_CONTEXT_MINOR_VERSION_ARB = 0
        context_attribs[3] = 0;
   
        ctxErrorOccurred = FALSE;
   
        printf( "Failed to create GL 3.0 context"
                " ... using old-style GLX context\n" );
        context->ctx = glXCreateContextAttribsARB( display, bestFbc, 0, 
                                          True, context_attribs );
      } else {
        // we explicitly requested context which we cannot obtain - return error then
        goto error;
      }
    }
  }
 
  // Sync to ensure any errors generated are processed.
  XSync( display, False );
 
  // Restore the original error handler
  XSetErrorHandler( oldHandler );
 
  if ( ctxErrorOccurred || !context->ctx )
  {
    printf( "Failed to create an OpenGL context\n" );
    goto error;
  }
 
  // Verifying that context is a direct context
  if ( ! glXIsDirect ( display, context->ctx ) )
  {
    printf( "Indirect GLX rendering context obtained\n" );
  }
  else
  {
    printf( "Direct GLX rendering context obtained\n" );
  }
 
  printf( "Making context current\n" );
  glXMakeCurrent( display, context->win, context->ctx );

        glewInit();
        
        x11_unlock();
  
        glx_validate(context);

#ifdef HAVE_GPUPERFAPI
        gpa_status = GPA_OpenContext( context->ctx );
        assert(gpa_status == GPA_STATUS_OK);
#endif

        return (void *) context;

error:
        free(context);
        x11_unlock();
        return NULL;
}

void glx_make_current(void *arg)
{
        if(arg) {
                struct state_glx *context = (struct state_glx *) arg;
                Bool res;

                res = glXMakeCurrent( x11_get_display(), context->win, context->ctx );
                if(res != True) {
                        fprintf(stderr, "Acquiring GLX context failed!\n");
                }
        } else {
                Bool res;

                res = glXMakeCurrent( x11_get_display(), None, NULL);
                if(res != True) {
                        fprintf(stderr, "Releasing GLX context failed!\n");
                }
        }
}

