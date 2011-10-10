/*
 * FILE:    dxt_glsl_compress.c
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

#include "config.h"
#include "dxt_glsl_compress.h"
#include "dxt_compress/dxt_encoder.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/gl.h>
#include <GL/glx.h>

/*
 * GLX context creation overtaken from:
 * http://www.opengl.org/wiki/Tutorial:_OpenGL_3.0_Context_Creation_%28GLX%29
 */

struct video_compress {
        struct dxt_encoder *encoder;

        struct video_frame out;
        char *decoded, *repacked;
        unsigned int configured:1;
};

static void configure_with(struct video_compress *s, struct video_frame *frame);
static int isExtensionSupported(const char *extList, const char *extension);
static int ctxErrorHandler( Display *dpy, XErrorEvent *ev );
void glx_init(void);

#define GLX_CONTEXT_MAJOR_VERSION_ARB       0x2091
#define GLX_CONTEXT_MINOR_VERSION_ARB       0x2092
typedef GLXContext (*glXCreateContextAttribsARBProc)(Display*, GLXFBConfig, GLXContext, Bool, const int*);
 
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
    ctxErrorOccurred = TRUE;
    return 0;
}

void glx_init()
{
        Display *display = XOpenDisplay(0);
 
  if ( !display )
  {
    printf( "Failed to open X display\n" );
    exit(1);
  }
 
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
 
  // FBConfigs were added in GLX version 1.3.
  if ( !glXQueryVersion( display, &glx_major, &glx_minor ) || 
       ( ( glx_major == 1 ) && ( glx_minor < 3 ) ) || ( glx_major < 1 ) )
  {
    printf( "Invalid GLX version" );
    exit(1);
  }
 
  printf( "Getting matching framebuffer configs\n" );
  int fbcount;
  GLXFBConfig *fbc = glXChooseFBConfig( display, DefaultScreen( display ), 
                                        visual_attribs, &fbcount );
  if ( !fbc )
  {
    printf( "Failed to retrieve a framebuffer config\n" );
    exit(1);
  }
  printf( "Found %d matching FB configs.\n", fbcount );
 
  // Pick the FB config/visual with the most samples per pixel
  printf( "Getting XVisualInfos\n" );
  int best_fbc = -1, worst_fbc = -1, best_num_samp = -1, worst_num_samp = 999;
 
  int i;
  for ( i = 0; i < fbcount; i++ )
  {
    XVisualInfo *vi = glXGetVisualFromFBConfig( display, fbc[i] );
    if ( vi )
    {
      int samp_buf, samples;
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLE_BUFFERS, &samp_buf );
      glXGetFBConfigAttrib( display, fbc[i], GLX_SAMPLES       , &samples  );
 
      printf( "  Matching fbconfig %d, visual ID 0x%2x: SAMPLE_BUFFERS = %d,"
              " SAMPLES = %d\n", 
              i, vi -> visualid, samp_buf, samples );
 
      if ( best_fbc < 0 || samp_buf && samples > best_num_samp )
        best_fbc = i, best_num_samp = samples;
      if ( worst_fbc < 0 || !samp_buf || samples < worst_num_samp )
        worst_fbc = i, worst_num_samp = samples;
    }
    XFree( vi );
  }
 
  GLXFBConfig bestFbc = fbc[ best_fbc ];
 
  // Be sure to free the FBConfig list allocated by glXChooseFBConfig()
  XFree( fbc );
 
  // Get a visual
  XVisualInfo *vi = glXGetVisualFromFBConfig( display, bestFbc );
  printf( "Chosen visual ID = 0x%x\n", vi->visualid );
 
  printf( "Creating colormap\n" );
  XSetWindowAttributes swa;
  Colormap cmap;
  swa.colormap = cmap = XCreateColormap( display,
                                         RootWindow( display, vi->screen ), 
                                         vi->visual, AllocNone );
  swa.background_pixmap = None ;
  swa.border_pixel      = 0;
  swa.event_mask        = StructureNotifyMask;
 
  printf( "Creating window\n" );
  Window win = XCreateWindow( display, RootWindow( display, vi->screen ), 
                              0, 0, 100, 100, 0, vi->depth, InputOutput, 
                              vi->visual, 
                              CWBorderPixel|CWColormap|CWEventMask, &swa );
  if ( !win )
  {
    printf( "Failed to create window.\n" );
    exit(1);
  }
 
  // Done with the visual info data
  XFree( vi );
 
 /* We don't need this for UG */
  /*XStoreName( display, win, "GL 3.0 Window" );
 
  printf( "Mapping window\n" );
  XMapWindow( display, win );*/
 
  // Get the default screen's GLX extension list
  const char *glxExts = glXQueryExtensionsString( display,
                                                  DefaultScreen( display ) );
 
  // NOTE: It is not necessary to create or make current to a context before
  // calling glXGetProcAddressARB
  glXCreateContextAttribsARBProc glXCreateContextAttribsARB = 0;
  glXCreateContextAttribsARB = (glXCreateContextAttribsARBProc)
           glXGetProcAddressARB( (const GLubyte *) "glXCreateContextAttribsARB" );
 
  GLXContext ctx = 0;
 
  // Install an X error handler so the application won't exit if GL 3.0
  // context allocation fails.
  //
  // Note this error handler is global.  All display connections in all threads
  // of a process use the same error handler, so be sure to guard against other
  // threads issuing X commands while this code is running.
  ctxErrorOccurred = FALSE;
  int (*oldHandler)(Display*, XErrorEvent*) =
      XSetErrorHandler(&ctxErrorHandler);
 
  // Check for the GLX_ARB_create_context extension string and the function.
  // If either is not present, use GLX 1.3 context creation method.
  if ( !isExtensionSupported( glxExts, "GLX_ARB_create_context" ) ||
       !glXCreateContextAttribsARB )
  {
    printf( "glXCreateContextAttribsARB() not found"
            " ... using old-style GLX context\n" );
    ctx = glXCreateNewContext( display, bestFbc, GLX_RGBA_TYPE, 0, True );
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
 
    printf( "Creating context\n" );
    ctx = glXCreateContextAttribsARB( display, bestFbc, 0,
                                      True, context_attribs );
 
    // Sync to ensure any errors generated are processed.
    XSync( display, False );
    if ( !ctxErrorOccurred && ctx )
      printf( "Created GL 3.0 context\n" );
    else
    {
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
      ctx = glXCreateContextAttribsARB( display, bestFbc, 0, 
                                        True, context_attribs );
    }
  }
 
  // Sync to ensure any errors generated are processed.
  XSync( display, False );
 
  // Restore the original error handler
  XSetErrorHandler( oldHandler );
 
  if ( ctxErrorOccurred || !ctx )
  {
    printf( "Failed to create an OpenGL context\n" );
    exit(1);
  }
 
  // Verifying that context is a direct context
  if ( ! glXIsDirect ( display, ctx ) )
  {
    printf( "Indirect GLX rendering context obtained\n" );
  }
  else
  {
    printf( "Direct GLX rendering context obtained\n" );
  }
 
  printf( "Making context current\n" );
  glXMakeCurrent( display, win, ctx );

  glewInit();
}

static void configure_with(struct video_compress *s, struct video_frame *frame)
{
        codec_t tmp = s->out.color_spec;
        enum dxt_format format;
        
        glx_init();
        
        memcpy(&s->out, frame, sizeof(struct video_frame));

        switch (s->out.color_spec) {
                case RGBA:
                        s->out.decoder = (decoder_t) memcpy;
                        format = DXT_FORMAT_RGB;
                        break;
                case R10k:
                        s->out.decoder = (decoder_t) vc_copyliner10k;
                        s->out.rshift = 0;
                        s->out.gshift = 8;
                        s->out.bshift = 16;
                        format = DXT_FORMAT_RGB;
                        break;
                case UYVY:
                case Vuy2:
                case DVS8:
                        s->out.decoder = (decoder_t) memcpy;
                        format = DXT_FORMAT_YUV;
                        break;
                case v210:
                        s->out.decoder = (decoder_t) vc_copylinev210;
                        format = DXT_FORMAT_YUV;
                        break;
                case DVS10:
                        s->out.decoder = (decoder_t) vc_copylineDVS10;
                        format = DXT_FORMAT_YUV;
                        break;
                /*case DXT1:
                        fprintf(stderr, "Input frame is already comperssed!");
                        exit(128);*/
        }
        s->out.src_bpp = get_bpp(s->out.color_spec);
        s->out.src_linesize = s->out.width * s->out.src_bpp;
        s->out.dst_linesize = s->out.width * 
                (format == DXT_FORMAT_RGB ? 4 /*RGBA*/: 2/*YUV 422*/);

        /* We will deinterlace the output frame */
        s->out.aux &= ~AUX_INTERLACED;
        
        s->out.color_spec = tmp;
        
        
        if(s->out.color_spec == DXT1) {
                s->encoder = dxt_encoder_create(DXT_TYPE_DXT1, frame->width, frame->height, format);
                s->out.aux |= AUX_RGB;
                s->out.data_len = frame->width * frame->height / 2;
        } else if(s->out.color_spec == DXT5){
                s->encoder = dxt_encoder_create(DXT_TYPE_DXT5_YCOCG, frame->width, frame->height, format);
                s->out.aux |= AUX_YUV; /* YCoCg */
                s->out.data_len = frame->width * frame->height;
        }
        
        s->out.data = (char *) malloc(s->out.data_len);
        s->decoded = malloc(4 * frame->width * frame->height);
        s->repacked = malloc(4 * frame->width * frame->height);
        
        s->configured = TRUE;
}

struct video_compress * dxt_glsl_init(char * opts)
{
        struct video_compress *s;
        
        s = (struct video_compress *) malloc(sizeof(struct video_compress));
        s->out.data = NULL;
        s->decoded = NULL;
        s->repacked = NULL;
        
        if(opts) {
                if(strcasecmp(opts, "DXT5_YCoCg") == 0) {
                        s->out.color_spec = DXT5;
                } else if(strcasecmp(opts, "DXT1") == 0) {
                        s->out.color_spec = DXT1;
                } else {
                        fprintf(stderr, "Unknown compression : %s\n", opts);
                        return NULL;
                }
        } else {
                s->out.color_spec = DXT1;
        }
                
        s->configured = FALSE;

        return s;
}

struct video_frame * dxt_glsl_compress(void *arg, struct video_frame * tx)
{
        struct video_compress *s = (struct video_compress *) arg;
        int x;
        unsigned char *line1, *line2;
        
        if(!s->configured)
                configure_with(s, tx);


        line1 = (unsigned char *)tx->data;
        line2 = s->decoded;
        
        for (x = 0; x < tx->height; ++x) {
                s->out.decoder(line2, line1, s->out.src_linesize,
                                s->out.rshift, s->out.gshift, s->out.bshift);
                line1 += s->out.src_linesize;
                line2 += s->out.dst_linesize;
        }
        
        if(tx->color_spec == RGBA || 
                        tx->color_spec == R10k) {
                dxt_encoder_compress(s->encoder, s->decoded, s->out.data);
        } else {
                const count = s->out.width * s->out.height;
                /* Repack the data to YUV 4:4:4 Format */
                for (x = 0; x < count; x += 2) {
                        s->repacked[4 * x] = s->decoded[2 * x + 1]; //Y1
                        s->repacked[4 * x + 1] = s->decoded[2 * x]; //U1
                        s->repacked[4 * x + 2] = s->decoded[2 * x + 2];     //V1
                        //s->repacked[4 * x + 3] = 255;  //Alpha
        
                        s->repacked[4 * x + 4] = s->decoded[2 * x + 3];     //Y2
                        s->repacked[4 * x + 5] = s->decoded[2 * x]; //U1
                        s->repacked[4 * x + 6] = s->decoded[2 * x + 2];     //V1
                        //s->repacked[4 * x + 7] = 255;  //Alpha
                }
                dxt_encoder_compress(s->encoder, s->repacked, s->out.data);
        }
        
        return &s->out;
}

void dxt_glsl_exit(void *arg)
{
        struct video_compress *s = (struct video_compress *) arg;
        free(s->out.data);
}
