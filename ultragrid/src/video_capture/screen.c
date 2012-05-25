/*
 * FILE:    screen.c
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


#include "host.h"
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "debug.h"
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#include "video_capture/screen.h"
#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "video_display.h"
#include "video.h"

#include <pthread.h>

#ifdef HAVE_MACOSX
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <Carbon/Carbon.h>
#else
#include <GL/glew.h>
#include <X11/Xlib.h>
#include <GL/glx.h>
#include "x11_common.h"
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


/* prototypes of functions defined in this module */
static void show_help(void);

static void show_help()
{
        printf("Screen capture\n");
        printf("Usage\n");
        printf("\t-t screen[:fps=<fps>]\n");
        printf("\t\t<fps> - preferred grabbing fps (otherwise unlimited)\n");
}

/* defined in main.c */
extern int uv_argc;
extern char **uv_argv;

static struct vidcap_screen_state *state;

struct vidcap_screen_state {
        struct video_frame       *frame; 
        struct tile       *tile; 
        int frames;
        struct       timeval t, t0;
#ifdef HAVE_MACOSX
        CGDirectDisplayID display;
#else
        Display *dpy;
        Window root;
        GLXContext glc;
#endif

        GLuint tex;
        GLuint tex_out;
        GLuint fbo;

        struct timeval prev_time;

        double fps;
};

pthread_once_t initialized = PTHREAD_ONCE_INIT;

static void initialize() {
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;


#ifndef HAVE_MACOSX
        XWindowAttributes        xattr;
#endif

        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);

#ifndef HAVE_MACOSX
        x11_lock();

        s->dpy = x11_acquire_display();

        x11_unlock();


        s->root = DefaultRootWindow(s->dpy);
        GLint att[] = {GLX_RGBA, None};
        XVisualInfo *vis = glXChooseVisual(s->dpy, 0, att);
        s->glc = glXCreateContext(s->dpy, vis, NULL, True);
        glXMakeCurrent(s->dpy, s->root, s->glc);

        XGetWindowAttributes(s->dpy, DefaultRootWindow(s->dpy), &xattr);
        s->tile->width = xattr.width;
        s->tile->height = xattr.height;

        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
                /* Problem: glewInit failed, something is seriously wrong. */
                fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
                goto error;
        }


        glEnable(GL_TEXTURE_2D);

        glGenTextures(1, &state->tex);
        glBindTexture(GL_TEXTURE_2D, state->tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, state->tile->width, state->tile->height,
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glGenTextures(1, &state->tex_out);
        glBindTexture(GL_TEXTURE_2D, state->tex_out);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, state->tile->width, state->tile->height,
                        0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

        glBindTexture(GL_TEXTURE_2D, 0);
        glGenFramebuffers(1, &state->fbo);

        glViewport(0, 0, state->tile->width, state->tile->height);
        glDisable(GL_DEPTH_TEST);

#else
        s->display = CGMainDisplayID();
        CGImageRef image = CGDisplayCreateImage(s->display);

        s->tile->width = CGImageGetWidth(image);
        s->tile->height = CGImageGetHeight(image);
        CFRelease(image);
#endif

        s->frame->color_spec = RGBA;
        if(s->fps > 0.0) {
                s->frame->fps = s->fps;
        } else {
                s->frame->fps = 30;
        }
        s->frame->interlacing = PROGRESSIVE;
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;
        s->tile->data = (char *) malloc(s->tile->data_len);

        return;
error:
        fprintf(stderr, "[Screen cap.] Initialization failed!\n");
        exit_uv(128);
}



struct vidcap_type * vidcap_screen_probe(void)
{
        struct vidcap_type*		vt;

        vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id          = VIDCAP_SCREEN_ID;
                vt->name        = "screen";
                vt->description = "Grabbing screen";
        }
        return vt;
}

void * vidcap_screen_init(char *init_fmt, unsigned int flags)
{
        struct vidcap_screen_state *s;

        printf("vidcap_screen_init\n");

        UNUSED(flags);


        state = s = (struct vidcap_screen_state *) malloc(sizeof(struct vidcap_screen_state));
        if(s == NULL) {
                printf("Unable to allocate screen capture state\n");
                return NULL;
        }

        s->fps = 0.0;

        s->frame = NULL;
        s->tile = NULL;

        s->prev_time.tv_sec = 
                s->prev_time.tv_usec = 0;


        s->frames = 0;

        if(init_fmt) {
                if (strcmp(init_fmt, "help") == 0) {
                        show_help();
                        return NULL;
                } else if (strncasecmp(init_fmt, "fps=", strlen("fps=")) == 0) {
                        s->fps = atoi(init_fmt + strlen("fps="));
                }
        }

        return s;
}

void vidcap_screen_finish(void *state)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        assert(s != NULL);
}

void vidcap_screen_done(void *state)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        assert(s != NULL);

        if(s->tile) {
                free(s->tile->data);
        }
        vf_free(s->frame);
        free(s);
}

struct video_frame * vidcap_screen_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        pthread_once(&initialized, initialize);

        *audio = NULL;

#ifndef HAVE_MACOSX
        glXMakeCurrent(s->dpy, s->root, s->glc);

        glDrawBuffer(GL_FRONT);

        /*                        
                                  glDrawBuffer(GL_FRONT);
                                  glx_swap(s->context);

                                  GLint ReadBuffer;
                                  glGetIntegerv(GL_READ_BUFFER,&ReadBuffer);
                                  glPixelStorei(GL_READ_BUFFER,GL_RGB);

                                  GLint PackAlignment;
                                  glGetIntegerv(GL_PACK_ALIGNMENT,&PackAlignment); 
                                  glPixelStorei(GL_PACK_ALIGNMENT,1);

                                  glPixelStorei(GL_PACK_ALIGNMENT, 3);
                                  glPixelStorei(GL_PACK_ROW_LENGTH, 0);
                                  glPixelStorei(GL_PACK_SKIP_ROWS, 0);
                                  glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
                                  */

        glBindTexture(GL_TEXTURE_2D, s->tex);

        glReadBuffer(GL_FRONT);

        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, s->tile->width, s->tile->height);
        //glCopyTexImage2D(GL_TEXTURE_2D,  0,  GL_RGBA,  0,  0,  s->tile->width,  s->tile->height,  0);
        //glReadPixels(0, 0, s->tile->width, s->tile->height, GL_RGBA, GL_UNSIGNED_BYTE, s->tile->data);
        //glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s->tile->width, s->tile->height,  GL_RGBA, GL_UNSIGNED_BYTE, s->tile->data);

        //gl_check_error();

        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->tex_out, 0);
        assert(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER));
        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); 

        glBindTexture(GL_TEXTURE_2D, s->tex);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0);
        glEnd();

        glReadBuffer(GL_COLOR_ATTACHMENT0_EXT); 

        glReadPixels(0, 0, s->tile->width, s->tile->height, GL_RGBA, GL_UNSIGNED_BYTE, s->tile->data);

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

#else
        CGImageRef image = CGDisplayCreateImage(s->display);
        CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(image));
        char *pixels = CFDataGetBytePtr(data);

        int linesize = s->tile->width * 4;
        int y;
        unsigned char *dst = s->tile->data;
        unsigned char *src = pixels;
        for(y = 0; y < s->tile->height; ++y) {
                vc_copylineRGBA (dst, src, linesize, 16, 8, 0);
                src += linesize;
                dst += linesize;
        }

        CFRelease(data);
        CFRelease(image);

#endif

        if(s->fps > 0.0) {
                struct timeval cur_time;

                gettimeofday(&cur_time, NULL);
                while(tv_diff_usec(cur_time, s->prev_time) < 1000000.0 / s->frame->fps) {
                        gettimeofday(&cur_time, NULL);
                }
                s->prev_time = cur_time;
        }

        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);        
        if (seconds >= 5) {
                float fps  = s->frames / seconds;
                fprintf(stderr, "[screen capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = s->t;
                s->frames = 0;
        }

        s->frames++;

#ifndef HAVE_MACOSX
        glXMakeCurrent(s->dpy, None, NULL);
#endif

        return s->frame;
}

