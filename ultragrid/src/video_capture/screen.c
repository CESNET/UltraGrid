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

#include "video_display.h"
#include "video.h"

#include <GL/glew.h>

#include <pthread.h>

#ifndef HAVE_MACOSX
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
#if 0
        printf("Aggregate capture\n");
        printf("Usage\n");
        printf("\t-t aggregate:<dev1_config>#<dev2_config>[#....]\n");
        printf("\t\twhere devn_config is a complete configuration string of device involved in an aggregate device\n");
#endif

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
        Display *dpy;
        Window root;
        GLXContext glc;

        GLuint tex;
        GLuint tex_out;
        GLuint fbo;
};

pthread_once_t initialized = PTHREAD_ONCE_INIT;

static void initialize() {
	struct vidcap_screen_state *s = (struct vidcap_aggregate_state *) state;


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
#endif
        
        GLenum err = glewInit();
        if (GLEW_OK != err)
        {
          /* Problem: glewInit failed, something is seriously wrong. */
          fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
          abort();
        }

        s->tile->width = 1920;
        s->tile->height = 1080;

        s->frame->color_spec = RGBA;
        s->frame->fps = 30;
        s->frame->interlacing = PROGRESSIVE;
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;
        s->tile->data = (char *) malloc(s->tile->data_len);

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

}



struct vidcap_type *
vidcap_screen_probe(void)
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

void *
vidcap_screen_init(char *init_fmt, unsigned int flags)
{
	struct vidcap_screen_state *s;

	printf("vidcap_screen_init\n");

        UNUSED(flags);


        state = s = (struct vidcap_screen_state *) malloc(sizeof(struct vidcap_screen_state));
	if(s == NULL) {
		printf("Unable to allocate screen capture state\n");
		return NULL;
	}

        s->frame = NULL;
        s->tile = NULL;


        s->frames = 0;

        if(init_fmt && strcmp(init_fmt, "help") == 0) {
                show_help();
                return NULL;
        }

	return s;
}

void
vidcap_screen_finish(void *state)
{
	struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

	assert(s != NULL);
}

void
vidcap_screen_done(void *state)
{
	struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

	assert(s != NULL);

        if(s->tile) {
                free(s->tile->data);
        }
        vf_free(s->frame);
        free(s);
}

struct video_frame *
vidcap_screen_grab(void *state, struct audio_frame **audio)
{
	struct vidcap_screen_state *s = (struct vidcap_aggregate_state *) state;

        pthread_once(&initialized, initialize);

        *audio = NULL;

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

        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, s->tile->height, s->tile->width, s->tile->height);
        //glCopyTexImage2D(GL_TEXTURE_2D,  0,  GL_RGBA,  0,  0,  s->tile->width,  s->tile->height,  0);
        glReadPixels(0, 0, s->tile->width, s->tile->height, GL_RGBA, GL_UNSIGNED_BYTE, s->tile->data);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, s->tile->width, s->tile->height,  GL_RGBA, GL_UNSIGNED_BYTE, s->tile->data);

        //gl_check_error();

        glBindFramebuffer(GL_FRAMEBUFFER, s->fbo);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, s->tex_out, 0);
        assert(GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER));
        glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT); 

        glBindTexture(GL_TEXTURE_2D, s->tex);

        glClearColor(0.5,0,0,0);
        glClear(GL_COLOR_BUFFER_BIT);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex2f(-1.0, 1.0);
        glTexCoord2f(1.0, 0.0); glVertex2f(1.0, 1.0);
        glTexCoord2f(1.0, 1.0); glVertex2f(1.0, -1.0);
        glTexCoord2f(0.0, 1.0); glVertex2f(-1.0, -1.0);
        glEnd();

        glReadBuffer(GL_COLOR_ATTACHMENT0_EXT); 

        glReadPixels(0, 0, s->tile->width, s->tile->height, GL_RGBA, GL_UNSIGNED_BYTE, s->tile->data);
static int i;
if (i++ == -1)  {
int fd = open("res.rgb", O_CREAT| O_WRONLY, 0666);
assert(fd != -1);
write(fd, s->tile->data, s->tile->data_len);
close(fd); abort();
}


        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);

        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);        
        if (seconds >= 5) {
                float fps  = s->frames / seconds;
                fprintf(stderr, "[screen capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = s->t;
                s->frames = 0;
        }

        s->frames++;

        glXMakeCurrent(s->dpy, None, NULL);

	return s->frame;
}

