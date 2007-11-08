/*
 * FILE:   video_display/x11.c
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *
 * Copyright (c) 2001-2003 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#define NDEF
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifndef X_DISPLAY_MISSING /* Don't try to compile if X is not present */

#include "debug.h"
#include "video_display.h"
#include "video_display/x11.h"

/* For X shared memory... */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <X11/extensions/XShm.h>
#include "host.h"

#define HD_WIDTH 	hd_size_x
#define HD_HEIGHT 	hd_size_y
#define MAGIC_X11	0xcafebabe

struct state_x11 {
	 Display		*display;
	 Window			 window;
	 GC			 gc;
	 int			 vw_depth;
	 Visual			*vw_visual;
	 XImage			*vw_image;
	 XShmSegmentInfo	 vw_shm_segment;
	 int			 xv_port;
	 uint32_t		 magic;			/* For debugging */
};

void *
display_x11_init(void)
{
#ifdef NDEF
	struct state_x11	*s;

	s = (struct state_x11 *) malloc(sizeof(struct state_x11));
	s->magic   = MAGIC_X11;

	/* Create a bare window to draw into... */
        if (!(s->display = XOpenDisplay(NULL))) {
                printf("Unable to open display.\n");
                abort();
        }
        s->window = XCreateSimpleWindow(s->display, DefaultRootWindow(s->display), 0, 0, HD_WIDTH, HD_HEIGHT, 0, BlackPixel(s->display, DefaultScreen(s->display)), BlackPixel(s->display,  DefaultScreen(s->display)));
        if (s->window == 0) {
		abort();
	};
        XMapWindow(s->display, s->window);
	XStoreName(s->display, s->window, "UltraGrid");

        s->vw_depth  = DefaultDepth(s->display, DefaultScreen(s->display));
        s->vw_visual = DefaultVisual(s->display, DefaultScreen(s->display));

	/* For now, we only support 24-bit TrueColor displays... */
	if (s->vw_depth != 24) {
		printf("Unable to open display: not 24 bit colour\n");
		return NULL;
	}
	if (s->vw_visual->class != TrueColor) {
		printf("Unable to open display: not TrueColor visual\n");
		return NULL;
	}

	/* Do the shared memory magic... */
        s->vw_image = XShmCreateImage(s->display, s->vw_visual, s->vw_depth, ZPixmap, NULL, &s->vw_shm_segment, HD_WIDTH, HD_HEIGHT);
	debug_msg("vw_image                   = %p\n", s->vw_image);
	debug_msg("vw_image->width            = %d\n", s->vw_image->width);
	debug_msg("vw_image->height           = %d\n", s->vw_image->height);
	if (s->vw_image->width != (int)HD_WIDTH) {
		printf("Display does not support %d pixel wide shared memory images\n", HD_WIDTH);
		abort();
	}
	if (s->vw_image->height != (int)HD_HEIGHT) {
		printf("Display does not support %d pixel tall shared memory images\n", HD_WIDTH);
		abort();
	}

	s->vw_shm_segment.shmid    = shmget(IPC_PRIVATE, s->vw_image->bytes_per_line * s->vw_image->height, IPC_CREAT|0777);
	s->vw_shm_segment.shmaddr  = shmat(s->vw_shm_segment.shmid, 0, 0);
	s->vw_shm_segment.readOnly = False;
	debug_msg("vw_shm_segment.shmid       = %d\n", s->vw_shm_segment.shmid);
	debug_msg("vw_shm_segment.shmaddr     = %d\n", s->vw_shm_segment.shmaddr);

	s->vw_image->data = s->vw_shm_segment.shmaddr;

        if (XShmAttach(s->display, &s->vw_shm_segment) == 0) {
		printf("Cannot attach shared memory segment\n");
		abort();
	}

	/* Get our window onto the screen... */
	XFlush(s->display);

	s->gc =  XCreateGC(s->display, s->window, 0, NULL);

	printf ("X11 init done\n");
	return (void *) s;
#endif
	return NULL;
}

void
display_x11_done(void *state)
{
	struct state_x11 *s = (struct state_x11 *) state;

	assert(s->magic == MAGIC_X11);

	XShmDetach(s->display, &(s->vw_shm_segment));
	XDestroyImage(s->vw_image);
	shmdt(s->vw_shm_segment.shmaddr);
	shmctl(s->vw_shm_segment.shmid, IPC_RMID, 0);
}

char *
display_x11_getf(void *state)
{
	struct state_x11 *s = (struct state_x11 *) state;
	assert(s->magic == MAGIC_X11);
	return s->vw_image->data;
}

int
display_x11_putf(void *state, char *frame)
{
	struct state_x11 *s = (struct state_x11 *) state;

	assert(s->magic == MAGIC_X11);
	assert(frame == s->vw_image->data);
	XShmPutImage(s->display, s->window, s->gc, s->vw_image, 0, 0, 0, 0, s->vw_image->width, s->vw_image->height, True);
	XFlush(s->display);
	return 0;
}

display_colour_t
display_x11_colour(void *state)
{
	struct state_x11 *s = (struct state_x11 *) state;
	assert(s->magic == MAGIC_X11);
	return DC_YUV;
}

display_type_t *
display_x11_probe(void)
{
	display_type_t		*dt;
	display_format_t	*dformat;

	dformat = malloc(3 * sizeof(display_format_t));
	dformat[0].size        = DS_176x144;
	dformat[0].colour_mode = DC_RGB;
	dformat[0].num_images  = -1;
	dformat[1].size        = DS_352x288;
	dformat[1].colour_mode = DC_RGB;
	dformat[1].num_images  = -1;
	dformat[2].size        = DS_702x576;
	dformat[2].colour_mode = DC_RGB;
	dformat[2].num_images  = -1;

	dt = malloc(sizeof(display_type_t));
	if (dt != NULL) {
		dt->id	        = DISPLAY_X11_ID;
		dt->name        = "x11";
		dt->description = "X Window System";
		dt->formats     = dformat;
		dt->num_formats = 3;
	}
	return dt;
}

#endif /* X_DISPLAY_MISSING */

