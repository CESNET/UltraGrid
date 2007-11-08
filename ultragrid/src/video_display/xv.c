/*
 * FILE:   video_display/xv.c
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#ifndef X_DISPLAY_MISSING /* Don't try to compile if X is not present */

#include "debug.h"
#include "video_display.h"
#include "video_display/xv.h"
#include "tcl.h"
#include "tk.h"
#include "tv.h"

/* For X shared memory... */
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <X11/extensions/Xv.h>
#include <X11/extensions/Xvlib.h>
#include <X11/extensions/XShm.h>
#include <host.h>

extern int      XShmQueryExtension(Display*);
extern int      XShmGetEventBase(Display*);
extern XvImage  *XvShmCreateImage(Display*, XvPortID, int, char*, int, int, XShmSegmentInfo*);

#define HD_WIDTH 	hd_size_x
#define HD_HEIGHT 	hd_size_y
#define MAGIC_XV	DISPLAY_XV_ID


struct state_xv {
	Display			*display;
	Window			 window;
	GC			 gc;
	int			 vw_depth;
	Visual			*vw_visual;
	XvImage			*vw_image[3];
	XShmSegmentInfo		 vw_shm_segment[2];
	int			 image_display, image_network;
	XvAdaptorInfo		*ai;
	int			 xv_port;
	/* Thread related information follows... */
	pthread_t		 thread_id;
	pthread_mutex_t		 lock;
	pthread_cond_t		 boss_cv;
	pthread_cond_t		 worker_cv;
	int			 work_to_do;
	int			 boss_waiting;
	int			 worker_waiting;
	/* For debugging... */
	uint32_t		 magic;	
};

static void*
display_thread_xv(void *arg)
{
	struct state_xv        *s = (struct state_xv *) arg;

	while (1) {
		pthread_mutex_lock(&s->lock);

		while (s->work_to_do == FALSE) {
			s->worker_waiting = TRUE;
			pthread_cond_wait(&s->worker_cv, &s->lock);
			s->worker_waiting = FALSE;
		}

		s->work_to_do     = FALSE;

		if (s->boss_waiting) {
			pthread_cond_signal(&s->boss_cv);
		}
		pthread_mutex_unlock(&s->lock);

		XvShmPutImage(s->display, s->xv_port, s->window, s->gc, s->vw_image[s->image_display], 0, 0, 
			      s->vw_image[s->image_display]->width, s->vw_image[s->image_display]->height, 0, 0, HD_WIDTH, HD_HEIGHT, False);

		XFlush(s->display);
	}
	return NULL;
}


void *
display_xv_init(void)
{
	struct state_xv		*s;
	unsigned int		 p_version, p_release, p_request_base, p_event_base, p_error_base, p; 
	unsigned int          	 p_num_adaptors, i, j;

	s = (struct state_xv *) malloc(sizeof(struct state_xv));
	s->magic   = MAGIC_XV;
	s->xv_port = -1;

	if (!(s->display = XOpenDisplay(NULL))) {
		printf("Unable to open display.\n");
		return NULL;
	}

	/* Do we support the Xv extension? */
	if (XvQueryExtension(s->display, &p_version, &p_release, &p_request_base, &p_event_base, &p_error_base) != Success) {
		printf("Cannot activate Xv extension\n");
		abort();
	}
	debug_msg("Xv version       = %u.%u\n", p_version, p_release);
	debug_msg("Xv request base  = %u\n", p_request_base);
	debug_msg("Xv event base    = %u\n", p_event_base);
	debug_msg("Xv error base    = %u\n", p_error_base);
	if (XvQueryAdaptors(s->display, DefaultRootWindow(s->display), &p_num_adaptors, &s->ai) != Success) {
		printf("Cannot query Xv adaptors\n");
		abort();
	}
	s->xv_port = 0;
	debug_msg("Xv adaptor count = %d\n", p_num_adaptors);
	for (i = 0;  i < p_num_adaptors && s->xv_port == 0; i++) {
		debug_msg("Xv adaptor %d name = %s\n", i, s->ai[i].name);
		debug_msg("Xv adaptor %d type = %s%s%s%s%s\n", i, (s->ai[i].type & XvInputMask)   ? "[input] "  : "",
						  		  (s->ai[i].type & XvOutputMask)  ? "[output] " : "",
								  (s->ai[i].type & XvVideoMask)   ? "[video] "  : "",
								  (s->ai[i].type & XvStillMask)   ? "[still] "  : "",
								  (s->ai[i].type & XvImageMask)   ? "[image] "  : "");
		for (p = s->ai[i].base_id; p < s->ai[i].base_id + s->ai[i].num_ports; p++) {
			unsigned int		 encodings;
			int			 formats;
			int			 k;
			XvEncodingInfo		*ei;
			XvImageFormatValues	*fo;

                        if(!XvGrabPort(s->display, p, CurrentTime)) {
                                 debug_msg("Grabed port: %d\n", p);
                                 s->xv_port = p;
                         } else {
                                 debug_msg("Cannot grab port: %d\n", p);
                                 abort();
                         }

			if (XvQueryEncodings(s->display, p, &encodings, &ei) != Success) {
				printf("Cannot query Xv encodings\n");
				abort();
			}
			for (j = 0; j < encodings; j++) {
				debug_msg("Xv adaptor %d port %d coding %d = %s\n", i, p - s->ai[i].base_id, j, ei[j].name);
			}
			XvFreeEncodingInfo(ei);
			fo = XvListImageFormats(s->display, p, &formats);
			for (k = 0; k < formats; k++) {
				debug_msg("Xv adaptor %d port %d format %d = 0x%08lx %s\n", i, p - s->ai[i].base_id, k, fo[k].id, fo[k].guid);
			}
			if (fo != NULL) {
				XFree(fo);
			}
		}
	}

	
	s->window = XCreateSimpleWindow(s->display, DefaultRootWindow(s->display), 0, 0, HD_WIDTH, HD_HEIGHT, 0, XWhitePixel(s->display, DefaultScreen(s->display)), XBlackPixel(s->display, DefaultScreen(s->display)));

	s->gc = XCreateGC(s->display, s->window, 0, 0);

	XMapWindow(s->display, s->window);

	XStoreName(s->display, s->window, "UltraGrid");

	/* Create the image buffer, shared with the X server... */
	for (i = 0; i < 2; i++) {
		s->vw_image[i] = XvShmCreateImage(s->display, s->xv_port, 0x59565955, 0, HD_WIDTH, HD_HEIGHT, &s->vw_shm_segment[i]);
		if (s->vw_image[i] == NULL) {
			printf("Cannot create XV shared memory image\n");
			abort();
		}
		debug_msg("vw_image                   = %p\n", s->vw_image[i]);
		debug_msg("vw_image->width            = %d\n", s->vw_image[i]->width);
		debug_msg("vw_image->height           = %d\n", s->vw_image[i]->height);
		debug_msg("vw_image->data_size        = %d\n", s->vw_image[i]->data_size);
		if (s->vw_image[i]->width != (int)HD_WIDTH) {
			printf("Display does not support %d pixel wide Xvideo shared memory images\n", HD_WIDTH);
			abort();
		}
		if (s->vw_image[i]->height != (int)HD_HEIGHT) {
			printf("Display does not support %d pixel tall Xvideo shared memory images\n", HD_WIDTH);
			abort();
		}

		s->vw_shm_segment[i].shmid    = shmget(IPC_PRIVATE, s->vw_image[i]->data_size, IPC_CREAT|0777);
		s->vw_shm_segment[i].shmaddr  = shmat(s->vw_shm_segment[i].shmid, 0, 0);
		s->vw_shm_segment[i].readOnly = False;
		debug_msg("vw_shm_segment.shmid       = %d\n", s->vw_shm_segment[i].shmid);
		debug_msg("vw_shm_segment.shmaddr     = %d\n", s->vw_shm_segment[i].shmaddr);

		s->vw_image[i]->data = s->vw_shm_segment[i].shmaddr;

		if (XShmAttach(s->display, &s->vw_shm_segment[i]) == 0) {
			printf("Cannot attach shared memory segment\n");
			abort();
		}
	}
	s->image_network = 0;
	s->image_display = 1;

	/* Get our window onto the screen... */
	XFlush(s->display);

	pthread_mutex_init(&s->lock, NULL);
	pthread_cond_init(&s->boss_cv, NULL);
	pthread_cond_init(&s->worker_cv, NULL);
	s->work_to_do     = FALSE;
	s->boss_waiting   = FALSE;
	s->worker_waiting = TRUE;
	if (pthread_create(&(s->thread_id), NULL, display_thread_xv, (void *) s) != 0) {
		perror("Unable to create display thread\n");
		return NULL;
	}

	debug_msg("Window initialized %p\n", s);
	return (void *) s;
}

void
display_xv_done(void *state)
{
	int		 i;
	struct state_xv *s = (struct state_xv *) state;

	assert(s->magic == MAGIC_XV);

	for (i = 0; i < 2; i++) {
		XShmDetach(s->display, &(s->vw_shm_segment[i]));
		shmdt(s->vw_shm_segment[i].shmaddr);
		shmctl(s->vw_shm_segment[i].shmid, IPC_RMID, 0);
		//XDestroyImage(s->vw_image[i]);
	}
	XvFreeAdaptorInfo(s->ai);
}

char *
display_xv_getf(void *state)
{
	struct state_xv *s = (struct state_xv *) state;
	assert(s->magic == MAGIC_XV);
	return s->vw_image[s->image_network]->data;
}

int
display_xv_putf(void *state, char *frame)
{
	int		 tmp;
	struct state_xv *s = (struct state_xv *) state;

	assert(s->magic == MAGIC_XV);
	assert(frame == s->vw_image[s->image_network]->data);

	pthread_mutex_lock(&s->lock);
	/* Wait for the worker to finish... */
	while (s->work_to_do) {
		s->boss_waiting = TRUE;
		pthread_cond_wait(&s->boss_cv, &s->lock);
		s->boss_waiting = FALSE;
	}

	/* ...and give it more to do... */
	tmp = s->image_display;
	s->image_display = s->image_network;
	s->image_network = tmp;
	s->work_to_do    = TRUE;

	/* ...and signal the worker */
	if (s->worker_waiting) {
		pthread_cond_signal(&s->worker_cv);
	}
	pthread_mutex_unlock(&s->lock);
	return 0;
}

display_colour_t
display_xv_colour(void *state)
{
	struct state_xv *s = (struct state_xv *) state;
	assert(s->magic == MAGIC_XV);
	return DC_YUV;
}

display_type_t *
display_xv_probe(void)
{
        display_type_t          *dt;
        display_format_t        *dformat;

	dformat = malloc(4 * sizeof(display_format_t));
	dformat[0].size        = DS_176x144;
        dformat[0].colour_mode = DC_YUV;
        dformat[0].num_images  = 1;
	dformat[1].size        = DS_352x288;
        dformat[1].colour_mode = DC_YUV;
        dformat[1].num_images  = 1;
	dformat[2].size        = DS_702x576;
        dformat[2].colour_mode = DC_YUV;
        dformat[2].num_images  = 1;
	dformat[3].size        = DS_1280x720;
        dformat[3].colour_mode = DC_YUV;
        dformat[3].num_images  = 1;

        dt = malloc(sizeof(display_type_t));
        if (dt != NULL) {
		dt->id	        = DISPLAY_XV_ID;
		dt->name        = "xv";
		dt->description = "X Window System with Xvideo extension";
                dt->formats     = dformat;
                dt->num_formats = 4;
        }
        return dt;
}

#endif /* X_DISPLAY_MISSING */

