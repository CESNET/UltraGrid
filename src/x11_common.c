/**
 * @file   x11_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2018 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
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
 */
/**
 * @todo
 * Perhaps remove the whole stuff (+ resouce manager). This used to be used to
 * solve some crashes with access to X11 from within multiple threads (IIRC
 * RTDXT and X11-based display - GL/SDL?).
 *
 * This may not be needed - individual modues create its own non-shared Xlib
 * connection and XInitThreads() is called (this also may not be required).
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "x11_common.h"

#include <pthread.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include "debug.h"
#include "utils/resource_manager.h"

#define resource_symbol "X11-state"

struct x11_state {
        Display *display;
        pthread_mutex_t lock;
        int display_opened_here; /* indicates wheather we opened the display
                                          in such a case, we count references and if 0,
                                          we close it */
        int ref_num;
        bool initialized;
};

struct x11_state *get_state(void);

struct x11_state *get_state() {
        struct x11_state *state;
        rm_lock();
        state = (struct x11_state *) rm_get_shm(resource_symbol, sizeof(struct x11_state));
        if(!state->initialized) {
                state->display = NULL;
                pthread_mutex_init(&state->lock, NULL);
                state->display_opened_here = TRUE;
                state->ref_num = 0;
                state->initialized = true;
        }
        rm_unlock();

        return state;
}

void x11_set_display(void *disp)
{
        struct x11_state *s = get_state();
        Display *d = disp;
        if (d == NULL)
                return;
        pthread_mutex_lock(&s->lock);
        if(s->display != NULL) {
                fprintf(stderr, __FILE__ ": Fatal error: Display already set.\n");
                abort();
        }
        s->display = d;
        s->display_opened_here = FALSE;
        pthread_mutex_unlock(&s->lock);
}

void * x11_acquire_display(void)
{
        struct x11_state *s = get_state();
        if(!s->display) {
                s->display = XOpenDisplay(0);
                s->display_opened_here = TRUE;
        }
        
        if ( !s->display )
        {
                fprintf(stderr, "Failed to open X display\n" );
                return NULL;
        }
        
        s->ref_num++;
        return s->display;
}

void * x11_get_display(void)
{
        struct x11_state *s = get_state();
        return s->display;
}

void x11_release_display() {
        struct x11_state *s = get_state();
        s->ref_num--;
        
        if(s->ref_num < 0) {
                fprintf(stderr, __FILE__ ": WARNING: Unpaired glx_free call.");
        }
        
        if(s->display_opened_here && s->ref_num == 0) {
                fprintf(stderr, "Display closed (last client disconnected)\n");
                XCloseDisplay( s->display );
                s->display = NULL;
        }
}

void x11_lock(void)
{
        struct x11_state *s = get_state();
        pthread_mutex_lock(&s->lock);
}

void x11_unlock(void)
{
        struct x11_state *s = get_state();
        pthread_mutex_unlock(&s->lock);
}

/*
 * Copyright (C) 2017 Alberts Muktupāvels
 * Code to disable window decorations taken from:
 * https://gist.github.com/muktupavels/d03bb14ea6042b779df89b4c87df975d
 */
typedef struct
{
	unsigned long flags;
	unsigned long functions;
	unsigned long decorations;
	long input_mode;
	unsigned long status;
} MotifWmHints;

static MotifWmHints * get_motif_wm_hints (Display *display, Window window)
{
	Atom property;
	int result;
	Atom actual_type;
	int actual_format;
	unsigned long nitems;
	unsigned long bytes_after;
	unsigned char *data;

	property = XInternAtom (display, "_MOTIF_WM_HINTS", False);
	result = XGetWindowProperty (display, window, property,
			0, LONG_MAX, False, AnyPropertyType,
			&actual_type, &actual_format,
			&nitems, &bytes_after, &data);

	if (result == Success && data != NULL)
	{
		size_t data_size;
		size_t max_size;
		MotifWmHints *hints;

		data_size = nitems * sizeof (long);
		max_size = sizeof (*hints);

		hints = calloc (1, max_size);

		memcpy (hints, data, data_size > max_size ? max_size : data_size);
		XFree (data);

		return hints;
	}

	return NULL;
}

void x11_unset_window_decorations (Display *display, Window window)
{
	MotifWmHints *hints;
	Atom property;
	int nelements;

	hints = get_motif_wm_hints (display, window);
	if (hints == NULL)
	{
		hints = calloc (1, sizeof (*hints));
		hints->decorations = (1L << 0);
	}

	hints->flags |= (1L << 1);
	hints->decorations = 0;

	property = XInternAtom (display, "_MOTIF_WM_HINTS", False);
	nelements = sizeof (*hints) / sizeof (long);

	XChangeProperty (display, window, property, property, 32, PropModeReplace,
			(unsigned char *) hints, nelements);

	free (hints);
}

