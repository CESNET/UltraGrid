/*
 * FILE:   video_display.c
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2003 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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
 * $Revision: 1.8.2.4 $
 * $Date: 2010/02/05 13:56:49 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "video_display.h"
#include "video_display/null.h"
#include "video_display/xv.h"
#include "video_display/x11.h"
#include "video_display/sdl.h"
#include "video_display/hdstation.h"
#include "video_display/gl_sdl.h"
#include "video_display/dxt.h"
#include "video_display/quicktime.h"
#include "video_display/sage.h"

/*
 * Interface to probing the valid display types. 
 *
 */

typedef struct {
	display_id_t		 id;
	display_type_t 		*(*func_probe)(void);
	void			*(*func_init)(char *fmt);
	void 			 (*func_done)(void *state);
	char 			*(*func_getf)(void *state);
	int  			 (*func_putf)(void *state, char *frame);
	display_colour_t	 (*func_colour)(void *state);
} display_table_t;

static display_table_t display_device_table[] = {
#ifndef X_DISPLAY_MISSING
	{
		0,
		display_xv_probe,
		display_xv_init,
		display_xv_done,
		display_xv_getf,
		display_xv_putf,
		display_xv_colour,
	},
	{
		0,
		display_x11_probe,
		display_x11_init,
		display_x11_done,
		display_x11_getf,
		display_x11_putf,
		display_x11_colour,
	},
#ifdef HAVE_SDL
	{	
		0,
                display_sdl_probe,
                display_sdl_init,
                display_sdl_done,
                display_sdl_getf,
                display_sdl_putf,
                display_sdl_colour, 
	},
#ifdef HAVE_GL
	{	
		0,
                display_gl_probe,
                display_gl_init,
                display_gl_done,
                display_gl_getf,
                display_gl_putf,
                display_gl_colour,
	},
#ifdef HAVE_SAGE
	{
        0,
                display_sage_probe,
                display_sage_init,
                display_sage_done,
                display_sage_getf,
                display_sage_putf,
                display_sage_colour,
	},
#endif /* HAVE_SAGE */
#ifdef HAVE_FASTDXT
	{	
		0,
                display_dxt_probe,
                display_dxt_init,
                display_dxt_done,
                display_dxt_getf,
                display_dxt_putf,
                display_dxt_colour, 
	},
#endif /* HAVE_FASTDXT */
#endif /* HAVE_GL */
#endif /* HAVE_SDL */
#endif /* X_DISPLAY_MISSING */
#ifdef HAVE_HDSTATION
	{
		0,
		display_hdstation_probe,
		display_hdstation_init,
		display_hdstation_done,
		display_hdstation_getf,
		display_hdstation_putf,
		display_hdstation_colour,
	},
#endif /* HAVE_HDSTATION */
#ifdef HAVE_MACOSX
	{
		0,
		display_quicktime_probe,
		display_quicktime_init,
		display_quicktime_done,
		display_quicktime_getf,
		display_quicktime_putf,
		display_quicktime_colour,
	},
#endif /* HAVE_MACOSX */
	{
		0,
		display_null_probe,
		display_null_init,
		display_null_done,
		display_null_getf,
		display_null_putf,
		display_null_colour,
	}
};

#define DISPLAY_DEVICE_TABLE_SIZE (sizeof(display_device_table) / sizeof(display_table_t))

static display_type_t	*available_devices[DISPLAY_DEVICE_TABLE_SIZE];
static int		 available_device_count = 0;

int
display_init_devices(void)
{
	unsigned int	 i;
	display_type_t	*dt;

	assert(available_device_count == 0);

	for (i = 0; i < DISPLAY_DEVICE_TABLE_SIZE; i++) {
		dt = display_device_table[i].func_probe();
		if (dt != NULL) {
			display_device_table[i].id = dt->id;
			available_devices[available_device_count++] = dt;
		}
	}
	return 0;
}

void 
display_free_devices(void)
{
	int	i;

	for (i = 0; i < available_device_count; i++) {
		free(available_devices[i]);
		available_devices[i] = NULL;
	}
	available_device_count = 0;
}

int
display_get_device_count(void)
{
	return available_device_count;
}

display_type_t *
display_get_device_details(int index)
{
	assert(index < available_device_count);
	assert(available_devices[index] != NULL);

	return available_devices[index];
}

display_id_t
display_get_null_device_id(void)
{
	return DISPLAY_NULL_ID;
}

/*
 * Display initialisation and playout routines...
 */

#define DISPLAY_MAGIC 0x01ba7ef1

struct display {
	uint32_t	 magic;
	int		 index;
	void		*state;
};

struct display *
display_init(display_id_t id, char *fmt)
{
	unsigned int	i;

	for (i = 0; i < DISPLAY_DEVICE_TABLE_SIZE; i++) {
		if (display_device_table[i].id == id) {
			struct display *d = (struct display *) malloc(sizeof(struct display));
			d->magic = DISPLAY_MAGIC;
			d->state = display_device_table[i].func_init(fmt);
			d->index = i;
			if (d->state == NULL) {
				debug_msg("Unable to start display 0x%08lx\n", id);
				free(d);
				return NULL;
			} 
			return d;
		}
	}
	debug_msg("Unknown display id: 0x%08x\n", id);
	return NULL;
}

void
display_done(struct display *d)
{
	assert(d->magic == DISPLAY_MAGIC);
	display_device_table[d->index].func_done(d->state);
}

struct video_frame*
display_get_frame(struct display *d)
{
	assert(d->magic == DISPLAY_MAGIC);
	return display_device_table[d->index].func_getf(d->state);
}

void
display_put_frame(struct display *d, char *frame)
{
	assert(d->magic == DISPLAY_MAGIC);
	display_device_table[d->index].func_putf(d->state, frame);
}

display_colour_t 
display_get_colour_mode(struct display *d)
{
	assert(d->magic == DISPLAY_MAGIC);
	return display_device_table[d->index].func_colour(d->state);
}

