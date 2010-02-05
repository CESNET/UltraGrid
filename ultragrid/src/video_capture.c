/*
 * FILE:   video_capture.c
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2004 University of Southern California
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
 * $Revision: 1.9.2.1 $
 * $Date: 2010/01/30 20:07:35 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "video_codec.h"
#include "video_capture.h"
#include "video_capture/firewire_dv_freebsd.h"
#include "video_capture/hdstation.h"
#include "video_capture/quicktime.h"
#include "video_capture/testcard.h"
#include "video_capture/null.h"
#include "video_capture/decklink.h"

#define VIDCAP_MAGIC	0x76ae98f0

struct vidcap {
	void			*state;
	int			 index;
	uint32_t		 magic;		/* For debugging */
};

struct vidcap_device_api {
	vidcap_id_t		 id;
	struct vidcap_type	*(*func_probe)(void);
	void			*(*func_init)(char *fmt);
	void			 (*func_done)(void *state);
	struct video_frame	*(*func_grab)(void *state);
};

struct vidcap_device_api vidcap_device_table[] = {
#ifdef HAVE_FIREWIRE_DV_FREEBSD
	{
		/* A FireWire DV capture card, on FreeBSD */
		0,
		vidcap_dvbsd_probe,
		vidcap_dvbsd_init,
		vidcap_dvbsd_done,
		vidcap_dvbsd_grab
	},
#endif /* HAVE_FIREWIRE_DV_FREEBSD */
#ifdef HAVE_HDSTATION
	{
		/* The DVS HDstation capture card */
		0,
		vidcap_hdstation_probe,
		vidcap_hdstation_init,
		vidcap_hdstation_done,
		vidcap_hdstation_grab
	},
#endif /* HAVE_HDSTATION */
#ifdef HAVE_DECKLINK
	{
		/* The Blackmagic DeckLink capture card */
		0,
		vidcap_decklink_probe,
		vidcap_decklink_init,
		vidcap_decklink_done,
		vidcap_decklink_grab
	},
#endif /* HAVE_DECKLINK */
#ifdef HAVE_QUAD 
        {
                /* The HD-SDI Master Quad capture card */
                0,
                vidcap_quad_probe,
                vidcap_quad_init,
                vidcap_quad_done,
                vidcap_quad_grab
        },
#endif /* HAVE_QUAD */ 
#ifdef HAVE_MACOSX
	{
		/* The QuickTime API */
		0,
		vidcap_quicktime_probe,
		vidcap_quicktime_init,
		vidcap_quicktime_done,
		vidcap_quicktime_grab
	},
#endif /* HAVE_MACOSX */
	{
		/* Dummy sender for testing purposes */
		0,
		vidcap_testcard_probe,
		vidcap_testcard_init,
		vidcap_testcard_done,
		vidcap_testcard_grab
	},
	{
		0,
		vidcap_null_probe,
		vidcap_null_init,
		vidcap_null_done,
		vidcap_null_grab
	}
};

#define VIDCAP_DEVICE_TABLE_SIZE (sizeof(vidcap_device_table)/sizeof(struct vidcap_device_api))

/* API for probing capture devices ****************************************************************/

static struct vidcap_type	*available_devices[VIDCAP_DEVICE_TABLE_SIZE];
static int			 available_device_count = 0;

int
vidcap_init_devices(void)
{
	unsigned int		 i;
	struct vidcap_type	*dt;

	assert(available_device_count == 0);

	for (i = 0; i < VIDCAP_DEVICE_TABLE_SIZE; i++) {
		//printf("probe: %d\n",i);
		dt = vidcap_device_table[i].func_probe();
		if (dt != NULL) {
			vidcap_device_table[i].id = dt->id;
			available_devices[available_device_count++] = dt;
		}
	}

	return available_device_count;
}

void
vidcap_free_devices(void)
{
	int	i;

	for (i = 0; i < available_device_count; i++) {
		free(available_devices[i]);
		available_devices[i] = NULL;
	}
	available_device_count = 0;
}

int
vidcap_get_device_count(void)
{
	return available_device_count;
}

struct vidcap_type *
vidcap_get_device_details(int index)
{
	assert(index < available_device_count);
	assert(available_devices[index] != NULL);

	return available_devices[index];
}

vidcap_id_t
vidcap_get_null_device_id(void)
{
	return VIDCAP_NULL_ID;
}

/* API for video capture **************************************************************************/

struct vidcap *
vidcap_init(vidcap_id_t id, char *fmt)
{
	unsigned int	i;

	for (i = 0; i < VIDCAP_DEVICE_TABLE_SIZE; i++) {
		if (vidcap_device_table[i].id == id) {
			struct vidcap *d = (struct vidcap *) malloc(sizeof(struct vidcap));
			d->magic = VIDCAP_MAGIC;
			d->state = vidcap_device_table[i].func_init(fmt);
			d->index = i;
			if (d->state == NULL) {
				debug_msg("Unable to start video capture device 0x%08lx\n", id);
				free(d);
				return NULL;
			} 
			return d;
		}
	}
	debug_msg("Unknown video capture device: 0x%08x\n", id);
	return NULL;
}

void 
vidcap_done(struct vidcap *state)
{
	assert(state->magic == VIDCAP_MAGIC);
	vidcap_device_table[state->index].func_done(state->state);
	free(state);
}

struct video_frame *
vidcap_grab(struct vidcap *state)
{
	assert(state->magic == VIDCAP_MAGIC);
	return vidcap_device_table[state->index].func_grab(state->state);
}

