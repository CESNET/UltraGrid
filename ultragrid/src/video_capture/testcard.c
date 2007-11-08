/*
 * FILE:   testcard.c
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *
 * A fake video capture device, used for systems that either have no capture
 * hardware or do not wish to transmit. This fits the interface of the other
 * capture devices, but never produces any video.
 *
 * Copyright (c) 2004 University of Glasgow
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
#include "debug.h"
#include "tv.h"
#include "video_types.h"
#include "video_capture.h"
#include "video_capture/testcard.h"
#include "host.h"

extern  char	testcard_image[];

struct testcard_state {
	struct timeval	last_frame_time;
	int		fps;
	int		count;
};

void *
vidcap_testcard_init(int fps)
{
	struct testcard_state	*s;

	s = malloc(sizeof(struct testcard_state));
	if (s != NULL) {
		s->fps   = fps;
		s->count = 0;
		gettimeofday(&(s->last_frame_time), NULL);
	}
	return s;
}

void
vidcap_testcard_done(void *state)
{
	free(state);
}

struct video_frame *
vidcap_testcard_grab(void *arg)
{
	struct timeval		 curr_time;
	struct testcard_state	*state;
	struct video_frame	*vf;

	state = (struct testcard_state *) arg;

	gettimeofday(&curr_time, NULL);
	if (tv_diff(curr_time, state->last_frame_time) > 1.0/(double)state->fps) {
		state->last_frame_time = curr_time;
		state->count++;

		printf("Sending frame %d\n", state->count);

		vf = (struct video_frame *) malloc(sizeof(struct video_frame));
		if (vf != NULL) {
			vf->colour_mode = YUV_422;
			vf->width       = hd_size_x;
			vf->height      = hd_size_y;
			vf->data        = testcard_image;
			vf->data_len	= hd_size_x * hd_size_y * hd_color_bpp;
		}
		return vf;
	}
	return NULL;
}

struct vidcap_type *
vidcap_testcard_probe(void)
{
	struct vidcap_type	*vt;

	vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
	if (vt != NULL) {
		vt->id          = VIDCAP_TESTCARD_ID;
		vt->name        = "testcard";
		vt->description = "Video testcard";
		vt->width       = hd_size_x;
		vt->height      = hd_size_y;
		vt->colour_mode = YUV_422;
	}
	return vt;
}

