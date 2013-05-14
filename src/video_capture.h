/*
 * FILE:   video_capture.h
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2002 University of Southern California
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
 */

/*
 * API for probing the valid video capture devices. 
 *
 */

#ifndef _VIDEO_CAPTURE_H_
#define _VIDEO_CAPTURE_H_

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#define VIDCAP_FLAG_AUDIO_EMBEDDED (1<<1)
#define VIDCAP_FLAG_AUDIO_AESEBU (1<<2)
#define VIDCAP_FLAG_AUDIO_ANALOG (1<<3)

typedef uint32_t	vidcap_id_t;

struct audio_frame;

struct vidcap_type {
	vidcap_id_t		 id;
	const char		*name;
	const char		*description;
	unsigned	 	 width;
	unsigned	 	 height;
	//video_colour_mode_t	 colour_mode;
};

int			 vidcap_init_devices(void);
void			 vidcap_free_devices(void);
int			 vidcap_get_device_count(void);
struct vidcap_type	*vidcap_get_device_details(int index);
vidcap_id_t 		 vidcap_get_null_device_id(void);

/*
 * API for video capture. Normal operation is something like:
 *
 * 	v = vidcap_init(id);
 *	...
 *	while (!done) {
 *		...
 *		f = vidcap_grab(v, timeout);
 *		...use the frame "f"
 *	}
 *	vidcap_stop(v);
 *
 * Where the "id" parameter to vidcap_init() is obtained from
 * the probing API. The vidcap_grab() function returns a pointer
 * to the frame, or NULL if no frame is currently available. It
 * does not block.
 *
 */

struct vidcap;

/**
 * Semantics is similar to the semantic of display_init
 *
 * @see display_init
 */
int                      vidcap_init(vidcap_id_t id, char *fmt, unsigned int flags, struct vidcap **);
void			 vidcap_done(struct vidcap *state);
void			 vidcap_finish(struct vidcap *state);
struct video_frame	*vidcap_grab(struct vidcap *state, struct audio_frame **audio);

extern int vidcap_init_noerr;

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _VIDEO_CAPTURE_H_

