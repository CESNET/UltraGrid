/*
 * FILE:   display.h
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
 * $Revision: 1.2 $
 * $Date: 2007/12/11 19:16:45 $
 *
 */

#ifndef _VIDEO_DISPLAY_H
#define _VIDEO_DISPLAY_H

/*
 * Interface to probing the valid display types. 
 *
 */

typedef uint32_t	display_id_t;

typedef enum {
	DS_176x144,	/* Quarter CIF */
	DS_352x288,	/* CIF         */
	DS_702x576,	/* Super CIF   */
	DS_1280x720,	/* SMPTE 296M  */
	DS_1920x1080,	/* SMPTE 274M  */
	DS_NONE,
} display_size_t;

typedef enum {
	DC_YUV,
	DC_RGB,
	DC_NONE,
} display_colour_t;

typedef struct {
	display_size_t		size;
	display_colour_t	colour_mode;
	int			num_images;	/* Maximum displayable images, -1 = unlimited */
} display_format_t;

typedef struct {
	display_id_t		 id;
	const char		*name;		/* Single word name 		*/
	const char		*description;
	display_format_t	*formats;	/* Array of supported formats 	*/
	unsigned int		 num_formats;	/* Size of the array 		*/
} display_type_t;

int		 display_init_devices(void);
void		 display_free_devices(void);
int		 display_get_device_count(void);
display_type_t	*display_get_device_details(int index);
display_id_t 	 display_get_null_device_id(void);

/* 
 * Interface to initialize displays, and playout video
 *
 */

struct display;

struct display	*display_init(display_id_t id);
void 		 display_done(struct display *d);
char		*display_get_frame(struct display *d);
void 		 display_put_frame(struct display *d, char *frame);
display_colour_t display_get_colour_mode(struct display *d);

#endif /* _VIDEO_DISPLAY_H */
