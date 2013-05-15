/*
 * FILE:   display.h
 * AUTHOR: Colin Perkins <csp@csperkins.org>
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
 * $Revision: 1.4.2.4 $
 * $Date: 2010/02/05 13:56:49 $
 *
 */

#ifndef _VIDEO_DISPLAY_H
#define _VIDEO_DISPLAY_H

#include "video.h"

#define DISPLAY_FLAG_AUDIO_EMBEDDED (1<<1)
#define DISPLAY_FLAG_AUDIO_AESEBU (1<<2)
#define DISPLAY_FLAG_AUDIO_ANALOG (1<<3)

struct audio_frame;

/*
 * Interface to probing the valid display types. 
 *
 */

typedef uint32_t	display_id_t;

typedef struct {
        display_id_t		 id;
        const char		*name;		/* Single word name 		*/
        const char		*description;
} display_type_t;

#define DISPLAY_PROPERTY_CODECS  0 /* codec_t[] */
#define DISPLAY_PROPERTY_RSHIFT  1 /* int */
#define DISPLAY_PROPERTY_GSHIFT  2 /* int */
#define DISPLAY_PROPERTY_BSHIFT  3 /* int */
#define DISPLAY_PROPERTY_BUF_PITCH  4 /* int */
#define DISPLAY_PROPERTY_VIDEO_MODE 5 /* int */
#define DISPLAY_PROPERTY_SUPPORTED_IL_MODES 6 /* enum interlacing_t[] */

#define PITCH_DEFAULT -1 /* default to linesize */
#define DISPLAY_PROPERTY_VIDEO_MERGED          0
#define DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES  1

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

extern int display_init_noerr;

/**
 * Initializes video display
 *
 * @param fmt    command-line entered format string
 * @param flags  bit sum of DISPLAY_FLAG_* params defined above
 * @param[out]   display state if available, may be NULL if driver only shows help
 *               defined only if sucessful
 * @retval    0  if sucessful
 * @retval   -1  if failed
 * @retval    1  if successfully shown help
 */
int display_init(display_id_t id, char *fmt, unsigned int flags, struct display **state);

/**
 * This call is entered in main thread and the display may stay in this call until end of the program.
 * This is mainly for GUI displays (GL/SDL), which usually need to be run from main thread of the
 * program.
 * The function must finish after calling display_finish.
 */
void                     display_run(struct display *d);

/**  
 * This function performs final cleanup after display.
 */
void 	                 display_done(struct display *d);

/**
 * This function should tell the driver to finish (eg. unlocking semaphores, locks, signalling cv etc.)
 * When this function is called, should_exit variable is set to TRUE
 */
void 		         display_finish(struct display *d);

/**
 * Returns video frame which will be written to.
 * Currently there is a restriction on number of concurrently acquired frames - only one frame
 * can be hold at the moment.
 *
 * @return               video frame
 */
struct video_frame      *display_get_frame(struct display *d);

#define PUTF_BLOCKING 0
#define PUTF_NONBLOCK 1
/* TODO: figure out what with frame parameter, which is no longer used. Leave out? */
/**
 * Puts filled video frame.
 * Currnetly, it must be the frame previously obtained by display_get_frame. Moreover, every frame
 * acquired from video display should be put.
 */
int 		         display_put_frame(struct display *d, struct video_frame *frame, int nonblock);
/**
 * Tells display to reconfigure according to video description
 */
int                      display_reconfigure(struct display *d, struct video_desc desc);

/**
 * Gets property from video display
 * @param       property  one of DISPLAY_PROPERTY_* defines listed above
 * @param       val       output value localtion
 * @param       len       IN provided buffer length
 *                        OUT actual size written
 * @return      true      if succeeded and result is contained in val and len
 *              false     if the query didn't succeeded (either not supported or error)
 */
int                      display_get_property(struct display *d, int property, void *val, size_t *len);

/* 
 * Audio related functions (embedded audio)
 */
/*
 * TODO: currently unused - should be implemented at least for SDL (toggling fullscreen for RGB videos)
 */
/**
 * Puts audio data
 */
void                     display_put_audio_frame(struct display *d, struct audio_frame *frame);
/**
 * This function instructs video driver to reconfigure itself
 *
 * @param               d               video display structure
 * @param               quant_samples   number of bits per sample
 * @param               channels        count of channels
 * @param               sample_rate     samples per second
 */
int                      display_reconfigure_audio(struct display *d, int quant_samples, int channels, int sample_rate);

#endif /* _VIDEO_DISPLAY_H */
