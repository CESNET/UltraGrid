/**
 * @file   video_display.h
 * @author Colin Perkins    <csp@csperkins.org>
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @ingroup display
 */
/* Copyright (c) 2001-2003 University of Southern California
 * Copyright (c) 2005-2013 CESNET z.s.p.o.
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
 */

/**
 * @defgroup display Video Display
 *
 * API for video display
 *
 * @{
 */
#ifndef _VIDEO_DISPLAY_H
#define _VIDEO_DISPLAY_H

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/** @anchor display_flags
 * @name Initialization Flags
 * @{ */
#define DISPLAY_FLAG_AUDIO_EMBEDDED (1<<1)
#define DISPLAY_FLAG_AUDIO_AESEBU (1<<2)
#define DISPLAY_FLAG_AUDIO_ANALOG (1<<3)
/** @} */

struct audio_frame;

/*
 * Interface to probing the valid display types. 
 */
typedef uint32_t	display_id_t; ///< driver unique ID

/** Defines video display device */
typedef struct {
        display_id_t		 id;          ///< @copydoc display_id_t
        const char		*name;        ///< single word name
        const char		*description; ///< longer device description
} display_type_t;

/** @name Display Properties
 * @{ */
enum display_property {
        DISPLAY_PROPERTY_CODECS = 0, ///< list of natively supported codecs - codec_t[]
        DISPLAY_PROPERTY_RGB_SHIFT = 1, ///< red,green,blue shift - int[3] (bits)
        DISPLAY_PROPERTY_BUF_PITCH = 2, ///< requested framebuffer pitch - int (bytes), may be @ref PITCH_DEFAULT
        DISPLAY_PROPERTY_VIDEO_MODE = 3, ///< requested video mode - int (one of @ref display_prop_vid_mode)
        DISPLAY_PROPERTY_SUPPORTED_IL_MODES = 4 ///< display supported interlacing modes - enum interlacing_t[]
};

#define PITCH_DEFAULT -1 ///< default pitch, i. e. respective linesize

enum display_prop_vid_mode {
        DISPLAY_PROPERTY_VIDEO_MERGED         = 0, ///< monolithic framebuffer
        DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES = 1  ///< framebuffer consists of separate tiles
};
/// @}

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

void                     list_video_display_devices(void);
int                      initialize_video_display(const char *requested_display,
                const char *fmt, unsigned int flags,
                struct display **out);
int                      display_init(display_id_t id, const char *fmt, unsigned int flags, struct display **state);
void                     display_run(struct display *d);
void 	                 display_done(struct display *d);
struct video_frame      *display_get_frame(struct display *d);

/** @brief putf blocking behavior control */
enum display_put_frame_flags {
        PUTF_BLOCKING = 0, ///< Block until frame can be displayed.
        PUTF_NONBLOCK = 1  ///< Do not block.
};

int 		         display_put_frame(struct display *d, struct video_frame *frame, int nonblock);
int                      display_reconfigure(struct display *d, struct video_desc desc);
int                      display_get_property(struct display *d, int property, void *val, size_t *len);
/**
 * @defgroup display_audio Audio
 * Audio related functions (embedded audio).
 * @note
 * This functions will be called from different thread than video functions.
 * @{
 */
void                     display_put_audio_frame(struct display *d, struct audio_frame *frame);
int                      display_reconfigure_audio(struct display *d, int quant_samples, int channels, int sample_rate);
/** @} */ // end of display_audio

#ifdef __cplusplus
}
#endif // __cplusplus

#endif /* _VIDEO_DISPLAY_H */

/** @} */ // end of display

