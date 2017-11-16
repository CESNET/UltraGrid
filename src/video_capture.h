/**
 * @file   video_capture.h
 * @author Colin Perkins <csp@csperkins.org>
 * @author Martin Benes     <martinbenesh@gmail.com>
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Petr Holub       <hopet@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Jiri Matela      <matela@ics.muni.cz>
 * @author Dalibor Matura   <255899@mail.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * @ingroup vidcap
 */
/**
 * Copyright (c) 2005-2013 CESNET z.s.p.o
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

/**
 * @defgroup vidcap Video Capture
 *
 * API for video capture. Normal operation is something like:
 * @code{.c}
 * v = vidcap_init(id);
 * ...
 * while (!done) {
 *     ...
 *     f = vidcap_grab(v, timeout);
 *     ...use the frame "f"
 * }
 * vidcap_done(v);
 * @endcode
 *
 * Where the "id" parameter to vidcap_init() is obtained from
 * the probing API. The vidcap_grab() function returns a pointer
 * to the frame, or NULL if no frame is currently available. It
 * does not block.
 *
 * @note
 * The vidcap_grab() API is currently slightly different - the function does
 * not take the timeout parameter and may block, but only for a short period
 * (ideally no longer than 2x frame time)
 *
 * @{
 */

#ifndef _VIDEO_CAPTURE_H_
#define _VIDEO_CAPTURE_H_

#include "types.h"

#define VIDEO_CAPTURE_ABI_VERSION 5

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct audio_frame;

/** Defines video capture device */
struct vidcap_type {
        const char              *name;        ///< short name (one word)
        const char              *description; ///< description of the video device

        int                      card_count;
        struct device_info      *cards;
};

struct vidcap_params;

#define VIDCAP_INIT_OK 0
#define VIDCAP_INIT_NOERR 1
#define VIDCAP_INIT_FAIL -1
#define VIDCAP_INIT_AUDIO_NOT_SUPPOTED -2

struct video_capture_info {
        struct vidcap_type    *(*probe) (bool verbose);
        /**
         * @param[in] driver configuration string
         * @param[in] param  driver parameters
         * @retval NULL if initialization failed
         * @retval &vidcap_init_noerr if initialization succeeded but a state was not returned (eg. help)
         * @retval other_ptr if initialization succeeded, contains pointer to state
         */
        int (*init) (const struct vidcap_params *param, void **state);
        void                   (*done) (void *state);
        struct video_frame    *(*grab) (void *state, struct audio_frame **audio);
};

struct module;
struct vidcap;

void                     list_video_capture_devices(void);
void                     print_available_capturers(void);
int initialize_video_capture(struct module *parent,
                struct vidcap_params *params,
                struct vidcap **state);
void			 vidcap_done(struct vidcap *state);
struct video_frame	*vidcap_grab(struct vidcap *state, struct audio_frame **audio);

#ifdef __cplusplus
}
#endif // __cplusplus

#include "video_capture_params.h"

#endif // _VIDEO_CAPTURE_H_
/**
 * @}
 */

