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
 * @ingroup vidcap
 */
/*
 * Copyright (c) 2005-2023 CESNET, z. s. p. o
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
 */

/**
 * @defgroup vidcap Video Capture
 *
 * ## Workflow for video capture
 *
 * Normal operation is something like:
 * @code{.c}
 * v = vidcap_init(id);
 * ...
 * while (!done) {
 *     ...
 *     f = vidcap_grab(v, &a);
 *     ...use the video frame "f"
 *     ...use the audio frame "a"
 * }
 * vidcap_done(v);
 * @endcode
 *
 * Where the "id" parameter to vidcap_init() is obtained from
 * the probing API. The vidcap_grab() function returns a pointer
 * to the frame, or NULL if no frame is currently available.
 *
 * The vidcap_grab() may block, but not indefinitely (ideally
 * no longer than 2x frame time) or it should observe global exit
 * status with register_should_exit_callback() and yield control
 * when notified.
 *
 * ## API for video capturers
 * Each module should implement API from @ref video_capture_info.
 * Furthermore, it should register the module with REGISTER_MODULE(). E.g.:
 *
 *     REGISTER_MODULE(dvs, &vidcap_dvs_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
 *
 * @note
 * The lifetime of captured frame is determined by the fact whether
 * video_frame_callbacks::dispose is set. If so, the caller calls the callback
 * when the frame is no longer needed. When set to NULL, the lifetime of the frame
 * is limited to a next call to vidcap_grab().
 *
 * @{
 */

#ifndef _VIDEO_CAPTURE_H_
#define _VIDEO_CAPTURE_H_

#include "types.h"
#include "video_capture_params.h"

#define VIDEO_CAPTURE_ABI_VERSION 12

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct audio_frame;

/// @name vidcap_retval
/// @anchor vidcap_retval
/// Video capture init return values
/// @{
#define VIDCAP_INIT_OK                   0  ///< initialization successful
#define VIDCAP_INIT_NOERR                1  ///< state not initialized, other action performed (typically help)
#define VIDCAP_INIT_FAIL               (-1) ///< error ocured
#define VIDCAP_INIT_AUDIO_NOT_SUPPORTED (-2) ///< card does not support audio
/// @}

#define VIDCAP_NO_GENERIC_FPS_INDICATOR NULL

/**
 * API for video capture modules
 */
struct video_capture_info {
        device_probe_func probe;
        /**
         * @param[in]  param  driver parameters
         * @param[out] state  returned capture state
         * @returns           one of @ref vidcap_retval
         */
        int                    (*init) (struct vidcap_params *param, void **state);
        void                   (*done) (void *state);
        struct video_frame    *(*grab) (void *state, struct audio_frame **audio);
        const char             *generic_fps_indicator_prefix; ///@todo use everywhere, then remove
};

struct module;
struct vidcap;

void                     list_video_capture_devices(bool);
int initialize_video_capture(struct module *parent,
                struct vidcap_params *params,
                struct vidcap **state);
void			 vidcap_done(struct vidcap *state);
struct video_frame	*vidcap_grab(struct vidcap *state, struct audio_frame **audio);
const char              *vidcap_get_fps_print_prefix(struct vidcap *state);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // _VIDEO_CAPTURE_H_
/**
 * @}
 */

