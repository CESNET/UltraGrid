/**
 * @file   video_capture.cpp
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
/*
 * Copyright (c) 2005-2023 CESNET, z. s. p. o.
 * Copyright (c) 2001-2004 University of Southern California
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
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "module.h"
#include "video_capture.h"
#include "video_capture_params.h"

#include <string>
#include <iomanip>

using namespace std;

#define VIDCAP_MAGIC	0x76ae98f0

/// @brief This struct represents video capture state.
struct vidcap {
        struct module mod;
        void    *state; ///< state of the created video capture driver
        const struct video_capture_info *funcs;
        uint32_t magic; ///< For debugging. Conatins @ref VIDCAP_MAGIC

        struct capture_filter *capture_filter; ///< capture_filter_state
};

/* API for probing capture devices ****************************************************************/
void list_video_capture_devices(bool full)
{
        printf("Available capture devices:\n");
        list_modules(LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION, full);
}

/** @brief Initializes video capture
 * @param[in] parent  parent module
 * @param[in] param   driver parameters
 * @param[out] state returned state
 * @retval 0    if initialization was successful
 * @retval <0   if initialization failed
 * @retval >0   if initialization was successful but no state was returned (eg. only having shown help).
 */
int initialize_video_capture(struct module *parent,
                struct vidcap_params *param,
                struct vidcap **state)
{
        /// check appropriate cmdline parameters order (--capture-filter then -t)
        /// only if one capture device specified, allow setting -F after -t
        struct vidcap_params *tlast = param;
        struct vidcap_params *tprev = param;
        while ((vidcap_params_get_next(tlast))) {
                tprev = tlast;
                tlast = vidcap_params_get_next(tlast);
        }
        if (vidcap_params_get_driver(tlast) == NULL
                        && vidcap_params_get_capture_filter(tlast) != NULL) {
                if (tprev != param) { // more than one -t
                        log_msg(LOG_LEVEL_ERROR, "Capture filter (--capture-filter) needs to be "
                                "specified before capture (-t)\n");
                        return -1;
                }
                if (vidcap_params_get_capture_filter(tprev) != NULL) { // one -t but -F specified both before and after it
                        log_msg(LOG_LEVEL_ERROR, "Multiple capture filter specification.\n");
                        return -1;
                }
                vidcap_params_set_capture_filter(tprev, vidcap_params_get_capture_filter(tlast));
        }
        // similarly for audio connection
        if (vidcap_params_get_driver(tlast) == nullptr
                        && vidcap_params_get_flags(tlast) != 0) {
                if (tprev != param) { // more than one -t
                        log_msg(LOG_LEVEL_ERROR, "Audio connection (-s) needs to be "
                                "specified before capture (-t)\n");
                        return -1;
                }
                if (vidcap_params_get_flags(tprev) != 0) { // one -t but -s specified both before and after it
                        log_msg(LOG_LEVEL_ERROR, "Multiple audio connection specification.\n");
                        return -1;
                }
                vidcap_params_set_flags(tprev, vidcap_params_get_flags(tlast));
        }

        const struct video_capture_info *vci = (const struct video_capture_info *)
                load_library(vidcap_params_get_driver(param), LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

        if (vci == nullptr) {
                log_msg(LOG_LEVEL_ERROR, "WARNING: Selected '%s' capture card "
                        "was not found.\n", vidcap_params_get_driver(param));
                return -1;
        }

        struct vidcap *d =
                (struct vidcap *)malloc(sizeof(struct vidcap));
        d->magic = VIDCAP_MAGIC;
        d->funcs = vci;

        module_init_default(&d->mod);
        d->mod.cls = MODULE_CLASS_CAPTURE;
        module_register(&d->mod, parent);

        vidcap_params_set_parent(param, &d->mod);
        int ret = vci->init(param, &d->state);

        switch (ret) {
        case VIDCAP_INIT_OK:
                break;
        case VIDCAP_INIT_NOERR:
                break;
        case VIDCAP_INIT_FAIL:
                log_msg(LOG_LEVEL_ERROR,
                                "Unable to start video capture device %s\n",
                                vidcap_params_get_driver(param));
                break;
        case VIDCAP_INIT_AUDIO_NOT_SUPPOTED:
                log_msg(LOG_LEVEL_ERROR,
                                "Video capture driver does not support selected embedded/analog/AESEBU audio.\n");
                break;
        }
        if (ret != 0) {
                module_done(&d->mod);
                free(d);
                return ret;
        }

        ret = capture_filter_init(&d->mod, vidcap_params_get_capture_filter(param),
                &d->capture_filter);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize capture filter: %s.\n",
                        vidcap_params_get_capture_filter(param));
        }

        if (ret != 0) {
                module_done(&d->mod);
                free(d);
                return ret;
        }

        *state = d;
        return 0;
}

/** @brief Destroys vidap state
 * @param state state to be destroyed (must have been successfully initialized with vidcap_init()) */
void vidcap_done(struct vidcap *state)
{
        assert(state->magic == VIDCAP_MAGIC);
        state->funcs->done(state->state);
        capture_filter_destroy(state->capture_filter);
        module_done(&state->mod);
        free(state);
}

/** @brief Grabs video frame.
 * This function may block for a short period if waiting for incoming frame. This period, however,
 * should not be longer than few frame times, a second at maximum.
 *
 * The decision of blocking behavior is on the vidcap driver.
 *
 * The returned video frame is valid only until next vidcap_grab() call.
 *
 * @param[in]  state vidcap state
 * @param[out] audio contains audio frame if driver is grabbing audio
 * @returns video frame. If no frame was grabbed (or timeout passed) NULL is returned.
 */
struct video_frame *vidcap_grab(struct vidcap *state, struct audio_frame **audio)
{
        assert(state->magic == VIDCAP_MAGIC);
        struct video_frame *frame;
        frame = state->funcs->grab(state->state, audio);
        if (frame != NULL)
                frame = capture_filter(state->capture_filter, frame);
        return frame;
}

/**
 * @brief If not-NULL returned, display doesn't hae own FPS indicator and wants
 * to use a generic one (prefixed with returned module name)
 */
const char *vidcap_get_fps_print_prefix(struct vidcap *state)
{
        assert(state->magic == VIDCAP_MAGIC);
        return state->funcs->generic_fps_indicator_prefix;
}

