/**
 * @file   video_capture.c
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
 * Copyright (c) 2005-2013 CESNET z.s.p.o.
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
#include "utils/config_file.h"
#include "video_capture.h"

#include <string>

using namespace std;

#define VIDCAP_MAGIC	0x76ae98f0

/**This variable represents a pseudostate and may be returned when initialization
 * of module was successful but no state was created (eg. when driver had displayed help).
 */
int vidcap_init_noerr;

struct vidcap_params;

/**
 * Defines parameters passed to video capture driver.
  */
struct vidcap_params {
        char  *driver; ///< driver name
        char  *fmt;    ///< driver options
        unsigned int flags;  ///< one of @ref vidcap_flags

        char *requested_capture_filter;
        char  *name;   ///< input name (capture alias in config file or complete config if not alias)
        struct vidcap_params *next; /**< Pointer to next vidcap params. Used by aggregate capture drivers.
                                     *   Last device in list has @ref driver set to NULL. */
        struct module *parent;
};

/// @brief This struct represents video capture state.
struct vidcap {
        struct module mod;
        void    *state; ///< state of the created video capture driver
        const struct video_capture_info *funcs;
        uint32_t magic; ///< For debugging. Conatins @ref VIDCAP_MAGIC

        struct capture_filter *capture_filter; ///< capture_filter_state
};

/* API for probing capture devices ****************************************************************/
void list_video_capture_devices()
{
        printf("Available capture devices:\n");
        list_modules(LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
}

void print_available_capturers()
{
        const auto & vidcaps = get_libraries_for_class(LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);
        for (auto && item : vidcaps) {
                auto vci = static_cast<const struct video_capture_info *>(item.second);

                struct vidcap_type *vt = vci->probe(true);
                for (int i = 0; i < vt->card_count; ++i) {
                        printf("[capability][capture][v1] {\"device\": \"%s%s%s\", \"name\": \"%s\", \"modes\": {", vt->name, strlen(vt->cards[i].id) > 0 ? ":" : "", vt->cards[i].id, vt->cards[i].name);
                        for (unsigned int j = 0; j < sizeof vt->cards[i].modes
                                        / sizeof vt->cards[i].modes[0]; j++) {
                                if (vt->cards[i].modes[j].id[0] == '\0') { // last item
                                        break;
                                }
                                if (j > 0) {
                                        printf(", ");
                                }
                                printf("\"%s\": \"%s\"", vt->cards[i].modes[j].id,
                                                vt->cards[i].modes[j].name);
                        }

                        printf("}}\n");
                }

        }

        char buf[1024] = "";
        struct config_file *conf = config_file_open(default_config_file(buf, sizeof buf));
        if (conf) {
                auto const & from_config_file = get_configured_capture_aliases(conf);
                for (auto const & it : from_config_file) {
                        printf("[cap] (%s;%s)\n", it.first.c_str(), it.second.c_str());
                }
        }
        config_file_close(conf);
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
        /// check appropriate cmdline parameters order (--capture-filter and -t)
        struct vidcap_params *t, *t0;
        t = t0 = param;
        while ((t = vidcap_params_get_next(t0))) {
                t0 = t;
        }
        if (t0->driver == NULL && t0->requested_capture_filter != NULL) {
                log_msg(LOG_LEVEL_ERROR, "Capture filter (--capture-filter) needs to be "
                                "specified before capture (-t)\n");
                return -1;
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

        param->parent = &d->mod;
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

        ret = capture_filter_init(&d->mod, param->requested_capture_filter,
                &d->capture_filter);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize capture filter: %s.\n",
                        param->requested_capture_filter);
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

