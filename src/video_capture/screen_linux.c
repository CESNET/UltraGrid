/**
 * @file   video_capture/screen_linux.c
 * @author Martin Pulec <pulec@cesnet.cz>
 *
 * X11/PipeWire screen capture abstraction
 */
/*
 * Copyright (c) 2023-2025 CESNET
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
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
 */

#include <stdio.h>                 // for printf
#include <stdlib.h>
#include <string.h>

#include "config.h"                // for HAVE_*
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/text.h"
#include "video_capture.h"
#include "video_capture_params.h"  // for vidcap_params_free_struct, vidcap_...

struct audio_frame;
struct device_info;
struct vidcap_params;


static void vidcap_screen_linux_probe(struct device_info **cards, int *count, void (**deleter)(void *))
{
        *cards = NULL;
        *count = 0;
        *deleter = free;
}

static int
vidcap_screen_linux_init(struct vidcap_params *params, void **state)
{
        struct vidcap *device = NULL;

        if (strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                char desc[] = TRED(TBOLD("screen")) " capture in Linux abstracts X11/PipeWire screen capture and tries to select "
                        "the right implementation. You can also specify directly " TBOLD("screen_x11") " or " TBOLD("screen_pw") ".";
                color_printf("%s\n\n", wrap_paragraph(desc));
                color_printf("Compiled modules:");
#ifdef HAVE_SCREEN_X11
                color_printf(TBOLD(" X11"));
#endif
#ifdef HAVE_SCREEN_PW
                color_printf(TBOLD(" pipewire"));
#endif
                printf("\n\nFollows help for the module that would have been selected:\n\n");
        }

        struct vidcap_params *params_new = vidcap_params_copy(params);
        if (getenv("WAYLAND_DISPLAY") != NULL) {
                vidcap_params_set_driver(params_new, "screen_pw");
                verbose_msg("Trying to initialize screen_pw\n");
                int ret = initialize_video_capture(NULL, params_new, &device);
                if (ret < 0) {
                        error_msg("screen_pw initialization failed\n");
                } else {
                        vidcap_params_free_struct(params_new);
                        *state = device;
                        return ret;
                }
        }
        verbose_msg("Trying to initialize screen_x11\n");
        if (getenv("DISPLAY") == NULL) {
                log_msg(LOG_LEVEL_WARNING, "Trying to initialize screen_x11 but DISPLAY environment variable is not set!\n");
        }
        vidcap_params_set_driver(params_new, "screen_x11");
        int ret = initialize_video_capture(NULL, params_new, &device);
        vidcap_params_free_struct(params_new);

        *state = device;
        return ret;
}

static void
vidcap_screen_linux_done(void *state)
{
        vidcap_done((struct vidcap *) state);
}

static struct video_frame *
vidcap_screen_linux_grab(void *state, struct audio_frame **audio)
{
        return vidcap_grab((struct vidcap *) state, audio);
}

static const struct video_capture_info vidcap_screen_linux_info = {
        vidcap_screen_linux_probe,
        vidcap_screen_linux_init,
        vidcap_screen_linux_done,
        vidcap_screen_linux_grab,
        "[screen] ",
};

REGISTER_MODULE(screen, &vidcap_screen_linux_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

