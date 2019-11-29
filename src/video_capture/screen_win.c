/**
 * @file   video_capture/screen_win.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * Screen pseudo capturer. Uses DirectShow with screen-capturer-recorder filter.
 *
 * @todo
 * - add more formats
 * - load the dll even if working directory is not the dir with the DLL
 */
/*
 * Copyright (c) 2019 CESNET, z.s.p.o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <stdio.h>
#include <stdlib.h>

#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture_params.h"

extern const struct video_capture_info vidcap_dshow_info;

static void show_help()
{
        printf("Screen capture\n");
        printf("Usage\n");
        color_out(COLOR_OUT_BOLD | COLOR_OUT_RED, "\t-t screen\n");
}


static struct vidcap_type * vidcap_screen_win_probe(bool verbose, void (**deleter)(void *))
{
        struct vidcap_type*		vt;
        *deleter = free;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt == NULL) {
                return NULL;
        }

        vt->name        = "screen";
        vt->description = "Grabbing screen";

        if (!verbose) {
                return vt;
        }

        vt->card_count = 1;
        vt->cards = calloc(vt->card_count, sizeof(struct device_info));
        // vt->cards[0].id can be "" since screen cap. doesn't require parameters
        snprintf(vt->cards[0].id, sizeof vt->cards[0].id, "screen");
        snprintf(vt->cards[0].name, sizeof vt->cards[0].name, "Screen capture");

        return vt;
}

#define CHECK_NOT_NULL_EX(cmd, err_action) do { if ((cmd) == NULL) { log_msg(LOG_LEVEL_ERROR, "[screen] %s\n", #cmd); err_action; } } while(0)
#define CHECK_NOT_NULL(cmd) CHECK_NOT_NULL_EX(cmd, return VIDCAP_INIT_FAIL);
static int vidcap_screen_win_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_fmt(params) &&
                        strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                show_help();
                return VIDCAP_INIT_NOERR;
        }

        HMODULE mod;
        CHECK_NOT_NULL(mod = LoadLibraryA("screen-capture-recorder-x64.dll"));
        typedef void (*func)();
        func register_filter;
        CHECK_NOT_NULL(register_filter = (func) GetProcAddress(mod, "DllRegisterServer"));
        register_filter();
        FreeLibrary(mod);
        struct vidcap_params *params_dshow = vidcap_params_allocate();
        vidcap_params_set_device(params_dshow, "dshow:device=screen-capture-recorder");
        int ret = vidcap_dshow_info.init(params_dshow, state);
        vidcap_params_free_struct(params_dshow);

        return ret;
}

#undef CHECK_NOT_NULL
#define CHECK_NOT_NULL(cmd) CHECK_NOT_NULL_EX(cmd, return);
static void vidcap_screen_win_done(void *state)
{
        vidcap_dshow_info.done(state);

        HMODULE mod;
        CHECK_NOT_NULL(mod = LoadLibraryA("screen-capture-recorder-x64.dll"));
        typedef void (*func)();
        func unregister_filter;
        CHECK_NOT_NULL(unregister_filter = (func) GetProcAddress(mod, "DllUnregisterServer"));
        unregister_filter();
        FreeLibrary(mod);
}

static struct video_frame * vidcap_screen_win_grab(void *state, struct audio_frame **audio)
{
        return vidcap_dshow_info.grab(state, audio);
}

static const struct video_capture_info vidcap_screen_win_info = {
        vidcap_screen_win_probe,
        vidcap_screen_win_init,
        vidcap_screen_win_done,
        vidcap_screen_win_grab,
        false
};

REGISTER_MODULE(screen, &vidcap_screen_win_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

