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
 * Copyright (c) 2019-2021 CESNET, z.s.p.o.
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

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/hresult.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture_params.h"

#define MOD_NAME "[screen win] "

extern const struct video_capture_info vidcap_dshow_info;

struct vidcap_screen_win_state {
        HMODULE screen_cap_lib;
        bool filter_registered;
        void *dshow_state;
};

static void show_help()
{
        printf("Screen capture\n");
        printf("Usage\n");
        color_printf(TERM_BOLD TERM_FG_RED "\t-t screen" TERM_FG_RESET "[:width=<w>][:height=<h>][:fps=<f>]\n" TERM_RESET);
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
        // vt->cards[0].dev can be "" since screen cap. doesn't require parameters
        snprintf(vt->cards[0].name, sizeof vt->cards[0].name, "Screen capture");

        return vt;
}

static bool set_key(const char *key, int val)
{
        HKEY hKey = NULL;
        if (RegCreateKeyEx(HKEY_CURRENT_USER, "Software\\screen-capture-recorder", 0L, NULL, REG_OPTION_NON_VOLATILE, KEY_ALL_ACCESS, NULL, &hKey, NULL ) != ERROR_SUCCESS) {
                // may already exist - try to open it
                if (RegOpenKeyEx(HKEY_CURRENT_USER, "Software\\screen-capture-recorder", 0L, KEY_ALL_ACCESS, &hKey ) != ERROR_SUCCESS) {
                        return false;
                }
        }
        DWORD val_dword = val;
        if (RegSetValueExA(hKey, key, 0L, REG_DWORD, (BYTE *) &val_dword, sizeof val_dword) != ERROR_SUCCESS) {
                return false;
        }

        return true;
}

static bool vidcap_screen_win_process_params(const char *fmt)
{
        if (!fmt || fmt[0] == '\0') {
                return true;
        }
        char *fmt_c = strdup(fmt);
        assert(fmt_c != NULL);

        char *save_ptr;
        char *tmp = fmt_c;
        char *tok;

        while ((tok = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                const char *key;
                char *val_c;
                if (strstr(tok, "width=") != NULL) {
                        key ="capture_width";
                        val_c = tok + strlen("width=");
                } else if (strstr(tok, "height=") != NULL) {
                        key ="capture_height";
                        val_c = tok + strlen("height=");
                } else if (strstr(tok, "fps=") != NULL) {
                        key ="default_max_fps";
                        val_c = tok + strlen("fps=");
                } else {
                        free(fmt_c);
                        return false;
                }
                char *endptr;
                long val = strtol(val_c, &endptr, 0);
                if (*endptr != '\0') {
                        log_msg(LOG_LEVEL_ERROR, "Wrong val: %s\n", val_c);
                        free(fmt_c);
                        return false;
                }
                if (!set_key(key, val)) {
                        log_msg(LOG_LEVEL_ERROR, "Cannot set %s=%ld\n", key, val);
                        free(fmt_c);
                        return false;
                }

                tmp = NULL;
        }

        free(fmt_c);
        return true;
}

typedef HRESULT __stdcall (*func)();

#define CHECK_NOT_NULL_EX(cmd, err_action) do { if ((cmd) == NULL) { log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s\n", #cmd); err_action; } } while(0)
#define CHECK_NOT_NULL(cmd) CHECK_NOT_NULL_EX(cmd, return);
static void cleanup(struct vidcap_screen_win_state *s) {
        assert(s != NULL);

        if (s->dshow_state) {
                vidcap_dshow_info.done(s->dshow_state);
        }

        if (s->filter_registered) {
                func unregister_filter = NULL;
                CHECK_NOT_NULL(unregister_filter = (func)(void *) GetProcAddress(s->screen_cap_lib, "DllUnregisterServer"));
                if (unregister_filter != NULL) {
                        unregister_filter();
                }
        }

        if (s->screen_cap_lib) {
                FreeLibrary(s->screen_cap_lib);
        }
        free(s);
}

#undef CHECK_NOT_NULL
#define CHECK_NOT_NULL(cmd) CHECK_NOT_NULL_EX(cmd, cleanup(s));
static int vidcap_screen_win_init(struct vidcap_params *params, void **state)
{
        const char *cfg = vidcap_params_get_fmt(params);
        if (cfg && strcmp(cfg, "help") == 0) {
                show_help();
                return VIDCAP_INIT_NOERR;
        }

        if (!vidcap_screen_win_process_params(cfg)) {
                show_help();
                return VIDCAP_INIT_FAIL;
        }

        struct vidcap_screen_win_state *s = calloc(1, sizeof *s);

        CHECK_NOT_NULL(s->screen_cap_lib = LoadLibraryA("screen-capture-recorder-x64.dll"));
        func register_filter;
        CHECK_NOT_NULL(register_filter = (func)(void *) GetProcAddress(s->screen_cap_lib, "DllRegisterServer"));
        HRESULT res = register_filter();
        if (FAILED(res)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Register failed: %s\n", hresult_to_str(res));
                cleanup(s);
                return VIDCAP_INIT_FAIL;
        }
        s->filter_registered = true;
        struct vidcap_params *params_dshow = vidcap_params_allocate();
        vidcap_params_set_device(params_dshow, "dshow:device=screen-capture-recorder");
        int ret = vidcap_dshow_info.init(params_dshow, &s->dshow_state);
        if (ret != 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "DirectShow init failed: %d\n", ret);
                cleanup(s);
                return VIDCAP_INIT_FAIL;
        }
        vidcap_params_free_struct(params_dshow);

        *state = s;
        return ret;
}

static void vidcap_screen_win_done(void *state)
{
        cleanup(state);
}

static struct video_frame * vidcap_screen_win_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_screen_win_state *s = state;
        return vidcap_dshow_info.grab(s->dshow_state, audio);
}

static const struct video_capture_info vidcap_screen_win_info = {
        vidcap_screen_win_probe,
        vidcap_screen_win_init,
        vidcap_screen_win_done,
        vidcap_screen_win_grab,
        false
};

REGISTER_MODULE(screen, &vidcap_screen_win_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

