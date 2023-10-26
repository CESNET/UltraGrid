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
 * Copyright (c) 2019-2023 CESNET, z.s.p.o.
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

#include <shellapi.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/text.h"
#include "utils/windows.h"
#include "video.h"
#include "video_capture.h"
#include "video_capture_params.h"

#define MOD_NAME "[screen win] "
#define FILTER_UPSTREAM_URL "https://github.com/rdp/screen-capture-recorder-to-video-windows-free/releases"

extern const struct video_capture_info vidcap_dshow_info;
static bool is_library_registered();
static bool unregister_filter();

struct vidcap_screen_win_state {
        bool filter_registered_here; ///< we were able to register filter as a normal user -> cleanup at end
        void *dshow_state;
};

static void show_help()
{
        char desc[] = "Windows " TBOLD("screen capture") " can be used to capture the whole desktop in Windows.\n\n"
                "It uses dynamically loaded DShow filter to get the screen data. Depending on the system configuration, "
                "it may be required to confirm the filter registration with elevated permissions. In that case the plugin "
                "remains registered permanently (you can unregister it with \"" TBOLD(":unregister") "\" (otherwise it is "
                "unloaded in the end. You can also download and install it from " FILTER_UPSTREAM_URL " to get latest "
                "version.\n\n";
        color_printf("%s", wrap_paragraph(desc));
        color_printf("Usage\n");
        color_printf(
            TBOLD(TRED("\t-t screen") "[:size=<w>x<h>][:fps=<f>]") "\n");
        color_printf(TBOLD("\t-t screen:help") " | " TBOLD(
            "-t screen:unregister") " | " TBOLD("-t screen:clear_prefs") "\n");
        color_printf("where:\n");
        color_printf(TBOLD("\tunregister") " - unregister DShow filter\n");
        color_printf(TBOLD("\tclear_prefs") " - clear screen capture preferences from registry\n");
        color_printf("\t\tIf an option follow, UG will run using new parameters.\n");
        color_printf("\n");
        color_printf(TBOLD("DShow") " filter " TBOLD("%s") " registered\n", is_library_registered() ? "is" : "is not");
}

static void vidcap_screen_win_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *count = 1;
        *deleter = free;
        struct device_info *cards = calloc(1, sizeof(struct device_info));
        // cards[0].dev can be "" since screen cap. doesn't require parameters
        snprintf(cards[0].name, sizeof cards[0].name, "Screen capture");
        *available_cards = cards;
}

#define SCREEN_CAP_REG_TREE "Software\\screen-capture-recorder"
#define REG_FRIENDLY_NAME   "\\HKEY_CURRENT_USER\\" SCREEN_CAP_REG_TREE

static bool
delete_tree()
{
        const LSTATUS err =
            RegDeleteTree(HKEY_CURRENT_USER, SCREEN_CAP_REG_TREE);
        if (err == ERROR_SUCCESS) {
                MSG(NOTICE,
                    "Register tree " REG_FRIENDLY_NAME " deleted succesfully.\n");
                return true;
        }
        MSG(ERROR, "Cannot delete " REG_FRIENDLY_NAME " from registry: %s\n",
            get_win_error(err));
        return false;
}

static LRESULT
set_key(const char *key, int val)
{
        static bool printed;
        if (!printed) {
                MSG(NOTICE, "Options are written permanently to system registry (" REG_FRIENDLY_NAME ").\n"
                        "Use `-t screen:clear_prefs` to erase.\n");
                printed = true;
        }
        HKEY hKey = NULL;
        LRESULT res = RegCreateKeyEx(HKEY_CURRENT_USER, SCREEN_CAP_REG_TREE, 0L,
                                     NULL, REG_OPTION_NON_VOLATILE,
                                     KEY_ALL_ACCESS, NULL, &hKey, NULL);
        if (res != ERROR_SUCCESS) {
                // may already exist - try to open it
                res = RegOpenKeyEx(HKEY_CURRENT_USER, SCREEN_CAP_REG_TREE, 0L,
                                   KEY_ALL_ACCESS, &hKey);
                if (res != ERROR_SUCCESS) {
                        return res;
                }
        }
        DWORD val_dword = val;
        return RegSetValueExA(hKey, key, 0L, REG_DWORD, (BYTE *) &val_dword, sizeof val_dword);
}

static bool
set_key_from_str(const char *key, const char *val_c)
{
        char *endptr;
        long  val = strtol(val_c, &endptr, 0);
        if (*endptr != '\0') {
                log_msg(LOG_LEVEL_ERROR, "Wrong val: %s\n", val_c);
                return false;
        }
        LRESULT res = set_key(key, val);
        if (res == ERROR_SUCCESS) {
                return true;
        }
        MSG(ERROR, "Cannot set %s=%ld: %s\n", key, val, get_win_error(res));
        return false;
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
                tmp = NULL;
                const char *key;
                char *val_c;
                if (IS_KEY_PREFIX(tok, "size") && strchr(tok, 'x') != NULL) {
                        char *width = strchr(tok, '=') + 1;
                        char *height = strchr(tok, 'x') + 1;
                        *strchr(tok, 'x') = '\0';
                        if (!set_key_from_str("capture_width", width) ||
                            !set_key_from_str("capture_height", height)) {
                                free(fmt_c);
                                return false;
                        }
                        continue;
                } else if (strstr(tok, "width=") != NULL) {
                        MSG(WARNING, "Deprecated width=, use size=\n");
                        key ="capture_width";
                        val_c = tok + strlen("width=");
                } else if (strstr(tok, "height=") != NULL) {
                        MSG(WARNING, "Deprecated height=, use size=\n");
                        key ="capture_height";
                        val_c = tok + strlen("height=");
                } else if (strstr(tok, "fps=") != NULL) {
                        key ="default_max_fps";
                        val_c = tok + strlen("fps=");
                } else {
                        MSG(ERROR, "Unknown parameter: %s\n", tok);
                        free(fmt_c);
                        return false;
                }
                if (!set_key_from_str(key, val_c)) {
                        free(fmt_c);
                        return false;
                }
        }

        free(fmt_c);
        return true;
}

typedef HRESULT __stdcall (*func)();

#define CHECK_NOT_NULL(cmd, err_action) do { if ((cmd) == NULL) { log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s\n", #cmd); err_action; } } while(0)
static void cleanup(struct vidcap_screen_win_state *s) {
        assert(s != NULL);

        if (s->dshow_state) {
                vidcap_dshow_info.done(s->dshow_state);
        }

        if (s->filter_registered_here) {
                unregister_filter();
        }

        free(s);
}

static bool is_library_registered() {
        void (*deleter)(void *) = NULL;
		struct device_info *cards = NULL;
		int count = 0;
        vidcap_dshow_info.probe(&cards, &count, &deleter);

        bool ret = false;
        for (int i = 0; i < count; ++i) {
                if (strcmp(cards[i].name, "screen-capture-recorder") == 0) {
                        ret = true;
                        break;
                }
        }
        deleter = IF_NOT_NULL_ELSE(deleter, free);
        deleter(cards);
        return ret;
}

static int register_screen_cap_rec_library(bool is_elevated) {
        HMODULE screen_cap_lib = NULL;
        CHECK_NOT_NULL(screen_cap_lib = LoadLibraryA("screen-capture-recorder-x64.dll"), return -1);
        func register_filter;
        CHECK_NOT_NULL(register_filter = (func)(void *) GetProcAddress(screen_cap_lib, "DllRegisterServer"), FreeLibrary(screen_cap_lib); return -1);
        HRESULT res = register_filter();
        FreeLibrary(screen_cap_lib);
        if (SUCCEEDED(res)) {
                return 0;
        }
        FreeLibrary(screen_cap_lib);
        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Register failed: %s\n", hresult_to_str(res));
        if (res == E_ACCESSDENIED) {
                if (!is_elevated) {
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Need to register DirectShow module for screen capture, confirm if you agree.\n");
                        HINSTANCE ret = ShellExecute( NULL,
                                        "runas",
                                        uv_argv[0], " -t screen:register_elevated",
                                        NULL,                        // default dir
                                        SW_SHOWNORMAL
                                        );
                        if ((INT_PTR) ret > 32) {
                                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Module installation successful.\n");
                                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "If you want to unregister the module, run 'uv -t screen:unregister'.\n");
                                sleep(2);
                                return 1;
                        }
                }
                log_msg(LOG_LEVEL_WARNING, "Cannot register DLL (access denied), please allow access or install the filter from:\n"
                                "  " FILTER_UPSTREAM_URL "\n");
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Register failed: %s\n", hresult_to_str(res));
        }
        return -1;
}

static bool
unregister_filter()
{
        HMODULE screen_cap_lib = NULL;
        CHECK_NOT_NULL(screen_cap_lib =
                           LoadLibraryA("screen-capture-recorder-x64.dll"),
                       return false);
        func unregister_filter;
        CHECK_NOT_NULL(unregister_filter = (func) (void *) GetProcAddress(
                           screen_cap_lib, "DllUnregisterServer"),
                       FreeLibrary(screen_cap_lib);
                       return false);
        HRESULT res = unregister_filter();
        if (SUCCEEDED(res)) {
                MSG(NOTICE, "Filter unregistered successfully\n");
                return true;
        }
        MSG(ERROR, "Cannot unregister filter, ret = %ld (%s)\n",
            res, hresult_to_str(res));
        return false;
}

static int load_screen_cap_rec_library(struct vidcap_screen_win_state *s) {
        if (is_library_registered()) {
                log_msg(LOG_LEVEL_VERBOSE, "Using already system-registered screen-capture-recorder library.\n");
                return 0;
        }

        int rc = 0;
        if ((rc = register_screen_cap_rec_library(false)) != 0) {
                return rc;
        }
        s->filter_registered_here = true;
        return 0;
}

static int run_child_process() {
        char cmd[1024] = "";
        for (int i = 0; i < uv_argc; ++i) {
                char tmp[1024];
                if (strstr(uv_argv[i], "screen") == uv_argv[i]) {
                        if (strlen(uv_argv[i]) >= 7) { // screen:
                                snprintf(tmp, sizeof tmp, "\"screen:child:%s\" ", uv_argv[i] +  6);
                        } else
                                snprintf(tmp, sizeof tmp, "\"screen:child\" ");
                } else {
                        snprintf(tmp, sizeof tmp, "\"%s\" ", uv_argv[i]);
                }
                strncat(cmd, tmp, sizeof cmd - strlen(cmd) - 1);
        }
        if (strlen(cmd) == sizeof cmd - 1) {
                return -1; // potential overflow
        }
        STARTUPINFO si = { 0 };
        PROCESS_INFORMATION pi = { 0 };
        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Starting child process:\n%s\n", cmd);
        const bool ret = CreateProcess(NULL, cmd, NULL, NULL, FALSE, 0, NULL,
                                       NULL, &si, &pi);
        if (ret) {
                WaitForSingleObject(pi.hProcess, INFINITE);
        }
        return ret;
}

static int vidcap_screen_win_init(struct vidcap_params *params, void **state)
{
        const char *cfg = vidcap_params_get_fmt(params);
        bool child = false; // to prevent fork bombs if error
        if (strcmp(cfg, "help") == 0) {
                show_help();
                return VIDCAP_INIT_NOERR;
        }
        if (strcmp(cfg, "register_elevated") == 0) {
                int rc = register_screen_cap_rec_library(true);
                system("pause");
                return rc >= 0 ? VIDCAP_INIT_NOERR : VIDCAP_INIT_FAIL;
        }
        if (strcmp(cfg, "unregister") == 0) {
                HINSTANCE inst = ShellExecute(NULL, "runas", uv_argv[0],
                                              " -t screen:unregister_elevated",
                                              NULL, // default dir
                                              SW_SHOWNORMAL);
                INT_PTR ret = (INT_PTR) inst;
                if (ret > 32) {
                        return VIDCAP_INIT_NOERR;
                }
                MSG(ERROR, "Elevated subshell exec failed, res = %lld (%s)\n",
                    ret, get_win_error(ret));
                return VIDCAP_INIT_FAIL;
        }
        if (strcmp(cfg, "unregister_elevated") == 0) {
                bool res = unregister_filter();
                system("pause");
                return res ? VIDCAP_INIT_NOERR : VIDCAP_INIT_FAIL;
        }
        if (strstr(cfg, "clear_prefs") == cfg) {
                const bool ret = delete_tree();
                cfg += strlen("clear_prefs");
                if (strlen(cfg) == 0) {
                        return ret ? VIDCAP_INIT_NOERR : VIDCAP_INIT_FAIL;
                }
        }
        if (strstr(cfg, "child") == cfg) {
                child = true;
                cfg = strchr(cfg, ':') ? strchr(cfg, ':') + 1 : "";
        }

        if (!vidcap_screen_win_process_params(cfg)) {
                show_help();
                return VIDCAP_INIT_FAIL;
        }

        struct vidcap_screen_win_state *s = calloc(1, sizeof *s);
        int rc = load_screen_cap_rec_library(s);
        if (rc != 0) {
                cleanup(s);
                if (rc < 0) {
                       return VIDCAP_INIT_FAIL;
                }
                return child ? VIDCAP_INIT_FAIL : run_child_process();
        }

        struct vidcap_params *params_dshow = vidcap_params_allocate();
        vidcap_params_set_device(params_dshow, "dshow:device=screen-capture-recorder");
        vidcap_params_set_parent(params_dshow, vidcap_params_get_parent(params));
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
        MOD_NAME,
};

REGISTER_MODULE(screen, &vidcap_screen_win_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

