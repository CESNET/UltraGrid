/**
 * @file   video_capture/ximea.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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
#endif // HAVE_CONFIG_H
#include "config_unix.h"
#include "config_win32.h"

#if defined(__APPLE__)
#include <m3api/xiApi.h>
#else
#include <xiApi.h>
#endif

#ifndef WIN32
#include <dlfcn.h>
#endif

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "video.h"
#include "video_capture.h"

#define DEFAULT_TIMEOUT_MS 100
#define EXPOSURE_DEFAULT_US 10000
#define MAGIC to_fourcc('X', 'I', 'M', 'E')
#define MOD_NAME "[XIMEA] "
#define MICROSEC_IN_SEC 1000000.0

struct ximea_functions {
        XI_RETURN __cdecl (*xiGetNumberDevices)(OUT PDWORD pNumberDevices);
        XI_RETURN __cdecl (*xiGetDeviceInfoString)(IN DWORD DevId, const char* prm, char* value, DWORD value_size);
        XI_RETURN __cdecl (*xiOpenDevice)(IN DWORD DevId, OUT PHANDLE hDevice);
        XI_RETURN __cdecl (*xiSetParamInt)(IN HANDLE hDevice, const char* prm, const int val);
        XI_RETURN __cdecl (*xiStartAcquisition)(IN HANDLE hDevice);
        XI_RETURN __cdecl (*xiGetImage)(IN HANDLE hDevice, IN DWORD timeout, OUT LPXI_IMG img);
        XI_RETURN __cdecl (*xiStopAcquisition)(IN HANDLE hDevice);
        XI_RETURN __cdecl (*xiCloseDevice)(IN HANDLE hDevice);
        LIB_HANDLE handle;
};

struct state_vidcap_ximea {
        uint32_t magic;
        long device_id;
        long exposure_time_us;

        HANDLE xiH;
	struct ximea_functions funcs;
};

#define GET_SYMBOL(sym) do {\
        f->sym = (void *) dlsym(f->handle, #sym);\
        if (f->sym == NULL) {\
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to find symbol %s: %s\n", #sym, dlerror());\
                return false;\
        }\
} while(0)
static bool vidcap_ximea_fill_symbols(struct ximea_functions *f) {
#ifdef XIMEA_RUNTIME_LINKING
        f->handle = dlopen(XIMEA_LIBRARY_NAME, RTLD_NOW);
        if (!f->handle) {
#ifdef WIN32
                char errstr[1024] = "";
                FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(),
                                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                (LPTSTR) &errstr, sizeof errstr, NULL );
#else
                const char *errstr = dlerror();
#endif
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Unable to open library %s): %s!\n",
                                XIMEA_LIBRARY_NAME, errstr);
                char path[strlen(XIMEA_LIBRARY_PATH) + 1 + strlen(XIMEA_LIBRARY_NAME) + 1];
                strcpy(path, XIMEA_LIBRARY_PATH);
                strcat(path, "/");
                strcat(path, XIMEA_LIBRARY_NAME);
                f->handle = dlopen(path, RTLD_NOW);
        }
        if (!f->handle) {
#ifdef WIN32
                char errstr[1024] = "";
                FormatMessage(FORMAT_MESSAGE_FROM_SYSTEM, NULL, GetLastError(),
                                MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
                                (LPTSTR) &errstr, sizeof errstr, NULL );
#else
                const char *errstr = dlerror();
#endif
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to open library (name: %s, path: %s): %s!\n",
                                XIMEA_LIBRARY_NAME, XIMEA_LIBRARY_PATH, errstr);
                return false;
        }
        GET_SYMBOL(xiGetNumberDevices);
        GET_SYMBOL(xiGetDeviceInfoString);
        GET_SYMBOL(xiOpenDevice);
        GET_SYMBOL(xiSetParamInt);
        GET_SYMBOL(xiStartAcquisition);
        GET_SYMBOL(xiGetImage);
        GET_SYMBOL(xiStopAcquisition);
        GET_SYMBOL(xiCloseDevice);
#else
	f->xiGetNumberDevices = xiGetNumberDevices;
	f->xiGetDeviceInfoString = xiGetDeviceInfoString;
	f->xiOpenDevice = xiOpenDevice;
	f->xiSetParamInt = xiSetParamInt;
	f->xiStartAcquisition = xiStartAcquisition;
	f->xiGetImage = xiGetImage;
	f->xiStopAcquisition = xiStopAcquisition;
	f->xiCloseDevice = xiCloseDevice;
#endif

        return true;
}

static void vidcap_ximea_close_lib(struct ximea_functions *f)
{
        if (f->handle == NULL) {
                return;
        }
#ifdef XIMEA_RUNTIME_LINKING
        dlclose(f->handle);
#endif
}


static void vidcap_ximea_show_help() {
        color_out(0, "XIMEA usage:\n");
        color_out(COLOR_OUT_RED | COLOR_OUT_BOLD, "\t-t ximea");
        color_out(COLOR_OUT_BOLD, "[:device=<d>][:exposure=<time_us>]\n");
        printf("\n");
        printf("Available devices:\n");

        struct ximea_functions funcs;
        if (!vidcap_ximea_fill_symbols(&funcs)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot load XIMEA library!\n");
                return;
        }

        DWORD count;
        XI_RETURN ret = funcs.xiGetNumberDevices(&count);
        if (ret != XI_OK) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to get device count!\n");
                return;
        }

        if (count == 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "No devices found!\n");
                return;
        }

        for (DWORD i = 0; i < count; ++i) {
                char name[256];
                color_out(COLOR_OUT_BOLD, "\t%d) ", (int) i);
                if (funcs.xiGetDeviceInfoString(i, XI_PRM_DEVICE_NAME, name, sizeof name) == XI_OK) {
                        color_out(0, "%s", name);
                }
                color_out(0, "\n");
        }
        vidcap_ximea_close_lib(&funcs);
}

static int vidcap_ximea_parse_params(struct state_vidcap_ximea *s, const char *cfg) {
        if (cfg == NULL || cfg[0] == '\0') {
                return 0;
        }

        char *fmt = strdup(cfg);
        assert(fmt != NULL);
        char *save_ptr;
        char *tmp = fmt;
        char *tok;
        while ((tok = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                if (!strcmp(tok, "help")) {
                        vidcap_ximea_show_help();
                        free(fmt);
                        return VIDCAP_INIT_NOERR;
                }
                if (strstr(tok, "device=")) {
                        char *endptr = NULL;
                        s->device_id = strtol(tok + strlen("device="), &endptr, 0);
                        if (*endptr != '\0') {
                                goto error;
                        }
                } else if (strstr(tok, "exposure=")) {
                        char *endptr = NULL;
                        s->exposure_time_us = strtol(tok + strlen("exposure="), &endptr, 0);
                        if (*endptr != '\0' || s->exposure_time_us < 0) {
                                goto error;
                        }
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s\n", tok);
                        free(fmt);
                        return VIDCAP_INIT_FAIL;
                }
                tmp = NULL;
        }

        free(fmt);
        return 0;

error:
        free(fmt);
        return VIDCAP_INIT_FAIL;
}

#define CHECK(cmd) do { \
        XI_RETURN ret = cmd; \
        if (ret != XI_OK) { \
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "%s: %d\n", #cmd, (int) ret); \
                goto error; \
        } \
} while(0)
static int vidcap_ximea_init(struct vidcap_params *params, void **state)
{
        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }
        struct state_vidcap_ximea *s = calloc(1, sizeof(struct state_vidcap_ximea));
        s->magic = MAGIC;
        s->exposure_time_us = EXPOSURE_DEFAULT_US;
        if (!vidcap_ximea_fill_symbols(&s->funcs)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot load XIMEA library!\n");
                goto error;
        }
        int ret = vidcap_ximea_parse_params(s, vidcap_params_get_fmt(params));
        if (ret != 0) {
                free(s);
                return ret;
        }
        
        CHECK(s->funcs.xiOpenDevice(s->device_id, &s->xiH));
        CHECK(s->funcs.xiSetParamInt(s->xiH, XI_PRM_EXPOSURE, s->exposure_time_us));
        CHECK(s->funcs.xiSetParamInt(s->xiH, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24));
        CHECK(s->funcs.xiSetParamInt(s->xiH, XI_PRM_AUTO_WB, XI_ON));
        CHECK(s->funcs.xiStartAcquisition(s->xiH));

        *state = s;
        return VIDCAP_INIT_OK;

error:
        free(s);
        return VIDCAP_INIT_FAIL;
}

static void vidcap_ximea_done(void *state)
{
        struct state_vidcap_ximea *s = (struct state_vidcap_ximea *) state;
        assert(s->magic == MAGIC);
        s->funcs.xiStopAcquisition(s->xiH);
        s->funcs.xiCloseDevice(s->xiH);
        vidcap_ximea_close_lib(&s->funcs);
        free(s);
}

static struct video_frame *vidcap_ximea_grab(void *state, struct audio_frame **audio)
{
        struct state_vidcap_ximea *s = (struct state_vidcap_ximea *) state;
        assert(s->magic == MAGIC);
        int timeout_ms = DEFAULT_TIMEOUT_MS;

        XI_IMG img;
        memset(&img, 0, sizeof img);
        img.size = sizeof img;

        *audio = NULL;
        XI_RETURN ret = s->funcs.xiGetImage(s->xiH, timeout_ms, &img);
        if (ret != XI_OK) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot capture frame: %d", (int) ret);
                return NULL;
        }

        struct video_desc d;
        d.width = img.width;
        d.height = img.height;
        d.color_spec = BGR;
        d.tile_count = 1;
        d.interlacing = PROGRESSIVE;
        d.fps = MICROSEC_IN_SEC / (double) s->exposure_time_us;

        struct video_frame *out = vf_alloc_desc(d);
        out->tiles[0].data = img.bp;

        return out;
}

static struct vidcap_type *vidcap_ximea_probe(bool verbose, void (**deleter)(void *))
{
        struct vidcap_type *vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt == NULL) {
                return NULL;
        }

        vt->name = "XIMEA";
        vt->description = "XIMEA capture card";
        *deleter = free;

        if (!verbose) {
                return vt;
        }

        struct ximea_functions funcs;
        if (!vidcap_ximea_fill_symbols(&funcs)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot load XIMEA library!\n");
                return vt;
        }

        DWORD count;
        XI_RETURN ret = funcs.xiGetNumberDevices(&count);
        if (ret != XI_OK) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to get device count!\n");
                vidcap_ximea_close_lib(&funcs);
                return vt;
        }

        vt->cards = calloc(count, sizeof(struct device_info));
        for (DWORD i = 0; i < count; ++i) {
                snprintf(vt->cards[i].id, sizeof vt->cards[i].id, "ximea:device=%d", (int) i);
                char name[256];
                color_out(COLOR_OUT_BOLD, "%d) ", (int) i);
                if (funcs.xiGetDeviceInfoString(i, XI_PRM_DEVICE_NAME, name, sizeof name) == XI_OK) {
                        strncpy(vt->cards[i].name, name, sizeof vt->cards[i].name);
                }
        }
        vidcap_ximea_close_lib(&funcs);

        return vt;
}

static const struct video_capture_info vidcap_ximea_info = {
        vidcap_ximea_probe,
        vidcap_ximea_init,
        vidcap_ximea_done,
        vidcap_ximea_grab,
        true
};

REGISTER_MODULE(ximea, &vidcap_ximea_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

