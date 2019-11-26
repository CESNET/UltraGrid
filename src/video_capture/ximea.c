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

#include <xiApi.h>

#include "debug.h"
#include "lib_common.h"
#include "utils/misc.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_capture.h"

#define EXPOSURE_DEFAULT_US 10000
#define MAGIC to_fourcc('X', 'I', 'M', 'E')
#define MOD_NAME "[XIMEA] "

struct state_vidcap_ximea {
        uint32_t magic;
        int device_id;
        int exposure_time_us;

        HANDLE xiH;
};

static void vidcap_ximea_show_help() {
        color_out(0, "XIMEA usage:\n");
        color_out(COLOR_OUT_RED | COLOR_OUT_BOLD, "\t-t ximea");
        color_out(COLOR_OUT_BOLD, "[:device=<d>][:exposure=<time_us>]\n");
        printf("\n");
        printf("Available devices:\n");

        DWORD count;
        XI_RETURN ret = xiGetNumberDevices(&count);
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
                if (xiGetDeviceInfoString(i, XI_PRM_DEVICE_NAME, name, sizeof name) == XI_OK) {
                        color_out(0, "%s", name);
                }
                color_out(0, "\n");
        }
}

static int vidcap_ximea_parse_params(struct state_vidcap_ximea *s, const char *cfg) {
        if (cfg == NULL || cfg[0] == '\0') {
                return 0;
        }

        char *fmt = strdup(cfg);
        char *save_ptr, *tmp = fmt, *tok;
        while ((tok = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                if (!strcmp(tok, "help")) {
                        vidcap_ximea_show_help();
                        free(fmt);
                        return VIDCAP_INIT_NOERR;
                } else if (strstr(tok, "device=")) {
                        s->device_id = atoi(tok + strlen("device="));
                } else if (strstr(tok, "exposure=")) {
                        s->exposure_time_us = atoi(tok + strlen("exposure="));
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option: %s\n", tok);
                        free(fmt);
                        return VIDCAP_INIT_FAIL;
                }
                tmp = NULL;
        }

        free(fmt);
        return 0;
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
        int ret = vidcap_ximea_parse_params(s, vidcap_params_get_fmt(params));
        if (ret != 0) {
                free(s);
                return ret;
        }
        
        CHECK(xiOpenDevice(s->device_id, &s->xiH));
        CHECK(xiSetParamInt(s->xiH, XI_PRM_EXPOSURE, s->exposure_time_us));
        CHECK(xiSetParamInt(s->xiH, XI_PRM_IMAGE_DATA_FORMAT, XI_RGB24));
        CHECK(xiSetParamInt(s->xiH, XI_PRM_AUTO_WB, XI_ON));
        CHECK(xiStartAcquisition(s->xiH));

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
        xiStopAcquisition(s->xiH);
        xiCloseDevice(s->xiH);
        free(s);
}

static struct video_frame *vidcap_ximea_grab(void *state, struct audio_frame **audio)
{
        struct state_vidcap_ximea *s = (struct state_vidcap_ximea *) state;
        assert(s->magic == MAGIC);
        int timeout_ms = 100;

        XI_IMG img;
        memset(&img, 0, sizeof img);
        img.size = sizeof img;

        *audio = NULL;
        XI_RETURN ret = xiGetImage(s->xiH, timeout_ms, &img);
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
        d.fps = 1000000l / s->exposure_time_us;

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

        DWORD count;
        XI_RETURN ret = xiGetNumberDevices(&count);
        if (ret != XI_OK) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to get device count!\n");
                return vt;
        }

        vt->cards = calloc(count, sizeof(struct device_info));
        for (DWORD i = 0; i < count; ++i) {
                snprintf(vt->cards[i].id, sizeof vt->cards[i].id, "ximea:device=%d", (int) i);
                char name[256];
                color_out(COLOR_OUT_BOLD, "%d) ", (int) i);
                if (xiGetDeviceInfoString(i, XI_PRM_DEVICE_NAME, name, sizeof name) == XI_OK) {
                        strncpy(vt->cards[i].name, name, sizeof vt->cards[i].name);
                }
        }

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

