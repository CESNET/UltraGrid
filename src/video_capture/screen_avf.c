/**
 * @file   video_capture/screen_avf.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * screen capture using AVFoundation
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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

#include <stdlib.h>                // for calloc, free
#include <strings.h>               // for strcasecmp

#include "compat/c23.h"            // C23 fallback defines (nullptr etc.)
#include "lib_common.h"            // for REGISTER_MODULE, library_class
#include "types.h"                 // for device_info
#include "utils/macros.h"          // for snprintf_ch
#include "video_capture.h"         // for video_capture_info, VIDCAP_INIT_NOERR
#include "video_capture_params.h"  // for vidcap_params_get_fmt

struct audio_frame;
struct vidcap_params;

#define MOD_NAME "[screen cap avf] "

// items defined in avfoundation.mm
extern const struct video_capture_info vidcap_avfoundation_info;
bool                  avfoundation_usage(const char *fmt, bool for_screen);
unsigned              avfoundation_get_screen_count(void);
extern const char     AFV_SCR_CAP_NAME_PREF[];
extern const unsigned AVF_SCR_CAP_OFF;

static void
vidcap_screen_avf_probe(struct device_info **available_cards, int *count,
                        void (**deleter)(void *))
{
        *available_cards = nullptr;
        *count           = (int) avfoundation_get_screen_count();
        *deleter         = free;
        if (*count == 0) {
                return;
        }
        *available_cards = calloc(*count, sizeof(struct device_info));

        for (int i = 0; i < *count; ++i) {
                snprintf_ch((*available_cards)[i].name, ":%u",
                            AVF_SCR_CAP_OFF + i);
                snprintf_ch((*available_cards)[i].dev, "%s%d",
                            AFV_SCR_CAP_NAME_PREF, i);
        }
}

/// @returns whether the config string contains device spec (d=/uid=/name=)
static bool
contains_dev_spec(const char *fmt)
{
        char *cpy     = strdup(fmt);
        char *tmp     = cpy;
        char *saveptr = nullptr;
        char *item    = nullptr;
        while ((item = strtok_r(tmp, ":", &saveptr)) != nullptr) {
                tmp = nullptr;
                char *delim = strchr(item, '=');
                if (!delim) {
                        continue;
                }
                *delim = '\0';
                if (strstr(item, "device") == item ||
                    strstr(item, "uid") == item ||
                    strstr(item, "name") == item) {
                        return true;
                }
        }
        return false;
}

static int
vidcap_screen_avf_init(struct vidcap_params *params, void **state)
{
        const char *fmt = vidcap_params_get_fmt(params);

        if (avfoundation_usage(fmt, true)) {
                return VIDCAP_INIT_NOERR; // help shown
        }

        if (contains_dev_spec(fmt)) {
                return vidcap_avfoundation_info.init(params, state);
        }

        // add "d=100" to initialize first screen cap av foundation device
        const size_t orig_len = strlen(fmt);
        const size_t new_len  = orig_len + 50;
        char        *new_fmt  = malloc(new_len);
        (void) snprintf(new_fmt, new_len, "%s%sd=%u", fmt,
                        orig_len == 0 ? "" : ":", AVF_SCR_CAP_OFF);
        vidcap_params_replace_fmt(params, new_fmt);
        free(new_fmt);
        return vidcap_avfoundation_info.init(params, state);
}

static void
vidcap_screen_avf_done(void *state)
{
        return vidcap_avfoundation_info.done(state);
}

static struct video_frame *
vidcap_screen_avf_grab(void *state, struct audio_frame **audio)
{
        return vidcap_avfoundation_info.grab(state, audio);
}

static const struct video_capture_info vidcap_screen_avf_info = {
        vidcap_screen_avf_probe,
        vidcap_screen_avf_init,
        vidcap_screen_avf_done,
        vidcap_screen_avf_grab,
        MOD_NAME,
};

REGISTER_MODULE(screen, &vidcap_screen_avf_info, LIBRARY_CLASS_VIDEO_CAPTURE,
                VIDEO_CAPTURE_ABI_VERSION);
