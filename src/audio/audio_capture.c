/**
 * @file   audio/audio_capture.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2019 CESNET, z. s. p. o.
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
#endif

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "debug.h"
#include "host.h"

#include "audio/audio_capture.h"
#include "audio/capture/sdi.h"
#include "audio/types.h"

/* vidcap flags */
#include "video_capture.h"

#include "lib_common.h"

struct state_audio_capture {
        char name[128];
        const struct audio_capture_info *funcs;
        void *state;
};

int audio_capture_init(const char *driver, char *cfg, struct state_audio_capture **state)
{
        struct state_audio_capture *s = calloc(1, sizeof(struct state_audio_capture));
        assert(s != NULL);

        s->funcs = load_library(driver, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);

        if (s->funcs == NULL) {
                log_msg(LOG_LEVEL_ERROR, "Unknown audio capture driver: %s\n", driver);
                goto error;
        }

        strncpy(s->name, driver, sizeof s->name - 1);

        s->state = s->funcs->init(cfg);

        if(!s->state) {
                log_msg(LOG_LEVEL_ERROR, "Error initializing audio capture.\n");
                goto error;
        }

        if(s->state == &audio_init_state_ok) {
                free(s);
                return 1;
        }

        *state = s;
        return 0;

error:
        free(s);
        return -1;
}

struct state_audio_capture *audio_capture_init_null_device()
{
        struct state_audio_capture *device = NULL;
        int ret = audio_capture_init("none", NULL, &device);
        if (ret != 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize null audio capture: %d\n", ret);
        }
        return device;
}

void audio_capture_print_help(bool full)
{
        printf("Available audio capture devices:\n");
        list_modules(LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION, full);
}

void audio_capture_done(struct state_audio_capture *s)
{
        if(s) {
                s->funcs->done(s->state);

                free(s);
        }
}

struct audio_frame * audio_capture_read(struct state_audio_capture *s)
{
        if(s) {
                return s->funcs->read(s->state);
        } else {
                return NULL;
        }
}

/**
 * @returns vidcap flags if audio should be taken from video
 * capture device.
 */
unsigned int audio_capture_get_vidcap_flags(const char *const_device_name)
{
        char *tmp = strdup(const_device_name);
        assert(tmp != NULL);
        char *save_ptr = NULL;
        char *device_name = strtok_r(tmp, ":", &save_ptr);
        assert(device_name != NULL);
        unsigned int ret;

        if(strcasecmp(device_name, "embedded") == 0) {
                ret = VIDCAP_FLAG_AUDIO_EMBEDDED;
        } else if(strcasecmp(device_name, "AESEBU") == 0) {
                ret = VIDCAP_FLAG_AUDIO_AESEBU;
        } else if(strcasecmp(device_name, "analog") == 0) {
                ret = VIDCAP_FLAG_AUDIO_ANALOG;
        } else {
                ret = 0u;
        }

        free(tmp);
        return ret;
}

/**
 * @returns optional index to video capture device to which
 * should be audio flags passed.
 * @see audio_capture_get_vidcap_flags
 */
int audio_capture_get_vidcap_index(const char *const_device_name)
{
        char *tmp = strdup(const_device_name);
        assert(tmp != NULL);
        char *save_ptr = NULL;
        unsigned int ret;

        strtok_r(tmp, ":", &save_ptr);
        char *vidcap_index_str = strtok_r(NULL, ":", &save_ptr);

        if (vidcap_index_str == NULL) {
                ret = -1;
        } else {
                ret = atoi(vidcap_index_str);
        }

        free(tmp);
        return ret;
}

const char *audio_capture_get_driver_name(struct state_audio_capture * s)
{
        return s->name;
}

void *audio_capture_get_state_pointer(struct state_audio_capture *s)
{
        return s->state;
}

