/**
 * @file   audio/capture/sdi.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2015 CESNET, z. s.p. o.
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
#endif // HAVE_CONFIG_H

#include "audio/audio.h" 
#include "audio/audio_capture.h"
#include "audio/capture/sdi.h" 

#include "compat/platform_semaphore.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"

#include <condition_variable>
#include <chrono>
#include <mutex>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FRAME_NETWORK 0
#define FRAME_CAPTURE 1

using std::condition_variable;
using std::cv_status;
using std::chrono::milliseconds;
using std::mutex;
using std::unique_lock;

struct state_sdi_capture {
        struct audio_frame audio_frame[2];
        mutex lock;
        condition_variable audio_frame_ready_cv;
};

static void audio_cap_sdi_help(const char *driver_name);

static void audio_cap_sdi_probe_common(struct device_info **available_devices, int *count, 
                const char *id, const char *name)
{
        *available_devices = (struct device_info *) calloc(1, sizeof(struct device_info));
        strncpy((*available_devices)[0].id, id, sizeof (*available_devices)[0].id - 1);
        strncpy((*available_devices)[0].name, name, sizeof (*available_devices)[0].name - 1);
        *count = 1;
}

static void audio_cap_sdi_probe_embedded(struct device_info **available_devices, int *count)
{
        audio_cap_sdi_probe_common(available_devices, count, "embedded", "Embedded SDI/HDMI audio");
}

static void audio_cap_sdi_probe_aesebu(struct device_info **available_devices, int *count)
{
        audio_cap_sdi_probe_common(available_devices, count, "AESEBU", "Digital AES/EBU audio");
}

static void audio_cap_sdi_probe_analog(struct device_info **available_devices, int *count)
{
        audio_cap_sdi_probe_common(available_devices, count, "analog", "Analog audio through capture card");
}

static void * audio_cap_sdi_init(const char *cfg)
{
        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Available vidcap audio devices:\n");
                audio_cap_sdi_help("embedded");
                audio_cap_sdi_help("AESEBU");
                audio_cap_sdi_help("analog");
                printf("\t\twhere <index> is index of vidcap device "
                                "to be taken audio from.\n");
                return &audio_init_state_ok;
        }
        
        return new state_sdi_capture();
}

static struct audio_frame * audio_cap_sdi_read(void *state)
{
        struct state_sdi_capture *s = (struct state_sdi_capture *) state;

        unique_lock<mutex> lk(s->lock);
        bool rc = s->audio_frame_ready_cv.wait_for(lk, milliseconds(100), [s]{return s->audio_frame[FRAME_CAPTURE].data_len > 0;});

        if (rc == false) {
                return NULL;
        }

        // FRAME_NETWORK has been "consumed"
        s->audio_frame[FRAME_NETWORK].data_len = 0;
        // swap
        struct audio_frame tmp;
        memcpy(&tmp, &s->audio_frame[FRAME_CAPTURE], sizeof(struct audio_frame));
        memcpy(&s->audio_frame[FRAME_CAPTURE], &s->audio_frame[FRAME_NETWORK], sizeof(struct audio_frame));
        memcpy(&s->audio_frame[FRAME_NETWORK], &tmp, sizeof(struct audio_frame));

        return &s->audio_frame[FRAME_NETWORK];
}

static void audio_cap_sdi_done(void *state)
{
        struct state_sdi_capture *s;

        s = (struct state_sdi_capture *) state;
        for(int i = 0; i < 2; ++i) {
                free(s->audio_frame[i].data);
        }
}

static void audio_cap_sdi_help(const char *driver_name)
{
        if(strcmp(driver_name, "embedded") == 0) {
                printf("\tembedded[:<index>] : SDI audio (if available)\n");
        } else if(strcmp(driver_name, "AESEBU") == 0) {
                printf("\tAESEBU[:<index>] : separately connected AES/EBU to a grabbing card (if available)\n");
        } else if(strcmp(driver_name, "analog") == 0) {
                printf("\tanalog[:<index>] : analog input of grabbing card (if available)\n");
        }
}

#define MAX_BUF_SIZE_MS 100l

void sdi_capture_new_incoming_frame(void *state, struct audio_frame *frame)
{
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) state;

        unique_lock<mutex> lk(s->lock);

        if(
                        s->audio_frame[FRAME_CAPTURE].bps != frame->bps ||
                        s->audio_frame[FRAME_CAPTURE].ch_count != frame->ch_count ||
                        s->audio_frame[FRAME_CAPTURE].sample_rate != frame->sample_rate
          ) {
                s->audio_frame[FRAME_CAPTURE].bps = frame->bps;
                s->audio_frame[FRAME_CAPTURE].ch_count = frame->ch_count;
                s->audio_frame[FRAME_CAPTURE].sample_rate = frame->sample_rate;
                s->audio_frame[FRAME_CAPTURE].data_len = 0;
        }

        int needed_size = frame->data_len + s->audio_frame[FRAME_CAPTURE].data_len;
        if (needed_size > frame->bps * frame->ch_count * frame->sample_rate / 1000l * MAX_BUF_SIZE_MS) {
                fprintf(stderr, "[SDI] Maximal audio buffer length %ld ms exceeded! Dropping samples.\n",
                                MAX_BUF_SIZE_MS);
        } else {
                if (needed_size > (int) s->audio_frame[FRAME_CAPTURE].max_size) {
                        free(s->audio_frame[FRAME_CAPTURE].data);
                        s->audio_frame[FRAME_CAPTURE].max_size = needed_size;
                        s->audio_frame[FRAME_CAPTURE].data = (char *) malloc(needed_size);
                }
                memcpy(s->audio_frame[FRAME_CAPTURE].data + s->audio_frame[FRAME_CAPTURE].data_len,
                                frame->data, frame->data_len);
                s->audio_frame[FRAME_CAPTURE].data_len += frame->data_len;
        }

        lk.unlock();
        s->audio_frame_ready_cv.notify_one();
}

static const struct audio_capture_info acap_sdi_info_embedded = {
        audio_cap_sdi_probe_embedded,
        audio_cap_sdi_help,
        audio_cap_sdi_init,
        audio_cap_sdi_read,
        audio_cap_sdi_done
};

static const struct audio_capture_info acap_sdi_info_aesebu = {
        audio_cap_sdi_probe_aesebu,
        audio_cap_sdi_help,
        audio_cap_sdi_init,
        audio_cap_sdi_read,
        audio_cap_sdi_done
};

static const struct audio_capture_info acap_sdi_info_analog = {
        audio_cap_sdi_probe_analog,
        audio_cap_sdi_help,
        audio_cap_sdi_init,
        audio_cap_sdi_read,
        audio_cap_sdi_done
};

REGISTER_MODULE(embedded, &acap_sdi_info_embedded, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);
REGISTER_MODULE(AESEBU, &acap_sdi_info_aesebu, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);
REGISTER_MODULE(analog, &acap_sdi_info_analog, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);

/* vim: set expandtab: sw=8 */
