/**
 * @file   audio/capture/sdi.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2026 CESNET, zájmové sdružení právnických osob
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

#include "audio/capture/sdi.h"

#include <assert.h>  // for assert
#include <pthread.h> // for pthread_mutex_unlock, pthread_mutex...
#include <stdio.h>   // for printf, NULL
#include <stdlib.h>  // for free, calloc, atoi, malloc
#include <string.h>  // for memcpy, strcmp

#include "audio/audio_capture.h" // for AUDIO_CAPTURE_ABI_VERSION, audio_ca...
#include "audio/types.h"         // for audio_frame
#include "compat/c23.h"          // IWYU pragma: keep
#include "debug.h"               // for LOG_LEVEL_ERROR, LOG_LEVEL_WARNING
#include "host.h"                // for INIT_NOERR, get_commandline_param
#include "lib_common.h"          // for REGISTER_MODULE, library_class
#include "tv.h"                  // for MS_TO_NS, time_ns_t
#include "types.h"               // for device_info, frame_flags_common
#include "utils/macros.h"        // for strcpy_ch
#include "utils/pthread.h"       // for CHK_PTHR, ug_pthread_cond_reltimedwait

struct module;

#define DEFAULT_BUF_SIZE_MS 100L

#define FRAME_NETWORK 0
#define FRAME_CAPTURE 1
#define MOD_NAME "[acap.] "

struct state_sdi_capture {
        long               buf_size_ms;
        struct audio_frame audio_frame[2];
        pthread_mutex_t    lock;
        pthread_cond_t     audio_frame_ready_cv;
};

static void audio_cap_sdi_help(const char *driver_name);

static void audio_cap_sdi_probe_common(struct device_info **available_devices, int *count, 
                const char *dev, const char *name)
{
        *available_devices = (struct device_info *) calloc(1, sizeof(struct device_info));
        strcpy_ch((*available_devices)[0].dev, dev);
        strcpy_ch((*available_devices)[0].name, name);
        strcpy_ch((*available_devices)[0].extra, "\"isEmbeddedAudio\":\"t\"");
        *count = 1;
}

static void audio_cap_sdi_probe_embedded(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        audio_cap_sdi_probe_common(available_devices, count, "", "Embedded SDI/HDMI audio");
}

static void audio_cap_sdi_probe_aesebu(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        audio_cap_sdi_probe_common(available_devices, count, "", "Digital AES/EBU audio");
}

static void audio_cap_sdi_probe_analog(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        audio_cap_sdi_probe_common(available_devices, count, "", "Analog audio through capture card");
}

static void * audio_cap_sdi_init(struct module *parent, const char *cfg)
{
        UNUSED(parent);
        if (strcmp(cfg, "help") == 0) {
                printf("Available vidcap audio devices:\n");
                audio_cap_sdi_help("embedded");
                audio_cap_sdi_help("AESEBU");
                audio_cap_sdi_help("analog");
                printf("\t\twhere <index> is index of vidcap device "
                                "to be taken audio from.\n");
                return INIT_NOERR;
        }

        struct state_sdi_capture *s = calloc(1, sizeof *s);
        s->buf_size_ms =  DEFAULT_BUF_SIZE_MS;
        ug_pthread_mutex_init(&s->lock);
        pthread_cond_init(&s->audio_frame_ready_cv, nullptr);

        // the audio-buffer-len is used mainly for playback buffers so this
        // place for the param is not much related... but it is just a
        // param so not intended for users....
        const char *buf_len = get_commandline_param("audio-buffer-len");
        if (buf_len != nullptr) {
                s->buf_size_ms = atoi(buf_len);
                assert(s->buf_size_ms > 0);
                if (s->buf_size_ms > 1000) {
                        MSG(WARNING, "Suspicions buffer length used: %ld ms\n",
                            s->buf_size_ms);
                }
        }

        return s;
}

static const struct audio_frame *
audio_cap_sdi_read(void *state)
{
        struct state_sdi_capture *s = state;

        CHK_PTHR(pthread_mutex_lock(&s->lock));
        {
                time_ns_t timeout = MS_TO_NS(100);
                int rc = 0;
                while (s->audio_frame[FRAME_CAPTURE].data_len == 0 &&
                       rc == 0) {
                        rc = ug_pthread_cond_reltimedwait(
                            &s->audio_frame_ready_cv, &s->lock, &timeout);
                }

                if (s->audio_frame[FRAME_CAPTURE].data_len == 0) {
                        CHK_PTHR(pthread_mutex_unlock(&s->lock));
                        return nullptr;
                }

                // FRAME_NETWORK has been "consumed"
                s->audio_frame[FRAME_NETWORK].data_len = 0;
                // swap
                struct audio_frame tmp;
                memcpy(&tmp, &s->audio_frame[FRAME_CAPTURE],
                       sizeof(struct audio_frame));
                memcpy(&s->audio_frame[FRAME_CAPTURE],
                       &s->audio_frame[FRAME_NETWORK],
                       sizeof(struct audio_frame));
                memcpy(&s->audio_frame[FRAME_NETWORK], &tmp,
                       sizeof(struct audio_frame));
        }
        CHK_PTHR(pthread_mutex_unlock(&s->lock));

        return &s->audio_frame[FRAME_NETWORK];
}

static void audio_cap_sdi_done(void *state)
{
        struct state_sdi_capture *s = state;
        for(int i = 0; i < 2; ++i) {
                free(s->audio_frame[i].data);
        }
        CHK_PTHR(pthread_mutex_destroy(&s->lock));
        CHK_PTHR(pthread_cond_destroy(&s->audio_frame_ready_cv));
        free(s);
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

static void
process_incoming_frame(struct state_sdi_capture *s, struct audio_frame *frame)
{
        if (s->audio_frame[FRAME_CAPTURE].bps != frame->bps ||
                        s->audio_frame[FRAME_CAPTURE].ch_count != frame->ch_count ||
                        s->audio_frame[FRAME_CAPTURE].sample_rate != frame->sample_rate) {
                s->audio_frame[FRAME_CAPTURE].bps = frame->bps;
                s->audio_frame[FRAME_CAPTURE].ch_count = frame->ch_count;
                s->audio_frame[FRAME_CAPTURE].sample_rate = frame->sample_rate;
                s->audio_frame[FRAME_CAPTURE].data_len = 0;
                s->audio_frame[FRAME_CAPTURE].max_size = frame->bps * frame->ch_count * frame->sample_rate / 1000L * s->buf_size_ms;
                s->audio_frame[FRAME_CAPTURE].data = malloc(s->audio_frame[FRAME_CAPTURE].max_size);
        }

        int len = frame->data_len;
        if (len + s->audio_frame[FRAME_CAPTURE].data_len > s->audio_frame[FRAME_CAPTURE].max_size) {
                MSG(WARNING,
                    "Maximal audio buffer length %ld ms exceeded! Dropping %d "
                    "samples.\n",
                    s->buf_size_ms,
                    len - (s->audio_frame[FRAME_CAPTURE].max_size -
                           s->audio_frame[FRAME_CAPTURE].data_len));
                len = s->audio_frame[FRAME_CAPTURE].max_size - s->audio_frame[FRAME_CAPTURE].data_len;
        }
        memcpy(s->audio_frame[FRAME_CAPTURE].data + s->audio_frame[FRAME_CAPTURE].data_len,
                        frame->data, len);
        if (s->audio_frame[FRAME_CAPTURE].data_len == 0 && (frame->flags & TIMESTAMP_VALID) != 0) {
                s->audio_frame[FRAME_CAPTURE].timestamp = frame->timestamp;
                s->audio_frame[FRAME_CAPTURE].flags |= TIMESTAMP_VALID;
        }
        s->audio_frame[FRAME_CAPTURE].data_len += len;
}

void sdi_capture_new_incoming_frame(void *state, struct audio_frame *frame)
{
        struct state_sdi_capture *s = state;

        CHK_PTHR(pthread_mutex_lock(&s->lock));
        {
                process_incoming_frame(s, frame);
        }
        CHK_PTHR(pthread_mutex_unlock(&s->lock));

        CHK_PTHR(pthread_cond_signal(&s->audio_frame_ready_cv));
}

static const struct audio_capture_info acap_sdi_info_embedded = {
        audio_cap_sdi_probe_embedded,
        audio_cap_sdi_init,
        audio_cap_sdi_read,
        audio_cap_sdi_done
};

static const struct audio_capture_info acap_sdi_info_aesebu = {
        audio_cap_sdi_probe_aesebu,
        audio_cap_sdi_init,
        audio_cap_sdi_read,
        audio_cap_sdi_done
};

static const struct audio_capture_info acap_sdi_info_analog = {
        audio_cap_sdi_probe_analog,
        audio_cap_sdi_init,
        audio_cap_sdi_read,
        audio_cap_sdi_done
};

REGISTER_MODULE(embedded, &acap_sdi_info_embedded, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);
REGISTER_MODULE(AESEBU, &acap_sdi_info_aesebu, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);
REGISTER_MODULE(analog, &acap_sdi_info_analog, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);

/* vim: set expandtab sw=8: */
