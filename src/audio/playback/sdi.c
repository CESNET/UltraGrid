/**
 * @file   audio/playback/sdi.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2016 CESNET z.s.p.o.
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

#include "audio/audio.h"
#include "audio/audio_playback.h"
#include "audio/playback/sdi.h"
#include "debug.h"
#include "lib_common.h"
#include "video_display.h"

#include <stdlib.h>
#include <string.h>

struct state_sdi_playback {
        void *udata;
        void (*put_callback)(void *, struct audio_frame *);
        int (*reconfigure_callback)(void *state, int quant_samples, int channels,
                int sample_rate);
        int (*get_property_callback)(void *, int, void *, size_t *);
};

static void audio_play_sdi_probe_common(struct device_info **available_devices, int *count, 
                const char *dev, const char *name)
{
        *available_devices = (struct device_info *) calloc(1, sizeof(struct device_info));
        strncpy((*available_devices)[0].dev, dev, sizeof (*available_devices)[0].dev - 1);
        strncpy((*available_devices)[0].name, name, sizeof (*available_devices)[0].name - 1);
        snprintf((*available_devices)[0].extra, sizeof (*available_devices)[0].extra, "\"isEmbeddedAudio\":\"t\"");
        *count = 1;
}

static void audio_play_sdi_probe_embedded(struct device_info **available_devices, int *count)
{
        audio_play_sdi_probe_common(available_devices, count, "", "Embedded SDI/HDMI audio");
}

static void audio_play_sdi_probe_aesebu(struct device_info **available_devices, int *count)
{
        audio_play_sdi_probe_common(available_devices, count, "", "Digital AES/EBU audio");
}

static void audio_play_sdi_probe_analog(struct device_info **available_devices, int *count)
{
        audio_play_sdi_probe_common(available_devices, count, "", "Analog audio through capture card");
}

static void audio_play_sdi_help(const char *driver_name)
{
        if(strcmp(driver_name, "embedded") == 0) {
                printf("\tembedded : SDI audio (if available)\n");
        } else if(strcmp(driver_name, "AESEBU") == 0) {
                printf("\tAESEBU : separately connected AES/EBU cabling to grabbing card (if available)\n");
        } else if(strcmp(driver_name, "analog") == 0) {
                printf("\tanalog : separately connected audio input (if available)\n");
        }
}

static void * audio_play_sdi_init(const char *cfg)
{
        if(cfg && strcmp(cfg, "help") == 0) {
                printf("Available embedded devices:\n");
                audio_play_sdi_help("embedded");
                audio_play_sdi_help("AESEBU");
                audio_play_sdi_help("analog");
                return &audio_init_state_ok;
        }
        struct state_sdi_playback *s = malloc(sizeof(struct state_sdi_playback));
        s->put_callback = NULL;
        s->reconfigure_callback = NULL;
        return s;
}

void sdi_register_display_callbacks(void *state, void *udata, void (*putf)(void *, struct audio_frame *), int (*reconfigure)(void *, int, int, int), int (*get_property)(void *, int, void *, size_t *))
{
        struct state_sdi_playback *s = (struct state_sdi_playback *) state;
        
        s->udata = udata;
        s->put_callback = putf;
        s->reconfigure_callback = reconfigure;
        s->get_property_callback = get_property;
}

static void audio_play_sdi_put_frame(void *state, struct audio_frame *frame)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;

        if(s->put_callback)
                s->put_callback(s->udata, frame);
}

static bool audio_play_sdi_query_format(struct state_sdi_playback *s, void *data, size_t *len)
{
        if (s->get_property_callback(s->udata, DISPLAY_PROPERTY_AUDIO_FORMAT, data, len)) {
                return true;
        } else {
                log_msg(LOG_LEVEL_WARNING, "Cannot get audio format from playback card!\n");
		struct audio_desc desc = {2, 48000, 2, AC_PCM};
		if (*len >= sizeof desc) {
			memcpy(data, &desc, sizeof desc);
			*len = sizeof desc;
                        return true;
                } else {
                        return false;
                }
        }
}

static bool audio_play_sdi_ctl(void *state, int request, void *data, size_t *len)
{
        struct state_sdi_playback *s = (struct state_sdi_playback *) state;
        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                return audio_play_sdi_query_format(s, data, len);
        default:
                return false;
        }
}

static int audio_play_sdi_reconfigure(void *state, struct audio_desc desc)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;

        if(s->reconfigure_callback) {
                return s->reconfigure_callback(s->udata, desc.bps * 8,
                                desc.ch_count, desc.sample_rate);
        } else {
                return FALSE;
        }
}

static void audio_play_sdi_done(void *s)
{
        UNUSED(s);
}

static const struct audio_playback_info aplay_sdi_info_embedded = {
        audio_play_sdi_probe_embedded,
        audio_play_sdi_help,
        audio_play_sdi_init,
        audio_play_sdi_put_frame,
        audio_play_sdi_ctl,
        audio_play_sdi_reconfigure,
        audio_play_sdi_done
};

static const struct audio_playback_info aplay_sdi_info_aesebu = {
        audio_play_sdi_probe_aesebu,
        audio_play_sdi_help,
        audio_play_sdi_init,
        audio_play_sdi_put_frame,
        audio_play_sdi_ctl,
        audio_play_sdi_reconfigure,
        audio_play_sdi_done
};

static const struct audio_playback_info aplay_sdi_info_analog = {
        audio_play_sdi_probe_analog,
        audio_play_sdi_help,
        audio_play_sdi_init,
        audio_play_sdi_put_frame,
        audio_play_sdi_ctl,
        audio_play_sdi_reconfigure,
        audio_play_sdi_done
};


REGISTER_MODULE(embedded, &aplay_sdi_info_embedded, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);
REGISTER_MODULE(AESEBU, &aplay_sdi_info_aesebu, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);
REGISTER_MODULE(analog, &aplay_sdi_info_analog, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);


/* vim: set expandtab sw=8: */

