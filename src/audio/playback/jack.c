/**
 * @file   audio/playback/jack.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2016 CESNET z.s.p.o.
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
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/ring_buffer.h"

#include <jack/jack.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_PORTS 64

#ifndef __cplusplus
#define min(a, b)      (((a) < (b))? (a): (b))
#endif

struct state_jack_playback {
        char *jack_ports_pattern;
        int jack_sample_rate;
        jack_client_t *client;
        jack_port_t *output_port[MAX_PORTS];
        struct audio_desc desc;
        char *channel;
        float *converted;

        int jack_ports_count;
        struct ring_buffer *data[MAX_PORTS];
};

static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg);
static int jack_process_callback(jack_nframes_t nframes, void *arg);

static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg)
{
        struct state_jack_playback *s = (struct state_jack_playback *) arg;

        log_msg(LOG_LEVEL_WARNING, "JACK sample rate changed from %d to %d. "
                        "Runtime change is not supported in UG and will likely "
                        "cause malfunctioning. If so, pleaser report to %s.",
                        s->jack_sample_rate, nframes, PACKAGE_BUGREPORT);

        s->jack_sample_rate = nframes;

        return 0;
}

static int jack_process_callback(jack_nframes_t nframes, void *arg)
{
        struct state_jack_playback *s = (struct state_jack_playback *) arg;
        int i;
	int channels; // actual written channels (max of available and required)

	channels = s->desc.ch_count;
	if(channels > s->jack_ports_count)
		channels = s->jack_ports_count;

        for (i = 0; i < channels; ++i) {
                if(ring_get_current_size(s->data[i]) <= (int) (nframes * sizeof(float))) {
                        fprintf(stderr, "[JACK playback] Buffer underflow detected.\n");
                }
        }
        for (i = 0; i < channels; ++i) {
                int ret;
                jack_default_audio_sample_t *out =
                        jack_port_get_buffer (s->output_port[i], nframes);
                ret = ring_buffer_read(s->data[i], (char *)out, nframes * sizeof(float));
                if((unsigned int) ret != nframes * sizeof(float)) {
                        fprintf(stderr, "[JACK playback] Buffer underflow detected (channel %d).\n", i);
                }
        }

        return 0;
}

static void audio_play_jack_probe(struct device_info **available_devices, int *count)
{
        *available_devices = malloc(sizeof(struct device_info));
        strcpy((*available_devices)[0].id, "jack");
        strcpy((*available_devices)[0].name, "JACK audio output");
        *count = 1;
}

static void audio_play_jack_help(const char *driver_name)
{
        UNUSED(driver_name);
        jack_client_t *client;
        jack_status_t status;
        char *last_name = NULL;
        int i;
        int channel_count;
        const char **ports;

        client = jack_client_open(PACKAGE_STRING, JackNullOption, &status);
        if(status & JackFailure) {
                fprintf(stderr, "[JACK playback] Opening JACK client failed.\n");
                return;
        }

        ports = jack_get_ports(client, NULL, NULL, JackPortIsInput);
        if(ports == NULL) {
                fprintf(stderr, "[JACK playback] Unable to enumerate ports.\n");
                return;
        }

        printf("JACK playback:\n");
        i = 0;
        channel_count = 0;
        while(ports[i] != NULL) {
                char *item = strdup(ports[i]);
                char *save_ptr = NULL;
                char *name;

                ++channel_count;
                name = strtok_r(item, "_", &save_ptr);
                if(last_name && strcmp(last_name, name) != 0) {
                        printf("\tjack:%s (%d channels)\n", last_name, channel_count);
                        channel_count = 0;
                }
                free(last_name);
                last_name = strdup(name);
                free(item);
                ++i;
        }
        if(last_name) {
                printf("\tjack:%s (%d channels)\n", last_name, channel_count);
        }
        free(last_name);
        jack_client_close(client);
}

static void * audio_play_jack_init(const char *cfg)
{
        struct state_jack_playback *s;
        const char **ports;
        jack_status_t status;

        if(!cfg || strcmp(cfg, "help") == 0) {
                audio_play_jack_help("jack");
                return &audio_init_state_ok;
        }

        s = calloc(1, sizeof(struct state_jack_playback));
        if(!s) {
                fprintf(stderr, "[JACK playback] Unable to allocate memory.\n");
                goto error;
        }

        s->jack_ports_pattern = strdup(cfg);

        s->client = jack_client_open(PACKAGE_STRING, JackNullOption, &status);
        if(status & JackFailure) {
                fprintf(stderr, "[JACK playback] Opening JACK client failed.\n");
                goto error;
        }

        if(jack_set_sample_rate_callback(s->client, jack_samplerate_changed_callback, (void *) s)) {
                fprintf(stderr, "[JACK capture] Registering callback problem.\n");
                goto release_client;
        }


        if(jack_set_process_callback(s->client, jack_process_callback, (void *) s) != 0) {
                fprintf(stderr, "[JACK capture] Process callback registration problem.\n");
                goto release_client;
        }

        s->jack_sample_rate = jack_get_sample_rate (s->client);
	fprintf(stderr, "JACK sample rate: %d\n", (int) s->jack_sample_rate);


        ports = jack_get_ports(s->client, cfg, NULL, JackPortIsInput);
        if(ports == NULL) {
                fprintf(stderr, "[JACK playback] Unable to input ports matching %s.\n", cfg);
                goto release_client;
        }

        s->jack_ports_count = 0;
        while(ports[s->jack_ports_count]) s->jack_ports_count++;
        free(ports);

        {
                char name[30];
                int i;
                
                for(i = 0; i < MAX_PORTS; ++i) {
                        snprintf(name, 30, "playback_%02u", i);
                        s->output_port[i] = jack_port_register (s->client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
                }
        }
                

        return s;

release_client:
        jack_client_close(s->client);
error:
        free(s);
        return NULL;
}

static bool audio_play_jack_query_format(struct state_jack_playback *s, void *data, size_t *len)
{
        struct audio_desc desc;
        if (*len < sizeof desc) {
                return false;
        } else {
                memcpy(&desc, data, sizeof desc);
        }

        desc = (struct audio_desc){4, s->jack_sample_rate, min(s->jack_ports_count, desc.ch_count), AC_PCM};

        memcpy(data, &desc, sizeof desc);
        *len = sizeof desc;
        return true;
}

static bool audio_play_jack_ctl(void *state, int request, void *data, size_t *len)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;

        switch (request) {
        case AUDIO_PLAYBACK_CTL_QUERY_FORMAT:
                return audio_play_jack_query_format(s, data, len);
        default:
                return false;
        }
}

static int audio_play_jack_reconfigure(void *state, struct audio_desc desc)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;
        const char **ports;
        int i;

        assert(desc.bps == 4 && desc.sample_rate == s->jack_sample_rate && desc.codec == AC_PCM);

        jack_deactivate(s->client);

        ports = jack_get_ports(s->client, s->jack_ports_pattern, NULL, JackPortIsInput);
        if(ports == NULL) {
                fprintf(stderr, "[JACK playback] Unable to input ports matching %s.\n", s->jack_ports_pattern);
                return FALSE;
        }

        if(desc.ch_count > s->jack_ports_count) {
                fprintf(stderr, "[JACK playback] Warning: received %d audio channels, JACK can process only %d.", desc.ch_count, s->jack_ports_count);
        }

        for(i = 0; i < MAX_PORTS; ++i) {
                ring_buffer_destroy(s->data[i]);
                s->data[i] = NULL;
        }
        /* for all channels previously connected */
        for(i = 0; i < desc.ch_count; ++i) {
                jack_disconnect(s->client, jack_port_name (s->output_port[i]), ports[i]);
		fprintf(stderr, "[JACK playback] Port %d: %s\n", i, ports[i]);
        }
        free(s->channel);
        free(s->converted);
        s->desc.bps = desc.bps;
        s->desc.ch_count = desc.ch_count;
        s->desc.sample_rate = desc.sample_rate;

        s->channel = malloc(s->desc.bps * desc.sample_rate);
        s->converted = (float *) malloc(desc.sample_rate * sizeof(float));

        for(i = 0; i < desc.ch_count; ++i) {
                s->data[i] = ring_buffer_init(sizeof(float) * s->jack_sample_rate);
        }

        if(jack_activate(s->client)) {
                fprintf(stderr, "[JACK capture] Cannot activate client.\n");
                return FALSE;
        }

        for(i = 0; i < desc.ch_count; ++i) {
                if (jack_connect (s->client, jack_port_name (s->output_port[i]), ports[i])) {
                        fprintf (stderr, "Cannot connect output port: %d.\n", i);
                        return FALSE;
                }
        }
        free(ports);

        return TRUE;
}

static void audio_play_jack_put_frame(void *state, struct audio_frame *frame)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;
        assert(frame->bps == 4);

        int channel_size = frame->data_len / frame->ch_count;

        for (int i = 0; i < frame->ch_count; ++i) {
                demux_channel(s->channel, frame->data, frame->bps, frame->data_len, frame->ch_count, i);
                int2float((char *) s->converted, (char *) s->channel, channel_size);
                ring_buffer_write(s->data[i], (char *) s->converted, channel_size);
        }
}

static void audio_play_jack_done(void *state)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;
        int i;

        jack_client_close(s->client);
        free(s->channel);
        free(s->converted);
        free(s->jack_ports_pattern);
        for(i = 0; i < MAX_PORTS; ++i) {
                ring_buffer_destroy(s->data[i]);
        }

        free(s);
}

static const struct audio_playback_info aplay_jack_info = {
        audio_play_jack_probe,
        audio_play_jack_help,
        audio_play_jack_init,
        audio_play_jack_put_frame,
        audio_play_jack_ctl,
        audio_play_jack_reconfigure,
        audio_play_jack_done
};

REGISTER_MODULE(jack, &aplay_jack_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

