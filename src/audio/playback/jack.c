/**
 * @file   audio/playback/jack.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2024 CESNET z.s.p.o.
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

#include <jack/jack.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "audio/audio_playback.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "config.h"                // for PACKAGE_NAME
#include "debug.h"
#include "host.h"
#include "jack_common.h"
#include "lib_common.h"
#include "utils/audio_buffer.h"
#include "utils/macros.h"
#include "utils/ring_buffer.h"

#define MOD_NAME "[JACK playback] "

enum {
        DEFAULT_AUDIO_BUF_LEN_MS = 50,
        MAX_LEN_MS               = 1000,
        MAX_PORTS                = 64,
};

struct state_jack_playback {
        struct libjack_connection *libjack;

        char *jack_ports_pattern;
        int jack_sample_rate;
        jack_client_t *client;
        jack_port_t *output_port[MAX_PORTS];
        struct audio_desc desc;
        int max_channel_len; ///< maximal length of channel data that is processed at once
        float *converted; ///< temporery buffer for int2float (put_frame)

        int jack_ports_count;
        void *data; // audio buffer
        struct audio_buffer_api *buffer_fns;
        char *tmp; ///< temporary buffer used to demux data

        long int first_channel;
};

static void audio_play_jack_done(void *state);
static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg);
static int jack_process_callback(jack_nframes_t nframes, void *arg);

static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg)
{
        struct state_jack_playback *s = (struct state_jack_playback *) arg;

        if (s->jack_sample_rate != 0) {
                bug_msg(LOG_LEVEL_WARNING, "JACK sample rate changed from %d to %d. "
                                "Runtime change is not supported in UG and will likely "
                                "cause malfunctioning. ",
                                s->jack_sample_rate, nframes);
        } else {
                log_msg(LOG_LEVEL_NOTICE, MOD_NAME "Sample rate changed to: %d\n",
                                nframes);
        }

        s->jack_sample_rate = nframes;

        return 0;
}

static int jack_process_callback(jack_nframes_t nframes, void *arg)
{
        struct state_jack_playback *s = (struct state_jack_playback *) arg;
        int len;
        int req_len = s->desc.ch_count * nframes * sizeof(float);
        int nframes_available = nframes;

        len = s->buffer_fns->read(s->data, s->tmp, req_len);
        if (len != req_len) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Buffer underflow detected.\n");
                nframes_available = len / s->desc.ch_count / sizeof(float);
        }

        for (int i = 0; i < s->desc.ch_count; ++i) {
                jack_default_audio_sample_t *out =
                        s->libjack->port_get_buffer (s->output_port[i], nframes_available);
                assert(out != NULL);
                demux_channel((char *) out, s->tmp, sizeof(float), len, s->desc.ch_count, i);
        }

        return 0;
}

static void audio_play_jack_probe(struct device_info **available_devices, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_devices = audio_jack_probe(PACKAGE_STRING, JackPortIsInput, count);
}

static void audio_play_jack_help(const char *client_name)
{
        printf("Usage:\n");
        printf("\t-r jack[:first_channel=<f>][:name=<n>][:<device>]\n");
        printf("\twhere\n");
        printf("\t\t<f> - index of first channel to capture (default: 0)\n");
        printf("\t\t<n> - name of the JACK client (default: %s)\n", PACKAGE_NAME);
        printf("\n");

        printf("Available devices:\n");
        int count = 0;
        struct device_info *available_devices =
            audio_jack_probe(client_name, JackPortIsInput, &count);
        for (int i = 0; i < count; i++) {
                printf("\t%s\n", available_devices[i].name);
        }
        free(available_devices);
}

static void *
audio_play_jack_init(const struct audio_playback_opts *opts)
{
        const char **ports;
        jack_status_t status;
        char client_name[STR_LEN];
        const char *source_name = "";

        snprintf_ch(client_name, "%s", PACKAGE_NAME);
        if (strcmp(opts->cfg, "help") == 0) {
                audio_play_jack_help(client_name);
                return INIT_NOERR;
        }

        struct state_jack_playback *s = calloc(1, sizeof(*s));
        if(!s) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to allocate memory.\n");
                return NULL;
        }

        s->libjack = open_libjack();
        if (s->libjack == NULL) {
                free(s);
                return NULL;
        }

        char dup[STR_LEN];
        snprintf_ch(dup, "%s", opts->cfg);
        char *tmp = dup, *item, *save_ptr;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                if (strstr(item, "first_channel=") == item) {
                        char *endptr;
                        char *val = item + strlen("first_channel=");
                        errno = 0;
                        s->first_channel = strtol(val, &endptr, 0);
                        if (errno == ERANGE || *endptr != '\0' || s->first_channel < 0) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong value '%s'.\n", val);
                                goto error;
                        }
                } else if (strstr(item, "name=") == item) {
                        snprintf_ch(client_name, "%s", strchr(item, '=') + 1);
                } else { // the rest is the device name
                        source_name = opts->cfg + (item - dup);
                        break;
                }
                tmp = NULL;
        }

        s->jack_ports_pattern = strdup(source_name);

        s->client = s->libjack->client_open(client_name, JackNullOption, &status);
        if(status & JackFailure) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Opening JACK client failed.\n");
                goto error;
        }

        if (s->libjack->set_sample_rate_callback(s->client, jack_samplerate_changed_callback, (void *) s)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Registering callback problem.\n");
                goto error;
        }


        if (s->libjack->set_process_callback(s->client, jack_process_callback, (void *) s) != 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Process callback registration problem.\n");
                goto error;
        }

        s->jack_sample_rate = s->libjack->get_sample_rate (s->client);
        log_msg(LOG_LEVEL_INFO, "JACK sample rate: %d\n", s->jack_sample_rate);


        ports = s->libjack->get_ports(s->client, s->jack_ports_pattern, NULL, JackPortIsInput);
        if(ports == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to input ports matching %s.\n", s->jack_ports_pattern);
                goto error;
        }

        s->jack_ports_count = 0;
        while(ports[s->jack_ports_count]) s->jack_ports_count++;
        free(ports);

        {
                char name[30];
                int i;
                
                for(i = 0; i < MAX_PORTS; ++i) {
                        snprintf(name, 30, "playback_%02u", i);
                        s->output_port[i] = s->libjack->port_register (s->client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
                }
        }
                

        return s;

error:
        audio_play_jack_done(s);
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

        if (s->jack_sample_rate <= 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong JACK sample rate detected: %d!",
                                s->jack_sample_rate);
                return false;
        }
        desc = (struct audio_desc){4, s->jack_sample_rate, MIN(s->jack_ports_count, desc.ch_count), AC_PCM};

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

static bool audio_play_jack_reconfigure(void *state, struct audio_desc desc)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;
        const char **ports;
        int i;

        assert(desc.bps == 4 && desc.sample_rate == s->jack_sample_rate && desc.codec == AC_PCM);

        s->libjack->deactivate(s->client);

        ports = s->libjack->get_ports(s->client, s->jack_ports_pattern, NULL, JackPortIsInput);
        if(ports == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to input ports matching %s.\n", s->jack_ports_pattern);
                return false;
        }

        if(desc.ch_count > s->jack_ports_count) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Warning: received %d audio channels, JACK can process only %d.", desc.ch_count, s->jack_ports_count);
        }

        if (s->buffer_fns) {
                s->buffer_fns->destroy(s->data);
                s->buffer_fns = NULL;
                s->data = NULL;
        }

        {
                int buf_len_ms = DEFAULT_AUDIO_BUF_LEN_MS;
                if (get_commandline_param("low-latency-audio")) {
                        buf_len_ms = 5;
                }
                if (get_commandline_param("audio-buffer-len")) {
                        buf_len_ms = atoi(get_commandline_param("audio-buffer-len"));
                        assert(buf_len_ms > 0 && buf_len_ms < MAX_LEN_MS);
                }
                if (get_commandline_param("audio-disable-adaptive-buffer")) {
                        int buf_len = desc.bps * desc.ch_count * (desc.sample_rate * buf_len_ms / 1000);
                        s->data = ring_buffer_init(buf_len);
                        s->buffer_fns = &ring_buffer_fns;
                } else {
                        s->data = audio_buffer_init(desc.sample_rate, desc.bps, desc.ch_count, buf_len_ms);
                        s->buffer_fns = &audio_buffer_fns;
                }
        }

        /* for all channels previously connected */
        for(i = 0; i < desc.ch_count; ++i) {
                s->libjack->disconnect(s->client, s->libjack->port_name (s->output_port[i]), ports[i]);
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Port %d: %s\n", i, ports[i]);
        }
        free(s->tmp);
        free(s->converted);
        s->desc.bps = desc.bps;
        s->desc.ch_count = desc.ch_count;
        s->desc.sample_rate = desc.sample_rate;

        s->max_channel_len = (desc.sample_rate / 1000) * MAX_LEN_MS * sizeof(float);
        s->tmp = malloc(s->max_channel_len);
        s->converted = malloc(desc.ch_count * s->max_channel_len);

        if (s->libjack->activate(s->client)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot activate client.\n");
                return false;
        }

        for(i = 0; i < desc.ch_count; ++i) {
                if (s->libjack->connect (s->client, s->libjack->port_name (s->output_port[i]), ports[i])) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot connect output port: %d.\n", i);
                        return false;
                }
        }
        free(ports);

        return true;
}

static void audio_play_jack_put_frame(void *state, const struct audio_frame *frame)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;
        assert(frame->bps == 4);
        int len = frame->data_len;

        if (len >= s->max_channel_len * frame->ch_count) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Long frame: %d!\n", frame->data_len);
                len = s->max_channel_len * frame->ch_count;
        }

        int2float((char *) s->converted, frame->data, len);
        s->buffer_fns->write(s->data, (char *) s->converted, len);
}

static void audio_play_jack_done(void *state)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;

        if (s->client != NULL) {
                s->libjack->client_close(s->client);
        }
        free(s->tmp);
        free(s->converted);
        free(s->jack_ports_pattern);
        if (s->buffer_fns) {
                s->buffer_fns->destroy(s->data);
        }

        close_libjack(s->libjack);

        free(s);
}

static const struct audio_playback_info aplay_jack_info = {
        audio_play_jack_probe,
        audio_play_jack_init,
        audio_play_jack_put_frame,
        audio_play_jack_ctl,
        audio_play_jack_reconfigure,
        audio_play_jack_done
};

REGISTER_MODULE(jack, &aplay_jack_info, LIBRARY_CLASS_AUDIO_PLAYBACK, AUDIO_PLAYBACK_ABI_VERSION);

/* vim: set expandtab sw=8: */
