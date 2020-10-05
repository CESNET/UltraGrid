/*
 * FILE:    audio/capture/jack.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif
#include "debug.h"
#include "host.h"

#include "audio/audio.h"
#include "audio/audio_capture.h"
#include "audio/utils.h"
#include "lib_common.h"
#include "utils/ring_buffer.h"
#include "jack_common.h"
#include <jack/jack.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_PORTS 64
#define MOD_NAME "[JACK capture] "

static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg);
static int jack_process_callback(jack_nframes_t nframes, void *arg);

struct state_jack_capture {
        struct audio_frame frame;
        jack_client_t *client;
        jack_port_t *input_ports[MAX_PORTS];
        char *tmp;

        struct ring_buffer *data;
        bool can_process;

        long int first_channel;
};

static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg)
{
        struct state_jack_capture *s = (struct state_jack_capture *) arg;

        s->frame.sample_rate = nframes;

        return 0;
}

static int jack_process_callback(jack_nframes_t nframes, void *arg)
{
        struct state_jack_capture *s = (struct state_jack_capture *) arg;
        int i;
        int channel_size = nframes * sizeof(int32_t);

        if (!s->can_process) {
                return 0;
        }

        for (i = 0; i < s->frame.ch_count; ++i) {
                jack_default_audio_sample_t *in = jack_port_get_buffer(s->input_ports[i], nframes);
                mux_channel(s->tmp, (char *) in, sizeof(int32_t), channel_size, s->frame.ch_count, i, 1.0);
        }

        ring_buffer_write(s->data, s->tmp, channel_size * s->frame.ch_count);

        return 0;
}

static void audio_cap_jack_probe(struct device_info **available_devices, int *count)
{
        *available_devices = audio_jack_probe(PACKAGE_STRING, JackPortIsOutput, count);
}

static void audio_cap_jack_help(const char *client_name)
{
        int count = 0;
        int i = 0;
        struct device_info *available_devices = audio_jack_probe(client_name, JackPortIsOutput, &count);

        printf("Usage:\n");
        printf("\t-s jack[:first_channel=<f>][:name=<n>][:<device>]\n");
        printf("\twhere\n");
        printf("\t\t<f> - index of first channel to capture (default: 0)\n");
        printf("\t\t<n> - name of the JACK client (default: %s)\n", PACKAGE_NAME);
        printf("\n");

        if(!available_devices)
                return;

        printf("Available devices:\n");
        for(i = 0; i < count; i++){
                printf("\t%s\n", available_devices[i].name);
        }
        free(available_devices);
}

static void * audio_cap_jack_init(const char *cfg)
{
        struct state_jack_capture *s;
        jack_status_t status;
        const char **ports;
        int i;
        char *client_name;
        const char *source_name = NULL;

        if (cfg == NULL) {
                cfg = "";
        }
        client_name = alloca(MAX(strlen(PACKAGE_NAME), strlen(cfg)) + 1);
        strcpy(client_name, PACKAGE_NAME);

        s = (struct state_jack_capture *) calloc(1, sizeof(struct state_jack_capture));
        if(!s) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to allocate memory.\n");
                return NULL;
        }

        char *dup = strdup(cfg);
        assert(dup != NULL);
        char *tmp = dup, *item, *save_ptr;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                if (strcmp(item, "help") == 0) {
                        audio_cap_jack_help(client_name);
                        free(dup);
                        free(s);
                        return &audio_init_state_ok;
                } else if (strstr(item, "first_channel=") == item) {
                        char *endptr;
                        char *val = item + strlen("first_channel=");
                        errno = 0;
                        s->first_channel = strtol(val, &endptr, 0);
                        if (errno == ERANGE || *endptr != '\0' || s->first_channel < 0) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong value '%s'.\n", val);
                                goto error;
                        }
                } else if (strstr(item, "name=") == item) {
                        strcpy(client_name, item + strlen("name="));
                } else { // this is the device name
                        source_name = cfg + (item - dup);
                        break;
                }

                tmp = NULL;
        }
        free(dup);
        dup = NULL;

        s->client = jack_client_open(client_name, JackNullOption, &status);
        if(status & JackFailure) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Opening JACK client failed.\n");
                goto error;
        }

        ports = jack_get_ports(s->client, source_name, NULL, JackPortIsOutput);
        if(ports == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to output ports matching \"%s\".\n", source_name);
                goto release_client;
        }

        i = 0;
        while(ports[i]) i++;

        s->frame.ch_count = audio_capture_channels > 0 ? audio_capture_channels : DEFAULT_AUDIO_CAPTURE_CHANNELS;
        if (i < s->frame.ch_count) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested channel count %d not found (matching pattern %s).\n",
                                s->frame.ch_count, cfg);
                goto release_client;

        }

        s->frame.bps = 4;
        if (audio_capture_sample_rate) {
                log_msg(LOG_LEVEL_WARNING, "[JACK capture] Ignoring user specified sample rate!\n");
        }
        s->frame.sample_rate = jack_get_sample_rate (s->client);
        s->frame.max_size = s->frame.ch_count * s->frame.bps * s->frame.sample_rate;
        s->frame.data = malloc(s->frame.max_size);

        s->tmp = malloc(s->frame.max_size);

        s->data = ring_buffer_init(s->frame.max_size);
        
        if(jack_set_sample_rate_callback(s->client, jack_samplerate_changed_callback, (void *) s)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Registring callback problem.\n");
                goto release_client;
        }

        if(jack_set_process_callback(s->client, jack_process_callback, (void *) s) != 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Process callback registration problem.\n");
                goto release_client;
        }

        if(jack_activate(s->client)) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot activate client.\n");
                goto release_client;
        }

        {
                int port;
                char name[32];

                for(port = 0; port < s->frame.ch_count; port++) {
                        snprintf(name, 32, "capture_%02u", port);
                        s->input_ports[port] = jack_port_register(s->client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
                        /* attach ports */
                        if(jack_connect(s->client, ports[port], jack_port_name(s->input_ports[port]))) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot connect input ports.\n");
                        }
                }
        }

        free(ports);

        s->can_process = true;

        return s;

release_client:
        jack_client_close(s->client);   
error:
        free(dup);
        free(s);
        return NULL;
}

static struct audio_frame *audio_cap_jack_read(void *state)
{
        struct state_jack_capture *s = (struct state_jack_capture *) state;

        s->frame.data_len = ring_buffer_read(s->data, s->frame.data, s->frame.max_size);
        float2int((char *) s->frame.data, (char *) s->frame.data, s->frame.max_size);

        if(!s->frame.data_len)
                return NULL;

        return &s->frame;
}

static void audio_cap_jack_done(void *state)
{
        struct state_jack_capture *s = (struct state_jack_capture *) state;

        jack_client_close(s->client);
        free(s->tmp);
        ring_buffer_destroy(s->data);
        free(s->frame.data);
        free(s);
}

static const struct audio_capture_info acap_jack_info = {
        audio_cap_jack_probe,
        audio_cap_jack_help,
        audio_cap_jack_init,
        audio_cap_jack_read,
        audio_cap_jack_done
};

REGISTER_MODULE(jack, &acap_jack_info, LIBRARY_CLASS_AUDIO_CAPTURE, AUDIO_CAPTURE_ABI_VERSION);

