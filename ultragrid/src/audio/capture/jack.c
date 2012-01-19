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
#endif
#include "debug.h"


#include "audio/audio.h"
#include "audio/utils.h"
#include "audio/capture/jack.h" 
#include "utils/ring_buffer.h"
#include <jack/jack.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#define MAX_PORTS 64

static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg);
static int jack_process_callback(jack_nframes_t nframes, void *arg);

struct state_jack_capture {
        struct audio_frame frame;
        jack_client_t *client;
        jack_port_t *input_ports[MAX_PORTS];
        char *tmp;

        struct ring_buffer *data;
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

        for (i = 0; i < s->frame.ch_count; ++i) {
                jack_default_audio_sample_t *in = jack_port_get_buffer(s->input_ports[i], nframes);
                float2int((char *) in, (char *) in, channel_size);
                mux_channel(s->tmp, (char *) in, sizeof(int32_t), channel_size, s->frame.ch_count, i);
        }

        ring_buffer_write(s->data, s->tmp, channel_size * s->frame.ch_count);

        return 0;
}

void audio_cap_jack_help(void)
{
        jack_client_t *client;
        jack_status_t status;
        const char **ports;
        char *last_name = NULL;
        int i;
        int channel_count;

        client = jack_client_open(PACKAGE_STRING, JackNullOption, &status);
        if(status == JackFailure) {
                fprintf(stderr, "[JACK capture] Opening JACK client failed.\n");
                return;
        }

        ports = jack_get_ports(client, NULL, NULL, JackPortIsOutput);
        if(ports == NULL) {
                fprintf(stderr, "[JACK capture] Unable to enumerate ports.\n");
                return;
        }

        printf("JACK capture:\n");
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


void * audio_cap_jack_init(char *cfg)
{
        struct state_jack_capture *s;
        jack_status_t status;
        const char **ports;
        int i;



        if(!cfg || strcmp(cfg, "help") == 0) {
                audio_cap_jack_help();
                return NULL;
        }


        s = (struct state_jack_capture *) calloc(1, sizeof(struct state_jack_capture));
        if(!s) {
                fprintf(stderr, "[JACK capture] Unable to allocate memory.\n");
                goto error;
        }

        s->client = jack_client_open(PACKAGE_STRING, JackNullOption, &status);
        if(status == JackFailure) {
                fprintf(stderr, "[JACK capture] Opening JACK client failed.\n");
                goto error;
        }

        ports = jack_get_ports(s->client, cfg, NULL, JackPortIsOutput);
        if(ports == NULL) {
                fprintf(stderr, "[JACK capture] Unable to output ports matching %s.\n", cfg);
                goto release_client;
        }

        i = 0;
        while(ports[i]) i++;

        s->frame.ch_count = i;
        s->frame.bps = 4;
        s->frame.sample_rate = jack_get_sample_rate (s->client);
        s->frame.max_size = s->frame.ch_count * s->frame.bps * s->frame.sample_rate;
        s->frame.data = malloc(s->frame.max_size);

        s->tmp = malloc(s->frame.max_size);

        s->data = ring_buffer_init(s->frame.max_size);
        
        free(ports);

        if(jack_set_sample_rate_callback(s->client, jack_samplerate_changed_callback, (void *) s)) {
                fprintf(stderr, "[JACK capture] Registring callback problem.\n");
                goto release_client;
        }

        if(jack_set_process_callback(s->client, jack_process_callback, (void *) s) != 0) {
                fprintf(stderr, "[JACK capture] Process callback registration problem.\n");
                goto release_client;
        }

        if(jack_activate(s->client)) {
                fprintf(stderr, "[JACK capture] Cannot activate client.\n");
                goto release_client;
        }

        {
                int port;
                char name[32];

                for(port = 0; port < i; port++) {
                        snprintf(name, 32, "capture_%02u", port);
                        s->input_ports[port] = jack_port_register(s->client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
                        /* attach ports */
                        if(jack_connect(s->client, ports[port], jack_port_name(s->input_ports[port]))) {
                                fprintf(stderr, "[JACK capture] Cannot connect input ports.\n");
                        }
                }
        }

        return s;

release_client:
        jack_client_close(s->client);   
error:
        free(s);
        return NULL;
}

struct audio_frame *audio_cap_jack_read(void *state)
{
        struct state_jack_capture *s = (struct state_jack_capture *) state;

        s->frame.data_len = ring_buffer_read(s->data, s->frame.data, s->frame.max_size);

        if(!s->frame.data_len)
                return NULL;

        return &s->frame;
}

void audio_cap_jack_finish(void *state)
{
        UNUSED(state);
}

void audio_cap_jack_done(void *state)
{
        struct state_jack_capture *s = (struct state_jack_capture *) state;

        jack_client_close(s->client);
        free(s->tmp);
        ring_buffer_destroy(s->data);
        free(s->frame.data);
        free(s);
}

