/*
 * FILE:    audio/playback/jack.c
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

#include "audio/audio.h"
#include "audio/playback/jack.h" 
#include "audio/utils.h"
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif
#include "debug.h"
#include "host.h"
#include "utils/ring_buffer.h"

#include <jack/jack.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifdef HAVE_SPEEX
#include <speex/speex_resampler.h> 
#endif

#define MAX_PORTS 64

struct state_jack_playback {
        const char *jack_ports_pattern;
        int jack_sample_rate;
        jack_client_t *client;
        jack_port_t *output_port[MAX_PORTS];
        struct audio_frame frame;
        char *channel;
        char *converted;
#ifdef HAVE_SPEEX
        char *converted_resampled;
        SpeexResamplerState *resampler; 
#endif


        int jack_ports_count;
        struct ring_buffer *data[MAX_PORTS];
};

static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg);
static int jack_process_callback(jack_nframes_t nframes, void *arg);

static int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg)
{
        struct state_jack_playback *s = (struct state_jack_playback *) arg;

        s->jack_sample_rate = nframes;

        return 0;
}

static int jack_process_callback(jack_nframes_t nframes, void *arg)
{
        struct state_jack_playback *s = (struct state_jack_playback *) arg;
        int i;

        if(should_exit)
                return 1;

        for (i = 0; i < s->jack_ports_count; ++i) {
                if(ring_get_current_size(s->data[i]) <= (int) (nframes * sizeof(float))) {
                        fprintf(stderr, "[JACK playback] Buffer underflow detected.\n");
                        return 0;
                }
        }
        for (i = 0; i < s->jack_ports_count; ++i) {
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



void audio_play_jack_help(void)
{
        jack_client_t *client;
        jack_status_t status;
        char *last_name = NULL;
        int i;
        int channel_count;
        const char **ports;

        client = jack_client_open(PACKAGE_STRING, JackNullOption, &status);
        if(status == JackFailure) {
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


void * audio_play_jack_init(char *cfg)
{
        struct state_jack_playback *s;
        const char **ports;
        jack_status_t status;

        if(!cfg || strcmp(cfg, "help") == 0) {
                audio_play_jack_help();
                return NULL;
        }

        s = calloc(1, sizeof(struct state_jack_playback));

        s->frame.data = NULL;
        s->jack_ports_pattern = cfg;

        if(!s) {
                fprintf(stderr, "[JACK playback] Unable to allocate memory.\n");
                goto error;
        }

        s->client = jack_client_open(PACKAGE_STRING, JackNullOption, &status);
        if(status == JackFailure) {
                fprintf(stderr, "[JACK playback] Opening JACK client failed.\n");
                goto error;
        }

        if(jack_set_sample_rate_callback(s->client, jack_samplerate_changed_callback, (void *) s)) {
                fprintf(stderr, "[JACK capture] Registring callback problem.\n");
                goto release_client;
        }


        if(jack_set_process_callback(s->client, jack_process_callback, (void *) s) != 0) {
                fprintf(stderr, "[JACK capture] Process callback registration problem.\n");
                goto release_client;
        }

        s->jack_sample_rate = jack_get_sample_rate (s->client);



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

int audio_play_jack_reconfigure(void *state, int quant_samples, int channels,
                                int sample_rate)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;
        const char **ports;
        int i;

        jack_deactivate(s->client);

        ports = jack_get_ports(s->client, s->jack_ports_pattern, NULL, JackPortIsInput);
        if(ports == NULL) {
                fprintf(stderr, "[JACK playback] Unable to input ports matching %s.\n", s->jack_ports_pattern);
                return FALSE;
        }

        if(channels > s->jack_ports_count) {
                fprintf(stderr, "[JACK playback] Warning: received %d audio channels, JACK can process only %d.", channels, s->jack_ports_count);
        }

        for(i = 0; i < MAX_PORTS; ++i) {
                ring_buffer_destroy(s->data[i]);
                s->data[i] = NULL;
        }
        /* for all channels previously connected */
        for(i = 0; i < s->frame.ch_count; ++i) {
                jack_disconnect(s->client, jack_port_name (s->output_port[i]), ports[i]);
        }
        free(s->frame.data);
        free(s->channel);
        free(s->converted);
        s->frame.bps = quant_samples / 8;
        s->frame.ch_count = channels;
        s->frame.sample_rate = sample_rate;

        s->frame.max_size = s->frame.bps * s->frame.ch_count * s->frame.sample_rate;
        s->frame.data = malloc(s->frame.max_size);

#ifdef HAVE_SPEEX
        free(s->converted_resampled);
        if(s->resampler) {
                speex_resampler_destroy(s->resampler);
        }
        s->converted_resampled = malloc(sizeof(float) * s->jack_sample_rate);

        {
                int err;
                s->resampler = speex_resampler_init(channels, sample_rate, s->jack_sample_rate, 10, &err); 
                if(err) {
                        fprintf(stderr, "[JACK playback] Unable to create resampler.\n");
                        return FALSE;
                }
        }
#endif

        s->channel = malloc(s->frame.bps * sample_rate);
        s->converted = malloc(sample_rate * sizeof(float));

        for(i = 0; i < channels; ++i) {
                s->data[i] = ring_buffer_init(sizeof(float) * s->jack_sample_rate);
        }

        if(jack_activate(s->client)) {
                fprintf(stderr, "[JACK capture] Cannot activate client.\n");
                return FALSE;
        }

        for(i = 0; i < channels; ++i) {
                if (jack_connect (s->client, jack_port_name (s->output_port[i]), ports[i])) {
                        fprintf (stderr, "Cannot connect output port: %d.\n", i);
                        return FALSE;
                }
        }
        free(ports);

        return TRUE;
}

struct audio_frame *audio_play_jack_get_frame(void *state)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;

        return &s->frame;
}

void audio_play_jack_put_frame(void *state, struct audio_frame *frame)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;
        int i;

        int channel_size = frame->data_len / frame->ch_count;
        int converted_size = channel_size * sizeof(int32_t) / frame->bps;


        for(i = 0; i < frame->ch_count; ++i) {
                demux_channel(s->channel, frame->data, frame->bps, frame->data_len, frame->ch_count, i);
                change_bps(s->converted, sizeof(int32_t), s->channel, frame->bps, channel_size);
                int2float(s->converted, s->converted, converted_size);
#ifdef HAVE_SPEEX
                spx_uint32_t in_len = channel_size / frame->bps;
                spx_uint32_t out_len;
                speex_resampler_process_float(s->resampler, 
                                           i, 
                                           (float *) s->converted, 
                                           &in_len, 
                                           (float *) s->converted_resampled, 
                                           &out_len);
                out_len *= sizeof(float);
                ring_buffer_write(s->data[i], s->converted_resampled, out_len);
#else
                ring_buffer_write(s->data[i], s->converted, converted_size);
#endif
        }
}

void audio_play_jack_done(void *state)
{
        struct state_jack_playback *s = (struct state_jack_playback *) state;
        int i;

        jack_client_close(s->client);
#ifdef HAVE_SPEEX
        free(s->converted_resampled);
        speex_resampler_destroy(s->resampler);
#endif
        free(s->channel);
        free(s->converted);
        for(i = 0; i < MAX_PORTS; ++i) {
                ring_buffer_destroy(s->data[i]);
        }

        free(s);
}

