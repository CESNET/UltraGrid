/*
 * FILE:    audio/jack.c
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
#include "audio/jack.h"
#include "config_unix.h"
#include "pthread.h"
#include <jack/jack.h>
#include <stdlib.h>
#include <stdio.h>
#include <limits.h>
#include <string.h>

#define CLIENT_NAME "UltraGrid Transport"
#define BUFF_ELEM (1<<16)
#define BUFF_SIZE (BUFF_ELEM * sizeof(float))
#define MAX_PORTS 8

struct state_jack {
        unsigned int sender:1,
                        receiver:1;
                        
        jack_client_t *client;
        
        struct audio_frame record;
/* TODO: consider using ring buffers abstraction instead implementing
 * by hand (see utils/ring_buffer.h) */
        char *play_buffer[MAX_PORTS];
        int play_buffer_start, play_buffer_end;
        char *rec_buffer;
        volatile int rec_buffer_start, rec_buffer_end;
        
        jack_port_t *input_port[MAX_PORTS];
        jack_port_t *output_port[MAX_PORTS];
        
        char *in_port_pattern;
        int in_ch_count; /* "requested", real received count is in
                          * record.ch_count */
        char *out_port_pattern;
        int out_channel_count; /* really obtained playback channels ( <= _req) */
        int out_channel_count_req; /* requested playback ch. count */
};

int jack_process_callback(jack_nframes_t nframes, void *arg);
int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg);
void reconfigure_send_ch_count(struct state_jack *s, int ch_count);
static int settings_init(struct state_jack *s, char *cfg);

int jack_process_callback(jack_nframes_t nframes, void *arg) {
        struct state_jack *s = (struct state_jack *) arg;
        int send_b;
        int i;
        
        send_b = s->play_buffer_end - s->play_buffer_start;
        if(send_b < 0) send_b += BUFF_SIZE;
        if(send_b > (int) (sizeof (jack_default_audio_sample_t) * nframes))
                send_b = sizeof (jack_default_audio_sample_t) * nframes;

        for (i = 0; i < s->out_channel_count; ++i) {
                
                int to_end = BUFF_SIZE - s->play_buffer_start;
                jack_default_audio_sample_t *out =
                        jack_port_get_buffer (s->output_port[i], nframes);
                if(to_end > send_b) {
                        memcpy (out, s->play_buffer[i] + s->play_buffer_start,
                                send_b);
                } else {
                        memcpy (out, s->play_buffer[i] + s->play_buffer_start,
                                to_end);
                        memcpy (out + to_end, s->play_buffer[i],
                                send_b - to_end);
                }
                
        }
        s->play_buffer_start = (s->play_buffer_start + send_b) % BUFF_SIZE;
        
        for(i = 0; i < s->record.ch_count; ++i) {
                int j;
                jack_default_audio_sample_t *in =
                        jack_port_get_buffer (s->input_port[i], nframes);
                for(j = 0; j < (int) nframes; ++j) {
                        *(int *)(s->rec_buffer + ((s->rec_buffer_end + (j * s->record.ch_count + i) * sizeof(int32_t)) % BUFF_SIZE)) =
                                        in[j] * INT_MAX;
                }
        }
        s->rec_buffer_end = (s->rec_buffer_end + nframes * s->record.ch_count * sizeof(int32_t)) % BUFF_SIZE;
        
        //fprintf(stderr, ".%d.", nframes);
        
        return 0;
}

int jack_samplerate_changed_callback(jack_nframes_t nframes, void *arg) {
        struct state_jack *s = (struct state_jack *) arg;
        
        s->record.sample_rate = nframes;
        return 0;
}

void reconfigure_send_ch_count(struct state_jack *s, int ch_count)
{
        const char **ports;
        int i;

        s->out_channel_count = s->out_channel_count_req = ch_count;

        if ((ports = jack_get_ports (s->client, s->out_port_pattern, NULL, JackPortIsInput)) == NULL) {
                fprintf(stderr, "Cannot find any ports matching pattern '%s'\n", s->out_port_pattern);
                s->out_channel_count = 0;
                return;
        }
        for (i = 0; i < s->record.ch_count; ++i) {
                jack_disconnect(s->client, jack_port_name (s->output_port[i]), ports[i]);
                free(s->play_buffer[i]);
        }

        i = 0;
        while (ports[i]) ++i;

        if(i < s->out_channel_count) {
                fprintf(stderr, "Not enought output ports found matching pattern '%s': "
                                "%d requested, %d found\n", s->out_port_pattern, s->record.ch_count, i);
                fprintf(stderr, "Reducing port count to %d\n", i);
                s->out_channel_count = i;
        }
         
        for(i = 0; i < s->out_channel_count; ++i) {
                fprintf(stderr, "%s\n\n\n", ports[i]);
                if (jack_connect (s->client, jack_port_name (s->output_port[i]), ports[i])) {
                        fprintf (stderr, "cannot connect output ports\n");
                }
                s->play_buffer[i] = malloc(BUFF_SIZE);
        }
        
        fprintf(stderr, "[JACK] Sending %d output audio streams (ports).\n", s->out_channel_count);
 
        free (ports);
}

static int settings_init(struct state_jack *s, char *cfg)
{
        char * save_ptr = NULL,
                *tok;
        if(!cfg) /* no explicit config */
                return 0;
                
        if(strncmp(cfg, "help", strlen("help")) == 0)
        {
                printf("JACK config:\n"
                        "\t-j ({tok1|tok2...tokn-1},)*{tokn}\n\n"
                        "\tTokens:\n"
                        "\t\tpi[=<name>]\t\treceive sound from JACK instead of from RTP (optionally select port)\n"
                        "\n"
                        "\t\tpo[=<name>]\t\tsend sound output via JACK with (optional) port name.\n"
                        "\t\t\t\t\tIf name not specified, let Jack select (probably system output!)\n"
                        "\n"
                        "\t\tch=<count>\t\tcount of input (!) JACK ports to listen at (matching RE above)\n"
                        "\t\t\t\t\tYou should set it for input, otherwise the count defaults to 0.\n"
                        );
                exit(0);
        }
        while((tok = strtok_r(cfg, ",", &save_ptr)) != NULL)
        {
                switch (tok[0]) {
                        case 'p':
                                if(tok[1] != 'i' && tok[1] != 'o')
                                        return -1;
                                if(tok[1] == 'i') {
                                        s->receiver = TRUE;
                                        if(tok[2] == '=')
                                                s->in_port_pattern = strdup(&tok[3]);
                                }
                                if(tok[1] == 'o') {
                                        s->sender = TRUE;
                                        if(tok[2] == '=')
                                                s->out_port_pattern = strdup(&tok[3]);
                                }
                                break;
                        case 'c':
                                if(strncmp(tok, "ch=", 3))
                                        return -1;
                                else
                                        s->in_ch_count = atoi(tok + 3);
                                break;        
                        default:
                                return -1;
                }
                cfg = NULL;
        }
        return 0;
}

static int attach_input_ports(struct state_jack *s)
{
        int i = 0;
        const char **ports;
        if ((ports = jack_get_ports (s->client, s->in_port_pattern, NULL, JackPortIsOutput)) == NULL) {
                 fprintf(stderr, "Cannot find any ports matching pattern '%s'\n", s->in_port_pattern);
                 return FALSE;
         }
         
         while (ports[i]) ++i;
         if(i < s->record.ch_count) {
                 fprintf(stderr, "Not enought input ports found matching pattern '%s': "
                                "%d requested, %d found\n", s->in_port_pattern, s->record.ch_count, i);
                fprintf(stderr, "Reducing port count to %d\n", i);
                s->record.ch_count = i;
         }
         
         for(i = 0; i < s->in_ch_count; ++i) {
                 if (jack_connect (s->client, ports[i], jack_port_name (s->input_port[i]))) {
                         fprintf (stderr, "cannot connect input ports\n");
                 }
         }
 
         free (ports);
         return TRUE;
}

void * jack_start(char *cfg)
{
        struct state_jack *s;
         
        s = (struct state_jack *) malloc(sizeof(struct state_jack));
        
        s->in_port_pattern = NULL;
        s->out_port_pattern = NULL;
        s->in_ch_count = 1;
        s->play_buffer_start = s->play_buffer_end = 0;
        s->rec_buffer_start = s->rec_buffer_end = 0;
        s->sender = FALSE;
        s->receiver = FALSE;
        s->out_channel_count = 0;
        
        if (settings_init(s, cfg)) {
                fprintf(stderr, "Setting JACK failed. Check configuration ('-j' option).\n");
                free(s);
                return NULL;
        }
        
        if(!s->sender && !s->receiver) {
                free(s);
                return NULL;
        }
        
        s->client = jack_client_open(CLIENT_NAME, JackNullOption, NULL);
        if(jack_set_process_callback(s->client, jack_process_callback, (void *) s)  != 0) {
                fprintf(stderr, "[jack] Callback initialization problem.\n");
                goto error;
        }
        
        if(jack_set_sample_rate_callback(s->client,
		jack_samplerate_changed_callback, (void *) s)) {
                        fprintf(stderr, "[jack] Callback initialization problem.\n");
                        goto error;
	}
        
        //jack_on_shutdown (client, jack_shutdown, 0);
        
        if(s->sender) {
                char name[30];
                int i;
                
                for(i = 0; i < MAX_PORTS; ++i) {
                        snprintf(name, 30, "out_%02u", i);
                        s->output_port[i] = jack_port_register (s->client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsOutput, 0);
                }
                
                s->out_channel_count = s->out_channel_count_req = 0;
        }
        
        if(s->receiver) {
                char name[30];
                int i;
                
                for(i = 0; i < s->in_ch_count; ++i) {
                        snprintf(name, 30, "in_%02u", i);
                        s->input_port[i] = jack_port_register (s->client, name, JACK_DEFAULT_AUDIO_TYPE, JackPortIsInput, 0);
                }
                
                s->record.sample_rate = jack_get_sample_rate (s->client);
                s->record.bps = sizeof(int32_t);
                s->record.ch_count = s->in_ch_count;
                s->rec_buffer = s->record.data = (void *) malloc(BUFF_SIZE);
        }
        
        if (jack_activate (s->client)) {
                 fprintf (stderr, "cannot activate client");
                 goto error;
        }
         
        if(s->receiver) {
                if(!attach_input_ports(s))
                        goto error;
         }
        
        return s;
error:        
        return NULL;
}

void jack_send(void *state, struct audio_frame *frame)
{
        //float *tmp = (char *) s->play_buffer + s->play_buffer_end;
        struct state_jack *s = (struct state_jack *) state;
        char *in_ptr = frame->data;
        
        int i;
        
        if(s->out_channel_count != frame->ch_count)
                reconfigure_send_ch_count(s, frame->ch_count);
                
        for(i = 0; i < frame->data_len; i+= frame->bps * frame->ch_count) {
                int channels;
                for (channels = 0; channels < s->out_channel_count; ++channels) {
                        float *pos = (float *) &s->play_buffer[channels][(s->play_buffer_end + i * 4 / (frame->ch_count * frame->bps)) % BUFF_SIZE];
                                        *pos = (*((int *) (in_ptr + channels * frame->bps)) << (32 - frame->bps * 8)) / (float) INT_MAX;
                }
                
                in_ptr += frame->bps * frame->ch_count;
        }
        s->play_buffer_end = (s->play_buffer_end + frame->data_len / (frame->bps * frame->ch_count) * sizeof(float)) % BUFF_SIZE;
}

void jack_receive(void *state, struct audio_frame *buffer)
{
        struct state_jack *s = (struct state_jack *) state;
        
        while(s->rec_buffer_start == s->rec_buffer_end);
        //fprintf(stderr, "%d ", s->record.data_len);
        int end = s->rec_buffer_end;
             
        if(buffer->ch_count != s->record.ch_count ||
                        buffer->bps != s->record.bps ||
                        buffer->sample_rate != s->record.sample_rate) {
                buffer->reconfigure_audio(buffer->state, s->record.bps * 8, s->record.ch_count,
                        s->record.sample_rate);
        }
        
        buffer->data_len = end - s->rec_buffer_start;
        if(buffer->data_len < 0) buffer->data_len += BUFF_SIZE;
        if (s->rec_buffer_start < end) {
                memcpy(buffer->data, s->rec_buffer + s->rec_buffer_start, 
                                buffer->data_len);
        } else {
                memcpy(buffer->data, s->rec_buffer + s->rec_buffer_start, 
                                BUFF_SIZE - s->rec_buffer_start);
                memcpy(buffer->data + BUFF_SIZE - s->rec_buffer_start,
                                s->rec_buffer, 
                                buffer->data_len + BUFF_SIZE - s->rec_buffer_start);
        }
        s->rec_buffer_start = (s->rec_buffer_start + buffer->data_len) % BUFF_SIZE;
}

int is_jack_sender(void *state)
{
        return ((struct state_jack *) state)->sender;
}

int is_jack_receiver(void *state)
{
        return ((struct state_jack *) state)->receiver;
}

