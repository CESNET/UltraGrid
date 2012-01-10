/*
 * FILE:    audio/audio.c
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
#include "audio/capture/portaudio.h" 
#include "audio/capture/alsa.h" 
#include "audio/playback/alsa.h" 
#include "audio/capture/coreaudio.h" 
#include "audio/playback/coreaudio.h" 
#include "audio/capture/none.h" 
#include "audio/playback/none.h" 
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "tv.h"
#include "transmit.h"
#include "pdb.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define EXIT_FAIL_USAGE		1
#define EXIT_FAIL_NETWORK	5

struct audio_device_t {
        int index;
        void *state;
};

enum audio_transport_device {
        NET_NATIVE,
        NET_JACK
};

struct state_audio {
        struct audio_device_t audio_capture_device;
        struct audio_device_t audio_playback_device;
        
        struct rtp *audio_network_device;
        struct pdb *audio_participants;
        void *jack_connection;
        enum audio_transport_device sender;
        enum audio_transport_device receiver;
        
        struct timeval start_time;

        struct tx *tx_session;
        
        pthread_t audio_sender_thread_id,
                  audio_receiver_thread_id;
};

struct state_sdi_capture {
        struct audio_frame * audio_buffer;
        sem_t audio_frame_ready;
};

struct state_sdi_playback {
        struct audio_frame * (*get_callback)(void *);
        void (*put_callback)(void *, struct audio_frame *);
        void *get_udata;
        void *put_udata;
};

void print_audio_capture_devices(void);
void print_audio_playback_devices(void);

/** 
 * Copies one input channel into n output (interlaced).
 * 
 * Input and output data may overlap. 
 */
void copy_channel(char *out, const char *in, int bps, int in_len /* bytes */, int out_channel_count); 


typedef void (*audio_device_help_t)(void);
/**
 * @return state
 */
typedef void * (*audio_init_t)(char *cfg);
typedef struct audio_frame* (*audio_read_t)(void *state);
typedef void (*audio_finish_t)(void *state);
typedef void (*audio_done_t)(void *state);

/* playback */
typedef struct audio_frame* (*audio_get_frame_t)(void *state);
typedef void (*audio_put_frame_t)(void *state, struct audio_frame *frame);
typedef void (*audio_playback_done_t)(void *s);

struct audio_capture_t {
        const char *name;
        audio_device_help_t audio_help;
        audio_init_t audio_init;
        audio_read_t audio_read;
        audio_finish_t audio_capture_finish;
        audio_done_t audio_capture_done;
};

struct audio_playback_t {
        const char *name;
        audio_device_help_t audio_help;
        audio_init_t audio_init;
        audio_get_frame_t audio_get_frame;
        audio_put_frame_t audio_put_frame;
        audio_playback_done_t playback_done;
};

void sdi_capture_help(void);
void sdi_playback_help(void);
void * sdi_capture_init(char *cfg);
void * sdi_capture_finish(void *state);
void * sdi_capture_done(void *state);
void * sdi_playback_init(char *cfg);
void sdi_playback_done(void *s);
struct audio_frame * sdi_read(void *state);
static void *audio_sender_thread(void *arg);
static void *audio_receiver_thread(void *arg);
static struct rtp *initialize_audio_network(char *addr, struct pdb *participants);
void print_audio_capture_devices();
void print_audio_playback_devices();

static struct audio_capture_t audio_capture[] = {
        { "embedded", sdi_capture_help, sdi_capture_init, sdi_read, sdi_capture_finish, sdi_capture_done },
#ifdef HAVE_ALSA
        { "alsa", audio_cap_alsa_help, audio_cap_alsa_init, audio_cap_alsa_read, audio_cap_alsa_finish, audio_cap_alsa_done },
#endif
#ifdef HAVE_COREAUDIO
        { "coreaudio", audio_cap_ca_help, audio_cap_ca_init, audio_cap_ca_read, audio_cap_ca_finish, audio_cap_ca_done },
#endif
#ifdef HAVE_PORTAUDIO
        { "portaudio", portaudio_capture_help, portaudio_capture_init, portaudio_read, portaudio_capture_finish, portaudio_capture_done },
#endif
        { "none", audio_cap_none_help, audio_cap_none_init, audio_cap_none_read, audio_cap_none_finish, audio_cap_none_done },
        { NULL, NULL, NULL, NULL, NULL, NULL }
};

static struct audio_playback_t audio_playback[] = {
        { "embedded", sdi_playback_help, sdi_playback_init, sdi_get_frame, sdi_put_frame, sdi_playback_done },
#ifdef HAVE_ALSA
        { "alsa", audio_play_alsa_help, audio_play_alsa_init, audio_play_alsa_get_frame, audio_play_alsa_put_frame, audio_play_alsa_done },
#endif
#ifdef HAVE_COREAUDIO
        { "coreaudio", audio_play_ca_help, audio_play_ca_init, audio_play_ca_get_frame, audio_play_ca_put_frame, audio_play_ca_done },
#endif
#ifdef HAVE_PORTAUDIO
        { "portaudio", portaudio_playback_help, portaudio_playback_init, portaudio_get_frame, portaudio_put_frame, portaudio_close_playback },
#endif
        { "none", audio_play_none_help, audio_play_none_init, audio_play_none_get_frame, audio_play_none_put_frame, audio_play_none_done },
        { NULL, NULL, NULL, NULL, NULL, NULL }
};

void sdi_capture_help(void)
{
        printf("\tembedded : SDI audio (if available)\n");
}

void sdi_playback_help(void)
{
        printf("\tembedded : SDI audio (if available)\n");
}

void print_audio_capture_devices()
{
        int i;
        printf("Available audio capture devices:\n");
        for (i = 0; audio_capture[i].name != NULL; ++i) {
                audio_capture[i].audio_help();
                printf("\n");
        }
}

void print_audio_playback_devices()
{
        int i;
        printf("Available audio playback devices:\n");
        for (i = 0; audio_playback[i].name != NULL; ++i) {
                audio_playback[i].audio_help();
                printf("\n");
        }
}

/**
 * take care that addrs can also be comma-separated list of addresses !
 */
struct state_audio * audio_cfg_init(char *addrs, char *send_cfg, char *recv_cfg, char *jack_cfg)
{
        struct state_audio *s = NULL;
        char *tmp, *unused = NULL;
        char *addr;
        int i;
        
        if (send_cfg != NULL &&
                        !strcmp("help", send_cfg)) {
                print_audio_capture_devices();
                exit_uv(0);
                return NULL;
        }
        
        if (recv_cfg != NULL &&
                        !strcmp("help", recv_cfg)) {
                print_audio_playback_devices();
                exit_uv(0);
                return NULL;
        }
        
        s = calloc(1, sizeof(struct state_audio));
        s->audio_participants = NULL;
        
        s->tx_session = tx_init(1500, NULL);
        gettimeofday(&s->start_time, NULL);        
        
        tmp = strdup(addrs);
        s->audio_participants = pdb_init();
        addr = strtok_r(tmp, ",", &unused);
        if ((s->audio_network_device =
             initialize_audio_network(addr,
                                      s->audio_participants)) ==
            NULL) {
                printf("Unable to open audio network\n");
                goto error;
        }
        free(tmp);

        if (send_cfg != NULL) {
                char *tmp = strtok(send_cfg, ":");
                for (i = 0; audio_capture[i].name != NULL; ++i) {
                        if(strcmp(tmp, audio_capture[i].name) == 0) {
                                s->audio_capture_device.index = i;
                                break;
                        }
                }

                if(audio_capture[i].name == NULL) {
                        fprintf(stderr, "Unknown audio driver: %s\n", tmp);
                        goto error;
                }
                
                tmp = strtok(NULL, ":");
                s->audio_capture_device.state =
                        audio_capture[s->audio_capture_device.index].audio_init(tmp);
                
                if(!s->audio_capture_device.state) {
                        fprintf(stderr, "Error initializing audio capture.\n");
                        goto error;
                }
        } else {
                for (i = 0; audio_capture[i].name != NULL; ++i) {
                        if(strcmp("none", audio_capture[i].name) == 0) {
                                s->audio_capture_device.index = i;
                        }
                }
        }
        
        if (recv_cfg != NULL) {
                char *tmp = strtok(recv_cfg, ":");
                for (i = 0; audio_playback[i].name != NULL; ++i) {
                        if(strcmp(tmp, audio_playback[i].name) == 0) {
                                s->audio_playback_device.index = i;
                                break;
                        }
                }
                if(audio_playback[i].name == NULL) {
                        fprintf(stderr, "Unknown audio driver: %s\n", tmp);
                        goto error;
                }
                
                tmp = strtok(NULL, ":");
                s->audio_playback_device.state =
                        audio_playback[s->audio_playback_device.index].audio_init(tmp);
                if(!s->audio_playback_device.state) {
                        fprintf(stderr, "Error initializing audio playback.\n");
                        goto error;
                }
        } else {
                for (i = 0; audio_playback[i].name != NULL; ++i) {
                        if(strcmp("none", audio_playback[i].name) == 0) {
                                s->audio_playback_device.index = i;
                        }
                }
        }

        if (send_cfg != NULL) {
                if (pthread_create
                    (&s->audio_sender_thread_id, NULL, audio_sender_thread, (void *)s) != 0) {
                        fprintf(stderr,
                                "Error creating audio thread. Quitting\n");
                        goto error;
                }
        }

        if (recv_cfg != NULL) {
                if (pthread_create
                    (&s->audio_receiver_thread_id, NULL, audio_receiver_thread, (void *)s) != 0) {
                        fprintf(stderr,
                                "Error creating audio thread. Quitting\n");
                        goto error;
                }
        }
        
        s->sender = NET_NATIVE;
        s->receiver = NET_NATIVE;
        
#ifdef HAVE_JACK
        s->jack_connection = jack_start(jack_cfg);
        if(s->jack_connection) {
                if(is_jack_sender(s->jack_connection))
                        s->sender = NET_JACK;
                if(is_jack_receiver(s->jack_connection))
                        s->receiver = NET_JACK;
        }
#else
        if(jack_cfg) {
                fprintf(stderr, "[Audio] JACK configuration string entered ('-j'), "
                                "but JACK support isn't compiled.\n");
                goto error;
        }
#endif


        return s;

error:
        if(s->tx_session)
                tx_done(s->tx_session);
        if(s->audio_participants)
                pdb_destroy(&s->audio_participants);
        free(s);
        exit_uv(1);
        return NULL;
}

void audio_join(struct state_audio *s) {
        if(s) {
                if(s->audio_receiver_thread_id)
                        pthread_join(s->audio_receiver_thread_id, NULL);
                if(s->audio_sender_thread_id)
                        pthread_join(s->audio_sender_thread_id, NULL);
        }
}
        
void audio_finish(struct state_audio *s)
{
        if(s) {
                audio_capture[s->audio_capture_device.index].audio_capture_finish(s->audio_capture_device.state);
        }
}

void audio_done(struct state_audio *s)
{
        if(s) {
                if(s->audio_participants)
                        pdb_destroy(&s->audio_participants);
                audio_playback[s->audio_playback_device.index].playback_done(s->audio_playback_device.state);
                audio_capture[s->audio_capture_device.index].audio_capture_done(s->audio_capture_device.state);
                tx_done(s->tx_session);
                free(s);
        }
}

void * sdi_capture_init(char *cfg)
{
        struct state_sdi_capture *s;
        UNUSED(cfg);
        
        s = (struct state_sdi_capture *) calloc(1, sizeof(struct state_sdi_capture));
        platform_sem_init(&s->audio_frame_ready, 0, 0);
        
        return s;
}

void * sdi_playback_init(char *cfg)
{
        struct state_sdi_playback *s = calloc(1, sizeof(struct state_sdi_playback));
        UNUSED(cfg);
        s->get_callback = NULL;
        s->put_callback = NULL;
        return s;
}


struct audio_frame * sdi_read(void *state)
{
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) state;
        platform_sem_wait(&s->audio_frame_ready);
        if(!should_exit)
                return s->audio_buffer;
        else
                return NULL;
}

static struct rtp *initialize_audio_network(char *addr, struct pdb *participants)       // GiX
{
        struct rtp *r;
        double rtcp_bw = 1024 * 512;    // FIXME:  something about 5% for rtcp is said in rfc

        r = rtp_init(addr, PORT_AUDIO, PORT_AUDIO, 255, rtcp_bw, FALSE, rtp_recv_callback,
                     (void *)participants);
        if (r != NULL) {
                pdb_add(participants, rtp_my_ssrc(r));
                rtp_set_option(r, RTP_OPT_WEAK_VALIDATION, TRUE);
                rtp_set_sdes(r, rtp_my_ssrc(r), RTCP_SDES_TOOL,
                             PACKAGE_STRING, strlen(PACKAGE_VERSION));
        }

        return r;
}

static void *audio_receiver_thread(void *arg)
{
        struct state_audio *s = arg;
        // rtp variables
        struct timeval timeout, curr_time;
        uint32_t ts;
        struct pdb_e *cp;
        struct audio_frame *frame;
        
        frame = audio_playback[s->audio_playback_device.index].audio_get_frame(
                        s->audio_playback_device.state);
                
        printf("Audio receiving started.\n");
        while (!should_exit) {
                if(s->receiver == NET_NATIVE) {
                        gettimeofday(&curr_time, NULL);
                        ts = tv_diff(curr_time, s->start_time) * 90000;
                        rtp_update(s->audio_network_device, curr_time);
                        rtp_send_ctrl(s->audio_network_device, ts, 0, curr_time);
                        timeout.tv_sec = 0;
                        timeout.tv_usec = 999999 / 59.94; /* audio goes almost always at the same rate
                                                             as video frames */
                        rtp_recv_r(s->audio_network_device, &timeout, ts);
                        cp = pdb_iter_init(s->audio_participants);
                
                        while (cp != NULL) {
                                if(frame != NULL) {
                                        if (audio_pbuf_decode(cp->playout_buffer, curr_time, frame)) {
                                                audio_playback[s->audio_playback_device.index].audio_put_frame(
                                                        s->audio_playback_device.state, frame);
                                                frame = audio_playback[s->audio_playback_device.index].audio_get_frame(
                                                        s->audio_playback_device.state);
                                        }
                                } else {
                                        frame = audio_playback[s->audio_playback_device.index].audio_get_frame(
                                                s->audio_playback_device.state);
                                }
                                pbuf_remove(cp->playout_buffer, curr_time);
                                cp = pdb_iter_next(s->audio_participants);
                        }
                        pdb_iter_done(s->audio_participants);
                } else { /* NET_JACK */
#ifdef HAVE_JACK
                        jack_receive(s->jack_connection, frame);
                        audio_playback[s->audio_playback_device.index].audio_put_frame(
                                                s->audio_playback_device.state, frame);
                        frame = audio_playback[s->audio_playback_device.index].audio_get_frame(
                                s->audio_playback_device.state);
#endif
                }
        }

        return NULL;
}

static void *audio_sender_thread(void *arg)
{
        struct state_audio *s = (struct state_audio *) arg;
        struct audio_frame *buffer = NULL;
        
        printf("Audio sending started.\n");
        while (!should_exit) {
                buffer = audio_capture[s->audio_capture_device.index].audio_read(
                        s->audio_capture_device.state);
                if(buffer) {
                        if(s->sender == NET_NATIVE)
                                audio_tx_send(s->tx_session, s->audio_network_device, buffer);
#ifdef HAVE_JACK
                        else
                                jack_send(s->jack_connection, buffer);
#endif
                }
        }

        return NULL;
}

void audio_sdi_send(struct state_audio *s, struct audio_frame *frame) {
        struct state_sdi_capture *sdi;
        if(strcmp(audio_capture[s->audio_capture_device.index].name, "embedded") != 0)
                return;
        
        sdi = (struct state_sdi_capture *) s->audio_capture_device.state;
        sdi->audio_buffer = frame;
        platform_sem_post(&sdi->audio_frame_ready);
}

void audio_register_get_callback(struct state_audio *s, struct audio_frame * (*callback)(void *),
                void *udata)
{
        struct state_sdi_playback *sdi;
        //assert(strcmp(audio_capture[s->audio_capture_device.index].name, "embedded") == 0);
        
        sdi = (struct state_sdi_playback *) s->audio_playback_device.state;
        sdi->get_callback = callback;
        sdi->get_udata = udata;
}

void audio_register_put_callback(struct state_audio *s, void (*callback)(void *, struct audio_frame *),
                void *udata)
{
        struct state_sdi_playback *sdi;
        //assert(strcmp(audio_capture[s->audio_capture_device.index].name, "embedded") == 0);
        
        sdi = (struct state_sdi_playback *) s->audio_playback_device.state;
        sdi->put_callback = callback;
        sdi->put_udata = udata;
}

int audio_does_send_sdi(struct state_audio *s)
{
        if(!s) 
                return FALSE;
        return strcmp(audio_capture[s->audio_capture_device.index].name, "embedded") == 0;
}

int audio_does_receive_sdi(struct state_audio *s)
{
        if(!s) 
                return FALSE;
        return strcmp(audio_playback[s->audio_playback_device.index].name, "embedded") == 0;
}

void sdi_put_frame(void *state, struct audio_frame *frame)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;

        if(s->put_callback)
                s->put_callback(s->put_udata, frame);
}

struct audio_frame * sdi_get_frame(void *state)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;
        
        if(s->get_callback)
                return s->get_callback(s->get_udata);
        else
                return NULL;
}

void sdi_playback_done(void *s)
{
        UNUSED(s);
}

void * sdi_capture_finish(void *state)
{
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) state;
        platform_sem_post(&s->audio_frame_ready);
}

void * sdi_capture_done(void *state)
{
        UNUSED(state);
}

