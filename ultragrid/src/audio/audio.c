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
#include "audio/portaudio.h" 
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "version.h"
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

enum audio_sound_device {
        AUDIO_DEV_NONE = 0,
        AUDIO_DEV_SDI = 1,
        AUDIO_DEV_PORTAUDIO = 2
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

/** 
 * Copies one input channel into n output (interlaced).
 * 
 * Input and output data may overlap. 
 */
void copy_channel(char *out, const char *in, int bps, int in_len /* bytes */, int out_channel_count); 


/**
 * @return state
 */
typedef void * (*audio_init_t)(char *cfg);
typedef struct audio_frame* (*audio_read_t)(void *state);
typedef struct audio_frame* (*audio_get_frame_t)(void *state);
typedef void (*audio_put_frame_t)(void *state, struct audio_frame *frame);
typedef void (*audio_playback_done_t)(void *s);

struct audio_capture_t {
        audio_init_t audio_init;
        audio_read_t audio_read;
};

struct audio_playback_t {
        audio_init_t audio_init;
        audio_get_frame_t audio_get_frame;
        audio_put_frame_t audio_put_frame;
        audio_playback_done_t playback_done;
};

void * sdi_capture_init(char *cfg);
void * sdi_playback_init(char *cfg);
void sdi_done(void *s);
struct audio_frame * sdi_read(void *state);
static void *audio_sender_thread(void *arg);
static void *audio_receiver_thread(void *arg);
static struct rtp *initialize_audio_network(char *addr, struct pdb *participants);
void print_audio_devices(enum audio_device_kind kind);

static struct audio_capture_t audio_capture[] = {
#ifdef HAVE_PORTAUDIO
        [AUDIO_DEV_PORTAUDIO] = { portaudio_capture_init, portaudio_read },
#endif
        [AUDIO_DEV_SDI] = { sdi_capture_init, sdi_read }
};

static struct audio_playback_t audio_playback[] = {
#ifdef HAVE_PORTAUDIO
        [AUDIO_DEV_PORTAUDIO] = { portaudio_playback_init, portaudio_get_frame, portaudio_put_frame, portaudio_close_playback },
#endif
        [AUDIO_DEV_SDI] = { sdi_playback_init, sdi_get_frame, sdi_put_frame, sdi_done }
};


void print_audio_devices(enum audio_device_kind kind)
{
        printf("Available audio %s devices:\n", kind == AUDIO_IN ? "input"
                        : "output");
        printf("\tembedded : SDI audio (if available)\n");
#ifdef HAVE_PORTAUDIO
        portaudio_print_available_devices(kind);
#endif
}

/**
 * take care that addrs can also be comma-separated list of addresses !
 */
struct state_audio * audio_cfg_init(char *addrs, char *send_cfg, char *recv_cfg, char *jack_cfg)
{
        struct state_audio *s = NULL;
        char *tmp, *unused;
        char *addr;
        
        if (send_cfg != NULL &&
                        !strcmp("help", send_cfg)) {
                print_audio_devices(AUDIO_IN);
                exit(0);
        }
        
        if (recv_cfg != NULL &&
                        !strcmp("help", recv_cfg)) {
                print_audio_devices(AUDIO_OUT);
                exit(0);
        }
        
        s = calloc(1, sizeof(struct state_audio));
        s->audio_participants = NULL;
        
        gettimeofday(&s->start_time, NULL);        
        
        tmp = strdup(addrs);
        s->audio_participants = pdb_init();
        addr = strtok_r(tmp, ",", &unused);
        if ((s->audio_network_device =
             initialize_audio_network(addr,
                                      s->audio_participants)) ==
            NULL) {
                printf("Unable to open audio network\n");
                free(tmp);
                exit(EXIT_FAIL_NETWORK);
        }
        free(tmp);

        if (send_cfg != NULL) {
                char *tmp = strtok(send_cfg, ":");
                if (!strcmp("embedded", tmp)) {
                        s->audio_capture_device.index = AUDIO_DEV_SDI;
                } 
#ifdef HAVE_PORTAUDIO
                else if (!strcmp("portaudio", tmp)) {
                        s->audio_capture_device.index = AUDIO_DEV_PORTAUDIO;
                }
#endif
                else {
                        fprintf(stderr, "Unknown audio driver: %s\n", tmp);
                        exit(EXIT_FAIL_USAGE);
                }
                
                tmp = strtok(NULL, ":");
                s->audio_capture_device.state =
                        audio_capture[s->audio_capture_device.index].audio_init(tmp);
                
                if(!s->audio_capture_device.state) {
                        error_with_code_msg(EXIT_FAILURE, "Error initializing audio capture.\n");
                }
                if (pthread_create
                    (&s->audio_sender_thread_id, NULL, audio_sender_thread, (void *)s) != 0) {
                        fprintf(stderr,
                                "Error creating audio thread. Quitting\n");
                        exit(EXIT_FAILURE);
                }
        } else {
                s->audio_capture_device.index = AUDIO_DEV_NONE;
        }
        
        if (recv_cfg != NULL) {
                char *tmp = strtok(recv_cfg, ":");
                if (!strcmp("embedded", tmp)) {
                        s->audio_playback_device.index = AUDIO_DEV_SDI;
                }
#ifdef HAVE_PORTAUDIO                        
                else if (!strcmp("portaudio", tmp)) {
                        s->audio_playback_device.index = AUDIO_DEV_PORTAUDIO;
                } 
#endif                  
                else {
                        fprintf(stderr, "Unknown audio driver: %s\n", tmp);
                        exit(EXIT_FAIL_USAGE);
                }
                
                tmp = strtok(NULL, ":");
                s->audio_playback_device.state =
                        audio_playback[s->audio_playback_device.index].audio_init(tmp);
                if(!s->audio_playback_device.state) {
                        error_with_code_msg(EXIT_FAILURE, "Error initializing audio playback.\n");
                }
                if (pthread_create
                    (&s->audio_receiver_thread_id, NULL, audio_receiver_thread, (void *)s) != 0) {
                        fprintf(stderr,
                                "Error creating audio thread. Quitting\n");
                        exit(EXIT_FAILURE);
                }
        } else {
                s->audio_playback_device.index = AUDIO_DEV_NONE;
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
                exit(EXIT_FAIL_USAGE);
        }
#endif

        return s;
}

void audio_join(struct state_audio *s) {
        if(s) {
                if(s->audio_playback_device.index)
                        pthread_join(s->audio_receiver_thread_id, NULL);
                if(s->audio_capture_device.index)
                        pthread_join(s->audio_sender_thread_id, NULL);
        }
}
        
void audio_done(struct state_audio *s) {
        if(s) {
                if(s->audio_participants)
                        pdb_destroy(&s->audio_participants);
                if(s->audio_playback_device.index)
                        audio_playback[s->audio_playback_device.index].playback_done(s->audio_playback_device.state);
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
        return s->audio_buffer;
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
                             ULTRAGRID_VERSION, strlen(ULTRAGRID_VERSION));
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
                        ts = tv_diff(curr_time, s->start_time) * 90000;        // What is this?
                        rtp_update(s->audio_network_device, curr_time);        // this is just some internal rtp housekeeping...nothing to worry about
                        rtp_send_ctrl(s->audio_network_device, ts, 0, curr_time);      // strange..
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
                                audio_tx_send(s->audio_network_device, buffer);
#ifdef HAVE_JACK
                        else
                                jack_send(s->jack_connection, buffer);
#endif
                }
        }
/*#ifdef HAVE_PORTAUDIO
        free_audio_frame(&buffer);
        portaudio_close(stream);
#endif*/ /* HAVE_PORTAUDIO */

        return NULL;
}

void audio_sdi_send(struct state_audio *s, struct audio_frame *frame) {
        struct state_sdi_capture *sdi;
        if(s->audio_capture_device.index != AUDIO_DEV_SDI)
                return;
        
        sdi = (struct state_sdi_capture *) s->audio_capture_device.state;
        sdi->audio_buffer = frame;
        platform_sem_post(&sdi->audio_frame_ready);
}

void audio_register_get_callback(struct state_audio *s, struct audio_frame * (*callback)(void *),
                void *udata)
{
        struct state_sdi_playback *sdi;
        assert(s->audio_playback_device.index == AUDIO_DEV_SDI);
        
        sdi = (struct state_sdi_playback *) s->audio_playback_device.state;
        sdi->get_callback = callback;
        sdi->get_udata = udata;
}

void audio_register_put_callback(struct state_audio *s, void (*callback)(void *, struct audio_frame *),
                void *udata)
{
        struct state_sdi_playback *sdi;
        assert(s->audio_playback_device.index == AUDIO_DEV_SDI);
        
        sdi = (struct state_sdi_playback *) s->audio_playback_device.state;
        sdi->put_callback = callback;
        sdi->put_udata = udata;
}

int audio_does_send_sdi(struct state_audio *s)
{
        if(!s) 
                return FALSE;
        return s->audio_capture_device.index == AUDIO_DEV_SDI;
}

int audio_does_receive_sdi(struct state_audio *s)
{
        if(!s) 
                return FALSE;
        return s->audio_playback_device.index == AUDIO_DEV_SDI;
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

void sdi_done(void *s)
{
        UNUSED(s);
}
