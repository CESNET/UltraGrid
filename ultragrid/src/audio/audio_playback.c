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
#include "audio/audio_playback.h" 
#include "audio/playback/portaudio.h" 
#include "audio/playback/alsa.h" 
#include "audio/playback/coreaudio.h" 
#include "audio/playback/none.h" 
#include "audio/playback/jack.h" 
#include "audio/playback/sdi.h" 
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif
#include "debug.h"
#include "host.h"
#include "lib_common.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>


struct state_audio_playback {
        int index;
        void *state;
};


typedef void (*audio_device_help_t)(void);
/**
 * @return state
 */
typedef void * (*audio_init_t)(char *cfg);

typedef struct audio_frame* (*audio_get_frame_t)(void *state);
typedef void (*audio_put_frame_t)(void *state, struct audio_frame *frame);
typedef void (*audio_finish_t)(void *state);
typedef void (*audio_done_t)(void *state);
/*
 * Returns TRUE if succeeded, FALSE otherwise
 */
typedef int (*audio_reconfigure_t)(void *state, int quant_samples, int channels,
                int sample_rate);
typedef void (*audio_playback_done_t)(void *s);

struct audio_playback_t {
        const char              *name;

        const char              *library_name;
        audio_device_help_t      audio_help;
        const char              *audio_help_str;
        audio_init_t             audio_init;
        const char              *audio_init_str;
        audio_get_frame_t        audio_get_frame;
        const char              *audio_get_frame_str;
        audio_put_frame_t        audio_put_frame;
        const char              *audio_put_frame_str;
        audio_playback_done_t    audio_playback_done;
        const char              *audio_playback_done_str;
        audio_reconfigure_t      audio_reconfigure;
        const char              *audio_reconfigure_str;

        void *handle;
};

static struct audio_playback_t audio_playback_table[] = {
        { "embedded",
               NULL,
               MK_STATIC(sdi_playback_help),
               MK_STATIC(sdi_playback_init),
               MK_STATIC(sdi_get_frame),
               MK_STATIC(sdi_put_frame),
               MK_STATIC(sdi_playback_done),
               MK_STATIC(sdi_reconfigure),
               NULL
        },
#if defined HAVE_ALSA || defined BUILD_LIBRARIES
        { "alsa",
                "alsa",
                MK_NAME(audio_play_alsa_help),
                MK_NAME(audio_play_alsa_init),
                MK_NAME(audio_play_alsa_get_frame),
                MK_NAME(audio_play_alsa_put_frame),
                MK_NAME(audio_play_alsa_done),
                MK_NAME(audio_play_alsa_reconfigure),
                NULL
        },
#endif
#if defined HAVE_COREAUDIO
        { "coreaudio",
                NULL,
                MK_STATIC(audio_play_ca_help),
                MK_STATIC(audio_play_ca_init),
                MK_STATIC(audio_play_ca_get_frame),
                MK_STATIC(audio_play_ca_put_frame),
                MK_STATIC(audio_play_ca_done),
                MK_STATIC(audio_play_ca_reconfigure),
                NULL
        },
#endif
#if defined HAVE_JACK || defined BUILD_LIBRARIES
        { "jack",
                "jack",
                MK_NAME(audio_play_jack_help),
                MK_NAME(audio_play_jack_init),
                MK_NAME(audio_play_jack_get_frame),
                MK_NAME(audio_play_jack_put_frame),
                MK_NAME(audio_play_jack_done),
                MK_NAME(audio_play_jack_reconfigure),
                NULL
        },
#endif
#if defined HAVE_PORTAUDIO || defined BUILD_LIBRARIES
        { "portaudio",
                "portaudio",
                MK_NAME(portaudio_playback_help),
                MK_NAME(portaudio_playback_init),
                MK_NAME(portaudio_get_frame),
                MK_NAME(portaudio_put_frame),
                MK_NAME(portaudio_close_playback),
                MK_NAME(portaudio_reconfigure),
                NULL
        },
#endif
        { "none",
                NULL,
                MK_STATIC(audio_play_none_help),
                MK_STATIC(audio_play_none_init),
                MK_STATIC(audio_play_none_get_frame),
                MK_STATIC(audio_play_none_put_frame),
                MK_STATIC(audio_play_none_done),
                MK_STATIC(audio_play_none_reconfigure),
                NULL
        }
};

#define MAX_AUDIO_PLAY (sizeof(audio_playback_table) / sizeof(struct audio_playback_t))

struct audio_playback_t *available_audio_playback[MAX_AUDIO_PLAY];
int available_audio_playback_count = 0;

#ifdef BUILD_LIBRARIES
static void *audio_playback_open_library(const char *playback_name)
{
        char name[128];
        snprintf(name, sizeof(name), "aplay_%s.so.%d", playback_name, AUDIO_PLAYBACK_ABI_VERSION);

        return open_library(name);
}

static int audio_playback_fill_symbols(struct audio_playback_t *device)
{
        void *handle = device->handle;

        device->audio_help = (audio_device_help_t)
                dlsym(handle, device->audio_help_str);
        device->audio_init = (audio_init_t)
                dlsym(handle, device->audio_init_str);
        device->audio_get_frame = (audio_get_frame_t)
                dlsym(handle, device->audio_get_frame_str);
        device->audio_put_frame = (audio_put_frame_t)
                dlsym(handle, device->audio_put_frame_str);
        device->audio_playback_done = (audio_done_t)
                dlsym(handle, device->audio_playback_done_str);
        device->audio_reconfigure = (audio_reconfigure_t)
                dlsym(handle, device->audio_reconfigure_str);

        if(!device->audio_help || !device->audio_init || !device->audio_get_frame ||
                        !device->audio_put_frame || !device->audio_playback_done || !device->audio_reconfigure) {
                fprintf(stderr, "Library %s opening error: %s \n", device->library_name, dlerror());
                return FALSE;
        }

        return TRUE;
}
#endif

void audio_playback_init_devices()
{
        unsigned int i;

        for(i = 0; i < MAX_AUDIO_PLAY; ++i) {
#ifdef BUILD_LIBRARIES
                if(audio_playback_table[i].library_name) {
                        int ret;
                        audio_playback_table[i].handle =
                                audio_playback_open_library(audio_playback_table[i].library_name);
                        if(!audio_playback_table[i].handle) {
                                continue;
                        }
                        ret = audio_playback_fill_symbols(&audio_playback_table[i]);
                        if(!ret) {
                                continue;
                        }
                }
#endif
                available_audio_playback[available_audio_playback_count] = &audio_playback_table[i];
                available_audio_playback_count++;
        }
}

void audio_playback_help()
{
        int i;
        printf("Available audio playback devices:\n");
        for (i = 0; i < available_audio_playback_count; ++i) {
                available_audio_playback[i]->audio_help();
                printf("\n");
        }
}

struct state_audio_playback * audio_playback_init(char *device, char *cfg)
{
        struct state_audio_playback *s;
        int i;
        
        s = (struct state_audio_playback *) malloc(sizeof(struct state_audio_playback));

        for (i = 0; i < available_audio_playback_count; ++i) {
                if(strcmp(device, available_audio_playback[i]->name) == 0) {
                        s->index = i;
                        break;
                }
        }

        if(i == available_audio_playback_count) {
                fprintf(stderr, "Unknown audio playback driver: %s\n", device);
                goto error;
        }
                
        s->state = available_audio_playback[s->index]->audio_init(cfg);
                
        if(!s->state) {
                fprintf(stderr, "Error initializing audio playback.\n");
                goto error;
        }

        return s;

error:
        free(s);
        return NULL;
}

struct state_audio_playback *audio_playback_init_null_device(void)
{
        return audio_playback_init("none", NULL);
}

void audio_playback_finish(struct state_audio_playback *s)
{
        UNUSED(s);
}

void audio_playback_done(struct state_audio_playback *s)
{
        if(s) {
                available_audio_playback[s->index]->audio_playback_done(s->state);
                free(s);
        }
}

int audio_playback_does_receive_sdi(struct state_audio_playback *s)
{
        if(!s) 
                return FALSE;

        return strcmp(available_audio_playback[s->index]->name, "embedded") == 0;
}

struct audio_frame * audio_playback_get_frame(struct state_audio_playback *s)
{
        return available_audio_playback[s->index]->audio_get_frame(
                                                        s->state);
}

void audio_playback_put_frame(struct state_audio_playback *s, struct audio_frame *frame)
{
        return available_audio_playback[s->index]->audio_put_frame(
                                                        s->state,
                                                        frame);
}

int audio_playback_reconfigure(struct state_audio_playback *s, int quant_samples, int channels,
                int sample_rate)
{
        return available_audio_playback[s->index]->audio_reconfigure(
                                                        s->state,
                                                        quant_samples, channels, sample_rate);
}

void  *audio_playback_get_state_pointer(struct state_audio_playback *s)
{
        return s->state;
}

/* vim: set expandtab: sw=8 */

