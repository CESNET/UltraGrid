/*
 * FILE:    audio/audio_capture.c
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

#include "audio/audio_capture.h" 
#include "audio/capture/portaudio.h" 
#include "audio/capture/alsa.h" 
#include "audio/capture/coreaudio.h" 
#include "audio/capture/none.h" 
#include "audio/capture/jack.h" 
#include "audio/capture/sdi.h" 

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif
#include "debug.h"
#include "host.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "lib_common.h"

struct state_audio_capture {
        int index;
        void *state;
};

struct audio_capture_t;

typedef void (*audio_device_help_t)(void);
/**
 * @return state
 */
typedef void * (*audio_init_t)(char *cfg);
typedef struct audio_frame* (*audio_read_t)(void *state);
typedef void (*audio_finish_t)(void *state);
typedef void (*audio_done_t)(void *state);

#ifdef BUILD_LIBRARIES
static void *audio_capture_open_library(const char *capture_name);
static int audio_capture_fill_symbols(struct audio_capture_t *device);
#endif

struct audio_capture_t {
        const char              *name;

        const char              *library_name;
        audio_device_help_t      audio_help;
        const char              *audio_help_str;
        audio_init_t             audio_init;
        const char              *audio_init_str;
        audio_read_t             audio_read;
        const char              *audio_read_str;
        audio_finish_t           audio_capture_finish;
        const char              *audio_capture_finish_str;
        audio_done_t             audio_capture_done;
        const char              *audio_capture_done_str;

        void                    *handle;
};

static struct audio_capture_t audio_capture_table[] = {
        { "embedded",
                NULL,
                MK_STATIC(sdi_capture_help),
                MK_STATIC(sdi_capture_init),
                MK_STATIC(sdi_read),
                MK_STATIC(sdi_capture_finish),
                MK_STATIC(sdi_capture_done),
                NULL
        },
#if defined HAVE_ALSA || defined BUILD_LIBRARIES
        { "alsa",
                "alsa",
                MK_NAME(audio_cap_alsa_help),
                MK_NAME(audio_cap_alsa_init),
                MK_NAME(audio_cap_alsa_read),
                MK_NAME(audio_cap_alsa_finish),
                MK_NAME(audio_cap_alsa_done),
                NULL
        },
#endif
#if defined HAVE_COREAUDIO
        { "coreaudio",
                NULL,
                MK_STATIC(audio_cap_ca_help),
                MK_STATIC(audio_cap_ca_init),
                MK_STATIC(audio_cap_ca_read),
                MK_STATIC(audio_cap_ca_finish),
                MK_STATIC(audio_cap_ca_done),
                NULL
        },
#endif
#if defined HAVE_PORTAUDIO || defined BUILD_LIBRARIES
        { "portaudio",
                "portaudio",
                MK_NAME(portaudio_capture_help),
                MK_NAME(portaudio_capture_init),
                MK_NAME(portaudio_read),
                MK_NAME(portaudio_capture_finish),
                MK_NAME(portaudio_capture_done),
                NULL
        },
#endif
#if defined HAVE_JACK || defined BUILD_LIBRARIES
        { "jack",
                "jack",
                MK_NAME(audio_cap_jack_help),
                MK_NAME(audio_cap_jack_init),
                MK_NAME(audio_cap_jack_read),
                MK_NAME(audio_cap_jack_finish),
                MK_NAME(audio_cap_jack_done),
                NULL
        },
#endif
        { "none",
                NULL,
                MK_STATIC(audio_cap_none_help),
                MK_STATIC(audio_cap_none_init),
                MK_STATIC(audio_cap_none_read),
                MK_STATIC(audio_cap_none_finish),
                MK_STATIC(audio_cap_none_done),
                NULL
        }
};

#define MAX_AUDIO_CAP (sizeof(audio_capture_table) / sizeof(struct audio_capture_t))

struct audio_capture_t *available_audio_capture[MAX_AUDIO_CAP];
int available_audio_capture_count = 0;

#ifdef BUILD_LIBRARIES
static void *audio_capture_open_library(const char *capture_name)
{
        char name[128];
        snprintf(name, sizeof(name), "acap_%s.so.%d", capture_name, AUDIO_CAPTURE_ABI_VERSION);

        return open_library(name);
}

static int audio_capture_fill_symbols(struct audio_capture_t *device)
{
        void *handle = device->handle;

        device->audio_help = (audio_device_help_t)
                dlsym(handle, device->audio_help_str);
        device->audio_init = (audio_init_t)
                dlsym(handle, device->audio_init_str);
        device->audio_read = (audio_read_t)
                dlsym(handle, device->audio_read_str);
        device->audio_capture_finish = (audio_finish_t)
                dlsym(handle, device->audio_capture_finish_str);
        device->audio_capture_done = (audio_done_t)
                dlsym(handle, device->audio_capture_done_str);

        if(!device->audio_help || !device->audio_init || !device->audio_read ||
                        !device->audio_capture_finish || !device->audio_capture_done) {
                fprintf(stderr, "Library %s opening error: %s \n", device->library_name, dlerror());
                return FALSE;
        }

        return TRUE;
}

#endif

void audio_capture_init_devices()
{
        unsigned int i;

        for(i = 0; i < MAX_AUDIO_CAP; ++i) {
#ifdef BUILD_LIBRARIES
                if(audio_capture_table[i].library_name) {
                        int ret;
                        audio_capture_table[i].handle =
                                audio_capture_open_library(audio_capture_table[i].library_name);
                        if(!audio_capture_table[i].handle) {
                                continue;
                        }
                        ret = audio_capture_fill_symbols(&audio_capture_table[i]);
                        if(!ret) {
                                continue;
                        }
                }
#endif
                available_audio_capture[available_audio_capture_count] = &audio_capture_table[i];
                available_audio_capture_count++;
        }
}

struct state_audio_capture *audio_capture_init(char *driver, char *cfg)
{
        struct state_audio_capture *s;
        int i;

        s = (struct state_audio_capture *) malloc(sizeof(struct state_audio_capture));
        assert(s != NULL);

        for (i = 0; i < available_audio_capture_count; ++i) {
                if(strcmp(driver, available_audio_capture[i]->name) == 0) {
                        s->index = i;
                        break;
                }
        }

        if(i == available_audio_capture_count) {
                fprintf(stderr, "Unknown audio capture driver: %s\n", driver);
                goto error;
        }
                
        s->state =
                available_audio_capture[s->index]->audio_init(cfg);
                
        if(!s->state) {
                fprintf(stderr, "Error initializing audio capture.\n");
                goto error;
        }

        return s;

error:
        free(s);
        return NULL;
}

struct state_audio_capture *audio_capture_init_null_device()
{
        return audio_capture_init("none", NULL);
}

void audio_capture_print_help()
{
        int i;
        printf("Available audio capture devices:\n");
        for (i = 0; i < available_audio_capture_count; ++i) {
                available_audio_capture[i]->audio_help();
                printf("\n");
        }
}

void audio_capture_finish(struct state_audio_capture *s)
{
        if(s) {
                available_audio_capture[s->index]->audio_capture_finish(s->state);
        }
}

void audio_capture_done(struct state_audio_capture *s)
{
        if(s) {
                available_audio_capture[s->index]->audio_capture_done(s->state);
                
                free(s);
        }
}

struct audio_frame * audio_capture_read(struct state_audio_capture *s)
{
        if(s) {
                return available_audio_capture[s->index]->audio_read(s->state);
        } else {
                return NULL;
        }
}

int audio_capture_does_send_sdi(struct state_audio_capture *s)
{
        if(!s) 
                return FALSE;
        return strcmp(available_audio_capture[s->index]->name, "embedded") == 0;
}

void *audio_capture_get_state_pointer(struct state_audio_capture *s)
{
        return s->state;
}

