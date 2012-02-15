/*
 * FILE:    audio/playback/sdi.c
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
#include "audio/playback/sdi.h" 
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#endif
#include "debug.h"

#include <stdlib.h>

struct state_sdi_playback {
        struct audio_frame * (*get_callback)(void *);
        void (*put_callback)(void *, struct audio_frame *);
        int (*reconfigure_callback)(void *state, int quant_samples, int channels,
                int sample_rate);
        void *get_udata;
        void *put_udata;
        void *reconfigure_udata;
};


void sdi_playback_help(void)
{
        printf("\tembedded : SDI audio (if available)\n");
}

void * sdi_playback_init(char *cfg)
{
        struct state_sdi_playback *s = malloc(sizeof(struct state_sdi_playback));
        UNUSED(cfg);
        s->get_callback = NULL;
        s->put_callback = NULL;
        s->reconfigure_callback = NULL;
        return s;
}

void sdi_register_get_callback(void *state, struct audio_frame * (*callback)(void *),
                void *udata)
{
        struct state_sdi_playback *s = (struct state_sdi_playback *) state;
        
        s->get_callback = callback;
        s->get_udata = udata;
}

void sdi_register_put_callback(void *state, void (*callback)(void *, struct audio_frame *),
                void *udata)
{
        struct state_sdi_playback *s = (struct state_sdi_playback *) state;
        
        s->put_callback = callback;
        s->put_udata = udata;
}

void sdi_register_reconfigure_callback(void *state, int (*callback)(void *, int, int,
                        int),
                void *udata)
{
        struct state_sdi_playback *s = (struct state_sdi_playback *) state;
        
        s->reconfigure_callback = callback;
        s->reconfigure_udata = udata;
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
        
        if(s->get_callback) {
                return s->get_callback(s->get_udata);
        } else {
                return NULL;
        }
}

int sdi_reconfigure(void *state, int quant_samples, int channels,
                int sample_rate)
{
        struct state_sdi_playback *s;
        s = (struct state_sdi_playback *) state;

        if(s->reconfigure_callback) {
                return s->reconfigure_callback(s->reconfigure_udata, quant_samples, channels, sample_rate);
        } else {
                return FALSE;
        }
}


void sdi_playback_done(void *s)
{
        UNUSED(s);
}

/* vim: set expandtab: sw=8 */

