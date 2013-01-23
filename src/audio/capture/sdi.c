/*
 * FILE:    audio/capture/sdi.c
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
#include "audio/capture/sdi.h" 

#include "compat/platform_semaphore.h"
#include "debug.h"
#include "host.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


struct state_sdi_capture {
        struct audio_frame * audio_buffer;
        sem_t audio_frame_ready;

};


void * sdi_capture_init(char *cfg)
{
        if(cfg && strcmp(cfg, "help")) {
                printf("Available embedded devices:\n");
                sdi_capture_help("embedded");
                sdi_capture_help("AESEBU");
                sdi_capture_help("analog");
                return NULL;
        }
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) calloc(1, sizeof(struct state_sdi_capture));
        platform_sem_init(&s->audio_frame_ready, 0, 0);
        
        return s;
}

struct audio_frame * sdi_read(void *state)
{
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) state;
        platform_sem_wait(&s->audio_frame_ready);
        return s->audio_buffer;
}

void sdi_capture_finish(void *state)
{
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) state;
        platform_sem_post(&s->audio_frame_ready);
}

void sdi_capture_done(void *state)
{
        UNUSED(state);
}

void sdi_capture_help(const char *driver_name)
{
        if(strcmp(driver_name, "embedded") == 0) {
                printf("\tembedded : SDI audio (if available)\n");
        } else if(strcmp(driver_name, "AESEBU") == 0) {
                printf("\tAESEBU : separately connected AES/EBU to a grabbing card (if available)\n");
        } else if(strcmp(driver_name, "analog") == 0) {
                printf("\tanalog : analog input of grabbing card (if available)\n");
        }
}

void sdi_capture_new_incoming_frame(void *state, struct audio_frame *frame)
{
        struct state_sdi_capture *s;
        
        s = (struct state_sdi_capture *) state;
        s->audio_buffer = frame;
        platform_sem_post(&s->audio_frame_ready);

}

/* vim: set expandtab: sw=8 */
