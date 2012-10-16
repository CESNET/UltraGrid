/*
 * FILE:    audio/playback/none.c
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
#include "audio/playback/none.h" 
#include "debug.h"
#include <stdlib.h>
#include <string.h>

#define AUDIO_PLAYBACK_NONE_MAGIC 0x3bcf376au

struct state_audio_playback_none
{
        uint32_t magic;
};

void audio_play_none_help(const char *driver_name)
{
        UNUSED(driver_name);
}

void * audio_play_none_init(char *cfg)
{
        UNUSED(cfg);
        struct state_audio_playback_none *s;

        s = (struct state_audio_playback_none *)
                malloc(sizeof(struct state_audio_playback_none));
        assert(s != NULL);
        s->magic = AUDIO_PLAYBACK_NONE_MAGIC;
                                
        return s;
}

struct audio_frame *audio_play_none_get_frame(void *state)
{
        struct state_audio_playback_none *s = 
                (struct state_audio_playback_none *) state;
        assert(s->magic == AUDIO_PLAYBACK_NONE_MAGIC);
        return NULL;
}

void audio_play_none_put_frame(void *state, struct audio_frame *frame)
{
        UNUSED(frame);
        struct state_audio_playback_none *s = 
                (struct state_audio_playback_none *) state;
        assert(s->magic == AUDIO_PLAYBACK_NONE_MAGIC);
}

void audio_play_none_done(void *state)
{
        struct state_audio_playback_none *s = 
                (struct state_audio_playback_none *) state;
        assert(s->magic == AUDIO_PLAYBACK_NONE_MAGIC);
        free(s);
}

int audio_play_none_reconfigure(void *state, int quant_samples, int channels,
                                                int sample_rate)
{
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);
        struct state_audio_playback_none *s = 
                (struct state_audio_playback_none *) state;
        assert(s->magic == AUDIO_PLAYBACK_NONE_MAGIC);

        return TRUE;
}
