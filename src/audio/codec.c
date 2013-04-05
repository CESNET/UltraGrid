/*
 * FILE:    audio/codec.c
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
#endif /* HAVE_CONFIG_H */

#include "audio/codec.h"
#include "audio/codec/dummy_pcm.h"

typedef struct {
        const char *name;
        uint32_t    tag;
} audio_codec_info_t;

static audio_codec_info_t audio_codec_names[] = {
        [AC_NONE] = { "(none)", 0 },
        [AC_PCM] = { "PCM", 0x0001 },
};

static struct audio_codec *audio_codecs[] = {
        &dummy_pcm_audio_codec,
};

audio_codec_t get_audio_codec_to_name(const char *codec) {
        for(unsigned int i = 0; i < sizeof(audio_codec_names)/sizeof(audio_codec_info_t); ++i) {
                if(strcasecmp(audio_codec_names[i].name, codec) == 0) {
                        return i;
                }
        }
        return AC_NONE;
}

const char *get_name_to_audio_codec(audio_codec_t codec)
{
        return audio_codec_names[codec].name;
}

uint32_t get_audio_tag(audio_codec_t codec)
{
        return audio_codec_names[codec].tag;
}

audio_codec_t get_audio_codec_to_tag(uint32_t tag)
{
        for(unsigned int i = 0; i < sizeof(audio_codec_names)/sizeof(audio_codec_info_t); ++i) {
                if(audio_codec_names[i].tag == tag) {
                        return i;
                }
        }
        return AC_NONE;
}

struct audio_codec_state {
        void *state;
        int index;
        audio_codec_t codec;
};

struct audio_codec_state *audio_codec_init(audio_codec_t audio_codec) {
        void *state = NULL;
        int index;
        for(unsigned int i = 0; i < sizeof(audio_codecs)/sizeof(struct audio_codec *); ++i) {
                for(unsigned int j = 0; audio_codecs[i]->supported_codecs[j] != AC_NONE; ++j) {
                        if(audio_codecs[i]->supported_codecs[j] == audio_codec) {
                                state = audio_codecs[i]->init(audio_codec);
                                index = i;
                                if(state)
                                        break;
                        }
                }
                if(state)
                        break;
        }

        if(!state) {
                fprintf(stderr, "Unable to find encoder for audio codec '%s'\n",
                                get_name_to_audio_codec(audio_codec));
                return NULL;
        }

        struct audio_codec_state *s = (struct audio_codec_state *) malloc(sizeof(struct audio_codec_state));

        s->state = state;
        s->index = index;
        s->codec = audio_codec;

        return s;
}

struct audio_codec_state *audio_codec_reconfigure(struct audio_codec_state *old,
                audio_codec_t audio_codec)
{
        if(old && old->codec == audio_codec)
                return old;
        audio_codec_done(old);
        return audio_codec_init(audio_codec);
}

audio_frame2 *audio_codec_compress(struct audio_codec_state *s, audio_frame2 *frame)
{
        return audio_codecs[s->index]->compress(s->state, frame);
}

audio_frame2 *audio_codec_decompress(struct audio_codec_state *s, audio_frame2 *frame)
{
        return audio_codecs[s->index]->decompress(s->state, frame);
}

void audio_codec_done(struct audio_codec_state *s)
{
        if(!s)
                return;
        audio_codecs[s->index]->done(s->state);
        free(s);
}

