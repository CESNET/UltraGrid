/*
 * FILE:    audio/echo.h
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

#ifndef AUDIO_CODEC_H_
#define AUDIO_CODEC_H_

#include "audio/audio.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
        AUDIO_CODER,
        AUDIO_DECODER
} audio_codec_direction_t;

struct audio_codec {
        const audio_codec_t *supported_codecs;
        const int *supported_bytes_per_second;
        void *(*init)(audio_codec_t, audio_codec_direction_t, bool);
        audio_channel *(*compress)(void *, audio_channel *);
        audio_channel *(*decompress)(void *, audio_channel *);
        void (*done)(void *);
};

extern void (*register_audio_codec)(struct audio_codec *);

typedef struct {
        const char *name;
        /** @var tag
         *  @brief TwoCC if defined, otherwise we define our tag
         */
        uint32_t    tag;
} audio_codec_info_t;

extern audio_codec_info_t audio_codec_info[];
extern int audio_codec_info_len;

struct audio_codec_state;

struct audio_codec_state *audio_codec_init(audio_codec_t audio_codec, audio_codec_direction_t);
struct audio_codec_state *audio_codec_reconfigure(struct audio_codec_state *old,
                audio_codec_t audio_codec, audio_codec_direction_t);
audio_frame2 *audio_codec_compress(struct audio_codec_state *, audio_frame2 *);
audio_frame2 *audio_codec_decompress(struct audio_codec_state *, audio_frame2 *);
const int *audio_codec_get_supported_bps(struct audio_codec_state *);
void audio_codec_done(struct audio_codec_state *);

void list_audio_codecs(void);

#ifdef __cplusplus
}
#endif

#endif /* AUDIO_CODEC_H */
