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
#include "audio/codec/libavcodec.h"
#include "audio/utils.h"

#include "lib_common.h"

#define MAX_AUDIO_CODECS 20

const int max_audio_codecs = MAX_AUDIO_CODECS;

audio_codec_info_t audio_codec_info[] = {
        [AC_NONE] = { "(none)", 0 },
        [AC_PCM] = { "PCM", 0x0001 },
        [AC_ALAW] = { "A-law", 0x0006 },
        [AC_MULAW] = { "u-law", 0x0007 },
        [AC_ADPCM_IMA_WAV] = { "ADPCM", 0x0011 },
        [AC_SPEEX] = { "speex", 0xA109 },
        [AC_OPUS] = { "OPUS", 0x7375704F }, // == Opus, the TwoCC isn't defined
        [AC_G722] = { "G.722", 0x028F },
        [AC_G726] = { "G.726", 0x0045 },
};

int audio_codec_info_len = sizeof(audio_codec_info)/sizeof(audio_codec_info_t);

#ifdef BUILD_LIBRARIES
static pthread_once_t libraries_initialized = PTHREAD_ONCE_INIT;
static void load_libraries(void);
#endif

static struct audio_codec *audio_codecs[MAX_AUDIO_CODECS] = {
        &dummy_pcm_audio_codec,
        NULL_IF_BUILD_LIBRARIES(LIBAVCODEC_AUDIO_CODEC_HANDLE),
};
static struct audio_codec_state *audio_codec_init_real(audio_codec_t audio_codec,
                audio_codec_direction_t direction, bool try_init);
static void register_audio_codec_real(struct audio_codec *);

void (*register_audio_codec)(struct audio_codec *) = register_audio_codec_real;

static void register_audio_codec_real(struct audio_codec *codec)
{
        for(int i = 0; i < MAX_AUDIO_CODECS; ++i) {
                if(audio_codecs[i] == 0) {
                        audio_codecs[i] = codec;
                        return;
                }
        }
        error_msg("Warning: not enough slots to register further audio codecs!!!\n");
}

struct audio_codec_state {
        void **state;
        int state_count;
        int index;
        audio_codec_t codec;
        audio_codec_direction_t direction;
        audio_frame2 *out;
};

void list_audio_codecs(void) {
        printf("Supported audio codecs:\n");
        for(int i = 0; i < audio_codec_info_len; ++i) {
                if(i != AC_NONE) {
                        printf("\t%s", audio_codec_info[i].name);
                        struct audio_codec_state *st = audio_codec_init_real(i, AUDIO_CODER, true);
                        if(!st) {
                                printf(" - unavailable");
                        } else {
                                audio_codec_done(st);
                        }
                        printf("\n");
                }
        }
}

#ifdef BUILD_LIBRARIES
static void load_libraries(void)
{
        char name[128];
        snprintf(name, sizeof(name), "acodec_*.so.%d", AUDIO_CODEC_ABI_VERSION);

        open_all(name);
}
#endif


struct audio_codec_state *audio_codec_init(audio_codec_t audio_codec,
                audio_codec_direction_t direction) {
        return audio_codec_init_real(audio_codec, direction, true);
}

static struct audio_codec_state *audio_codec_init_real(audio_codec_t audio_codec,
                audio_codec_direction_t direction, bool try_init) {
        void *state = NULL;
        int index;
#ifdef BUILD_LIBRARIES
        pthread_once(&libraries_initialized, load_libraries);
#endif
        for(unsigned int i = 0; i < sizeof(audio_codecs)/sizeof(struct audio_codec *); ++i) {
                if(!audio_codecs[i])
                        continue;
                for(unsigned int j = 0; audio_codecs[i]->supported_codecs[j] != AC_NONE; ++j) {
                        if(audio_codecs[i]->supported_codecs[j] == audio_codec) {
                                state = audio_codecs[i]->init(audio_codec, direction, try_init);
                                index = i;
                                if(state)
                                        break;
                                else
                                        try_init || fprintf(stderr, "Error: initialization of audio codec failed!\n");
                        }
                }
                if(state)
                        break;
        }

        if(!state) {
                try_init || fprintf(stderr, "Unable to find encoder for audio codec '%s'\n",
                                get_name_to_audio_codec(audio_codec));
                return NULL;
        }

        struct audio_codec_state *s = (struct audio_codec_state *) malloc(sizeof(struct audio_codec_state));

        s->state = calloc(1, sizeof(void**));;
        s->state[0] = state;
        s->state_count = 1;
        s->index = index;
        s->codec = audio_codec;
        s->direction = direction;

        s->out = audio_frame2_init();
        s->out->ch_count = 1;

        return s;
}

struct audio_codec_state *audio_codec_reconfigure(struct audio_codec_state *old,
                audio_codec_t audio_codec, audio_codec_direction_t direction)
{
        if(old && old->codec == audio_codec)
                return old;
        audio_codec_done(old);
        return audio_codec_init(audio_codec, direction);
}

/**
 * Audio_codec_compress compresses given audio frame.
 *
 * This function has to be called iterativelly, in first iteration with frame, the others with NULL
 *
 * @param s state
 * @param frame in first iteration audio frame to be compressed, in following NULL
 * @retval pointer pointing to data
 * @retval NULL indicating that there are no data left
 */
audio_frame2 *audio_codec_compress(struct audio_codec_state *s, audio_frame2 *frame)
{
        if(frame && s->state_count < frame->ch_count) {
                s->state = realloc(s->state, sizeof(void **) * frame->ch_count);
                for(int i = s->state_count; i < frame->ch_count; ++i) {
                        s->state[i] = audio_codecs[s->index]->init(s->codec, s->direction, false);
                        if(s->state[i] == NULL) {
                                        fprintf(stderr, "Error: initialization of audio codec failed!\n");
                                        return NULL;
                        }
                }
                s->state_count = frame->ch_count;
                s->out->ch_count = frame->ch_count;
        }

        audio_channel channel;
        int nonzero_channels = 0;
        for(int i = 0; i < s->state_count; ++i) {
                audio_channel *encode_channel = NULL;
                if(frame) {
                        audio_channel_demux(frame, i, &channel);
                        encode_channel = &channel;
                }
                audio_channel *out = audio_codecs[s->index]->compress(s->state[i], encode_channel);
                if(!out) {
                        s->out->data_len[i] = 0;
                } else {
                        audio_channel_mux(s->out, i, out);
                        nonzero_channels += 1;
                }
        }

        if(nonzero_channels > 0) {
                return s->out;
        } else {
                return NULL;
        }
}

audio_frame2 *audio_codec_decompress(struct audio_codec_state *s, audio_frame2 *frame)
{
        if(s->state_count < frame->ch_count) {
                s->state = realloc(s->state, sizeof(void **) * frame->ch_count);
                for(int i = s->state_count; i < frame->ch_count; ++i) {
                        s->state[i] = audio_codecs[s->index]->init(s->codec, s->direction, false);
                        if(s->state[i] == NULL) {
                                        fprintf(stderr, "Error: initialization of audio codec failed!\n");
                                        return NULL;
                        }
                }
                s->state_count = frame->ch_count;
                s->out->ch_count = frame->ch_count;
        }
        audio_channel channel;
        int nonzero_channels = 0;
        for(int i = 0; i < frame->ch_count; ++i) {
                audio_channel_demux(frame, i, &channel);
                audio_channel *out = audio_codecs[s->index]->decompress(s->state[i], &channel);
                if(out) {
                        audio_channel_mux(s->out, i, out);
                        nonzero_channels += 1;
                }
        }

        if(nonzero_channels != frame->ch_count) {
                fprintf(stderr, "[Audio decompress] Empty channel returned !\n");
                return NULL;
        }
        for(int i = 0; i < frame->ch_count; ++i) {
                if(s->out->data_len[i] != s->out->data_len[0]) {
                        fprintf(stderr, "[Audio decompress] Inequal channel lenghth detected!\n");
                        return NULL;
                }
        }

        return s->out;
}

void audio_codec_done(struct audio_codec_state *s)
{
        if(!s)
                return;
        for(int i = 0; i < s->state_count; ++i) {
                audio_codecs[s->index]->done(s->state[i]);
        }
        free(s->state);

        for(int i = 0; i < MAX_AUDIO_CHANNELS; ++i) {
                s->out->data[i] = NULL;
        }
        audio_frame2_free(s->out);
        free(s);
}

const int *audio_codec_get_supported_bps(struct audio_codec_state *s)
{
        return audio_codecs[s->index]->supported_bytes_per_second;
}

