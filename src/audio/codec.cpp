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
#include "audio/utils.h"
#include "debug.h"
#include "utils/misc.h"

#include "lib_common.h"

#include <limits.h>
#include <unordered_map>

using namespace std;

static const unordered_map<audio_codec_t, audio_codec_info_t, hash<int>> audio_codec_info = {
        {AC_NONE, { "(none)", 0 }},
        {AC_PCM, { "PCM", 0x0001 }},
        {AC_ALAW, { "A-law", 0x0006 }},
        {AC_MULAW, { "u-law", 0x0007 }},
        {AC_SPEEX, { "speex", 0xA109 }},
        {AC_OPUS, { "OPUS", 0x7375704F }}, // == Opus, the TwoCC isn't defined
        {AC_G722, { "G.722", 0x028F }},
        {AC_MP3, { "MP3", 0x0055 }},
        {AC_AAC, { "AAC", 0x00FF }},
        {AC_FLAC, { "FLAC", 0xF1AC }},
};

static struct audio_codec_state *audio_codec_init_real(const char *audio_codec_cfg,
                audio_codec_direction_t direction, bool try_init);

struct audio_codec_state {
        void **state;
        int state_count;
        const struct audio_compress_info *funcs;
        audio_codec_t codec;
        audio_codec_direction_t direction;
        audio_frame2 *out;
        int bitrate;
};

void list_audio_codecs(void) {
        printf("Syntax:\n");
        printf("\t--audio-codec <audio_codec>[:sample_rate=<sampling_rate>][:bitrate=<bitrate>]\n");
        printf("\n");
        printf("Supported audio codecs:\n");
        for (auto const &it : audio_codec_info) {
                if(it.first != AC_NONE) {
                        printf("\t%s", it.second.name);
                        struct audio_codec_state *st = (struct audio_codec_state *)
                                audio_codec_init_real(get_name_to_audio_codec(it.first),
                                                AUDIO_CODER, true);
                        if(!st) {
                                printf(" - unavailable");
                        } else {
                                audio_codec_done(st);
                        }
                        printf("\n");
                }
        }
}

struct audio_codec_state *audio_codec_init(audio_codec_t audio_codec,
                audio_codec_direction_t direction) {
        return audio_codec_init_real(get_name_to_audio_codec(audio_codec), direction, false);
}

struct audio_codec_state *audio_codec_init_cfg(const char *audio_codec_cfg,
                audio_codec_direction_t direction) {
        return audio_codec_init_real(audio_codec_cfg, direction, false);
}


static struct audio_codec_state *audio_codec_init_real(const char *audio_codec_cfg,
                audio_codec_direction_t direction, bool silent) {
        audio_codec_t audio_codec = get_audio_codec(audio_codec_cfg);
        int bitrate = get_audio_codec_bitrate(audio_codec_cfg);
        void *state = NULL;
        const struct audio_compress_info *aci = nullptr;

        auto audio_compressions = get_libraries_for_class(LIBRARY_CLASS_AUDIO_COMPRESS, AUDIO_COMPRESS_ABI_VERSION);

        for (auto it : audio_compressions) {
                aci = static_cast<const struct audio_compress_info *>(it.second);
                for (unsigned int j = 0; aci->supported_codecs[j] != AC_NONE; ++j) {
                        if (aci->supported_codecs[j] == audio_codec) {
                                state = aci->init(audio_codec, direction, silent, bitrate);
                                if(state) {
                                        break;
                                } else {
                                        if (!silent) {
                                                fprintf(stderr, "Error: initialization of audio codec failed!\n");
                                        }
                                }
                        }
                }
                if(state)
                        break;
        }

        if(!state) {
                if (!silent) {
                        fprintf(stderr, "Unable to find encoder for audio codec '%s'\n",
                                        get_name_to_audio_codec(audio_codec));
                }
                return NULL;
        }

        struct audio_codec_state *s = (struct audio_codec_state *) malloc(sizeof(struct audio_codec_state));

        s->state = (void **) calloc(1, sizeof(void*));
        s->state[0] = state;
        s->state_count = 1;
        s->funcs = aci;
        s->codec = audio_codec;
        s->direction = direction;
        s->bitrate = bitrate;

        s->out = new audio_frame2;

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
const audio_frame2 *audio_codec_compress(struct audio_codec_state *s, const audio_frame2 *frame)
{
        if(frame && s->state_count < frame->get_channel_count()) {
                s->state = (void **) realloc(s->state, sizeof(void *) * frame->get_channel_count());
                for(int i = s->state_count; i < frame->get_channel_count(); ++i) {
                        s->state[i] = s->funcs->init(s->codec, s->direction, false, s->bitrate);
                        if(s->state[i] == NULL) {
                                        fprintf(stderr, "Error: initialization of audio codec failed!\n");
                                        return NULL;
                        }
                }
                s->state_count = frame->get_channel_count();
        }

        audio_channel channel;
        int nonzero_channels = 0;
        bool out_frame_initialized = false;
        for (int i = 0; i < s->state_count; ++i) {
                audio_channel *encode_channel = NULL;
                if(frame) {
                        audio_channel_demux(frame, i, &channel);
                        encode_channel = &channel;
                }
                audio_channel *out = s->funcs->compress(s->state[i], encode_channel);
                if (out) {
                        if (!out_frame_initialized) {
                                if (frame) {
                                        s->out->init(frame->get_channel_count(), s->codec, out->bps, out->sample_rate);
                                } else {
                                        s->out->reset();
                                }
                                s->out->set_duration(out->duration);
                                out_frame_initialized = true;
                        } else {
                                assert(out->bps == s->out->get_bps()
                                                && out->sample_rate == s->out->get_sample_rate());
                        }
                        s->out->append(i, out->data, out->data_len);
                        nonzero_channels += 1;
                }
        }

        if(nonzero_channels > 0) {
                return s->out;
        } else {
                return NULL;
        }
}

audio_frame2 audio_codec_decompress(struct audio_codec_state *s, audio_frame2 *frame)
{
        if (s->state_count < frame->get_channel_count()) {
                s->state = (void **) realloc(s->state, sizeof(void *) * frame->get_channel_count());
                for(int i = s->state_count; i < frame->get_channel_count(); ++i) {
                        s->state[i] = s->funcs->init(s->codec, s->direction, false, 0);
                        if(s->state[i] == NULL) {
                                        fprintf(stderr, "Error: initialization of audio codec failed!\n");
                                        return {};
                        }
                }
                s->state_count = frame->get_channel_count();
        }

#if 0
        if (s->out->ch_count != frame->ch_count) {
                s->out->ch_count = frame->ch_count;
        }
#endif

        audio_frame2 ret;
        audio_channel channel;
        int nonzero_channels = 0;
        bool out_frame_initialized = false;
        for (int i = 0; i < frame->get_channel_count(); ++i) {
                audio_channel_demux(frame, i, &channel);
                audio_channel *out = s->funcs->decompress(s->state[i], &channel);
                if (out) {
                        if (!out_frame_initialized) {
                                ret.init(frame->get_channel_count(), AC_PCM, out->bps, out->sample_rate);
                                out_frame_initialized = true;
                        } else {
                                assert(out->bps == ret.get_bps()
                                                && out->sample_rate == ret.get_sample_rate());
                        }
                        ret.append(i, out->data, out->data_len);
                        nonzero_channels += 1;
                }
        }

        if(nonzero_channels != frame->get_channel_count()) {
                fprintf(stderr, "[Audio decompress] Empty channel returned !\n");
                return {};
        }
        /// @todo
        /// Drop the whole frame? It seems to be better to avoid filling remaining with silence
        /// since it
        /// could cause problems if we didn't miss any data but we are processing data from
        /// different channels not simultaneously.
        for(int i = 1; i < frame->get_channel_count(); ++i) {
                if(ret.get_data_len(i) != ret.get_data_len(0)) {
                        fprintf(stderr, "[Audio decompress] Inequal channel lenghth detected (%zd vs %zd)!\n",
                                        ret.get_data_len(0), ret.get_data_len(i));
                        return {};
                }
        }

        return ret;
}

const int *audio_codec_get_supported_samplerates(struct audio_codec_state *s)
{
        return s->funcs->get_samplerates(s->state[0]);
}

void audio_codec_done(struct audio_codec_state *s)
{
        if(!s)
                return;
        for(int i = 0; i < s->state_count; ++i) {
                s->funcs->done(s->state[i]);
        }
        free(s->state);

        delete s->out;
        free(s);
}

audio_codec_t get_audio_codec(const char *codec_str) {
        char *codec = strdup(codec_str);
        if (strchr(codec, ':')) {
                *strchr(codec, ':') = '\0';
        }
        for (auto const &it : audio_codec_info) {
                if(strcasecmp(it.second.name, codec) == 0) {
                        free(codec);
                        return it.first;
                }
        }
        free(codec);
        return AC_NONE;
}

/**
 * Caller must free() the returned buffer
 */
static char *get_val_from_cfg(const char *audio_codec_cfg, const char *key)
{
        char *cfg = strdup(audio_codec_cfg);
        char *tmp = cfg;
        char *item, *save_ptr;

        while ((item = strtok_r(cfg, ":", &save_ptr)) != NULL) {
                if (strncasecmp(key, item, strlen(key)) == 0) {
                        free(tmp);
                        return strdup(item + strlen(key));
                }
                cfg = NULL;
        }
        free(tmp);
        return NULL;
}

/**
 * @returns user specified sample rate or 0 if unspecified
 */
int get_audio_codec_sample_rate(const char *audio_codec_cfg)
{
        char *val = get_val_from_cfg(audio_codec_cfg, "sample_rate=");
        if (val) {
                int ret =  atoi(val);
                free(val);
                return ret;
        } else {
                return 0;
        }
}

int get_audio_codec_bitrate(const char *audio_codec_cfg)
{
        char *val = get_val_from_cfg(audio_codec_cfg, "bitrate=");
        if (val) {
                long long ret =  unit_evaluate(val);
                assert(ret > 0 && ret <= INT_MAX);
                free(val);
                return ret;
        } else {
                return 0;
        }
}

const char *get_name_to_audio_codec(audio_codec_t codec)
{
        return audio_codec_info.at(codec).name;
}

uint32_t get_audio_tag(audio_codec_t codec)
{
        return audio_codec_info.at(codec).tag;
}

audio_codec_t get_audio_codec_to_tag(uint32_t tag)
{
        for (auto const &it : audio_codec_info) {
                if(it.second.tag == tag) {
                        return it.first;
                }
        }
        return AC_NONE;
}

