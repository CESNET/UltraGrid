/**
 * @file   audio/codec.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2023 CESNET, z. s. p. o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
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
 */

#ifndef AUDIO_CODEC_H_
#define AUDIO_CODEC_H_

#include "audio/types.h"

#define AUDIO_COMPRESS_ABI_VERSION 3

typedef enum {
        AUDIO_CODER,
        AUDIO_DECODER
} audio_codec_direction_t;

struct audio_compress_info {
        const audio_codec_t *supported_codecs;
        void *(*init)(audio_codec_t, audio_codec_direction_t, bool, int bitrate);
        audio_channel *(*compress)(void *, audio_channel *);
        audio_channel *(*decompress)(void *, audio_channel *);
        const int *(*get_samplerates)(void *);
        void (*done)(void *);
};

typedef struct {
        const char *name;
        /** @var tag
         *  @brief TwoCC if defined, otherwise we define our tag
         */
        uint32_t    tag;
} audio_codec_info_t;

#ifdef __cplusplus
struct audio_codec_state;

struct audio_codec_state *audio_codec_init(audio_codec_t audio_codec, audio_codec_direction_t);
struct audio_codec_state *audio_codec_init_cfg(const char *audio_codec_cfg, audio_codec_direction_t);
struct audio_codec_state *audio_codec_reconfigure(struct audio_codec_state *old,
                audio_codec_t audio_codec, audio_codec_direction_t);
audio_frame2 audio_codec_compress(struct audio_codec_state *, const audio_frame2 *);
audio_frame2 audio_codec_decompress(struct audio_codec_state *, audio_frame2 *);
const int *audio_codec_get_supported_samplerates(struct audio_codec_state *);
void audio_codec_done(struct audio_codec_state *);

std::vector<std::pair<std::string, bool>> get_audio_codec_list(void);
#endif

#ifdef __cplusplus
extern "C" {
#endif // defined __cplusplus
void list_audio_codecs(void);

audio_codec_t get_audio_codec(const char *audio_codec_cfg);
int get_audio_codec_sample_rate(const char *audio_codec_cfg);
int get_audio_codec_bitrate(const char *audio_codec_cfg);
const char *get_name_to_audio_codec(audio_codec_t codec);
uint32_t get_audio_tag(audio_codec_t codec);
audio_codec_t get_audio_codec_to_tag(uint32_t audio_tag);
bool check_audio_codec(const char *audio_codec_cfg);

#ifdef __cplusplus
}
#endif // defined __cplusplus

#endif /* AUDIO_CODEC_H */
