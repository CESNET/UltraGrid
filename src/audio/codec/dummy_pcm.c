/**
 * @file   audio/codec/dummy_pcm.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2025 CESNET
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

#include <assert.h>       // for assert
#include <stdbool.h>      // for bool
#include <stdint.h>       // for uint32_t
#include <stdlib.h>       // for free, malloc, NULL

#include "audio/codec.h"  // for audio_codec_direction_t, AUDIO_COMPRESS_ABI...
#include "audio/types.h"  // for audio_channel, AC_PCM, AC_NONE, audio_codec_t
#include "debug.h"        // for UNUSED
#include "lib_common.h"   // for REGISTER_MODULE, library_class

#define MAGIC 0x552bca11

static void *dummy_pcm_init(audio_codec_t audio_codec, audio_codec_direction_t direction, bool try_init,
                int bitrate);
static audio_channel *dummy_pcm_compress(void *, audio_channel *);
static audio_channel *dummy_pcm_decompress(void *, audio_channel *);
static void dummy_pcm_done(void *);

struct dummy_pcm_codec_state {
        uint32_t magic;
};

static void *dummy_pcm_init(audio_codec_t audio_codec, audio_codec_direction_t direction, bool try_init,
                int bitrate)
{
        UNUSED(direction);
        UNUSED(try_init);
        UNUSED(bitrate);
        assert(audio_codec == AC_PCM);
        struct dummy_pcm_codec_state *s = malloc(sizeof(struct dummy_pcm_codec_state));
        s->magic = MAGIC;
        return s;
}

static audio_channel *dummy_pcm_compress(void *state, audio_channel * channel)
{
        struct dummy_pcm_codec_state *s = (struct dummy_pcm_codec_state *) state;
        assert(s->magic == MAGIC);

        return channel;
}

static audio_channel *dummy_pcm_decompress(void *state, audio_channel * channel)
{
        struct dummy_pcm_codec_state *s = (struct dummy_pcm_codec_state *) state;
        assert(s->magic == MAGIC);

        return channel;
}

static const int *dummy_pcm_get_sample_rates(void *state)
{
        UNUSED(state);
        return NULL;
}

static void dummy_pcm_done(void *state)
{
        struct dummy_pcm_codec_state *s = (struct dummy_pcm_codec_state *) state;
        assert(s->magic == MAGIC);
        free(s);
}

static const struct audio_compress_info dummy_pcm_audio_codec = {
        .supported_codecs = (audio_codec_t[]){ AC_PCM, AC_NONE },
        .init = dummy_pcm_init,
        .compress = dummy_pcm_compress,
        .decompress = dummy_pcm_decompress,
        .get_samplerates = dummy_pcm_get_sample_rates,
        .done = dummy_pcm_done
};

REGISTER_MODULE(dummy_pcm,  &dummy_pcm_audio_codec, LIBRARY_CLASS_AUDIO_COMPRESS, AUDIO_COMPRESS_ABI_VERSION);


