/**
 * @file   audio/codec.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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

#include "audio/codec.h"

#include <assert.h>  // for assert
#include <limits.h>  // for INT_MAX
#include <stdio.h>   // for printf
#include <stdlib.h>  // for calloc, free, realloc, atoi
#include <string.h>  // for strchr, strlen, memset, strcmp, strtok_r
#include <strings.h> // for strcasecmp

#include "audio/types.h"     // for audio_desc, AC_NONE, audio_codec_t, aud...
#include "audio/utils.h"     // for audio_channel_demux
#include "compat/c23.h"      // IWYU pragma: keep for countof
#include "debug.h"           // for LOG_LEVEL_ERROR, MSG, log_msg, LOG_LEVE...
#include "lib_common.h"      // for class_modules, class_modules::(anonymous)
#include "utils/color_out.h" // for color_printf, TBOLD, TRED
#include "utils/macros.h"    // for strcpy_ch, IS_KEY_PREFIX, MAX
#include "utils/misc.h"      // for unit_evaluate

struct audio_frame2;

#define MOD_NAME "[acodec] "

static const struct ac_info {
        const char *name;
        uint32_t    audio_tag; // usually twocc
} audio_codec_info[] = {
        [AC_NONE]  = { "(none)", 0          },
        [AC_PCM]   = { "PCM",    0x0001     },
        [AC_ALAW]  = { "A-law",  0x0006     },
        [AC_MULAW] = { "u-law",  0x0007     },
        [AC_SPEEX] = { "speex",  0xA109     },
        // fcc is "Opus", TwoCC isn't defined for Opus
        [AC_OPUS] = { "Opus",   0x7375704F },
        [AC_G722] = { "G.722",  0x028F     },
        [AC_MP3]  = { "MP3",    0x0055     },
        [AC_AAC]  = { "AAC",    0x00FF     },
        [AC_FLAC] = { "FLAC",   0xF1AC     },
};

struct audio_codec_state {
        void                            **state;
        int                               state_count;
        const struct audio_compress_info *funcs;
        struct audio_desc                 desc;
        audio_codec_direction_t           direction;
        int                               bitrate;
};

static void
get_codec_desc(struct audio_codec_state *st, size_t buflen,
               char buf[static buflen])
{
        buf[0] = '\0';
        if (st->funcs->get_codec_info == nullptr) {
                return;
        }
        st->funcs->get_codec_info(st->state[0], buflen, buf);
}

struct audio_codec_list
get_audio_codec_list()
{
        struct audio_codec_list ret = { .count = 0 };

        for (unsigned i = 0; i < countof(audio_codec_info); ++i) {
                const struct ac_info *ai = &audio_codec_info[i];
                if (ai->audio_tag == 0) {
                        continue;
                }
                strcpy_ch(ret.item[ret.count++], ai->name);
        }

        return ret;
}

static struct audio_codec_state *
audio_codec_init_real(const char             *audio_codec_cfg,
                      audio_codec_direction_t direction, bool silent)
{
        const struct audio_codec_params params =
            parse_audio_codec_params(audio_codec_cfg);
        if (params.codec == AC_NONE) {
                return nullptr;
        }
        void                             *state = NULL;
        const struct audio_compress_info *aci   = nullptr;

        struct class_modules audio_compressions = get_libraries_for_class(
            LIBRARY_CLASS_AUDIO_COMPRESS, AUDIO_COMPRESS_ABI_VERSION);

        for (unsigned i = 0; i < audio_compressions.count; i++) {
                aci = audio_compressions.item[i].info;
                for (unsigned int j = 0; aci->supported_codecs[j] != AC_NONE;
                     ++j) {
                        if (aci->supported_codecs[j] != params.codec) {
                                continue;
                        }
                        state = aci->init(params.codec, direction,
                                          silent, params.bitrate);
                        if (state) {
                                goto found;
                        }
                        if (!silent) {
                                log_msg(LOG_LEVEL_ERROR,
                                        "Error: initialization of "
                                        "audio codec failed!\n");
                        }
                }
        }
found:

        if (!state) {
                if (!silent) {
                        log_msg(LOG_LEVEL_ERROR,
                                "Unable to find encoder for audio codec '%s'\n",
                                get_audio_codec_name(params.codec));
                }
                return nullptr;
        }

        struct audio_codec_state *s = (struct audio_codec_state *) calloc(
            1, sizeof(struct audio_codec_state));

        s->state       = (void **) calloc(1, sizeof(void *));
        s->state[0]    = state;
        s->state_count = 1;
        s->funcs       = aci;
        s->desc.codec  = params.codec;
        s->direction   = direction;
        s->bitrate     = params.bitrate;

        return s;
}

void
list_audio_codecs(void)
{
        printf("Syntax:\n");
        color_printf(
            TBOLD(TRED("\t-A <codec_name>") "[:sample_rate=<sampling_rate>][:"
                                            "bitrate=<bitrate>]\n"));
        color_printf("\nwhere\n");
        color_printf("\t" TBOLD("codec_name ")
                     " - one of the list below\n");
        color_printf(
            "\t" TBOLD("sample_rate")
            " - sample rate that will the codec used (may differ from\n"
            "\t\t      captured)\n");
        color_printf("\t" TBOLD("bitrate    ")
                     " - codec bitrate " TBOLD("per channel")
                     " (with optional k/M suffix)\n");
        color_printf("\n");

        printf("Supported audio codecs:\n");
        for (unsigned i = 0; i < countof(audio_codec_info); ++i) {
                const struct ac_info *ai = &audio_codec_info[i];
                if (ai->audio_tag == 0) {
                        continue;
                }
                struct audio_codec_state *st = audio_codec_init_real(
                    get_audio_codec_name(i), AUDIO_CODER, true);
                char notes[128] = TRED("unavailable");
                if (st) {
                        get_codec_desc(st, sizeof notes, notes);
                        audio_codec_done(st);
                }
                color_printf(TBOLD("\t%s")
                             "%s%s\n",
                             ai->name, strlen(notes) > 0 ? " - " : "", notes);
        }
        color_printf(
            "\nCodecs marked as \"deprecated\" may be removed in future.\n"
            "Codec coder marked with '*' is default.\n");
}

struct audio_codec_state *
audio_codec_init(audio_codec_t audio_codec, audio_codec_direction_t direction)
{
        return audio_codec_init_real(get_audio_codec_name(audio_codec),
                                     direction, false);
}

struct audio_codec_state *
audio_codec_init_cfg(const char             *audio_codec_cfg,
                     audio_codec_direction_t direction)
{
        return audio_codec_init_real(audio_codec_cfg, direction, false);
}

struct audio_codec_state *
audio_codec_reconfigure(struct audio_codec_state *old,
                        audio_codec_t             audio_codec,
                        audio_codec_direction_t   direction)
{
        if (old != nullptr && old->desc.codec == audio_codec) {
                return old;
        }
        audio_codec_done(old);
        return audio_codec_init(audio_codec, direction);
}

/**
 * Audio_codec_compress compresses given audio frame.
 *
 * This function has to be called iterativelly, in first iteration with frame,
 * the others with NULL
 *
 * @param s state
 * @param frame in first iteration audio frame to be compressed, in following
 * NULL
 * @retval pointer pointing to data
 * @retval NULL indicating that there are no data left
 */
struct audio_frame2 *
audio_codec_compress(struct audio_codec_state  *s,
                     const struct audio_frame2 *frame)
{
        if (frame != nullptr) {
                int ch_count = audio_frame2_get_channel_count(frame);
                if (s->state_count < ch_count) {
                        s->state = (void **) realloc(s->state,
                                                     sizeof(void *) * ch_count);
                        for (int i = s->state_count; i < ch_count; ++i) {
                                s->state[i] =
                                    s->funcs->init(s->desc.codec, s->direction,
                                                   false, s->bitrate);
                                if (s->state[i] == nullptr) {
                                        MSG(ERROR, "Error: initialization of "
                                                   "audio codec failed!\n");
                                        return nullptr;
                                }
                        }
                        s->state_count = ch_count;
                }

                s->desc.ch_count    = ch_count;
                s->desc.bps         = audio_frame2_get_bps(frame);
                s->desc.sample_rate = audio_frame2_get_sample_rate(frame);
        }

        struct audio_frame2 *res = nullptr;

        audio_channel channel;
        channel.timestamp =
            frame == nullptr ? -1 : audio_frame2_get_timestamp(frame);
        int nonzero_channels = 0;
        for (int i = 0; i < s->desc.ch_count; ++i) {
                audio_channel *encode_channel = NULL;
                if (frame) {
                        audio_channel_demux(frame, i, &channel);
                        encode_channel = &channel;
                }
                audio_channel *out =
                    s->funcs->compress(s->state[i], encode_channel);
                if (out == nullptr) {
                        continue;
                }
                if (!res) {
                        res = audio_frame2_alloc(s->desc.ch_count, out->codec,
                                                 out->bps, out->sample_rate);
                        if (frame != nullptr) {
                                audio_frame2_set_timestamp(
                                    res, audio_frame2_get_timestamp(frame));
                        }
                        audio_frame2_set_duration(res, out->duration);
                } else {
                        assert(out->bps == audio_frame2_get_bps(res) &&
                               out->sample_rate ==
                                   audio_frame2_get_sample_rate(res));
                }
                audio_frame2_append_channel(res, i, out->data, out->data_len);
                audio_frame2_set_timestamp(res, out->timestamp);
                nonzero_channels += 1;
        }

        if (nonzero_channels == 0) {
                audio_frame2_delete(res);
                return nullptr;
        }
        return res;
}

struct audio_frame2 *
audio_codec_decompress(struct audio_codec_state *s, struct audio_frame2 *frame)
{
        int ch_count = audio_frame2_get_channel_count(frame);
        if (s->state_count < ch_count) {
                s->state =
                    (void **) realloc(s->state, sizeof(void *) * ch_count);
                for (int i = s->state_count; i < ch_count; ++i) {
                        s->state[i] = s->funcs->init(s->desc.codec,
                                                     s->direction, false, 0);
                        if (s->state[i] == NULL) {
                                log_msg(LOG_LEVEL_ERROR,
                                        "Error: initialization of audio codec "
                                        "failed!\n");
                                return nullptr;
                        }
                }
                s->state_count = ch_count;
        }

#if 0
        if (s->out->ch_count != frame->ch_count) {
                s->out->ch_count = frame->ch_count;
        }
#endif

        struct audio_frame2 *ret = nullptr;
        audio_channel        channel;
        int                  nonzero_channels = 0;
        int in_ch_count = audio_frame2_get_channel_count(frame);
        for (int i = 0; i < in_ch_count; ++i) {
                audio_channel_demux(frame, i, &channel);
                if (channel.data_len == 0) {
                        continue;
                }
                audio_channel *out =
                    s->funcs->decompress(s->state[i], &channel);
                if (out) {
                        if (ret == nullptr) {
                                ret = audio_frame2_alloc(in_ch_count, AC_PCM,
                                                         out->bps,
                                                         out->sample_rate);
                                audio_frame2_set_timestamp(
                                    frame, audio_frame2_get_timestamp(frame));
                        } else {
                                assert(out->bps == audio_frame2_get_bps(ret) &&
                                       out->sample_rate ==
                                           audio_frame2_get_sample_rate(ret));
                        }
                        audio_frame2_append_channel(ret, i, out->data,
                                                    out->data_len);
                        nonzero_channels += 1;
                }
        }

        if (nonzero_channels == 0 && audio_frame2_get_all_data_len(frame) ==
                                         0) { // produced by acap/passive
                ret = audio_frame2_alloc(in_ch_count, AC_PCM,
                                         audio_frame2_get_bps(frame),
                                         audio_frame2_get_sample_rate(frame));
                return ret;
        }

        if (nonzero_channels != in_ch_count) {
                log_msg(LOG_LEVEL_WARNING,
                        "[Audio decompress] %d empty channel(s) returned!\n",
                        in_ch_count - nonzero_channels);
                audio_frame2_delete(ret);
                return nullptr;
        }
        int max_len = 0;
        for (int i = 0; i < in_ch_count; ++i) {
                max_len = MAX(max_len, (int) audio_frame2_get_data_len(ret, i));
        }
        for (int i = 0; i < in_ch_count; ++i) {
                int len = (int) audio_frame2_get_data_len(ret, i);
                if (len != max_len) {
                        MSG(WARNING,
                            "Inequal channel length detected (%d vs %d)!\n",
                            len, max_len);
                        audio_frame2_resize(ret, i, max_len);
                        memset(audio_frame2_get_data(ret, i) + len, 0,
                               max_len - len);
                }
        }

        return ret;
}

const int *
audio_codec_get_supported_samplerates(struct audio_codec_state *s)
{
        return s->funcs->get_samplerates(s->state[0]);
}

void
audio_codec_done(struct audio_codec_state *s)
{
        if (!s) {
                return;
        }

        for (int i = 0; i < s->state_count; ++i) {
                s->funcs->done(s->state[i]);
        }
        free((void *) s->state);
        free(s);
}

static audio_codec_t
get_audio_codec(const char *codec)
{
        for (unsigned i = 0; i < countof(audio_codec_info); ++i) {
                const struct ac_info *ai = &audio_codec_info[i];
                if (ai->name != nullptr && !strcmp(ai->name, codec)) {
                        return i;
                }
        }
        // aliases
        if (strcasecmp(codec, "PCMA") == 0) {
                return AC_ALAW;
        }
        if (strcasecmp(codec, "PCMU") == 0) {
                return AC_MULAW;
        }
        return AC_NONE;
}

struct audio_codec_params
parse_audio_codec_params(const char *ccfg)
{
        assert(ccfg != nullptr);
        struct audio_codec_params params = { 0 };

        char cfg[strlen(ccfg) + 1];
        strcpy_ch(cfg, ccfg);
        char *tmp    = cfg;
        char *tok    = nullptr;
        char *endptr = nullptr;
        while ((tok = strtok_r(tmp, ":", &endptr)) != nullptr) {
                tmp = nullptr;
                if (params.codec == AC_NONE) {
                        params.codec = get_audio_codec(tok);
                        if (params.codec == AC_NONE) {
                                MSG(ERROR,
                                    "Unknown audio codec \"%s\" given!\n", tok);
                                return (struct audio_codec_params){ 0 };
                        }
                        continue;
                }
                if (IS_KEY_PREFIX(tok, "sample_rate")) {
                        const char *val    = strchr(tok, '=') + 1;
                        params.sample_rate = atoi(val);
                        if (params.sample_rate <= 0) {
                                MSG(ERROR,
                                    "Sample rate must be positive! given "
                                    "%s\n",
                                    val);
                                return (struct audio_codec_params){ 0 };
                        }
                } else if (IS_KEY_PREFIX(tok, "bitrate")) {
                        const char *val    = strchr(tok, '=') + 1;
                        const char *endptr = nullptr;
                        long long   rate   = unit_evaluate(val, &endptr);
                        if (rate <= 0 || rate > INT_MAX || endptr[0] != '\0') {
                                MSG(ERROR, "Wrong bitrate: %lld bps (%s)\n",
                                    rate, val);
                                return (struct audio_codec_params){ 0 };
                        }
                        params.bitrate = (int) rate;
                } else {
                        MSG(ERROR, "Unknown audio option: %s\n", tok);
                        return (struct audio_codec_params){ 0 };
                }
        }
        return params;
}

const char *
get_audio_codec_name(audio_codec_t codec)
{
        return audio_codec_info[codec].name;
}

uint32_t
get_audio_tag(audio_codec_t codec)
{
        return audio_codec_info[codec].audio_tag;
}

audio_codec_t
get_audio_codec_to_tag(uint32_t audio_tag)
{
        for (unsigned i = 0; i < countof(audio_codec_info); ++i) {
                const struct ac_info *ai = &audio_codec_info[i];
                if (ai->audio_tag == audio_tag) {
                        return i;
                }
        }
        return AC_NONE;
}
