/**
 * @file   audio/codec.cpp
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "audio/codec.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/misc.h"
#include "utils/packet_counter.h"

#include <algorithm>
#include <climits>
#include <unordered_map>

static constexpr const char *MOD_NAME = "[acodec] ";

using std::hash;
using std::max;
using std::pair;
using std::unordered_map;
using std::string;
using std::vector;

static const unordered_map<audio_codec_t, audio_codec_info_t, hash<int>>
    audio_codec_info = {
            {AC_NONE,   { "(none)", 0, 0 }                     },
            { AC_PCM,   { "PCM", 0x0001, 0 }                   },
            { AC_ALAW,  { "A-law", 0x0006, 0 }                 },
            { AC_MULAW, { "u-law", 0x0007, 0 }                 },
            { AC_SPEEX, { "speex", 0xA109, AC_FLAG_DEPRECATED }},
            { AC_OPUS,  // fcc is "Opus", the TwoCC isn't defined
              { "Opus", 0x7375704F, 0 }  },
            { AC_G722,  { "G.722", 0x028F, 0 }                 },
            { AC_MP3,   { "MP3", 0x0055, 0 }                   },
            { AC_AAC,   { "AAC", 0x00FF, 0 }                   },
            { AC_FLAC,  { "FLAC", 0xF1AC, 0 }                  },
};

static struct audio_codec_state *audio_codec_init_real(const char *audio_codec_cfg,
                audio_codec_direction_t direction, bool try_init);

struct audio_codec_state {
        void **state;
        int state_count;
        const struct audio_compress_info *funcs;
        audio_desc desc;
        audio_codec_direction_t direction;
        int bitrate;
};

vector<pair<audio_codec_info_t, bool>> get_audio_codec_list() {
        vector<pair<audio_codec_info_t, bool>> ret;

        for (auto const &it : audio_codec_info) {
                if(it.first != AC_NONE) {
                        struct audio_codec_state *st = (struct audio_codec_state *)
                                audio_codec_init_real(get_name_to_audio_codec(it.first),
                                                AUDIO_CODER, true);
                        bool available = false;
                        if(st){
                                available = true;
                                audio_codec_done(st);
                        }
                        ret.emplace_back(it.second, available);
                }
        }

        return ret;
}

void list_audio_codecs(void) {
        printf("Syntax:\n");
        col() << SBOLD(
            SRED("\t-A <codec_name>")
            << "[:sample_rate=<sampling_rate>][:bitrate=<bitrate>]\n");
        col() << "\nwhere\n";
        col() << "\t" << SBOLD("codec_name ") << " - one of the list below\n";
        col() << "\t" << SBOLD("sample_rate")
              << " - sample rate that will the codec used (may differ from\n"
              << "\t\t      captured)\n";
        col() << "\t" << SBOLD("bitrate    ") << " - codec bitrate "
              << SBOLD("per channel") << " (with optional k/M suffix)\n";
        col() << "\n";
        bool deprecated_present = false;
        printf("Supported audio codecs:\n");
        for (auto const &it : get_audio_codec_list()) {
                string notes;
                if (!it.second) {
                        notes += " " TRED("unavailable");
                }
                if ((it.first.flags & AC_FLAG_DEPRECATED) != 0) {
                        notes += " " TYELLOW("deprecated");
                        deprecated_present = true;
                }
                col() << SBOLD("\t" << it.first.name)
                      << (notes.empty() ? "" : " -") << notes << "\n";
        }
        if (deprecated_present) {
                col() << "\nCodecs marked as \"deprecated\" may be removed in future.\n";
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
        if (bitrate < 0) {
                return nullptr;
        }
        void *state = NULL;
        const struct audio_compress_info *aci = nullptr;

        auto audio_compressions = get_libraries_for_class(LIBRARY_CLASS_AUDIO_COMPRESS, AUDIO_COMPRESS_ABI_VERSION);

        for (auto const &it : audio_compressions) {
                aci = static_cast<const struct audio_compress_info *>(it.second);
                for (unsigned int j = 0; aci->supported_codecs[j] != AC_NONE; ++j) {
                        if (aci->supported_codecs[j] == audio_codec) {
                                state = aci->init(audio_codec, direction, silent, bitrate);
                                if(state) {
                                        break;
                                }
                                if (!silent) {
                                        log_msg(LOG_LEVEL_ERROR,
                                                "Error: initialization of "
                                                "audio codec failed!\n");
                                }
                        }
                }
                if(state)
                        break;
        }

        if(!state) {
                if (!silent) {
                        log_msg(LOG_LEVEL_ERROR,
                                "Unable to find encoder for audio codec '%s'\n",
                                get_name_to_audio_codec(audio_codec));
                }
                return NULL;
        }

        struct audio_codec_state *s = (struct audio_codec_state *) calloc(1, sizeof(struct audio_codec_state));

        s->state = (void **) calloc(1, sizeof(void*));
        s->state[0] = state;
        s->state_count = 1;
        s->funcs = aci;
        s->desc.codec = audio_codec;
        s->direction = direction;
        s->bitrate = bitrate;

        return s;
}

struct audio_codec_state *audio_codec_reconfigure(struct audio_codec_state *old,
                audio_codec_t audio_codec, audio_codec_direction_t direction)
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
 * This function has to be called iterativelly, in first iteration with frame, the others with NULL
 *
 * @param s state
 * @param frame in first iteration audio frame to be compressed, in following NULL
 * @retval pointer pointing to data
 * @retval NULL indicating that there are no data left
 */
audio_frame2 audio_codec_compress(struct audio_codec_state *s, const audio_frame2 *frame)
{
        audio_frame2 res;

        if (frame != nullptr) {
                if (s->state_count < frame->get_channel_count()) {
                        s->state = (void **) realloc(s->state, sizeof(void *) * frame->get_channel_count());
                        for (int i = s->state_count; i < frame->get_channel_count(); ++i) {
                                s->state[i] = s->funcs->init(s->desc.codec, s->direction, false, s->bitrate);
                                if (s->state[i] == nullptr) {
                                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Error: initialization of audio codec failed!\n";
                                        return {};
                                }
                        }
                        s->state_count = frame->get_channel_count();
                }

                s->desc.ch_count = frame->get_channel_count();
                s->desc.bps = frame->get_bps();
                s->desc.sample_rate = frame->get_sample_rate();
                res.set_timestamp(frame->get_timestamp());
        }

        audio_channel channel;
        int nonzero_channels = 0;
        for (int i = 0; i < s->desc.ch_count; ++i) {
                audio_channel *encode_channel = NULL;
                if(frame) {
                        audio_channel_demux(frame, i, &channel);
                        encode_channel = &channel;
                }
                audio_channel *out = s->funcs->compress(s->state[i], encode_channel);
                if (out == nullptr) {
                        continue;
                }
                if (!res) {
                        res.init(s->desc.ch_count, out->codec, out->bps, out->sample_rate);
                        res.set_duration(out->duration);
                } else {
                        assert(out->bps == res.get_bps()
                                        && out->sample_rate == res.get_sample_rate());
                }
                res.append(i, out->data, out->data_len);
                nonzero_channels += 1;
        }

        if (nonzero_channels == 0) {
                return {};
        }
        return res;
}

audio_frame2
audio_codec_decompress(struct audio_codec_state *s, audio_frame2 *frame,
                       packet_counter *counter)
{
        if (s->state_count < frame->get_channel_count()) {
                s->state = (void **) realloc(s->state, sizeof(void *) * frame->get_channel_count());
                for(int i = s->state_count; i < frame->get_channel_count(); ++i) {
                        s->state[i] = s->funcs->init(s->desc.codec, s->direction, false, 0);
                        if(s->state[i] == NULL) {
                                log_msg(LOG_LEVEL_ERROR,
                                        "Error: initialization of audio codec "
                                        "failed!\n");
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
                if (channel.data_len == 0) {
                        continue;
                }
                struct packet_iterator it{};
                packet_iterator_init(counter, i, &it);
                audio_channel *out = s->funcs->decompress(s->state[i], &channel, &it);
                if (out) {
                        if (!out_frame_initialized) {
                                ret.init(frame->get_channel_count(), AC_PCM, out->bps, out->sample_rate);
                                ret.set_timestamp(frame->get_timestamp());
                                out_frame_initialized = true;
                        } else {
                                assert(out->bps == ret.get_bps()
                                                && out->sample_rate == ret.get_sample_rate());
                        }
                        ret.append(i, out->data, out->data_len);
                        nonzero_channels += 1;
                }
        }
        packet_counter_clear_current_frame(counter);

        if (nonzero_channels != frame->get_channel_count()) {
                log_msg(LOG_LEVEL_WARNING,
                        "[Audio decompress] %d empty channel(s) returned!\n",
                        frame->get_channel_count() - nonzero_channels);
                return {};
        }
        int max_len = 0;
        for(int i = 0; i < frame->get_channel_count(); ++i) {
                max_len = max<int>(max_len, ret.get_data_len(i));
        }
        for(int i = 0; i < frame->get_channel_count(); ++i) {
                int len = ret.get_data_len(i);
                if (len != max_len) {
                        LOG(LOG_LEVEL_WARNING) << "[Audio decompress] Inequal channel length detected (" << ret.get_data_len(i) << " vs " << max_len << ")!\n",
                        ret.resize(i, max_len);
                        memset(ret.get_data(i) + len, 0, max_len - len);
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
        char *item, *save_ptr, *ret;

        while ((item = strtok_r(cfg, ":", &save_ptr)) != NULL) {
                if (strncasecmp(key, item, strlen(key)) == 0) {
                        ret = strdup(item + strlen(key));
                        free(tmp);
                        return ret;
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
                if (ret <= 0 && ret > INT_MAX) {
                        LOG(LOG_LEVEL_ERROR) << "Wrong bitrate: " << val << "\n";
                        return -1;
                }
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

bool check_audio_codec(const char *audio_codec_cfg)
{
        if (get_audio_codec(audio_codec_cfg) == AC_NONE) {
                LOG(LOG_LEVEL_ERROR) << "Unknown audio codec given!\n";
                return false;
        }

        auto tmp = static_cast<char *>(alloca(strlen(audio_codec_cfg) + 1));
        strcpy(tmp, audio_codec_cfg);
        char *item, *save_ptr;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != nullptr) {
                if (tmp != nullptr) { // skip first && set tmp to nullptr
                        tmp = nullptr;
                        continue;
                }
                if (strstr(item, "bitrate=") == item) {
                        continue;
                }
                if (strstr(item, "sample_rate=") == item) {
                        continue;
                }
                LOG(LOG_LEVEL_ERROR) << "Unknown audio option: " << item << "\n";
                return false;
        }
        return true;
}

