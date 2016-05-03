/**
 * @file   audio/types.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This file contains definition of some common audio types.
 */
/*
 * Copyright (c) 2011-2015 CESNET, z. s. p. o.
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

#ifndef AUDIO_TYPES_H
#define AUDIO_TYPES_H

typedef enum {
        AC_NONE,
        AC_PCM,
        AC_ALAW,
        AC_MULAW,
        AC_SPEEX,
        AC_OPUS,
        AC_G722,
        AC_MP3,
        AC_AAC,
        AC_FLAC,
} audio_codec_t;

#ifdef __cplusplus
#include <string>
#endif

struct audio_desc {
        int bps;                /* bytes per sample */
        int sample_rate;
        int ch_count;		/* count of channels */
        audio_codec_t codec;
#ifdef __cplusplus
        bool operator!() const;
        operator std::string() const;
#endif
};

#ifdef __cplusplus
#include <ostream>
std::ostream& operator<<(std::ostream& os, const audio_desc& desc);
#endif

/**
 * @brief struct used to comunicate with audio capture/playback device
 *
 * Always hold uncompressed (PCM) packed interleaved signed data.
 */
typedef struct audio_frame
{
        int bps;                /* bytes per sample */
        int sample_rate;
        char *data; /* data should be at least 4B aligned */
        int data_len;           /* size of useful data in buffer */
        int ch_count;		/* count of channels */
        unsigned int max_size;  /* maximal size of data in buffer */
}
audio_frame;

typedef struct
{
        int bps;                /* bytes per sample */
        int sample_rate;
        const char *data; /* data should be at least 4B aligned */
        int data_len;           /* size of useful data in buffer */
        audio_codec_t codec;
        double duration;
} audio_channel;

#ifdef __cplusplus
#include <memory>
#include <utility>
#include <vector>

class audio_frame2;

class audio_frame2_resampler {
public:
        audio_frame2_resampler();
        ~audio_frame2_resampler();
private:
        void *resampler; // type is (SpeexResamplerState *)
        int resample_from;
        size_t resample_ch_count;
        int resample_to;

        friend class audio_frame2;
};

/**
 * More versatile than audio_frame
 *
 * Can hold also compressed audio data. Audio channels are non-interleaved.
 */
class audio_frame2
{
public:
        audio_frame2();
        audio_frame2(audio_frame2 &&) = default;
        explicit audio_frame2(const struct audio_frame *);
        audio_frame2& operator=(audio_frame2 &&) = default;
        bool operator!() const;
        void init(int nr_channels, audio_codec_t codec, int bps, int sample_rate);
        void append(audio_frame2 const &frame);
        void append(int channel, const char *data, size_t length);
        void replace(int channel, size_t offset, const char *data, size_t length);
        void reserve(size_t len);
        void resize(int channel, size_t len);
        void reset();
        int get_bps() const;
        audio_codec_t get_codec() const;
        const char *get_data(int channel) const;
        size_t get_data_len(int channel) const;
        double get_duration() const;
        int get_channel_count() const;
        int get_sample_count() const;
        int get_sample_rate() const;
        bool has_same_prop_as(audio_frame2 const &frame) const;
        void set_duration(double duration);
        static audio_frame2 copy_with_bps_change(audio_frame2 const &frame, int new_bps);
        void change_bps(int new_bps);
        /**
         * @note
         * bps of the frame needs to be 16 bits!
         *
         * @param resampler_state opaque state that can holds resampler that dosn't need
         *                        to be reinitalized during calls on various audio frames.
         *                        It reinitializes itself when needed (when source or new
         *                        sample rate changes). Therefore, it is very recommended
         *                        to use it only in a stream that may change sometimes but
         *                        do not eg. share it between two streams that has different
         *                        properties.
         */
        void resample(audio_frame2_resampler &resampler_state, int new_sample_rate);
private:
        struct channel {
                std::unique_ptr<char []> data;
                size_t len;
                size_t max_len;
        };
        void reserve(int channel, size_t len);
        int bps;                /* bytes per sample */
        int sample_rate;
        std::vector<channel> channels; /* data should be at least 4B aligned */
        audio_codec_t codec;
        double duration; /// @note currently unused
};
#endif // __cplusplus

#endif // defined AUDIO_TYPES_H

