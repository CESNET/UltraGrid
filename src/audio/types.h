/**
 * @file   audio/types.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This file contains definition of some common audio types.
 */
/*
 * Copyright (c) 2011-2021 CESNET, z. s. p. o.
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

#include "../types.h" // fec_desc

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
        bool operator==(audio_desc const &) const;
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
        int max_size;           /* maximal size of data in buffer */
        void *network_source;   /* pointer to sockaddr_storage containing
                                   network source that the audio stream came
                                   from. Relevant only for receiver. */
        void (*dispose)(struct audio_frame *); ///< called by sender when frame was processed
        void *dispose_udata;    ///< additional data that may the caller use in dispose
}
audio_frame;

#define AUDIO_FRAME_DISPOSE(frame) if ((frame) && (frame)->dispose) \
        (frame)->dispose(frame)

/// Audio channel is a non-owning view on audio_frame2 isolating one channel.
typedef struct
{
        int bps;                /* bytes per sample */
        int sample_rate;
        const char *data;       /* data should be at least 4B aligned */
        int data_len;           /* size of useful data in buffer */
        audio_codec_t codec;
        double duration;
} audio_channel;

#ifdef __cplusplus
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

class audio_frame2;

class audio_frame2_resampler {
public:
        ~audio_frame2_resampler();
        int get_resampler_numerator();
        int get_resampler_denominator();
        int get_resampler_output_latency();
        int get_resampler_input_latency();
        int get_resampler_from_sample_rate();
        int get_resampler_initial_bps();
        size_t get_resampler_channel_count();
        bool resampler_is_set();
        bool create_resampler(uint32_t original_sample_rate, uint32_t new_sample_rate_num, uint32_t new_sample_rate_den, size_t channel_size, int bps);
        void resample_set_destroy_flag(bool destroy);
private:
        void *resampler{nullptr}; // type is (SpeexResamplerState *)
        int resample_from{0};
        int resample_to_num{0};
        int resample_to_den{1};
        int resample_output_latency{0};
        int resample_input_latency{0};
        int resample_initial_bps{0};
        size_t resample_ch_count{0};
        bool destroy_resampler{false};
        bool initial{}; 

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
        audio_frame2(audio_frame2 const &) = delete;
        audio_frame2(audio_frame2 &&) = default;
        explicit audio_frame2(const struct audio_frame *);
        audio_frame2& operator=(audio_frame2 const &) = delete;
        audio_frame2& operator=(audio_frame2 &&) = default;
        bool operator!() const;
        explicit operator bool() const;
        void init(int nr_channels, audio_codec_t codec, int bps, int sample_rate);
        void append(audio_frame2 const &frame);
        void append(int channel, const char *data, size_t length);
        void replace(int channel, size_t offset, const char *data, size_t length);
        void reserve(size_t len);
        void resize(int channel, size_t len);
        void reset();
        int get_bps() const;
        audio_codec_t get_codec() const;
        char *get_data(int channel);
        const char *get_data(int channel) const;
        size_t get_data_len(int channel) const;
        size_t get_data_len() const;
        double get_duration() const;
        int get_channel_count() const;
        fec_desc const &get_fec_params(int channel) const;
        int get_sample_count() const;
        int get_sample_rate() const;
        bool has_same_prop_as(audio_frame2 const &frame) const;
        void set_duration(double duration);
        void set_fec_params(int channel, fec_desc const &);
        static audio_frame2 copy_with_bps_change(audio_frame2 const &frame, int new_bps);
        void change_bps(int new_bps);
        void convert_int32_to_float();
        void convert_float_to_int32();

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
         * @retval false          if SpeexDSP was not compiled in
         */
        std::tuple<bool, bool> resample(audio_frame2_resampler &resampler_state, int new_sample_rate);

        ///@ resamples to new sample rate while keeping nominal sample rate intact
        std::tuple<bool, bool, audio_frame2> resample_fake(audio_frame2_resampler & resampler_state, int new_sample_rate_num, int new_sample_rate_den);
private:
        struct channel {
                std::unique_ptr<char []> data;
                size_t len;
                size_t max_len;
                struct fec_desc fec_params;
        };
        static void resample_channel(audio_frame2_resampler* resampler_state, int channel_index, 
                                     const uint16_t *in, uint32_t in_len, channel *new_channel, audio_frame2 *remainder);
        static void resample_channel_float(audio_frame2_resampler* resampler_state, int channel_index, 
                                           const float *in, uint32_t in_len, channel *new_channel, audio_frame2 *remainder);
        void reserve(int channel, size_t len);
        int bps;                /* bytes per sample */
        int sample_rate;
        std::vector<channel> channels; /* data should be at least 4B aligned */
        audio_codec_t codec;
        double duration; ///< for compressed formats where this cannot be directly determined from samples/sample_rate
};

#endif // __cplusplus

#endif // defined AUDIO_TYPES_H

