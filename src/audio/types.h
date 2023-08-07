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

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

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
        AC_COUNT,
} audio_codec_t;

/// packet format maximal allowed values defined in:
/// http://www.sitola.cz/files/4K-packet-format.pdf
enum {
        MAX_AUD_CH_COUNT    = 1 << 10,
        MAX_AUD_SAMPLE_RATE = 1 << 24,
};

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
        uint32_t timestamp;     ///< frame timestamp
        int flags;
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

enum {
        TX_MAX_AUDIO_PACKETS = 16,
};
struct audio_tx_pkt {
        char *data;
        int   len;
        struct fec_desc fec_desc;
};
struct audio_tx_channel {
        int pkt_count;
        struct audio_tx_pkt pkts[TX_MAX_AUDIO_PACKETS];
};
struct audio_tx_data {
        struct audio_desc        desc;
        int64_t                  timestamp; ///< -1 if not valid
        struct audio_tx_channel *channels;
};
void audio_tx_data_cleanup(struct audio_tx_data *d);

#ifdef __cplusplus
#include <memory>
#include <tuple>
#include <vector>

class audio_frame2_resampler;

/**
 * More versatile than audio_frame
 *
 * Can hold also compressed audio data. Audio channels are non-interleaved.
 */
class audio_frame2
{
public:
        audio_frame2() = default;
        ~audio_frame2() = default;
        audio_frame2(audio_frame2 const &) = delete;
        audio_frame2(audio_frame2 &&) = default;
        explicit audio_frame2(const struct audio_frame *);
        audio_frame2& operator=(audio_frame2 const &) = delete;
        audio_frame2& operator=(audio_frame2 &&) = default;
        explicit operator bool() const;
        void init(int nr_channels, audio_codec_t codec, int bps, int sample_rate);
        void append(audio_frame2 const &frame);
        void append(int channel, const char *data, size_t length);
        void replace(int channel, size_t offset, const char *data, size_t length);
        void reserve(size_t len);
        void resize(int channel, size_t len);
        void reset();
        void set_timestamp(int64_t ts);
        [[nodiscard]] int get_bps() const;
        [[nodiscard]] audio_codec_t get_codec() const;
        [[nodiscard]] char *get_data(int channel);
        [[nodiscard]] audio_desc get_desc() const;
        [[nodiscard]] const char *get_data(int channel) const;
        [[nodiscard]] size_t get_data_len(int channel) const;
        [[nodiscard]] size_t get_data_len() const;
        [[nodiscard]] double get_duration() const;
        [[nodiscard]] int get_channel_count() const;
        [[nodiscard]] fec_desc const &get_fec_params(int channel) const;
        [[nodiscard]] int get_sample_count() const;
        [[nodiscard]] int get_sample_rate() const;
        [[nodiscard]] int64_t get_timestamp() const; // u32 TS, -1 if unavail
        [[nodiscard]] bool has_same_prop_as(audio_frame2 const &frame) const;
        void set_duration(double duration);
        void set_fec_params(int channel, fec_desc const &);
        [[nodiscard]] static audio_frame2 copy_with_bps_change(audio_frame2 const &frame, int new_bps);
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
         * @retval false          if SpeexDSP was not compiled in
         */
        bool resample(audio_frame2_resampler &resampler_state, int new_sample_rate);
        ///@ resamples to new sample rate while keeping nominal sample rate intact
        std::tuple<bool, audio_frame2> resample_fake(audio_frame2_resampler & resampler_state, int new_sample_rate_num, int new_sample_rate_den);
        audio_tx_data *append_tx_data(audio_tx_data *src);
private:
        struct channel {
                std::unique_ptr<char []> data;
                size_t len;
                size_t max_len;
                struct fec_desc fec_params;
        };
        void reserve(int channel, size_t len);
        struct audio_desc
            desc = {}; ///< desc.ch_count not set, use channels.size() instead
        std::vector<channel> channels; /* data should be at least 4B aligned */
        double duration = 0.0; ///< for compressed formats where this cannot be directly determined from samples/sample_rate
        int64_t timestamp = -1;

        friend class audio_frame2_resampler;
        friend class soxr_resampler;
        friend class speex_resampler;
};

#endif // __cplusplus

#endif // defined AUDIO_TYPES_H

