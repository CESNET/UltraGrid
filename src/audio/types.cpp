/**
 * @file   audio/types.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


#include "audio/audio.h"
#include "audio/utils.h"
#include "debug.h"
#include <speex/speex_resampler.h>

#include <stdexcept>

using namespace std;

bool audio_desc::operator!() const
{
        return codec == AC_NONE;
}

audio_desc::operator string() const
{
        ostringstream oss;
        oss << *this;
        return oss.str();
}


audio_frame2_resampler::audio_frame2_resampler() : resampler(nullptr), resample_from(0),
        resample_ch_count(0), resample_to(0)
{
}

audio_frame2_resampler::~audio_frame2_resampler() {
        if (resampler) {
                speex_resampler_destroy((SpeexResamplerState *) resampler);
        }
}

/**
 * @brief Creates empty audio_frame2
 */
audio_frame2::audio_frame2() :
        bps(0), sample_rate(0), codec(AC_NONE), duration(0.0)
{
}

/**
 * @brief creates audio_frame2 from POD audio_frame
 */
audio_frame2::audio_frame2(const struct audio_frame *old) :
                bps(old ? old->bps : 0), sample_rate(old ? old->sample_rate : 0),
                channels(old ? old->ch_count : 0),
                codec(old ? AC_PCM : AC_NONE), duration(0.0)
{
        if (old) {
                for (int i = 0; i < old->ch_count; i++) {
                        resize(i, old->data_len / old->ch_count);
                        char *data = channels[i].data.get();
                        demux_channel(data, old->data, old->bps, old->data_len, old->ch_count, i);
                }
        }
}

bool audio_frame2::operator!() const
{
        return codec == AC_NONE;
}

/**
 * @brief Initializes audio_frame2 for use. If already initialized, data are dropped.
 */
void audio_frame2::init(int nr_channels, audio_codec_t c, int b, int sr)
{
        channels.clear();
        channels.resize(nr_channels);
        bps = b;
        codec = c;
        sample_rate = sr;
        duration = 0.0;
}

void audio_frame2::append(audio_frame2 const &src)
{
        if (bps != src.bps || sample_rate != src.sample_rate ||
                        channels.size() != src.channels.size()) {
                throw std::logic_error("Trying to append frame with different parameters!");
        }

        for (size_t i = 0; i < channels.size(); i++) {
                append(i, src.get_data(i), src.get_data_len(i));
        }
}

void audio_frame2::append(int channel, const char *data, size_t length)
{
        // allocate twice as much as we need to avoid frequent reallocations
        // when append is called repeatedly
        reserve(channel, 2 * (channels[channel].len + length));
        copy(data, data + length, channels[channel].data.get() + channels[channel].len);
        channels[channel].len += length;
}


/**
 * @brief replaces portion of data of specified channel. If the size of the channel is not sufficient,
 * it is extended and old data are copied.
 */
void audio_frame2::replace(int channel, size_t offset, const char *data, size_t length)
{
        resize(channel, offset + length);
        copy(data, data + length, channels[channel].data.get() + offset);
}

/**
 * Reserves data for every channel with the specified length.
 */
void audio_frame2::reserve(size_t length)
{
        for (size_t channel = 0; channel < channels.size(); ++channel) {
                reserve(channel, length);
        }
}

void audio_frame2::reserve(int channel, size_t length)
{
        if (channels[channel].max_len < length) {
                unique_ptr<char []> new_data(new char[length]);
                copy(channels[channel].data.get(), channels[channel].data.get() +
                                channels[channel].len, new_data.get());

                channels[channel].max_len = length;
                channels[channel].data = std::move(new_data);
        }
}

/**
 * Changes actual size of channel.
 */
void audio_frame2::resize(int channel, size_t length)
{
        reserve(channel, length);
        channels[channel].len = length;
}

/**
 * Removes all data from audio_frame2.
 */
void audio_frame2::reset()
{
        for (size_t i = 0; i < channels.size(); i++) {
                channels[i].len = 0;
        }
        duration = 0.0;
}

int audio_frame2::get_bps() const
{
        return bps;
}

audio_codec_t audio_frame2::get_codec() const
{
        return codec;
}

const char *audio_frame2::get_data(int channel) const
{
        return channels[channel].data.get();
}

size_t audio_frame2::get_data_len(int channel) const
{
        return channels[channel].len;
}

double audio_frame2::get_duration() const
{
        if (codec == AC_PCM) {
                int samples = get_sample_count();
                return (double) samples / get_sample_rate();
        } else {
                return duration;
        }
}

int audio_frame2::get_channel_count() const
{
        return channels.size();
}

int audio_frame2::get_sample_count() const
{
        // for PCM, we can deduce samples count from length of the data
        if (codec == AC_PCM) {
                return channels[0].len / get_bps();
        } else {
                throw logic_error("Unknown sample count for compressed audio!");
        }
}

int audio_frame2::get_sample_rate() const
{
        return sample_rate;
}

bool audio_frame2::has_same_prop_as(audio_frame2 const &frame) const
{
        return bps == frame.bps &&
                sample_rate == frame.sample_rate &&
                codec == frame.codec &&
                channels.size() == frame.channels.size();
}

void audio_frame2::set_duration(double new_duration)
{
        duration = new_duration;
}

audio_frame2 audio_frame2::copy_with_bps_change(audio_frame2 const &frame, int new_bps)
{
        audio_frame2 ret;
        ret.init(frame.get_channel_count(), frame.get_codec(), new_bps, frame.get_sample_rate());

        for (size_t i = 0; i < ret.channels.size(); i++) {
                ret.channels[i].len = frame.get_data_len(i) / frame.get_bps() * new_bps;
                ret.channels[i].data = unique_ptr<char []>(new char[ret.channels[i].len]);
                ::change_bps(ret.channels[i].data.get(), new_bps, frame.get_data(i), frame.get_bps(),
                                frame.get_data_len(i));
        }

        return ret;
}

void  audio_frame2::change_bps(int new_bps)
{
        if (new_bps == bps) {
                return;
        }

        std::vector<channel> new_channels(channels.size());

        for (size_t i = 0; i < channels.size(); i++) {
                size_t new_size = channels[i].len / bps * new_bps;
                new_channels[i] = {unique_ptr<char []>(new char[new_size]), new_size, new_size};
        }

        for (size_t i = 0; i < channels.size(); i++) {
                ::change_bps(new_channels[i].data.get(), new_bps, get_data(i), get_bps(),
                                get_data_len(i));
        }

        bps = new_bps;
        channels = move(new_channels);
}

void audio_frame2::resample(audio_frame2_resampler & resampler_state, int new_sample_rate)
{
        if (new_sample_rate == sample_rate) {
                return;
        }

        /// @todo
        /// speex supports also floats so there could be possibility also to add support for more bps
        if (bps != 2) {
                throw logic_error("Only 16 bits per sample are currently for resamling supported!");
        }

        std::vector<channel> new_channels(channels.size());

        if (sample_rate != resampler_state.resample_from || new_sample_rate != resampler_state.resample_to || channels.size() != resampler_state.resample_ch_count) {
                if (resampler_state.resampler) {
                        speex_resampler_destroy((SpeexResamplerState *) resampler_state.resampler);
                }
                resampler_state.resampler = nullptr;

                int err;
                /// @todo
                /// Consider lower quality than 10 (max). This will improve both latency and
                /// performance.
                resampler_state.resampler = speex_resampler_init(channels.size(), sample_rate,
                                new_sample_rate, 10, &err);
                if(err) {
                        abort();
                }
                resampler_state.resample_from = sample_rate;
                resampler_state.resample_to = new_sample_rate;
                resampler_state.resample_ch_count = channels.size();
        }

        for (size_t i = 0; i < channels.size(); i++) {
                // allocate new storage + 10 ms headroom
                size_t new_size = channels[i].len * new_sample_rate / sample_rate + new_sample_rate * sizeof(int16_t) / 100;
                new_channels[i] = {unique_ptr<char []>(new char[new_size]), new_size, new_size};
        }

        /// @todo
        /// Consider doing this in parallel - complex resampling requires some milliseconds.
        /// Parallel resampling would reduce latency (and improve performance if there is not
        /// enough single-core power).
        for (size_t i = 0; i < channels.size(); i++) {
                uint32_t in_frames = get_data_len(i) / sizeof(int16_t);
                uint32_t in_frames_orig = in_frames;
                uint32_t write_frames = new_channels[i].len;

                speex_resampler_process_int(
                                (SpeexResamplerState *) resampler_state.resampler,
                                i,
                                (spx_int16_t *)get_data(i), &in_frames,
                                (spx_int16_t *)(void *) new_channels[i].data.get(), &write_frames);
                if (in_frames != in_frames_orig) {
                        LOG(LOG_LEVEL_WARNING) << "Audio frame resampler: not all samples resampled!\n";
                }
                new_channels[i].len = write_frames * sizeof(int16_t);
        }

        sample_rate = new_sample_rate;
        channels = move(new_channels);
}

