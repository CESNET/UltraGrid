/**
 * @file   audio/types.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


#include "audio/resampler.hpp"
#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "utils/macros.h"

#include <chrono>
#include <sstream>
#include <stdexcept>

using std::copy;
using std::logic_error;
using std::ostringstream;
using std::string;
using std::tuple;
using std::unique_ptr;
using std::vector;

bool audio_desc::operator!() const
{
        return codec == AC_NONE;
}

bool audio_desc::operator==(audio_desc const & other) const
{
        return bps == other.bps &&
                sample_rate == other.sample_rate &&
                ch_count == other.ch_count &&
                codec == other.codec;
}

audio_desc::operator string() const
{
        ostringstream oss;
        oss << *this;
        return oss.str();
}

/**
 * @brief creates audio_frame2 from POD audio_frame
 */
audio_frame2::audio_frame2(const struct audio_frame *old) :
                channels(old ? old->ch_count : 0)
{
        if (old == nullptr) {
                return;
        }
        desc.bps = old->bps;
        desc.sample_rate = old->sample_rate;
        desc.codec = AC_PCM;
        for (int i = 0; i < old->ch_count; i++) {
                resize(i, old->data_len / old->ch_count);
                char *data = channels[i].data.get();
                demux_channel(data, old->data, old->bps, old->data_len,
                              old->ch_count, i);
        }
        if ((old->flags & TIMESTAMP_VALID) != 0) {
                timestamp = old->timestamp;
        }
}

audio_frame2::operator bool() const
{
        return desc.codec != AC_NONE;
}

/**
 * @brief Initializes audio_frame2 for use. If already initialized, data are dropped.
 */
void audio_frame2::init(int nr_channels, audio_codec_t c, int b, int sr)
{
        channels.clear();
        channels.resize(nr_channels);
        desc.bps = b;
        desc.codec = c;
        desc.sample_rate = sr;
        duration = 0.0;
}

void audio_frame2::append(audio_frame2 const &src)
{
        if (desc.bps != src.desc.bps || desc.sample_rate != src.desc.sample_rate ||
            channels.size() != src.channels.size()) {
                throw std::logic_error(
                    "Trying to append frame with different parameters!");
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
        return desc.bps;
}

audio_codec_t audio_frame2::get_codec() const
{
        return desc.codec;
}

char *audio_frame2::get_data(int channel)
{
        return channels[channel].data.get();
}

audio_desc audio_frame2::get_desc() const
{
        struct audio_desc ret = desc;
        ret.ch_count = channels.size();
        return ret;
}

const char *audio_frame2::get_data(int channel) const
{
        return channels[channel].data.get();
}

size_t audio_frame2::get_data_len(int channel) const
{
        return channels[channel].len;
}

/**
 * Returns length of all channels in bytes
 */
size_t audio_frame2::get_data_len() const
{
        size_t len = 0;
        for (int i = 0; i < get_channel_count(); ++i) {
                len += get_data_len(i);
        }

        return len;
}

double audio_frame2::get_duration() const
{
        if (desc.codec == AC_PCM) {
                int samples = get_sample_count();
                return (double) samples / get_sample_rate();
        }
        return duration;
}

fec_desc const &audio_frame2::get_fec_params(int channel) const
{
        return channels[channel].fec_params;
}

int audio_frame2::get_channel_count() const
{
        return channels.size();
}

int audio_frame2::get_sample_count() const
{
        // for PCM, we can deduce samples count from length of the data
        if (desc.codec == AC_PCM) {
                return channels[0].len / get_bps();
        } else {
                throw logic_error("Unknown sample count for compressed audio!");
        }
}

int audio_frame2::get_sample_rate() const
{
        return desc.sample_rate;
}

bool audio_frame2::has_same_prop_as(audio_frame2 const &frame) const
{
        return desc.bps == frame.desc.bps &&
                desc.sample_rate == frame.desc.sample_rate &&
                desc.codec == frame.desc.codec &&
                channels.size() == frame.channels.size();
}

void audio_frame2::set_duration(double new_duration)
{
        duration = new_duration;
}

void audio_frame2::set_fec_params(int channel, fec_desc const &fec_params)
{
        channels[channel].fec_params = fec_params;
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
        if (new_bps == desc.bps) {
                return;
        }

        vector<channel> new_channels(channels.size());

        for (size_t i = 0; i < channels.size(); i++) {
                size_t new_size = channels[i].len / desc.bps * new_bps;
                new_channels[i] = {unique_ptr<char []>(new char[new_size]), new_size, new_size, {}};
        }

        for (size_t i = 0; i < channels.size(); i++) {
                ::change_bps(new_channels[i].data.get(), new_bps, get_data(i), get_bps(),
                                get_data_len(i));
        }

        desc.bps = new_bps;
        channels = std::move(new_channels);
}

void audio_frame2::set_timestamp(int64_t ts)
{
        assert(ts >= -1);
        timestamp = ts;
}
int64_t audio_frame2::get_timestamp() const { return timestamp; }

tuple<bool, audio_frame2> audio_frame2::resample_fake(audio_frame2_resampler & resampler_state, int new_sample_rate_num, int new_sample_rate_den)
{
        vector<channel> new_channels(channels.size());
        for (size_t i = 0; i < channels.size(); i++) {
                // allocate new storage + 10 ms headroom
                size_t new_size = (long long) channels[i].len * new_sample_rate_num / desc.sample_rate / new_sample_rate_den
                        + new_sample_rate_num * desc.bps / 100 / new_sample_rate_den;
                new_channels[i] = {unique_ptr<char []>(new char[new_size]), new_size, new_size, {}};
        }

        auto [ret, remainder] = resampler_state.resample(*this, new_channels, new_sample_rate_num, new_sample_rate_den);
        if (!ret) {
                return {false, audio_frame2{}};
        }

        channels = std::move(new_channels);
        return {ret, std::move(remainder)};
}

bool audio_frame2::resample(audio_frame2_resampler & resampler_state, int new_sample_rate)
{
        auto [ret, remainder] = resample_fake(resampler_state, new_sample_rate, 1);
        if (!ret) {
                return false;
        }
        if (remainder.get_data_len() > 0) {
                LOG(LOG_LEVEL_WARNING) << "Audio frame resampler: not all samples resampled!\n";
        }
        desc.sample_rate = new_sample_rate;

        return true;
}
