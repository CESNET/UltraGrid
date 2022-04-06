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


#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "utils/misc.h"
#ifdef HAVE_SPEEXDSP
#include <speex/speex_resampler.h>
#endif // HAVE_SPEEXDSP

#include <sstream>
#include <stdexcept>
#include <chrono>
#include <thread>

#define DEFAULT_RESAMPLE_QUALITY 10 // in range [0,10] - 10 best

using namespace std;

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


audio_frame2_resampler::~audio_frame2_resampler() {
        if (resampler) {
#ifdef HAVE_SPEEXDSP
                speex_resampler_destroy((SpeexResamplerState *) resampler);
#endif
        }
}

/**
 * @brief Returns the numerator for the fractional sample rate in the resampler.
 * 
 * @return int The numerator for the fractional sample rate.
 */
int audio_frame2_resampler::get_resampler_numerator() {
        return this->resample_to_num;
}

/**
 * @brief Returns the denominator for the fractional sample rate in the resampler.
 * 
 * @return int The denominator of the sample applied to the resampler.
 */
int audio_frame2_resampler::get_resampler_denominator() {
        return this->resample_to_den;
}

/**
 * @brief Returns the input latency of the resampler. This is how many audio samples
 *        the resampler has stored that will need to be extracted when resampling is
 *        stopped.
 * 
 * @return int The input latency of the resampler.
 */
int audio_frame2_resampler::get_resampler_input_latency() {
        return this->resample_input_latency;
}

/**
 * @brief Returns the output latency of the resampler.
 * 
 * @return int The output latency of the resampler.
 */
int audio_frame2_resampler::get_resampler_output_latency() {
        return this->resample_output_latency;
}

/**
 * @brief Returns the sample rate that the resampler is sampling from.
 * 
 * @return int The sample rate the resampler is sampling from.
 */
int audio_frame2_resampler::get_resampler_from_sample_rate() {
        return this->resample_from;
}

/**
 * @brief Returns the channel count that the resampler has been initialised for.
 * 
 * @return size_t The channel count that the resampler was initiated with.
 */
size_t audio_frame2_resampler::get_resampler_channel_count() {
        return this->resample_ch_count;
}

/**
 * @brief Checks whether the resampler has been set.
 * 
 * @return true  The resampler has been initialised.
 * @return false The resampler has not been initialised.
 */
bool audio_frame2_resampler::resampler_is_set() {
        return this->resampler != nullptr;
}

/**
 * @brief Sets a flag to let the resampling function know that the resampler should
 *        be destroyed.
 * 
 * @param destroy A boolean indicating if the resampler should be destroyed on the next
 *                resample. This should be used after inserting useless data into the resampler
 *                to collect the buffer stored within it.
 */
void audio_frame2_resampler::resample_set_destroy_flag(bool destroy) {
        this->destroy_resampler = destroy;
}

/**
 * @brief Returns the initial BPS when the resampler is initialised so we can analyse what BPS the held buffer will
 *        be.
 * 
 * @return int The BPS of the audio frame when the resampler was initialised. 
 */
int audio_frame2_resampler::get_resampler_initial_bps() {
        return this->resample_initial_bps;
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

audio_frame2::operator bool() const
{
        return codec != AC_NONE;
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

char *audio_frame2::get_data(int channel)
{
        return channels[channel].data.get();
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
        if (codec == AC_PCM) {
                int samples = get_sample_count();
                return (double) samples / get_sample_rate();
        } else {
                return duration;
        }
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
        if (new_bps == bps) {
                return;
        }

        std::vector<channel> new_channels(channels.size());

        for (size_t i = 0; i < channels.size(); i++) {
                size_t new_size = channels[i].len / bps * new_bps;
                new_channels[i] = {unique_ptr<char []>(new char[new_size]), new_size, new_size, {}};
        }

        for (size_t i = 0; i < channels.size(); i++) {
                ::change_bps(new_channels[i].data.get(), new_bps, get_data(i), get_bps(),
                                get_data_len(i));
        }

        bps = new_bps;
        channels = move(new_channels);
}

/**
 * @brief A helper function for detecting whether or not there are two instances of a "zero" in the output
 *        in a row. This would indicate a period of silence in the audio, which during resampling, indicates
 *        that the buffer from the resampler has not been extracted properly (or is not being removed at the 
 *        beginning of the resampler delay). In production code this should function should not be required.
 * 
 * @param location Used in the log output so that this call can be placed in multiple places in the code and this argument
 *                 can be used to distinguish them.
 */
void audio_frame2::check_data(const char* location) {
        for(size_t i = 0; i < channels.size(); i++) {
                auto channelData = this->get_data(i);
                auto channelDataLength =  this->get_data_len(i);
                int16_t previousValue = 1;
                for(size_t j = 0; j < channelDataLength / sizeof(uint16_t); j++) {
                        auto currValue = *(int16_t *)(channelData + (sizeof(int16_t) * j));
                        // Check to see if the current value is zero and if the previous value was also zero. If true, then output a log line.
                        if(currValue == previousValue && currValue == 0) {
                                LOG(LOG_LEVEL_INFO) << " FOUND SET OF ZEROES IN CHANNEL " << i << " FOUND AT " << location << " " << j * sizeof(uint16_t) << " SAMPLES IN" << "\n";
                        }
                        previousValue = currValue;
                }
        }
}

void audio_frame2::resample_channel(audio_frame2_resampler* resampler_state, int channel_index, const uint16_t *in, uint32_t in_len, channel *new_channel, audio_frame2 *remainder) {
#ifdef HAVE_SPEEXDSP
        uint32_t in_len_orig = in_len;
        uint32_t out_len = new_channel->len;

        speex_resampler_process_int(
                        (SpeexResamplerState *) resampler_state->resampler,
                        channel_index,
                        (const spx_int16_t *)in, &in_len,
                        (spx_int16_t *)(void *) new_channel->data.get(), &out_len);
        if (in_len != in_len_orig) {
                remainder->append(channel_index, (char *)(in + (in_len * sizeof(int16_t))), in_len_orig - in_len);
        }
        // The speex resampler process returns the number of frames written + 1 (so ensure we subtract 1 when setting the length)
        new_channel->len = (out_len - 1) * sizeof(int16_t);
#else
        LOG(LOG_LEVEL_ERROR) << "Audio frame resampler: cannot resample, SpeexDSP was not compiled in!\n";
#endif
}

void audio_frame2::resample_channel_float(audio_frame2_resampler* resampler_state, int channel_index, const float *in, uint32_t in_len, channel *new_channel, audio_frame2 *remainder) {
#ifdef HAVE_SPEEXDSP
        uint32_t in_len_orig = in_len;
        uint32_t out_len = new_channel->len;

        speex_resampler_process_float(
                        (SpeexResamplerState *) resampler_state->resampler,
                        channel_index,
                        in, &in_len,
                        (float *)(void *) new_channel->data.get(), &out_len);
        if (in_len != in_len_orig) {
                remainder->append(channel_index, (char *)(in + (in_len * sizeof(float))), in_len_orig - in_len);
        }
        // The speex resampler process returns the number of frames written + 1 (so ensure we subtract 1 when setting the length)
        new_channel->len = (out_len - 1) * sizeof(float);
#else
        LOG(LOG_LEVEL_ERROR) << "Audio frame resampler: cannot resample, SpeexDSP was not compiled in!\n";
#endif
}

ADD_TO_PARAM("resampler-quality", "* resampler-quality=[0-10]\n"
                "  Sets audio resampler quality in range 0 (worst) and 10 (best), default " TOSTRING(DEFAULT_RESAMPLE_QUALITY) "\n");

tuple<bool, bool, audio_frame2> audio_frame2::resample_fake([[maybe_unused]] audio_frame2_resampler & resampler_state, int new_sample_rate_num, int new_sample_rate_den)
{
        if (new_sample_rate_num / new_sample_rate_den == sample_rate && new_sample_rate_num % new_sample_rate_den == 0) {
                return {true, false, audio_frame2()};
        }

        std::chrono::high_resolution_clock::time_point resample_begin = std::chrono::high_resolution_clock::now();

        bool reinitialised_resampler = false;
#ifdef HAVE_SPEEXDSP
        /// @todo
        /// speex supports also floats so there could be possibility also to add support for more bps
        LOG(LOG_LEVEL_VERBOSE) << " BPS " << bps << "\n";
        if (bps != 2 && bps != 4) {
                LOG(LOG_LEVEL_VERBOSE) << " Unsupported BPS " << bps << "\n";
                throw logic_error("Only 16 or 32 bits per sample are currently supported for resampling!");
        }

        if ((sample_rate != resampler_state.resample_from
                        || new_sample_rate_num != resampler_state.resample_to_num || new_sample_rate_den != resampler_state.resample_to_den
                        || channels.size() != resampler_state.resample_ch_count) || resampler_state.resample_initial_bps != this->bps
                        || resampler_state.destroy_resampler) {
                if (resampler_state.resampler) {
                        speex_resampler_destroy((SpeexResamplerState *) resampler_state.resampler);
                        resampler_state.destroy_resampler = false;
                }
                resampler_state.resampler = nullptr;
                
                LOG(LOG_LEVEL_VERBOSE) << " REINIT Reinitialising because of BPS change. newBPS " << this->bps << " oldBPS " << resampler_state.resample_initial_bps << "\n";

                int quality = DEFAULT_RESAMPLE_QUALITY;
                if (commandline_params.find("resampler-quality") != commandline_params.end()) {
                        quality = stoi(commandline_params.at("resampler-quality"));
                        assert(quality >= 0 && quality <= 10);
                }
                int err = 0;
                resampler_state.resampler = speex_resampler_init_frac(channels.size(), sample_rate * new_sample_rate_den, new_sample_rate_num,
                                                                      sample_rate, new_sample_rate_num / new_sample_rate_den, quality, &err);
                if (err) {
                        LOG(LOG_LEVEL_ERROR) << "[audio_frame2] Cannot initialize resampler: " << speex_resampler_strerror(err) << "\n";
                        return {false, reinitialised_resampler, audio_frame2{}};
                }
                // Ignore resampler delay. The speex resampler silently adds a delay to the resampler by adding silence at the length
                // of the input latency and stored a buffered amount for itself. This is extracted outside of this function on the final
                // call before a resampler is marked for destruction.
                speex_resampler_skip_zeros((SpeexResamplerState *) resampler_state.resampler);
                resampler_state.resample_from = sample_rate;

                // Setup resampler values
                resampler_state.resample_to_num = new_sample_rate_num;
                resampler_state.resample_to_den = new_sample_rate_den;
                resampler_state.resample_ch_count = channels.size();
                // Capture the input and output latency. Generally, there is not a difference between the two.
                // The input latency is used to calculate leftover audio in the resampler that is collected on the
                // audio frame before the resampler is destroyed.
                resampler_state.resample_input_latency = speex_resampler_get_input_latency((SpeexResamplerState *) resampler_state.resampler);
                resampler_state.resample_output_latency = speex_resampler_get_output_latency((SpeexResamplerState *) resampler_state.resampler);

                resampler_state.resample_initial_bps = this->bps;

                reinitialised_resampler = true;
        }

        // Initialise the new channels that the resampler is going to write into
        std::vector<channel> new_channels(channels.size());
        for (size_t i = 0; i < channels.size(); i++) {
                // allocate new storage + 10 ms headroom
                size_t new_size = (long long) channels[i].len * new_sample_rate_num / sample_rate / new_sample_rate_den
                        + new_sample_rate_num * this->bps / 100 / new_sample_rate_den;
                new_channels[i] = {unique_ptr<char []>(new char[new_size]), new_size, new_size, {}};
        }

        audio_frame2 remainder;
        remainder.init(get_channel_count(), get_codec(), get_bps(), get_sample_rate());

        /// @todo
        /// Consider doing this in parallel - complex resampling requires some milliseconds.
        /// Parallel resampling would reduce latency (and improve performance if there is not
        /// enough single-core power).
        std::vector<std::thread> resampleChannelThreads;
        for (size_t i = 0; i < channels.size(); i++) {
                if(bps == 2) {
                        // resampleChannelThreads.push_back(std::thread(audio_frame2::resample_channel, &resampler_state, i,  
                        //                                  (const uint16_t *)(const void *) get_data(i), 
                        //                                  (int)(get_data_len(i) / sizeof(int16_t)), &(new_channels[i]), &remainder));
                        audio_frame2::resample_channel(&resampler_state, i,  
                                                (const uint16_t *)(const void *) get_data(i), 
                                                (int)(get_data_len(i) / sizeof(int16_t)), &(new_channels[i]), &remainder);
                        LOG(LOG_LEVEL_VERBOSE) << "Calling int resampler\n";
                }
                else if(bps == 4) {
                        // resampleChannelThreads.push_back(std::thread(audio_frame2::resample_channel, &resampler_state, i,  
                        //                                  (const float *)(const void *) get_data(i), 
                        //                                  (int)(get_data_len(i) / sizeof(int16_t)), &(new_channels[i]), &remainder));
                        audio_frame2::resample_channel_float(&resampler_state, i,  
                                                             (const float *)(const void *) get_data(i), 
                                                             (int)(get_data_len(i) / sizeof(float)), &(new_channels[i]), &remainder);
                        LOG(LOG_LEVEL_VERBOSE) << "Calling float resampler\n";
                }
        }

        // for(size_t i = 0; i < channels.size(); i++) {
        //         resampleChannelThreads[i].join();
        // }

        if (remainder.get_data_len() == 0) {
                remainder = {};
        }

        channels = move(new_channels);

        std::chrono::high_resolution_clock::time_point resample_end = std::chrono::high_resolution_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::duration<double>>(resample_end - resample_begin);
        LOG(LOG_LEVEL_VERBOSE) << " call diff resampler " << setprecision(30) << time_diff.count() << "\n";

        return {true, reinitialised_resampler, std::move(remainder)};
#else
        UNUSED(resampler_state.resample_from);
        UNUSED(resampler_state.resample_to_num);
        UNUSED(resampler_state.resample_to_den);
        UNUSED(resampler_state.resample_ch_count);
        LOG(LOG_LEVEL_ERROR) << "Audio frame resampler: cannot resample, SpeexDSP was not compiled in!\n";
        return {false, reinitialised_resampler, audio_frame2{}};
#endif
}

tuple<bool, bool> audio_frame2::resample(audio_frame2_resampler & resampler_state, int new_sample_rate)
{
        auto [ret, reinitResampler, remainder] = resample_fake(resampler_state, new_sample_rate, 1);
        if (!ret) {
                return {false, reinitResampler};
        }
        if (remainder.get_data_len() > 0) {
                LOG(LOG_LEVEL_WARNING) << "Audio frame resampler: not all samples resampled!\n";
        }
        sample_rate = new_sample_rate;

        return {true, reinitResampler};
}

