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

#include <cmath>
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

ADD_TO_PARAM("resampler-quality", "* resampler-quality=[0-10]\n"
                "  Sets audio resampler quality in range 0 (worst) and 10 (best), default " TOSTRING(DEFAULT_RESAMPLE_QUALITY) "\n");

bool audio_frame2_resampler::create_resampler(uint32_t original_sample_rate, uint32_t new_sample_rate_num, uint32_t new_sample_rate_den, size_t channel_size, int bps) {
#ifdef HAVE_SPEEXDSP
        LOG(LOG_LEVEL_VERBOSE) << "Destroying Resampler\n";
        if (this->resampler) {
                speex_resampler_destroy((SpeexResamplerState *) this->resampler);
                this->destroy_resampler = false;
        }
        this->resampler = nullptr;

        int quality = DEFAULT_RESAMPLE_QUALITY;
        if (commandline_params.find("resampler-quality") != commandline_params.end()) {
                quality = stoi(commandline_params.at("resampler-quality"));
                assert(quality >= 0 && quality <= 10);
        }
        int err = 0;
        this->resampler = speex_resampler_init_frac(channel_size, original_sample_rate * new_sample_rate_den, new_sample_rate_num,
                                                    original_sample_rate, new_sample_rate_num / new_sample_rate_den, quality, &err);
        if (err) {
                LOG(LOG_LEVEL_ERROR) << "[audio_frame2_resampler] Cannot initialize resampler: " << speex_resampler_strerror(err) << "\n";
                return false;
        }
        // Ignore resampler delay. The speex resampler silently adds a delay to the resampler by adding silence at the length
        // of the input latency and stored a buffered amount for itself. This is extracted outside of this function on the final
        // call before a resampler is marked for destruction.
        speex_resampler_skip_zeros((SpeexResamplerState *) this->resampler);
        this->resample_from = original_sample_rate;

        // Setup resampler values
        this->resample_to_num = new_sample_rate_num;
        this->resample_to_den = new_sample_rate_den;
        this->resample_ch_count = channel_size;
        // Capture the input and output latency. Generally, there is not a difference between the two.
        // The input latency is used to calculate leftover audio in the resampler that is collected on the
        // audio frame before the resampler is destroyed.
        this->resample_input_latency = speex_resampler_get_input_latency((SpeexResamplerState *) this->resampler);
        this->resample_output_latency = speex_resampler_get_output_latency((SpeexResamplerState *) this->resampler);
        this->resample_initial_bps = bps;
        LOG(LOG_LEVEL_ERROR) << "[audio_frame2] Resampler (re)made at " << new_sample_rate_num / new_sample_rate_den << "\n";
        return true;
#endif
        return false;
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
 * @brief This will convert 32-bit integer audio into a 32-bit floating point audio format.
 *        Doing so will allow 32-bit integer audio data to be resampled using the speex floating point resampler.
 *        Converting between 32-bit floating point audio and 32-bit integer audio is likely to cause
 *        some data loss due to rounding issues on conversion (and the precision of floating point data types when converting to)
 * 
 */
void audio_frame2::convert_int32_to_float() {
        for(size_t i = 0; i < this->channels.size(); i++) {
                auto channel_data = this->get_data(i);
                auto channel_data_length =  this->get_data_len(i);
                for(size_t j = 0; j < channel_data_length / this->bps; j++) {
                        int32_t *p_curr_value = (int32_t *)(channel_data + (this->bps * j));
                        float *p_curr_value_float = (float *)p_curr_value;
                        *p_curr_value_float = ((float)(*p_curr_value) / (float)std::numeric_limits<int32_t>::max());
                }
        }
}

/**
 * @brief This will convert 32-bit floating point audio data into a 32-bit integer audio format.
 *        Doing so will allow 32-bit integer audio data to be resampled using the speex floating point resampler.
 *        Converting between 32-bit floating point audio and 32-bit integer audio is likely to cause
 *        some data loss due to the precision of floating point data types (and rounding issues on conversion back).
 * 
 */
void audio_frame2::convert_float_to_int32() {
        for(size_t i = 0; i < this->channels.size(); i++) {
                auto channel_data = this->get_data(i);
                auto channel_data_length =  this->get_data_len(i);
                for(size_t j = 0; j < channel_data_length / this->bps; j++) {
                        float *p_curr_value = (float *)(channel_data + (this->bps * j));
                        int32_t *p_curr_value_int = (int32_t *)p_curr_value;

                        if((*p_curr_value) > 1) {
                                *p_curr_value_int = std::numeric_limits<int32_t>::max();
                        }
                        else if((*p_curr_value) < -1) {
                                *p_curr_value_int = std::numeric_limits<int32_t>::min();
                        }
                        else {
                                *p_curr_value_int = (int32_t)roundf((*p_curr_value) * std::numeric_limits<int32_t>::max());
                        }
                }
        }
}

/**
 * @brief A static function for resampling a single channel using the speex integer resampler.
 *        This is used to thread the resampling for all channels simultaneously.
 * 
 * @param resampler_state A pointer to resample state object which contains the Speex resampler state.
 * @param channel_index The channel index which is resampled.
 * @param in  A pointer to the 16bit integer data for the channel that is being resampled.
 * @param in_len The length in samples of the inputted data.
 * @param new_channel A pointer to the channel that is going to be written to.
 * @param remainder A pointer to an audio frame to capture lost audio if the resampler fails to resample all of the given data.
 */
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
        new_channel->len = out_len * sizeof(int16_t);
#else
        LOG(LOG_LEVEL_ERROR) << "Audio frame resampler: cannot resample, SpeexDSP was not compiled in!\n";
#endif
}

/**
 * @brief A static function for resampling a single channel using the speex floating point resampler.
 *        This is used to thread the resampling for all channels simultaneously.
 * 
 * @param resampler_state A pointer to resample state object which contains the Speex resampler state.
 * @param channel_index The channel index which is resampled.
 * @param in  A pointer to the floating point data for the channel that is being resampled.
 * @param in_len The length in samples of the inputted data.
 * @param new_channel A pointer to the channel that is going to be written to.
 * @param remainder A pointer to an audio frame to capture lost audio if the resampler fails to resample all of the given data.
 */
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
        new_channel->len = out_len * sizeof(float);
#else
        LOG(LOG_LEVEL_ERROR) << "Audio frame resampler: cannot resample, SpeexDSP was not compiled in!\n";
#endif
}

tuple<bool, bool, audio_frame2> audio_frame2::resample_fake([[maybe_unused]] audio_frame2_resampler & resampler_state, int new_sample_rate_num, int new_sample_rate_den)
{
        if (new_sample_rate_num / new_sample_rate_den == sample_rate && new_sample_rate_num % new_sample_rate_den == 0) {
                return {true, false, audio_frame2()};
        }

        // If there is resampling occuring then time how long the function takes.
        std::chrono::high_resolution_clock::time_point resample_begin = std::chrono::high_resolution_clock::now();

        // Track whether or not the resampler was reinitialised so that there is not an attempt to pull the latency buffer
        // from the resampler
        bool reinitialised_resampler = false;
#ifdef HAVE_SPEEXDSP
        // Speex has support for both 16bit audio and floating point 32bit audio
        if (this->bps != 2 && this->bps != 4) {
                LOG(LOG_LEVEL_DEBUG) << " Resample unsupported BPS " << bps << "\n";
                throw logic_error("Only 16 bits per sample are currently supported for resampling!");
        }

        if ((sample_rate != resampler_state.resample_from
                        || new_sample_rate_num != resampler_state.resample_to_num || new_sample_rate_den != resampler_state.resample_to_den
                        || channels.size() != resampler_state.resample_ch_count) || resampler_state.resample_initial_bps != this->bps
                        || resampler_state.destroy_resampler) {
                reinitialised_resampler = resampler_state.create_resampler(this->sample_rate, new_sample_rate_num, new_sample_rate_den, this->channels.size(), this->bps);
                if(!reinitialised_resampler) {
                        return {false, false, audio_frame2{}};
                }
                LOG(LOG_LEVEL_ERROR) << "[audio_frame2] Resampler (re)made at " << new_sample_rate_num / new_sample_rate_den << "\n";
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

        // Thread pool the resampling of the threads
        std::vector<std::thread> resampleChannelThreads;
        for (size_t i = 0; i < channels.size(); i++) {
                // If the bytes per sample is 2, then use the integer based speex resampler
                if(bps == 2) {
                        resampleChannelThreads.push_back(std::thread(audio_frame2::resample_channel, &resampler_state, i,  
                                                         (const uint16_t *)(const void *) get_data(i), 
                                                         (int)(get_data_len(i) / sizeof(int16_t)), &(new_channels[i]), &remainder));
                }
                // If the bytes per sample is 4, then use the floating point based speex resampler
                else if(bps == 4) {
                        resampleChannelThreads.push_back(std::thread(audio_frame2::resample_channel_float, &resampler_state, i,  
                                                         (const float *)(const void *) get_data(i), 
                                                         (int)(get_data_len(i) / sizeof(float)), &(new_channels[i]), &remainder));
                }
        }

        // Join the threads before copying the data across
        for(size_t i = 0; i < channels.size(); i++) {
                resampleChannelThreads[i].join();
        }

        if (remainder.get_data_len() == 0) {
                remainder = {};
        }

        channels = move(new_channels);

        std::chrono::high_resolution_clock::time_point resample_end = std::chrono::high_resolution_clock::now();
        auto time_diff = std::chrono::duration_cast<std::chrono::duration<double>>(resample_end - resample_begin);
        LOG(LOG_LEVEL_DEBUG) << "CALL LENGTH RESAMPLER " << setprecision(30) << time_diff.count() << "\n";

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