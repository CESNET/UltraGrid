/**
 * @file   audio/resampler.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Andrew Walker    <andrew.walker@sohonet.com>
 */
/*
 * Copyright (c) 2011-2022 CESNET, z. s. p. o.
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

#ifdef HAVE_SPEEXDSP
#include <speex/speex_resampler.h>
#endif // HAVE_SPEEXDSP

#ifdef HAVE_SOXR
#include <soxr.h>
#endif // HAVE_SOXR

#define DEFAULT_RESAMPLE_QUALITY 10 // in range [0,10] - 10 best

using namespace std;

tuple<bool, audio_frame2> audio_frame2_resampler::resample(audio_frame2 &a, vector<audio_frame2::channel> &out, int new_sample_rate_num, int new_sample_rate_den)
{
        if (!impl) {
                 LOG(LOG_LEVEL_ERROR) << "Audio frame resampler: cannot resample, Soxr/SpeexDSP was not compiled in!\n";
                 return { false, audio_frame2{} };
        }
        return impl->resample(a, out, new_sample_rate_num, new_sample_rate_den);
}

#ifdef HAVE_SOXR
class soxr_resampler : public audio_frame2_resampler::interface {
public:
        tuple<bool, audio_frame2> resample(audio_frame2 &a, vector<audio_frame2::channel> &new_channels, int new_sample_rate_num, int new_sample_rate_den);

private:
        bool check_reconfigure(uint32_t original_sample_rate, uint32_t new_sample_rate_num, uint32_t new_sample_rate_den, size_t channel_size, int bps);

        soxr_t resampler{nullptr};
        uint32_t resample_from{0};
        uint32_t resample_to_num{0};
        uint32_t resample_to_den{1};
        size_t resample_ch_count{0};
        int resample_bps{0};
};

/**
 * @brief This function will create (and destroy) a new resampler.
 * 
 * @param original_sample_rate The original sample rate in Hz
 * @param new_sample_rate_num  The numerator of the new sample rate
 * @param new_sample_rate_den  The denominator of the new sample rate
 * @param channel_size         The number of channels that will be resampled
 * @param bps                  The bit rate (in bytes) of the incoming audio
 * 
 * @return true  Successfully created the resampler
 * @return false Initialisation of the resampler failed
 */
bool soxr_resampler::check_reconfigure(uint32_t original_sample_rate, uint32_t new_sample_rate_num, uint32_t new_sample_rate_den, size_t channel_size, int bps) {
        if (resampler != nullptr && bps == resample_bps) {
                if (original_sample_rate != resample_from
                                || new_sample_rate_num != resample_to_num
                                || new_sample_rate_den != resample_to_den) {
                        // Update the resampler numerator and denomintors
                        resample_from = original_sample_rate;
                        resample_to_num = new_sample_rate_num;
                        resample_to_den = new_sample_rate_den;
                        soxr_set_io_ratio(resampler, ((double)resample_from / ((double)new_sample_rate_num / (double)new_sample_rate_den)), 0);
                }
                return true;
        }

        if (this->resampler) {
                soxr_delete((soxr_t)this->resampler);
        }
        this->resampler = nullptr;

        /* When creating a var-rate resampler, q_spec must be set as follows: */
        soxr_quality_spec_t q_spec = soxr_quality_spec(SOXR_HQ, SOXR_VR);
        soxr_runtime_spec_t const runtime_spec = soxr_runtime_spec(1);
        soxr_io_spec_t io_spec;
        if(bps == 2) {
                io_spec = soxr_io_spec(SOXR_INT16_S, SOXR_INT16_S);
        }
        else if (bps == 4) {
                io_spec = soxr_io_spec(SOXR_INT32_S, SOXR_INT32_S);
        }
        else {
                LOG(LOG_LEVEL_ERROR) << "[audio_frame2_resampler] Unsupported BPS of: " << bps << "\n";
                return false;
        }

        soxr_error_t error;
        /* The ratio of the given input rate and output rates must equate to the
         * maximum I/O ratio that will be used. A resample rate of 2 to 1 would be excessive,
           but provides a sensible ceiling */
        this->resampler = soxr_create(2, 1, channel_size, &error, &io_spec, &q_spec, &runtime_spec);

        if (error) {
                LOG(LOG_LEVEL_ERROR) << "[audio_frame2_resampler] Cannot initialize resampler: " << soxr_strerror(error) << "\n";
                return false;
        }
        // Immediately change the resample rate to be the correct value for the audio frame
        soxr_set_io_ratio((soxr_t)this->resampler, ((double)original_sample_rate / ((double)new_sample_rate_num / (double)new_sample_rate_den)), 0);

        // Setup resampler values
        this->resample_from = original_sample_rate;
        this->resample_to_num = new_sample_rate_num;
        this->resample_to_den = new_sample_rate_den;
        this->resample_ch_count = channel_size;
        this->resample_bps = bps;
        LOG(LOG_LEVEL_DEBUG) << "[audio_frame2] Resampler (re)made at " << new_sample_rate_num / new_sample_rate_den << "\n";
        return true;
}

tuple<bool, audio_frame2> soxr_resampler::resample(audio_frame2 &a, vector<audio_frame2::channel> &new_channels, int new_sample_rate_num, int new_sample_rate_den) {
        std::chrono::high_resolution_clock::time_point funcBegin = std::chrono::high_resolution_clock::now();

        bool ret = check_reconfigure(a.get_sample_rate(), new_sample_rate_num, new_sample_rate_den, a.get_channel_count(), a.get_bps());
        if (!ret) {
                return {false, audio_frame2{}};
        }

        // Initialise the new channels that the resampler is going to write into
        void * * const obuf_ptrs = (void * *) malloc(sizeof(void *) * a.get_channel_count());
        void * *       ibuf_ptrs = (void * *) malloc(sizeof(void *) * a.get_channel_count());

        for (size_t i = 0; i < new_channels.size(); i++) {
                // Setup the buffers
                obuf_ptrs[i] = new_channels[i].data.get();
                ibuf_ptrs[i] = a.get_data(i);
        }

        size_t inlen = a.get_data_len(0) / a.get_bps();
        size_t outlen = new_channels[0].len / a.get_bps();
        size_t odone = 0;
        soxr_error_t error;
        error = soxr_process(resampler, ibuf_ptrs, inlen, NULL, obuf_ptrs, outlen, &odone);
        if (error) {
                LOG(LOG_LEVEL_ERROR) << "[audio_frame2_resampler] resampler failed: " << soxr_strerror(error) << "\n";
                return {false, audio_frame2{}};
        }
        for (unsigned int i = 0; i < new_channels.size(); i++) {
                new_channels[i].len = odone * a.get_bps();
        }

        free(obuf_ptrs); free(ibuf_ptrs);

        std::chrono::high_resolution_clock::time_point funcEnd = std::chrono::high_resolution_clock::now();
        long long resamplerDuration = std::chrono::duration_cast<std::chrono::milliseconds>(funcEnd - funcBegin).count();
        LOG(LOG_LEVEL_VERBOSE) << "[audio_frame2_resampler] resampler_duration " << resamplerDuration << "\n";

        // Remainders aren't as relevant when using SOXR
        audio_frame2 remainder = {};
        return {true, std::move(remainder)};
}
#endif

audio_frame2_resampler::audio_frame2_resampler() {
#ifdef HAVE_SOXR
        impl = unique_ptr<audio_frame2_resampler::interface>(new soxr_resampler());
#endif
}

