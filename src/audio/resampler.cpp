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
#include "ug_runtime_error.hpp"
#include "utils/macros.h"
#include "utils/misc.h"
#include "utils/worker.h"

#ifdef HAVE_SPEEXDSP
#include <speex/speex_resampler.h>
#endif // HAVE_SPEEXDSP

#ifdef HAVE_SOXR
#include <soxr.h>
#endif // HAVE_SOXR

#define DEFAULT_SPEEX_RESAMPLE_QUALITY 10 // in range [0,10] - 10 best
#define MOD_NAME "[audio_resampler] "

using namespace std;

class audio_frame2_resampler::impl {
        public:
                virtual std::tuple<bool, audio_frame2> resample(audio_frame2 &a, std::vector<audio_frame2::channel> &out, int new_sample_rate_num, int new_sample_rate_den) = 0;
                /// @returns 0-terminated C array of suppored BPS in _ascending_ (!) order
                virtual const int *get_supported_bps() = 0;
                virtual ~impl() {}
};

tuple<bool, audio_frame2> audio_frame2_resampler::resample(audio_frame2 &a, vector<audio_frame2::channel> &out, int new_sample_rate_num, int new_sample_rate_den)
{
        if (!m_impl) {
                 LOG(LOG_LEVEL_ERROR) << "Audio frame resampler: cannot resample, Soxr/SpeexDSP was not compiled in!\n";
                 return { false, audio_frame2{} };
        }
        return m_impl->resample(a, out, new_sample_rate_num, new_sample_rate_den);
}

struct resample_prop {
        unsigned rate_from{0};
        unsigned rate_to_num{0};
        unsigned rate_to_den{1};
        unsigned ch_count{0};
        unsigned bps{0};
};

#ifdef HAVE_SOXR
class soxr_resampler : public audio_frame2_resampler::impl {
public:
        tuple<bool, audio_frame2> resample(audio_frame2 &a, vector<audio_frame2::channel> &new_channels, int new_sample_rate_num, int new_sample_rate_den) override;
        const int *get_supported_bps() override {
                static const int ret[] = { 2, 4, 0 };
                return ret;
        }
        ~soxr_resampler() {
                if (resampler) {
                        soxr_delete(resampler);
                }
        }

private:
        bool check_reconfigure(uint32_t original_sample_rate, uint32_t new_sample_rate_num, uint32_t new_sample_rate_den, size_t nb_channels, unsigned bps);

        soxr_t resampler{nullptr};
        struct resample_prop prop;
};

/**
 * @brief This function will create (and destroy) a new resampler if needed.
 * 
 * @param original_sample_rate The original sample rate in Hz
 * @param new_sample_rate_num  The numerator of the new sample rate
 * @param new_sample_rate_den  The denominator of the new sample rate
 * @param nb_channels          The number of channels that will be resampled
 * @param bps                  The bit rate (in bytes) of the incoming audio
 * 
 * @return true  Successfully created the resampler
 * @return false Initialisation of the resampler failed
 */
bool soxr_resampler::check_reconfigure(uint32_t original_sample_rate, uint32_t new_sample_rate_num, uint32_t new_sample_rate_den, size_t nb_channels, unsigned bps) {
        if (resampler != nullptr && nb_channels == prop.ch_count && bps == prop.bps) {
                if (original_sample_rate != prop.rate_from
                                || new_sample_rate_num != prop.rate_to_num
                                || new_sample_rate_den != prop.rate_to_den) {
                        // Update the resampler numerator and denomintors
                        prop.rate_from = original_sample_rate;
                        prop.rate_to_num = new_sample_rate_num;
                        prop.rate_to_den = new_sample_rate_den;
                        soxr_set_io_ratio(resampler, ((double)prop.rate_from / ((double)new_sample_rate_num / (double)new_sample_rate_den)), 0);
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
        this->resampler = soxr_create(2, 1, nb_channels, &error, &io_spec, &q_spec, &runtime_spec);

        if (error) {
                LOG(LOG_LEVEL_ERROR) << "[audio_frame2_resampler] Cannot initialize resampler: " << soxr_strerror(error) << "\n";
                return false;
        }
        // Immediately change the resample rate to be the correct value for the audio frame
        soxr_set_io_ratio((soxr_t)this->resampler, ((double)original_sample_rate / ((double)new_sample_rate_num / (double)new_sample_rate_den)), 0);

        // Setup resampler values
        this->prop.rate_from = original_sample_rate;
        this->prop.rate_to_num = new_sample_rate_num;
        this->prop.rate_to_den = new_sample_rate_den;
        this->prop.ch_count = nb_channels;
        this->prop.bps = bps;
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "Soxr resampler (re)made at " << new_sample_rate_num / new_sample_rate_den << "\n";
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
        LOG(LOG_LEVEL_DEBUG) << "[audio_frame2_resampler] resampler_duration " << resamplerDuration << "\n";

        // Remainders aren't as relevant when using SOXR
        audio_frame2 remainder = {};
        return {true, std::move(remainder)};
}
#endif

#ifdef HAVE_SPEEXDSP
class speex_resampler : public audio_frame2_resampler::impl {
public:
        tuple<bool, audio_frame2> resample(audio_frame2 &a, vector<audio_frame2::channel> &new_channels, int new_sample_rate_num, int new_sample_rate_den) override;

        speex_resampler(int q) : quality(q) {}

        const int *get_supported_bps() override {
                static const int ret[] = { 2, 4, 0 };
                return ret;
        }

        ~speex_resampler() {
                if (state) {
                        speex_resampler_destroy(state);
                }
        }
private:
        bool check_reconfigure(unsigned original_sample_rate, unsigned new_sample_rate_num, unsigned new_sample_rate_den, unsigned channel_size, unsigned bps);

        int quality;
        SpeexResamplerState *state{nullptr};
        struct resample_prop prop;
};

bool speex_resampler::check_reconfigure(unsigned original_sample_rate, unsigned new_sample_rate_num, unsigned new_sample_rate_den, unsigned nb_channels, unsigned bps) {
        if (state != nullptr && original_sample_rate == prop.rate_from
                                && new_sample_rate_num == prop.rate_to_num
                                && new_sample_rate_den == prop.rate_to_den
                                && nb_channels == prop.ch_count
                                && bps == prop.bps) {
                return true;
        }
        if (bps != 2 && bps != 4) {
                throw logic_error("Only 16 or 32 bits per sample are supported for resampling!");
        }

        if (state) {
                 speex_resampler_destroy(state);
        }
        state = nullptr;
        int err = 0;
        state = speex_resampler_init_frac(nb_channels, original_sample_rate * new_sample_rate_den,
                                new_sample_rate_num, original_sample_rate, new_sample_rate_num, quality, &err);
        if (err) {
                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Cannot initialize SpeexDSP resampler: " << speex_resampler_strerror(err) << "\n";
                return false;
        }
        prop.rate_from = original_sample_rate;
        prop.rate_to_num = new_sample_rate_num;
        prop.rate_to_den = new_sample_rate_den;
        prop.ch_count = nb_channels;
        prop.bps = bps;
        LOG(LOG_LEVEL_VERBOSE) << MOD_NAME "SpeexDSP resampler (re)made at " << new_sample_rate_num / new_sample_rate_den << "\n";
        return true;
}

struct speex_process_channel_data {
        SpeexResamplerState *state;
        int channel_idx;
        void *in;
        void *out;
        uint32_t in_frames;
        uint32_t in_frames_orig;
        uint32_t write_frames;
};

static void *speex_process_channel_short(void *arg) {
        auto *d = (speex_process_channel_data *) arg;
        speex_resampler_process_int(d->state,
                        d->channel_idx,
                        (spx_int16_t *) d->in, &d->in_frames,
                        (spx_int16_t *) d->out, &d->write_frames);
        return NULL;
}

static void *speex_process_channel_int(void *arg) {
        auto *d = (speex_process_channel_data *) arg;
        int2float((char *) d->in, (char *) d->in, sizeof(float) * d->in_frames);
        speex_resampler_process_float(d->state,
                        d->channel_idx,
                        (float *) d->in, &d->in_frames,
                        (float *) d->out, &d->write_frames);
        float2int((char *) d->out, (char *) d->out, sizeof(float) * d->write_frames);
        return NULL;
}

tuple<bool, audio_frame2> speex_resampler::resample(audio_frame2 &a, vector<audio_frame2::channel> &new_channels, int new_sample_rate_num, int new_sample_rate_den) {
        bool ret = check_reconfigure(a.get_sample_rate(), new_sample_rate_num, new_sample_rate_den, a.get_channel_count(), a.get_bps());
        if (!ret) {
                return {false, audio_frame2{}};
        }

        audio_frame2 remainder;
        remainder.init(new_channels.size(), AC_PCM, prop.bps, prop.rate_from);

        vector <speex_process_channel_data> speex_worker_data(new_channels.size());
        for (size_t i = 0; i < new_channels.size(); i++) {
                speex_worker_data.at(i).state = state;
                speex_worker_data.at(i).channel_idx = i;
                speex_worker_data.at(i).in = (void *) a.get_data(i);
                speex_worker_data.at(i).out = (void *) new_channels[i].data.get();
                speex_worker_data.at(i).in_frames_orig =
                        speex_worker_data.at(i).in_frames = a.get_data_len(i) / a.get_bps();
                speex_worker_data.at(i).write_frames = new_channels[i].len / a.get_bps();
        }
        if (a.get_bps() == 2) {
                task_run_parallel(speex_process_channel_short, new_channels.size(), speex_worker_data.data(), sizeof speex_worker_data[0], NULL);
        } else {
                task_run_parallel(speex_process_channel_int, new_channels.size(), speex_worker_data.data(), sizeof speex_worker_data[0], NULL);
        }
        for (size_t i = 0; i < new_channels.size(); i++) {
                if (speex_worker_data.at(i).in_frames != speex_worker_data.at(i).in_frames_orig) {
                        remainder.append(i, a.get_data(i) + speex_worker_data.at(i).in_frames * a.get_bps(),
                                        speex_worker_data.at(i).in_frames_orig - speex_worker_data.at(i).in_frames);
                }
                new_channels[i].len = speex_worker_data.at(i).write_frames * a.get_bps();
        }

        if (remainder.get_data_len() == 0) {
                remainder = {};
        }
        return {true, std::move(remainder)};
}
#endif // defined HAVE_SPEEXDSP

ADD_TO_PARAM("resampler", "* resampler=[speex|soxr][[:]quality=[0-10]]\n"
                "  Select resampler; set quality for Speex in range 0 (worst) and 10 (best), default " TOSTRING(DEFAULT_SPEEX_RESAMPLE_QUALITY) "\n");

audio_frame2_resampler::audio_frame2_resampler()
{
        enum { RESAMPLER_DEFAULT, RESAMPLER_SPEEX, RESAMPLER_SOXR } resampler_type = RESAMPLER_DEFAULT;
        const char *cfg_c = get_commandline_param("resampler");
        std::string_view sv = cfg_c ? cfg_c : "";
        int quality = DEFAULT_SPEEX_RESAMPLE_QUALITY;
        while (!sv.empty()) {
                const auto &tok = tokenize(sv, ':');
                if (tok == "speex") {
                        resampler_type = RESAMPLER_SPEEX;
                } else if (tok == "soxr") {
                        resampler_type = RESAMPLER_SOXR;
                } else if (tok.compare(0, "quality="sv.length(), "quality=") == 0) {
                        quality = stoi(string(tok.substr("quality="sv.length())));
                        if (quality < 0 || quality > 10) {
                                throw ug_runtime_error("Quality " + to_string(quality) + " out of range 0-10"s);
                        }
                } else {
                        throw ug_runtime_error("Unknown resampler option: "s + string(tok));
                }
        }
        switch (resampler_type) {
                case RESAMPLER_DEFAULT:
#ifdef HAVE_SPEEXDSP
                        m_impl = unique_ptr<audio_frame2_resampler::impl>(new speex_resampler(quality));
#elif defined HAVE_SOXR
                        m_impl = unique_ptr<audio_frame2_resampler::impl>(new soxr_resampler());
#endif
                        break;
                case RESAMPLER_SPEEX:
#ifdef HAVE_SPEEXDSP
                        m_impl = unique_ptr<audio_frame2_resampler::impl>(new speex_resampler(quality));
#else
                        throw ug_runtime_error("SpeexDSP not compiled in!");
#endif
                        break;
                case RESAMPLER_SOXR:
#if defined HAVE_SOXR
                        m_impl = unique_ptr<audio_frame2_resampler::impl>(new soxr_resampler());
                        break;
#else
                        throw ug_runtime_error("Soxr not compiled in!");
#endif
        }
}

/**
 * @returns the orig (if supported) or nearest higher of resampler supported BPS (highest if orig is higher)
 */
int audio_frame2_resampler::align_bps(int orig) {
        if (!m_impl) {
                 LOG(LOG_LEVEL_ERROR) << "Audio frame resampler: cannot resample, Soxr/SpeexDSP was not compiled in!\n";
                 return 0;
        }
        const int *sup = m_impl->get_supported_bps();
        int last = 0;
        while (*sup != 0) {
                if (orig <= *sup) {
                        return *sup;
                }
                last = *sup++;
        }
        return last;
}

audio_frame2_resampler::~audio_frame2_resampler() = default;
audio_frame2_resampler::audio_frame2_resampler(audio_frame2_resampler&&) = default;
audio_frame2_resampler& audio_frame2_resampler::operator=(audio_frame2_resampler&&) = default;

