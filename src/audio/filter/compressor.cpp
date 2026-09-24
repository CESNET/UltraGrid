/**
 * @file   audio/filter/compressor.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2026 CESNET, zájmové sdružení právnických osob
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

#include <algorithm>
#include <cassert>
#include <cmath>

#include "debug.h"
#include "module.h"
#include "audio/audio_filter.h"
#include "audio/types.h"
#include "lib_common.h"
#include "utils/string_view_utils.hpp"

#define MOD_NAME "afilter/compressor"

namespace{

class Attack_release_envelope{
public:
        Attack_release_envelope() = default;

        void set_attack(double time_ms, unsigned sample_rate);
        void set_release(double time_ms, unsigned sample_rate);

        float update(float level);
private:
        static double time_to_coeff(double time_ms, unsigned sample_rate);

        float attack_coeff = 0.0f;
        float release_coeff = 0.0f;

        float currLevel = 0.0f;
};

float Attack_release_envelope::update(float level){
        const float coeff = level > currLevel ? attack_coeff : release_coeff;

        currLevel = coeff * currLevel + (1.f - coeff) * level;
        return currLevel;
}

double Attack_release_envelope::time_to_coeff(const double time_ms, const unsigned sample_rate){
        return std::exp(-1.f / (time_ms * 0.001 * sample_rate));
}

void Attack_release_envelope::set_attack(double time_ms, unsigned sample_rate){
        attack_coeff = time_to_coeff(time_ms, sample_rate);
}

void Attack_release_envelope::set_release(double time_ms, unsigned sample_rate){
        release_coeff = time_to_coeff(time_ms, sample_rate);
}

struct state_audio_compressor{
        explicit state_audio_compressor(struct module *mod) : mod(MODULE_CLASS_DATA, mod, this)
        {
        }

        module_raii mod;

        int bps = 0;
        int ch_count = 0;
        int sample_rate = 0;

        Attack_release_envelope envelope;
        float threshold = -8.f;
        double attack_ms = 0.5;
        double release_ms = 1000;
        float ratio = 4.f;
        float makeup_gain = 3.f;
};

void usage(){
        color_printf("Audio filter " TBOLD(
            "compressor") " hard knee compressor with make up gain\n\n");
        color_printf("Usage:\n");
        color_printf("\t" TBOLD("--audio-filter compressor[:attack=<time_ms>][:release=<time_ms>][:ratio=<ratio>][:threshold=<dB>][:makeup=<gain_dB>]\n\n"));
        color_printf(TBOLD("\tratio")      "\t\tcompression ratio\n");
        color_printf(TBOLD("\tmakeup")      "\t\tamplification of the resulting signal\n");
}

bool parse_config(state_audio_compressor *s, std::string_view cfg){
        while(!cfg.empty()){
                auto tok = tokenize(cfg, ':', '"');

                const auto key = tokenize(tok, '=');
                const auto val = tokenize(tok, '=');

                if (key == "attack"){
                        parse_num(val, s->attack_ms);
                } else if(key == "release"){
                        parse_num(val, s->release_ms);
                } else if(key == "threshold"){
                        parse_num(val, s->threshold);
                } else if(key == "ratio"){
                        parse_num(val, s->ratio);
                } else if(key == "makeup"){
                        parse_num(val, s->makeup_gain);
                } else {
                        log_msg(LOG_LEVEL_FATAL, MOD_NAME "Unknown parameter %s\n", SV_TO_CSTR(key));
                        return false;
                }
        }

        return true;
}

af_result_code compressor_init(struct module *parent, const char *cfg, void **state){
        auto s = std::make_unique<state_audio_compressor>(parent);

        if(strcmp(cfg, "help") == 0){
                usage();
                return AF_HELP_SHOWN;
        }

        parse_config(s.get(), cfg);

        *state = s.release();
        return AF_OK;
}

void compressor_done(void *state){
        auto *s = static_cast<state_audio_compressor *>(state);
        delete s;
}

af_result_code compressor_configure(void *state, int bps, int ch_count, int sample_rate){
        auto *s = static_cast<state_audio_compressor *>(state);

        if(bps != 2){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Only 16bit is supported\n");
                return AF_FAILURE;
        }

        if(ch_count != 1){
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Only mono is supported\n");
                return AF_FAILURE;
        }

        s->bps  = bps;
        s->ch_count = ch_count;
        s->sample_rate = sample_rate;

        s->envelope.set_attack(s->attack_ms, sample_rate);
        s->envelope.set_release(s->release_ms, sample_rate);

        return AF_OK;
}

af_result_code compressor_filter(void *state, const audio_frame **f){
        auto *s = static_cast<state_audio_compressor *>(state);

        auto frame = *f;
        constexpr int bps = 2;
        assert(frame->ch_count == 1);
        assert(frame->bps == bps);

        auto samples = frame->data_len / frame->bps / frame->ch_count;

        for(int i = 0; i < samples; i++){
                auto sample = *reinterpret_cast<const int16_t *>(frame->data + bps * i);
                auto sample_float = static_cast<float>(sample) / INT16_MAX;

                constexpr float epsilon = 0.00001f; //To avoid -inf in dBFS
                auto dBFS = 20 * std::log10(std::max(epsilon, std::fabs(sample_float)));

                auto env = s->envelope.update(dBFS);

                if(env > s->threshold){
                        float reduction_dB = (1 / s->ratio - 1) * (env - s->threshold);
                        float gain = std::pow(10.f, reduction_dB / 20.f);
                        sample_float *= gain;
                }

                sample_float *= std::pow(10.f, s->makeup_gain / 20.f);
                sample_float = std::clamp(sample_float, -1.0f, 1.0f);
                sample = sample_float * INT16_MAX;
                *reinterpret_cast<int16_t *>(frame->data + bps * i) = sample;
        }

        return AF_OK;
}

void compressor_get_configured(void *state, int *bps, int *ch_count, int *sample_rate)
{
        auto *s = static_cast<state_audio_compressor *>(state);

        if(bps) *bps = s->bps;
        if(ch_count) *ch_count = s->ch_count;
        if(sample_rate) *sample_rate = s->sample_rate;
}

constexpr audio_filter_info compressor_info = []{
        audio_filter_info info{};
        info.name = "compressor";
        info.init = compressor_init;
        info.done = compressor_done;
        info.configure = compressor_configure;
        info.filter = compressor_filter;
        info.get_configured_in = compressor_get_configured;
        info.get_configured_out = compressor_get_configured;
        return info;
}();

}

REGISTER_MODULE(compressor, &compressor_info, LIBRARY_CLASS_AUDIO_FILTER, AUDIO_FILTER_ABI_VERSION);