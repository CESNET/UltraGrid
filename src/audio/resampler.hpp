/**
 * @file   audio/resampler.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
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

#ifndef AUDIO_RESAMPLER_HPP_60C123AE_99B1_4726_AA2C_EFDB2C723952
#define AUDIO_RESAMPLER_HPP_60C123AE_99B1_4726_AA2C_EFDB2C723952

#include "audio/types.h"

#include <tuple>
#include <memory>
#include <vector>

class audio_frame2_resampler {
public:
        audio_frame2_resampler();
        class interface {
        public:
                virtual std::tuple<bool, audio_frame2> resample(audio_frame2 &a, std::vector<audio_frame2::channel> &out, int new_sample_rate_num, int new_sample_rate_den) = 0;
                virtual ~interface() {}
        };
        std::tuple<bool, audio_frame2> resample(audio_frame2 &a, std::vector<audio_frame2::channel> &out, int new_sample_rate_num, int new_sample_rate_den);
private:
        std::unique_ptr<interface> impl;
};

#endif // defined AUDIO_RESAMPLER_HPP_60C123AE_99B1_4726_AA2C_EFDB2C723952

