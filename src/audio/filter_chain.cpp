/**
 * @file   filter_chain.cpp
 * @author Martin Piatka     <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2021 CESNET, z. s. p. o.
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

#include "audio/types.h"
#include "filter_chain.hpp"

Filter_chain::~Filter_chain(){
        clear();
}

void Filter_chain::push_back(struct audio_filter filter){
        filters.push_back(filter);
}

void Filter_chain::clear(){
        for(auto& i : filters){
                audio_filter_destroy(&i);
        }
}

af_result_code Filter_chain::filter(struct audio_frame **frame){
        auto f = *frame;
        af_result_code res = reconfigure(f->bps, f->ch_count, f->sample_rate);
        if(res != AF_OK)
                return AF_MISCONFIGURED;

        for(auto& i : filters){
                res = audio_filter(&i, frame);
                if(res != AF_OK)
                        return res;
        }

        return res;
}

af_result_code Filter_chain::reconfigure(int bps, int ch_count, int sample_rate){
        if(this->bps == bps
                        && this->ch_count == ch_count
                        && this->sample_rate == sample_rate)
        {
                return AF_OK;
        }

        this->bps = bps;
        this->ch_count = ch_count;
        this->sample_rate = sample_rate;

        af_result_code res = AF_OK;
        for(auto& i : filters){
                af_result_code r = audio_filter_configure(&i, bps, ch_count, sample_rate);
                if(res != AF_OK)
                        return res;

                audio_filter_get_configured_out(&i, &bps, &ch_count, &sample_rate);
        }

        return res;
}
