/**
 * @file   filter_chain.hpp
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

#ifndef FILTER_CHAIN_HPP_
#define FILTER_CHAIN_HPP_

#include <vector>
#include <string_view>
#include "audio_filter.h"
#include "module.h"

struct audio_filter;

class Filter_chain{
public:
        Filter_chain(struct module *parent);
        Filter_chain(const Filter_chain&) = delete;
        ~Filter_chain();

        Filter_chain& operator=(const Filter_chain&) = delete;

        void push_back(struct audio_filter filter);
        bool emplace_new(std::string_view cfg);
        void clear();

        af_result_code reconfigure(int bps, int ch_count, int sample_rate);

        af_result_code filter(struct audio_frame **frame);

        struct module *get_module() { return mod.get(); }
private:
        std::vector<struct audio_filter> filters;
        int bps = 0;
        int ch_count = 0;
        int sample_rate = 0;

        module_raii mod;
};


#endif //FILTER_CHAIN_HPP_

