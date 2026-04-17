/**
 * @file   rtp/ldgm.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2026 CESNET, zájmové sdružení právnických osob
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

#ifndef __LDGM_H__
#define __LDGM_H__

#define LDGM_MAXIMAL_SIZE_RATIO 1

#include <map>
#include <memory>

#include "fec.h"

#define DEFAULT_LDGM_SEED 1

#define LDGM_GPU_API_VERSION 1

class LDGM_session;
struct video_frame;

struct ldgm : public fec{
        ldgm(unsigned int k, unsigned int m, unsigned int c, unsigned int seed);
        ldgm(int packet_size, int frame_size, double max_expected_loss);
        ldgm(const char *cfg);
        void set_params(unsigned int k, unsigned int m, unsigned int c, unsigned int seed);

        struct video_frame *
        encode_video_frame(const struct video_frame *video_frame) override;

        bool decode(char *in, int in_len, char **out, int *len,
                const std::map<int, int> &packets) override;

private:
        void init(unsigned int k, unsigned int m, unsigned int c, unsigned int seed = DEFAULT_LDGM_SEED);

        std::shared_ptr<LDGM_session> m_coding_session;
        unsigned int m_k, m_m, m_c;
        unsigned int m_seed;
};

#endif /* __LDGM_H__ */
