/*
 * FILE:   ldgm.h
 * AUTHOR: Martin Pulec <pulec@cesnet.cz>
 *
 * Copyright (c) 1998-2000 University College London
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions 
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *      This product includes software developed by the Computer Science
 *      Department at University College London.
 * 4. Neither the name of the University nor of the Department may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 * 
 */

#ifndef __LDGM_H__
#define __LDGM_H__

#define LDGM_MAXIMAL_SIZE_RATIO 1

#include <map>
#include <memory>

#include "fec.h"

#define DEFAULT_LDGM_SEED 1

class LDGM_session;
struct video_frame;

struct ldgm : public fec{
        ldgm(unsigned int k, unsigned int m, unsigned int c, unsigned int seed);
        ldgm(int packet_size, int frame_size, double max_expected_loss);
        ldgm(const char *cfg);
        std::shared_ptr<video_frame> encode(std::shared_ptr<video_frame>);
        void decode(const char *in, int in_len, char **out, int *len,
                const std::map<int, int> &);
        void freeBuffer(char *buffer);

private:
        void init(unsigned int k, unsigned int m, unsigned int c, unsigned int seed = DEFAULT_LDGM_SEED);

        std::unique_ptr<LDGM_session> m_coding_session;
        unsigned int m_k, m_m, m_c;
        unsigned int m_seed;
};

#endif /* __LDGM_H__ */
