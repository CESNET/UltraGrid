/**
 * @file   rtp/rs.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2015 CESNET, z. s. p. o.
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
#endif

#include <bitset>
#include <stdlib.h>
#include "rtp/rs.h"
#include "rtp/rtp_callback.h"
#include "transmit.h"
#include "video.h"

#define DEFAULT_K 200
#define DEFAULT_N 240

#define MAX_K 255
#define MAX_N 255

extern "C" {
#include "rs/fec.h"
}

static void usage();

using namespace std;

rs::rs(unsigned int k, unsigned int n)
        : m_k(k), m_n(n)
{
        assert (k <= MAX_K);
        assert (n <= MAX_N);
        assert (m_k <= m_n);
        state = fec_new(m_k, m_n);
        assert(state != NULL);
}

rs::rs(const char *c_cfg)
{
        if (strcmp(c_cfg, "help") == 0) {
                usage();
                throw 0;
        }
        char *cfg = strdup(c_cfg);
        char *item, *save_ptr;
        item = strtok_r(cfg, ":", &save_ptr);
        if (item != NULL) {
                m_k = atoi(item);
                item = strtok_r(NULL, ":", &save_ptr);
                assert(item != NULL);
                m_n = atoi(item);
        } else {
                m_k = DEFAULT_K;
                m_n = DEFAULT_N;
        }
        free(cfg);
        if (m_k > MAX_K || m_n > MAX_N || m_k >= m_n) {
                usage();
                throw 1;
        }

        state = fec_new(m_k, m_n);
        assert(state != NULL);
}

rs::~rs()
{
        fec_free((fec_t *) state);
}

shared_ptr<video_frame> rs::encode(shared_ptr<video_frame> in)
{
        video_payload_hdr_t hdr;
        format_video_header(in.get(), 0, 0, hdr);
        size_t hdr_len = sizeof(hdr);
        size_t len = in->tiles[0].data_len;
        char *data = in->tiles[0].data;

        struct video_frame *out = vf_alloc_desc(video_desc_from_frame(in.get()));
      
        //int encode(char *hdr, int hdr_len, char *in, int len, char **out) {
        int ss = get_ss(hdr_len, len);
        int buffer_len = ss * m_n;
        char *out_data;
        out_data = out->tiles[0].data = (char *) malloc(buffer_len);
        uint32_t len32 = len + hdr_len;
        memcpy(out_data, &len32, sizeof(len32));
        memcpy(out_data + sizeof(len32), hdr, hdr_len);
        memcpy(out_data + sizeof(len32) + hdr_len, data, len);
        memset(out_data + sizeof(len32) + hdr_len + len, 0, ss * m_k - (sizeof(len32) + hdr_len + len));

#if 0
        void *src[m_k];
        for (int k = 0; k < m_k; ++k) {
                src[k] = *out + ss * k;
        }

        for (int m = 0; m < m_n - m_k; ++m) {
                fec_encode(state, src, *out + ss * (m_k + m), m, ss);
        }
#else
        void *src[m_k];
        for (unsigned int k = 0; k < m_k; ++k) {
                src[k] = out_data + ss * k;
        }
        void *dst[m_n-m_k];
        unsigned int dst_idx[m_n-m_k];
        for (unsigned int m = 0; m < m_n-m_k; ++m) {
                dst[m] = out_data + ss * (m_k + m);
                dst_idx[m] = m_k + m;
        }

        fec_encode((const fec_t *)state, (gf **) src,
                        (gf **) dst, dst_idx, m_n-m_k, ss);
#endif

        out->tiles[0].data_len = buffer_len;
        out->fec_params = fec_desc(FEC_RS, m_k, m_n - m_k, 0, 0, ss);

        return {out,
                [](video_frame *frame) {
                        free(frame->tiles[0].data);
                        vf_free(frame);
                }
        };
}

int rs::get_ss(int hdr_len, int len) {
        return ((sizeof(uint32_t) + hdr_len + len) + m_k - 1) / m_k;
}

void rs::decode(char *in, int in_len, char **out, int *len,
                std::map<int, int> const & c_m)
{
        std::map<int, int> m = c_m; // make private copy
        unsigned int ss = in_len / m_n;
        void *pkt[m_n];
        unsigned int index[m_n];
        unsigned int i = 0;
#if 0

        ///fprintf(stderr, "%d\n\n%d\n%d\n", in_len, malloc_usable_size((void *)in), sizeof(short));


        for (auto it = m.begin(); it != m.end(); ++it) {
                int start = it->first;
                int offset = it->second;

                int first_symbol_start = (start + ss - 1) / ss * ss;
                int last_symbol_end = (start + offset) / ss * ss;
                //fprintf(stderr, "%d %d %d\n", first_symbol_start, last_symbol_end, start);
                for (int j = first_symbol_start; j < last_symbol_end; j += ss) {
                        //fprintf(stderr, "%d\n", j);
                        pkt[i] = (void *) (in + j);
                        index[i] = j / ss;
                        i++;
                        if (i == m_k) break;
                }
                if (i == m_k) break;
        }

        if (i != m_k) {
                *len = 0;
                return;
        }

        assert (i == m_k);

        int ret = fec_decode(state, pkt, index, ss);
        if (ret != 0) {
                *len = 0;
                return;
        }
        uint32_t out_sz;
        memcpy(&out_sz,  pkt[0], sizeof(out_sz));
        fprintf(stderr, "%d %d\n\n", out_sz, index[0]);
        *len = out_sz;
        *out = (char *) in + 4;
#else
        for (auto it = m.begin(); it != m.end(); ++it) {
                int start = it->first;
                int size = it->second;

                while (m.find(start + size) != m.end()) {
                        auto item = m.find(start + size);
                        it->second += m[start + size];
                        size = it->second;
                        m.erase(item);
                }
        }

        //const unsigned int bitset_size = m_k;

        std::bitset<MAX_K> empty_slots;
        std::bitset<MAX_K> repaired_slots;

        for (auto it = m.begin(); it != m.end(); ++it) {
                int start = it->first;
                int size = it->second;

                unsigned int first_symbol_start = (start + ss - 1) / ss * ss;
                unsigned int last_symbol_end = (start + size) / ss * ss;
                for (unsigned int j = first_symbol_start; j < last_symbol_end; j += ss) {
                        if (j/ss < m_k) {
                                pkt[j/ss] = in + j;
                                index[j/ss] = j/ss;
                                empty_slots.set(j/ss);
                                //fprintf(stderr, "%d\n", j/ss);
                        } else {
                                for (unsigned int k = 0; k < m_k; ++k) {
                                        if (!empty_slots.test(k)) {
                                                pkt[k] = in + j;
                                                index[k] = j/ss;
                                                //fprintf(stderr, "%d\n", j/ss);
                                                empty_slots.set(k);
                                                repaired_slots.set(k);
                                                break;
                                        }
                                        //fprintf(stderr, "what???\n", j/ss);
                                }
                        }
                        i++;
                        //fprintf(stderr, " %d\n", i);
                        if (i == m_k) break;
                }
                if (i == m_k) break;
        }

        //fprintf(stderr, "       %d\n", i);

        if (i != m_k) {
                *len = 0;
                return;
        }

        char **output = (char **) malloc(m_k * sizeof(char *));
        for (unsigned int i = 0; i < m_k; ++i) {
                output[i] = (char *) malloc(ss);
        }

        fec_decode((const fec_t *) state, (const gf *const *) pkt,
                        (gf *const *) output, index, ss);

        i = 0;
        for (unsigned int j = 0; j < m_k; ++j) {
                if (repaired_slots.test(j)) {
                        memcpy((void *) (in + j * ss), output[i], ss);
                        i++;
                }
        }

        for (unsigned int i = 0; i < m_k; ++i) {
                free(output[i]);
        }
        free(output);

        uint32_t out_sz;
        memcpy(&out_sz, in, sizeof(out_sz));
        //fprintf(stderr, "       %d\n", out_sz);
        *len = out_sz;
        *out = (char *) in + sizeof(uint32_t);
#endif 
}

static void usage() {
        printf("RS usage:\n"
                        "\t-f rs[:<k>:<n>]\n"
                        "\n"
                        "\t\t<k> - block length (default %d, max %d)\n"
                        "\t\t<n> - length of block + parity (default %d, max %d)\n\t\t\tmust be > <k>\n"
                        "\n",
                        DEFAULT_K, MAX_K, DEFAULT_N, MAX_N);
}

