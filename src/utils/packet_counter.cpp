/**
 * @file   utils/packet_conter.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2025 CESNET
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

#include "utils/packet_counter.h"

#include <map>      // for map, _Rb_tree_iterator
#include <utility>  // for pair
#include <vector>   // for vector

using std::map;
using std::vector;

struct packet_counter {
        explicit packet_counter(int ns) : substream_data(ns) {}

        void register_packet(int substream_id, int bufnum, int offset, int len) {
                substream_data.at(substream_id)[bufnum][offset] = len;
        }

        int get_total_bytes() {
                int ret = 0;

                for (auto &&chan : substream_data) {
                        for (auto &&buffer : chan) {
                                for (auto &&packet : buffer.second) {
                                        ret += packet.second;
                                }
                        }
                }

                return ret;
        }

        int get_all_bytes() {
                int ret = 0;

                for (auto &&chan : substream_data) {
                        for (auto &buf : chan) {
                                if (!buf.second.empty()) {
                                        ret += (--buf.second.end())->first +
                                               (--buf.second.end())->second;
                                }
                        }
                }

                return ret;
        }

        void clear() {
                for (auto &&chan : substream_data) {
                        chan.clear();
                }
        }

        vector<map<int, map<int, int> > > substream_data;
};

struct packet_counter *packet_counter_init(int num_substreams) {
        return new packet_counter(num_substreams);
}

void packet_counter_destroy(struct packet_counter *state) {
        delete state;
}

void packet_counter_register_packet(struct packet_counter *state, unsigned int substream_id, unsigned int bufnum,
                unsigned int offset, unsigned int len)
{
        state->register_packet(substream_id, bufnum, offset, len);
}

int packet_counter_get_total_bytes(struct packet_counter *state)
{
        return state->get_total_bytes();
}

int packet_counter_get_all_bytes(struct packet_counter *state)
{
        return state->get_all_bytes();
}

int packet_counter_get_channels(struct packet_counter *state)
{
        return (int) state->substream_data.size();
}

void packet_counter_clear(struct packet_counter *state)
{
        state->clear();
}

