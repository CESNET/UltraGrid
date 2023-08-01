/**
 * @file   utils/packet_conter.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012 CESNET, z. s. p. o.
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
#endif // defined HAVE_CONFIG_H

#include "utils/packet_counter.h"
#include <cassert>
#include <map>
#include <vector>

using std::map;
using std::vector;

struct packet_counter {
        explicit packet_counter(int ns) : num_substreams(ns), packets(ns) {}

        void register_packet(int substream_id, int bufnum, int offset, int len) {
                assert(substream_id < num_substreams); 

                packets[substream_id][bufnum][offset] = len;
                current_bufnum = bufnum;
        }

        int get_total_bytes() {
                int ret = 0;

                for(int i = 0; i < num_substreams; ++i) {
                        for(map<int, map<int, int> >::const_iterator it = packets[i].begin();
                                       it != packets[i].end();
                                       ++it) {
                                for(map<int, int>::const_iterator it2 = it->second.begin();
                                               it2 != it->second.end();
                                               ++it2) {
                                        ret += it2->second;
                                }
                        }
                }

                return ret;
        }

        int get_all_bytes() {
                int ret = 0;

                for(int i = 0; i < num_substreams; ++i) {
                        for(map<int, map<int, int> >::const_iterator it = packets[i].begin();
                                       it != packets[i].end();
                                       ++it) {
                                if(!it->second.empty()) {
                                        ret += (--it->second.end())->first + (--it->second.end())->second;
                                }
                        }
                }

                return ret;
        }

        void clear() {
                for(int i = 0; i < num_substreams; ++i) {
                        packets[i].clear();
                }
        }

        void iterator_init(int channel, packet_iterator *it) {
                it->counter = this;
                it->channel = channel;
                auto &&chan      = packets.at(channel).at(current_bufnum);
                auto &&first_pkt = chan.begin();
                it->offset       = first_pkt->first;
                it->len          = first_pkt->second;
        }
        bool next_packet(packet_iterator *it) {
                auto &&chan      = packets.at(it->channel).at(current_bufnum);
                auto &&cur_pkt = chan.find(it->offset);
                if (++cur_pkt == chan.end()) {
                        return false;
                }
                it->offset       = cur_pkt->first;
                it->len          = cur_pkt->second;
                return true;
        }

      private:
        int                             num_substreams;
        vector<map<int, map<int, int>>> packets; ///< channel, bufnum, off, len
        int current_bufnum = 0;

        friend int packet_counter_get_channels(struct packet_counter *state);
};

struct packet_counter *packet_counter_init(int num_substreams) {
        struct packet_counter *state;
        
        state = new packet_counter(num_substreams);

        return state;
}

void packet_counter_destroy(struct packet_counter *state) {
        if(state) {
                delete state;
        }
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
        return state->num_substreams;
}

void packet_counter_clear(struct packet_counter *state)
{
        state->clear();
}

/**
 * Initializes iterator @ref it to first packet of the channel.
 * Channel must be non-empty.
 *
 * @param[out] it  output iterator, struct contents shouls not be modified by the
 *                 caller
 */
void
packet_iterator_init(struct packet_counter *state, int channel,
                     struct packet_iterator *it)
{
        state->iterator_init(channel, it);
}

/**
 * @returns if there is another packet to process (and set its data to @ref it)
 */
bool
packet_next(struct packet_iterator *it)
{
        return it->counter->next_packet(it);
}
