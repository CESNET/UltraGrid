/*
 * FILE:    utils/packet_counter.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */

#include "utils/packet_counter.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <map>
#include <vector>

using namespace std;

struct packet_counter {
        packet_counter(int num_substreams) {
                this->num_substreams = num_substreams;

                substream_data.reserve(num_substreams);
                for(int i = 0; i < num_substreams; ++i) {
                        substream_data.push_back(map<int, map<int, int> > ());
                }
        }

        ~packet_counter() {
        }

        void register_packet(int substream_id, int bufnum, int offset, int len) {
                assert(substream_id < num_substreams); 

                substream_data[substream_id][bufnum][offset] = len;
        }

        bool has_packet(int substream_id, int bufnum, int offset, int len) {
                assert(substream_id < num_substreams); 

                return substream_data[substream_id][bufnum][offset] != 0;
        }


        int get_total_bytes() {
                int ret = 0;

                for(int i = 0; i < num_substreams; ++i) {
                        for(map<int, map<int, int> >::const_iterator it = substream_data[i].begin();
                                       it != substream_data[i].end();
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
                        for(map<int, map<int, int> >::const_iterator it = substream_data[i].begin();
                                       it != substream_data[i].end();
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
                        substream_data[i].clear();
                }
        }

        vector<map<int, map<int, int> > > substream_data;
        int num_substreams;
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

bool packet_counter_has_packet(struct packet_counter *state, unsigned int substream_id,
                unsigned int bufnum, unsigned int offset, unsigned int len) 
{
        return state->has_packet(substream_id, bufnum, offset, len);
}

int packet_counter_get_total_bytes(struct packet_counter *state)
{
        return state->get_total_bytes();
}

int packet_counter_get_all_bytes(struct packet_counter *state)
{
        return state->get_all_bytes();
}

void packet_counter_clear(struct packet_counter *state)
{
        state->clear();
}

