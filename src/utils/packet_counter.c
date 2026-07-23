/**
 * @file   utils/packet_counter.c
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

#include "utils/packet_counter.h"

#include <assert.h>   // for assert
#include <inttypes.h> // for uint32_t, uint16_t
#include <limits.h>   // for ULONG_MAX
#include <stdint.h>
#include <stdlib.h> // for free, size_t, calloc, qsort, realloc

#include "compat/c23.h"   // IWYU pragma: keep
#include "utils/macros.h" // for to_fourcc

#define MAGIC to_fourcc('U', 'T', 'p', 'c')

/**
 * The member field order is just for better memory efficiency,
 * for sorting, the packet_len goes at the end (otherwise the
 * order matches).
 */
struct packet {
        uint16_t substream_id;
        uint16_t packet_len;
        uint32_t buffer_number;
        uint32_t offset;
};

struct packet_counter {
        uint32_t magic;
        int      num_substreams;

        struct packet *packets;
        size_t         packets_allocated;
        size_t         packets_count;
};

struct packet_counter *
packet_counter_init()
{
        struct packet_counter *s = calloc(1, sizeof *s);
        s->magic                 = MAGIC;
        return s;
}

void
packet_counter_destroy(struct packet_counter *s)
{
        if (s == nullptr) {
                return;
        }
        assert(s->magic == MAGIC);
        free(s->packets);
        free(s);
}

void
packet_counter_register_packet(struct packet_counter *s,
                               unsigned int substream_id, unsigned int bufnum,
                               unsigned int offset, unsigned int len)
{
        assert(len <= UINT16_MAX);
        if (s->packets_count == s->packets_allocated) {
                s->packets_allocated = 2 * (s->packets_allocated + 1);
                s->packets = realloc(s->packets, s->packets_allocated *
                                                     sizeof(struct packet));
                assert(s->packets != nullptr);
        }
        s->packets[s->packets_count].substream_id  = substream_id;
        s->packets[s->packets_count].packet_len    = len;
        s->packets[s->packets_count].buffer_number = bufnum;
        s->packets[s->packets_count].offset        = offset;
        s->packets_count += 1;
}

static int
compare(const void *a, const void *b)
{
        const struct packet *packet_a = a;
        const struct packet *packet_b = b;
        if (packet_a->substream_id != packet_b->substream_id) {
                return packet_a->substream_id - packet_b->substream_id;
        }
        if (packet_a->buffer_number != packet_b->buffer_number) {
                return packet_a->buffer_number < packet_b->buffer_number ? -1
                                                                         : 1;
        }
        if (packet_a->offset != packet_b->offset) {
                return packet_a->offset < packet_b->offset ? -1 : 1;
        }
        if (packet_a->packet_len != packet_b->packet_len) {
                return packet_a->packet_len - packet_b->packet_len;
        }

        return 0;
}

/**
 * @param[out] expected_o  number of expected bytes
 * @param[out] received_o  actual number of received bytes
 *
 * @note
 * The reported number of expected bytes may be lees than the actual,
 * see the inline comment below. Usually it works ok for big buffers
 * but not for small containing few or even one packet (as in audio).
 */
void
packet_counter_get_bytes(struct packet_counter *s, long *expected_o,
                         long *received_o)
{
        qsort(s->packets, s->packets_count, sizeof s->packets[0], compare);
        long expected = 0;
        long received = 0;

        unsigned long last_stream = ULONG_MAX;
        unsigned long last_offset = ULONG_MAX;
        unsigned long last_bufnum = ULONG_MAX;

        for (unsigned long i = 0; i < s->packets_count; ++i) {
                const struct packet *p = s->packets + i;
                if (p->substream_id == last_stream &&
                    p->buffer_number == last_bufnum &&
                    p->offset == last_offset) {
                        continue; // skip pkt dup
                }
                received += p->packet_len;
                // last packet of a buffer in a substeram (count expected as its
                // offset+len); buffers with no packes can therefor not be
                // counted, also not detected are missing packets at the end of
                // the buffer
                if (i == s->packets_count - 1 ||
                    p[1].substream_id != p->substream_id ||
                    p[1].buffer_number != p->buffer_number) {
                        expected += p->offset + p->packet_len;
                }
                last_stream = p->substream_id;
                last_offset = p->offset;
                last_bufnum = p->buffer_number;
        }
        *expected_o = expected;
        *received_o = received;
}

void
packet_counter_clear(struct packet_counter *s)
{
        s->packets_count = 0;
}
