/**
 * @file   utils/audio_buffer.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016 CESNET z.s.p.o.
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

#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "utils/audio_buffer.h"
#include "utils/ring_buffer.h"

#define WINDOW 50

#define max(a, b)      (((a) > (b))? (a): (b))
#define min(a, b)      (((a) < (b))? (a): (b))

#define BUF_LAST_UNDERRUN_MAX 1000000000
#define BUF_LAST_UNDERRUN_THRESHOLD 10000
#define AGGRESIVITY_MAX 4
#define AGGRESIVITY_STEP 100

static const int occupacy_windows[3] = { 50, 200 };

struct audio_buffer {
        struct audio_desc desc;
        ring_buffer_t *ring;
        int suggested_latency_ms;

        // moving averages
        int in_pkt_size;
        int out_pkt_size;
        int avg_occupancy[2]; // at read time
        int last_underrun; // last underrun n output frames ago
        int aggresivity;
        int last_aggresivity_change;
};

struct audio_buffer *audio_buffer_init(int sample_rate, int bps, int ch_count, int suggested_latency_ms)
{
        struct audio_buffer *buf = calloc(1, sizeof(struct audio_buffer));
        buf->desc.sample_rate = sample_rate;
        buf->desc.bps = bps;
        buf->desc.ch_count = ch_count;

        buf->last_underrun = BUF_LAST_UNDERRUN_MAX;

        buf->ring = ring_buffer_init(sample_rate * bps * ch_count);

        buf->suggested_latency_ms = suggested_latency_ms;

        buf->aggresivity = 1;
        buf->last_aggresivity_change = AGGRESIVITY_STEP;

        return buf;
}

void audio_buffer_destroy(struct audio_buffer *buf)
{
        if (buf) {
                ring_buffer_destroy(buf->ring);
        }
}

int audio_buffer_read(struct audio_buffer *buf, char *out, int max_len)
{
        if (buf->out_pkt_size > 0) {
                buf->out_pkt_size = (max_len + (buf->out_pkt_size * (WINDOW-1))) / WINDOW;
        } else {
                buf->out_pkt_size = max_len;
        }

        int ring_size = ring_get_current_size(buf->ring);

        for (unsigned int i = 0; i < sizeof buf->avg_occupancy / sizeof buf->avg_occupancy; ++i) {
                if (buf->avg_occupancy[i] > 0) {
                        buf->avg_occupancy[i] = (ring_size + (buf->avg_occupancy[i] * (occupacy_windows[i] -1))) / occupacy_windows[i];
                } else {
                        buf->avg_occupancy[i] = ring_size;
                }
        }

        if (ring_size < max_len) {
                buf->last_underrun = 0;
        } else {
                if (buf->last_underrun < BUF_LAST_UNDERRUN_MAX) {
                        buf->last_underrun += 1;
                }
        }

        int suggested_latency_bytes = buf->suggested_latency_ms * buf->desc.bps * buf->desc.ch_count * buf->desc.sample_rate / 1000;
        int requested_latency_bytes = max(suggested_latency_bytes, 2*max(buf->in_pkt_size, buf->out_pkt_size));

        int ret = ring_buffer_read(buf->ring, out, max_len);

        int remaining_bytes = ring_size - ret;

        if (buf->last_aggresivity_change >= AGGRESIVITY_STEP) {
                buf->last_aggresivity_change = 0;
                if ((buf->avg_occupancy[0] > buf->avg_occupancy[1] && buf->last_underrun > BUF_LAST_UNDERRUN_THRESHOLD / 10)) {
                        buf->aggresivity = min(buf->aggresivity + 1, AGGRESIVITY_MAX);
                } else if (buf->avg_occupancy[0] < buf->avg_occupancy[1] || buf->last_underrun < BUF_LAST_UNDERRUN_THRESHOLD / 100) {
                        buf->aggresivity = max(buf->aggresivity - 1, 1);
                }
        } else {
                buf->last_aggresivity_change += 1;
        }

        if (requested_latency_bytes < remaining_bytes) {
                int len_drop = (1<<buf->aggresivity) * buf->desc.bps * buf->desc.ch_count;
                len_drop = min(len_drop, remaining_bytes / 2);

                char *tmp = alloca(len_drop);
                ring_buffer_read(buf->ring, tmp, len_drop);
        }

        log_msg(LOG_LEVEL_DEBUG, "buf - in avg %d, out avg %d, occupancy avg %d, last underrun %d, aggresivity %d\n", buf->in_pkt_size, buf->out_pkt_size, buf->avg_occupancy[0], buf->last_underrun, buf->aggresivity);

        return ret;
}

void audio_buffer_write(struct audio_buffer *buf, const char *in, int len)
{
        if (buf->in_pkt_size > 0) {
                buf->in_pkt_size = (len + (buf->in_pkt_size * (WINDOW-1))) / WINDOW;
        } else {
                buf->in_pkt_size = len;
        }
        ring_buffer_write(buf->ring, in, len);
}

