/*
 * FILE:     transmit.c
 * AUTHOR:  Colin Perkins <csp@csperkins.org>
 *          Ladan Gharai
 *          Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2004 University of Southern California
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
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "audio/audio.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "tv.h"
#include "transmit.h"
#include "host.h"
#include "video_codec.h"
#include "compat/platform_time.h"

#define TRANSMIT_MAGIC	0xe80ab15f

extern long packet_rate;

#if HAVE_MACOSX
#define GET_STARTTIME gettimeofday(&start, NULL)
#define GET_STOPTIME gettimeofday(&stop, NULL)
#define GET_DELTA delta = (stop.tv_usec - start.tv_usec) * 1000L
#else                           /* HAVE_MACOSX */
#define GET_STARTTIME clock_gettime(CLOCK_REALTIME, &start)
#define GET_STOPTIME clock_gettime(CLOCK_REALTIME, &stop)
#define GET_DELTA delta = stop.tv_nsec - start.tv_nsec
#endif                          /* HAVE_MACOSX */

void
tx_send_base(struct tx *tx, struct tile *tile, struct rtp *rtp_session,
                uint32_t ts, int send_m, codec_t color_spec, double fps,
                enum interlacing_t interlacing, unsigned int substream);

struct tx {
        uint32_t magic;
        unsigned mtu;
        unsigned int buffer:20;
};

struct tx *tx_init(unsigned mtu)
{
        struct tx *tx;

        tx = (struct tx *)malloc(sizeof(struct tx));
        if (tx != NULL) {
                tx->magic = TRANSMIT_MAGIC;
                tx->mtu = mtu;
                tx->buffer = lrand48() & 0x3ff;
        }
        return tx;
}

void tx_done(struct tx *tx)
{
        assert(tx->magic == TRANSMIT_MAGIC);
        free(tx);
}

/*
 * sends one or more frames (tiles) with same TS in one RTP stream. Only one m-bit is set.
 */
void
tx_send(struct tx *tx, struct video_frame *frame, struct rtp *rtp_session)
{
        int i, j;
        uint32_t ts = 0;
        struct tile *tile;

        ts = get_local_mediatime();

        for(i = 0; i < frame->tile_count; ++i)
        {
                int last = FALSE;
                
                if (i == frame->tile_count - 1)
                        last = TRUE;
                tx_send_base(tx, vf_get_tile(frame, i), rtp_session, ts, last,
                                frame->color_spec, frame->fps, frame->interlacing,
                                i);
        }
}


void
tx_send_tile(struct tx *tx, struct video_frame *frame, int pos, struct rtp *rtp_session)
{
        struct tile *tile;
        
        tile = vf_get_tile(frame, pos);
        uint32_t ts = 0;
        ts = get_local_mediatime();
        tx_send_base(tx, tile, rtp_session, ts, TRUE, frame->color_spec, frame->fps, frame->interlacing, pos);
}

void
tx_send_base(struct tx *tx, struct tile *tile, struct rtp *rtp_session,
                uint32_t ts, int send_m, codec_t color_spec, double fps,
                enum interlacing_t interlacing, unsigned int substream)
{
        int m, data_len;
        video_payload_hdr_t payload_hdr;
        int pt = 96;            /* A dynamic payload type for the tests... */
        char *data;
        unsigned int pos;
#if HAVE_MACOSX
        struct timeval start, stop;
#else                           /* HAVE_MACOSX */
        struct timespec start, stop;
#endif                          /* HAVE_MACOSX */
        long delta;
        uint32_t tmp;

        assert(tx->magic == TRANSMIT_MAGIC);

        perf_record(UVP_SEND, ts);

        m = 0;
        pos = 0;

        payload_hdr.hres = htons(tile->width);
        payload_hdr.vres = htons(tile->height);
        payload_hdr.fourcc = htonl(get_fourcc(color_spec));
        payload_hdr.length = htonl(tile->data_len);
        payload_hdr.fps = htonl(fps * 65536.0);
        tmp = substream << 22;
        tmp |= tx->buffer << 2;
        tmp |= interlacing;
        payload_hdr.substream_bufnum_il = htonl(tmp);

        do {
                payload_hdr.offset = htonl(pos);

                data = tile->data + pos;
                data_len = tx->mtu - 40 - (sizeof(video_payload_hdr_t));
                data_len = (data_len / 48) * 48;
                if (pos + data_len >= tile->data_len) {
                        if (send_m)
                                m = 1;
                        data_len = tile->data_len - pos;
                }
                pos += data_len;
                GET_STARTTIME;
                rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                  (char *)&payload_hdr, sizeof(video_payload_hdr_t),
                                  data, data_len, 0, 0, 0);
                do {
                        GET_STOPTIME;
                        GET_DELTA;
                        if (delta < 0)
                                delta += 1000000000L;
                } while (packet_rate - delta > 0);

        } while (pos < tile->data_len);

        tx->buffer ++;
}

void audio_tx_send(struct tx* tx, struct rtp *rtp_session, audio_frame * buffer)
{
        const int mtu = 1500; // perhaps to be added as parameter to function call ?
        //audio_frame_to_network_buffer(buffer->tmp_buffer, buffer);
        unsigned int pos = 0u,
                     m = 0u;
        int channel;
        char *chan_data = (char *) malloc(buffer->data_len);
        int buffer_len;
        int data_len;
        char *data;
        audio_payload_hdr_t payload_hdr;
        uint32_t timestamp;
#if HAVE_MACOSX
        struct timeval start, stop;
#else                           /* HAVE_MACOSX */
        struct timespec start, stop;
#endif                          /* HAVE_MACOSX */
        long delta;
        
        timestamp = get_local_mediatime();
        perf_record(UVP_SEND, timestamp);

        for(channel = 0; channel < buffer->ch_count; ++channel)
        {
                demux_channel(chan_data, buffer->data, buffer->bps, buffer->data_len, buffer->ch_count, channel);
                pos = 0u;

                uint32_t tmp;
                tmp = channel << 22; /* bits 0-9 */
                tmp |= tx->buffer << 2; /* bits 10-29 */
                payload_hdr.substream_bufnum = htonl(tmp);

                payload_hdr.length = htonl(buffer->data_len / buffer->ch_count);

                /* fourth word */
                tmp = (buffer->bps * 8) << 26;
                tmp |= buffer->sample_rate;
                payload_hdr.quant_sample_rate = htonl(tmp);

                /* fifth word */
                payload_hdr.audio_tag = htonl(1); /* PCM */

                do {
                        data = chan_data + pos;
                        data_len = mtu - 40 - sizeof(audio_payload_hdr_t);
                        if(pos + data_len >= buffer->data_len / buffer->ch_count) {
                                data_len = buffer->data_len / buffer->ch_count - pos;
                                if(channel == buffer->ch_count - 1)
                                        m = 1;
                        }
                        payload_hdr.offset = htonl(pos);
                        pos += data_len;
                        
                        GET_STARTTIME;
                        
                        rtp_send_data_hdr(rtp_session, timestamp, audio_payload_type, m, 0,        /* contributing sources */
                              0,        /* contributing sources length */
                              (char *) &payload_hdr, sizeof(payload_hdr),
                              data, data_len,
                              0, 0, 0);
                        do {
                                GET_STOPTIME;
                                GET_DELTA;
                                if (delta < 0)
                                        delta += 1000000000L;
                        } while (packet_rate - delta > 0);
                      
                } while (pos < buffer->data_len / buffer->ch_count);
        }

        tx->buffer ++;
        free(chan_data);
}
