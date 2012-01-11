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
#include "audio/utils.h"
#include "rtp/xor.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "tv.h"
#include "transmit.h"
#include "host.h"
#include "video_codec.h"
#include "compat/platform_time.h"

#define TRANSMIT_MAGIC	0xe80ab15f

extern long packet_rate;

enum fec_scheme_t {
        FEC_NONE,
        FEC_XOR,
        FEC_MULT
};

#define FEC_MAX_MULT 10

#if HAVE_MACOSX
#define GET_STARTTIME gettimeofday(&start, NULL)
#define GET_STOPTIME gettimeofday(&stop, NULL)
#define GET_DELTA delta = (stop.tv_usec - start.tv_usec) * 1000L
#else                           /* HAVE_MACOSX */
#define GET_STARTTIME clock_gettime(CLOCK_REALTIME, &start)
#define GET_STOPTIME clock_gettime(CLOCK_REALTIME, &stop)
#define GET_DELTA delta = stop.tv_nsec - start.tv_nsec
#endif                          /* HAVE_MACOSX */

/**
 * @param total_packets packets sent so far (for another tiles but within same session)
 *                    Used only when m bit is sent.
 * @return      packets sent within this function (only!!)
 */
int
tx_send_base(struct tx *tx, struct tile *tile, struct rtp *rtp_session,
                uint32_t ts, int send_m, unsigned long int total_packets,
                codec_t color_spec, double input_fps,
                enum interlacing_t interlacing, unsigned int substream);

struct tx {
        uint32_t magic;
        unsigned mtu;

        unsigned int buffer:20;

        enum fec_scheme_t fec_scheme;
        unsigned xor_leap;
        unsigned xor_streams;
        int mult_count;
};

struct tx *tx_init(unsigned mtu, char *fec)
{
        struct tx *tx;

        tx = (struct tx *)malloc(sizeof(struct tx));
        if (tx != NULL) {
                tx->magic = TRANSMIT_MAGIC;
                tx->mtu = mtu;
                tx->buffer = lrand48() & 0x3ff;
                tx->fec_scheme = FEC_NONE;
                if (fec) {
                        char *save_ptr = NULL;
                        char *item;

                        item = strtok_r(fec, ":", &save_ptr);
                        if(strcasecmp(item, "XOR") == 0) {
                                item = strtok_r(NULL, ":", &save_ptr);
                                assert(item);
                                tx->xor_leap = (unsigned int) atoi(item);
                                item = strtok_r(NULL, ":", &save_ptr);
                                assert(item);
                                tx->xor_streams = (unsigned int) atoi(item);
                                tx->fec_scheme = FEC_XOR;
                        } else if(strcasecmp(item, "mult") == 0) {
                                tx->fec_scheme = FEC_MULT;
                                item = strtok_r(NULL, ":", &save_ptr);
                                assert(item);
                                tx->mult_count = (unsigned int) atoi(item);
                                assert(tx->mult_count <= FEC_MAX_MULT);
                        }
                }
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
        unsigned int i;
        uint32_t ts = 0;
        unsigned long int packets_sent = 0u;

        ts = get_local_mediatime();

        for(i = 0; i < frame->tile_count; ++i)
        {
                int last = FALSE;
                
                if (i == frame->tile_count - 1)
                        last = TRUE;
                packets_sent += tx_send_base(tx, vf_get_tile(frame, i), rtp_session, ts, last, packets_sent,
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
        tx_send_base(tx, tile, rtp_session, ts, TRUE, 0 /* packets sent */, frame->color_spec, frame->fps, frame->interlacing, pos);
}

int
tx_send_base(struct tx *tx, struct tile *tile, struct rtp *rtp_session,
                uint32_t ts, int send_m, unsigned long int total_packets,
                codec_t color_spec, double input_fps,
                enum interlacing_t interlacing, unsigned int substream)
{
        int m, data_len;
        video_payload_hdr_t payload_hdr;
        int pt = 20;            /* A value specified in our packet format */
        const int xor_pt = 98;
        char *data;
        unsigned int pos;
#if HAVE_MACOSX
        struct timeval start, stop;
#else                           /* HAVE_MACOSX */
        struct timespec start, stop;
#endif                          /* HAVE_MACOSX */
        long delta;
        uint32_t tmp;
        unsigned int fps, fpsd, fd, fi;
        struct xor_session **xor = NULL;
        int *xor_pkts = NULL;
        int mult_pos[FEC_MAX_MULT];
        int mult_index = 0;
        int mult_first_sent = 0;
        unsigned long int packets = 0u;
        int hdrs_len = 40 + (sizeof(video_payload_hdr_t));

        if(tx->fec_scheme == FEC_XOR) {
                hdrs_len += xor_get_hdr_size();
        }

        assert(tx->magic == TRANSMIT_MAGIC);

        perf_record(UVP_SEND, ts);

        if(tx->fec_scheme == FEC_XOR) {
                unsigned int i;
                xor = calloc(50, sizeof(struct xor_session *));
                xor_pkts = calloc(50, sizeof(int));
                for (i = 0; i < tx->xor_streams; ++i) {
                        xor[i] = xor_init(sizeof(video_payload_hdr_t), tx->mtu - 40 - (sizeof(video_payload_hdr_t))); 
                        xor_clear(xor[i]);
                        xor_pkts[i] = i % tx->xor_leap;
                }
        }
        if(tx->fec_scheme == FEC_MULT) {
                int i;
                for (i = 0; i < tx->mult_count; ++i) {
                        mult_pos[i] = 0;
                }
                mult_index = 0;
        }

        m = 0;
        pos = 0;

        payload_hdr.hres = htons(tile->width);
        payload_hdr.vres = htons(tile->height);
        payload_hdr.fourcc = htonl(get_fourcc(color_spec));
        payload_hdr.length = htonl(tile->data_len);
        tmp = substream << 22;
        tmp |= tx->buffer;
        payload_hdr.substream_bufnum = htonl(tmp);

        /* word 6 */
        tmp = interlacing << 29;
        fps = round(input_fps);
        fpsd = 1;
        if(fabs(input_fps - round(input_fps) / 1.001) < 0.005)
                fd = 1;
        else
                fd = 0;
        fi = 0;

        tmp |= fps << 19;
        tmp |= fpsd << 15;
        tmp |= fd << 14;
        tmp |= fi << 13;
        payload_hdr.il_fps = htonl(tmp);

        do {
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[mult_index];
                }

                payload_hdr.offset = htonl(pos);


                data = tile->data + pos;
                data_len = tx->mtu - hdrs_len;
                data_len = (data_len / 48) * 48;
                if (pos + data_len >= tile->data_len) {
                        if (send_m && tx->fec_scheme == FEC_NONE)
                                m = 1;
                        data_len = tile->data_len - pos;
                }
                pos += data_len;
                GET_STARTTIME;
                if(data_len) { /* check needed for FEC_MULT */
                        rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                  (char *)&payload_hdr, sizeof(video_payload_hdr_t),
                                  data, data_len, 0, 0, 0);
                        packets++;
                }

                if(tx->fec_scheme == FEC_MULT) {
                        mult_pos[mult_index] = pos;
                        mult_first_sent ++;
                        if(mult_index != 0 || mult_first_sent >= (tx->mult_count - 1))
                                        mult_index = (mult_index + 1) % tx->mult_count;
                }

                if(tx->fec_scheme == FEC_XOR) {
                        unsigned int i;
                        for (i = 0; i < tx->xor_streams; ++i) {
                                xor_add_packet(xor[i], (const char *) &payload_hdr, data, data_len);
                        }
                }

                do {
                        GET_STOPTIME;
                        GET_DELTA;
                        if (delta < 0)
                                delta += 1000000000L;
                } while (packet_rate - delta > 0);

                if(tx->fec_scheme == FEC_XOR) {
                        char *hdr;
                        size_t hdr_len;
                        char *payload;
                        size_t payload_len;
                        unsigned int i;

                        for (i = 0; i < tx->xor_streams; ++i) {
                                if((unsigned int) ++xor_pkts[i] == tx->xor_leap) {
                                        xor_emit_xor_packet(xor[i], (const char **) &hdr, &hdr_len, (const char **) &payload, &payload_len);
                                
                                        if (pos + data_len >= tile->data_len) {
                                                if (send_m)
                                                        m = 1;
                                        }
                                        GET_STARTTIME;
                                        rtp_send_data_hdr(rtp_session, ts, xor_pt + i, 0 /* mbit */, 0, 0,
                                                  (char *)hdr, hdr_len,
                                                  (char *)payload, payload_len, 0, 0, 0);
                                        do {
                                                GET_STOPTIME;
                                                GET_DELTA;
                                                if (delta < 0)
                                                        delta += 1000000000L;
                                        } while (packet_rate - delta > 0);
                                        xor_clear(xor[i]);
                                        xor_pkts[i] = 0;
                                }
                        }
                }
                /* when trippling, we need all streams goes to end */
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[tx->mult_count - 1];
                }

        } while (pos < tile->data_len);

        tx->buffer ++;
        if(tx->fec_scheme == FEC_XOR) {
                char *hdr;
                size_t hdr_len;
                char *payload;
                size_t payload_len;
                unsigned int i;
                for (i = 0; i < tx->xor_streams; ++i) {
                        xor_emit_xor_packet(xor[i], (const char **) &hdr, &hdr_len, (const char **) &payload, &payload_len);
                
                        GET_STARTTIME;
                        rtp_send_data_hdr(rtp_session, ts, xor_pt + i, 0 /* mbit */, 0, 0,
                                  (char *)hdr, hdr_len,
                                  (char *)payload, payload_len, 0, 0, 0);
                        do {
                                GET_STOPTIME;
                                GET_DELTA;
                                if (delta < 0)
                                        delta += 1000000000L;
                        } while (packet_rate - delta > 0);
                        xor_clear(xor[i]);
                        xor_pkts[i] = 0;

                        xor_destroy(xor[i]);
                }
                free(xor);
                free(xor_pkts);
        }

        if(tx->fec_scheme == FEC_MULT) {
                packets /= tx->mult_count;
        }

        if(tx->fec_scheme != FEC_NONE) {
                if(send_m) {
                        uint32_t pckts_n;
                        int i;
                        pckts_n = htonl((uint32_t) (total_packets + packets));
                        /* send 3-times - only for redundancy */
                        for(i = 0; i < 3; ++i) {
                                rtp_send_data_hdr(rtp_session, ts, 120, 1 /* mbit */, 0, 0,
                                          (char *)NULL, 0,
                                          (char *) &pckts_n, sizeof(uint32_t), 0, 0, 0);
                        }
                }
        }
        return packets;
}

void audio_tx_send(struct tx* tx, struct rtp *rtp_session, audio_frame * buffer)
{
        const int pt = 21; /* PT set for audio in our packet format */
        unsigned int pos = 0u,
                     m = 0u;
        int channel;
        char *chan_data = (char *) malloc(buffer->data_len);
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
                tmp |= tx->buffer; /* bits 10-31 */
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
                        data_len = tx->mtu - 40 - sizeof(audio_payload_hdr_t);
                        if(pos + data_len >= (unsigned int) buffer->data_len / buffer->ch_count) {
                                data_len = buffer->data_len / buffer->ch_count - pos;
                                if(channel == buffer->ch_count - 1)
                                        m = 1;
                        }
                        payload_hdr.offset = htonl(pos);
                        pos += data_len;
                        
                        GET_STARTTIME;
                        
                        rtp_send_data_hdr(rtp_session, timestamp, pt, m, 0,        /* contributing sources */
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
                      
                } while (pos < (unsigned int) buffer->data_len / buffer->ch_count);
        }

        tx->buffer ++;
        free(chan_data);
}
