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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "crypto/random.h"
#include "debug.h"
#include "perf.h"
#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"
#include "rtp/ldgm.h"
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
        FEC_MULT,
        FEC_LDGM
};

#define FEC_MAX_MULT 10

#ifdef HAVE_MACOSX
#define GET_STARTTIME gettimeofday(&start, NULL)
#define GET_STOPTIME gettimeofday(&stop, NULL)
#define GET_DELTA delta = (stop.tv_usec - start.tv_usec) * 1000L
#elif defined HAVE_LINUX
#define GET_STARTTIME clock_gettime(CLOCK_REALTIME, &start)
#define GET_STOPTIME clock_gettime(CLOCK_REALTIME, &stop)
#define GET_DELTA delta = stop.tv_nsec - start.tv_nsec
#else // Windows
#define GET_STARTTIME {QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&start); }
#define GET_STOPTIME { QueryPerformanceCounter(&stop); }
#define GET_DELTA delta = (long)((double)(stop.QuadPart - start.QuadPart) * 1000 * 1000 * 1000 / freq.QuadPart);
#endif


static bool fec_is_ldgm(struct tx *tx);
static void tx_update(struct tx *tx, struct tile *tile);

static void
tx_send_base(struct tx *tx, struct tile *tile, struct rtp *rtp_session,
                uint32_t ts, int send_m,
                codec_t color_spec, double input_fps,
                enum interlacing_t interlacing, unsigned int substream,
                int fragment_offset);

struct tx {
        uint32_t magic;
        unsigned mtu;
        double max_loss;

        uint32_t last_ts;
        int      last_frame_fragment_id;

        unsigned int buffer:22;
        unsigned long int sent_frames;
        int32_t avg_len;
        int32_t avg_len_last;

        enum fec_scheme_t fec_scheme;
        void *fec_state;
        int mult_count;

        int last_fragment;
};

static bool fec_is_ldgm(struct tx *tx)
{
        return tx->fec_scheme == FEC_LDGM && tx->fec_state;
}

static void tx_update(struct tx *tx, struct tile *tile)
{
        if(!tile) {
                return;
        }
        
        uint64_t tmp_avg = tx->avg_len * tx->sent_frames + tile->data_len;
        tx->sent_frames++;
        tx->avg_len = tmp_avg / tx->sent_frames;
        if(tx->sent_frames >= 100) {
                if(tx->fec_scheme == FEC_LDGM && tx->max_loss > 0.0) {
                        if(abs(tx->avg_len_last - tx->avg_len) > tx->avg_len / 3) {
                                int data_len = tx->mtu -  (40 + (sizeof(ldgm_video_payload_hdr_t)));
                                data_len = (data_len / 48) * 48;
                                void *fec_state_old = tx->fec_state;
                                tx->fec_state = ldgm_encoder_init_with_param(data_len, tx->avg_len, tx->max_loss);
                                if(tx->fec_state != NULL) {
                                        tx->avg_len_last = tx->avg_len;
                                        ldgm_encoder_destroy(fec_state_old);
                                } else {
                                        tx->fec_state = fec_state_old;
                                        if(!tx->fec_state) {
                                                fprintf(stderr, "Unable to initialize FEC.\n");
                                                exit_uv(1);
					}
                                }
                        }
                }
                tx->avg_len = 0;
                tx->sent_frames = 0;
        }
}

struct tx *tx_init(unsigned mtu, char *fec)
{
        struct tx *tx;

        tx = (struct tx *)malloc(sizeof(struct tx));
        if (tx != NULL) {
                tx->magic = TRANSMIT_MAGIC;
                tx->mult_count = 0;
                tx->max_loss = 0.0;
                tx->fec_state = NULL;
                tx->mtu = mtu;
                tx->buffer = lrand48() & 0x3fffff;
                tx->avg_len = tx->avg_len_last = tx->sent_frames = 0u;
                tx->fec_scheme = FEC_NONE;
                tx->last_frame_fragment_id = -1;
                if (fec) {
			char *fec_cfg = NULL;
			if(strchr(fec, ':')) {
				char *delim = strchr(fec, ':');
				*delim = '\0';
				fec_cfg = delim + 1;
			}

                        if(strcasecmp(fec, "none") == 0) {
                                tx->fec_scheme = FEC_NONE;
                        } else if(strcasecmp(fec, "mult") == 0) {
                                tx->fec_scheme = FEC_MULT;
                                assert(fec_cfg);
                                tx->mult_count = (unsigned int) atoi(fec_cfg);
                                assert(tx->mult_count <= FEC_MAX_MULT);
                        } else if(strcasecmp(fec, "LDGM") == 0) {
                                tx->fec_scheme = FEC_LDGM;
                                if(!fec_cfg || (strlen(fec_cfg) > 0 && strchr(fec_cfg, '%') == NULL)) {
                                        tx->fec_state = ldgm_encoder_init_with_cfg(fec_cfg);
                                        if(tx->fec_state == NULL) {
                                                fprintf(stderr, "Unable to initialize LDGM.\n");
                                                free(tx);
                                                return NULL;
                                        }
                                } else { // delay creation until we have avarage frame size
                                        tx->max_loss = atof(fec_cfg);
                                }
                        } else {
                                fprintf(stderr, "Unknown FEC: %s\n", fec);
                                free(tx);
                                return NULL;
                        }
                }
        }
        return tx;
}

void tx_done(struct tx *tx)
{
        assert(tx->magic == TRANSMIT_MAGIC);
        ldgm_encoder_destroy(tx->fec_state);
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

        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tile are not currently supported for fragmented send

        ts = get_local_mediatime();
        if(frame->fragment &&
                        tx->last_frame_fragment_id == frame->frame_fragment_id) {
                ts = tx->last_ts;
        } else {
                tx->last_frame_fragment_id = frame->frame_fragment_id;
                tx->last_ts = ts;
        }

        for(i = 0; i < frame->tile_count; ++i)
        {
                int last = FALSE;
                int fragment_offset = 0;
                
                if (i == frame->tile_count - 1) {
                        if(!frame->fragment || frame->last_fragment)
                                last = TRUE;
                }
                if(frame->fragment)
                        fragment_offset = vf_get_tile(frame, i)->offset;

                tx_send_base(tx, vf_get_tile(frame, i), rtp_session, ts, last,
                                frame->color_spec, frame->fps, frame->interlacing,
                                i, fragment_offset);
                tx->buffer ++;
        }
}


void
tx_send_tile(struct tx *tx, struct video_frame *frame, int pos, struct rtp *rtp_session)
{
        struct tile *tile;
        int last = FALSE;
        uint32_t ts = 0;
        int fragment_offset = 0;

        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tile are not currently supported for fragmented send
        
        tile = vf_get_tile(frame, pos);
        ts = get_local_mediatime();
        if(frame->fragment &&
                        tx->last_frame_fragment_id == frame->frame_fragment_id) {
                ts = tx->last_ts;
        } else {
                tx->last_frame_fragment_id = frame->frame_fragment_id;
                tx->last_ts = ts;
        }
        if(!frame->fragment || frame->last_fragment)
                last = TRUE;
        if(frame->fragment)
                fragment_offset = vf_get_tile(frame, pos)->offset;
        tx_send_base(tx, tile, rtp_session, ts, last, frame->color_spec, frame->fps, frame->interlacing, pos,
                        fragment_offset);
        tx->buffer ++;
}

static void
tx_send_base(struct tx *tx, struct tile *tile, struct rtp *rtp_session,
                uint32_t ts, int send_m,
                codec_t color_spec, double input_fps,
                enum interlacing_t interlacing, unsigned int substream,
                int fragment_offset)
{
        int m, data_len;
        // see definition in rtp_callback.h
        video_payload_hdr_t video_hdr;
        ldgm_video_payload_hdr_t ldgm_hdr;
        int pt = PT_VIDEO;            /* A value specified in our packet format */
        char *data;
        unsigned int pos;
#ifdef HAVE_LINUX
        struct timespec start, stop;
#elif defined HAVE_MACOSX
        struct timeval start, stop;
#else // Windows
	LARGE_INTEGER start, stop, freq;
#endif
        long delta;
        uint32_t tmp;
        unsigned int fps, fpsd, fd, fi;
        int mult_pos[FEC_MAX_MULT];
        int mult_index = 0;
        int mult_first_sent = 0;
        int hdrs_len = 40 + (sizeof(video_payload_hdr_t));
        char *data_to_send;
        int data_to_send_len;

        assert(tx->magic == TRANSMIT_MAGIC);

        tx_update(tx, tile);

        perf_record(UVP_SEND, ts);

        data_to_send = tile->data;
        data_to_send_len = tile->data_len;

        if(tx->fec_scheme == FEC_MULT) {
                int i;
                for (i = 0; i < tx->mult_count; ++i) {
                        mult_pos[i] = 0;
                }
                mult_index = 0;
        }

        m = 0;
        pos = 0;

        video_hdr[3] = htonl(tile->width << 16 | tile->height);
        video_hdr[4] = get_fourcc(color_spec);
        video_hdr[2] = htonl(data_to_send_len);
        tmp = substream << 22;
        tmp |= 0x3fffff & tx->buffer;
        video_hdr[0] = htonl(tmp);

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
        video_hdr[5] = htonl(tmp);

        char *hdr;
        int hdr_len;

        if(fec_is_ldgm(tx)) {
                hdrs_len = 40 + (sizeof(ldgm_video_payload_hdr_t));
                ldgm_encoder_encode(tx->fec_state, (char *) &video_hdr, sizeof(video_hdr),
                                tile->data, tile->data_len, &data_to_send, &data_to_send_len);
                tmp = substream << 22;
                tmp |= 0x3fffff & tx->buffer;
                // see definition in rtp_callback.h
                ldgm_hdr[0] = htonl(tmp);
                ldgm_hdr[2] = htonl(data_to_send_len);
                ldgm_hdr[3] = htonl(
                                (ldgm_encoder_get_k(tx->fec_state)) << 19 |
                                (ldgm_encoder_get_m(tx->fec_state)) << 6 |
                                ldgm_encoder_get_c(tx->fec_state));
                ldgm_hdr[4] = htonl(ldgm_encoder_get_seed(tx->fec_state));

                pt = PT_VIDEO_LDGM;

                hdr = (char *) &ldgm_hdr;
                hdr_len = sizeof(ldgm_hdr);
        } else {
                hdr = (char *) &video_hdr;
                hdr_len = sizeof(video_hdr);
        }

        do {
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[mult_index];
                }

                int offset = pos + fragment_offset;

                video_hdr[1] = htonl(offset);
                if(fec_is_ldgm(tx)) {
                        ldgm_hdr[1] = htonl(offset);
                }


                data = data_to_send + pos;
                data_len = tx->mtu - hdrs_len;
                data_len = (data_len / 48) * 48;
                if (pos + data_len >= (unsigned int) data_to_send_len) {
                        if (send_m) {
                                m = 1;
                        }
                        data_len = data_to_send_len - pos;
                }
                pos += data_len;
                GET_STARTTIME;
                if(data_len) { /* check needed for FEC_MULT */
                        rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                  hdr, hdr_len,
                                  data, data_len, 0, 0, 0);
                        if(m && tx->fec_scheme != FEC_NONE) {
                                int i;
                                for(i = 0; i < 5; ++i) {
                                        rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                                  hdr, hdr_len,
                                                  data, data_len, 0, 0, 0);
                                }
                        }
                }

                if(tx->fec_scheme == FEC_MULT) {
                        mult_pos[mult_index] = pos;
                        mult_first_sent ++;
                        if(mult_index != 0 || mult_first_sent >= (tx->mult_count - 1))
                                        mult_index = (mult_index + 1) % tx->mult_count;
                }

                do {
                        GET_STOPTIME;
                        GET_DELTA;
                        if (delta < 0)
                                delta += 1000000000L;
                } while (packet_rate - delta > 0);

                /* when trippling, we need all streams goes to end */
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[tx->mult_count - 1];
                }

        } while (pos < (unsigned int) data_to_send_len);

        if(fec_is_ldgm(tx)) {
               ldgm_encoder_free_buffer(tx->fec_state, data_to_send);
        }
}

/* 
 * This multiplication scheme relies upon the fact, that our RTP/pbuf implementation is
 * not sensitive to packet duplication. Otherwise, we can get into serious problems.
 */
void audio_tx_send(struct tx* tx, struct rtp *rtp_session, audio_frame2 * buffer)
{
        const int pt = PT_AUDIO; /* PT set for audio in our packet format */
        unsigned int pos = 0u,
                     m = 0u;
        int channel;
        char *chan_data;
        int data_len;
        char *data;
        // see definition in rtp_callback.h
        audio_payload_hdr_t payload_hdr;
        uint32_t timestamp;
#ifdef HAVE_LINUX
        struct timespec start, stop;
#elif defined HAVE_MACOSX
        struct timeval start, stop;
#else // Windows
	LARGE_INTEGER start, stop, freq;
#endif
        long delta;
        int mult_pos[FEC_MAX_MULT];
        int mult_index = 0;
        int mult_first_sent = 0;
        
        if(fec_is_ldgm(tx)) {
                fprintf(stderr, "LDGM is not currently supported for audio! "
                                "Exitting...\n");
                exit_uv(129);
        }

        timestamp = get_local_mediatime();
        perf_record(UVP_SEND, timestamp);

        for(channel = 0; channel < buffer->ch_count; ++channel)
        {
                chan_data = buffer->data[channel];
                pos = 0u;

                if(tx->fec_scheme == FEC_MULT) {
                        int i;
                        for (i = 0; i < tx->mult_count; ++i) {
                                mult_pos[i] = 0;
                        }
                        mult_index = 0;
                }

                uint32_t tmp;
                tmp = channel << 22; /* bits 0-9 */
                tmp |= tx->buffer; /* bits 10-31 */
                payload_hdr[0] = htonl(tmp);

                payload_hdr[2] = htonl(buffer->data_len[channel]);

                /* fourth word */
                tmp = (buffer->bps * 8) << 26;
                tmp |= buffer->sample_rate;
                payload_hdr[3] = htonl(tmp);

                /* fifth word */
                payload_hdr[4] = htonl(get_audio_tag(buffer->codec));

                do {
                        if(tx->fec_scheme == FEC_MULT) {
                                pos = mult_pos[mult_index];
                        }

                        data = chan_data + pos;
                        data_len = tx->mtu - 40 - sizeof(audio_payload_hdr_t);
                        if(pos + data_len >= (unsigned int) buffer->data_len[channel]) {
                                data_len = buffer->data_len[channel] - pos;
                                if(channel == buffer->ch_count - 1)
                                        m = 1;
                        }
                        payload_hdr[1] = htonl(pos);
                        pos += data_len;
                        
                        GET_STARTTIME;
                        
                        if(data_len) { /* check needed for FEC_MULT */
                                rtp_send_data_hdr(rtp_session, timestamp, pt, m, 0,        /* contributing sources */
                                      0,        /* contributing sources length */
                                      (char *) &payload_hdr, sizeof(payload_hdr),
                                      data, data_len,
                                      0, 0, 0);
                        }

                        if(tx->fec_scheme == FEC_MULT) {
                                mult_pos[mult_index] = pos;
                                mult_first_sent ++;
                                if(mult_index != 0 || mult_first_sent >= (tx->mult_count - 1))
                                                mult_index = (mult_index + 1) % tx->mult_count;
                        }

                        do {
                                GET_STOPTIME;
                                GET_DELTA;
                                if (delta < 0)
                                        delta += 1000000000L;
                        } while (packet_rate - delta > 0);

                        /* when trippling, we need all streams goes to end */
                        if(tx->fec_scheme == FEC_MULT) {
                                pos = mult_pos[tx->mult_count - 1];
                        }

                      
                } while (pos < (unsigned int) buffer->data_len[channel]);
        }

        tx->buffer ++;
}
