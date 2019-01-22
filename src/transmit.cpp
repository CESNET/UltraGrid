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
 *          David Cassany    <david.cassany@i2cat.net>
 *          Ignacio Contreras <ignacio.contreras@i2cat.net>
 *          Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Jordi "Txor" Casas Ríos <txorlings@gmail.com>
 *          Martin Pulec     <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2001-2004 University of Southern California
 * Copyright (c) 2005-2017 CESNET z.s.p.o.
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

#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"
#include "crypto/random.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "perf.h"
#include "crypto/openssl_encrypt.h"
#include "module.h"
#include "rang.hpp"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/rtpenc_h264.h"
#include "tv.h"
#include "transmit.h"
#include "utils/jpeg_reader.h"
#include "video.h"
#include "video_codec.h"

#include <algorithm>

#define TRANSMIT_MAGIC	0xe80ab15f

#define FEC_MAX_MULT 10

#ifdef HAVE_MACOSX
#define GET_STARTTIME gettimeofday(&start, NULL)
#define GET_STOPTIME gettimeofday(&stop, NULL)
#define GET_DELTA delta = (stop.tv_sec - start.tv_sec) * 1000000000l + (stop.tv_usec - start.tv_usec) * 1000L
#elif defined HAVE_LINUX
#define GET_STARTTIME clock_gettime(CLOCK_REALTIME, &start)
#define GET_STOPTIME clock_gettime(CLOCK_REALTIME, &stop)
#define GET_DELTA delta = (stop.tv_sec - start.tv_sec) * 1000000000l + stop.tv_nsec - start.tv_nsec
#else // Windows
#define GET_STARTTIME {QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&start); }
#define GET_STOPTIME { QueryPerformanceCounter(&stop); }
#define GET_DELTA delta = (long)((double)(stop.QuadPart - start.QuadPart) * 1000 * 1000 * 1000 / freq.QuadPart);
#endif

#define DEFAULT_CIPHER_MODE MODE_AES128_CFB

static void tx_update(struct tx *tx, struct video_frame *frame, int substream);
static void tx_done(struct module *tx);
static uint32_t format_interl_fps_hdr_row(enum interlacing_t interlacing, double input_fps);

static void
tx_send_base(struct tx *tx, struct video_frame *frame, struct rtp *rtp_session,
                uint32_t ts, int send_m,
                unsigned int substream,
                int fragment_offset);


static bool set_fec(struct tx *tx, const char *fec);
static void fec_check_messages(struct tx *tx);

struct tx {
        struct module mod;

        uint32_t magic;
        enum tx_media_type media_type;
        unsigned mtu;
        double max_loss;

        uint32_t last_ts;
        int      last_frame_fragment_id;

        unsigned int buffer:22;
        unsigned long int sent_frames;
        int32_t avg_len;
        int32_t avg_len_last;

        enum fec_type fec_scheme;
        int mult_count;

        int last_fragment;

        const struct openssl_encrypt_info *enc_funcs;
        struct openssl_encrypt *encryption;
        long long int bitrate;
		
        struct rtpenc_h264_state *rtpenc_h264_state;
        char tmp_packet[RTP_MAX_MTU];
};

static void tx_update(struct tx *tx, struct video_frame *frame, int substream)
{
        if(!frame) {
                return;
        }
        
        uint64_t tmp_avg = tx->avg_len * tx->sent_frames + frame->tiles[substream].data_len *
                (frame->fec_params.type != FEC_NONE ?
                 (double) frame->fec_params.k / (frame->fec_params.k + frame->fec_params.m) :
                 1);
        tx->sent_frames++;
        tx->avg_len = tmp_avg / tx->sent_frames;
        if(tx->sent_frames >= 100) {
                if(tx->fec_scheme == FEC_LDGM && tx->max_loss > 0.0) {
                        if(abs(tx->avg_len_last - tx->avg_len) > tx->avg_len / 3) {
                                int data_len = tx->mtu -  (40 + (sizeof(fec_video_payload_hdr_t)));
                                data_len = (data_len / 48) * 48;
                                //void *fec_state_old = tx->fec_state;

                                struct msg_sender *msg = (struct msg_sender *)
                                        new_message(sizeof(struct msg_sender));
                                snprintf(msg->fec_cfg, sizeof(msg->fec_cfg), "LDGM percents %d %d %f",
                                                data_len, tx->avg_len, tx->max_loss);
                                msg->type = SENDER_MSG_CHANGE_FEC;
                                struct response *resp = send_message_to_receiver(get_parent_module(&tx->mod),
                                                (struct message *) msg);
                                free_response(resp);
                                tx->avg_len_last = tx->avg_len;
                        }
                }
                tx->avg_len = 0;
                tx->sent_frames = 0;
        }
}

struct tx *tx_init(struct module *parent, unsigned mtu, enum tx_media_type media_type,
                const char *fec, const char *encryption, long long int bitrate)
{
        struct tx *tx;

        if (mtu > RTP_MAX_MTU) {
                log_msg(LOG_LEVEL_ERROR, "Requested MTU exceeds maximal value allowed by RTP library (%d B).\n", RTP_MAX_MTU);
                return NULL;
        }

        tx = (struct tx *) calloc(1, sizeof(struct tx));
        if (tx != NULL) {
                module_init_default(&tx->mod);
                tx->mod.cls = MODULE_CLASS_TX;
                tx->mod.priv_data = tx;
                tx->mod.deleter = tx_done;
                module_register(&tx->mod, parent);

                tx->magic = TRANSMIT_MAGIC;
                tx->media_type = media_type;
                tx->mult_count = 1;
                tx->max_loss = 0.0;
                tx->mtu = mtu;
                tx->buffer = lrand48() & 0x3fffff;
                tx->avg_len = tx->avg_len_last = tx->sent_frames = 0u;
                tx->fec_scheme = FEC_NONE;
                tx->last_frame_fragment_id = -1;
                if (fec) {
                        if(!set_fec(tx, fec)) {
                                module_done(&tx->mod);
                                return NULL;
                        }
                }
                if (encryption) {
                        tx->enc_funcs = static_cast<const struct openssl_encrypt_info *>(load_library("openssl_encrypt",
                                        LIBRARY_CLASS_UNDEFINED, OPENSSL_ENCRYPT_ABI_VERSION));
                        if (!tx->enc_funcs) {
                                fprintf(stderr, "UltraGrid was build without OpenSSL support!\n");
                                module_done(&tx->mod);
                                return NULL;
                        }
                        if (tx->enc_funcs->init(&tx->encryption,
                                                encryption, DEFAULT_CIPHER_MODE) != 0) {
                                fprintf(stderr, "Unable to initialize encryption\n");
                                module_done(&tx->mod);
                                return NULL;
                        }
                }

                tx->bitrate = bitrate;
                tx->rtpenc_h264_state = rtpenc_h264_init_state();
        }
		return tx;
}

struct tx *tx_init_h264(struct module *parent, unsigned mtu, enum tx_media_type media_type,
                const char *fec, const char *encryption, long long int bitrate)
{
  return tx_init(parent, mtu, media_type, fec, encryption, bitrate);
}

static bool set_fec(struct tx *tx, const char *fec_const)
{
        char *fec = strdup(fec_const);
        bool ret = true;

        char *fec_cfg = NULL;
        if(strchr(fec, ':')) {
                char *delim = strchr(fec, ':');
                *delim = '\0';
                fec_cfg = delim + 1;
        }

        struct msg_sender *msg = (struct msg_sender *)
                new_message(sizeof(struct msg_sender));
        msg->type = SENDER_MSG_CHANGE_FEC;

        snprintf(msg->fec_cfg, sizeof(msg->fec_cfg), "flush");

        if (strcasecmp(fec, "none") == 0) {
                tx->fec_scheme = FEC_NONE;
        } else if(strcasecmp(fec, "mult") == 0) {
                tx->fec_scheme = FEC_MULT;
                assert(fec_cfg);
                tx->mult_count = (unsigned int) atoi(fec_cfg);
                assert(tx->mult_count <= FEC_MAX_MULT);
        } else if(strcasecmp(fec, "LDGM") == 0) {
                if(tx->media_type == TX_MEDIA_AUDIO) {
                        fprintf(stderr, "LDGM is not currently supported for audio!\n");
                        ret = false;
                } else {
                        if(!fec_cfg || (strlen(fec_cfg) > 0 && strchr(fec_cfg, '%') == NULL)) {
                                snprintf(msg->fec_cfg, sizeof(msg->fec_cfg), "LDGM cfg %s",
                                                fec_cfg ? fec_cfg : "");
                        } else { // delay creation until we have avarage frame size
                                tx->max_loss = atof(fec_cfg);
                        }
                        tx->fec_scheme = FEC_LDGM;
                }
        } else if(strcasecmp(fec, "RS") == 0) {
                if(tx->media_type == TX_MEDIA_AUDIO) {
                        fprintf(stderr, "LDGM is not currently supported for audio!\n");
                        ret = false;
                } else {
                        snprintf(msg->fec_cfg, sizeof(msg->fec_cfg), "RS cfg %s",
                                        fec_cfg ? fec_cfg : "");
                        tx->fec_scheme = FEC_RS;
                }
        } else {
                fprintf(stderr, "Unknown FEC: %s\n", fec);
                ret = false;
        }

        if (ret) {
                struct response *resp = send_message_to_receiver(get_parent_module(&tx->mod),
                                (struct message *) msg);
                free_response(resp);
        } else {
                free_message((struct message *) msg, NULL);
        }

        free(fec);
        return ret;
}

static void fec_check_messages(struct tx *tx)
{
        struct message *msg;
        while ((msg = check_message(&tx->mod))) {
                struct msg_change_fec_data *data = (struct msg_change_fec_data *) msg;
                if(tx->media_type != data->media_type) {
                        fprintf(stderr, "[Transmit] FEC media type mismatch!\n");
                        free_message(msg, new_response(RESPONSE_BAD_REQUEST, NULL));
                        continue;
                }
                struct response *r;
                if (set_fec(tx, data->fec)) {
                        r = new_response(RESPONSE_OK, NULL);
                        printf("[Transmit] FEC set to new setting.\n");
                } else {
                        r = new_response(RESPONSE_INT_SERV_ERR, NULL);
                        fprintf(stderr, "[Transmit] Unable to reconfigure FEC!\n");
                }

                free_message(msg, r);
        }
}

static void tx_done(struct module *mod)
{
        struct tx *tx = (struct tx *) mod->priv_data;
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

        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tile are not currently supported for fragmented send
        fec_check_messages(tx);

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

                tx_send_base(tx, frame, rtp_session, ts, last,
                                i, fragment_offset);
        }
        tx->buffer++;
}

void format_video_header(struct video_frame *frame, int tile_idx, int buffer_idx, uint32_t *video_hdr)
{
        uint32_t tmp;

        video_hdr[3] = htonl(frame->tiles[tile_idx].width << 16 | frame->tiles[tile_idx].height);
        video_hdr[4] = get_fourcc(frame->color_spec);
        video_hdr[2] = htonl(frame->tiles[tile_idx].data_len);
        tmp = tile_idx << 22;
        tmp |= 0x3fffff & buffer_idx;
        video_hdr[0] = htonl(tmp);

        /* word 6 */
        video_hdr[5] = format_interl_fps_hdr_row(frame->interlacing, frame->fps);
}

void
tx_send_tile(struct tx *tx, struct video_frame *frame, int pos, struct rtp *rtp_session)
{
        int last = FALSE;
        uint32_t ts = 0;
        int fragment_offset = 0;

        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tile are not currently supported for fragmented send
        fec_check_messages(tx);

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
        tx_send_base(tx, frame, rtp_session, ts, last, pos,
                        fragment_offset);
        tx->buffer ++;
}

static uint32_t format_interl_fps_hdr_row(enum interlacing_t interlacing, double input_fps)
{
        unsigned int fpsd, fd, fps, fi;
        uint32_t tmp;

        tmp = interlacing << 29;
        fps = round(input_fps);
        fpsd = 1; /// @todo make use of this value (for now it is always one)
        fd = 0;
        fi = 0;
        if (input_fps > 1.0 && fabs(input_fps - round(input_fps) / 1.001) < 0.005) { // 29.97 etc.
                fd = 1;
        } else if (fps < 1.0) {
                fps = round(1.0 / input_fps);
                fi = 1;
        }

        tmp |= fps << 19;
        tmp |= fpsd << 15;
        tmp |= fd << 14;
        tmp |= fi << 13;
        return htonl(tmp);
}
static inline int get_data_len(bool with_fec, int mtu, int hdrs_len,
                int fec_symbol_size, int *fec_symbol_offset, int pf_block_size)
{
        int data_len;
        data_len = mtu - hdrs_len;
        if (with_fec) {
                if (fec_symbol_size <= mtu - hdrs_len) {
                        data_len = data_len / fec_symbol_size * fec_symbol_size;
                } else {
                        if (fec_symbol_size - *fec_symbol_offset <= mtu - hdrs_len) {
                                data_len = fec_symbol_size - *fec_symbol_offset;
                                *fec_symbol_offset = 0;
                        } else {
                                *fec_symbol_offset += data_len;
                        }
                }
        } else {
                pf_block_size = MAX(pf_block_size, 1); // compressed formats have usually 0
                data_len = (data_len / pf_block_size) * pf_block_size;
        }
        return data_len;
}

static void
tx_send_base(struct tx *tx, struct video_frame *frame, struct rtp *rtp_session,
                uint32_t ts, int send_m,
                unsigned int substream,
                int fragment_offset)
{
        struct tile *tile = &frame->tiles[substream];

        int m, data_len;
        // see definition in rtp_callback.h

        uint32_t rtp_hdr[100];
        int rtp_hdr_len;
        uint32_t tmp_hdr[100];
        uint32_t *video_hdr;
        uint32_t *fec_hdr;
        int pt;            /* A value specified in our packet format */
        char *data;
        unsigned int pos;
#ifdef HAVE_LINUX
        struct timespec start, stop;
#elif defined HAVE_MACOSX
        struct timeval start, stop;
#else // Windows
	LARGE_INTEGER start, stop, freq;
#endif
        long delta, overslept = 0;
        uint32_t tmp;
        int mult_pos[FEC_MAX_MULT];
        int mult_index = 0;
        int mult_first_sent = 0;

        int hdrs_len = (rtp_is_ipv6(rtp_session) ? 40 : 20) + 8 + 12; // IP hdr size + UDP hdr size + RTP hdr size
        unsigned int fec_symbol_size = frame->fec_params.symbol_size;

        assert(tx->magic == TRANSMIT_MAGIC);

        tx_update(tx, frame, substream);

        perf_record(UVP_SEND, ts);

        if(tx->fec_scheme == FEC_MULT) {
                int i;
                for (i = 0; i < tx->mult_count; ++i) {
                        mult_pos[i] = 0;
                }
                mult_index = 0;
        }

        m = 0;
        pos = 0;

        if (tx->encryption) {
                uint32_t *encryption_hdr;
                rtp_hdr_len = sizeof(crypto_payload_hdr_t);

                if (frame->fec_params.type != FEC_NONE) {
                        video_hdr = tmp_hdr;
                        fec_hdr = rtp_hdr;
                        encryption_hdr = rtp_hdr + sizeof(fec_video_payload_hdr_t)/sizeof(uint32_t);
                        rtp_hdr_len += sizeof(fec_video_payload_hdr_t);
                        pt = fec_pt_from_fec_type(frame->fec_params.type, true);
                        hdrs_len += (sizeof(fec_video_payload_hdr_t));
                } else {
                        video_hdr = rtp_hdr;
                        encryption_hdr = rtp_hdr + sizeof(video_payload_hdr_t)/sizeof(uint32_t);
                        rtp_hdr_len += sizeof(video_payload_hdr_t);
                        pt = PT_ENCRYPT_VIDEO;
                        hdrs_len += (sizeof(video_payload_hdr_t));
                }

                encryption_hdr[0] = htonl(DEFAULT_CIPHER_MODE << 24);
                hdrs_len += sizeof(crypto_payload_hdr_t) + tx->enc_funcs->get_overhead(tx->encryption);
        } else {
                if (frame->fec_params.type != FEC_NONE) {
                        video_hdr = tmp_hdr;
                        fec_hdr = rtp_hdr;
                        rtp_hdr_len = sizeof(fec_video_payload_hdr_t);
                        pt = fec_pt_from_fec_type(frame->fec_params.type, false);
                        hdrs_len += (sizeof(fec_video_payload_hdr_t));
                } else {
                        video_hdr = rtp_hdr;
                        rtp_hdr_len = sizeof(video_payload_hdr_t);
                        pt = PT_VIDEO;
                        hdrs_len += (sizeof(video_payload_hdr_t));
                }
        }

        if (frame->fec_params.type != FEC_NONE) {
                static bool status_printed = false;
                if (!status_printed) {
                        if (fec_symbol_size > tx->mtu - hdrs_len) {
                                fprintf(stderr, "Warning: FEC symbol size exceeds payload size! "
                                                "FEC symbol size: %d\n", fec_symbol_size);
                        } else {
                                printf("FEC symbol size: %d, symbols per packet: %d, payload size: %d\n",
                                                fec_symbol_size, (tx->mtu - hdrs_len) / fec_symbol_size,
                                                (tx->mtu - hdrs_len) / fec_symbol_size * fec_symbol_size);
                        }
                        status_printed = true;
                }
        }

        format_video_header(frame, substream, tx->buffer, video_hdr);

        if (frame->fec_params.type != FEC_NONE) {
                tmp = substream << 22;
                tmp |= 0x3fffff & tx->buffer;
                // see definition in rtp_callback.h
                fec_hdr[0] = htonl(tmp);
                fec_hdr[2] = htonl(tile->data_len);
                fec_hdr[3] = htonl(
                                frame->fec_params.k << 19 |
                                frame->fec_params.m << 6 |
                                frame->fec_params.c);
                fec_hdr[4] = htonl(frame->fec_params.seed);
        }

        int fec_symbol_offset = 0;

        // calculate number of packets
        int packet_count = 0;
        do {
                pos += get_data_len(frame->fec_params.type != FEC_NONE, tx->mtu, hdrs_len,
                                fec_symbol_size, &fec_symbol_offset,
                                get_pf_block_size(frame->color_spec));
                packet_count += 1;
        } while (pos < (unsigned int) tile->data_len);
        if(tx->fec_scheme == FEC_MULT) {
                packet_count *= tx->mult_count;
        }
        pos = 0;
        fec_symbol_offset = 0;

        long packet_rate;
        if (tx->bitrate == RATE_UNLIMITED) {
                packet_rate = 0;
        } else if (tx->bitrate == RATE_AUTO) {
                double time_for_frame = 1.0 / frame->fps / frame->tile_count;
                double interval_between_pkts = time_for_frame / tx->mult_count / packet_count;
                // use only 75% of the time
                interval_between_pkts = interval_between_pkts * 0.75;
                // prevent bitrate to be "too low", here 1 Mbps at minimum
                interval_between_pkts = std::min<double>(interval_between_pkts, tx->mtu / 1000000.0);
                packet_rate = interval_between_pkts * 1000ll * 1000 * 1000;
        } else { // bitrate given manually
                int avg_packet_size = tile->data_len / packet_count;
                packet_rate = 1000ll * 1000 * 1000 * avg_packet_size * 8 / tx->bitrate;
        }

        // initialize header array with values (except offset which is different among
        // different packts)
        void *rtp_headers = malloc(packet_count * rtp_hdr_len);
        uint32_t *rtp_hdr_packet = (uint32_t *) rtp_headers;
        for (int i = 0; i < packet_count; ++i) {
                memcpy(rtp_hdr_packet, rtp_hdr, rtp_hdr_len);
                rtp_hdr_packet += rtp_hdr_len / sizeof(uint32_t);
        }
        rtp_hdr_packet = (uint32_t *) rtp_headers;

        if (!tx->encryption) {
                rtp_async_start(rtp_session, packet_count);
        }

        do {
                GET_STARTTIME;
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[mult_index];
                }

                int offset = pos + fragment_offset;

                rtp_hdr_packet[1] = htonl(offset);

                data = tile->data + pos;
                data_len = get_data_len(frame->fec_params.type != FEC_NONE, tx->mtu, hdrs_len,
                                fec_symbol_size, &fec_symbol_offset,
                                get_pf_block_size(frame->color_spec));
                if (pos + data_len >= (unsigned int) tile->data_len) {
                        if (send_m) {
                                m = 1;
                        }
                        data_len = tile->data_len - pos;
                }
                pos += data_len;
                if(data_len) { /* check needed for FEC_MULT */
                        char encrypted_data[data_len + MAX_CRYPTO_EXCEED];

                        if (tx->encryption) {
                                data_len = tx->enc_funcs->encrypt(tx->encryption,
                                                data, data_len,
                                                (char *) rtp_hdr_packet,
                                                frame->fec_params.type != FEC_NONE ? sizeof(fec_video_payload_hdr_t) :
                                                sizeof(video_payload_hdr_t),
                                                encrypted_data);
                                data = encrypted_data;
                        }

                        rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                  (char *) rtp_hdr_packet, rtp_hdr_len,
                                  data, data_len, 0, 0, 0);
                }

                if(tx->fec_scheme == FEC_MULT) {
                        mult_pos[mult_index] = pos;
                        mult_first_sent ++;
                        if(mult_index != 0 || mult_first_sent >= (tx->mult_count - 1))
                                        mult_index = (mult_index + 1) % tx->mult_count;
                }

                /* when trippling, we need all streams goes to end */
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[tx->mult_count - 1];
                }
                rtp_hdr_packet += rtp_hdr_len / sizeof(uint32_t);

                // TRAFFIS SHAPER
                if (pos < (unsigned int) tile->data_len) { // wait for all but last packet
                        do {
                                GET_STOPTIME;
                                GET_DELTA;
                        } while (packet_rate - delta - overslept > 0);
                        overslept = -(packet_rate - delta - overslept);
                        //fprintf(stdout, "%ld ", overslept);
                }
        } while (pos < (unsigned int) tile->data_len);

        if (!tx->encryption) {
                rtp_async_wait(rtp_session);
        }
        free(rtp_headers);
}

/* 
 * This multiplication scheme relies upon the fact, that our RTP/pbuf implementation is
 * not sensitive to packet duplication. Otherwise, we can get into serious problems.
 */
void audio_tx_send(struct tx* tx, struct rtp *rtp_session, const audio_frame2 * buffer)
{
        int pt; /* PT set for audio in our packet format */
        unsigned int pos = 0u,
                     m = 0u;
        int channel;
        const char *chan_data;
        int data_len;
        const char *data;
        // see definition in rtp_callback.h
        uint32_t hdr_data[100];
        uint32_t *audio_hdr = hdr_data;
        uint32_t *crypto_hdr = audio_hdr + sizeof(audio_payload_hdr_t) / sizeof(uint32_t);
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
        int rtp_hdr_len;

        fec_check_messages(tx);

        timestamp = get_local_mediatime();
        perf_record(UVP_SEND, timestamp);

        if(tx->encryption) {
                rtp_hdr_len = sizeof(crypto_payload_hdr_t) + sizeof(audio_payload_hdr_t);
                pt = PT_ENCRYPT_AUDIO;
        } else {
                rtp_hdr_len = sizeof(audio_payload_hdr_t);
                pt = PT_AUDIO; /* PT set for audio in our packet format */
        }

        int hdrs_len = (rtp_is_ipv6(rtp_session) ? 40 : 20) + 8 + 12 + sizeof(audio_payload_hdr_t); // MTU - IP hdr - UDP hdr - RTP hdr - payload_hdr
        if(tx->encryption) {
                hdrs_len += sizeof(crypto_payload_hdr_t);
        }

        for(channel = 0; channel < buffer->get_channel_count(); ++channel)
        {
                chan_data = buffer->get_data(channel);
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
                audio_hdr[0] = htonl(tmp);

                audio_hdr[2] = htonl(buffer->get_data_len(channel));

                /* fourth word */
                tmp = (buffer->get_bps() * 8) << 26;
                tmp |= buffer->get_sample_rate();
                audio_hdr[3] = htonl(tmp);

                /* fifth word */
                audio_hdr[4] = htonl(get_audio_tag(buffer->get_codec()));

                int packet_rate;
                if (tx->bitrate > 0) {
                        //packet_rate = 1000ll * 1000 * 1000 * tx->mtu * 8 / tx->bitrate;
			packet_rate = 0;
                } else if (tx->bitrate == RATE_UNLIMITED) {
                        packet_rate = 0;
                } else if (tx->bitrate == RATE_AUTO) {
                        /**
                         * @todo
                         * Following code would actually work but seems to be useless in most of cases (eg.
                         * PCM 2 channels 2 Bps takes 5 std. Eth frames). On the other hand it could cause
                         * unexpectable problems (I'd write them here but if I'd expect them they wouldn't
                         * be unexpectable.)
                         */
#if 0
                        double time_for_frame = buffer->get_duration() / buffer->get_channel_count();
                        if (time_for_frame > 0.0) {
                                long long req_bitrate = buffer->get_data_len(channel) * 8 / time_for_frame * tx->mult_count;
                                // adjust computed value to 3
                                req_bitrate = req_bitrate * 3;
                                packet_rate = compute_packet_rate(req_bitrate, tx->mtu);
                        } else {
                                packet_rate = 0;
                        }
#endif
                        packet_rate = 0;
                } else {
                        abort();
                }

                do {
                        if(tx->fec_scheme == FEC_MULT) {
                                pos = mult_pos[mult_index];
                        }

                        data = chan_data + pos;
                        data_len = tx->mtu - hdrs_len;
                        if(pos + data_len >= (unsigned int) buffer->get_data_len(channel)) {
                                data_len = buffer->get_data_len(channel) - pos;
                                if(channel == buffer->get_channel_count() - 1)
                                        m = 1;
                        }
                        audio_hdr[1] = htonl(pos);
                        pos += data_len;
                        
                        GET_STARTTIME;
                        
                        if(data_len) { /* check needed for FEC_MULT */
                                char encrypted_data[data_len + MAX_CRYPTO_EXCEED];
                                if(tx->encryption) {
                                        crypto_hdr[0] = htonl(DEFAULT_CIPHER_MODE << 24);
                                        data_len = tx->enc_funcs->encrypt(tx->encryption,
                                                        const_cast<char *>(data), data_len,
                                                        (char *) audio_hdr, sizeof(audio_payload_hdr_t),
                                                        encrypted_data);
                                        data = encrypted_data;
                                }

                                rtp_send_data_hdr(rtp_session, timestamp, pt, m, 0,        /* contributing sources */
                                      0,        /* contributing sources length */
                                      (char *) audio_hdr, rtp_hdr_len,
                                      const_cast<char *>(data), data_len,
                                      0, 0, 0);
                        }

                        if(tx->fec_scheme == FEC_MULT) {
                                mult_pos[mult_index] = pos;
                                mult_first_sent ++;
                                if(mult_index != 0 || mult_first_sent >= (tx->mult_count - 1))
                                                mult_index = (mult_index + 1) % tx->mult_count;
                        }

                        if (pos < buffer->get_data_len(channel)) {
                                do {
                                        GET_STOPTIME;
                                        GET_DELTA;
                                        if (delta < 0)
                                                delta += 1000000000L;
                                } while (packet_rate - delta > 0);
                        }

                        /* when trippling, we need all streams goes to end */
                        if(tx->fec_scheme == FEC_MULT) {
                                pos = mult_pos[tx->mult_count - 1];
                        }

                      
                } while (pos < buffer->get_data_len(channel));
        }

        tx->buffer ++;
}

/**
 * audio_tx_send_standard - Send interleaved channels from the audio_frame2,
 *                       	as the mulaw and A-law standards (dynamic or std PT).
 */
void audio_tx_send_standard(struct tx* tx, struct rtp *rtp_session,
		const audio_frame2 * buffer) {
	//TODO to be more abstract in order to accept A-law too and other supported standards with such implementation
	assert(buffer->get_codec() == AC_MULAW || buffer->get_codec() == AC_ALAW || buffer->get_codec() == AC_OPUS);

	int pt;
	uint32_t ts;
	static uint32_t ts_prev = 0;
	struct timeval curr_time;

	// Configure the right Payload type,
	// 8000 Hz, 1 channel and 2 bps is the ITU-T G.711 standard (should be 1 bps...)
	// Other channels or Hz goes to DynRTP-Type97
	if (buffer->get_channel_count() == 1 && buffer->get_sample_rate() == 8000) {
		if (buffer->get_codec() == AC_MULAW)
			pt = PT_ITU_T_G711_PCMU;
		else if (buffer->get_codec() == AC_ALAW)
			pt = PT_ITU_T_G711_PCMA;
		else pt = PT_DynRTP_Type97;
	} else {
		pt = PT_DynRTP_Type97;
	}

	// The sizes for the different audio_frame2 channels must be the same.
	for (int i = 1; i < buffer->get_channel_count(); i++)
		assert(buffer->get_data_len(0) == buffer->get_data_len(i));

	int data_len = buffer->get_data_len(0) * buffer->get_channel_count(); 	/* Number of samples to send 			*/
	int payload_size = tx->mtu - 40 - 8 - 12; /* Max size of an RTP payload field (minus IPv6, UDP and RTP header lengths) */

        if (buffer->get_codec() == AC_OPUS) { // OPUS needs to fit one package
                if (payload_size < data_len) {
                        log_msg(LOG_LEVEL_ERROR, "Transmit: OPUS frame larger than packet! Discarding...\n");
                        return;
                }
        } else { // we may split the data into more packets, compute chunk size
                int frame_size = buffer->get_channel_count() * buffer->get_bps();
                payload_size = payload_size / frame_size * frame_size; // align to frame size
        }

	int pos = 0;
	do {
                int pkt_len = std::min(payload_size, data_len - pos);

                // interleave
                if (buffer->get_codec() == AC_OPUS) {
                        assert(buffer->get_channel_count() == 1); // we cannot interleave OPUS here
                        memcpy(tx->tmp_packet, buffer->get_data(0), pkt_len);
                } else {
                        for (int ch = 0; ch < buffer->get_channel_count(); ch++) {
                                remux_channel(tx->tmp_packet, buffer->get_data(ch) + pos / buffer->get_channel_count(), buffer->get_bps(), pkt_len / buffer->get_channel_count(), 1, buffer->get_channel_count(), 0, ch);
                        }
                }

                // Update first sample timestamp
                if (buffer->get_codec() == AC_OPUS) {
                        /* OPUS packet will be the whole contained in one packet
                         * according to RFC 7587. For PCMA/PCMU there may be more
                         * packets so we cannot use the whole frame duration. */
                        ts = get_std_audio_local_mediatime(buffer->get_duration(), 48000);
                } else {
                        ts = get_std_audio_local_mediatime((double) pkt_len / (double) buffer->get_channel_count() / (double) buffer->get_sample_rate(), buffer->get_sample_rate());
                }
                gettimeofday(&curr_time, NULL);
                rtp_send_ctrl(rtp_session, ts_prev, 0, curr_time); //send RTCP SR
                ts_prev = ts;
                // Send the packet
                rtp_send_data(rtp_session, ts, pt, 0, 0, /* contributing sources 		*/
                                0, 												/* contributing sources length 	*/
                                tx->tmp_packet, pkt_len, 0, 0, 0);
                pos += pkt_len;
	} while (pos < data_len);
}

/**
 *  H.264 standard transmission
 */
void tx_send_h264(struct tx *tx, struct video_frame *frame,
		struct rtp *rtp_session) {
        assert(frame->tile_count == 1); // std transmit doesn't handle more than one tile
        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tiles are not currently supported for fragmented send
        uint32_t ts = get_std_video_local_mediatime();
        struct tile *tile = &frame->tiles[0];

	char pt = RTPENC_H264_PT;
	unsigned char hdr[2];
	int cc = 0;
	uint32_t csrc = 0;
	int m = 0;
	char *extn = 0;
	uint16_t extn_len = 0;
	uint16_t extn_type = 0;
	unsigned nalsize = 0;
	uint8_t *data = (uint8_t *) tile->data;
	int data_len = tile->data_len;
	tx->rtpenc_h264_state->maxPacketSize = tx->mtu - 40;
	tx->rtpenc_h264_state->haveSeenEOF = false;
	tx->rtpenc_h264_state->haveSeenFirstStartCode = false;

	while ((nalsize = rtpenc_h264_frame_parse(tx->rtpenc_h264_state, data, data_len)) > 0) {

		tx->rtpenc_h264_state->curNALOffset = 0;
		tx->rtpenc_h264_state->lastNALUnitFragment = false; // by default

		while(!tx->rtpenc_h264_state->lastNALUnitFragment){
			// We have NAL unit data in the buffer.  There are three cases to consider:
			// 1. There is a new NAL unit in the buffer, and it's small enough to deliver
			//    to the RTP sink (as is).
			// 2. There is a new NAL unit in the buffer, but it's too large to deliver to
			//    the RTP sink in its entirety.  Deliver the first fragment of this data,
			//    as a FU packet, with one extra preceding header byte (for the "FU header").
			// 3. There is a NAL unit in the buffer, and we've already delivered some
			//    fragment(s) of this.  Deliver the next fragment of this data,
			//    as a FU packet, with two (H.264) extra preceding header bytes
			//    (for the "NAL header" and the "FU header").
			if (tx->rtpenc_h264_state->curNALOffset == 0) { // case 1 or 2
				if (nalsize	<= tx->rtpenc_h264_state->maxPacketSize) { // case 1

					if (tx->rtpenc_h264_state->haveSeenEOF) m = 1;
					if (rtp_send_data(rtp_session, ts, pt, m, cc, &csrc,
							(char *) tx->rtpenc_h264_state->from, nalsize,
							extn, extn_len, extn_type) < 0) {
						error_msg("There was a problem sending the RTP packet\n");
					}
					tx->rtpenc_h264_state->lastNALUnitFragment = true;
				} else { // case 2
					// We need to send the NAL unit data as FU packets.  Deliver the first
					// packet now.  Note that we add "NAL header" and "FU header" bytes to the front
					// of the packet (overwriting the existing "NAL header").
					hdr[0] = (tx->rtpenc_h264_state->firstByteOfNALUnit & 0xE0) | 28; //FU indicator
					hdr[1] = 0x80 | (tx->rtpenc_h264_state->firstByteOfNALUnit & 0x1F); // FU header (with S bit)

					if (rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
									(char *) hdr, 2,
									(char *) tx->rtpenc_h264_state->from + 1, tx->rtpenc_h264_state->maxPacketSize - 2,
									extn, extn_len, extn_type) < 0) {
										error_msg("There was a problem sending the RTP packet\n");
					}
					tx->rtpenc_h264_state->curNALOffset += tx->rtpenc_h264_state->maxPacketSize - 1;
					tx->rtpenc_h264_state->lastNALUnitFragment = false;
					nalsize -= tx->rtpenc_h264_state->maxPacketSize - 1;
				}
			} else { // case 3
				// We are sending this NAL unit data as FU packets.  We've already sent the
				// first packet (fragment).  Now, send the next fragment.  Note that we add
				// "NAL header" and "FU header" bytes to the front.  (We reuse these bytes that
				// we already sent for the first fragment, but clear the S bit, and add the E
				// bit if this is the last fragment.)
				hdr[1] = hdr[1] & ~0x80;// FU header (no S bit)

				if (nalsize + 1 > tx->rtpenc_h264_state->maxPacketSize) {
					// We can't send all of the remaining data this time:
					if (rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
							(char *) hdr, 2,
							(char *) tx->rtpenc_h264_state->from + tx->rtpenc_h264_state->curNALOffset,
							tx->rtpenc_h264_state->maxPacketSize - 2, extn, extn_len,
							extn_type) < 0) {
								error_msg("There was a problem sending the RTP packet\n");
					}
					tx->rtpenc_h264_state->curNALOffset += tx->rtpenc_h264_state->maxPacketSize - 2;
					tx->rtpenc_h264_state->lastNALUnitFragment = false;
					nalsize -= tx->rtpenc_h264_state->maxPacketSize - 2;

				} else {
					// This is the last fragment:
					if (tx->rtpenc_h264_state->haveSeenEOF) m = 1;

					hdr[1] |= 0x40;// set the E bit in the FU header

					if (rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
									(char *) hdr, 2,
									(char *) tx->rtpenc_h264_state->from + tx->rtpenc_h264_state->curNALOffset,
									nalsize, extn, extn_len, extn_type) < 0) {
										error_msg("There was a problem sending the RTP packet\n");
					}
					tx->rtpenc_h264_state->lastNALUnitFragment = true;
				}
			}
		}

		if (tx->rtpenc_h264_state->haveSeenEOF){
			return;
		}
	}
}

void tx_send_jpeg(struct tx *tx, struct video_frame *frame,
               struct rtp *rtp_session) {
        uint32_t ts = 0;

        assert(frame->tile_count == 1); // std transmit doesn't handle more than one tile
        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tiles are not currently supported for fragmented send

        ts = get_std_video_local_mediatime();

        struct tile *tile = &frame->tiles[0];
        char pt = PT_JPEG;
        struct jpeg_rtp_data d;

        if (!jpeg_get_rtp_hdr_data((uint8_t *) frame->tiles[0].data, frame->tiles[0].data_len, &d)) {
                exit_uv(1);
                return;
        }

        uint32_t jpeg_hdr[2 /* JPEG hdr */ + 1 /* RM hdr */ + 129 /* QT hdr */];
        int hdr_off = 0;
        unsigned int type_spec = 0u;
        jpeg_hdr[hdr_off++] = htonl(type_spec << 24u);
        jpeg_hdr[hdr_off++] = htonl(d.type << 24u | d.q << 16u | d.width / 8u << 8u | d.height / 8u);
        if (d.restart_interval != 0) {
                // we do not align restart interval on packet boundaries yet
                jpeg_hdr[hdr_off++] = htonl(d.restart_interval << 16u | 1u << 15u | 1u << 14u | 0x3fffu);
        }
        // quantization headers
        if (d.q == 255u) { // we must include the tables
                unsigned int mbz = 0u; // must be zero
                unsigned int precision = 0u;
                unsigned int qt_len = 2 * 64u;
                jpeg_hdr[hdr_off++] = htonl(mbz << 24u | precision << 16u | qt_len);
                memcpy(&jpeg_hdr[hdr_off], d.quantization_tables[0], 64);
                hdr_off += 64 / sizeof(uint32_t);
                memcpy(&jpeg_hdr[hdr_off], d.quantization_tables[1], 64);
                hdr_off += 64 / sizeof(uint32_t);
        }

        char *data = (char *) d.data;
        int bytes_left = tile->data_len - ((char *) d.data - tile->data);
        int max_mtu = tx->mtu - ((rtp_is_ipv6(rtp_session) ? 40 : 20) + 8 + 12); // IP hdr size + UDP hdr size + RTP hdr size

        int fragment_offset = 0;
        do {
                int hdr_len;
                if (fragment_offset == 0) { // include quantization header only in 1st pkt
                        hdr_len = hdr_off * sizeof(uint32_t);
                } else {
                        hdr_len = 8 + (d.restart_interval > 0 ? 4 : 0);
                }
                int data_len = max_mtu - hdr_len;
                int m = 0;
                if (bytes_left <= data_len) {
                        data_len = bytes_left;
                        m = 1;
                }
                jpeg_hdr[0] = htonl(type_spec << 24u | fragment_offset);

                int ret = rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                (char *) &jpeg_hdr, hdr_len,
                                data, data_len, 0, 0, 0);
                if (ret < 0) {
                        log_msg(LOG_LEVEL_ERROR, "Error sending RTP/JPEG packet!\n");
                }
                data += data_len;
                bytes_left -= data_len;
                fragment_offset += data_len;
        } while (bytes_left > 0);
}

int tx_get_buffer_id(struct tx *tx)
{
        return tx->buffer;

}

