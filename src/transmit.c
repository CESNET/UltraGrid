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
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
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
#include "host.h"
#include "perf.h"
#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h"
#include "crypto/openssl_encrypt.h"
#include "module.h"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "tv.h"
#include "transmit.h"
#include "video.h"
#include "video_codec.h"
#include "compat/platform_spin.h"

#define TRANSMIT_MAGIC	0xe80ab15f

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

#define RTPENC_H264_MAX_NALS 1024*2*2
#define RTPENC_H264_PT 96

// Mulaw audio memory reservation
#define BUFFER_MTU_SIZE 1500
static char *data_buffer_mulaw;
static int buffer_mulaw_init = 0;

struct rtp_nal_t {
    uint8_t *data;
    int size;
};

static void tx_update(struct tx *tx, struct video_frame *frame, int substream);
static void tx_done(struct module *tx);
static uint32_t format_interl_fps_hdr_row(enum interlacing_t interlacing, double input_fps);

static void
tx_send_base(struct tx *tx, struct video_frame *frame, struct rtp *rtp_session,
                uint32_t ts, int send_m,
                unsigned int substream,
                int fragment_offset);


static struct response *fec_change_callback(struct module *mod, struct message *msg);
static bool set_fec(struct tx *tx, const char *fec);

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

        platform_spin_t spin;

        struct openssl_encrypt *encryption;
        long packet_rate;
};

// Mulaw audio memory reservation
static void init_tx_mulaw_buffer() {
    if (!buffer_mulaw_init) {
        data_buffer_mulaw = malloc(BUFFER_MTU_SIZE*20);
        buffer_mulaw_init = 1;
    }
}

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
                                send_message_to_receiver(get_parent_module(&tx->mod), (struct message *) msg);
                                tx->avg_len_last = tx->avg_len;
                        }
                }
                tx->avg_len = 0;
                tx->sent_frames = 0;
        }
}

struct tx *tx_init(struct module *parent, unsigned mtu, enum tx_media_type media_type,
                const char *fec, const char *encryption, long packet_rate)
{
        struct tx *tx;

        tx = (struct tx *) calloc(1, sizeof(struct tx));
        if (tx != NULL) {
                module_init_default(&tx->mod);
                tx->mod.cls = MODULE_CLASS_TX;
                tx->mod.msg_callback = fec_change_callback;
                tx->mod.priv_data = tx;
                tx->mod.deleter = tx_done;
                module_register(&tx->mod, parent);

                tx->magic = TRANSMIT_MAGIC;
                tx->media_type = media_type;
                tx->mult_count = 0;
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
                if(encryption) {
                        if(openssl_encrypt_init(&tx->encryption,
                                                encryption, MODE_AES128_CTR) != 0) {
                                fprintf(stderr, "Unable to initialize encryption\n");
                                module_done(&tx->mod);
                                return NULL;
                        }
                }

                tx->packet_rate = packet_rate;

                platform_spin_init(&tx->spin);
        }
        return tx;
}

struct tx *tx_init_h264(struct module *parent, unsigned mtu, enum tx_media_type media_type,
                const char *fec, const char *encryption, long packet_rate)
{
  return tx_init(parent, mtu, media_type, fec, encryption, packet_rate);
}

static struct response *fec_change_callback(struct module *mod, struct message *msg)
{
        struct tx *tx = (struct tx *) mod->priv_data;

        struct msg_change_fec_data *data = (struct msg_change_fec_data *) msg;
        struct response *response;

        if(tx->media_type != data->media_type)
                return NULL;

        platform_spin_lock(&tx->spin);
        if(set_fec(tx, data->fec)) {
                response = new_response(RESPONSE_OK, NULL);
        } else {
                response = new_response(RESPONSE_BAD_REQUEST, NULL);
        }
        platform_spin_unlock(&tx->spin);

        free_message(msg);

        return response;
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

        if(strcasecmp(fec, "none") == 0) {
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
                                struct msg_sender *msg = (struct msg_sender *)
                                        new_message(sizeof(struct msg_sender));
                                snprintf(msg->fec_cfg, sizeof(msg->fec_cfg), "LDGM cfg %s",
                                                fec_cfg ? fec_cfg : "");
                                msg->type = SENDER_MSG_CHANGE_FEC;
                                send_message_to_receiver(get_parent_module(&tx->mod), (struct message *) msg);
                        } else { // delay creation until we have avarage frame size
                                tx->max_loss = atof(fec_cfg);
                        }
                        tx->fec_scheme = FEC_LDGM;
                }
        } else {
                fprintf(stderr, "Unknown FEC: %s\n", fec);
                ret = false;
        }

        free(fec);
        return ret;
}

static void tx_done(struct module *mod)
{
        struct tx *tx = (struct tx *) mod->priv_data;
        assert(tx->magic == TRANSMIT_MAGIC);
        platform_spin_destroy(&tx->spin);
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

        platform_spin_lock(&tx->spin);

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
                tx->buffer ++;
        }
        platform_spin_unlock(&tx->spin);
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
        struct tile *tile;
        int last = FALSE;
        uint32_t ts = 0;
        int fragment_offset = 0;

        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tile are not currently supported for fragmented send

        platform_spin_lock(&tx->spin);

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
        tx_send_base(tx, frame, rtp_session, ts, last, pos,
                        fragment_offset);
        tx->buffer ++;

        platform_spin_unlock(&tx->spin);
}

static uint32_t format_interl_fps_hdr_row(enum interlacing_t interlacing, double input_fps)
{
        unsigned int fpsd, fd, fps, fi;
        uint32_t tmp;

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
        return htonl(tmp);
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
        long delta;
        uint32_t tmp;
        int mult_pos[FEC_MAX_MULT];
        int mult_index = 0;
        int mult_first_sent = 0;
        int hdrs_len = 40; // for computing max payload size
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

                encryption_hdr[0] = htonl(CRYPTO_TYPE_AES128_CTR << 24);
                hdrs_len += sizeof(crypto_payload_hdr_t) + openssl_get_overhead(tx->encryption);
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

        do {
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[mult_index];
                }

                int offset = pos + fragment_offset;

                rtp_hdr[1] = htonl(offset);

                data = tile->data + pos;
                data_len = tx->mtu - hdrs_len;
                if (frame->fec_params.type != FEC_NONE) {
                        if (fec_symbol_size <= tx->mtu - hdrs_len) {
                                data_len = data_len / fec_symbol_size * fec_symbol_size;
                        } else {
                                if (fec_symbol_size - fec_symbol_offset <= tx->mtu - hdrs_len) {
                                        data_len = fec_symbol_size - fec_symbol_offset;
                                        fec_symbol_offset = 0;
                                } else {
                                        fec_symbol_offset += data_len;
                                }
                        }
                } else {
                        data_len = (data_len / 48) * 48;
                }
                if (pos + data_len >= (unsigned int) tile->data_len) {
                        if (send_m) {
                                m = 1;
                        }
                        data_len = tile->data_len - pos;
                }
                pos += data_len;
                GET_STARTTIME;
                if(data_len) { /* check needed for FEC_MULT */
                        char encrypted_data[data_len + MAX_CRYPTO_EXCEED];

                        if (tx->encryption) {
                                data_len = openssl_encrypt(tx->encryption,
                                                data, data_len,
                                                (char *) rtp_hdr,
                                                frame->fec_params.type != FEC_NONE ? sizeof(fec_video_payload_hdr_t) :
                                                sizeof(video_payload_hdr_t),
                                                encrypted_data);
                                data = encrypted_data;
                        }

                        rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                  (char *) rtp_hdr, rtp_hdr_len,
                                  data, data_len, 0, 0, 0);
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
                } while (tx->packet_rate - delta > 0);

                /* when trippling, we need all streams goes to end */
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[tx->mult_count - 1];
                }

        } while (pos < (unsigned int) tile->data_len);
}

/* 
 * This multiplication scheme relies upon the fact, that our RTP/pbuf implementation is
 * not sensitive to packet duplication. Otherwise, we can get into serious problems.
 */
void audio_tx_send(struct tx* tx, struct rtp *rtp_session, audio_frame2 * buffer)
{
        int pt; /* PT set for audio in our packet format */
        unsigned int pos = 0u,
                     m = 0u;
        int channel;
        char *chan_data;
        int data_len;
        char *data;
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

        platform_spin_lock(&tx->spin);

        timestamp = get_local_mediatime();
        perf_record(UVP_SEND, timestamp);

        if(tx->encryption) {
                rtp_hdr_len = sizeof(crypto_payload_hdr_t) + sizeof(audio_payload_hdr_t);
                pt = PT_ENCRYPT_AUDIO;
        } else {
                rtp_hdr_len = sizeof(audio_payload_hdr_t);
                pt = PT_AUDIO; /* PT set for audio in our packet format */
        }

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
                audio_hdr[0] = htonl(tmp);

                audio_hdr[2] = htonl(buffer->data_len[channel]);

                /* fourth word */
                tmp = (buffer->bps * 8) << 26;
                tmp |= buffer->sample_rate;
                audio_hdr[3] = htonl(tmp);

                /* fifth word */
                audio_hdr[4] = htonl(get_audio_tag(buffer->codec));

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
                        audio_hdr[1] = htonl(pos);
                        pos += data_len;
                        
                        GET_STARTTIME;
                        
                        if(data_len) { /* check needed for FEC_MULT */
                                char encrypted_data[data_len + MAX_CRYPTO_EXCEED];
                                if(tx->encryption) {
                                        crypto_hdr[0] = htonl(CRYPTO_TYPE_AES128_CTR << 24);
                                        data_len = openssl_encrypt(tx->encryption,
                                                        data, data_len,
                                                        (char *) audio_hdr, sizeof(audio_payload_hdr_t),
                                                        encrypted_data);
                                        data = encrypted_data;
                                }

                                rtp_send_data_hdr(rtp_session, timestamp, pt, m, 0,        /* contributing sources */
                                      0,        /* contributing sources length */
                                      (char *) audio_hdr, rtp_hdr_len,
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
                        } while (tx->packet_rate - delta > 0);

                        /* when trippling, we need all streams goes to end */
                        if(tx->fec_scheme == FEC_MULT) {
                                pos = mult_pos[tx->mult_count - 1];
                        }

                      
                } while (pos < (unsigned int) buffer->data_len[channel]);
        }

        tx->buffer ++;

        platform_spin_unlock(&tx->spin);
}

/*
 * audio_tx_send_mulaw - Send interleaved channels from the audio_frame2 at 1 bps,
 *                       as the mulaw standard.
 */
void audio_tx_send_mulaw(struct tx* tx, struct rtp *rtp_session, audio_frame2 * buffer)
{
    int pt;
    uint32_t timestamp;

    platform_spin_lock(&tx->spin);

    // Configure the right Payload type,
    // 8000 Hz, 1 channel is the ITU-T G.711 standard
    // More channels or Hz goes to DynRTP-Type97

//TODO CHECK ACTUAL CHCOUNT IN ORDER TO PROPERLY CREATE PAYLOAD TYPE
    if (buffer->ch_count == 1 && buffer->sample_rate == 8000) {
        pt = PT_ITU_T_G711_PCMU;
    } else {
        pt = PT_DynRTP_Type97;
    }

    // The sizes for the different audio_frame2 channels must be the same.
    for (int i = 1 ; i < buffer->ch_count ; i++) assert(buffer->data_len[0] == buffer->data_len[i]);

    int data_len = buffer->data_len[0] * buffer->ch_count;  /* Number of samples to send (bps=1)*/
    int data_remainig = data_len;
    int payload_size = tx->mtu - 40;                        /* Max size of an RTP payload field */
    int packets = data_len / payload_size;
    if (data_len % payload_size != 0) packets++;            /* Number of RTP packets needed */

    init_tx_mulaw_buffer();
    char *curr_sample = data_buffer_mulaw;

    // For each interval that fits in an RTP payload field.
    for (int p = 0 ; p < packets ; p++) {

        int samples_per_packet;
        int data_to_send;
        if (data_remainig >= payload_size) {
            samples_per_packet = payload_size / buffer->ch_count;
            data_to_send = payload_size;
        }
        else {
            samples_per_packet = data_remainig / buffer->ch_count;
            data_to_send = data_remainig;
        }

        // Interleave the samples
        for (int ch_sample = 0 ; ch_sample < samples_per_packet ; ch_sample++){
            for (int ch = 0 ; ch < buffer->ch_count ; ch++) {
 //TODO to be checked prepiously -> if(buffer->data[ch]!=NULL){
                    memcpy(curr_sample, (char *)(buffer->data[ch] + ch_sample), sizeof(uint8_t));
                    curr_sample += sizeof(uint8_t);
                    data_remainig--;
 //               }
            }
        }

        // Update first sample timestamp
        timestamp = get_std_audio_local_mediatime((buffer->data_len[0] - (data_remainig/buffer->ch_count)));

        // Send the packet
        rtp_send_data(rtp_session, timestamp, pt, 0, 0,        /* contributing sources */
                0,        /* contributing sources length */
                data_buffer_mulaw, data_to_send,
                0, 0, 0);
    }

    tx->buffer ++;

    platform_spin_unlock(&tx->spin);
}

static uint8_t *rtpenc_h264_find_startcode_internal(uint8_t *start,
        uint8_t *end);
uint8_t *rtpenc_h264_find_startcode(uint8_t *p, uint8_t *end);
int rtpenc_h264_parse_nal_units(uint8_t *buf_in, int size,
        struct rtp_nal_t *nals, int *nnals);

static uint8_t *rtpenc_h264_find_startcode_internal(uint8_t *start,
        uint8_t *end)
{
    //uint8_t *p = start;
    //uint8_t *pend = end; // - 3; // XXX: w/o -3, p[1] and p[2] may fail.

    for (uint8_t *p = start; p < end; p++) {
        if (p[0] == 0 && p[1] == 0 && p[2] == 1) {
            return p;
        }
    }

    return (uint8_t *) NULL;
}

uint8_t *rtpenc_h264_find_startcode(uint8_t *p, uint8_t *end)
{
    uint8_t *out = rtpenc_h264_find_startcode_internal(p, end);
    if (out != NULL) {
        if (p < out && out < end && !out[-1]) {
            out--;
        }
    } else {
        debug_msg("No NAL start code found\n"); // It's not an error per se.
    }
    return out;
}

int rtpenc_h264_parse_nal_units(uint8_t *buf_in, int size,
                                struct rtp_nal_t *nals, int *nnals)
{
    uint8_t *p = buf_in;
    uint8_t *end = p + size;
    uint8_t *nal_start;
    uint8_t *nal_end = NULL;

    size = 0;
    *nnals = 0;
    // TODO: control error
    nal_start = rtpenc_h264_find_startcode(p, end);
    for (;;) {
        if (nal_start == end || nal_start == NULL) {
            break;
        }

        nal_end = rtpenc_h264_find_startcode(nal_start + 3, end);
        if (nal_end == NULL) {
            nal_end = end;
        }
        int nal_size = nal_end - nal_start;

        if(nal_size > 4){
            size += nal_size;
            nals[(*nnals)].data = nal_start;
            nals[(*nnals)].size = nal_size;
            (*nnals)++;

            nal_start = nal_end;
    	}else nal_start += 3;
    }
    return size;
}

static void tx_send_base_h264(struct tx *tx, struct tile *tile, struct rtp *rtp_session, uint32_t ts,
        int send_m, codec_t color_spec, double input_fps,
        enum interlacing_t interlacing, unsigned int substream,
        int fragment_offset)
{

    UNUSED(color_spec);
    UNUSED(input_fps);
    UNUSED(interlacing);
    UNUSED(substream);
    UNUSED(fragment_offset);

    assert(tx->magic == TRANSMIT_MAGIC);
        //tx_update(tx, tile);

    uint8_t *data = (uint8_t *) tile->data;
    int data_len = tile->data_len;

    struct rtp_nal_t nals[RTPENC_H264_MAX_NALS];
    int nnals = 0;
    rtpenc_h264_parse_nal_units(data, data_len, nals, &nnals);

    debug_msg("%d NAL units found in buffer\n", nnals);

    char pt = RTPENC_H264_PT;
    int cc = 0;
    uint32_t csrc = 0;

    char *extn = 0;
    uint16_t extn_len = 0;
    uint16_t extn_type = 0;

    int i;
    for (i = 0; i < nnals; i++) {
        struct rtp_nal_t nal = nals[i];

        int fragmentation = 0;
        int nal_max_size = tx->mtu - 40;

        if (nal.size > nal_max_size) {
            debug_msg("RTP packet size exceeds the MTU size\n");
            fragmentation = 1;
        }

        uint8_t *nal_header = nal.data;

        // skip startcode
        int startcode_size = 0;
        uint8_t *p = nal_header;
        while ((*(p++)) == (uint8_t)0) {
            startcode_size++;
        }
        startcode_size++;

        nal_header += startcode_size;
        int nal_header_size = 1;

        uint8_t *nal_payload = nal.data + nal_header_size + startcode_size; // nal.data + nal_header_size;
        int nal_payload_size = nal.size - (int)(nal_header_size + startcode_size); //nal.size - nal_header_size;

        const char type = (char) (*nal_header & 0x1f);
        const char nri = (char) ((*nal_header & 0x60) >> 5);

        debug_msg("NAL recv | %d bytes | header: %d %d %d %d %d %d %d %d | type: %d | NRI: %d\n",
                nal_header_size + nal_payload_size,
                ((*nal_header) & 0x80) >> 7, ((*nal_header) & 0x40) >> 6,
                ((*nal_header) & 0x20) >> 5, ((*nal_header) & 0x10) >> 4,
                ((*nal_header) & 0x08) >> 3, ((*nal_header) & 0x04) >> 2,
                ((*nal_header) & 0x02) >> 1, ((*nal_header) & 0x01),
                (int)(*nal_header & 0x1f), (int)((*nal_header & 0x60) >> 5));

        char info_type;
        if (type >= 1 && type <= 23) {
            info_type = 1;
        } else {
            info_type = type;
        }

        switch (info_type) {
        case 0:
        case 1:
            debug_msg("Unfragmented or reconstructed NAL type\n");
            break;
        default:
            error_msg("Non expected NAL type %d\n", (int)info_type);
            return; // TODO maybe just warn and don't fail?
            break;
        }

        int m = 0;
        if (!fragmentation) {

            if (i == nnals - 1) {
                m = send_m;
                debug_msg("NAL with M bit\n");
            }

            int err = rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
                        (char *)nal_header, nal_header_size,
                        (char *)nal_payload, nal_payload_size, extn, extn_len,
                        extn_type);

            /*unsigned char *dst = (unsigned char *)(nal.data);
            unsigned char *end = (unsigned char *)(nal.data + nal.size);
            debug_msg("\n\nFirst four bytes: %02x %02x %02x %02x\n", dst[0], dst[1], dst[2], dst[3]);
            debug_msg("Last four bytes: %02x %02x %02x %02x\n",
                    end[-4],
                    end[-3],
                    end[-2],
                    end[-1]);
            debug_msg("NAL size: %d\n\n", nal.size); // - startcode_size); */

            if (err < 0) {
                error_msg("There was a problem sending the RTP packet\n");
            }
            else {
                debug_msg("NAL sent | %d bytes | header: %d %d %d %d %d %d %d %d | type: %d | NRI: %d\n",
                        nal_payload_size + nal_header_size,
                        ((*nal_header) & 0x80) >> 7, ((*nal_header) & 0x40) >> 6,
                        ((*nal_header) & 0x20) >> 5, ((*nal_header) & 0x10) >> 4,
                        ((*nal_header) & 0x08) >> 3, ((*nal_header) & 0x04) >> 2,
                        ((*nal_header) & 0x02) >> 1, ((*nal_header) & 0x01),
                        (int)(*nal_header & 0x1f), (int)((*nal_header & 0x60) >> 5));
            }
        }
        else {
            uint8_t frag_header[2];
            int frag_header_size = 2;

            frag_header[0] = 28 | (nri << 5); // fu_indicator, new type, same nri
            frag_header[1] = type | (1 << 7);// start, initial fu_header

            uint8_t *frag_payload = nal_payload;
            int frag_payload_size = nal_max_size - frag_header_size;

            int remaining_payload_size = nal_payload_size;

            while (remaining_payload_size + 2 > nal_max_size) {

                debug_msg("NAL 1st 6 payload bytes: %x %x %x %x %x %x\n",
                        (unsigned char)frag_payload[0], (unsigned char)frag_payload[1],
                        (unsigned char)frag_payload[2], (unsigned char)frag_payload[3],
                        (unsigned char)frag_payload[4], (unsigned char)frag_payload[5]);

                int err = rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
                            (char *)frag_header, frag_header_size,
                            (char *)frag_payload, frag_payload_size, extn, extn_len,
                            extn_type);
                if (err < 0) {
                    error_msg("There was a problem sending the RTP packet\n");
                }
                else {
                    debug_msg("NAL fragment send | %d bytes | flag %d\n", frag_payload_size + frag_header_size, (int)((frag_header[1] & 0xE0) >> 5)); //flag: 0 -> O , 2 -> E , 4 -> S , other -> !
                }

                remaining_payload_size -= frag_payload_size;
                frag_payload += frag_payload_size;

                frag_header[1] = type;
            }

            if (i == nnals - 1) {
                m = send_m;
                debug_msg("NAL fragment (E) with M bit\n");
            }

            frag_header[1] = type | (1 << 6); // end

            debug_msg("NAL 1st 6 payload bytes: %x %x %x %x %x %x\n",
                                    (unsigned char)frag_payload[0], (unsigned char)frag_payload[1],
                                    (unsigned char)frag_payload[2], (unsigned char)frag_payload[3],
                                    (unsigned char)frag_payload[4], (unsigned char)frag_payload[5]);

            int err = rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
                    (char *)frag_header, frag_header_size,
                    (char *)frag_payload, remaining_payload_size, extn, extn_len,
                    extn_type);
            if (err < 0) {
                error_msg("There was a problem sending the RTP packet\n");
            }
            else {
                debug_msg("NAL fragment send | %d bytes | flag %d\n", remaining_payload_size + frag_header_size, (int)((frag_header[1] & 0xE0) >> 5)); //flag: 0 -> O , 2 -> E , 4 -> S , other -> !
            }
        }
    }
}

/*
 * sends one or more frames (tiles) with same TS in one RTP stream. Only one m-bit is set.
 */
void
tx_send_h264(struct tx *tx, struct video_frame *frame, struct rtp *rtp_session)
{
        unsigned int i;
        uint32_t ts = 0;

        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tile are not currently supported for fragmented send

        platform_spin_lock(&tx->spin);

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

                tx_send_base_h264(tx, vf_get_tile(frame, i), rtp_session, ts, last,
                                frame->color_spec, frame->fps, frame->interlacing,
                                i, fragment_offset);
                tx->buffer ++;
        }
        platform_spin_unlock(&tx->spin);
}
