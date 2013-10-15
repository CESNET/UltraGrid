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
#include "rtp/ldgm.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "tv.h"
#include "transmit.h"
#include "video.h"
#include "video_codec.h"
#include "compat/platform_spin.h"

#define TRANSMIT_MAGIC	0xe80ab15f

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


#define RTPENC_H264_MAX_NALS 1024*2*2
#define RTPENC_H264_PT 96

struct rtp_nal_t {
    uint8_t *data;
    int size;
};

int rtpenc_h264_nals_recv;
int rtpenc_h264_nals_sent_nofrag;
int rtpenc_h264_nals_sent_frag;
int rtpenc_h264_nals_sent;


static bool fec_is_ldgm(struct tx *tx);
static void tx_update(struct tx *tx, struct tile *tile);
static void tx_done(struct module *tx);

static void
tx_send_base(struct tx *tx, struct tile *tile, struct rtp *rtp_session,
                uint32_t ts, int send_m,
                codec_t color_spec, double input_fps,
                enum interlacing_t interlacing, unsigned int substream,
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

        enum fec_scheme_t fec_scheme;
        void *fec_state;
        int mult_count;

        int last_fragment;

        platform_spin_t spin;

        struct openssl_encrypt *encryption;
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

struct tx *tx_init(struct module *parent, unsigned mtu, enum tx_media_type media_type,
                char *fec, const char *encryption)
{
        struct tx *tx;

        tx = (struct tx *) calloc(1, sizeof(struct tx));
        if (tx != NULL) {
                module_init_default(&tx->mod);
                tx->mod.cls = MODULE_CLASS_TX;
                tx->mod.msg_callback = fec_change_callback;
                tx->mod.priv_data = tx;
                tx->mod.deleter = tx_done;

                tx->magic = TRANSMIT_MAGIC;
                tx->media_type = media_type;
                tx->mult_count = 0;
                tx->max_loss = 0.0;
                tx->fec_state = NULL;
                tx->mtu = mtu;
                tx->buffer = lrand48() & 0x3fffff;
                tx->avg_len = tx->avg_len_last = tx->sent_frames = 0u;
                tx->fec_scheme = FEC_NONE;
                tx->last_frame_fragment_id = -1;
                if (fec) {
                        if(!set_fec(tx, fec)) {
                                free(tx);
                                return NULL;
                        }
                }
                if(encryption) {
                        if(openssl_encrypt_init(&tx->encryption,
                                                encryption, MODE_AES128_CTR) != 0) {
                                fprintf(stderr, "Unable to initialize encryption\n");
                                return NULL;
                        }
                }

                platform_spin_init(&tx->spin);

                module_register(&tx->mod, parent);
        }
        return tx;
}

struct tx *tx_init_h264(struct module *parent, unsigned mtu, enum tx_media_type media_type,
                char *fec, const char *encryption)
{
  rtpenc_h264_nals_recv = 0;
  rtpenc_h264_nals_sent_nofrag = 0;
  rtpenc_h264_nals_sent_frag = 0;
  rtpenc_h264_nals_sent = 0;
  return tx_init(parent, mtu, media_type, fec, encryption);
}

static struct response *fec_change_callback(struct module *mod, struct message *msg)
{
        struct tx *tx = (struct tx *) mod->priv_data;

        struct msg_change_fec_data *data = (struct msg_change_fec_data *) msg;
        struct response *response;

        if(tx->media_type != data->media_type)
                return NULL;

        platform_spin_lock(&tx->spin);
        void *old_fec_state = tx->fec_state;
        tx->fec_state = NULL;
        if(set_fec(tx, data->fec)) {
                ldgm_encoder_destroy(old_fec_state);
                response = new_response(RESPONSE_OK, NULL);
        } else {
                tx->fec_state = old_fec_state;
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
                                tx->fec_state = ldgm_encoder_init_with_cfg(fec_cfg);
                                if(tx->fec_state == NULL) {
                                        fprintf(stderr, "Unable to initialize LDGM.\n");
                                        ret = false;
                                }
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
        ldgm_encoder_destroy(tx->fec_state);
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

                tx_send_base(tx, vf_get_tile(frame, i), rtp_session, ts, last,
                                frame->color_spec, frame->fps, frame->interlacing,
                                i, fragment_offset);
                tx->buffer ++;
        }
        platform_spin_unlock(&tx->spin);
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
        tx_send_base(tx, tile, rtp_session, ts, last, frame->color_spec, frame->fps, frame->interlacing, pos,
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
tx_send_base(struct tx *tx, struct tile *tile, struct rtp *rtp_session,
                uint32_t ts, int send_m,
                codec_t color_spec, double fps,
                enum interlacing_t interlacing, unsigned int substream,
                int fragment_offset)
{
        int m, data_len;
        // see definition in rtp_callback.h

        uint32_t hdr_data[100];
        uint32_t *ldgm_payload_hdr = hdr_data;
        uint32_t *video_hdr = ldgm_payload_hdr + 1;
        uint32_t *ldgm_hdr = video_hdr + sizeof(video_payload_hdr_t)/sizeof(uint32_t);
        uint32_t *encryption_hdr;
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
        int mult_pos[FEC_MAX_MULT];
        int mult_index = 0;
        int mult_first_sent = 0;
        int hdrs_len = 40 + (sizeof(video_payload_hdr_t)); // for computing max payload size
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

        char *rtp_hdr;
        int rtp_hdr_len;

        if(tx->encryption && !fec_is_ldgm(tx)) {
                /*
                 * Important
                 * Crypto and video header must be in specified order placed one right after
                 * the another since both will be sent as a RTP header.
                 */
                encryption_hdr = video_hdr + sizeof(video_payload_hdr_t)/sizeof(uint32_t);

                encryption_hdr[0] = htonl(CRYPTO_TYPE_AES128_CTR << 24);
                hdrs_len += sizeof(crypto_payload_hdr_t) + openssl_get_overhead(tx->encryption);
                rtp_hdr = (char *) video_hdr;
                rtp_hdr_len = sizeof(crypto_payload_hdr_t) + sizeof(video_payload_hdr_t);
                pt = PT_ENCRYPT_VIDEO;
        }

        video_hdr[3] = htonl(tile->width << 16 | tile->height);
        video_hdr[4] = get_fourcc(color_spec);
        video_hdr[2] = htonl(data_to_send_len);
        tmp = substream << 22;
        tmp |= 0x3fffff & tx->buffer;
        video_hdr[0] = htonl(tmp);

        /* word 6 */
        video_hdr[5] = format_interl_fps_hdr_row(interlacing, fps);

        if(fec_is_ldgm(tx)) {
                hdrs_len = 40 + (sizeof(ldgm_video_payload_hdr_t));
                char *tmp_data = NULL;
                char *ldgm_input_data;
                int ldgm_input_len;
                int ldgm_payload_hdr_len = sizeof(ldgm_payload_hdr_t) + sizeof(video_payload_hdr_t);
                if(tx->encryption) {
                        ldgm_input_len = tile->data_len + sizeof(crypto_payload_hdr_t) +
                                MAX_CRYPTO_EXCEED;
                        ldgm_input_data = tmp_data = malloc(ldgm_input_len);
                        char *ciphertext = tmp_data + sizeof(crypto_payload_hdr_t);
                        encryption_hdr = (uint32_t *)(void *) tmp_data;
                        encryption_hdr[0] = htonl(CRYPTO_TYPE_AES128_CTR << 24);
                        ldgm_payload_hdr[0] = ntohl(PT_ENCRYPT_VIDEO);
                        int ret = openssl_encrypt(tx->encryption,
                                        tile->data, tile->data_len,
                                        (char *) ldgm_payload_hdr, ldgm_payload_hdr_len,
                                        ciphertext);
                        ldgm_input_len = sizeof(crypto_payload_hdr_t) + ret;

                } else {
                        ldgm_input_data = tile->data;
                        ldgm_input_len = tile->data_len;
                        ldgm_payload_hdr[0] = ntohl(PT_VIDEO);
                }
                ldgm_encoder_encode(tx->fec_state, (char *) ldgm_payload_hdr,
                                ldgm_payload_hdr_len,
                                ldgm_input_data, ldgm_input_len, &data_to_send, &data_to_send_len);
                free(tmp_data);
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

                rtp_hdr = (char *) ldgm_hdr;
                rtp_hdr_len = sizeof(ldgm_video_payload_hdr_t);
        } else if(!tx->encryption) {
                rtp_hdr = (char *) video_hdr;
                rtp_hdr_len = sizeof(video_payload_hdr_t);
        }

        uint32_t *hdr_offset; // data offset pointer - contains field that needs to be updated
                              // every cycle
        if(fec_is_ldgm(tx)) {
                hdr_offset = ldgm_hdr + 1;
        } else {
                hdr_offset = video_hdr + 1;
        }

        do {
                if(tx->fec_scheme == FEC_MULT) {
                        pos = mult_pos[mult_index];
                }

                int offset = pos + fragment_offset;

                *hdr_offset = htonl(offset);

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
                        char encrypted_data[data_len + MAX_CRYPTO_EXCEED];

                        if(tx->encryption && tx->fec_scheme != FEC_LDGM) {
                                data_len = openssl_encrypt(tx->encryption,
                                                data, data_len,
                                                (char *) video_hdr, sizeof(video_payload_hdr_t),
                                                encrypted_data);
                                data = encrypted_data;
                        }

                        rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                  rtp_hdr, rtp_hdr_len,
                                  data, data_len, 0, 0, 0);
                        if(m && tx->fec_scheme != FEC_NONE) {
                                int i;
                                for(i = 0; i < 5; ++i) {
                                        rtp_send_data_hdr(rtp_session, ts, pt, m, 0, 0,
                                                  rtp_hdr, rtp_hdr_len,
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
                        } while (packet_rate - delta > 0);

                        /* when trippling, we need all streams goes to end */
                        if(tx->fec_scheme == FEC_MULT) {
                                pos = mult_pos[tx->mult_count - 1];
                        }

                      
                } while (pos < (unsigned int) buffer->data_len[channel]);
        }

        tx->buffer ++;

        platform_spin_unlock(&tx->spin);
}


void rtpenc_h264_stats_print()
{
    printf("[RTPENC][STATS] Total recv NALs: %d\n", rtpenc_h264_nals_recv);
    printf("[RTPENC][STATS] Unfragmented sent NALs: %d\n",
            rtpenc_h264_nals_sent_nofrag);
    printf("[RTPENC][STATS] Fragmented sent NALs: %d\n",
            rtpenc_h264_nals_sent_frag);
    printf("[RTPENC][STATS] NAL fragments sent: %d\n",
            rtpenc_h264_nals_sent - rtpenc_h264_nals_sent_nofrag);
    printf("[RTPENC][STATS] Total sent NALs: %d\n", rtpenc_h264_nals_sent);
}

static uint8_t *rtpenc_h264_find_startcode_internal(uint8_t *start,
        uint8_t *end);
uint8_t *rtpenc_h264_find_startcode(uint8_t *p, uint8_t *end);
int rtpenc_h264_parse_nal_units(uint8_t *buf_in, int size,
        struct rtp_nal_t *nals, int *nnals);

static void rtpenc_h264_debug_print_nal_recv_info(uint8_t *header, int size);
static void rtpenc_h264_debug_print_nal_sent_info(uint8_t *header, int size);
static void rtpenc_h264_debug_print_fragment_sent_info(uint8_t *header, int size);
static void rtpenc_h264_debug_print_payload_bytes(uint8_t *payload);

static void rtpenc_h264_debug_print_payload_bytes(uint8_t *payload)
{
    #ifdef DEBUG
    debug_msg("NAL 1st 6 payload bytes: %x %x %x %x %x %x\n",
            (unsigned char)payload[0], (unsigned char)payload[1],
            (unsigned char)payload[2], (unsigned char)payload[3],
            (unsigned char)payload[4], (unsigned char)payload[5]);
    #else
    UNUSED(payload);
    #endif
}

static void rtpenc_h264_debug_print_nal_recv_info(uint8_t *header, int size)
{
    #ifdef DEBUG
    int type = (int)(*header & 0x1f);
    int nri = (int)((*header & 0x60) >> 5);
    debug_msg("NAL recv | %d bytes | header: %d %d %d %d %d %d %d %d | NRI: %d | type: %d\n",
            size,
            ((*header) & 0x80) >> 7, ((*header) & 0x40) >> 6,
            ((*header) & 0x20) >> 5, ((*header) & 0x10) >> 4,
            ((*header) & 0x08) >> 3, ((*header) & 0x04) >> 2,
            ((*header) & 0x02) >> 1, ((*header) & 0x01),
            nri, type);
    #else
    UNUSED(header);
    UNUSED(size);
    #endif
}

static void rtpenc_h264_debug_print_nal_sent_info(uint8_t *header, int size)
{
    #ifdef DEBUG
    int type = (int)(*header & 0x1f);
    int nri = (int)((*header & 0x60) >> 5);
    debug_msg("NAL sent | %d bytes | header: %d %d %d %d %d %d %d %d | NRI: %d | type: %d\n",
            size,
            ((*header) & 0x80) >> 7, ((*header) & 0x40) >> 6,
            ((*header) & 0x20) >> 5, ((*header) & 0x10) >> 4,
            ((*header) & 0x08) >> 3, ((*header) & 0x04) >> 2,
            ((*header) & 0x02) >> 1, ((*header) & 0x01),
            type, nri);
    #else
    UNUSED(header);
    UNUSED(size);
    #endif
}

static void rtpenc_h264_debug_print_fragment_sent_info(uint8_t *header, int size)
{
    #ifdef DEBUG
    char frag_class;
    switch((header[1] & 0xE0) >> 5) {
        case 0:
            frag_class = '0';
            break;
        case 2:
            frag_class = 'E';
            break;
        case 4:
            frag_class = 'S';
            break;
        default:
            frag_class = '!';
            break;
    }
    debug_msg("NAL fragment send | %d bytes | flag %c\n", size, frag_class);
    #else
    UNUSED(header);
    UNUSED(size);
    #endif
}

static uint8_t *rtpenc_h264_find_startcode_internal(uint8_t *start,
        uint8_t *end)
{
    uint8_t *p = start;
    uint8_t *pend = end; // - 3; // XXX: w/o -3, p[1] and p[2] may fail.

    for (p = start; p < pend; p++) {
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

        size += nal_size;

        nals[(*nnals)].data = nal_start;
        nals[(*nnals)].size = nal_size;
        (*nnals)++;

        nal_start = nal_end;
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
        tx_update(tx, tile);

    uint8_t *data = (uint8_t *) tile->data;
    int data_len = tile->data_len;

    struct rtp_nal_t nals[RTPENC_H264_MAX_NALS];
    int nnals = 0;
    rtpenc_h264_parse_nal_units(data, data_len, nals, &nnals);

    rtpenc_h264_nals_recv += nnals;
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

        rtpenc_h264_debug_print_nal_recv_info(nal_header, nal_header_size + nal_payload_size);

        const char type = (char) (*nal_header & 0x1f);
        const char nri = (char) ((*nal_header & 0x60) >> 5);

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
                rtpenc_h264_nals_sent_nofrag++;
                rtpenc_h264_nals_sent++;
                rtpenc_h264_debug_print_nal_sent_info(nal_header, nal_payload_size + nal_header_size);
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

                rtpenc_h264_debug_print_payload_bytes(frag_payload);

                int err = rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
                            (char *)frag_header, frag_header_size,
                            (char *)frag_payload, frag_payload_size, extn, extn_len,
                            extn_type);
                if (err < 0) {
                    error_msg("There was a problem sending the RTP packet\n");
                }
                else {
                    rtpenc_h264_nals_sent++;
                    rtpenc_h264_debug_print_fragment_sent_info(frag_header, frag_payload_size + frag_header_size);
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

            rtpenc_h264_debug_print_payload_bytes(frag_payload);

            int err = rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
                    (char *)frag_header, frag_header_size,
                    (char *)frag_payload, remaining_payload_size, extn, extn_len,
                    extn_type);
            if (err < 0) {
                error_msg("There was a problem sending the RTP packet\n");
            }
            else {
                rtpenc_h264_nals_sent_frag++; // Each fragmented NAL has one E (end) NAL fragment
                rtpenc_h264_nals_sent++;
                rtpenc_h264_debug_print_fragment_sent_info(frag_header, remaining_payload_size + frag_header_size);
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
