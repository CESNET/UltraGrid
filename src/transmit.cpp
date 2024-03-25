/*
 * FILE:    transmit.cpp
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
 * Copyright (c) 2005-2023 CESNET z.s.p.o.
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

#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <vector>

#include "audio/codec.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "control_socket.h"
#include "crypto/openssl_encrypt.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "module.h"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/rtpenc_h264.h"
#include "transmit.h"
#include "tv.h"
#include "utils/jpeg_reader.h"
#include "utils/misc.h" // unit_evaluate
#include "utils/random.h"
#include "video.h"
#include "video_codec.h"

#define MOD_NAME "[transmit] "
#define TRANSMIT_MAGIC	0xe80ab15f

#define FEC_MAX_MULT 10

#define CONTROL_PORT_BANDWIDTH_REPORT_INTERVAL_NS NS_IN_SEC

#ifdef __APPLE__
#define GET_STARTTIME gettimeofday(&start, NULL)
#define GET_STOPTIME gettimeofday(&stop, NULL)
#define GET_DELTA delta = (stop.tv_sec - start.tv_sec) * 1000000000l + (stop.tv_usec - start.tv_usec) * 1000L
#elif defined __linux__
#define GET_STARTTIME clock_gettime(CLOCK_REALTIME, &start)
#define GET_STOPTIME clock_gettime(CLOCK_REALTIME, &stop)
#define GET_DELTA delta = (stop.tv_sec - start.tv_sec) * 1000000000l + stop.tv_nsec - start.tv_nsec
#else // Windows
#define GET_STARTTIME {QueryPerformanceFrequency(&freq); QueryPerformanceCounter(&start); }
#define GET_STOPTIME { QueryPerformanceCounter(&stop); }
#define GET_DELTA delta = (long)((double)(stop.QuadPart - start.QuadPart) * 1000 * 1000 * 1000 / freq.QuadPart);
#endif

#define DEFAULT_CIPHER_MODE MODE_AES128_GCM

using std::array;
using std::vector;

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

struct rate_limit_dyn {
        unsigned long avg_frame_size;   ///< moving average
        long long last_excess; ///< nr of frames last excessive frame was emitted
        static constexpr int EXCESS_GAP = 4; ///< minimal gap between excessive frames
};

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

        struct control_state *control = nullptr;
        size_t sent_since_report = 0;
        uint64_t last_stat_report = 0;

        const struct openssl_encrypt_info *enc_funcs;
        struct openssl_encrypt *encryption;
        long long int bitrate;
        struct rate_limit_dyn dyn_rate_limit_state;
		
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
                                int data_len = tx->mtu -  (40 + (sizeof(fec_payload_hdr_t)));
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
        if (mtu > RTP_MAX_MTU) {
                log_msg(LOG_LEVEL_ERROR, "Requested MTU exceeds maximal value allowed by RTP library (%d B).\n", RTP_MAX_MTU);
                return NULL;
        }

        if (bitrate < RATE_MIN) {
                log_msg(LOG_LEVEL_ERROR, "Invalid bitrate value %lld passed (either positive bitrate or magic values from %d supported)!\n", bitrate, RATE_MIN);
                return NULL;
        }

        struct tx *tx = (struct tx *) calloc(1, sizeof(struct tx));
        if (tx == nullptr) {
                return tx;
        }
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
        tx->buffer = ug_rand() & 0x3fffff;
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
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to initialize encryption\n");
                        module_done(&tx->mod);
                        return NULL;
                }
        }

        tx->bitrate = bitrate;

        if(parent)
                tx->control = (struct control_state *) get_module(get_root_module(parent), "control");

        return tx;
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

        tx->mult_count = 1; // default
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
                snprintf(msg->fec_cfg, sizeof(msg->fec_cfg), "RS cfg %s",
                                fec_cfg ? fec_cfg : "");
                tx->fec_scheme = FEC_RS;
        } else if(strcasecmp(fec, "help") == 0) {
                color_printf("Usage:\n");
                color_printf("\t" TBOLD("-f [A:|V:]{mult:count|ldgm[:params]|"
                             "rs[:params]}") "\n");
                color_printf("\nIf neither A: or V: is speciefied, FEC is set "
                             "to the video (backward compat).\n\n");
                ret = false;
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
                auto *data = reinterpret_cast<struct msg_universal *>(msg);
                const char *text = data->text;
                if (strstr(text, MSG_UNIVERSAL_TAG_TX) != text) {
                        LOG(LOG_LEVEL_ERROR) << "[Transmit] Unexpected TX message: " << text << "\n";
                        free_message(msg, new_response(RESPONSE_BAD_REQUEST, "Unexpected message"));
                        continue;
                }
                text += strlen(MSG_UNIVERSAL_TAG_TX);
                struct response *r = nullptr;
                if (strstr(text, "fec ") == text) {
                        text += strlen("fec ");
                        if (set_fec(tx, text)) {
                                r = new_response(RESPONSE_OK, nullptr);
                                LOG(LOG_LEVEL_NOTICE) << "[Transmit] FEC set to new setting: " << text << "\n";
                        } else {
                                r = new_response(RESPONSE_INT_SERV_ERR, "cannot set FEC");
                                LOG(LOG_LEVEL_ERROR) << "[Transmit] Unable to reconfiure FEC to: " << text << "\n";
                        }
                } else if (strstr(text, "rate ") == text) {
                        text += strlen("rate ");
                        auto new_rate = unit_evaluate(text, nullptr);
                        if (new_rate >= RATE_MIN) {
                                tx->bitrate = new_rate;
                                r = new_response(RESPONSE_OK, nullptr);
                                LOG(LOG_LEVEL_NOTICE) << "[Transmit] Bitrate set to: " << text << (new_rate > 0 ? "B" : "") << "\n";
                        } else {
                                r = new_response(RESPONSE_BAD_REQUEST, "Wrong value for bitrate");
                                LOG(LOG_LEVEL_ERROR) << "[Transmit] Wrong bitrate: " << text << "\n";
                        }
                } else {
                        r = new_response(RESPONSE_BAD_REQUEST, "Unknown TX message");
                        LOG(LOG_LEVEL_ERROR) << "[Transmit] Unknown TX message: " << text << "\n";
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

        assert(!frame->fragment || tx->fec_scheme == FEC_NONE); // currently no support for FEC with fragments
        assert(!frame->fragment || frame->tile_count); // multiple tile are not currently supported for fragmented send
        fec_check_messages(tx);

        uint32_t ts =
            (frame->flags & TIMESTAMP_VALID) == 0
                ? get_local_mediatime()
                : get_local_mediatime_offset() + frame->timestamp;
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

void format_audio_header(const audio_frame2 *frame, int channel, int buffer_idx, uint32_t *audio_hdr)
{
        uint32_t tmp = 0;
        tmp = channel << 22U; /* bits 0-9 */
        tmp |= buffer_idx; /* bits 10-31 */
        audio_hdr[0] = htonl(tmp);

        audio_hdr[2] = htonl(frame->get_data_len(channel));

        /* fourth word */
        tmp = (frame->get_bps() * 8) << 26U;
        tmp |= frame->get_sample_rate();
        audio_hdr[3] = htonl(tmp);

        /* fifth word */
        audio_hdr[4] = htonl(get_audio_tag(frame->get_codec()));
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

static inline void check_symbol_size(int fec_symbol_size, int payload_len)
{
        thread_local static bool status_printed = false;

        if (status_printed && log_level < LOG_LEVEL_DEBUG2) {
                return;
        }

        if (fec_symbol_size > payload_len) {
                LOG(LOG_LEVEL_WARNING) << MOD_NAME
                    "Warning: FEC symbol size exceeds payload size! "
                    "FEC symbol size: " << fec_symbol_size
                                       << "\n";
        } else {
                const int ll =
                    status_printed ?  LOG_LEVEL_DEBUG2 : LOG_LEVEL_INFO;
                LOG(ll) << MOD_NAME "FEC symbol size: " << fec_symbol_size
                        << ", symbols per packet: "
                        << payload_len / fec_symbol_size << ", payload size: "
                        << payload_len / fec_symbol_size * fec_symbol_size
                        << "\n";
        }
        status_printed = true;
}

/**
 * Splits symbol (FEC symbol or uncompressed line) to 1 or more MTUs. Symbol starts
 * always on beginning of packet.
 *
 * If symbol_size is longer than MTU (more symbols fit one packet), the aligned
 * packet size is always the same.
 *
 * @param symbol_size  FEC symbol size or linesize for uncompressed
 */
static inline int get_video_pkt_len(int mtu,
                int symbol_size, int *symbol_offset)
{
        if (symbol_size > mtu) {
                if (symbol_size - *symbol_offset <= mtu) {
                        mtu = symbol_size - *symbol_offset;
                        *symbol_offset = 0;
                } else {
                        *symbol_offset += mtu;
                }
                return mtu;
        }
        return mtu / symbol_size * symbol_size;
}

/// @param mtu is tx->mtu - hdrs_len
static vector<int> get_packet_sizes(struct video_frame *frame, int substream, int mtu) {
        if (frame->fec_params.type != FEC_NONE) {
                check_symbol_size(frame->fec_params.symbol_size, mtu);
        }

        unsigned int symbol_size = 1;
        int symbol_offset = 0;
        if (frame->fec_params.type == FEC_NONE && !is_codec_opaque(frame->color_spec)) {
                symbol_size = vc_get_linesize(frame->tiles[substream].width, frame->color_spec);
                int pf_block_size = PIX_BLOCK_LCM / get_pf_block_pixels(frame->color_spec) * get_pf_block_bytes(frame->color_spec);
                assert(pf_block_size <= mtu);
                mtu = mtu / pf_block_size * pf_block_size;
        } else if (frame->fec_params.type != FEC_NONE) {
                symbol_size = frame->fec_params.symbol_size;
        }
        vector<int> ret;
        unsigned pos = 0;
        do {
                int len = symbol_size == 1
                        ? mtu
                        : get_video_pkt_len(mtu, symbol_size, &symbol_offset);
                pos += len;
                ret.push_back(len);
        } while (pos < frame->tiles[substream].data_len);

        if (pos > frame->tiles[substream].data_len) {
                ret[ret.size() - 1] -=
                    (int) (pos - frame->tiles[substream].data_len);
        }
        return ret;
}

static void
report_stats(struct tx *tx, struct rtp *rtp_session, long data_sent)
{
        if (!tx->control || !control_stats_enabled(tx->control)) {
                return;
        }

        tx->sent_since_report += data_sent;

        const time_ns_t current_time_ns = get_time_in_ns();
        if (current_time_ns - tx->last_stat_report <
            CONTROL_PORT_BANDWIDTH_REPORT_INTERVAL_NS) {
                return;
        }

        const char *media =
            tx->media_type == TX_MEDIA_VIDEO ? "video" : "audio";
        std::ostringstream oss;
        oss << "tx_send " << std::hex << rtp_my_ssrc(rtp_session) << std::dec
            << " " << media << " " << tx->sent_since_report;

        control_report_stats(tx->control, oss.str());
        tx->last_stat_report  = current_time_ns;
        tx->sent_since_report = 0;
}

/**
 * Returns inter-packet interval in nanoseconds.
 */
static long
get_packet_rate(struct tx *tx, struct video_frame *frame, int substream, long packet_count)
{
        if (tx->bitrate == RATE_UNLIMITED) {
                return 0;
        }
        double time_for_frame = 1.0 / frame->fps / frame->tile_count;
        double interval_between_pkts = time_for_frame / tx->mult_count / packet_count;
        // use only 75% of the time - we less likely overshot the frame time and
        // can minimize risk of swapping packets between 2 frames (out-of-order ones)
        interval_between_pkts = interval_between_pkts * 0.75;
        // prevent bitrate to be "too low", here 1 Mbps at minimum
        interval_between_pkts = std::min<double>(interval_between_pkts, tx->mtu / 1000'000.0);
        long packet_rate_auto = interval_between_pkts * 1000'000'000L;

        if (tx->bitrate == RATE_AUTO) { // adaptive (spread packets to 75% frame time)
               return packet_rate_auto;
        }
        if (tx->bitrate == RATE_DYNAMIC) {
                if (frame->tiles[substream].data_len > 2 * tx->dyn_rate_limit_state.avg_frame_size
                                && tx->dyn_rate_limit_state.last_excess > rate_limit_dyn::EXCESS_GAP) {
                        packet_rate_auto /= 2; // double packet rate for this frame
                        tx->dyn_rate_limit_state.last_excess = 0;
                } else {
                        tx->dyn_rate_limit_state.last_excess += 1;
                }
                tx->dyn_rate_limit_state.avg_frame_size = (9 * tx->dyn_rate_limit_state.avg_frame_size + frame->tiles[substream].data_len) / 10;
                return packet_rate_auto;
        }
        long long int bitrate = tx->bitrate & ~RATE_FLAG_FIXED_RATE;
        int avg_packet_size = frame->tiles[substream].data_len / packet_count;
        long packet_rate = 1000'000'000L * avg_packet_size * 8 / bitrate; // fixed rate
        if ((tx->bitrate & RATE_FLAG_FIXED_RATE) == 0) { // adaptive capped rate
                packet_rate = std::max(packet_rate, packet_rate_auto);
        }
        return packet_rate;
}

static int
get_tx_hdr_len(bool is_ipv6)
{
        return (is_ipv6 ? IPV6_HDR_LEN : IPV4_HDR_LEN) + UDP_HDR_LEN +
               RTP_HDR_LEN;
}

static void
tx_send_base(struct tx *tx, struct video_frame *frame, struct rtp *rtp_session,
                uint32_t ts, int send_m,
                unsigned int substream,
                int fragment_offset)
{
        assert(fragment_offset == 0); // no longer supported
        if (!rtp_has_receiver(rtp_session)) {
                return;
        }

        struct tile *tile = &frame->tiles[substream];

        // see definition in rtp_callback.h
        uint32_t rtp_hdr[100];
        int rtp_hdr_len;
        int pt = fec_pt_from_fec_type(TX_MEDIA_VIDEO, frame->fec_params.type, tx->encryption);            /* A value specified in our packet format */
#ifdef __linux__
        struct timespec start, stop;
#elif defined __APPLE__
        struct timeval start, stop;
#else // Windows
	LARGE_INTEGER start, stop, freq;
#endif
        long delta, overslept = 0;
        int hdrs_len = get_tx_hdr_len(rtp_is_ipv6(rtp_session));

        assert(tx->magic == TRANSMIT_MAGIC);

        tx_update(tx, frame, substream);

        if (frame->fec_params.type == FEC_NONE) {
                hdrs_len += (sizeof(video_payload_hdr_t));
                rtp_hdr_len = sizeof(video_payload_hdr_t);
                format_video_header(frame, substream, tx->buffer, rtp_hdr);
        } else {
                hdrs_len += (sizeof(fec_payload_hdr_t));
                rtp_hdr_len = sizeof(fec_payload_hdr_t);
                uint32_t tmp = substream << 22;
                tmp |= 0x3fffff & tx->buffer;
                // see definition in rtp_callback.h
                rtp_hdr[0] = htonl(tmp);
                rtp_hdr[2] = htonl(tile->data_len);
                rtp_hdr[3] = htonl(
                             frame->fec_params.k << 19 |
                             frame->fec_params.m << 6 |
                             frame->fec_params.c);
                rtp_hdr[4] = htonl(frame->fec_params.seed);
        }

        if (tx->encryption) {
                hdrs_len += sizeof(crypto_payload_hdr_t) + tx->enc_funcs->get_overhead(tx->encryption);
                rtp_hdr[rtp_hdr_len / sizeof(uint32_t)] = htonl(DEFAULT_CIPHER_MODE << 24);
                rtp_hdr_len += sizeof(crypto_payload_hdr_t);
        }

        vector<int> packet_sizes = get_packet_sizes(frame, substream, tx->mtu - hdrs_len);
        const long mult_pkt_cnt = (long) packet_sizes.size() * tx->mult_count;
        const long packet_rate =
            get_packet_rate(tx, frame, (int) substream, mult_pkt_cnt);

        // initialize header array with values (except offset which is different among
        // different packts)
        void *rtp_headers = malloc(mult_pkt_cnt * rtp_hdr_len);
        uint32_t *rtp_hdr_packet = (uint32_t *) rtp_headers;
        for (int m = 0; m < tx->mult_count; ++m) {
                unsigned pos = 0;
                for (unsigned i = 0; i < packet_sizes.size(); ++i) {
                        memcpy(rtp_hdr_packet, rtp_hdr, rtp_hdr_len);
                        rtp_hdr_packet[1] = htonl(pos);
                        rtp_hdr_packet += rtp_hdr_len / sizeof(uint32_t);
                        pos += packet_sizes.at(i);
                }
        }

        if (!tx->encryption) {
                rtp_async_start(rtp_session, mult_pkt_cnt);
        }

        rtp_hdr_packet = (uint32_t *) rtp_headers;
        for (long i = 0; i < mult_pkt_cnt; ++i) {
                GET_STARTTIME;
                const int m        = i == mult_pkt_cnt - 1 ? send_m : 0;
                char     *data     = tile->data + ntohl(rtp_hdr_packet[1]);
                int       data_len = packet_sizes.at(i % packet_sizes.size());

                char encrypted_data[RTP_MAX_PACKET_LEN + MAX_CRYPTO_EXCEED];
                if (tx->encryption != nullptr) {
                        data_len = tx->enc_funcs->encrypt(
                            tx->encryption, data, data_len,
                            (char *) rtp_hdr_packet,
                            frame->fec_params.type != FEC_NONE
                                ? sizeof(fec_payload_hdr_t)
                                : sizeof(video_payload_hdr_t),
                            encrypted_data);
                        data = encrypted_data;
                }

                rtp_send_data_hdr(rtp_session, ts, pt, m, 0, nullptr,
                                  (char *) rtp_hdr_packet, rtp_hdr_len, data,
                                  data_len, nullptr, 0, 0);
                rtp_hdr_packet += rtp_hdr_len / sizeof(uint32_t);

                // TRAFFIC SHAPER
                if (m != 1) { // wait for all but last packet
                        do {
                                GET_STOPTIME;
                                GET_DELTA;
                        } while (packet_rate - delta - overslept > 0);
                        overslept = -(packet_rate - delta - overslept);
                        //fprintf(stdout, "%ld ", overslept);
                }
        }

        const long data_sent = tile->data_len + rtp_hdr_len * mult_pkt_cnt;
        report_stats(tx, rtp_session, data_sent);

        if (!tx->encryption) {
                rtp_async_wait(rtp_session);
        }
        free(rtp_headers);
}

static void audio_tx_send_chan(struct tx *tx, struct rtp *rtp_session,
                               uint32_t timestamp, const audio_frame2 *buffer,
                               int channel, bool send_m);

/* 
 * This multiplication scheme relies upon the fact, that our RTP/pbuf implementation is
 * not sensitive to packet duplication. Otherwise, we can get into serious problems.
 */
void
audio_tx_send(struct tx *tx, struct rtp *rtp_session,
              const audio_frame2 *buffer)
{
        if (!rtp_has_receiver(rtp_session)) {
                return;
        }

        fec_check_messages(tx);

        const uint32_t timestamp =
            buffer->get_timestamp() == -1
                ? get_local_mediatime()
                : get_local_mediatime_offset() + buffer->get_timestamp();

        for (int iter = 0; iter < tx->mult_count; ++iter) {
                for (int chan = 0; chan < buffer->get_channel_count(); ++chan) {
                        bool send_m = iter == tx->mult_count - 1 &&
                                      chan == buffer->get_channel_count() - 1;
                        audio_tx_send_chan(tx, rtp_session, timestamp, buffer,
                                           chan, send_m);
                }
        }

        tx->buffer++;
}

static void
audio_tx_send_chan(struct tx *tx, struct rtp *rtp_session, uint32_t timestamp,
                   const audio_frame2 *buffer, int channel, bool send_m)
{
        int pt = fec_pt_from_fec_type(
            TX_MEDIA_AUDIO, buffer->get_fec_params(0).type,
            tx->encryption); /* PT set for audio in our packet format */
        unsigned m = 0U;
        // see definition in rtp_callback.h
        uint32_t rtp_hdr[100];

        int rtp_hdr_len = 0;
        int hdrs_len = get_tx_hdr_len(rtp_is_ipv6(rtp_session));
        unsigned int fec_symbol_size =
            buffer->get_fec_params(channel).symbol_size;

        const char *chan_data = buffer->get_data(channel);
        unsigned    pos       = 0U;

        if (buffer->get_fec_params(0).type == FEC_NONE) {
                hdrs_len += (sizeof(audio_payload_hdr_t));
                rtp_hdr_len = sizeof(audio_payload_hdr_t);
                format_audio_header(buffer, channel, tx->buffer, rtp_hdr);
        } else {
                hdrs_len += (sizeof(fec_payload_hdr_t));
                rtp_hdr_len  = sizeof(fec_payload_hdr_t);
                uint32_t tmp = channel << 22;
                tmp |= 0x3fffff & tx->buffer;
                // see definition in rtp_callback.h
                rtp_hdr[0] = htonl(tmp);
                rtp_hdr[2] = htonl(buffer->get_data_len(channel));
                rtp_hdr[3] = htonl(buffer->get_fec_params(channel).k << 19 |
                                   buffer->get_fec_params(channel).m << 6 |
                                   buffer->get_fec_params(channel).c);
                rtp_hdr[4] = htonl(buffer->get_fec_params(channel).seed);
        }

        if (tx->encryption) {
                hdrs_len += sizeof(crypto_payload_hdr_t) +
                            tx->enc_funcs->get_overhead(tx->encryption);
                rtp_hdr[rtp_hdr_len / sizeof(uint32_t)] =
                    htonl(DEFAULT_CIPHER_MODE << 24);
                rtp_hdr_len += sizeof(crypto_payload_hdr_t);
        }

        if (buffer->get_fec_params(0).type != FEC_NONE) {
                check_symbol_size(fec_symbol_size, tx->mtu - hdrs_len);
        }

        long data_sent = 0;
        do {
                const char *data     = chan_data + pos;
                int         data_len = tx->mtu - hdrs_len;
                if (pos + data_len >=
                    (unsigned int) buffer->get_data_len(channel)) {
                        data_len = buffer->get_data_len(channel) - pos;
                        if (send_m) {
                                m = 1;
                        }
                }
                rtp_hdr[1] = htonl(pos);
                pos += data_len;

                char encrypted_data[RTP_MAX_PACKET_LEN + MAX_CRYPTO_EXCEED];
                if (tx->encryption) {
                        data_len = tx->enc_funcs->encrypt(
                            tx->encryption, const_cast<char *>(data), data_len,
                            (char *) rtp_hdr,
                            rtp_hdr_len - sizeof(crypto_payload_hdr_t),
                            encrypted_data);
                        data = encrypted_data;
                }

                data_sent += data_len + rtp_hdr_len;

                rtp_send_data_hdr(rtp_session, timestamp, pt, m,
                                  0, /* contributing sources */
                                  0, /* contributing sources length */
                                  (char *) rtp_hdr, rtp_hdr_len,
                                  const_cast<char *>(data), data_len, 0, 0, 0);

        } while (pos < buffer->get_data_len(channel));

        report_stats(tx, rtp_session, data_sent);
}

static bool
validate_std_audio(const audio_frame2 * buffer, int payload_size)
{
        if ((buffer->get_codec() == AC_MP3 ||
             buffer->get_codec() == AC_OPUS) &&
            buffer->get_channel_count() > 1) { // we cannot interleave Opus here
                const uint32_t msg_id = to_fourcc('t', 'x', 'v', 'a');
                log_msg_once(LOG_LEVEL_ERROR, msg_id,
                             MOD_NAME
                             "%s can currently have only 1 channel in "
                             "RFC-compliant mode! Discarding channels but the "
                             "first one...\n",
                             get_name_to_audio_codec(buffer->get_codec()));
        }
        if (buffer->get_codec() == AC_OPUS &&
            payload_size < (int) buffer->get_data_len(0)) {
                MSG(ERROR, "Opus frame larger than packet! Discarding...\n");
                return false;
        }
        return true;
}

/**
 * audio_tx_send_standard - Send interleaved channels from the audio_frame2,
 *                       	as the mulaw and A-law standards (dynamic or std PT).
 */
void audio_tx_send_standard(struct tx* tx, struct rtp *rtp_session,
		const audio_frame2 * buffer) {
	//TODO to be more abstract in order to accept A-law too and other supported standards with such implementation
        assert(buffer->get_codec() == AC_MULAW ||
               buffer->get_codec() == AC_ALAW ||
               buffer->get_codec() == AC_MP3 ||
               buffer->get_codec() == AC_OPUS);

	uint32_t ts;
	static uint32_t ts_prev = 0;

	// Configure the right Payload type,
	// 8000 Hz, 1 channel and 2 bps is the ITU-T G.711 standard (should be 1 bps...)
	// Other channels or Hz goes to DynRTP-Type97
	int pt = PT_DynRTP_Type97;
	if (buffer->get_channel_count() == 1 && buffer->get_sample_rate() == 8000) {
		if (buffer->get_codec() == AC_MULAW)
			pt = PT_ITU_T_G711_PCMU;
		else if (buffer->get_codec() == AC_ALAW)
			pt = PT_ITU_T_G711_PCMA;
        } else if (buffer->get_codec() == AC_MP3) {
                pt = PT_MPA;
        }

	int data_len = buffer->get_data_len(0); 	/* Number of samples to send 			*/
	int payload_size = tx->mtu - 40 - 8 - 12; /* Max size of an RTP payload field (minus IPv6, UDP and RTP header lengths) */

        if (pt == PT_ITU_T_G711_PCMU ||
            pt == PT_ITU_T_G711_PCMA) { // we may split the data into more
                                        // packets, compute chunk size
                int frame_size =
                    buffer->get_channel_count() * buffer->get_bps();
                payload_size = payload_size / frame_size * frame_size; // align to frame size
                // The sizes for the different channels must be the same.
                for (int i = 1; i < buffer->get_channel_count(); i++) {
                        assert(buffer->get_data_len(0) ==
                               buffer->get_data_len(i));
                }
                data_len *= buffer->get_channel_count();
        } else if (pt == PT_MPA) {
                payload_size -= sizeof(mpa_hdr_t);
        }

        if (!validate_std_audio(buffer, payload_size)) {
                return;
        }

	int pos = 0;
	do {
                int pkt_len = std::min(payload_size, data_len - pos);

                if (buffer->get_codec() == AC_OPUS) {
                        memcpy(tx->tmp_packet, buffer->get_data(0), pkt_len);
                } else if (buffer->get_codec() == AC_MP3) {
                        memset(tx->tmp_packet, 0, 2);
                        const uint16_t offset = htons(pos);
                        memcpy(tx->tmp_packet + 2, &offset, sizeof offset);
                        pkt_len += sizeof(mpa_hdr_t);
                        memcpy(tx->tmp_packet + 4, buffer->get_data(0), pkt_len);
                } else { // interleave
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
                rtp_send_ctrl(rtp_session, ts_prev, 0, get_time_in_ns()); //send RTCP SR
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

	char pt =  PT_DynRTP_Type96;
	unsigned char hdr[2];
	int cc = 0;
	uint32_t csrc = 0;
	int m = 0;
	char *extn = 0;
	uint16_t extn_len = 0;
	uint16_t extn_type = 0;
	const uint8_t *start = (uint8_t *) tile->data;
	int data_len = tile->data_len;
	unsigned maxPacketSize = tx->mtu - 40;

        const unsigned char *endptr = 0;
        const unsigned char *nal = start;

        while ((nal = rtpenc_get_next_nal(nal, data_len - (nal - start), &endptr))) {
                unsigned int nalsize = endptr - nal;
                bool eof = endptr == start + data_len;
                bool lastNALUnitFragment = false; // by default
                unsigned curNALOffset = 0;
                char *nalc = const_cast<char *>(reinterpret_cast<const char *>(nal));

		while(!lastNALUnitFragment){
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
			if (curNALOffset == 0) { // case 1 or 2
				if (nalsize	<= maxPacketSize) { // case 1

					if (eof) m = 1;
					if (rtp_send_data(rtp_session, ts, pt, m, cc, &csrc,
							nalc, nalsize,
							extn, extn_len, extn_type) < 0) {
						error_msg("There was a problem sending the RTP packet\n");
					}
					lastNALUnitFragment = true;
				} else { // case 2
					// We need to send the NAL unit data as FU packets.  Deliver the first
					// packet now.  Note that we add "NAL header" and "FU header" bytes to the front
					// of the packet (overwriting the existing "NAL header").
					hdr[0] = (nal[0] & 0xE0) | 28; //FU indicator
					hdr[1] = 0x80 | (nal[0] & 0x1F); // FU header (with S bit)

					if (rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
									(char *) hdr, 2,
									nalc + 1, maxPacketSize - 2,
									extn, extn_len, extn_type) < 0) {
										error_msg("There was a problem sending the RTP packet\n");
					}
					curNALOffset += maxPacketSize - 1;
					lastNALUnitFragment = false;
					nalsize -= maxPacketSize - 1;
				}
			} else { // case 3
				// We are sending this NAL unit data as FU packets.  We've already sent the
				// first packet (fragment).  Now, send the next fragment.  Note that we add
				// "NAL header" and "FU header" bytes to the front.  (We reuse these bytes that
				// we already sent for the first fragment, but clear the S bit, and add the E
				// bit if this is the last fragment.)
				hdr[1] = hdr[1] & ~0x80;// FU header (no S bit)

				if (nalsize + 1 > maxPacketSize) {
					// We can't send all of the remaining data this time:
					if (rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
							(char *) hdr, 2,
							nalc + curNALOffset,
							maxPacketSize - 2, extn, extn_len,
							extn_type) < 0) {
								error_msg("There was a problem sending the RTP packet\n");
					}
					curNALOffset += maxPacketSize - 2;
					lastNALUnitFragment = false;
					nalsize -= maxPacketSize - 2;

				} else {
					// This is the last fragment:
					if (eof) m = 1;

					hdr[1] |= 0x40;// set the E bit in the FU header

					if (rtp_send_data_hdr(rtp_session, ts, pt, m, cc, &csrc,
									(char *) hdr, 2,
									nalc + curNALOffset,
									nalsize, extn, extn_len, extn_type) < 0) {
										error_msg("There was a problem sending the RTP packet\n");
					}
					lastNALUnitFragment = true;
				}
			}
		}
	}
        if (endptr != start + data_len) {
                error_msg("No NAL found!\n");
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

