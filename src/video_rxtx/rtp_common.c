/**
 * @file   video_rxtx/rtp_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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

#include "video_rxtx/rtp_common.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>            // for strdup, strcmp, strlen, strstr
#include <sys/time.h>            // for timeval

#include "compat/c23.h"          // IWYU pragma: keep
#include "compat/net.h"          // for sockaddr_storage
#include "config.h"              // for PACKAGE_STRING
#include "debug.h"
#include "host.h"
#include "messaging.h"
#include "module.h"
#include "pdb.h"
#include "rtp/audio_decoders.h"
#include "rtp/fec.h"
#include "rtp/pbuf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "transmit.h"
#include "tv.h"                  // for time_ns_t, get_time_in_ns, NS_IN_SEC
#include "types.h"
#include "utils/pthread.h" // for CHK_PTHR
#include "utils/string.h"      // for strprintf
#include "video_rxtx.h"

#define MAGIC to_fourcc('V', 'X', 'r', ' ')
#define MOD_NAME "[video_rxtx/rtp] "

#if !defined _WIN32
#define VIDEO_MT true
#else
#define VIDEO_MT false
#endif

const enum module_class path_sender_video[] = { MODULE_CLASS_SENDER,
                                                MODULE_CLASS_VIDEO,
                                                MODULE_CLASS_NONE };
const enum module_class path_sender_audio[] = { MODULE_CLASS_SENDER,
                                                MODULE_CLASS_AUDIO,
                                                MODULE_CLASS_NONE };

struct pdb;
struct rtp;

struct rtp_medium_priv {
        int   rx_port;
        int   tx_port;
        bool  mutex_initialized;
        char *requested_receiver;

        /// This is child of vrxtx sender module to process specific messages.
        /// The receiver module is used directly (vrxtx doesn't process any
        /// message and also the send/receive handling is not entirely
        /// symetric).
        struct module sender_mod;
};

struct rtp_rxtx_common_priv_state {
        uint32_t magic;

        struct module *parent;

        time_ns_t start_time;
        // stored for reconfiguration
        int   force_ip_version;
        char *mcast_if;
        int   ttl;
        struct rtp_medium_priv medium[NUM_TX_MEDIA];

        struct rtp_rxtx_common pub;

        bool used;
        // audio
        time_ns_t a_last_not_timeout;
};

static struct rtp *initialize_network(const char *addr, int recv_port,
                                      int send_port, struct pdb *participants,
                                      int         force_ip_version,
                                      const char *mcast_if, int ttl,
                                      enum tx_media_type medium);
static void        destroy_rtp_device(struct rtp *network_device);

static struct response *
rtp_process_sender_message(struct rtp_rxtx_common_priv_state *s,
                           struct msg_sender *msg, enum tx_media_type t)
{
        struct rtp_rxtx_medium *medium_pub  = &s->pub.medium[t];
        struct rtp_medium_priv *medium_priv = &s->medium[t];
        const char             *medium_name = get_tx_name(t);
        struct response        *r           = nullptr;
        pthread_mutex_lock(&medium_pub->lock);
        switch (msg->type) {
        case SENDER_MSG_CHANGE_RECEIVER: {
                assert(medium_pub->rxtx_mode == MODE_SENDER); // sender only
                struct rtp *old_device   = medium_pub->network_device;
                char       *old_receiver = medium_priv->requested_receiver;
                medium_priv->requested_receiver = strdup(msg->receiver);
                medium_pub->network_device      = initialize_network(
                    medium_priv->requested_receiver, medium_priv->rx_port,
                    medium_priv->tx_port, medium_pub->participants,
                    s->force_ip_version, s->mcast_if, s->ttl, t);
                if (medium_pub->network_device == nullptr) {
                        medium_pub->network_device = old_device;
                        const char *err=
                            strprintf("Failed to set %s receiver to %s!",
                                      medium_name, msg->receiver);
                        free(medium_priv->requested_receiver);
                        medium_priv->requested_receiver = old_receiver;
                        MSG(ERROR, "%s\n", err);
                        r = new_response(RESPONSE_INT_SERV_ERR, err);
                } else {
                        MSG(NOTICE, "Changed %s receiver to %s.\n", medium_name,
                            msg->receiver);
                        destroy_rtp_device(old_device);
                        free(old_receiver);
                }
                break;
        }
        case SENDER_MSG_CHANGE_PORT: {
                assert(medium_pub->rxtx_mode == MODE_SENDER); // sender only
                struct rtp *old_device = medium_pub->network_device;
                int         old_port   = medium_priv->tx_port;

                medium_priv->tx_port = msg->tx_port;
                if (msg->rx_port != 0) {
                        medium_priv->rx_port = msg->rx_port;
                }
                medium_pub->network_device = initialize_network(
                    medium_priv->requested_receiver, medium_priv->rx_port,
                    medium_priv->tx_port, medium_pub->participants,
                    s->force_ip_version, s->mcast_if, s->ttl, t);

                if (medium_pub->network_device == nullptr) {
                        medium_pub->network_device = old_device;
                        medium_priv->tx_port       = old_port;
                        const char *err=
                            strprintf("Failed to change %s TX port port to %d!",
                                      medium_name, msg->tx_port);
                        MSG(ERROR, "%s.\n", err);
                        r = new_response(RESPONSE_INT_SERV_ERR, err);
                } else {
                        MSG(NOTICE, "Changed %s TX port to %d.\n", medium_name,
                            msg->tx_port);
                        destroy_rtp_device(old_device);
                }
                break;
        }
        case SENDER_MSG_CHANGE_FEC: {
                struct fec *old_fec_state = medium_pub->fec_state;
                medium_pub->fec_state     = nullptr;
                if (strcmp(msg->fec_cfg, "flush") == 0) {
                        fec_destroy(old_fec_state);
                        break;
                }
                medium_pub->fec_state =
                    fec_create_from_config(msg->fec_cfg, false);
                if (medium_pub->fec_state == nullptr) {
                        int rc = 0;
                        if (strstr(msg->fec_cfg, "help") == nullptr) {
                                MSG(ERROR, "Unable to initialize %s FEC!\n",
                                    medium_name);
                                rc = 1;
                        }

                        // Exit only if we failed because of command line
                        // params, not control port msg
                        if (s->used) {
                                exit_uv(rc);
                        }

                        medium_pub->fec_state = old_fec_state;
                        r = new_response(RESPONSE_INT_SERV_ERR, nullptr);
                } else {
                        fec_destroy(old_fec_state);
                        MSG(NOTICE, "%s FEC changed successfully\n",
                            medium_name);
                }
                break;
        }
        default:
                MSG(ERROR, "Unsupported message ID %d!\n", msg->type);
                r = new_response(RESPONSE_INT_SERV_ERR, nullptr);
        }

        pthread_mutex_unlock(&medium_pub->lock);
        if (r == nullptr) { // implicitly success
                r = new_response(RESPONSE_OK, nullptr);
        }
        return r;
}

void
rtp_rxtx_sender_do_housekeeping(struct rtp_rxtx_common *pub,
                                enum tx_media_type      t)
{
        struct rtp_rxtx_common_priv_state *s           = pub->priv;
        struct rtp_rxtx_medium            *medium_pub  = &s->pub.medium[t];
        struct rtp_medium_priv            *medium_priv = &s->medium[t];

        if (medium_priv->requested_receiver == nullptr) { // medium not used
                return;
        }

        s->used                                        = true;

        struct message *msg_external = nullptr;
        while ((msg_external = check_message(&medium_priv->sender_mod)) !=
               nullptr) {
                struct msg_sender *msg = (struct msg_sender *) msg_external;
                struct response *r = rtp_process_sender_message(s, msg, t);
                free_message(msg_external, r);
        }

        // do the house keeping if no receiver thread, otherwise it does
        // the stuff...
        if (t == TX_MEDIA_AUDIO &&
            (medium_pub->rxtx_mode & MODE_RECEIVER) == 0) {
                time_ns_t curr_time = get_time_in_ns();
                uint32_t  ts = (curr_time - s->start_time) / (100 * 1000) *
                               9; // at 90000 Hz
                rtp_update(medium_pub->network_device, curr_time);
                rtp_send_ctrl(medium_pub->network_device, ts, 0, curr_time);

                // receive RTCP
                struct timeval timeout;
                timeout.tv_sec  = 0;
                timeout.tv_usec = 0;
                rtcp_recv_r(medium_pub->network_device, &timeout, ts);
        }
}

static bool
init_medium_state(struct rtp_rxtx_common_priv_state *s,
                  const struct vrxtx_params *params, enum tx_media_type t)
{
        const struct rxtx_medium_params *params_medium = &params->medium[t];
        struct rtp_medium_priv          *medium_priv   = &s->medium[t];
        struct rtp_rxtx_medium          *medium_pub    = &s->pub.medium[t];
        const struct {
                volatile int *medium_offset;
                long long     bitrate_limit;
                enum module_class mod_cls;

        } medium_defaults[NUM_TX_MEDIA] = {
                { &audio_offset, 0,                     MODULE_CLASS_AUDIO },
                { &video_offset, params->bitrate_limit, MODULE_CLASS_VIDEO },
        };
        const char       *medium_str    = get_tx_name(t);
        volatile int     *medium_offset = medium_defaults[t].medium_offset;
        long long         bitrate_limit = medium_defaults[t].bitrate_limit;
        enum module_class mod_cls       = medium_defaults[t].mod_cls;
        bool              fec_help      = strstr(params_medium->fec, "help");

        if (params_medium->rxtx_mode == 0 &&
            !fec_help) { // no RX or TX for medium
                return true;
        }
        medium_priv->rx_port = params_medium->rx_port;
        medium_priv->tx_port = params_medium->tx_port;
        medium_priv->requested_receiver = strdup(params->receiver),

        module_init_default(&medium_priv->sender_mod);
        medium_priv->sender_mod.cls = mod_cls;
        module_register(&medium_priv->sender_mod, params->sender_mod);

        medium_pub->rxtx_mode      = params_medium->rxtx_mode;
        medium_pub->participants   = pdb_init(medium_str, medium_offset);
        medium_pub->network_device = initialize_network(
            params->receiver, medium_priv->rx_port, medium_priv->tx_port,
            medium_pub->participants, params->force_ip_version, params->mcast_if,
            params->ttl, t);
        if (medium_pub->network_device == nullptr) {
                MSG(ERROR, "Unable to open %s network!\n",  medium_str);
                return false;
        }
        if (params_medium->rxtx_mode & MODE_SENDER || fec_help) {
                medium_pub->tx = tx_init(&medium_priv->sender_mod, params->mtu, t,
                                         params_medium->fec, params->encryption,
                                         bitrate_limit);
                if (medium_pub->tx == nullptr) {
                        MSG(ERROR, "Unable to open %s transmitter!\n",  medium_str);
                        return false;
                }
        }

        pthread_mutex_init(&medium_pub->lock, nullptr);
        medium_priv->mutex_initialized = true;
        return true;
}

struct rtp_rxtx_common *rtp_rxtx_common_init(const struct vrxtx_params *params)
{
        struct rtp_rxtx_common_priv_state *s = calloc(
            1, sizeof(struct rtp_rxtx_common_priv_state));
        struct rtp_rxtx_common *pub = &s->pub;
        pub->magic = RTP_COMMON_MAGIC;

        pub->priv = s;
        pub->encryption       = strdup(params->encryption);
        s->magic              = MAGIC;
        s->parent             = params->parent;
        s->force_ip_version   = params->force_ip_version,
        s->mcast_if           = strdup(params->mcast_if);
        s->ttl                = params->ttl;
        s->start_time         = params->start_time;

        for (unsigned i = 0; i < NUM_TX_MEDIA; ++i) {
                bool rc = init_medium_state(s, params, i);
                if (!rc) {
                        rtp_rxtx_common_done(pub);
                        return nullptr;
                }
        }

        return pub;
}

void
rtp_rxtx_common_done(struct rtp_rxtx_common *pub)
{
        if (pub == nullptr) {
                return;
        }

        struct rtp_rxtx_common_priv_state *s = pub->priv;
        assert(s->magic == MAGIC);

        for (unsigned i = 0; i < NUM_TX_MEDIA; ++i) {
                struct rtp_rxtx_medium *medium_pub = &pub->medium[i];
                // skip unread messages - eg. gen by rtsp deleteStream
                rtp_rxtx_sender_do_housekeeping(pub, i);
                if (medium_pub->network_device != nullptr) {
                        destroy_rtp_device(medium_pub->network_device);
                }
                if (medium_pub->participants != nullptr) {
                        pdb_destroy(&medium_pub->participants);
                }
                if (medium_pub->tx != nullptr) {
                        tx_done(medium_pub->tx);
                }
                struct rtp_medium_priv *medium_priv = &s->medium[i];
                if (medium_priv->mutex_initialized) {
                        CHK_PTHR(pthread_mutex_destroy(&medium_pub->lock));
                }
                free(medium_priv->requested_receiver);
                fec_destroy(medium_pub->fec_state);
                module_done(&medium_priv->sender_mod);
        }

        free(pub->priv->mcast_if);
        free(pub->encryption);

        free(s);
}

static struct rtp *
initialize_network(const char *addr, int recv_port, int send_port,
                   struct pdb *participants, int force_ip_version,
                   const char *mcast_if, int ttl, enum tx_media_type medium)
{
        // FIXME:  something about 5% for rtcp is said in rfc
        const double rtcp_bw =
            medium == TX_MEDIA_VIDEO ? 5 * 1024 * 1024 : 1024 * 512;
        const bool multithreaded = medium == TX_MEDIA_VIDEO ? VIDEO_MT : false;
        const int  buf_size      = medium == TX_MEDIA_VIDEO
                                       ? INITIAL_VIDEO_RECV_BUFFER_SIZE
                                       : DEFAULT_AUDIO_RECV_BUF_SIZE;

        if (strlen(mcast_if) == 0) {
                mcast_if = nullptr;
        }
        struct rtp *device =
            rtp_init_if(addr, mcast_if, recv_port, send_port, ttl, rtcp_bw,
                        false, rtp_recv_callback, (uint8_t *) participants,
                        force_ip_version, multithreaded);
        if (device == nullptr) {
                return nullptr;
        }
        rtp_set_option(device, RTP_OPT_WEAK_VALIDATION, true);
        rtp_set_option(device, RTP_OPT_PROMISC, true);
        rtp_set_sdes(device, rtp_my_ssrc(device),
                     RTCP_SDES_TOOL, PACKAGE_STRING, strlen(PACKAGE_STRING));

        if (medium == TX_MEDIA_VIDEO) {
                rtp_set_send_buf(device, INITIAL_VIDEO_SEND_BUFFER_SIZE);
        } else {
                rtp_set_option(device, RTP_OPT_RECORD_SOURCE, true);
        }
        rtp_set_recv_buf(device, buf_size);

        pdb_add(participants, rtp_my_ssrc(device));

        return device;
}

static void
destroy_rtp_device(struct rtp *network_device)
{
        if (network_device != nullptr) {
                rtp_done(network_device);
        }
        network_device = nullptr;
}

void
rtp_rxtx_set_pbuf_delay(struct rtp_rxtx_medium *s, double delay)
{
        pthread_mutex_lock(&s->lock);
        pdb_iter_t          it;
        /// @todo should be set only to relevant participant,
        /// not all
        struct pdb_e *cp = pdb_iter_init(s->participants, &it);
        while (cp != nullptr) {
                pbuf_set_playout_delay(cp->playout_buffer, delay);

                cp = pdb_iter_next(&it);
        }
        pthread_mutex_unlock(&s->lock);
}

bool
rtp_rxtx_common_is_ipv6(struct rtp_rxtx_common *pub)
{
        struct rtp *a_net_dev = pub->medium[TX_MEDIA_AUDIO].network_device;
        struct rtp *v_net_dev = pub->medium[TX_MEDIA_VIDEO].network_device;

        if (a_net_dev != nullptr) {
                const bool a_is_ipv6 = rtp_is_ipv6(a_net_dev);
                if (v_net_dev != nullptr &&
                    rtp_is_ipv6(v_net_dev) != a_is_ipv6) {
                        MSG(ERROR, "IP protocol version mismatch for A/V! This "
                                   "should not happen!\n");
                }
                return a_is_ipv6;
        }
        if (v_net_dev != nullptr) {
                return rtp_is_ipv6(v_net_dev);
        }
        return false;
}

struct rtp_audio_decoder {
        bool enabled;
        struct acodec_state pbuf_data;
};

static struct rtp_audio_decoder *
audio_decoder_state_create(struct rtp_rxtx_common *s)
{
        struct rtp_audio_decoder *dec_state =
            calloc(1, sizeof(struct rtp_audio_decoder));
        assert(dec_state != nullptr);
        dec_state->enabled = true;
        dec_state->pbuf_data.decoder =
            (struct state_audio_decoder *) audio_decoder_init(s->encryption,
                                                              s->priv->parent);
        if (!dec_state->pbuf_data.decoder) {
                free(dec_state);
                return nullptr;
        }
        return dec_state;
}

static void audio_decoder_state_deleter(void *state)
{
        struct rtp_audio_decoder *s = state;

        audio_decoder_destroy(s->pbuf_data.decoder);

        free(s);
}

struct rx_audio_frames *
rtp_recv_audio_frame(void *state, decode_audio_frame_fn decode)
{
        struct rtp_rxtx_common            *pub   = state;
        struct rtp_rxtx_common_priv_state *priv  = pub->priv;
        struct rtp_rxtx_medium            *audio = &pub->medium[TX_MEDIA_AUDIO];

        time_ns_t curr_time = get_time_in_ns();
        uint32_t  ts =
            (curr_time - priv->start_time) / (100*1000) * 9; // at 90000 Hz
        rtp_update(audio->network_device, curr_time);
        rtp_send_ctrl(audio->network_device, ts, 0, curr_time);
        struct timeval timeout;
        timeout.tv_sec = 0;
        // timeout.tv_usec = 999999 / 59.94; // audio goes almost always at the
        // same rate as video frames
        if ((curr_time - priv->a_last_not_timeout) > NS_IN_SEC) {
                timeout.tv_usec = 100000;
        } else {
                timeout.tv_usec = 1000; // this stuff really smells !!!
        }
        bool ret = rtp_recv_r(audio->network_device, &timeout, ts);
        if (ret) {
                priv->a_last_not_timeout = curr_time;
        }
        pdb_iter_t it;
        struct pdb_e *cp = pdb_iter_init(audio->participants, &it);

        struct rx_audio_frames *retval = nullptr;
        struct rx_audio_frames **retval_next = &retval;

        while (cp != nullptr) {
                if (cp->decoder_state == nullptr &&
                    !pbuf_is_empty(
                        cp->playout_buffer)) { // the second check is need ed
                                               // because we want to assign
                                               // display to participant that
                                               // really sends data
                        // disable all previous sources
                        if (!pub->playback_supports_multiple_streams) {
                                pdb_iter_t    it;
                                struct pdb_e *cp =
                                    pdb_iter_init(audio->participants, &it);
                                while (cp != nullptr) {
                                        if (cp->decoder_state) {
                                                ((struct rtp_audio_decoder *)
                                                     cp->decoder_state)
                                                    ->enabled = false;
                                        }
                                        cp = pdb_iter_next(&it);
                                }
                                pdb_iter_done(&it);
                        }

                        if (get_commandline_param("low-latency-audio")) {
                                pbuf_set_playout_delay(
                                    cp->playout_buffer,
                                    strcmp(get_commandline_param(
                                               "low-latency-audio"),
                                           "ultra") == 0
                                        ? 0.001
                                        : 0.005);
                        }
                        cp->decoder_state = audio_decoder_state_create(pub);
                        if (!cp->decoder_state) {
                                exit_uv(1);
                                break;
                        }
                        cp->decoder_state_deleter = audio_decoder_state_deleter;
                }

                struct rtp_audio_decoder *dec_state = cp->decoder_state;
                if (dec_state && dec_state->enabled) {
                        dec_state->pbuf_data.decoded = nullptr;
                        struct rx_audio_frames *decoded_frame = nullptr;
                        // dec_state->pbuf_data.buffer.data_len  = 0;
                        // dec_state->pbuf_data.buffer.timestamp = -1;

                        // We iterate in loop since there can be more than one
                        // frmae present in the playout buffer and it would be
                        // discarded by following pbuf_remove() call.
                        while (pbuf_decode(cp->playout_buffer, curr_time,
                                           decode, &dec_state->pbuf_data)) {
                                if (!decoded_frame) {
                                        decoded_frame =
                                            calloc(1, sizeof *decoded_frame);
                                        decoded_frame->frame =
                                            dec_state->pbuf_data.decoded;
                                        size_t slen =
                                            sizeof *decoded_frame->source;
                                        decoded_frame->source = malloc(slen);
                                        memcpy(decoded_frame->source,
                                               &dec_state->pbuf_data.source,
                                               slen);
                                        *retval_next = decoded_frame;
                                        retval_next  = &decoded_frame->next;
                                }
                                decoded_frame->expected_bytes +=
                                    dec_state->pbuf_data.expected_bytes;
                                decoded_frame->received_bytes +=
                                    dec_state->pbuf_data.received_bytes;
                        }
                }

                pbuf_remove(cp->playout_buffer, curr_time);
                cp = pdb_iter_next(&it);
        }
        pdb_iter_done(&it);
        return retval;
}
