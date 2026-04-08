/**
 * @file   video_rxtx/rtp.cpp
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

#include <cassert>
#include <cinttypes>
#include <sstream>
#include <string>

#include "config.h"              // for PACKAGE_STRING
#include "debug.h"
#include "host.h"
#include "messaging.h"
#include "module.h"
#include "pdb.h"
#include "rtp/fec.h"
#include "rtp/pbuf.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/video_decoders.h"
#include "transmit.h"
#include "types.h"
#include "ug_runtime_error.hpp"
#include "utils/lock_guard.h"    // for ultragrid::pthread_mutex_guard
#include "utils/net.h" // IN6_BLACKHOLE_STR
#include "utils/pthread.h" // for CHK_PTHR
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "video_display.h"
#include "video_rxtx.h"
#include "video_rxtx/rtp.hpp"

#define DEFAULT_AUDIO_RECV_BUF_SIZE (256 * 1024)

#define MAGIC to_fourcc('V', 'X', 'r', ' ')
#define MOD_NAME "[video_rxtx/rtp] "

#if !defined _WIN32
constexpr bool VIDEO_MT = true;
#else
constexpr bool VIDEO_MT = false;
#endif

using std::ostringstream;
using std::string;
using ultragrid::pthread_mutex_guard;

struct rtp_medium_priv {
        int rx_port;
        int tx_port;
        bool mutex_init;
};

struct rtp_rxtx_common_priv_state {
        uint32_t magic;

        // stored for reconfiguration
        int   force_ip_version;
        char *mcast_if;
        int   ttl;
        char *requested_receiver;
        struct rtp_medium_priv medium[NUM_TX_MEDIA];

        struct rtp_rxtx_common pub;
        /// This is child of vrxtx sender module to process specific messages.
        /// The receiver module is used directly (vrxtx doesn't process any
        /// message and also the send/receive handling is not entirely
        /// symetric).
        struct module m_rtp_sender_mod;

        bool used;
};

static struct rtp *initialize_network(const char *addr, int recv_port,
                                      int send_port, struct pdb *participants,
                                      int         force_ip_version,
                                      const char *mcast_if, int ttl,
                                      enum tx_media_type medium);
static void        destroy_rtp_device(struct rtp *network_device);

static struct response *
rtp_process_sender_message(struct rtp_rxtx_common_priv_state *s, struct msg_sender *msg)
{
        /// @todo audio
        struct rtp_rxtx_medium *video = &s->pub.medium[TX_MEDIA_VIDEO];
        struct rtp_medium_priv *video_priv = &s->medium[TX_MEDIA_VIDEO];
        switch (msg->type) {
        case SENDER_MSG_CHANGE_RECEIVER: {
                assert(video->rxtx_mode == MODE_SENDER); // sender only
                pthread_mutex_guard lock(video->lock);
                auto             *old_device   = video->network_device;
                char *old_receiver = s->requested_receiver;
                s->requested_receiver = strdup(msg->receiver);
                video->network_device            = initialize_network(
                    s->requested_receiver, video_priv->rx_port,
                    video_priv->tx_port, video->participants,
                    s->force_ip_version, s->mcast_if, s->ttl, TX_MEDIA_VIDEO);
                if (video->network_device == nullptr) {
                        video->network_device = old_device;
                        free(s->requested_receiver);
                        s->requested_receiver = old_receiver;
                        MSG(ERROR, "Failed receiver to %s.\n", msg->receiver);
                        return new_response(RESPONSE_INT_SERV_ERR,
                                            "Changing receiver failed!");
                }
                MSG(NOTICE, "Changed receiver to %s.\n", msg->receiver);
                destroy_rtp_device(old_device);
                free(old_receiver);
        } break;
        case SENDER_MSG_CHANGE_PORT: {
                assert(video->rxtx_mode == MODE_SENDER); // sender only
                pthread_mutex_guard lock(video->lock);
                auto             *old_device = video->network_device;
                auto              old_port   = video_priv->tx_port;

                video_priv->tx_port = msg->tx_port;
                if (msg->rx_port != 0) {
                        video_priv->rx_port = msg->rx_port;
                }
                video->network_device = initialize_network(
                    s->requested_receiver, video_priv->rx_port,
                    video_priv->tx_port, video->participants,
                    s->force_ip_version,
                    s->mcast_if, s->ttl, TX_MEDIA_VIDEO);

                if (video->network_device == nullptr) {
                        video->network_device   = old_device;
                        video_priv->tx_port = old_port;
                        MSG(ERROR, "Failed to Change TX port to %d.\n",
                            msg->tx_port);
                        return new_response(RESPONSE_INT_SERV_ERR,
                                            "Changing TX port failed!");
                }
                MSG(NOTICE, "Changed TX port to %d.\n", msg->tx_port);
                destroy_rtp_device(old_device);
        } break;
        case SENDER_MSG_CHANGE_FEC: {
                pthread_mutex_guard lock(video->lock);
                auto               *old_fec_state = s->pub.fec_state;
                s->pub.fec_state                  = nullptr;
                if (strcmp(msg->fec_cfg, "flush") == 0) {
                        delete old_fec_state;
                        break;
                }
                s->pub.fec_state = fec::create_from_config(msg->fec_cfg, false);
                if (s->pub.fec_state == nullptr) {
                        int rc = 0;
                        if (strstr(msg->fec_cfg, "help") == nullptr) {
                                MSG(ERROR, "Unable to initialize FEC!\n");
                                rc = 1;
                        }

                        // Exit only if we failed because of command line
                        // params, not control port msg
                        if (s->used) {
                                exit_uv(rc);
                        }

                        s->pub.fec_state = old_fec_state;
                        return new_response(RESPONSE_INT_SERV_ERR, nullptr);
                }
                delete old_fec_state;
                MSG(NOTICE, "Fec changed successfully\n");
        } break;
        case SENDER_MSG_GET_STATUS:
        case SENDER_MSG_MUTE:
        case SENDER_MSG_UNMUTE:
        case SENDER_MSG_MUTE_TOGGLE:
                MSG(ERROR, "Unexpected audio message ID %d!\n", msg->type);
                return new_response(RESPONSE_INT_SERV_ERR, nullptr);
        default:
                MSG(ERROR, "Unsupported message ID %d!\n", msg->type);
                return new_response(RESPONSE_INT_SERV_ERR, nullptr);
        }

        return new_response(RESPONSE_OK, nullptr);
}

void rtp_rxtx_sender_do_housekeeping(struct rtp_rxtx_common *pub)
{
        struct rtp_rxtx_common_priv_state *s = pub->priv;
        s->used = true;

        struct message *msg_external = nullptr;
        while ((msg_external = check_message(&s->m_rtp_sender_mod)) !=
               nullptr) {
                auto *msg = (struct msg_sender *) msg_external;
                struct response *r = rtp_process_sender_message(s, msg);
                free_message(msg_external, r);
        }
}

static void
init_medium_state(struct rtp_rxtx_common_priv_state *s,
                  const struct common_opts          *opts,
                  const struct vrxtx_params *params, enum tx_media_type t)
{
        const struct rxtx_medium_params *params_medium = &params->medium[t];
        struct rtp_medium_priv          *medium_priv   = &s->medium[t];
        struct rtp_rxtx_medium          *medium_pub    = &s->pub.medium[t];
        const struct {
                const char   *medium_str;
                volatile int *medium_offset;
                long long     bitrate_limit;

        } medium_defaults[NUM_TX_MEDIA] = {
                { "audio", &audio_offset, 0                     },
                { "video", &video_offset, params->bitrate_limit },
        };
        const char   *medium_str    = medium_defaults[t].medium_str;
        volatile int *medium_offset = medium_defaults[t].medium_offset;
        long long     bitrate_limit = medium_defaults[t].bitrate_limit;

        if (params_medium->rxtx_mode == 0) { // no RX or TX for medium
                return;
        }
        medium_priv->rx_port = params_medium->rx_port;
        medium_priv->tx_port = params_medium->tx_port;

        medium_pub->rxtx_mode      = params_medium->rxtx_mode;
        medium_pub->participants   = pdb_init(medium_str, medium_offset);
        medium_pub->network_device = initialize_network(
            opts->receiver, medium_priv->rx_port, medium_priv->tx_port,
            medium_pub->participants, opts->force_ip_version, opts->mcast_if,
            opts->ttl, t);
        if (medium_pub->network_device == nullptr) {
                throw ug_runtime_error(string("Unable to open ") + medium_str +
                                           " network",
                                       EXIT_FAIL_NETWORK);
        }
        if (params_medium->rxtx_mode & MODE_SENDER) {
                medium_pub->tx = tx_init(&s->m_rtp_sender_mod, opts->mtu,
                                         TX_MEDIA_VIDEO, params_medium->fec,
                                         opts->encryption, bitrate_limit);
                if (medium_pub->tx == nullptr) {
                        throw ug_runtime_error(string("Unable to initialize ") +
                                                   medium_str + " transmitter",
                                               EXIT_FAIL_TRANSMIT);
                }
        }
        pthread_mutex_init(&medium_pub->lock, nullptr);
        medium_priv->mutex_init = true;
}

struct rtp_rxtx_common *rtp_rxtx_common_init(const struct vrxtx_params *params,
                       const struct common_opts  *common)
{
        auto *s = (struct rtp_rxtx_common_priv_state *) calloc(
            1, sizeof(struct rtp_rxtx_common_priv_state));
        struct rtp_rxtx_common *pub = &s->pub;
        pub->magic = RTP_COMMON_MAGIC;

        pub->priv = s;
        s->magic = MAGIC;
        s->force_ip_version   = common->force_ip_version,
        s->mcast_if           = strdup(common->mcast_if);
        s->ttl                = common->ttl;
        s->requested_receiver = strdup(common->receiver),

        module_init_default(&s->m_rtp_sender_mod);
        s->m_rtp_sender_mod.cls = MODULE_CLASS_DATA;
        module_register(&s->m_rtp_sender_mod, params->sender_mod);

        for (unsigned i = 0; i < NUM_TX_MEDIA; ++i) {
                try {
                        init_medium_state(s, common, params, (enum tx_media_type) i);
                } catch (...) {
                        rtp_rxtx_common_done(pub);
                        throw;
                }
        }

        // The idea of doing that is to display help on '-f ldgm:help' even if UG would exit
        // immediately. The encoder is actually created by a message.
        // Also for `-x sdp:help` the message will get discarded and the warning that message quie
        rtp_rxtx_sender_do_housekeeping(pub);

        return pub;
}

void
rtp_rxtx_common_done(struct rtp_rxtx_common *pub)
{
        auto *s = pub->priv;
        assert(s->magic == MAGIC);

        for (unsigned i = 0; i < NUM_TX_MEDIA; ++i) {
                struct rtp_rxtx_medium *medium = &pub->medium[i];
                if (medium->network_device != nullptr) {
                        destroy_rtp_device(medium->network_device);
                }
                if (medium->participants != nullptr) {
                        pdb_destroy(&medium->participants);
                }
                if (medium->tx != nullptr) {
                        tx_done(medium->tx);
                }
                if (s->medium[i].mutex_init) {
                        CHK_PTHR(pthread_mutex_destroy(&medium->lock));
                }
        }

        delete pub->fec_state;
        free(pub->priv->mcast_if);
        free(pub->priv->requested_receiver);
        module_done(&s->m_rtp_sender_mod);

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
        pthread_mutex_guard lock(s->lock);
        pdb_iter_t          it;
        /// @todo should be set only to relevant participant,
        /// not all
        struct pdb_e *cp = pdb_iter_init(s->participants, &it);
        while (cp != nullptr) {
                pbuf_set_playout_delay(cp->playout_buffer, delay);

                cp = pdb_iter_next(&it);
        }
}
