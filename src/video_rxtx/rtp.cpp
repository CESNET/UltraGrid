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

#define MAGIC to_fourcc('V', 'X', 'r', ' ')
#define MOD_NAME "[video_rxtx/rtp] "

using std::ostringstream;
using std::string;
using ultragrid::pthread_mutex_guard;

struct rtp_rxtx_common_priv_state {
        uint32_t magic;

        // stored for reconfiguration
        int   force_ip_version;
        char *mcast_if;
        int   ttl;
        char *requested_receiver;
        int   rx_port;
        int   tx_port;

        struct rtp_rxtx_common info;
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
                                      const char *mcast_if, int ttl);
static void        destroy_rtp_device(struct rtp *network_device);

static struct response *
rtp_process_sender_message(struct rtp_rxtx_common *s, struct msg_sender *msg)
{
        switch (msg->type) {
        case SENDER_MSG_CHANGE_RECEIVER: {
                assert(s->priv->info.rxtx_mode == MODE_SENDER); // sender only
                ultragrid::pthread_mutex_guard lock(s->network_devices_lock);
                auto             *old_device   = s->network_device;
                char *old_receiver = s->priv->requested_receiver;
                s->priv->requested_receiver = strdup(msg->receiver);
                s->network_device           = initialize_network(
                    s->priv->requested_receiver, s->priv->rx_port,
                    s->priv->tx_port, s->participants,
                    s->priv->force_ip_version, s->priv->mcast_if, s->priv->ttl);
                if (s->network_device == nullptr) {
                        s->network_device     = old_device;
                        free(s->priv->requested_receiver);
                        s->priv->requested_receiver = old_receiver;
                        MSG(ERROR, "Failed receiver to %s.\n", msg->receiver);
                        return new_response(RESPONSE_INT_SERV_ERR,
                                            "Changing receiver failed!");
                }
                MSG(NOTICE, "Changed receiver to %s.\n", msg->receiver);
                destroy_rtp_device(old_device);
                free(old_receiver);
        } break;
        case SENDER_MSG_CHANGE_PORT: {
                assert(s->rxtx_mode == MODE_SENDER); // sender only
                ultragrid::pthread_mutex_guard lock(s->network_devices_lock);
                auto             *old_device = s->network_device;
                auto              old_port   = s->priv->tx_port;

                s->priv->tx_port = msg->tx_port;
                if (msg->rx_port != 0) {
                        s->priv->rx_port = msg->rx_port;
                }
                s->network_device = initialize_network(
                    s->priv->requested_receiver, s->priv->rx_port,
                    s->priv->tx_port, s->participants,
                    s->priv->force_ip_version,
                    s->priv->mcast_if, s->priv->ttl);

                if (s->network_device == nullptr) {
                        s->network_device   = old_device;
                        s->priv->tx_port = old_port;
                        MSG(ERROR, "Failed to Change TX port to %d.\n",
                            msg->tx_port);
                        return new_response(RESPONSE_INT_SERV_ERR,
                                            "Changing TX port failed!");
                }
                MSG(NOTICE, "Changed TX port to %d.\n", msg->tx_port);
                destroy_rtp_device(old_device);
        } break;
        case SENDER_MSG_CHANGE_FEC: {
                ultragrid::pthread_mutex_guard lock(s->network_devices_lock);
                auto             *old_fec_state = s->fec_state;
                s->fec_state                     = nullptr;
                if (strcmp(msg->fec_cfg, "flush") == 0) {
                        delete old_fec_state;
                        break;
                }
                s->fec_state = fec::create_from_config(msg->fec_cfg, false);
                if (s->fec_state == nullptr) {
                        int rc = 0;
                        if (strstr(msg->fec_cfg, "help") == nullptr) {
                                MSG(ERROR, "Unable to initialize FEC!\n");
                                rc = 1;
                        }

                        // Exit only if we failed because of command line
                        // params, not control port msg
                        if (s->priv->used) {
                                exit_uv(rc);
                        }

                        s->fec_state = old_fec_state;
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

void rtp_rxtx_sender_do_housekeeping(struct rtp_rxtx_common *s)
{
        s->priv->used = true;

        struct message *msg_external = nullptr;
        while ((msg_external = check_message(&s->priv->m_rtp_sender_mod)) !=
               nullptr) {
                auto *msg = (struct msg_sender *) msg_external;
                struct response *r = rtp_process_sender_message(s, msg);
                free_message(msg_external, r);
        }
}

struct rtp_rxtx_common *rtp_rxtx_common_init(const struct vrxtx_params *params,
                       const struct common_opts  *common)
{
        auto *priv = (struct rtp_rxtx_common_priv_state *) calloc(
            1, sizeof(struct rtp_rxtx_common_priv_state));
        struct rtp_rxtx_common *s = &priv->info;

        s->priv = priv;
        s->priv->magic = MAGIC;
        pthread_mutex_init(&s->network_devices_lock, nullptr);
        s->priv->force_ip_version   = common->force_ip_version,
        s->priv->mcast_if           = strdup(common->mcast_if);
        s->priv->ttl                = common->ttl;
        s->priv->requested_receiver = strdup(common->receiver),
        s->priv->rx_port            = params->rx_port,
        s->priv->tx_port            = params->tx_port;
        s->rxtx_mode                = params->rxtx_mode;

        s->participants = pdb_init("video", &video_offset);

        s->network_device = initialize_network(
            s->priv->requested_receiver, s->priv->rx_port,
            s->priv->tx_port, s->participants, common->force_ip_version,
            common->mcast_if, common->ttl);
        if (s->network_device == nullptr) {
                rtp_rxtx_common_done(s);
                throw ug_runtime_error("Unable to open network",
                                       EXIT_FAIL_NETWORK);
        }

        module_init_default(&priv->m_rtp_sender_mod);
        priv->m_rtp_sender_mod.cls = MODULE_CLASS_DATA;
        module_register(&priv->m_rtp_sender_mod, params->sender_mod);

        s->tx = tx_init(&priv->m_rtp_sender_mod, common->mtu, TX_MEDIA_VIDEO,
                       params->fec, common->encryption, params->bitrate_limit);
        if (s->tx == nullptr) {
                rtp_rxtx_common_done(s);
                throw ug_runtime_error("Unable to initialize transmitter",
                                       EXIT_FAIL_TRANSMIT);
        }

        // The idea of doing that is to display help on '-f ldgm:help' even if UG would exit
        // immediately. The encoder is actually created by a message.
        // Also for `-x sdp:help` the message will get discarded and the warning that message quie
        rtp_rxtx_sender_do_housekeeping(s);

        return s;
}

void
rtp_rxtx_common_done(struct rtp_rxtx_common *s)
{
        auto *priv = s->priv;
        assert(priv->magic == MAGIC);
        if (s->tx != nullptr) {
                tx_done(s->tx);
        }

        pthread_mutex_lock(&s->network_devices_lock);
        destroy_rtp_device(s->network_device);
        pthread_mutex_unlock(&s->network_devices_lock);

        if (s->participants != nullptr) {
                pdb_destroy(&s->participants);
        }

        delete s->fec_state;
        free(s->priv->mcast_if);
        free(s->priv->requested_receiver);
        module_done(&priv->m_rtp_sender_mod);

        CHK_PTHR(pthread_mutex_destroy(&s->network_devices_lock));

        free(priv);
}

static struct rtp *
initialize_network(const char *addr, int recv_port, int send_port,
                   struct pdb *participants, int force_ip_version,
                   const char *mcast_if, int ttl)
{
        double rtcp_bw = 5 * 1024 * 1024;       /* FIXME */

#if !defined _WIN32
        const bool multithreaded = true;
#else
        const bool multithreaded = false;
#endif

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

        rtp_set_recv_buf(device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
        rtp_set_send_buf(device, INITIAL_VIDEO_SEND_BUFFER_SIZE);

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

void rtp_rxtx_set_pbuf_delay(struct rtp_rxtx_common *s, double delay) {
        pthread_mutex_guard lock(s->network_devices_lock);
        pdb_iter_t          it;
        /// @todo should be set only to relevant participant,
        /// not all
        struct pdb_e *cp = pdb_iter_init(s->participants, &it);
        while (cp != nullptr) {
                pbuf_set_playout_delay(cp->playout_buffer, delay);

                cp = pdb_iter_next(&it);
        }
}
