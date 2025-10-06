/**
 * @file   video_rxtx/rtp.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2025 CESNET
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
#include <map>
#include <mutex>
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
#include "utils/misc.h"          // for format_in_si_units
#include "utils/net.h" // IN6_BLACKHOLE_STR
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "video_display.h"
#include "video_rxtx.hpp"
#include "video_rxtx/rtp.hpp"

#define MOD_NAME "[video_rxtx/rtp] "

using std::lock_guard;
using std::map;
using std::mutex;
using std::ostringstream;
using std::string;

struct response *
rtp_video_rxtx::process_sender_message(struct msg_sender *msg)
{
        switch (msg->type) {
        case SENDER_MSG_CHANGE_RECEIVER: {
                assert(m_rxtx_mode == MODE_SENDER); // sender only
                lock_guard<mutex> lock(m_network_devices_lock);
                auto             *old_device   = m_network_device;
                auto              old_receiver = m_requested_receiver;
                m_requested_receiver           = msg->receiver;
                m_network_device               = initialize_network(
                    m_requested_receiver.c_str(), m_recv_port_number,
                    m_send_port_number, m_participants,
                    m_common.force_ip_version,
                    m_common.mcast_if, m_common.ttl);
                if (m_network_device == nullptr) {
                        m_network_device     = old_device;
                        m_requested_receiver = std::move(old_receiver);
                        MSG(ERROR, "Failed receiver to %s.\n", msg->receiver);
                        return new_response(RESPONSE_INT_SERV_ERR,
                                            "Changing receiver failed!");
                }
                MSG(NOTICE, "Changed receiver to %s.\n", msg->receiver);
                destroy_rtp_device(old_device);
        } break;
        case SENDER_MSG_CHANGE_PORT: {
                assert(m_rxtx_mode == MODE_SENDER); // sender only
                lock_guard<mutex> lock(m_network_devices_lock);
                auto             *old_device = m_network_device;
                auto              old_port   = m_send_port_number;

                m_send_port_number = msg->tx_port;
                if (msg->rx_port != 0) {
                        m_recv_port_number = msg->rx_port;
                }
                m_network_device = initialize_network(
                    m_requested_receiver.c_str(), m_recv_port_number,
                    m_send_port_number, m_participants,
                    m_common.force_ip_version,
                    m_common.mcast_if, m_common.ttl);

                if (m_network_device == nullptr) {
                        m_network_device   = old_device;
                        m_send_port_number = old_port;
                        MSG(ERROR, "Failed to Change TX port to %d.\n",
                            msg->tx_port);
                        return new_response(RESPONSE_INT_SERV_ERR,
                                            "Changing TX port failed!");
                }
                MSG(NOTICE, "Changed TX port to %d.\n", msg->tx_port);
                destroy_rtp_device(old_device);
        } break;
        case SENDER_MSG_CHANGE_FEC: {
                lock_guard<mutex> lock(m_network_devices_lock);
                auto             *old_fec_state = m_fec_state;
                m_fec_state                     = nullptr;
                if (strcmp(msg->fec_cfg, "flush") == 0) {
                        delete old_fec_state;
                        break;
                }
                m_fec_state = fec::create_from_config(msg->fec_cfg, false);
                if (m_fec_state == nullptr) {
                        int rc = 0;
                        if (strstr(msg->fec_cfg, "help") == nullptr) {
                                MSG(ERROR, "Unable to initialize FEC!\n");
                                rc = 1;
                        }

                        // Exit only if we failed because of command line
                        // params, not control port msg
                        if (m_frames_sent == 0ULL) {
                                exit_uv(rc);
                        }

                        m_fec_state = old_fec_state;
                        return new_response(RESPONSE_INT_SERV_ERR, nullptr);
                }
                delete old_fec_state;
                MSG(NOTICE, "Fec changed successfully\n");
        } break;
        case SENDER_MSG_RESET_SSRC: {
                lock_guard<mutex> lock(m_network_devices_lock);
                const uint32_t    old_ssrc   = rtp_my_ssrc(m_network_device);
                auto             *old_device = m_network_device;
                m_network_device             = initialize_network(
                    m_requested_receiver.c_str(), m_recv_port_number,
                    m_send_port_number, m_participants,
                    m_common.force_ip_version,
                    m_common.mcast_if, m_common.ttl);
                if (m_network_device == nullptr) {
                        m_network_device = old_device;
                        MSG(ERROR, "Unable to change SSRC!\n");
                        return new_response(RESPONSE_INT_SERV_ERR, nullptr);
                }
                destroy_rtp_device(old_device);
                MSG(NOTICE,
                    "Changed SSRC from 0x%08" PRIx32 " to "
                    "0x%08" PRIx32 ".\n",
                    old_ssrc, rtp_my_ssrc(m_network_device));
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

rtp_video_rxtx::rtp_video_rxtx(map<string, param_u> const &params) :
        video_rxtx(params), m_fec_state(NULL)
{
        m_participants = pdb_init("video", &video_offset);
        m_requested_receiver = params.at("receiver").str;
        m_recv_port_number = params.at("rx_port").i;
        m_send_port_number = params.at("tx_port").i;

        m_network_device = initialize_network(
            m_requested_receiver.c_str(), m_recv_port_number,
            m_send_port_number, m_participants, m_common.force_ip_version,
            m_common.mcast_if, m_common.ttl);
        if (m_network_device == nullptr) {
                throw ug_runtime_error("Unable to open network",
                                       EXIT_FAIL_NETWORK);
        }

        if ((m_tx = tx_init(&m_sender_mod, m_common.mtu,
                                        TX_MEDIA_VIDEO,
                                        params.at("fec").str,
                                        m_common.encryption,
                                        params.at("bitrate").ll)) == NULL) {
                throw ug_runtime_error("Unable to initialize transmitter", EXIT_FAIL_TRANSMIT);
        }

        // The idea of doing that is to display help on '-f ldgm:help' even if UG would exit
        // immediately. The encoder is actually created by a message.
        check_sender_messages();
}

rtp_video_rxtx::~rtp_video_rxtx()
{
        if (m_tx) {
                tx_done(m_tx);
        }

        m_network_devices_lock.lock();
        destroy_rtp_device(m_network_device);
        m_network_devices_lock.unlock();

        if (m_participants != NULL) {
                pdb_destroy(&m_participants);
        }

        delete m_fec_state;
}

void rtp_video_rxtx::display_buf_increase_warning(int size)
{
        log_msg(LOG_LEVEL_INFO, "\n***\nUnable to set buffer size to %sB.\n",
                format_in_si_units(size));

#if defined _WIN32
        log_msg(LOG_LEVEL_INFO, "See "
                                "https://github.com/CESNET/UltraGrid/wiki/"
                                "Extending-Network-Buffers-%%28Windows%%29 "
                                "for details.\n");
        return;
#endif /* defined _WIN32 */

#ifdef __APPLE__
#define SYSCTL_ENTRY "net.inet.udp.recvspace"
#else
#define SYSCTL_ENTRY "net.core.rmem_max"
#endif
        log_msg(
            LOG_LEVEL_INFO,
            "Please set " SYSCTL_ENTRY " value to %d or greater (see also\n"
            "https://github.com/CESNET/UltraGrid/wiki/OS-Setup-UltraGrid):\n"
#ifdef __APPLE__
            "\tsysctl -w kern.ipc.maxsockbuf=%d\n"
#endif
            "\tsysctl -w " SYSCTL_ENTRY "=%d\n"
            "To make this persistent, add these options (key=value) to "
            "/etc/sysctl.d/60-ultragrid.conf\n"
            "\n***\n\n",
            size,
#ifdef __APPLE__
            size * 4,
#endif /* __APPLE__ */
            size
        );
#undef SYSCTL_ENTRY
}

struct rtp *rtp_video_rxtx::initialize_network(const char *addr, int recv_port,
                int send_port, struct pdb *participants, int force_ip_version,
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
        if (strcmp(addr, IN6_BLACKHOLE_SERVER_MODE_STR) == 0) {
                rtp_set_option(device, RTP_OPT_SEND_BACK, true);
        }

        rtp_set_recv_buf(device, INITIAL_VIDEO_RECV_BUFFER_SIZE);
        rtp_set_send_buf(device, INITIAL_VIDEO_SEND_BUFFER_SIZE);

        pdb_add(participants, rtp_my_ssrc(device));

        return device;
}

void rtp_video_rxtx::destroy_rtp_device(struct rtp *network_device)
{
        rtp_done(network_device);
}

