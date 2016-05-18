/**
 * @file   video_rxtx/rtp.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2014 CESNET z.s.p.o.
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


#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video_rxtx/rtp.h"

#include "debug.h"

#include <sstream>
#include <string>
#include <stdexcept>

#include "host.h"
#include "ihdtv.h"
#include "messaging.h"
#include "module.h"
#include "pdb.h"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/video_decoders.h"
#include "rtp/pbuf.h"
#include "rtp/rtp_callback.h"
#include "tfrc.h"
#include "transmit.h"
#include "tv.h"
#include "ug_runtime_error.h"
#include "utils/vf_split.h"
#include "video.h"
#include "video_compress.h"
#include "video_decompress.h"
#include "video_display.h"
#include "video_export.h"
#include "video_rxtx.h"

using namespace std;

struct response *rtp_video_rxtx::process_sender_message(struct msg_sender *msg)
{
        switch(msg->type) {
                case SENDER_MSG_CHANGE_RECEIVER:
                        {
                                assert(m_rxtx_mode == MODE_SENDER); // sender only
                                lock_guard<mutex> lock(m_network_devices_lock);
                                auto old_devices = m_network_devices;
                                auto old_receiver = m_requested_receiver;
                                m_requested_receiver = msg->receiver;
                                m_network_devices = initialize_network(m_requested_receiver.c_str(),
                                                m_recv_port_number,
                                                m_send_port_number, m_participants, m_ipv6,
                                                m_requested_mcast_if);
                                if (!m_network_devices) {
                                        m_network_devices = old_devices;
                                        m_requested_receiver = old_receiver;
                                        log_msg(LOG_LEVEL_ERROR, "[control] Failed receiver to %s.\n",
                                                        msg->receiver);
                                        return new_response(RESPONSE_INT_SERV_ERR, "Changing receiver failed!");
                                } else {
                                        log_msg(LOG_LEVEL_NOTICE, "[control] Changed receiver to %s.\n",
                                                        msg->receiver);
                                        destroy_rtp_devices(old_devices);
                                }
                        }
                        break;
                case SENDER_MSG_CHANGE_PORT:
                        {
                                assert(m_rxtx_mode == MODE_SENDER); // sender only
                                lock_guard<mutex> lock(m_network_devices_lock);
                                auto old_devices = m_network_devices;
                                auto old_port = m_send_port_number;

                                m_send_port_number = msg->tx_port;
                                if (msg->rx_port) {
                                        m_recv_port_number = msg->rx_port;
                                }
                                m_network_devices = initialize_network(m_requested_receiver.c_str(), m_recv_port_number,
                                                m_send_port_number, m_participants, m_ipv6,
                                                m_requested_mcast_if);

                                if (!m_network_devices) {
                                        m_network_devices = old_devices;
                                        m_send_port_number = old_port;
                                        log_msg(LOG_LEVEL_ERROR, "[control] Failed to Change TX port to %d.\n",
                                                        msg->tx_port);
                                        return new_response(RESPONSE_INT_SERV_ERR, "Changing TX port failed!");
                                } else {
                                        log_msg(LOG_LEVEL_NOTICE, "[control] Changed TX port to %d.\n",
                                                        msg->tx_port);
                                        destroy_rtp_devices(old_devices);
                                }
                        }
                        break;
                case SENDER_MSG_PAUSE:
                        {
                                lock_guard<mutex> lock(m_network_devices_lock);
                                log_msg(LOG_LEVEL_ERROR, "[control] Paused.\n");
                                m_paused = true;
                                break;
                        }
                case SENDER_MSG_PLAY:
                        {
                                lock_guard<mutex> lock(m_network_devices_lock);
                                log_msg(LOG_LEVEL_ERROR, "[control] Playing again.\n");
                                m_paused = false;
                                break;
                        }
                case SENDER_MSG_CHANGE_FEC:
                        {
                                lock_guard<mutex> lock(m_network_devices_lock);
                                auto old_fec_state = m_fec_state;
                                m_fec_state = NULL;
                                if (strcmp(msg->fec_cfg, "flush") == 0) {
                                        delete old_fec_state;
                                } else {
                                        int ret = -1;
                                        try {
                                                m_fec_state = fec::create_from_config(msg->fec_cfg);
                                        } catch (int i) {
                                                ret = i;
                                        } catch (...) {
                                        }
                                        if (!m_fec_state) {
                                                if (ret != 0) {
                                                        log_msg(LOG_LEVEL_ERROR, "[control] Unable to initalize FEC!\n");
                                                        m_fec_state = old_fec_state;
                                                } else { // -f LDGM:help or so
                                                        exit_uv(0);
                                                }
                                                return new_response(RESPONSE_INT_SERV_ERR, NULL);
                                        } else {
                                                delete old_fec_state;
                                                log_msg(LOG_LEVEL_NOTICE, "[control] Fec changed successfully\n");
                                        }
                                }
                        }
                        break;
                case SENDER_MSG_QUERY_VIDEO_MODE:
                        if (!m_video_desc) {
                                return new_response(RESPONSE_NO_CONTENT, NULL);
                        } else {
                                ostringstream oss;
                                oss << m_video_desc;
                                return new_response(RESPONSE_OK, oss.str().c_str());
                        }
                        break;
                case SENDER_MSG_RESET_SSRC:
                        {
                                lock_guard<mutex> lock(m_network_devices_lock);
                                uint32_t old_ssrc = rtp_my_ssrc(m_network_devices[0]);
                                auto old_devices = m_network_devices;
                                m_network_devices = initialize_network(m_requested_receiver.c_str(),
                                                m_recv_port_number,
                                                m_send_port_number, m_participants, m_ipv6,
                                                m_requested_mcast_if);
                                if (!m_network_devices) {
                                        m_network_devices = old_devices;
                                        log_msg(LOG_LEVEL_ERROR, "[control] Unable to change SSRC!\n");
                                        return new_response(RESPONSE_INT_SERV_ERR, NULL);
                                } else {
                                        destroy_rtp_devices(old_devices);
                                        log_msg(LOG_LEVEL_NOTICE, "[control] Changed SSRC from 0x%08lx to "
                                                        "0x%08lx.\n", old_ssrc, rtp_my_ssrc(m_network_devices[0]));
                                }
                        }
                        break;
        }

        return new_response(RESPONSE_OK, NULL);
}

rtp_video_rxtx::rtp_video_rxtx(map<string, param_u> const &params) :
        video_rxtx(params), m_fec_state(NULL), m_start_time(*(const std::chrono::steady_clock::time_point *) params.at("start_time").ptr), m_video_desc{}
{
        m_participants = pdb_init((volatile int *) params.at("video_delay").ptr);
        m_requested_receiver = (const char *) params.at("receiver").ptr;
        m_recv_port_number = params.at("rx_port").i;
        m_send_port_number = params.at("tx_port").i;
        m_ipv6 = params.at("use_ipv6").b;
        m_requested_mcast_if = (const char *) params.at("mcast_if").ptr;

        if ((m_network_devices = initialize_network(m_requested_receiver.c_str(), m_recv_port_number, m_send_port_number,
                                        m_participants, m_ipv6, m_requested_mcast_if))
                        == NULL) {
                throw ug_runtime_error("Unable to open network", EXIT_FAIL_NETWORK);
        } else {
                struct rtp **item;
                m_connections_count = 0;
                /* only count how many connections has initialize_network opened */
                for(item = m_network_devices; *item != NULL; ++item)
                        ++m_connections_count;
        }

        if ((m_tx = tx_init(&m_sender_mod,
                                        params.at("mtu").i, TX_MEDIA_VIDEO,
                                        static_cast<const char *>(params.at("fec").ptr),
                                        static_cast<const char *>(params.at("encryption").ptr),
                                        params.at("packet_rate").i)) == NULL) {
                throw ug_runtime_error("Unable to initialize transmitter", EXIT_FAIL_TRANSMIT);
        }

        // The idea of doing that is to display help on '-f ldgm:help' even if UG would exit
        // immediatelly. The encoder is actually created by a message.
        check_sender_messages();
}

rtp_video_rxtx::~rtp_video_rxtx()
{
        if (m_tx) {
                module_done(CAST_MODULE(m_tx));
        }

        m_network_devices_lock.lock();
        destroy_rtp_devices(m_network_devices);
        m_network_devices_lock.unlock();

        if (m_participants != NULL) {
                pdb_destroy(&m_participants);
        }

        delete m_fec_state;
}

void rtp_video_rxtx::display_buf_increase_warning(int size)
{
        fprintf(stderr, "\n***\n"
                        "Unable to set buffer size to %d B.\n"
#if defined WIN32
                        "See https://www.sitola.cz/igrid/index.php/Extending_Network_Buffers_%%28Windows%%29 for details.\n",
#else
                        "Please set net.core.rmem_max value to %d or greater. (see also\n"
                        "https://www.sitola.cz/igrid/index.php/OS_Setup_UltraGrid)\n"
#ifdef HAVE_MACOSX
                        "\tsysctl -w kern.ipc.maxsockbuf=%d\n"
                        "\tsysctl -w net.inet.udp.recvspace=%d\n"
#else
                        "\tsysctl -w net.core.rmem_max=%d\n"
#endif
                        "To make this persistent, add these options (key=value) to /etc/sysctl.conf\n"
                        "\n***\n\n",
                        size, size,
#ifdef HAVE_MACOSX
                        size * 4,
#endif /* HAVE_MACOSX */
#endif /* ! defined WIN32 */
                        size);

}

struct rtp **rtp_video_rxtx::initialize_network(const char *addrs, int recv_port_base,
                int send_port_base, struct pdb *participants, bool use_ipv6,
                const char *mcast_if)
{
        struct rtp **devices = NULL;
        double rtcp_bw = 5 * 1024 * 1024;       /* FIXME */
        int ttl = 255;
        char *saveptr = NULL;
        char *addr;
        char *tmp;
        int required_connections, index;
        int recv_port = recv_port_base;
        int send_port = send_port_base;

        tmp = strdup(addrs);
        if(strtok_r(tmp, ",", &saveptr) == NULL) {
                free(tmp);
                return NULL;
        }
        else required_connections = 1;
        while(strtok_r(NULL, ",", &saveptr) != NULL)
                ++required_connections;

        free(tmp);
        tmp = strdup(addrs);

        devices = (struct rtp **)
                malloc((required_connections + 1) * sizeof(struct rtp *));

        for(index = 0, addr = strtok_r(tmp, ",", &saveptr);
                index < required_connections;
                ++index, addr = strtok_r(NULL, ",", &saveptr), recv_port += 2, send_port += 2)
        {
                /* port + 2 is reserved for audio */
                if (recv_port == recv_port_base + 2)
                        recv_port += 2;
                if (send_port == send_port_base + 2)
                        send_port += 2;

                devices[index] = rtp_init_if(addr, mcast_if, recv_port,
                                send_port, ttl, rtcp_bw, FALSE,
                                rtp_recv_callback, (uint8_t *)participants,
                                use_ipv6, true);
                if (devices[index] != NULL) {
                        rtp_set_option(devices[index], RTP_OPT_WEAK_VALIDATION,
                                TRUE);
                        rtp_set_option(devices[index], RTP_OPT_PROMISC,
                                TRUE);
                        rtp_set_sdes(devices[index], rtp_my_ssrc(devices[index]),
                                RTCP_SDES_TOOL,
                                PACKAGE_STRING, strlen(PACKAGE_STRING));

                        int size = INITIAL_VIDEO_RECV_BUFFER_SIZE;
                        int ret = rtp_set_recv_buf(devices[index], INITIAL_VIDEO_RECV_BUFFER_SIZE);
                        if(!ret) {
                                display_buf_increase_warning(size);
                        }

                        rtp_set_send_buf(devices[index], INITIAL_VIDEO_SEND_BUFFER_SIZE);

                        pdb_add(participants, rtp_my_ssrc(devices[index]));
                }
                else {
                        int index_nest;
                        for(index_nest = 0; index_nest < index; ++index_nest) {
                                rtp_done(devices[index_nest]);
                        }
                        free(devices);
                        devices = NULL;
                }
        }
        if(devices != NULL) devices[index] = NULL;
        free(tmp);

        return devices;
}

void rtp_video_rxtx::destroy_rtp_devices(struct rtp ** network_devices)
{
        struct rtp ** current = network_devices;
        if(!network_devices)
                return;
        while(*current != NULL) {
                rtp_done(*current++);
        }
        free(network_devices);
}

