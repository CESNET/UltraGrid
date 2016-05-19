/**
 * @file   video_rxtx/rtp.h
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

#ifndef VIDEO_RXTX_RTP_H_
#define VIDEO_RXTX_RTP_H_

#include "video_rxtx.h"

#include <chrono>
#include <mutex>
#include <string>

#ifdef HAVE_MACOSX
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  5944320
#else
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((4*1920*1080)*110/100)
#endif

#define INITIAL_VIDEO_SEND_BUFFER_SIZE  (1024*1024)

struct rtp;
struct fec;

class rtp_video_rxtx : public video_rxtx {
        friend class video_rxtx;
public:
        rtp_video_rxtx(std::map<std::string, param_u> const &params);
        virtual ~rtp_video_rxtx();

        static struct rtp **initialize_network(const char *addrs, int recv_port_base,
                        int send_port_base, struct pdb *participants, bool use_ipv6,
                        const char *mcast_if);
        void destroy_rtp_devices(struct rtp ** network_devices);
        static void display_buf_increase_warning(int size);

protected:
        int m_connections_count;
        struct rtp **m_network_devices; // ULTRAGRID_RTP
        std::mutex m_network_devices_lock;
        struct tx *m_tx;
        struct pdb *m_participants;
        std::string      m_requested_receiver;
        int              m_recv_port_number;
        int              m_send_port_number;
        bool             m_ipv6;
        const char      *m_requested_mcast_if;
        fec             *m_fec_state;
        const std::chrono::steady_clock::time_point m_start_time;
        video_desc       m_video_desc;
private:
        struct response *process_sender_message(struct msg_sender *i, int *status);
};

#endif // VIDEO_RXTX_RTP_H_

