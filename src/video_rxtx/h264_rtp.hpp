/**
 * @file   video_rxtx/h264_rtp.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author David Cassany    <david.cassany@i2cat.net>
 * @author Ignacio Contreras <ignacio.contreras@i2cat.net>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2013-2023 CESNET, z. s. p. o.
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

#ifndef VIDEO_RXTX_H264_RTP_H_
#define VIDEO_RXTX_H264_RTP_H_

#include <atomic>                        // for atomic_bool
#include <map>                           // for map
#include <memory>                        // for shared_ptr
#include <string>                        // for string

#include "rtsp/c_basicRTSPOnlyServer.h"
#include "video_rxtx/rtp.hpp"

union param_u;
struct video_frame;

class h264_rtp_video_rxtx : public rtp_video_rxtx {
public:
        h264_rtp_video_rxtx(std::map<std::string, param_u> const &, int);
        virtual ~h264_rtp_video_rxtx();
        void join() override;
        void set_audio_spec(const struct audio_desc *desc, int audio_rx_port,
                            int audio_tx_port, bool ipv6) override;

private:
        virtual void send_frame(std::shared_ptr<video_frame>) noexcept override;
        virtual void *(*get_receiver_thread() noexcept)(void *arg) override {
                return nullptr;
        }
        void                          configure_rtsp_server_video();
        struct rtsp_server_parameters rtsp_params{};
        std::atomic<bool>             audio_params_set = false;
        rtsp_serv_t                  *m_rtsp_server    = nullptr;
        void (*tx_send_std)(struct tx *tx_session, struct video_frame *frame,
                            struct rtp *rtp_session) = nullptr;

        bool m_sent_compress_change = false;
};

#endif // VIDEO_RXTX_H264_RTP_H_

