/**
 * @file   video_rxtx/h264_sdp.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author David Cassany    <david.cassany@i2cat.net>
 * @author Ignacio Contreras <ignacio.contreras@i2cat.net>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2013-2018 CESNET, z. s. p. o.
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

#ifndef VIDEO_RXTX_H264_SDP_H_
#define VIDEO_RXTX_H264_SDP_H_

#include <string>

#include "utils/sdp.h"
#include "video_rxtx.h"
#include "video_rxtx/rtp.h"

class h264_sdp_video_rxtx : public rtp_video_rxtx {
public:
        h264_sdp_video_rxtx(std::map<std::string, param_u> const &);
        virtual ~h264_sdp_video_rxtx();
private:
        static void change_address_callback(void *udata, const char *address);
        virtual void send_frame(std::shared_ptr<video_frame>);
        virtual void *(*get_receiver_thread())(void *arg) {
                return NULL;
        }
        void sdp_add_video(codec_t codec);
        struct sdp *m_sdp;
        codec_t m_sdp_configured_codec = VIDEO_CODEC_NONE;
        int m_saved_tx_port;
        std::string m_requested_file;
        int m_requested_http_port = DEFAULT_SDP_HTTP_PORT;
        bool m_sent_compress_change = false;

        bool m_autorun = false;
        std::string m_saved_addr; ///< for dynamic address reconfiguration, @see m_autorun
};

#endif // VIDEO_RXTX_H264_SDP_H_

