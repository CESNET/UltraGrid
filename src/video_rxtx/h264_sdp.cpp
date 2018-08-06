/**
 * @file   video_rxtx/h264_rtp.h
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "transmit.h"
#include "rtp/rtp.h"
#include "rtp/rtpenc_h264.h"
#include "utils/sdp.h"
#include "video_rxtx.h"
#include "video_rxtx/h264_sdp.h"
#include "video.h"

using namespace std;

h264_sdp_video_rxtx::h264_sdp_video_rxtx(std::map<std::string, param_u> const &params)
        : rtp_video_rxtx(params)
{
        log_msg(LOG_LEVEL_WARNING, "Warning: SDP support is experimental only. Things may be broken - feel free to report them but the support may be limited.\n");
        m_sdp = new_sdp(std_H264, params.at("tx_port").i);
        if (params.at("a_tx_port").i) {
                new_stream(m_sdp);
                sprintf(m_sdp->stream[1].media_info, "m=audio %d RTP/AVP 97\n", params.at("a_tx_port").i);
                const char *audio_codec = NULL;
                switch (params.at("audio_codec").l) {
                        case AC_ALAW:
                                audio_codec = "PCMA";
                                break;
                        case AC_MULAW:
                                audio_codec = "PCMU";
                                break;
                        case AC_OPUS:
                                audio_codec = "OPUS";
                                break;
                }

                assert(audio_codec);

                sprintf(m_sdp->stream[1].rtpmap, "a=rtpmap:97 %s/%i/%i", audio_codec,
                                params.at("audio_sample_rate").i, params.at("audio_channels").i);
        }
        if (m_sdp == NULL) {
                throw string("[SDP] File creation failed\n");
        }
        if (!get_sdp(m_sdp)){
                throw string("[SDP] File creation failed\n");
        }
}

void h264_sdp_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame)
{
        if (m_connections_count == 1) { /* normal/default case - only one connection */
            tx_send_h264(m_tx, tx_frame.get(), m_network_devices[0]);
        } else {
            //TODO to be tested, the idea is to reply per destiny
                for (int i = 0; i < m_connections_count; ++i) {
                    tx_send_h264(m_tx, tx_frame.get(),
                                        m_network_devices[i]);
                }
        }
        if ((m_rxtx_mode & MODE_RECEIVER) == 0) { // send RTCP (receiver thread would otherwise do this
                struct timeval curr_time;
                uint32_t ts;
                gettimeofday(&curr_time, NULL);
                ts = std::chrono::duration_cast<std::chrono::duration<double>>(m_start_time - std::chrono::steady_clock::now()).count() * 90000;
                rtp_update(m_network_devices[0], curr_time);
                rtp_send_ctrl(m_network_devices[0], ts, 0, curr_time);

                // receive RTCP
                struct timeval timeout;
                timeout.tv_sec = 0;
                timeout.tv_usec = 0;
                rtp_recv_r(m_network_devices[0], &timeout, ts);
        }
}

h264_sdp_video_rxtx::~h264_sdp_video_rxtx()
{
        clean_sdp(m_sdp);
}

static video_rxtx *create_video_rxtx_h264_sdp(std::map<std::string, param_u> const &params)
{
        return new h264_sdp_video_rxtx(params);
}

static const struct video_rxtx_info h264_sdp_video_rxtx_info = {
        "RTP standard (SDP version)",
        create_video_rxtx_h264_sdp
};

REGISTER_MODULE(sdp, &h264_sdp_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

