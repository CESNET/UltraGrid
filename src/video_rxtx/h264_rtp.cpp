/**
 * @file   video_rxtx/h264_rtp.cpp
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
/**
 * @file
 * @note
 * Currently incompatible with upstream version of live555. Works with older
 * version from https://github.com/xanview/live555/, commit 35c375 (live555
 * version from 7th Aug 2015).
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "compat/misc.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "transmit.h"
#include "tv.h"
#include "rtp/rtp.h"
#include "rtp/rtpenc_h264.h"
#include "utils/color_out.h"
#include "video_rxtx.hpp"
#include "video_rxtx/h264_rtp.hpp"
#include "video.h"

using namespace std;

h264_rtp_video_rxtx::h264_rtp_video_rxtx(std::map<std::string, param_u> const &params,
                int rtsp_port) :
        rtp_video_rxtx(params)
        #ifdef HAVE_RTSP_SERVER
        , m_rtsp_server(rtsp_port,
                        static_cast<struct module *>(params.at("parent").ptr),
                        static_cast<rtsp_media_type_t>(params.at("media_type").l),
                        static_cast<audio_codec_t>(params.at("audio_codec").l),
                        params.at("audio_sample_rate").i, params.at("audio_channels").i,
                        params.at("audio_bps").i, params.at("rx_port").i, params.at("a_rx_port").i)
        #endif // HAVE_RTSP_SERVER
{
#ifdef HAVE_RTSP_SERVER
        m_rtsp_server.start_server();
#endif // HAVE_RTSP_SERVER
}

void
h264_rtp_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame) noexcept
{
        tx_send_h264(m_tx, tx_frame.get(), m_network_device);
        if ((m_rxtx_mode & MODE_RECEIVER) == 0) { // send RTCP (receiver thread would otherwise do this
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (curr_time - m_start_time) / 100'000 * 9; // at 90000 Hz
                rtp_update(m_network_device, curr_time);
                rtp_send_ctrl(m_network_device, ts, nullptr, curr_time);

                // receive RTCP
                struct timeval timeout;
                timeout.tv_sec = 0;
                timeout.tv_usec = 0;
                rtp_recv_r(m_network_device, &timeout, ts);
        }
}

h264_rtp_video_rxtx::~h264_rtp_video_rxtx()
{
}

static void rtps_server_usage(){
        printf("\n[RTSP SERVER] usage:\n");
        color_printf("\t" TBOLD("--video-protocol rtsp[=port:number]") "\n");
        printf("\t\tdefault rtsp server port number: 8554\n\n");
}

static int get_rtsp_server_port(const char *cconfig) {
        char *save_ptr = NULL;
        char *config = strdupa(cconfig);
        char *tok = strtok_r(config, ":", &save_ptr);
        if (!tok || strcmp(tok,"port") != 0) {
                log_msg(LOG_LEVEL_ERROR, "\n[RTSP SERVER] ERROR - please, check usage.\n");
                rtps_server_usage();
                return -1;
        }
        if (!(tok = strtok_r(NULL, ":", &save_ptr))) {
                log_msg(LOG_LEVEL_ERROR, "\n[RTSP SERVER] ERROR - please, enter a port number.\n");
                rtps_server_usage();
                return -1;
        }
        int port = atoi(tok);
        if (port < 0 || port > 65535) {
                log_msg(LOG_LEVEL_ERROR, "\n[RTSP SERVER] ERROR - please, enter a valid port number.\n");
                rtps_server_usage();
                return -1;
        }
        return port;
}

static video_rxtx *create_video_rxtx_h264_std(std::map<std::string, param_u> const &params)
{
        int rtsp_port;
        const char *rtsp_port_str = static_cast<const char *>(params.at("opts").ptr);
        if (strlen(rtsp_port_str) == 0) {
                rtsp_port = 0;
        } else {
                if (!strcmp(rtsp_port_str, "help")) {
#ifdef HAVE_RTSP_SERVER
                        rtps_server_usage();
#endif
                        return 0;
                } else {
                        rtsp_port = get_rtsp_server_port(rtsp_port_str);
                        if (rtsp_port == -1) return 0;
                }
        }
        return new h264_rtp_video_rxtx(params, rtsp_port);
}

static const struct video_rxtx_info h264_video_rxtx_info = {
        "H264 standard",
        create_video_rxtx_h264_std
};

REGISTER_MODULE(rtsp, &h264_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

