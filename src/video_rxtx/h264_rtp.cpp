/**
 * @file   video_rxtx/h264_rtp.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author David Cassany    <david.cassany@i2cat.net>
 * @author Ignacio Contreras <ignacio.contreras@i2cat.net>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2013-2024 CESNET, z. s. p. o.
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

#include <cctype>
#include <cstdint>            // for uint32_t
#include <cstdio>             // for printf
#include <cstdlib>
#include <cstring>
#include <memory>

#include "compat/strings.h" // strdupa
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "rtp/rtp.h"
#include "rtsp/rtsp_utils.h"  // for rtsp_types_t
#include "transmit.h"
#include "tv.h"
#include "types.h"            // for video_frame, H264, JPEG
#include "utils/color_out.h"
#include "utils/sdp.h"        // for sdp_print_supported_codecs
#include "video_codec.h"      // for get_codec_name
#include "video_rxtx.hpp"
#include "video_rxtx/h264_rtp.hpp"

constexpr char DEFAULT_RTSP_COMPRESSION[] = "lavc:enc=libx264:safe";
#define MOD_NAME "[vrxtx/h264_rtp] "

using std::shared_ptr;

h264_rtp_video_rxtx::h264_rtp_video_rxtx(std::map<std::string, param_u> const &params,
                int rtsp_port) :
        rtp_video_rxtx(params)
{
        rtsp_params.rtsp_port = (unsigned) rtsp_port;
        rtsp_params.parent = m_common.parent;;
        rtsp_params.avType = static_cast<rtsp_types_t>(params.at("avType").l);
        rtsp_params.rtp_port_video = params.at("rx_port").i;  //server rtp port
}

/**
 * this function is used to configure ther RTSP server either
 * for video-only or using both audio and video. For audio-only
 * RTSP server, the server is run directly from
 * h264_rtp_video_rxtx::set_audio_spec().
 */
void
h264_rtp_video_rxtx::configure_rtsp_server_video()
{
        assert((rtsp_params.avType & rtsp_type_video) != 0);
        switch (rtsp_params.video_codec) {
        case H264:
                tx_send_std = tx_send_h264;
                break;
        case H265:
                tx_send_std = tx_send_h265;
                break;
        case JPEG:
                tx_send_std = tx_send_jpeg;
                break;
        default:
                MSG(ERROR,
                    "codecs other than H.264/H.265 and JPEG currently not "
                    "supported, got %s\n",
                    get_codec_name(rtsp_params.video_codec));
                return;
        }

        if ((rtsp_params.avType & rtsp_type_audio) != 0) {
                if (!audio_params_set) {
                        MSG(INFO, "Waiting for audio specs...\n");
                        return;
                }
        }
        m_rtsp_server = c_start_server(rtsp_params);
}

void
h264_rtp_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame) noexcept
{
        // requestt compress reconfiguration if receivng raw data
        if (!is_codec_opaque(tx_frame->color_spec)) {
                if (!m_sent_compress_change) {
                        send_compess_change(m_common.parent,
                                            DEFAULT_RTSP_COMPRESSION);
                        m_sent_compress_change = true;
                }
                return;
        }

        if (m_rtsp_server == nullptr) {
                rtsp_params.video_codec = tx_frame->color_spec;
                configure_rtsp_server_video();
        }
        if (m_rtsp_server == nullptr) {
                return;
        }

        if (tx_frame->color_spec != rtsp_params.video_codec) {
                MSG(ERROR, "Video codec reconfiguration is not supported!\n");
                return;
        }

        tx_send_std(m_tx, tx_frame.get(), m_network_device);

        if ((m_rxtx_mode & MODE_RECEIVER) == 0) { // send RTCP (receiver thread would otherwise do this
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (curr_time - m_common.start_time) / 100'000 * 9; // at 90000 Hz
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
        free(m_rtsp_server);
}

void h264_rtp_video_rxtx::join()
{
        c_stop_server(m_rtsp_server);
        video_rxtx::join();
}

void
h264_rtp_video_rxtx::set_audio_spec(const struct audio_desc *desc,
                                    int  audio_rx_port, int /* audio_tx_port */,
                                    bool /* ipv6 */)
{
        rtsp_params.adesc = *desc;
        rtsp_params.rtp_port_audio = audio_rx_port;
        audio_params_set = true;

        if ((rtsp_params.avType & rtsp_type_video) == 0U) {
                m_rtsp_server = c_start_server(rtsp_params);
        }
}

static void rtps_server_usage(){
        printf("\n[RTSP SERVER] usage:\n");
        color_printf("\t" TBOLD("-x rtsp[:port=number]") "\n");
        printf("\t\tdefault rtsp server port number: 8554\n\n");

        sdp_print_supported_codecs();
}

static int get_rtsp_server_port(const char *config) {
        if (strncmp(config, "port:", 5) != 0 &&
            strncmp(config, "port=", 5) != 0) {
                log_msg(LOG_LEVEL_ERROR, "\n[RTSP SERVER] ERROR - please, check usage.\n");
                rtps_server_usage();
                return -1;
        }
        if (strlen(config) == 5) {
                log_msg(LOG_LEVEL_ERROR, "\n[RTSP SERVER] ERROR - please, enter a port number.\n");
                rtps_server_usage();
                return -1;
        }
        if (config[4] == ':') {
                MSG(WARNING, "deprecated usage - use port=number, not port:number!\n");
        }
        int port = atoi(config + 5);
        if (port < 0 || port > 65535 || !isdigit(config[5])) {
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
                if (strcmp(rtsp_port_str, "help") == 0) {
                        rtps_server_usage();
                        return nullptr;
                }
                rtsp_port = get_rtsp_server_port(rtsp_port_str);
                if (rtsp_port == -1) {
                        return nullptr;
                }
        }
        return new h264_rtp_video_rxtx(params, rtsp_port);
}

static const struct video_rxtx_info h264_video_rxtx_info = {
        "RTP standard (using RTSP)",
        create_video_rxtx_h264_std
};

REGISTER_MODULE(rtsp, &h264_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

