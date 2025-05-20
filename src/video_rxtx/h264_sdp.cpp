/**
 * @file   video_rxtx/h264_sdp.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author David Cassany    <david.cassany@i2cat.net>
 * @author Ignacio Contreras <ignacio.contreras@i2cat.net>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2013-2024 CESNET
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

#include "video_rxtx/h264_sdp.hpp"

#include <cstdint>               // for uint32_t
#include <cstdlib>               // for abort
#include <cstring>               // for strncpy
#include <exception>             // for exception
#include <iostream>

#include "audio/audio.h"
#include "audio/types.h"         // for audio_desc
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"           // for msg_change_compress_data, free_response
#include "module.h"
#include "rtp/rtp.h"
#include "transmit.h"
#include "tv.h"
#include "ug_runtime_error.hpp"
#include "utils/sdp.h"
#include "video_codec.h"         // for is_codec_opaque
#include "video_rxtx.hpp"

#define DEFAULT_SDP_COMPRESSION "lavc:codec=MJPEG:safe"
#define MOD_NAME "[vrxtx/sdp] "

using std::cout;
using std::exception;
using std::shared_ptr;
using std::string;

h264_sdp_video_rxtx::h264_sdp_video_rxtx(std::map<std::string, param_u> const &params)
        : rtp_video_rxtx(params)
{
        auto opts = params.at("opts").str;
        LOG(LOG_LEVEL_WARNING) << "Warning: SDP support is experimental only. Things may be broken - feel free to report them but the support may be limited.\n";
        m_saved_addr = m_requested_receiver;
        m_saved_tx_port = params.at("tx_port").i;
        if (int ret = sdp_set_options(opts)) {
                throw ret == 1 ? 0 : 1;
        }
}

void h264_sdp_video_rxtx::change_address_callback(void *udata, const char *address)
{
        auto *s = static_cast<h264_sdp_video_rxtx *>(udata);
        if (s->m_saved_addr == address) {
                return;
        }
        s->m_saved_addr = address;

        constexpr enum module_class path_sender[] = { MODULE_CLASS_SENDER,
                                                      MODULE_CLASS_NONE };
        sdp_send_change_address_message(get_root_module(s->m_common.parent),
                                        path_sender, address);
}

void h264_sdp_video_rxtx::sdp_add_video(codec_t codec)
{
        const int rc = ::sdp_add_video(
            rtp_is_ipv6(m_network_device), m_saved_tx_port, codec,
            h264_sdp_video_rxtx::change_address_callback, this);
        if (rc == -2) {
                throw ug_runtime_error("[SDP] Unsupported video codec for SDP (allowed H.264 and JPEG)!\n");
        }
        if (rc != 0) {
		abort();
	}
}

/**
 * @note
 * This function sets compression just after first frame is received. The delayed initialization is to allow devices
 * producing H.264/JPEG natively (eg. v4l2) to be passed untouched to transport. Fallback H.264 is applied when uncompressed
 * stream is detected.
 */
void
h264_sdp_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame) noexcept
{
        if (!is_codec_opaque(tx_frame->color_spec)) {
		if (m_sent_compress_change) {
			return;
		}
		send_compess_change(m_common.parent, DEFAULT_SDP_COMPRESSION);
		m_sent_compress_change = true;
		return;
        }
        if (m_sdp_configured_codec == VIDEO_CODEC_NONE) {
                try {
                        sdp_add_video(tx_frame->color_spec);
                        m_sdp_configured_codec = tx_frame->color_spec;
                } catch (exception const &e) {
                        LOG(LOG_LEVEL_ERROR) << e.what();
                        exit_uv(1);
                        return;
                }
        }

        if (m_sdp_configured_codec != tx_frame->color_spec) {
                LOG(LOG_LEVEL_ERROR) << "[SDP] Video codec reconfiguration is not supported!\n";
                return;
        }

        if (m_sdp_configured_codec == H264) {
                tx_send_h264(m_tx, tx_frame.get(), m_network_device);
        } else {
                tx_send_jpeg(m_tx, tx_frame.get(), m_network_device);
        }
        if ((m_rxtx_mode & MODE_RECEIVER) ==
            0) { // send RTCP (receiver thread would otherwise do this)
                uint32_t ts = get_std_video_local_mediatime();
                time_ns_t curr_time = get_time_in_ns();
                rtp_update(m_network_device, curr_time);
                rtp_send_ctrl(m_network_device, ts, nullptr, curr_time);

                // receive RTCP
                struct timeval timeout;
                timeout.tv_sec = 0;
                timeout.tv_usec = 0;
                rtp_recv_r(m_network_device, &timeout, ts);
        }
}

static void
sdp_change_address_callback(void *udata, const char *address)
{
        const enum module_class path_sender[] = { MODULE_CLASS_AUDIO,
                                                  MODULE_CLASS_SENDER,
                                                  MODULE_CLASS_NONE };
        sdp_send_change_address_message((module *) udata, path_sender, address);
}

void
h264_sdp_video_rxtx::set_audio_spec(const struct audio_desc *desc,
                                    int /* audio_rx_port */, int audio_tx_port, bool ipv6)
{
        if (sdp_add_audio(ipv6, audio_tx_port,
                          desc->sample_rate,
                          desc->ch_count, desc->codec,
                          sdp_change_address_callback,
                          get_root_module(m_common.parent)) != 0) {
                MSG(ERROR, "Cannot add audio to SDP!\n");
        }
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

