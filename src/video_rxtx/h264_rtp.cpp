/**
 * @file   video_rxtx/h264_rtp.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author David Cassany    <david.cassany@i2cat.net>
 * @author Ignacio Contreras <ignacio.contreras@i2cat.net>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2013-2014 CESNET, z. s. p. o.
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

#include "transmit.h"
#include "video_rxtx/h264_rtp.h"
#include "video.h"

h264_rtp_video_rxtx::h264_rtp_video_rxtx(struct module *parent, struct video_export *video_exporter,
                const char *requested_compression, const char *requested_encryption,
                const char *receiver, int rx_port, int tx_port,
                bool use_ipv6, const char *mcast_if, const char *requested_video_fec, int mtu,
                long packet_rate, uint8_t avType) :
        rtp_video_rxtx(parent, video_exporter, requested_compression, requested_encryption,
                        receiver, rx_port, tx_port,
                        use_ipv6, mcast_if, requested_video_fec, mtu, packet_rate)
{
#ifdef HAVE_RTSP_SERVER
        m_rtsp_server = init_rtsp_server(0, parent, avType); //port, root_module, avType
        c_start_server(m_rtsp_server);
#endif
}

void h264_rtp_video_rxtx::send_frame(struct video_frame *tx_frame)
{
        if (m_connections_count == 1) { /* normal/default case - only one connection */
            tx_send_h264(m_tx, tx_frame, m_network_devices[0]);
        } else {
            //TODO to be tested, the idea is to reply per destiny
                for (int i = 0; i < m_connections_count; ++i) {
                    tx_send_h264(m_tx, tx_frame,
                                        m_network_devices[i]);
                }
        }
        VIDEO_FRAME_DISPOSE(tx_frame);
}

h264_rtp_video_rxtx::~h264_rtp_video_rxtx()
{
#ifdef HAVE_RTSP_SERVER
        c_stop_server(m_rtsp_server);
#endif
}

