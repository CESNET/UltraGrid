/**
 * @file   video_rxtx/h264_rtp.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author David Cassany    <david.cassany@i2cat.net>
 * @author Ignacio Contreras <ignacio.contreras@i2cat.net>
 * @author Gerard Castillo  <gerard.castillo@i2cat.net>
 */
/*
 * Copyright (c) 2013-2014 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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

#include <atomic>                        // for atomic
#include <cassert>            // for assert
#include <cctype>
#include <cstdint>            // for uint32_t
#include <cstdio>             // for printf
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>                       // for move

#include "audio/types.h"                 // for audio_desc
#include "audio/utils.h"                 // for audio_desc_to_cstring
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "rtp/rtp.h"
#include "rtsp/c_basicRTSPOnlyServer.h"  // for rtsp_server_parameters, c_st...
#include "rtsp/rtsp_utils.h"  // for rtsp_types_t
#include "transmit.h"
#include "tv.h"
#include "types.h"            // for video_frame, H264, JPEG
#include "utils/color_out.h"
#include "utils/sdp.h"        // for sdp_print_supported_codecs
#include "video_codec.h"      // for get_codec_name
#include "video_rxtx.h"
#include "video_rxtx/rtp_common.h"

constexpr char DEFAULT_RTSP_COMPRESSION[] = "lavc:enc=libx264:safe";
#define MOD_NAME "[vrxtx/h264_rtp] "
constexpr uint32_t MAGIC = to_fourcc('V', 'X', 'h', 'r');

using std::shared_ptr;

struct h264_rtp_video_rxtx {
        uint32_t magic = MAGIC;
        h264_rtp_video_rxtx(const struct vrxtx_params *params,
                            const struct common_opts  *common, int rtsp_port);
        ~h264_rtp_video_rxtx();
        void join();
        void send_frame(std::shared_ptr<video_frame>) noexcept;

        struct rtp_rxtx_common *m_rtp_common;
        void                          configure_rtsp_server_video();
        struct rtsp_server_parameters rtsp_params{};
        std::atomic<bool>             audio_params_set = false;
        rtsp_serv_t                  *m_rtsp_server    = nullptr;
        void (*tx_send_std)(struct tx *tx_session, struct video_frame *frame,
                            struct rtp *rtp_session) = nullptr;

        bool m_sent_compress_change = false;
        struct module *m_parent;
        time_ns_t      m_start_time;
};

h264_rtp_video_rxtx::h264_rtp_video_rxtx(const struct vrxtx_params *params,
                            const struct common_opts *common, int rtsp_port) :
        m_parent(common->parent),
        m_start_time(common->start_time)
{
        rtsp_params.rtsp_port = (unsigned) rtsp_port;
        rtsp_params.parent = common->parent;

        auto avType = (rtsp_types_t) (SENDS_MEDIUM(params, TX_MEDIA_AUDIO)
                                          ? rtsp_type_audio
                                          : 0);
        avType = (rtsp_types_t) (avType | (SENDS_MEDIUM(params, TX_MEDIA_VIDEO)
                                               ? rtsp_type_video
                                               : 0));
        if (avType == rtsp_type_none) {
                printf("[RTSP SERVER CHECK] no stream type... check capture devices input...\n");
                throw -1;
        }
        rtsp_params.avType = avType;;

        rtsp_params.rtp_audio_src_port = params->medium[TX_MEDIA_AUDIO].rx_port;
        rtsp_params.rtp_video_src_port = params->medium[TX_MEDIA_VIDEO].rx_port;
        m_rtp_common                   = rtp_rxtx_common_init(params, common);
        if (m_rtp_common == nullptr) {
                throw -1;
        }
}

/**
 * this function is used to configure their RTSP server either
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
        struct rtp_rxtx_medium *video = &m_rtp_common->medium[TX_MEDIA_VIDEO];

        rtp_rxtx_sender_do_housekeeping(m_rtp_common, TX_MEDIA_VIDEO);
        // requestt compress reconfiguration if receivng raw data
        if (!is_codec_opaque(tx_frame->color_spec)) {
                if (!m_sent_compress_change) {
                        send_compess_change(m_parent,
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

        tx_send_std(video->tx, tx_frame.get(), video->network_device);

        if (video->rxtx_mode & MODE_RECEIVER) { // send RTCP (receiver thread would otherwise do this
                time_ns_t curr_time = get_time_in_ns();
                uint32_t ts = (curr_time - m_start_time) / 100'000 * 9; // at 90000 Hz
                rtp_update(video->network_device, curr_time);
                rtp_send_ctrl(video->network_device, ts, nullptr, curr_time);

                // receive RTCP
                struct timeval timeout;
                timeout.tv_sec = 0;
                timeout.tv_usec = 0;
                rtp_recv_r(video->network_device, &timeout, ts);
        }
}

h264_rtp_video_rxtx::~h264_rtp_video_rxtx()
{
        free(m_rtsp_server);
        rtp_rxtx_common_done(m_rtp_common);
}

void h264_rtp_video_rxtx::join()
{
        c_stop_server(m_rtsp_server);
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

static void *
create_video_rxtx_h264_std(const struct vrxtx_params *params,
                           const struct common_opts  *common)
{
        int rtsp_port;
        const char *rtsp_port_str = params->protocol_opts;
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
        return new h264_rtp_video_rxtx(params, common, rtsp_port);
}

static void done(void *state) {
        auto *s = static_cast<h264_rtp_video_rxtx *>(state);
        delete s;
}

static void
send_frame(void *state, std::shared_ptr<video_frame> f)
{
        auto *s = static_cast<h264_rtp_video_rxtx *>(state);
        s->send_frame(std::move(f));
}

static void join(void *state) {
        auto *s = static_cast<h264_rtp_video_rxtx*>(state);
        s->join();
}

static void
configure_audio(struct h264_rtp_video_rxtx *s, const struct audio_frame2 *frame)
{
        s->rtsp_params.adesc =  frame->get_desc();
        MSG(VERBOSE, "Setting audio desc %s to RTSP.\n",
            audio_desc_to_cstring(s->rtsp_params.adesc));

        s->audio_params_set = true;

        if ((s->rtsp_params.avType & rtsp_type_video) == 0U) {
                s->m_rtsp_server = c_start_server(s->rtsp_params);
        }
}

static void
h264_rtp_send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        auto *s = static_cast<h264_rtp_video_rxtx *>(state);

        rtp_rxtx_sender_do_housekeeping(s->m_rtp_common, TX_MEDIA_AUDIO);

        if (!s->audio_params_set) {
                configure_audio(s, frame);
        }
        audio_tx_send_standard(
            s->m_rtp_common->medium[TX_MEDIA_AUDIO].tx,
            s->m_rtp_common->medium[TX_MEDIA_AUDIO].network_device, frame);
}

static bool
h264_rtp_ctl_property(void *state, enum rxtx_property p,
                           void *val, size_t *len)
{
        auto *s = static_cast<h264_rtp_video_rxtx *>(state);
        assert(s->magic == MAGIC);
        switch (p) {
        case GET_RTP_COMMON_STATE: {
                // NOLINTBEGIN(bugprone-sizeof-expression)
                assert(*len >= sizeof s->m_rtp_common);
                *len = sizeof s->m_rtp_common;
                // NOLINTEND(bugprone-sizeof-expression)
                memcpy(val, (void *) &s->m_rtp_common, *len);
                return true;
        }
        }
        MSG(WARNING, "Unexpected property %d queiried!\n", (int) p);
        return false;
}

static const struct video_rxtx_info h264_video_rxtx_info = {
        .long_name        = "RTP standard (using RTSP)",
        .create           = create_video_rxtx_h264_std,
        .done             = done,
        .send_video_frame = send_frame,
        .join_sender      = join,
        .send_audio_frame = h264_rtp_send_audio_frame,
        .receiver_routine = nullptr,
        .ctl_property     = h264_rtp_ctl_property,
};

REGISTER_MODULE(rtsp, &h264_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

