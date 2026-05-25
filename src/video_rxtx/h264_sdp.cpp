/**
 * @file   video_rxtx/h264_sdp.cpp
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

#include <cassert>               // for assert
#include <cstdint>               // for uint32_t
#include <cstdlib>               // for abort
#include <exception>             // for exception
#include <memory>                // for shared_ptr
#include <ostream>               // for basic_ostream, operator<<
#include <string>                // for basic_string, string, operator==
#include <utility>               // for move

#include "audio/audio.h"
#include "audio/types.h"         // for audio_desc
#include "audio/utils.h"         // for audio_desc_to_cstring
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"           // for msg_change_compress_data, free_response
#include "module.h"
#include "rtp/rtp.h"
#include "transmit.h"
#include "tv.h"
#include "types.h"               // for video_frame, VIDEO_CODEC_NONE, codec_t
#include "ug_runtime_error.hpp"
#include "utils/sdp.h"
#include "video_codec.h"         // for is_codec_opaque
#include "video_rxtx.h"
#include "video_rxtx/rtp_common.h"

#define DEFAULT_SDP_COMPRESSION "lavc:codec=MJPG:safe"
#define MOD_NAME "[vrxtx/sdp] "

constexpr uint32_t MAGIC = to_fourcc('V', 'X', 'h', 's');

struct h264_sdp_video_rxtx {
        uint32_t magic = MAGIC;
        struct sdp *sdp_state = nullptr;
        h264_sdp_video_rxtx(const struct vrxtx_params *params,
                            const struct common_opts  *common);
        ~h264_sdp_video_rxtx();
        void send_frame(std::shared_ptr<video_frame>) noexcept;
        void set_audio_spec(const struct audio_desc *desc, int audio_rx_port,
                            int audio_tx_port, bool ipv6);

        struct rtp_rxtx_common *m_rtp_common;
        static void change_address_callback(void *udata, const char *address);
        void sdp_add_video(codec_t codec);
        codec_t m_sdp_configured_codec = VIDEO_CODEC_NONE;
        int m_audio_tx_port = -1;
        int m_video_tx_port = -1;
        bool m_sent_compress_change = false;

        std::string m_saved_addr; ///< for dynamic address reconfiguration, @see m_autorun
        struct module *m_parent;

        bool audio_params_set = false;
};


using std::exception;
using std::shared_ptr;
using std::string;

h264_sdp_video_rxtx::h264_sdp_video_rxtx(const struct vrxtx_params *params,
                                         const struct common_opts  *common)
        : m_parent(common->parent)
{
        const struct rxtx_medium_params *audio =
            &params->medium[TX_MEDIA_AUDIO];
        const struct rxtx_medium_params *video =
            &params->medium[TX_MEDIA_VIDEO];
        auto opts = params->protocol_opts;
        LOG(LOG_LEVEL_WARNING) << "Warning: SDP support is experimental only. Things may be broken - feel free to report them but the support may be limited.\n";
        if (audio->rxtx_mode & MODE_SENDER) {
                m_audio_tx_port = audio->tx_port;
        }
        if (video->rxtx_mode & MODE_SENDER) {
                m_video_tx_port = video->tx_port;
        }

        m_rtp_common = rtp_rxtx_common_init(params, common);
        if (m_rtp_common == nullptr) {
                throw -1;
        }

        bool is_ipv6 = rtp_rxtx_common_is_ipv6(m_rtp_common);
        sdp_state =
            sdp_init(opts, is_ipv6, common->receiver,
                     SENDS_MEDIUM(params, TX_MEDIA_VIDEO),
                     SENDS_MEDIUM(params, TX_MEDIA_AUDIO),
                     h264_sdp_video_rxtx::change_address_callback, this);
        if (sdp_state == nullptr) {
                this->~h264_sdp_video_rxtx();
                throw strcmp(opts, "help") == 0 ? 1 : -1;
        }
        m_saved_addr = common->receiver;
}

h264_sdp_video_rxtx::~h264_sdp_video_rxtx() {
        rtp_rxtx_common_done(m_rtp_common);
        sdp_done(sdp_state);
}

void
sdp_send_change_address_message(struct module           *root,
                                const enum module_class *path,
                                const char              *address)
{
        char pathV[1024];

        set_message_path(pathV, sizeof pathV, path);

        // CHANGE DST ADDRESS
        auto *msgV2 = reinterpret_cast<struct msg_sender *>(
            new_message(sizeof(struct msg_sender)));
        strncpy(static_cast<char *>(msgV2->receiver), address,
                sizeof(msgV2->receiver) - 1);
        msgV2->type = SENDER_MSG_CHANGE_RECEIVER;

        auto *resp = send_message(root, pathV, (struct message *) msgV2);
        if (response_get_status(resp) == RESPONSE_OK) {
                LOG(LOG_LEVEL_NOTICE)
                    << "[SDP] Changing address to " << address << "\n";
        } else {
                LOG(LOG_LEVEL_WARNING)
                    << "[SDP] Unable to change address to " << address << " ("
                    << response_get_status(resp) << ")\n";
        }
        free_response(resp);
}

void h264_sdp_video_rxtx::change_address_callback(void *udata, const char *address)
{
        auto *s = static_cast<h264_sdp_video_rxtx *>(udata);
        if (s->m_saved_addr == address) {
                return;
        }
        s->m_saved_addr = address;

        if (s->m_video_tx_port != -1) {
                sdp_send_change_address_message(get_root_module(s->m_parent),
                                                path_sender_video, address);
        }
        if (s->m_audio_tx_port != -1) {
                sdp_send_change_address_message(get_root_module(s->m_parent),
                                                path_sender_audio, address);
        }
}

void h264_sdp_video_rxtx::sdp_add_video(codec_t codec)
{
        const int rc = ::sdp_add_video(sdp_state, m_video_tx_port, codec);
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
        struct rtp_rxtx_medium *video = &m_rtp_common->medium[TX_MEDIA_VIDEO];

        rtp_rxtx_sender_do_housekeeping(m_rtp_common, TX_MEDIA_VIDEO);
        if (!is_codec_opaque(tx_frame->color_spec)) {
		if (m_sent_compress_change) {
			return;
		}
		send_compess_change(m_parent, DEFAULT_SDP_COMPRESSION);
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
                tx_send_h264(video->tx, tx_frame.get(), video->network_device);
        } else {
                tx_send_jpeg(video->tx, tx_frame.get(), video->network_device);
        }
        if (video->rxtx_mode & MODE_RECEIVER) {
                // send RTCP (receiver thread would otherwise do this)
                uint32_t ts = get_std_video_local_mediatime();
                time_ns_t curr_time = get_time_in_ns();
                rtp_update(video->network_device, curr_time);
                rtp_send_ctrl(video->network_device, ts, nullptr, curr_time);

                // receive RTCP
                struct timeval timeout;
                timeout.tv_sec = 0;
                timeout.tv_usec = 0;
                rtp_recv_r(video->network_device, &timeout, ts);
        }
}

static void *
create_video_rxtx_h264_sdp(const struct vrxtx_params *params,
                           const struct common_opts  *common)
{
        return new h264_sdp_video_rxtx(params, common);
}

static void done(void *state) {
        auto *s = static_cast<h264_sdp_video_rxtx *>(state);
        delete s;
}

static void
send_frame(void *state, std::shared_ptr<video_frame> f)
{
        auto *s = static_cast<h264_sdp_video_rxtx *>(state);
        s->send_frame(std::move(f));
}

static void
configure_audio(struct h264_sdp_video_rxtx *s, const struct audio_frame2 *frame)
{
        const struct audio_desc desc =  frame->get_desc();
        MSG(VERBOSE, "Setting audio desc %s to SDP.\n",
            audio_desc_to_cstring(desc));

        s->audio_params_set = true;

        int ret = sdp_add_audio(s->sdp_state, s->m_audio_tx_port,
                                desc.sample_rate, desc.ch_count, desc.codec);
        if (ret != 0) {
                MSG(ERROR, "Cannot add audio to SDP!\n");
        }
}

static void
h264_sdp_send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        auto *s = static_cast<h264_sdp_video_rxtx *>(state);

        rtp_rxtx_sender_do_housekeeping(s->m_rtp_common, TX_MEDIA_AUDIO);

        if (!s->audio_params_set) {
                configure_audio(s, frame);
        }
        audio_tx_send_standard(
            s->m_rtp_common->medium[TX_MEDIA_AUDIO].tx,
            s->m_rtp_common->medium[TX_MEDIA_AUDIO].network_device, frame);
}


static bool
h264_sdp_ctl_property(void *state, enum rxtx_property p,
                           void *val, size_t *len)
{
        auto *s = static_cast<h264_sdp_video_rxtx *>(state);
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

static const struct video_rxtx_info h264_sdp_video_rxtx_info = {
        .long_name        = "RTP standard (SDP version)",
        .create           = create_video_rxtx_h264_sdp,
        .done             = done,
        .send_video_frame = send_frame,
        .join_sender      = nullptr,
        .send_audio_frame = h264_sdp_send_audio_frame,
        .receiver_routine = nullptr,
        .ctl_property     = h264_sdp_ctl_property,
};

REGISTER_MODULE(sdp, &h264_sdp_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

