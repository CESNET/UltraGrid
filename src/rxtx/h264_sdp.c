/**
 * @file   rxtx/h264_sdp.c
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

#include <assert.h>   // for assert
#include <stdint.h>   // for uint32_t
#include <stdlib.h>   // for abort
#include <string.h>   // for memcpy, strdup, strcmp
#include <sys/time.h> // for timeval

#include "audio/audio.h"
#include "audio/types.h"         // for audio_desc
#include "audio/utils.h"         // for audio_desc_to_cstring
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"           // for msg_change_compress_data, free_response
#include "module.h"
#include "rtp/audio_decoders.h"  // for decode_audio_frame_mulaw
#include "rtp/rtp.h"
#include "rxtx.h"
#include "rxtx/rtp_common.h"
#include "transmit.h"
#include "tv.h"
#include "types.h"               // for video_frame, VIDEO_CODEC_NONE, codec_t
#include "utils/macros.h"        // for strcpy_ch, to_fourcc
#include "utils/sdp.h"
#include "video_codec.h"         // for is_codec_opaque

struct audio_frame2;

#define DEFAULT_SDP_COMPRESSION "lavc:codec=MJPG:safe"
#define MOD_NAME "[rxtx/sdp] "

#define MAGIC to_fourcc('V', 'X', 'h', 's')

struct h264_sdp_video_rxtx {
        uint32_t    magic;
        struct sdp *sdp_state;

        struct rtp_rxtx_common *rtp_common;
        codec_t                 sdp_configured_codec;
        int                     audio_tx_port;
        int                     video_tx_port;
        bool                    sent_compress_change;

        /// for dynamic address reconfiguration, @see sdp_set_options autorun
        char          *saved_addr;
        struct module *parent;

        bool audio_params_set;
};

static void change_address_callback(void *udata, const char *address);
static void done(void *state);

static void *
create_video_rxtx_h264_sdp(const struct rxtx_params *params)
{
        struct h264_sdp_video_rxtx *s = calloc(1, sizeof *s);

        s->magic    = MAGIC;
        s->audio_tx_port = -1;
        s->video_tx_port = -1;
        s->parent = params->parent;

        const struct rxtx_medium_params *audio =
            &params->medium[TX_MEDIA_AUDIO];
        const struct rxtx_medium_params *video =
            &params->medium[TX_MEDIA_VIDEO];
        const char *opts = params->protocol_opts;
        MSG(WARNING,
            "Warning: SDP support is experimental only. Things may be broken - "
            "feel free to report them but the support may be limited.\n");
        if (audio->rxtx_mode & MODE_SENDER) {
                s->audio_tx_port = audio->tx_port;
        }
        if (video->rxtx_mode & MODE_SENDER) {
                s->video_tx_port = video->tx_port;
        }

        s->rtp_common = rtp_rxtx_common_init(params);
        if (s->rtp_common == nullptr) {
                done(s);
                return nullptr;
        }

        bool is_ipv6 = rtp_rxtx_common_is_ipv6(s->rtp_common);
        s->sdp_state =
            sdp_init(opts, is_ipv6, params->receiver,
                     SENDS_MEDIUM(params, TX_MEDIA_VIDEO),
                     SENDS_MEDIUM(params, TX_MEDIA_AUDIO),
                     change_address_callback, s);
        if (s->sdp_state == nullptr) {
                done(s);
                return strcmp(opts, "help") == 0 ? INIT_NOERR : nullptr;
        }
        s->saved_addr = strdup(params->receiver);

        return s;
}

void
sdp_send_change_address_message(struct module           *root,
                                const enum module_class *path,
                                const char              *address)
{
        char pathV[1024];

        set_message_path(pathV, sizeof pathV, path);

        // CHANGE DST ADDRESS
        struct msg_sender *msgV2 = (struct msg_sender *)
            new_message(sizeof(struct msg_sender));
        strcpy_ch(msgV2->receiver, address);
        msgV2->type = SENDER_MSG_CHANGE_RECEIVER;

        struct response *resp = send_message(root, pathV, (struct message *) msgV2);
        if (response_get_status(resp) == RESPONSE_OK) {
                MSG(NOTICE, "Changing address to %s\n", address);
        } else {
                MSG(WARNING, "Unable to change address to %s (%d)\n", address,
                    response_get_status(resp));
        }
        free_response(resp);
}

static void
change_address_callback(void *udata, const char *address)
{
        struct h264_sdp_video_rxtx *s = udata;
        if (s->saved_addr == address) {
                return;
        }
        free(s->saved_addr);
        s->saved_addr = strdup(address);

        if (s->video_tx_port != -1) {
                sdp_send_change_address_message(get_root_module(s->parent),
                                                path_sender_video, address);
        }
        if (s->audio_tx_port != -1) {
                sdp_send_change_address_message(get_root_module(s->parent),
                                                path_sender_audio, address);
        }
}

static bool
h264_sdp_add_video(struct h264_sdp_video_rxtx *s, codec_t codec)
{
        const int rc = sdp_add_video(s->sdp_state, s->video_tx_port, codec);
        if (rc == -2) {
                MSG(ERROR, "Unsupported video codec for SDP (allowed H.264 and JPEG)!\n");
                return false;
        }
        if (rc != 0) {
		abort();
	}
	return true;
}

/**
 * @note
 * This function sets compression just after first frame is received. The delayed initialization is to allow devices
 * producing H.264/JPEG natively (eg. v4l2) to be passed untouched to transport. Fallback H.264 is applied when uncompressed
 * stream is detected.
 */
static void
send_frame_impl(struct h264_sdp_video_rxtx *s, struct video_frame *tx_frame)
{
        struct rtp_rxtx_medium *video = &s->rtp_common->medium[TX_MEDIA_VIDEO];

        rtp_rxtx_sender_do_housekeeping(s->rtp_common, TX_MEDIA_VIDEO);
        if (!is_codec_opaque(tx_frame->color_spec)) {
		if (s->sent_compress_change) {
			return;
		}
		send_compess_change(s->parent, DEFAULT_SDP_COMPRESSION);
		s->sent_compress_change = true;
		return;
        }
        if (s->sdp_configured_codec == VIDEO_CODEC_NONE) {
                if (!h264_sdp_add_video(s, tx_frame->color_spec)) {
                        exit_uv(1);
                        return;
                }
                s->sdp_configured_codec = tx_frame->color_spec;
        }

        if (s->sdp_configured_codec != tx_frame->color_spec) {
                MSG(ERROR, "Video codec reconfiguration is not supported!\n");
                return;
        }

        if (s->sdp_configured_codec == H264) {
                tx_send_h264(video->tx, tx_frame, video->network_device);
        } else {
                tx_send_jpeg(video->tx, tx_frame, video->network_device);
        }
}

/// wraps send_frame_impl to ensure tx_frame is disposed across all code paths
static void
send_frame(void *state, struct video_frame *tx_frame) {
        struct h264_sdp_video_rxtx *s = state;
        send_frame_impl(s, tx_frame);
        tx_frame->callbacks.dispose(tx_frame);
}

static void done(void *state) {
        struct h264_sdp_video_rxtx *s = state;

        rtp_rxtx_common_done(s->rtp_common);
        sdp_done(s->sdp_state);
        free(s->saved_addr);
        free(s);
}

static void
configure_audio(struct h264_sdp_video_rxtx *s, const struct audio_frame2 *frame)
{
        const struct audio_desc desc = audio_frame2_get_desc(frame);
        MSG(VERBOSE, "Setting audio desc %s to SDP.\n",
            audio_desc_to_cstring(desc));

        s->audio_params_set = true;

        int ret = sdp_add_audio(s->sdp_state, s->audio_tx_port,
                                desc.sample_rate, desc.ch_count, desc.codec);
        if (ret != 0) {
                MSG(ERROR, "Cannot add audio to SDP!\n");
        }
}

static void
h264_sdp_send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        struct h264_sdp_video_rxtx *s = state;

        rtp_rxtx_sender_do_housekeeping(s->rtp_common, TX_MEDIA_AUDIO);

        if (!s->audio_params_set) {
                configure_audio(s, frame);
        }
        audio_tx_send_standard(
            s->rtp_common->medium[TX_MEDIA_AUDIO].tx,
            s->rtp_common->medium[TX_MEDIA_AUDIO].network_device, frame);
}

static bool
h264_sdp_ctl_property(void *state, enum rxtx_property p,
                           void *val, size_t *len)
{
        struct h264_sdp_video_rxtx *s = state;
        assert(s->magic == MAGIC);
        switch (p) {
        case GET_RTP_COMMON_STATE: {
                // NOLINTBEGIN(bugprone-sizeof-expression)
                assert(*len >= sizeof s->rtp_common);
                *len = sizeof s->rtp_common;
                // NOLINTEND(bugprone-sizeof-expression)
                memcpy(val, (void *) &s->rtp_common, *len);
                return true;
        }
        case SET_RTP_AUD_FRM_SZ: {
                int sz = 0;
                assert(*len >= sizeof sz);
                memcpy((void *) &sz, val, sizeof sz);
                rtp_set_recv_buf(
                    s->rtp_common->medium[TX_MEDIA_AUDIO].network_device, sz);
                return true;
        }
        case SET_ULTRAGRID_RTP_MUTLI_OUT:
                abort();
        }
        MSG(WARNING, "Unexpected property %d queiried!\n", (int) p);
        return false;
}

// I don't believe this works (and worked before rework).
static struct rx_audio_frames *
h264_sdp_recv_audio_frame(void *state)
{
        struct h264_sdp_video_rxtx *s = state;
        return rtp_recv_audio_frame(s->rtp_common, decode_audio_frame_mulaw);
}

static const struct rxtx_info h264_sdp_video_rxtx_info = {
        .long_name    = "RTP standard (SDP version)",
        .create       = create_video_rxtx_h264_sdp,
        .done         = done,
        .ctl_property = h264_sdp_ctl_property,

        .send_audio_frame = h264_sdp_send_audio_frame,
        .recv_audio_frame = h264_sdp_recv_audio_frame,

        .send_video_frame   = nullptr,
        .send_video_frame_c = send_frame,
        .video_recv_routine = nullptr,
        .join_video_sender  = nullptr,
};

REGISTER_MODULE(sdp, &h264_sdp_video_rxtx_info, LIBRARY_CLASS_RXTX, RXTX_ABI_VERSION);

