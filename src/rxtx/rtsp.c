/**
 * @file   rxtx/rtsp.c
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

#include <assert.h>            // for assert
#include <ctype.h>
#include <stdatomic.h>
#include <stdint.h>            // for uint32_t
#include <stdio.h>             // for printf
#include <stdlib.h>
#include <string.h>

#include "audio/types.h"                 // for audio_desc
#include "audio/utils.h"                 // for audio_desc_to_cstring
#include "compat/c23.h"                  // IWYU pragma: keep
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "messaging.h"
#include "rtp/audio_decoders.h"  // for decode_audio_frame_mulaw
#include "rtp/rtp.h"
#include "rtsp/BasicRTSPOnlyServer.hh"
#include "rtsp/rtsp_utils.h"  // for rtsp_types_t
#include "rxtx.h"
#include "rxtx/rtp_common.h"
#include "transmit.h"
#include "tv.h"
#include "types.h"            // for video_frame, H264, JPEG
#include "utils/color_out.h"
#include "utils/macros.h"     // for to_fourcc
#include "utils/sdp.h"        // for sdp_print_supported_codecs
#include "video_codec.h"      // for get_codec_name

struct audio_frame2;
struct rtp;
struct tx;

#define DEFAULT_RTSP_COMPRESSION "lavc:enc=libx264:safe"
#define MOD_NAME "[rxtx/rtsp] "
#define MAGIC to_fourcc('R', 'T', 'r', 's')

struct h264_rtp_rxtx {
        uint32_t magic;

        struct rtp_rxtx_common       *rtp_common;
        struct rtsp_server_parameters rtsp_params;
        atomic_bool                   audio_params_set;
        struct BasicRTSPOnlyServer   *rtsp_server;
        void (*tx_send_std)(struct tx *tx_session, struct video_frame *frame,
                            struct rtp *rtp_session);

        bool           sent_compress_change;
        struct module *parent;
        time_ns_t      start_time;
};

// protoypes
static void rtps_server_usage();
static int  get_rtsp_server_port(const char *config);
static void done(void *state);

static void *
create_rxtx_rtsp(struct rxtx_params *params)
{
        int rtsp_port = 0;
        const char *rtsp_port_str = params->protocol_opts;
        if (strlen(rtsp_port_str) > 0) {
                if (strcmp(rtsp_port_str, "help") == 0) {
                        rtps_server_usage();
                        return INIT_NOERR;
                }
                rtsp_port = get_rtsp_server_port(rtsp_port_str);
                if (rtsp_port == -1) {
                        return nullptr;
                }
        }
        struct h264_rtp_rxtx *s = calloc(1, sizeof *s);
        s->magic                      = MAGIC;
        s->parent                     = params->parent;
        s->start_time                 = params->start_time;
        s->rtsp_params.rtsp_port      = (unsigned) rtsp_port;
        s->rtsp_params.parent         = params->parent;

        rtsp_types_t avType =
            (rtsp_types_t) (SENDS_MEDIUM(params, TX_MEDIA_AUDIO)
                                ? rtsp_type_audio
                                : 0);
        avType = (rtsp_types_t) (avType | (SENDS_MEDIUM(params, TX_MEDIA_VIDEO)
                                               ? rtsp_type_video
                                               : 0));
        if (avType == rtsp_type_none) {
                printf("[RTSP SERVER CHECK] no stream type... check capture devices input...\n");
                done(s);
                return nullptr;
        }
        s->rtsp_params.avType = avType;

        s->rtsp_params.rtp_audio_src_port = params->medium[TX_MEDIA_AUDIO].rx_port;
        s->rtsp_params.rtp_video_src_port = params->medium[TX_MEDIA_VIDEO].rx_port;
        s->rtp_common                   = rtp_rxtx_common_init(params);
        if (s->rtp_common == nullptr) {
                done(s);
                return nullptr;
        }
        if (strlen(params->video_compression) == 0) {
                strcpy_ch(params->video_compression,
                          DEFAULT_RTSP_COMPRESSION " (tentatively)");
        }
        return s;
}


/**
 * this function is used to configure their RTSP server either
 * for video-only or using both audio and video. For audio-only
 * RTSP server, the server is run directly from
 * configure_audio().
 */
static void
configure_rtsp_server_video(struct h264_rtp_rxtx *s)
{
        assert((s->rtsp_params.avType & rtsp_type_video) != 0);
        switch (s->rtsp_params.video_codec) {
        case H264:
                s->tx_send_std = tx_send_h264;
                break;
        case H265:
                s->tx_send_std = tx_send_h265;
                break;
        case JPEG:
                s->tx_send_std = tx_send_jpeg;
                break;
        default:
                MSG(ERROR,
                    "codecs other than H.264/H.265 and JPEG currently not "
                    "supported, got %s\n",
                    get_codec_name(s->rtsp_params.video_codec));
                return;
        }

        if ((s->rtsp_params.avType & rtsp_type_audio) != 0) {
                if (!s->audio_params_set) {
                        MSG(INFO, "Waiting for audio specs...\n");
                        return;
                }
        }
        s->rtsp_server = start_rtsp_server(s->rtsp_params);
}

static void
send_frame_impl(struct h264_rtp_rxtx *s, struct video_frame *tx_frame)
{
        struct rtp_rxtx_medium *video = &s->rtp_common->medium[TX_MEDIA_VIDEO];

        rtp_rxtx_sender_do_housekeeping(s->rtp_common, TX_MEDIA_VIDEO);
        // requestt compress reconfiguration if receivng raw data
        if (!is_codec_opaque(tx_frame->color_spec)) {
                if (!s->sent_compress_change) {
                        send_compess_change(s->parent,
                                            DEFAULT_RTSP_COMPRESSION);
                        s->sent_compress_change = true;
                }
                return;
        }

        if (s->rtsp_server == nullptr) {
                s->rtsp_params.video_codec = tx_frame->color_spec;
                configure_rtsp_server_video(s);
        }
        if (s->rtsp_server == nullptr) {
                return;
        }

        if (tx_frame->color_spec != s->rtsp_params.video_codec) {
                MSG(ERROR, "Video codec reconfiguration is not supported!\n");
                return;
        }

        s->tx_send_std(video->tx, tx_frame, video->network_device);
}

/// wraps send_frame_impl to ensure tx_frame is disposed across all code paths
static void
send_frame(void *state, struct video_frame *tx_frame)
{
        struct h264_rtp_rxtx *s = state;
        send_frame_impl(s, tx_frame);
        tx_frame->callbacks.dispose(tx_frame);
}

static void
join(void *state)
{
        struct h264_rtp_rxtx *s = state;
        stop_rtsp_server(s->rtsp_server);
        s->rtsp_server = nullptr;
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

static void done(void *state) {
        struct h264_rtp_rxtx *s = state;
        rtp_rxtx_common_done(s->rtp_common);
        free(s);
}

static void
configure_audio(struct h264_rtp_rxtx *s, const struct audio_frame2 *frame)
{
        s->rtsp_params.adesc =  audio_frame2_get_desc(frame);
        MSG(VERBOSE, "Setting audio desc %s to RTSP.\n",
            audio_desc_to_cstring(s->rtsp_params.adesc));

        s->audio_params_set = true;

        if ((s->rtsp_params.avType & rtsp_type_video) == 0U) {
                s->rtsp_server = start_rtsp_server(s->rtsp_params);
        }
}

static void
h264_rtp_send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        struct h264_rtp_rxtx *s = state;

        rtp_rxtx_sender_do_housekeeping(s->rtp_common, TX_MEDIA_AUDIO);

        if (!s->audio_params_set) {
                configure_audio(s, frame);
        }
        audio_tx_send_standard(
            s->rtp_common->medium[TX_MEDIA_AUDIO].tx,
            s->rtp_common->medium[TX_MEDIA_AUDIO].network_device, frame);
}

static bool
h264_rtp_ctl_property(void *state, enum rxtx_property p,
                           void *val, size_t *len)
{
        struct h264_rtp_rxtx *s = state;
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
h264_rtp_recv_audio_frame(void *state)
{
        struct h264_rtp_rxtx *s = state;
        return rtp_recv_audio_frame(s->rtp_common, decode_audio_frame_mulaw);
}

static const struct rxtx_info rtsp_rxtx_info = {
        .long_name          = "RTP standard (using RTSP)",
        .create             = create_rxtx_rtsp,
        .done               = done,
        .ctl_property       = h264_rtp_ctl_property,

        .send_audio_frame   = h264_rtp_send_audio_frame,
        .recv_audio_frame   = h264_rtp_recv_audio_frame,

        .send_video_frame   = nullptr,
        .send_video_frame_c = send_frame,
        .video_recv_routine = nullptr,
        .join_video_sender  = join,
};

REGISTER_MODULE(rtsp, &rtsp_rxtx_info, LIBRARY_CLASS_RXTX, RXTX_ABI_VERSION);
