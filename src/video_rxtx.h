/**
 * @file   video_rxtx.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET zájmové sdružení právnických osob
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

#ifndef VIDEO_RXTX_H_
#define VIDEO_RXTX_H_

#ifdef __cplusplus
#include <memory>     // for std::shared_ptr
#endif

#include "host.h"
#include "types.h"    // for codec_t, video_desc, video_frame (ptr only)

#define VIDEO_RXTX_ABI_VERSION 4

struct audio_desc;

struct rxtx_medium_params  {
        enum rxtx_mode  rxtx_mode;      ///< sender, receiver or both
        int             rx_port;
        int             tx_port;
        const char     *fec;
};

struct vrxtx_params {
        struct rxtx_medium_params medium[NUM_TX_MEDIA];

        const char     *compression; ///< nullptr selects proto dfl
        struct display *display_device; ///< only iHDTV, UG RTP
        struct vidcap  *capture_device; ///< iHDTV only
        long long       bitrate_limit; ///< rate limiter in bps or RATE_ constantts
        enum video_mode decoder_mode;
        const char     *protocol_opts;
        bool            send_audio;   ///< RTSP+SDP
        bool            send_video;   ///< RTSP+SDP
        struct module  *sender_mod;   ///< set by video_rxtx::create
        struct module  *receiver_mod; ///< set by video_rxtx::create
};

#define VRXTX_INIT \
        { \
                .medium         = { { }, \
                                   { \
                                        .rxtx_mode      = RXTX_MODE_NONE, \
                                        .rx_port       = -1, \
                                        .tx_port       = -1, \
                                        .fec           = "none", \
                                    } }, \
                .compression    = nullptr, \
                .display_device = nullptr, \
                .capture_device = nullptr, \
                .bitrate_limit  = RATE_UNLIMITED, \
                .decoder_mode   = VIDEO_NORMAL, \
                .protocol_opts  = "", \
                .send_audio     = false, \
                .send_video     = false, \
                .sender_mod     = nullptr, \
                .receiver_mod   = nullptr, \
}

#ifdef __cplusplus
struct video_rxtx_info {
        const char *long_name;
        void *(*create)(const struct vrxtx_params *params,
                        const struct common_opts  *common);
        void (*done)(void *state);
        /// this may be set optional if we had some receive-only mod
        void (*send_frame)(void *state, std::shared_ptr<video_frame>);

        // following callbacks are optional
        void (*join_sender)(void *state);
        void (*set_sender_audio_spec)(void                    *state,
                                      const struct audio_desc *desc,
                                      int audio_rx_port, int audio_tx_port,
                                      bool ipv6);
        void *(*receiver_routine)(void *state);
};
#endif // defined __cplusplus

#ifdef __cplusplus
extern "C" {
#endif

struct video_rxtx;

int  vrxtx_init(const char *proto_name, const struct vrxtx_params *params,
                const struct common_opts *opts, struct video_rxtx **state);
void vrxtx_destroy(struct video_rxtx *state);
void        vrxtx_list_protocols(bool full);
const char *vrxtx_get_proto_long_name(const char *short_name);
const char *vrxtx_get_compression(const char *video_protocol,
                                  const char *req_compression);
void        vrxtx_join(struct video_rxtx *state);
void  vrxtx_set_audio_spec(struct video_rxtx       *state,
                           const struct audio_desc *desc, int audio_rx_port,
                           int audio_tx_port, bool ipv6);
void *vrxtx_get_impl_state(struct video_rxtx *state);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
void vrxtx_send(struct video_rxtx *state, std::shared_ptr<struct video_frame>);
#endif

#endif // VIDEO_RXTX_H_

