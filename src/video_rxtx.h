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
#include <cstddef>    // for size_t
#include <memory>     // for std::shared_ptr
#else
#include <stddef.h>   // for size_t
#endif

#include "compat/c23.h" // IWYU pragma: keep for nullptr
#include "host.h"
#include "types.h"    // for codec_t, video_desc, video_frame (ptr only)

#define VIDEO_RXTX_ABI_VERSION 4

struct audio_desc;
struct audio_frame2;

struct rxtx_medium_params  {
        enum rxtx_mode  rxtx_mode;      ///< sender, receiver or both
        int             rx_port;
        int             tx_port;
        const char     *fec;
};

struct vrxtx_params {
        struct rxtx_medium_params medium[NUM_TX_MEDIA];

        struct module  *parent;
        struct exporter *video_exporter;
        const char     *receiver;
        char            encryption[STR_LEN];
        char            mcast_if[STR_LEN];
        int             mtu;
        int             ttl;
        int             force_ip_version;
        time_ns_t       start_time;

        const char     *compression; ///< nullptr selects proto dfl
        struct display *display_device; ///< only iHDTV, UG RTP
        struct vidcap  *capture_device; ///< iHDTV only
        long long       bitrate_limit; ///< rate limiter in bps or RATE_ constantts
        enum video_mode decoder_mode;
        char            protocol_opts[STR_LEN];
        struct module  *sender_mod;   ///< set by video_rxtx::create
        struct module  *receiver_mod; ///< set by video_rxtx::create
};

#define VRXTX_INIT \
        { \
                .medium         = { { \
                                        .rxtx_mode = RXTX_MODE_NONE, \
                                        .rx_port   = -1, \
                                        .tx_port   = -1, \
                                        .fec       = "none", \
                                    }, { \
                                        .rxtx_mode = RXTX_MODE_NONE, \
                                        .rx_port   = -1, \
                                        .tx_port   = -1, \
                                        .fec       = "none", \
                                    } }, \
                .parent           = nullptr, \
                .video_exporter   = nullptr, \
                .receiver         = nullptr, \
                .encryption       = "", \
                .mcast_if         = "", \
                .mtu              = 1500, \
                .ttl              = -1, \
                .force_ip_version = 0, \
                .start_time       = get_time_in_ns(), \
                .compression    = nullptr, \
                .display_device = nullptr, \
                .capture_device = nullptr, \
                .bitrate_limit  = RATE_UNLIMITED, \
                .decoder_mode   = VIDEO_NORMAL, \
                .protocol_opts  = "", \
                .sender_mod     = nullptr, \
                .receiver_mod   = nullptr, \
}

#define SENDS_MEDIUM(params, medium_type)                                      \
        (((params)->medium[medium_type].rxtx_mode & MODE_SENDER) != 0U)

enum rxtx_property {
        GET_RTP_COMMON_STATE, ///< RTP state - pointer to struct rtp_rxtx_common
        SET_RTP_AUD_FRM_SZ,   ///< pointer to int
        SET_ULTRAGRID_RTP_MUTLI_OUT, ///< pointer to bool
};

//
// API for modules
//
/**
 * creates RX/TX state
 * @retval nullptr     on error
 * @retval INIT_NOERR  if help requested
 * @return the actual state
 */
typedef void *rxtx_create_fn(const struct vrxtx_params *params);
typedef void  rxtx_done_fn(void *state);
typedef bool  rxtx_ctl_property_fn(void *state, enum rxtx_property p, void *val,
                                   size_t *len);

typedef void  rxtx_send_audio_frame_fn(void                      *state,
                                       const struct audio_frame2 *frame);
struct rx_audio_frames {
        struct audio_frame2     *frame;
        struct sockaddr_storage *source; // network source address
        long long int            expected_bytes;
        long long int            received_bytes;
        struct rx_audio_frames  *next;
};
/**
 * receive one or more audio frames
 * @note mutiple frames is just a special case, it occurs in case of
 * ultragrid_rtp and aplay/mixer (@ref AUDIO_PLAYBACK_CTL_MULTIPLE_STREAMS)
 */
typedef struct rx_audio_frames *rxtx_recv_audio_frame_fn(void *state);

#ifdef __cplusplus
typedef void  rxtx_send_shr_ptr_video_frame_fn(void *state,
                                               std::shared_ptr<video_frame>);
#else
typedef nullptr_t rxtx_send_shr_ptr_video_frame_fn;
#endif // defined __cplusplus
/**
 * @param f  f->callbacks.dispose(f) must be called
 */
typedef void  rxtx_send_video_frame_fn(void *state, struct video_frame *f);
typedef void *rxtx_vrecv_routine_fn(void *state);
typedef void  rxtx_join_video_sender_fn(void *state);

struct video_rxtx_info {
        const char     *long_name;
        rxtx_create_fn *create;
        rxtx_done_fn   *done;
        // the callbacks below are optional
        rxtx_ctl_property_fn *ctl_property;

        rxtx_send_audio_frame_fn *send_audio_frame;
        rxtx_recv_audio_frame_fn *recv_audio_frame;

        rxtx_send_shr_ptr_video_frame_fn *send_video_frame;
        rxtx_send_video_frame_fn         *send_video_frame_c;
        rxtx_vrecv_routine_fn            *video_recv_routine;
        rxtx_join_video_sender_fn        *join_video_sender;
};

#ifdef __cplusplus
extern "C" {
#endif

struct audio_frame2;
struct video_rxtx;

int  vrxtx_init(const char *proto_name, const struct vrxtx_params *params,
                struct video_rxtx **state);
void vrxtx_destroy(struct video_rxtx *state);
void vrxtx_list_protocols(bool full);
const char *vrxtx_get_proto_long_name(const char *short_name);
void        vrxtx_join(struct video_rxtx *state);
bool        rxtx_ctl_property(struct video_rxtx *state, enum rxtx_property p,
                              void *val, size_t *len);
void        rxtx_send_audio(struct video_rxtx         *state,
                            const struct audio_frame2 *frame);
struct rx_audio_frames *rxtx_recv_audio_frame(struct video_rxtx *s);
void rxtx_free_audio_frames(struct rx_audio_frames *frames);
enum rxtx_mode rxtx_get_mode(struct video_rxtx *s, enum tx_media_type t);
void rxtx_send_video(struct video_rxtx *state, struct video_frame *tx_frame);

// utils
const char *get_tx_name(enum tx_media_type);

// get protocol eligible params if user doesn't specify any
// (passthrough otherwise)
unsigned int rxtx_get_achannels(const char  *net_protocol,
                                unsigned int req_channels);
const char  *rxtx_get_acompression(const char *net_protocol,
                                   const char *req_codec);
const char  *rxtx_get_vcompression(const char *net_protocol,
                                   const char *req_compression);

#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus
void vrxtx_send(struct video_rxtx *state, std::shared_ptr<struct video_frame>);
#endif

#endif // VIDEO_RXTX_H_

