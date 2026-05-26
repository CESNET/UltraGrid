/**
 * @file   video_rxtx/rtp_common.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
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

#ifndef VIDEO_RXTX_RTP_COMMON_H_227CB2D9_DA5A_4A09_AE77_CC92F81C8D5A
#define VIDEO_RXTX_RTP_COMMON_H_227CB2D9_DA5A_4A09_AE77_CC92F81C8D5A

#include <pthread.h>
#ifdef __cplusplus
#include <cstdint>           // for uint32_t
#else
#include <stdint.h>          // for uint32_t
#endif

#include "types.h"
#include "utils/macros.h"    // for to_fourcc

#ifdef __cplusplus
extern "C" {
#endif // defined __cplusplus

#ifdef __APPLE__
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  5944320
#else
#define INITIAL_VIDEO_RECV_BUFFER_SIZE  ((4*1920*1080)*110/100)
#endif

#define INITIAL_VIDEO_SEND_BUFFER_SIZE  (1024*1024)

#define RTP_COMMON_MAGIC to_fourcc('V', 'R', 'r', 'c')

struct rtp_rxtx_medium {
        struct rtp     *network_device;
        struct tx      *tx;
        struct pdb     *participants;
        int             rxtx_mode;
        struct fec     *fec_state;
        pthread_mutex_t lock;
};

struct rtp_rxtx_common {
        uint32_t magic;
        struct rtp_rxtx_medium medium[NUM_TX_MEDIA];
        struct rtp_rxtx_common_priv_state *priv;
        char                              *encryption;
        bool playback_supports_multiple_streams; // set by impl
};

struct common_opts;
struct vrxtx_params;

struct rtp_rxtx_common *rtp_rxtx_common_init(const struct vrxtx_params *params,
                       const struct common_opts  *common);
void                    rtp_rxtx_common_done(struct rtp_rxtx_common *state);

void rtp_rxtx_sender_do_housekeeping(struct rtp_rxtx_common *pub,
                                     enum tx_media_type      t);
void rtp_rxtx_set_pbuf_delay(struct rtp_rxtx_medium *s, double delay);
bool rtp_rxtx_common_is_ipv6(struct rtp_rxtx_common *s);

struct coded_data;
struct pbuf_stats;
typedef int decode_audio_frame_fn(struct coded_data *cdata, void *pbuf_data,
                                  struct pbuf_stats *);
struct rx_audio_frames *rtp_recv_audio_frame(void                 *state,
                                             decode_audio_frame_fn decode);

#ifdef __cplusplus
}
#endif // defined __cplusplus

#endif // VIDEO_RXTX_RTP_COMMON_H_227CB2D9_DA5A_4A09_AE77_CC92F81C8D5A

