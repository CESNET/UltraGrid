/**
 * @file   video_rxtx/rtp.hpp
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

#ifndef VIDEO_RXTX_RTP_H_
#define VIDEO_RXTX_RTP_H_

#include <cstdint>           // for uint32_t
#include <pthread.h>

#include "types.h"
#include "video_rxtx.h"
#include "utils/macros.h"    // for to_fourcc

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
        pthread_mutex_t lock;
};

struct rtp_rxtx_common {
        uint32_t magic;
        struct rtp_rxtx_medium medium[NUM_TX_MEDIA];
        struct fec     *fec_state;
        struct rtp_rxtx_common_priv_state *priv;
};

struct rtp_rxtx_common *rtp_rxtx_common_init(const struct vrxtx_params *params,
                       const struct common_opts  *common);
void                    rtp_rxtx_common_done(struct rtp_rxtx_common *state);

void rtp_rxtx_sender_do_housekeeping(struct rtp_rxtx_common *s);
void rtp_rxtx_set_pbuf_delay(struct rtp_rxtx_medium *s, double delay);

#endif // VIDEO_RXTX_RTP_H_

