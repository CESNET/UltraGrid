/*
 * FILE:   transmit.h
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *         David Cassany    <david.cassany@i2cat.net>
 *         Ignacio Contreras <ignacio.contreras@i2cat.net>
 *         Gerard Castillo  <gerard.castillo@i2cat.net>
 *         Jordi "Txor" Casas Ríos <txorlings@gmail.com>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2001-2002 University of Southern California
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
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
 *
 */

#ifndef TRANSMIT_H_
#define TRANSMIT_H_

#include "audio/audio.h"
#include "rtp/rtpenc_h264.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct module;
struct rtp;
struct tx;
struct video_frame;

struct tx *tx_init(struct module *parent, unsigned mtu, enum tx_media_type media_type,
                const char *fec, const char *encryption, long long bitrate);
void		 tx_send_tile(struct tx *tx_session, struct video_frame *frame, int pos, struct rtp *rtp_session);
void             tx_send(struct tx *tx_session, struct video_frame *frame, struct rtp *rtp_session);
void             format_video_header(struct video_frame *frame, int tile_idx, int buffer_idx,
                uint32_t *hdr);

struct tx *tx_init_h264(struct module *parent, unsigned mtu, enum tx_media_type media_type,
                const char *fec, const char *encryption, long long bitrate);
void tx_send_h264(struct tx *tx_session, struct video_frame *frame, struct rtp *rtp_session);
void tx_send_jpeg(struct tx *tx_session, struct video_frame *frame, struct rtp *rtp_session);

/**
 * Returns buffer ID to be sent with next tx_send() call
 */
int tx_get_buffer_id(struct tx *tx_session);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
void             audio_tx_send(struct tx *tx_session, struct rtp *rtp_session, const audio_frame2 *buffer);
void             audio_tx_send_standard(struct tx* tx, struct rtp *rtp_session, const audio_frame2 * buffer);
#endif

#endif // TRANSMIT_H_

