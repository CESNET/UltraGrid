/*
 * FILE:    sdp.h
 * AUTHORS: Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2018 CESNET, z. s. p. o.
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
 *      This product includes software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */
#ifndef __sdp_h
#define __sdp_h

#include "audio/types.h"
#include "types.h"

#ifdef __cplusplus
extern "C" {
#else
#include <stdbool.h>
#endif

#define DEFAULT_SDP_HTTP_PORT 8554

typedef void (*address_callback_t)(void *udata, const char *address);

struct sdp *new_sdp(int ip_version, const char *receiver);
int sdp_add_audio(struct sdp *sdp, int port, int sample_rate, int channels, audio_codec_t codec);
int sdp_add_video(struct sdp *sdp, int port, codec_t codec);
/**
 * @param sdp           SDP struct to be generated file to
 * @param sdp_file_name name of the created file, may be empty in which case
 *                      default is created and returned or "no" - no file is created
 * @param output        textual representation of sdp
 */
bool gen_sdp(struct sdp *sdp, const char *sdp_file_name);
bool sdp_run_http_server(struct sdp *sdp, int port, address_callback_t addr_callback, void *addr_callback_udata);
void sdp_stop_http_server(struct sdp *sdp);
void clean_sdp(struct sdp *sdp);

#ifdef __cplusplus
}
#endif

#endif
