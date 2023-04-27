/*
 * FILE:    sdp.h
 * AUTHORS: Gerard Castillo  <gerard.castillo@i2cat.net>
 *          Martin Pulec <pulec@cesnet.cz>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2018-2023 CESNET, z. s. p. o.
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

#define DEFAULT_SDP_COMPRESSION "libavcodec:encoder=libx264:subsampling=420:disable_intra_refresh:disable_interlaced_dct"

int sdp_set_options(const char *options);
void sdp_set_properties(const char *receiver, bool has_sdp_video, bool has_sdp_audio);

typedef void (*address_callback_t)(void *udata, const char *address);

int sdp_add_audio(bool ipv6, int port, int sample_rate, int channels, audio_codec_t codec, address_callback_t addr_callback, void *addr_callback_udata);
int sdp_add_video(bool ipv6, int port, codec_t codec, address_callback_t addr_callback, void *addr_callback_udata);

#ifdef __cplusplus
}
#endif

#endif
