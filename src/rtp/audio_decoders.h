/**
 * @file   rtp/audio_decoders.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2014 CESNET z.s.p.o.
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

#include "audio/audio.h"

#ifdef __cplusplus
extern "C" {
#endif

struct coded_data;

typedef struct audio_desc (*query_supported_format_t)(void *state, struct audio_desc prop);

int decode_audio_frame(struct coded_data *cdata, void *data, struct pbuf_stats *stats);
int decode_audio_frame_mulaw(struct coded_data *cdata, void *data, struct pbuf_stats *stats);
void *audio_decoder_init(char *audio_channel_map, const char *audio_scale,
                const char *encryption, query_supported_format_t q, void *q_state);
void audio_decoder_destroy(void *state);
double audio_decoder_get_volume(void *state);
void audio_decoder_increase_volume(void *state);
void audio_decoder_decrease_volume(void *state);
void audio_decoder_mute(void *state);

bool parse_audio_hdr(uint32_t *hdr, struct audio_desc *desc);

#ifdef __cplusplus
}
#endif

