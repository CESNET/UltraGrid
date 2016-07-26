/**
 * @file    video_decoders.h
 * @author  Colin Perkins
 * @author  Ladan Gharai
 * @author  Martin Pulec <pulec@cesnet.cz>

 * @ingroup video_rtp_decoder
 *
 * @brief Video RTP decoder.
 */
/*
 * Copyright (c) 2003 University of Southern California
 * Copyright (c) 2005-2013 CESNET z.s.p.o.
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
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
/**
 * @defgroup video_rtp_decoder Video Decoder
 * @{
 */

#include "types.h"

struct coded_data;
struct display;
struct module;
struct state_video_decoder;
struct video_desc;
struct video_frame;
struct state_decompress;
struct tile;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int decode_video_frame(struct coded_data *received_data, void *decoder_data, struct pbuf_stats *stats);

struct state_video_decoder *video_decoder_init(struct module *parent, enum video_mode,
                struct display *display, const char *encryption);
void video_decoder_destroy(struct state_video_decoder *decoder);
bool video_decoder_register_display(struct state_video_decoder *decoder, struct display *display);
void video_decoder_remove_display(struct state_video_decoder *decoder);
bool parse_video_hdr(uint32_t *hdr, struct video_desc *desc);

/** @} */ // end of video_rtp_decoder

// used also by hd_rum_translator
bool init_decompress(codec_t in_codec, codec_t out_codec,
                struct state_decompress **state, int state_count);

#ifdef __cplusplus
}
#endif // __cplusplus

