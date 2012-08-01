/*
 * FILE:   rtp_callback.h
 * AUTHOR: Colin Perkins <csp@csperkins.org>
 *
 * Copyright (c) 2001-2003 University of Southern California
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
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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

#include "host.h"

/*
 * Packet formats:
 * http://www.sitola.cz/files/4K-packet-format.pdf
 */
#define PT_VIDEO 20
#define PT_AUDIO 21
#define PT_VIDEO_LDGM 22

/*
 * Video payload
 *
 * 1st word
 * bits 0 - 9 substream
 * bits 10 - 31 buffer
 *
 * 2nd word
 * bits 0 - 31 offset
 *
 * 3rd word
 * bits 0 - 31 length
 *
 * 4rd word
 * bits 0-15 horizontal resolution
 * bits 16-31 vertical resolution
 *
 * 5th word
 * bits 0 - 31 FourCC
 *
 * 6th word
 * bits 0 - 2 interlace flag
 * bits 3 - 12 FPS
 * bits 13 - 16 FPSd
 * bit 17 Fd
 * bit 18 Fi
 */
typedef uint32_t video_payload_hdr_t[6];

/*
 * Audio payload
 *
 * 1st word
 * bits 0 - 9 substream
 * bits 10 - 31 buffer
 *
 * 2nd word
 * bits 0 - 31 offset
 *
 * 3rd word
 * bits 0 - 31 length
 *
 * 4rd word
 * bits 0-5 audio quantization
 * bits 6-31 audio sample rate
 *
 * 5th word
 * bits 0 - 31 AudioTag
 */
typedef uint32_t audio_payload_hdr_t[5];

/*
 * LDGM video payload
 *
 * 1st word
 * bits 0 - 9 substream
 * bits 10 - 31 buffer
 *
 * 2nd word
 * bits 0 - 31 offset
 *
 * 3rd word
 * bits 0 - 31 length
 *
 * 4rd word
 * bits 0-12 K
 * bits 13-25 M
 * bits 26-31 C
 *
 * 5th word
 * bits 0 - 31 LDGM random generator seed
 */
typedef uint32_t ldgm_video_payload_hdr_t[5];

void rtp_recv_callback(struct rtp *session, rtp_event *e);
int handle_with_buffer(struct rtp *session,rtp_event *e);
int check_for_frame_completion(struct rtp *);
void process_packet_for_display(char *);
void call_display_frame(void);
