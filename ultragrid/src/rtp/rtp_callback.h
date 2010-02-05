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
 * $Revision: 1.1.2.1 $
 * $Date: 2010/01/28 18:17:28 $
 *
 */

#include "host.h"

#define HD_WIDTH hd_size_x
#define HD_HEIGHT hd_size_y
#define HD_DEPTH hd_color_bpp


typedef struct {
    uint16_t    width;      /* pixels */
    uint16_t    height;     /* pixels */
    uint32_t    offset;     /* in bytes */
    uint16_t    length;     /* octets */
    uint8_t     colorspc;
    uint8_t     flags;
} payload_hdr_t;

/* FIXME: this is only needed because fdisplay() takes "struct display" as a parameter */
/*        should probably not have such tight coupling between modules                 */
#include "video_display.h"

void fdisplay(struct display *display_device, unsigned char *inframe);
void rtp_recv_callback(struct rtp *session, rtp_event *e);
int handle_with_buffer(struct rtp *session,rtp_event *e);
int check_for_frame_completion(struct rtp *);
void process_packet_for_display(char *);
void call_display_frame(void);
