/*
 * AUTHOR:   Ladan Gharai/Colin Perkins
 * 
 * Copyright (c) 2003-2004 University of Southern California
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
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/decoders.h"

#define DXT_WIDTH 1920/4
#define DXT_DEPTH 8

static void 
copy_p2f (char *frame, rtp_packet *pckt)
{
	/* Copy 1 rtp packet to frame for uncompressed HDTV data. */
	/* We limit packets to having up to 10 payload headers... */
	char                    *offset;
	payload_hdr_t		*curr_hdr;
	payload_hdr_t		*hdr[10];
	int			 hdr_count = 0, i;
	int		 	 frame_offset = 0;
	char 			*base;
	int  			 len;

	/* figure out how many headers ? */
	curr_hdr = (payload_hdr_t *) pckt->data;
	while (1) {
		hdr[hdr_count++] = curr_hdr;
		if ((ntohs(curr_hdr->flags) & (1<<15)) != 0) {
				/* Last header... */
				break;
		}
		if (hdr_count == 10) {
				/* Out of space... */
			break;
		}
		curr_hdr++;
	}

        /* OK, now we can copy the data */
	offset=(char *) (pckt->data) + hdr_count * 8;
	for (i = 1; i < hdr_count + 1; i++) {
		unsigned int y=ntohs(hdr[i - 1]->y_offset);
                /*if(y < HD_HEIGHT/2) {
                        y = y *2;
                } else {
                        y = (y-HD_HEIGHT/2) * 2 + 1;
                }*/
		frame_offset = ((ntohs(hdr[i - 1]->x_offset) + y * HD_WIDTH)) * HD_DEPTH;
		base = frame + frame_offset;
		len  = ntohs(hdr[i - 1]->length);
		memcpy(base,offset,len);
		offset+=len;
	}
}

static void 
dxt_copy_p2f (char *frame, rtp_packet *pckt)
{
	/* Copy 1 rtp packet to frame for uncompressed HDTV data. */
	/* We limit packets to having up to 10 payload headers... */
	char                    *offset;
	payload_hdr_t		*curr_hdr;
	payload_hdr_t		*hdr[10];
	int			 hdr_count = 0, i;
	int		 	 frame_offset = 0;
	char 			*base;
	int  			 len;
	unsigned int		 y=0;

	/* figure out how many headers ? */
	curr_hdr = (payload_hdr_t *) pckt->data;
	while (1) {
		hdr[hdr_count++] = curr_hdr;
		if ((ntohs(curr_hdr->flags) & (1<<15)) != 0) {
				/* Last header... */
				break;
		}
		if (hdr_count == 10) {
				/* Out of space... */
			break;
		}
		curr_hdr++;
	}

        /* OK, now we can copy the data */
	offset=(char *) (pckt->data) + hdr_count * 8;
	for (i = 0; i < hdr_count ; i++) {
		y=ntohs(hdr[i]->y_offset);
                /*if(y < HD_HEIGHT/2) {
                        y = y *2;
                } else {
                        y = (y-HD_HEIGHT/2) * 2 + 1;
                }*/
		frame_offset = ((ntohs(hdr[i]->x_offset) + y * DXT_WIDTH)) * DXT_DEPTH;
		base = frame + frame_offset;
		len  = ntohs(hdr[i]->length);
		memcpy(base,offset,len);
		offset+=len;
	}
}

void
decode_frame(struct coded_data *cdata, char *frame, int compression)
{
	/* Given a list of coded_data, try to decode it. This is mostly  */
 	/* a placeholder function: once we have multiple codecs, it will */
	/* get considerably more content...                              */
	if(compression) {
		while (cdata != NULL) {
			dxt_copy_p2f(frame, cdata->data);
			cdata = cdata->nxt;
		}
	}else{
		while (cdata != NULL) {
			copy_p2f(frame, cdata->data);
			cdata = cdata->nxt;
		}
	}
}

