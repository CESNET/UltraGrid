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
 * $Revision: 1.1.2.8 $
 * $Date: 2010/02/05 13:56:49 $
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
#include "video_codec.h"

//#define DEBUG 1
//#define DEBUG_TIMING 1

void
decode_frame(struct coded_data *cdata, struct video_frame *frame)
{
        uint32_t width;
        uint32_t height;
        uint32_t offset;
        int      len;
        codec_t color_spec;
        rtp_packet *pckt;
        unsigned char *source;
        payload_hdr_t   *hdr;
        uint32_t data_pos;
#ifdef DEBUG_TIMING
        struct timeval tv,tv1;
        int packets=0;
#endif
#ifdef DEBUG
        long pos=0;
#endif
#ifdef DEBUG_TIMING
        gettimeofday(&tv, NULL);
#endif

        while (cdata != NULL) {
#ifdef DEBUG_TIMING
                packets++;
#endif
                pckt = cdata->data;
                hdr = (payload_hdr_t *)pckt->data;
                width = ntohs(hdr->width);
                height = ntohs(hdr->height);
                color_spec = hdr->colorspc;
                len = ntohs(hdr->length);
                data_pos = ntohl(hdr->offset);

                /* Critical section 
                 * each thread *MUST* wait here if this condition is true
                 */
                if(!(frame->width == width &&
                     frame->height == height &&
                     frame->color_spec == color_spec)) {
                        frame->reconfigure(frame->state, width, height, color_spec);
                        frame->src_linesize = vc_getsrc_linesize(width, color_spec);
                }
                /* End of critical section */
        
#ifdef DEBUG
                fprintf(stdout, "Setup: src line size: %d, dst line size %d, pitch %d\n",
                        frame->src_linesize, frame->dst_linesize, frame->dst_pitch);
                int b=0;
#endif
                /* MAGIC, don't touch it, you definitely break it */
                int y = (data_pos / frame->src_linesize)*frame->dst_pitch;
                int s_x = data_pos % frame->src_linesize;
                int d_x = ((int)((s_x)/frame->src_bpp))*frame->dst_bpp;
                source = pckt->data + sizeof(payload_hdr_t);
#ifdef DEBUG
                fprintf(stdout, "Computed start x %d, %d (%d %d), start y %d\n", s_x, d_x, (int)(s_x/frame->src_bpp),
                        (int)(d_x/frame->dst_bpp),  y/frame->dst_linesize);
#endif
                while(len > 0){
                        int l = ((int)(len/frame->src_bpp))*frame->dst_bpp;
                        if(l + d_x > frame->dst_linesize) {
                                l = frame->dst_linesize - d_x;
                        }
                        offset = y + d_x;
                        if(l + offset < frame->data_len) {
#ifdef DEBUG
                                if(b < 5) {
                                        fprintf(stdout, "Computed offset: %d, original offset %d, memcpy length %d (pixels %d), original length %d, stored length %d, next line %d\n",
                                                        offset, data_pos, l, (int)(l/frame->dst_bpp), len, pos, offset + l);
                                        b++;
                                }   
#endif
                                frame->decoder(frame->data+offset, source, l, 
                                              frame->rshift, frame->gshift, frame->bshift);
                                len -= frame->src_linesize - s_x;
                                source += frame->src_linesize - s_x;
#ifdef DEBUG
                                data_pos += frame->src_linesize - s_x;
                                pos += l;
#endif
                        } else {
#ifdef DEBUG
                                fprintf(stderr, "Discarding data, framebuffer too small.\n");
#endif
                                len = 0;
                        }
                        d_x = 0; /* next line from beginning */
                        s_x = 0;
                        y += frame->dst_pitch; /* next line */
                }

		cdata = cdata->nxt;
	}
#ifdef DEBUG_TIMING
    gettimeofday(&tv1, NULL);
    fprintf(stdout, "Frame encoded in %fms, %d packets\n", (tv1.tv_usec - tv.tv_usec)/1000.0, packets);
#endif
#ifdef DEBUG
    fprintf(stdout, "Frame end\n");
#endif
}

