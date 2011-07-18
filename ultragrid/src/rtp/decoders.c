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

void decode_frame(struct coded_data *cdata, struct video_frame *frame)
{
        uint32_t width;
        uint32_t height;
        uint32_t offset;
        uint32_t aux;
        struct tile_info tile_info;
        int len;
        codec_t color_spec;
        rtp_packet *pckt;
        unsigned char *source;
        payload_hdr_t *hdr;
        uint32_t data_pos;
        int prints=0;
        double fps;
        struct video_frame *tile, *data;

        if(!frame)
                return;

        tile = malloc(sizeof(struct video_frame));
        data = tile;

        while (cdata != NULL) {
                pckt = cdata->data;
                hdr = (payload_hdr_t *) pckt->data;
                width = ntohs(hdr->width);
                height = ntohs(hdr->height);
                color_spec = hdr->colorspc;
                len = ntohs(hdr->length);
                data_pos = ntohl(hdr->offset);
                fps = ntohl(hdr->fps)/65536.0;
                aux = ntohl(hdr->aux);
                tile_info = ntoh_uint2tileinfo(hdr->tileinfo);

                /* Critical section 
                 * each thread *MUST* wait here if this condition is true
                 */
                if (!(frame->width == (aux & AUX_TILED ? width * tile_info.x_count : width) &&
                      frame->height == (aux & AUX_TILED ? height * tile_info.y_count : height) &&
                      frame->color_spec == color_spec &&
                      frame->aux == aux &&
                      frame->fps == fps
                      )) {
                        int frame_width = aux & AUX_TILED ? width * tile_info.x_count : width;
                        int frame_height = aux & AUX_TILED ? height * tile_info.y_count : height;
                        frame->reconfigure(frame->state, frame_width,
                                        frame_height,
                                        color_spec, fps, aux);
                        frame->src_linesize =
                            vc_getsrc_linesize(frame->width, color_spec);
                }
                if (aux & AUX_TILED) {
                        frame->get_sub_frame(frame->state,
                                        tile_info.pos_x * frame->width / tile_info.x_count,
                                        tile_info.pos_y * frame->height / tile_info.y_count,
                                        frame->width / tile_info.x_count,
                                        frame->height / tile_info.y_count,
                                        data);
                        tile = data;
                } else {
                        tile = frame;
                }
                /* End of critical section */

                /* MAGIC, don't touch it, you definitely break it 
                 *  *source* is data from network, *destination* is frame buffer
                 */

                /* compute Y pos in source frame and convert it to 
                 * byte offset in the destination frame
                 */
                int y = (data_pos / tile->src_linesize) * tile->dst_pitch;

                /* compute X pos in source frame */
                int s_x = data_pos % tile->src_linesize;

                /* convert X pos from source frame into the destination frame.
                 * it is byte offset from the beginning of a line. 
                 */
                int d_x = tile->dst_x_offset + ((int)((s_x) / tile->src_bpp)) *
                        tile->dst_bpp;

                /* pointer to data payload in packet */
                source = (unsigned char*)(pckt->data + sizeof(payload_hdr_t));

                /* copy whole packet that can span several lines. 
                 * we need to clip data (v210 case) or center data (RGBA, R10k cases)
                 */
                while (len > 0) {
                        /* len id payload length in source BPP
                         * decoder needs len in destination BPP, so convert it 
                         */                        
                        int l = ((int)(len / tile->src_bpp)) * tile->dst_bpp;

                        /* do not copy multiple lines, we need to 
                         * copy (& clip, center) line by line 
                         */
                        if (l + d_x > (int)tile->dst_linesize) {
                                l = tile->dst_linesize - d_x;
                        }

                        /* compute byte offset in destination frame */
                        offset = y + d_x;

                        /* watch the SEGV */
                        if (l + offset <= tile->data_len) {
                                /*decode frame:
                                 * we have offset for destination
                                 * we update source contiguously
                                 * we pass {r,g,b}shifts */
                                tile->decoder((unsigned char*)tile->data + offset, source, l,
                                               tile->rshift, tile->gshift,
                                               tile->bshift);
                                /* we decoded one line (or a part of one line) to the end of the line
                                 * so decrease *source* len by 1 line (or that part of the line */
                                len -= tile->src_linesize - s_x;
                                /* jump in source by the same amount */
                                source += tile->src_linesize - s_x;
                        } else {
                                /* this should not ever happen as we call reconfigure before each packet
                                 * iff reconfigure is needed. But if it still happens, something is terribly wrong
                                 * say it loudly 
                                 */
                                if((prints % 100) == 0) {
                                        fprintf(stderr, "WARNING!! Discarding input data as frame buffer is too small.\n"
                                                        "Well this should not happened. Expect troubles pretty soon.\n");
                                }
                                prints++;
                                len = 0;
                        }
                        /* each new line continues from the beginning */
                        d_x = tile->dst_x_offset;        /* next line from beginning */
                        s_x = 0;
                        y += tile->dst_pitch;  /* next line */
                }

                cdata = cdata->nxt;
        }

        free(data);
}

