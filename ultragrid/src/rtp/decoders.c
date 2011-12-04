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
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "perf.h"
#include "rtp/fec.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/decoders.h"
#include "video_codec.h"
#include "video_decompress.h"
#include "video_display.h"
#include "vo_postprocess.h"

struct state_decoder;

struct video_frame * reconfigure_decoder(struct state_decoder * const decoder, struct video_desc desc,
                                struct tile_info tileinfo,
                                struct video_frame *frame);

enum decoder_type_t {
        UNSET,
        LINE_DECODER,
        EXTERNAL_DECODER
};

struct line_decoder {
        int                  base_offset; /* from the beginning of buffer */
        double               src_bpp;
        double               dst_bpp;
        int                  rshift;
        int                  gshift;
        int                  bshift;
        decoder_t            decode_line;
        unsigned int         dst_linesize; /* framebuffer pitch */
        unsigned int         dst_pitch; /* framebuffer pitch - it can be larger if SDL resolution is larger than data */
        unsigned int         src_linesize; /* display data pitch */
};

struct state_decoder {
        struct video_desc received_vid_desc;
        struct video_desc display_desc;
        
        /* requested values */
        int               requested_pitch;
        int               rshift, gshift, bshift;
        
        struct display   *display;
        codec_t          *native_codecs;
        int               native_count;
        
        /* actual values */
        enum decoder_type_t decoder_type; 
        struct {
                struct line_decoder *line_decoder;
                struct {                           /* OR - it is not union for easier freeing*/
                        const struct decode_from_to *ext_decoder_funcs;
                        int *total_bytes;
                        void *ext_decoder_state;
                        char **ext_recv_buffer;
                };
        };
        codec_t           out_codec;
        int               pitch;
        
        struct {
                struct vo_postprocess *postprocess;
                struct video_frame *pp_frame;
        };
        
        unsigned          merged_fb:1;
};

struct state_decoder *decoder_init(char *requested_mode)
{
        struct state_decoder *s;
        
        s = (struct state_decoder *) calloc(1, sizeof(struct state_decoder));
        s->native_codecs = NULL;
        
        if(requested_mode) {
                s->postprocess = vo_postprocess_init(requested_mode);
                if(!s->postprocess) {
                        fprintf(stderr, "Initializing postprocessor \"%s\" failed.\n", requested_mode);
                        return NULL;
                }
        } else {
                s->postprocess = NULL;
        }
        
        return s;
}

void decoder_register_video_display(struct state_decoder *decoder, struct display *display)
{
        int ret, i;
        decoder->display = display;
        
        free(decoder->native_codecs);
        decoder->native_count = 20 * sizeof(codec_t);
        decoder->native_codecs = (codec_t *)
                malloc(decoder->native_count * sizeof(codec_t));
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_CODECS, decoder->native_codecs, &decoder->native_count);
        decoder->native_count /= sizeof(codec_t);
        if(!ret) {
                error_with_code_msg(129, "Failed to query codecs from video display.");
        }
        
        /* next check if we didn't receive alias for UYVY */
        for(i = 0; i < decoder->native_count; ++i) {
                if(decoder->native_codecs[i] == Vuy2 ||
                                decoder->native_codecs[i] == DVS8)
                        error_with_code_msg(128, "Logic error: received alias for UYVY.");
        }
}

void decoder_destroy(struct state_decoder *decoder)
{
        if(decoder->ext_decoder_funcs) {
                decoder->ext_decoder_funcs->done(decoder->ext_decoder_state);
                decoder->ext_decoder_funcs = NULL;
        }
        if(decoder->ext_recv_buffer) {
                char **buf = decoder->ext_recv_buffer;
                while(*buf != NULL) {
                        free(*buf);
                        buf++;
                }
                free(decoder->total_bytes);
                free(decoder->ext_recv_buffer);
                decoder->ext_recv_buffer = NULL;
        }
        if(decoder->pp_frame) {
                vo_postprocess_done(decoder->postprocess);
                decoder->pp_frame = NULL;
        }
        free(decoder->native_codecs);
        free(decoder);
}

static codec_t choose_codec_and_decoder(struct state_decoder * const decoder, struct video_desc desc,
                                struct tile_info tile_info, codec_t *in_codec, decoder_t *decode_line)
{
        codec_t out_codec;
        *decode_line = NULL;
        *in_codec = desc.color_spec;
        
        /* first deal with aliases */
        if(*in_codec == DVS8 || *in_codec == Vuy2) {
                *in_codec = UYVY;
        }
        
        int native;
        /* first check if the codec is natively supported */
        for(native = 0; native < decoder->native_count; ++native)
        {
                out_codec = decoder->native_codecs[native];
                if(out_codec == DVS8 || out_codec == Vuy2)
                        out_codec = UYVY;
                if(*in_codec == out_codec) {
                        if((out_codec == DXT1 || out_codec == DXT1_YUV ||
                                        out_codec == DXT5)
                                        && (tile_info.x_count > 1 ||
                                        tile_info.y_count > 1))
                                continue; /* it is a exception, see NOTES #1 */
                        if(*in_codec == RGBA || /* another exception - we may change shifts */
                                        *in_codec == RGB)
                                continue;
                        
                        *decode_line = (decoder_t) memcpy;
                        decoder->decoder_type = LINE_DECODER;
                        
                        goto after_linedecoder_lookup;
                }
        }
        /* otherwise if we have line decoder */
        int trans;
        for(trans = 0; line_decoders[trans].line_decoder != NULL;
                                ++trans) {
                
                for(native = 0; native < decoder->native_count; ++native)
                {
                        out_codec = decoder->native_codecs[native];
                        if(out_codec == DVS8 || out_codec == Vuy2)
                                out_codec = UYVY;
                        if(*in_codec == line_decoders[trans].from &&
                                        out_codec == line_decoders[trans].to) {
                                                
                                *decode_line = line_decoders[trans].line_decoder;
                                
                                decoder->decoder_type = LINE_DECODER;
                                goto after_linedecoder_lookup;
                        }
                }
        }
        
after_linedecoder_lookup:

        /* we didn't find line decoder. So try now regular (aka DXT) decoder */
        if(*decode_line == NULL) {
                for(native = 0; native < decoder->native_count; ++native)
                {
                        int trans;
                        out_codec = decoder->native_codecs[native];
                        if(out_codec == DVS8 || out_codec == Vuy2)
                                out_codec = UYVY;
                                
                        for(trans = 0; decoders[trans].init != NULL;
                                        ++trans) {
                                if(*in_codec == decoders[trans].from &&
                                                out_codec == decoders[trans].to) {
                                        decoder->decoder_type = EXTERNAL_DECODER;
                                        decoder->ext_decoder_funcs = &decoders[trans];
                                        goto after_decoder_lookup;
                                }
                        }
                }
        }
after_decoder_lookup:

        if(decoder->decoder_type == UNSET) {
                error_with_code_msg(128, "Unable to find decoder for input codec!!!");
        }
        
        decoder->out_codec = out_codec;
        return out_codec;
}

struct video_frame * reconfigure_decoder(struct state_decoder * const decoder, struct video_desc desc,
                                struct tile_info tile_info,
                                struct video_frame *frame_display)
{
        codec_t out_codec, in_codec;
        decoder_t decode_line;
        struct video_frame *frame;
        
        assert(decoder != NULL);
        assert(decoder->native_codecs != NULL);
        
        free(decoder->line_decoder);
        decoder->line_decoder = NULL;
        decoder->decoder_type = UNSET;
        if(decoder->ext_decoder_funcs) {
                decoder->ext_decoder_funcs->done(decoder->ext_decoder_state);
                decoder->ext_decoder_funcs = NULL;
        }
        if(decoder->ext_recv_buffer) {
                char **buf = decoder->ext_recv_buffer;
                while(*buf != NULL) {
                        free(*buf);
                        buf++;
                }
                free(decoder->total_bytes);
                free(decoder->ext_recv_buffer);
                decoder->ext_recv_buffer = NULL;
        }
        if(decoder->pp_frame) {
                vo_postprocess_done(decoder->postprocess);
                decoder->pp_frame = NULL;
        }
        
        out_codec = choose_codec_and_decoder(decoder, desc, tile_info, &in_codec, &decode_line);
        if(decoder->postprocess) {
                decoder->pp_frame = vo_postprocess_reconfigure(decoder->postprocess, desc, tile_info);
                struct video_desc_ti desc_ti;
                vo_postprocess_get_out_desc(decoder->postprocess, &desc_ti);
                desc = desc_ti.desc;
                tile_info = desc_ti.ti;
        }
        
        struct video_desc cur_desc = desc;
        cur_desc.color_spec = out_codec;
        if(!video_desc_eq(decoder->display_desc, cur_desc))
        {
                /*
                 * TODO: put frame should be definitely here. On the other hand, we cannot be sure
                 * that vo driver is initialized so far:(
                 */
                //display_put_frame(decoder->display, frame);
                /* reconfigure VO and give it opportunity to pass us pitch */        
                display_reconfigure(decoder->display, cur_desc);
                frame_display = display_get_frame(decoder->display);
                decoder->display_desc = cur_desc;
        }
        if(decoder->postprocess) {
                frame = decoder->pp_frame;
        } else {
                frame = frame_display;
        }
        
        int len = sizeof(int);
        int ret;
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_RSHIFT,
                        &decoder->rshift, &len);
        if(!ret) {
                debug_msg("Failed to get properties from video driver.");
        }
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_GSHIFT,
                        &decoder->gshift, &len);
        if(!ret) {
                debug_msg("Failed to get properties from video driver.");
        }
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_BSHIFT,
                        &decoder->bshift, &len);
        if(!ret) {
                debug_msg("Failed to get properties from video driver.");
        }
        
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_BUF_PITCH,
                        &decoder->requested_pitch, &len);
        if(!ret) {
                debug_msg("Failed to get pitch from video driver.");
                decoder->requested_pitch = PITCH_DEFAULT;
        }
        
        if(!decoder->postprocess) {
                if(decoder->requested_pitch == PITCH_DEFAULT)
                        decoder->pitch = vc_get_linesize(desc.width / frame->grid_width, out_codec);
                else
                        decoder->pitch = decoder->requested_pitch;
                        
        } else {
                decoder->pitch = vc_get_linesize(desc.width, out_codec);
        }
        
        if(decoder->decoder_type == LINE_DECODER) {
                decoder->line_decoder = malloc(tile_info.x_count * tile_info.y_count *
                                        sizeof(struct line_decoder));                
                if(frame->grid_width == 1 && frame->grid_height == 1 && tile_info.x_count == 1 && tile_info.y_count == 1) {
                        struct line_decoder *out = &decoder->line_decoder[0];
                        out->base_offset = 0;
                        out->src_bpp = get_bpp(in_codec);
                        out->dst_bpp = get_bpp(out_codec);
                        out->rshift = decoder->rshift;
                        out->gshift = decoder->gshift;
                        out->bshift = decoder->bshift;
                
                        out->decode_line = decode_line;
                        out->dst_pitch = decoder->pitch;
                        out->src_linesize = vc_get_linesize(desc.width, in_codec);
                        out->dst_linesize = vc_get_linesize(desc.width, out_codec);
                        decoder->merged_fb = TRUE;
                } else if(frame->grid_width == 1 && frame->grid_height == 1
                                && (tile_info.x_count != 1 || tile_info.y_count != 1)) {
                        int x, y;
                        for(x = 0; x < tile_info.x_count; ++x) {
                                for(y = 0; y < tile_info.y_count; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x + 
                                                        tile_info.x_count * y];
                                        out->base_offset = y * (desc.height / tile_info.y_count)
                                                        * decoder->pitch + 
                                                        vc_get_linesize(x * desc.width / tile_info.x_count, out_codec);

                                        out->src_bpp = get_bpp(in_codec);
                                        out->dst_bpp = get_bpp(out_codec);

                                        out->rshift = decoder->rshift;
                                        out->gshift = decoder->gshift;
                                        out->bshift = decoder->bshift;
                
                                        out->decode_line = decode_line;

                                        out->dst_pitch = decoder->pitch;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width / tile_info.x_count, in_codec);
                                        out->dst_linesize =
                                                vc_get_linesize(desc.width / tile_info.x_count, out_codec);
                                }
                        }
                        decoder->merged_fb = TRUE;
                } else if(frame->grid_width == tile_info.x_count && frame->grid_height == tile_info.y_count) {
                        int x, y;
                        for(x = 0; x < tile_info.x_count; ++x) {
                                for(y = 0; y < tile_info.y_count; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x + 
                                                        tile_info.x_count * y];
                                        out->base_offset = 0;
                                        out->src_bpp = get_bpp(in_codec);
                                        out->dst_bpp = get_bpp(out_codec);
                                        out->rshift = decoder->rshift;
                                        out->gshift = decoder->gshift;
                                        out->bshift = decoder->bshift;
                
                                        out->decode_line = decode_line;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width / tile_info.x_count, in_codec);
                                        out->dst_pitch = 
                                                out->dst_linesize =
                                                vc_get_linesize(desc.width / tile_info.x_count, out_codec);
                                }
                        }
                        decoder->merged_fb = FALSE;
                }
        } else if (decoder->decoder_type == EXTERNAL_DECODER) {
                int buf_size;
                int i;
                
                desc.width /= tile_info.x_count;
                desc.height /= tile_info.y_count;
                decoder->ext_decoder_state = decoder->ext_decoder_funcs->init();
                buf_size = decoder->ext_decoder_funcs->reconfigure(decoder->ext_decoder_state, desc, 
                                decoder->rshift, decoder->gshift, decoder->bshift, decoder->pitch, out_codec);
                
                decoder->ext_recv_buffer = malloc((tile_info.x_count * tile_info.y_count + 1) * sizeof(char *));
                decoder->total_bytes = calloc(1, (tile_info.x_count * tile_info.y_count) * sizeof(int));
                for (i = 0; i < tile_info.x_count * tile_info.y_count; ++i)
                        decoder->ext_recv_buffer[i] = malloc(buf_size);
                decoder->ext_recv_buffer[i] = NULL;
                if(frame->grid_width == tile_info.x_count && frame->grid_height == tile_info.y_count) {
                        decoder->merged_fb = FALSE;
                } else {
                        decoder->merged_fb = TRUE;
                }
        }
        
        return frame_display;
}

struct node {
        struct node *next;
        int val;
};

struct linked_list {
        struct node *head;
};


struct linked_list  *ll_create()
{
        return (struct linked_list *) calloc(1, sizeof(struct linked_list));
}

void ll_insert(struct linked_list *ll, int val) {
        struct node *cur;
        struct node **ref;
        if(!ll->head) {
                ll->head = malloc(sizeof(struct node));
                ll->head->val = val;
                ll->head->next = NULL;
                return;
        }
        ref = &ll->head;
        cur = ll->head;
        while (cur != NULL) {
                if (val == cur->val) return;
                if (val < cur->val) {
                        struct node *new_node = malloc(sizeof(struct node));
                        (*ref) = new_node; 
                        new_node->val = val;
                        new_node->next = cur;
                        return;
                }
                ref = &cur->next;
                cur = cur->next;
        }
        struct node *new_node = malloc(sizeof(struct node));
        (*ref) = new_node; 
        new_node->val = val;
        new_node->next = NULL;
}

void ll_destroy(struct linked_list *ll) {
        struct node *cur = ll->head;
        struct node *tmp = tmp;

        while (cur != NULL) {
                tmp = cur->next;
                free(cur);
                cur = tmp;
        }
        free(ll);
}

int ll_count (struct linked_list *ll) {
        int ret = 0;
        struct node *cur = ll->head;
        while(cur != NULL) {
                ++ret;
                cur = cur->next;
        }
        return ret;
}

int decode_frame(struct coded_data *cdata, struct video_frame *frame, struct state_decoder *decoder)
{
        int ret = TRUE;
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
        char *data;
        uint32_t data_pos;
        int prints=0;
        double fps;
        struct tile *tile = NULL;

        struct fec_session **fecs = calloc(10, sizeof(struct fec_session *));
        uint16_t last_rtp_seq;
        struct linked_list *pckt_list = ll_create();
        uint32_t total_packets_sent = 0u;

        perf_record(UVP_DECODEFRAME, frame);

        if(!frame)
                return;

        if(decoder->decoder_type == EXTERNAL_DECODER) {
                memset(decoder->total_bytes, 0, sizeof(int) * 2); 
        }

        while (cdata != NULL) {
                pckt = cdata->data;
                hdr = (payload_hdr_t *) pckt->data;
                if(pckt->pt == 120) {
                                total_packets_sent = ntohl(* (uint32_t *) pckt->data);
                                cdata = cdata->nxt;
                                continue;
                }
                if(pckt->pt >= 98) {
                        struct fec_session *fec;
                        fec = fecs[pckt->pt - 98];
                        if(fec && last_rtp_seq < pckt->seq) {
                                        fec_restore_invalidate(fec);
                        }
                        last_rtp_seq = pckt->seq;
                        if(!fec) {
                                fec = fecs[pckt->pt - 98] = 
                                                fec_restore_init();
                                /* register the FEC packet */
                                fec_restore_start(fec, pckt->data);
                                cdata = cdata->nxt;
                                /* and jump to next */
                                continue;
                        } else {
                                int ret = FALSE;
                                rtp_packet *pckt_old = pckt;
                                /* try to restore packet */
                                ret = fec_restore_packet(fec, &hdr);
                                /* register current FEC packet */
                                fec_restore_start(fec, pckt_old->data);
                                /* if we didn't recovered any packet, jump to next */
                                if(!ret) {
                                        cdata = cdata->nxt;
                                        continue;
                                }
                                /* otherwise process the restored packet */
                        }
                } else {
                        int i = 0;
                        while (fecs[i]) {
                                if(fecs[i]) {
                                        if(last_rtp_seq >= pckt->seq) {
                                                fec_add_packet(fecs[i], hdr, (char *) hdr + sizeof(payload_hdr_t), ntohs(hdr->length));
                                        } else {
                                                fec_restore_invalidate(fecs[i]);
                                        }

                                        last_rtp_seq = pckt->seq;
                                }
                                i++;
                        }
                }
                data = (char *) hdr + sizeof(payload_hdr_t);
                width = ntohs(hdr->width);
                height = ntohs(hdr->height);
                color_spec = hdr->colorspc;
                len = ntohs(hdr->length);
                data_pos = ntohl(hdr->offset);
                fps = ntohl(hdr->fps)/65536.0;
                aux = ntohl(hdr->aux);
                tile_info = ntoh_uint2tileinfo(hdr->tileinfo);
                
                ll_insert(pckt_list, (tile_info.pos_x + tile_info.pos_y * tile_info.x_count) * (1<<24) + data_pos);

                if(aux & AUX_TILED) {
                        width = width * tile_info.x_count;
                        height = height * tile_info.y_count;
                } else {
                        tile_info.x_count = 1;
                        tile_info.y_count = 1;
                        tile_info.pos_x = 0;
                        tile_info.pos_y = 0;
                }
                
                /* Critical section 
                 * each thread *MUST* wait here if this condition is true
                 */
                if (!(decoder->received_vid_desc.width == width &&
                      decoder->received_vid_desc.height == height &&
                      decoder->received_vid_desc.color_spec == color_spec &&
                      decoder->received_vid_desc.aux == aux &&
                      decoder->received_vid_desc.fps == fps
                      )) {
                        decoder->received_vid_desc.width = width;
                        decoder->received_vid_desc.height = height;
                        decoder->received_vid_desc.color_spec = color_spec;
                        decoder->received_vid_desc.aux = aux;
                        decoder->received_vid_desc.fps = fps;

                        frame = reconfigure_decoder(decoder, decoder->received_vid_desc,
                                        tile_info, frame);
                }
                
                if(!decoder->postprocess) {
                        if (aux & AUX_TILED && !decoder->merged_fb) {
                                tile = tile_get(frame, tile_info.pos_x, tile_info.pos_y);
                        } else {
                                tile = tile_get(frame, 0, 0);
                        }
                } else {
                        if (aux & AUX_TILED && !decoder->merged_fb) {
                                tile = tile_get(decoder->pp_frame, tile_info.pos_x, tile_info.pos_y);
                        } else {
                                tile = tile_get(decoder->pp_frame, 0, 0);
                        }
                }
                
                if(decoder->decoder_type == LINE_DECODER) {
                        struct line_decoder *line_decoder = 
                                &decoder->line_decoder[tile_info.pos_x +
                                        tile_info.pos_y * tile_info.x_count];
                        
                        /* End of critical section */
        
                        /* MAGIC, don't touch it, you definitely break it 
                         *  *source* is data from network, *destination* is frame buffer
                         */
        
                        /* compute Y pos in source frame and convert it to 
                         * byte offset in the destination frame
                         */
                        int y = (data_pos / line_decoder->src_linesize) * line_decoder->dst_pitch;
        
                        /* compute X pos in source frame */
                        int s_x = data_pos % line_decoder->src_linesize;
        
                        /* convert X pos from source frame into the destination frame.
                         * it is byte offset from the beginning of a line. 
                         */
                        int d_x = ((int)((s_x) / line_decoder->src_bpp)) *
                                line_decoder->dst_bpp;
        
                        /* pointer to data payload in packet */
                        source = (unsigned char*)(data);
        
                        /* copy whole packet that can span several lines. 
                         * we need to clip data (v210 case) or center data (RGBA, R10k cases)
                         */
                        while (len > 0) {
                                /* len id payload length in source BPP
                                 * decoder needs len in destination BPP, so convert it 
                                 */                        
                                int l = ((int)(len / line_decoder->src_bpp)) * line_decoder->dst_bpp;
        
                                /* do not copy multiple lines, we need to 
                                 * copy (& clip, center) line by line 
                                 */
                                if (l + d_x > (int) line_decoder->dst_linesize) {
                                        l = line_decoder->dst_linesize - d_x;
                                }
        
                                /* compute byte offset in destination frame */
                                offset = y + d_x;
        
                                /* watch the SEGV */
                                if (l + line_decoder->base_offset + offset <= tile->data_len) {
                                        /*decode frame:
                                         * we have offset for destination
                                         * we update source contiguously
                                         * we pass {r,g,b}shifts */
                                        line_decoder->decode_line((unsigned char*)tile->data + line_decoder->base_offset + offset, source, l,
                                                       line_decoder->rshift, line_decoder->gshift,
                                                       line_decoder->bshift);
                                        /* we decoded one line (or a part of one line) to the end of the line
                                         * so decrease *source* len by 1 line (or that part of the line */
                                        len -= line_decoder->src_linesize - s_x;
                                        /* jump in source by the same amount */
                                        source += line_decoder->src_linesize - s_x;
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
                                d_x = 0;        /* next line from beginning */
                                s_x = 0;
                                y += line_decoder->dst_pitch;  /* next line */
                        }
                } else if(decoder->decoder_type == EXTERNAL_DECODER) {
                        int pos = tile_info.pos_x + tile_info.x_count * tile_info.pos_y;
                        memcpy(decoder->ext_recv_buffer[pos] + data_pos, (unsigned char*)(data),
                                len);
                        decoder->total_bytes[pos] = max(decoder->total_bytes[pos], data_pos + len);
                }

                cdata = cdata->nxt;
        }

        if(total_packets_sent && total_packets_sent != ll_count(pckt_list)) {
                fprintf(stderr, "Frame incomplete: expected %u packets, got %u.\n",
                                (unsigned int) total_packets_sent, (unsigned int) ll_count(pckt_list));
                ret = FALSE;
                goto cleanup;
        }
        
        if(decoder->decoder_type == EXTERNAL_DECODER) {
                int tile_width = decoder->received_vid_desc.width / tile_info.x_count;
                int tile_height = decoder->received_vid_desc.height / tile_info.y_count;
                int x, y;
                for (x = 0; x < tile_info.x_count; ++x) {
                        for (y = 0; y < tile_info.y_count; ++y) {
                                char *out;
                                if(decoder->merged_fb) {
                                        tile = tile_get(frame, 0, 0);
                                        out = tile->data + y * decoder->pitch * tile_height +
                                                vc_get_linesize(tile_width, decoder->out_codec) * x;
                                } else {
                                        tile = tile_get(frame, x, y);
                                        out = tile->data;
                                }
                                decoder->ext_decoder_funcs->decompress(decoder->ext_decoder_state,
                                                (unsigned char *) out,
                                                (unsigned char *) decoder->ext_recv_buffer[x + tile_info.x_count * y],
                                                decoder->total_bytes[x + tile_info.x_count * y]);
                        }
                }
        }
        
        if(decoder->postprocess) {
                vo_postprocess(decoder->postprocess,
                               decoder->pp_frame,
                               frame,
                               decoder->requested_pitch);
        }

cleanup:
        ll_destroy(pckt_list);
        int i = 0;
        while (fecs[i]) {
                fec_restore_destroy(fecs[i]);
                ++i;
        }

        return ret;
}
