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
#include "host.h"
#include "perf.h"
#include "rtp/xor.h"
#include "rtp/rtp.h"
#include "rtp/rtp_callback.h"
#include "rtp/pbuf.h"
#include "rtp/decoders.h"
#include "video.h"
#include "video_codec.h"
#include "video_decompress.h"
#include "video_display.h"
#include "vo_postprocess.h"

struct state_decoder;

struct video_frame * reconfigure_decoder(struct state_decoder * const decoder, struct video_desc desc,
                                struct video_frame *frame);
typedef void (*change_il_t)(char *dst, char *src, int linesize, int height);

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
        unsigned int      max_substreams;
        
        struct display   *display;
        codec_t          *native_codecs;
        size_t            native_count;
        enum interlacing_t    *disp_supported_il;
        size_t            disp_supported_il_cnt;
        change_il_t       change_il;
        
        /* actual values */
        enum decoder_type_t decoder_type; 
        struct {
                struct line_decoder *line_decoder;
                struct {                           /* OR - it is not union for easier freeing*/
                        struct state_decompress *ext_decoder;
                        unsigned int *total_bytes;
                        char **ext_recv_buffer;
                };
        };
        codec_t           out_codec;
        int               pitch;
        
        struct {
                struct vo_postprocess_state *postprocess;
                struct video_frame *pp_frame;
                int pp_output_frames_count;
        };

        unsigned int      video_mode;
        
        unsigned          merged_fb:1;
};

struct state_decoder *decoder_init(char *requested_mode, char *postprocess)
{
        struct state_decoder *s;
        
        s = (struct state_decoder *) calloc(1, sizeof(struct state_decoder));
        s->native_codecs = NULL;
        s->disp_supported_il = NULL;
        s->postprocess = NULL;
        s->change_il = NULL;
        s->video_mode = VIDEO_NORMAL;
        
        if(requested_mode) {
                /* these are data comming from newtork ! */
                if(strcasecmp(requested_mode, "help") == 0) {
                        printf("Video mode options\n\n");
                        printf("-M {tiled-4K | 3D | dual-link }\n");
                        free(s);
                        exit_uv(129);
                        return NULL;
                } else if(strcasecmp(requested_mode, "tiled-4K") == 0) {
                        s->video_mode = VIDEO_4K;
                } else if(strcasecmp(requested_mode, "3D") == 0) {
                        s->video_mode = VIDEO_STEREO;
                } else if(strcasecmp(requested_mode, "dual-link") == 0) {
                        s->video_mode = VIDEO_DUAL;
                } else {
                        fprintf(stderr, "[decoder] Unknown video mode (see -M help)\n");
                        free(s);
                        exit_uv(129);
                        return NULL;
                }
        }
        s->max_substreams = get_video_mode_tiles_x(s->video_mode)
                        * get_video_mode_tiles_y(s->video_mode);

        if(postprocess) {
                s->postprocess = vo_postprocess_init(postprocess);
                if(strcmp(postprocess, "help") == 0) {
                        exit_uv(0);
                        return NULL;
                }
                if(!s->postprocess) {
                        fprintf(stderr, "Initializing postprocessor \"%s\" failed.\n", postprocess);
                        free(s);
                        exit_uv(129);
                        return NULL;
                }
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
                fprintf(stderr, "Failed to query codecs from video display.\n");
                exit_uv(129);
                return;
        }
        
        /* next check if we didn't receive alias for UYVY */
        for(i = 0; i < (int) decoder->native_count; ++i) {
                assert(decoder->native_codecs[i] != Vuy2 &&
                                decoder->native_codecs[i] != DVS8);
        }


        free(decoder->disp_supported_il);
        decoder->disp_supported_il_cnt = 20 * sizeof(enum interlacing_t);
        decoder->disp_supported_il = malloc(decoder->disp_supported_il_cnt);
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_SUPPORTED_IL_MODES, decoder->disp_supported_il, &decoder->disp_supported_il_cnt);
        if(ret) {
                decoder->disp_supported_il_cnt /= sizeof(enum interlacing_t);
        } else {
                enum interlacing_t tmp[] = { PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME}; /* default if not said othervise */
                memcpy(decoder->disp_supported_il, tmp, sizeof(tmp));
                decoder->disp_supported_il_cnt = sizeof(tmp) / sizeof(enum interlacing_t);
        }
}

void decoder_destroy(struct state_decoder *decoder)
{
        if(!decoder)
                return;

        if(decoder->ext_decoder) {
                decompress_done(decoder->ext_decoder);
                decoder->ext_decoder = NULL;
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
        free(decoder->disp_supported_il);
        free(decoder);
}

static codec_t choose_codec_and_decoder(struct state_decoder * const decoder, struct video_desc desc,
                                codec_t *in_codec, decoder_t *decode_line)
{
        codec_t out_codec = (codec_t) -1;
        *decode_line = NULL;
        *in_codec = desc.color_spec;
        
        /* first deal with aliases */
        if(*in_codec == DVS8 || *in_codec == Vuy2) {
                *in_codec = UYVY;
        }
        
        size_t native;
        /* first check if the codec is natively supported */
        for(native = 0u; native < decoder->native_count; ++native)
        {
                out_codec = decoder->native_codecs[native];
                if(out_codec == DVS8 || out_codec == Vuy2)
                        out_codec = UYVY;
                if(*in_codec == out_codec) {
                        if((out_codec == DXT1 || out_codec == DXT1_YUV ||
                                        out_codec == DXT5)
                                        && decoder->video_mode != VIDEO_NORMAL)
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
                                
                        for(trans = 0; trans < decoders_for_codec_count;
                                        ++trans) {
                                if(*in_codec == decoders_for_codec[trans].from &&
                                                out_codec == decoders_for_codec[trans].to) {
                                        decoder->ext_decoder = decompress_init(decoders_for_codec[trans].decompress_index);
                                        if(!decoder->ext_decoder) {
                                                debug_msg("Decompressor with magic %x was not found.\n");
                                                continue;
                                        }
                                        decoder->decoder_type = EXTERNAL_DECODER;

                                        goto after_decoder_lookup;
                                }
                        }
                }
        }
after_decoder_lookup:

        if(decoder->decoder_type == UNSET) {
                fprintf(stderr, "Unable to find decoder for input codec!!!\n");
                exit_uv(128);
                return (codec_t) -1;
        }
        
        decoder->out_codec = out_codec;
        return out_codec;
}

static change_il_t select_il_func(enum interlacing_t in_il, enum interlacing_t *supported, int il_out_cnt, /*out*/ enum interlacing_t *out_il)
{
        struct transcode_t { enum interlacing_t in; enum interlacing_t out; change_il_t func; };

        struct transcode_t transcode[] = {
                {UPPER_FIELD_FIRST, INTERLACED_MERGED, il_upper_to_merged},
                {INTERLACED_MERGED, UPPER_FIELD_FIRST, il_merged_to_upper}
        };

        int i;
        /* first try to check if it can be nativelly displayed */
        for (i = 0; i < il_out_cnt; ++i) {
                if(in_il == supported[i]) {
                        *out_il = in_il;
                        return NULL;
                }
        }

        for (i = 0; i < il_out_cnt; ++i) {
                size_t j;
                for (j = 0; j < sizeof(transcode) / sizeof(struct transcode_t); ++j) {
                        if(in_il == transcode[j].in && supported[i] == transcode[j].out) {
                                *out_il = transcode[j].out;
                                return transcode[j].func;
                        }
                }
        }

        return NULL;
}

struct video_frame * reconfigure_decoder(struct state_decoder * const decoder, struct video_desc desc,
                                struct video_frame *frame_display)
{
        codec_t out_codec, in_codec;
        decoder_t decode_line;
        enum interlacing_t display_il = 0;
        struct video_frame *frame;
        int render_mode;

        assert(decoder != NULL);
        assert(decoder->native_codecs != NULL);
        
        free(decoder->line_decoder);
        decoder->line_decoder = NULL;
        decoder->decoder_type = UNSET;
        if(decoder->ext_decoder) {
                decompress_done(decoder->ext_decoder);
                decoder->ext_decoder = NULL;
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

        desc.tile_count = get_video_mode_tiles_x(decoder->video_mode)
                        * get_video_mode_tiles_y(decoder->video_mode);
        
        out_codec = choose_codec_and_decoder(decoder, desc, &in_codec, &decode_line);
        if(out_codec == (codec_t) -1)
                return NULL;
        struct video_desc display_desc = desc;

        if(decoder->postprocess) {
                struct video_desc pp_desc = desc;
                pp_desc.color_spec = out_codec;
                vo_postprocess_reconfigure(decoder->postprocess, pp_desc);
                decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
                vo_postprocess_get_out_desc(decoder->postprocess, &display_desc, &render_mode, &decoder->pp_output_frames_count);
        }
        
        if(!is_codec_opaque(out_codec)) {
                decoder->change_il = select_il_func(desc.interlacing, decoder->disp_supported_il, decoder->disp_supported_il_cnt, &display_il);
        } else {
                decoder->change_il = NULL;
        }


        size_t len = sizeof(int);
        int ret;

        int display_mode;
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_VIDEO_MODE,
                        &display_mode, &len);
        if(!ret) {
                debug_msg("Failed to get video display mode.");
                display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        }

        if (!decoder->postprocess) { /* otherwise we need postprocessor mode, which we obtained before */
                render_mode = display_mode;
        }

        display_desc.color_spec = out_codec;
        display_desc.interlacing = display_il;
        if(!decoder->postprocess && display_mode == DISPLAY_PROPERTY_VIDEO_MERGED) {
                display_desc.width *= get_video_mode_tiles_x(decoder->video_mode);
                display_desc.height *= get_video_mode_tiles_y(decoder->video_mode);
                display_desc.tile_count = 1;
        }

        if(!video_desc_eq(decoder->display_desc, display_desc))
        {
                int ret;
                /*
                 * TODO: put frame should be definitely here. On the other hand, we cannot be sure
                 * that vo driver is initialized so far:(
                 */
                //display_put_frame(decoder->display, frame);
                /* reconfigure VO and give it opportunity to pass us pitch */        
                ret = display_reconfigure(decoder->display, display_desc);
                if(!ret) {
                        fprintf(stderr, "[decoder] Unable to reconfigure display.\n");
                        exit_uv(128);
                        return NULL;
                }
                frame_display = display_get_frame(decoder->display);
                decoder->display_desc = display_desc;
        }
        if(decoder->postprocess) {
                frame = decoder->pp_frame;
        } else {
                frame = frame_display;
        }
        
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_RSHIFT,
                        &decoder->rshift, &len);
        if(!ret) {
                debug_msg("Failed to get rshift property from video driver.\n");
                decoder->rshift = 0;
        }
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_GSHIFT,
                        &decoder->gshift, &len);
        if(!ret) {
                debug_msg("Failed to get gshift property from video driver.\n");
                decoder->gshift = 8;
        }
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_BSHIFT,
                        &decoder->bshift, &len);
        if(!ret) {
                debug_msg("Failed to get bshift property from video driver.\n");
                decoder->bshift = 16;
        }
        
        ret = display_get_property(decoder->display, DISPLAY_PROPERTY_BUF_PITCH,
                        &decoder->requested_pitch, &len);
        if(!ret) {
                debug_msg("Failed to get pitch from video driver.\n");
                decoder->requested_pitch = PITCH_DEFAULT;
        }
        



        int linewidth;
        if(render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                linewidth = desc.width; 
        } else {
                linewidth = desc.width * get_video_mode_tiles_x(decoder->video_mode);
        }


        if(!decoder->postprocess) {
                if(decoder->requested_pitch == PITCH_DEFAULT)
                        decoder->pitch = vc_get_linesize(linewidth, out_codec);
                else
                        decoder->pitch = decoder->requested_pitch;
        } else {
                decoder->pitch = vc_get_linesize(linewidth, out_codec);
        }

        int src_x_tiles = get_video_mode_tiles_x(decoder->video_mode);
        int src_y_tiles = get_video_mode_tiles_y(decoder->video_mode);
        
        if(decoder->decoder_type == LINE_DECODER) {
                decoder->line_decoder = malloc(src_x_tiles * src_y_tiles *
                                        sizeof(struct line_decoder));                
                if(render_mode == DISPLAY_PROPERTY_VIDEO_MERGED && decoder->video_mode == VIDEO_NORMAL) {
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
                } else if(render_mode == DISPLAY_PROPERTY_VIDEO_MERGED
                                && decoder->video_mode != VIDEO_NORMAL) {
                        int x, y;
                        for(x = 0; x < src_x_tiles; ++x) {
                                for(y = 0; y < src_y_tiles; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x + 
                                                        src_x_tiles * y];
                                        out->base_offset = y * (desc.height)
                                                        * decoder->pitch + 
                                                        vc_get_linesize(x * desc.width, out_codec);

                                        out->src_bpp = get_bpp(in_codec);
                                        out->dst_bpp = get_bpp(out_codec);

                                        out->rshift = decoder->rshift;
                                        out->gshift = decoder->gshift;
                                        out->bshift = decoder->bshift;
                
                                        out->decode_line = decode_line;

                                        out->dst_pitch = decoder->pitch;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width, in_codec);
                                        out->dst_linesize =
                                                vc_get_linesize(desc.width, out_codec);
                                }
                        }
                        decoder->merged_fb = TRUE;
                } else if (render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
                        int x, y;
                        for(x = 0; x < src_x_tiles; ++x) {
                                for(y = 0; y < src_y_tiles; ++y) {
                                        struct line_decoder *out = &decoder->line_decoder[x + 
                                                        src_x_tiles * y];
                                        out->base_offset = 0;
                                        out->src_bpp = get_bpp(in_codec);
                                        out->dst_bpp = get_bpp(out_codec);
                                        out->rshift = decoder->rshift;
                                        out->gshift = decoder->gshift;
                                        out->bshift = decoder->bshift;
                
                                        out->decode_line = decode_line;
                                        out->src_linesize =
                                                vc_get_linesize(desc.width, in_codec);
                                        out->dst_pitch = 
                                                out->dst_linesize =
                                                vc_get_linesize(desc.width, out_codec);
                                }
                        }
                        decoder->merged_fb = FALSE;
                }
        } else if (decoder->decoder_type == EXTERNAL_DECODER) {
                int buf_size;
                int i;
                
                buf_size = decompress_reconfigure(decoder->ext_decoder, desc, 
                                decoder->rshift, decoder->gshift, decoder->bshift, decoder->pitch , out_codec);
                if(!buf_size) {
                        return NULL;
                }
                decoder->ext_recv_buffer = malloc((src_x_tiles * src_y_tiles + 1) * sizeof(char *));
                decoder->total_bytes = calloc(1, (src_x_tiles * src_y_tiles) * sizeof(unsigned int));
                for (i = 0; i < src_x_tiles * src_y_tiles; ++i)
                        decoder->ext_recv_buffer[i] = malloc(buf_size);
                decoder->ext_recv_buffer[i] = NULL;
                if(render_mode == DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES) {
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

struct linked_list  *ll_create(void);
void ll_insert(struct linked_list *ll, int val);
void ll_destroy(struct linked_list *ll);
unsigned int ll_count (struct linked_list *ll);

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

unsigned int ll_count (struct linked_list *ll) {
        unsigned int ret = 0u;
        struct node *cur = ll->head;
        while(cur != NULL) {
                ++ret;
                cur = cur->next;
        }
        return ret;
}

int decode_frame(struct coded_data *cdata, void *decode_data)
{
        struct pbuf_video_data *pbuf_data = (struct pbuf_video_data *) decode_data;
        struct video_frame *frame = pbuf_data->frame_buffer;
        struct state_decoder *decoder = pbuf_data->decoder;

        int ret = TRUE;
        uint32_t width;
        uint32_t height;
        uint32_t offset;
        enum interlacing_t interlacing;
        int len;
        codec_t color_spec;
        rtp_packet *pckt;
        unsigned char *source;
        video_payload_hdr_t *hdr;
        char *data;
        uint32_t data_pos;
        int prints=0;
        double fps;
        struct tile *tile = NULL;
        uint32_t tmp;
        uint32_t substream;
        int fps_pt, fpsd, fd, fi;

        struct xor_session **xors = calloc(10, sizeof(struct xor_session *));
        uint16_t last_rtp_seq = 0;
        struct linked_list *pckt_list = ll_create();
        uint32_t total_packets_sent = 0u;

        perf_record(UVP_DECODEFRAME, frame);

        if(decoder->decoder_type == EXTERNAL_DECODER) {
                memset(decoder->total_bytes, 0, sizeof(unsigned int) * 2); 
        }

        while (cdata != NULL) {
                pckt = cdata->data;
                hdr = (video_payload_hdr_t *) pckt->data;
                if(pckt->pt == 120) {
                                total_packets_sent = ntohl(* (uint32_t *) pckt->data);
                                cdata = cdata->nxt;
                                continue;
                }
                if(pckt->pt >= 98) {
                        struct xor_session *xor;
                        xor = xors[pckt->pt - 98];
                        if(xor && last_rtp_seq < pckt->seq) {
                                        xor_restore_invalidate(xor);
                        }
                        last_rtp_seq = pckt->seq;
                        if(!xor) {
                                xor = xors[pckt->pt - 98] = 
                                                xor_restore_init();
                                /* register the xor packet */
                                xor_restore_start(xor, pckt->data);
                                cdata = cdata->nxt;
                                /* and jump to next */
                                continue;
                        } else {
                                int ret = FALSE;
                                uint16_t payload_len;
                                rtp_packet *pckt_old = pckt;
                                /* try to restore packet */
                                ret = xor_restore_packet(xor, (char **) &hdr, &payload_len);
                                /* register current xor packet */
                                xor_restore_start(xor, pckt_old->data);
                                /* if we didn't recovered any packet, jump to next */
                                if(!ret) {
                                        cdata = cdata->nxt;
                                        continue;
                                }
                                /* otherwise process the restored packet */
                                data = (char *) hdr + sizeof(video_payload_hdr_t);
                                len = payload_len;
                                goto packet_restored;
                        }
                } else {
                        int i = 0;
                        while (xors[i]) {
                                if(xors[i]) {
                                        if(last_rtp_seq >= pckt->seq) {
                                                xor_add_packet(xors[i], (char *) hdr, (char *) (hdr + sizeof(video_payload_hdr_t)), pckt->data_len - sizeof(video_payload_hdr_t));
                                        } else {
                                                xor_restore_invalidate(xors[i]);
                                        }

                                        last_rtp_seq = pckt->seq;
                                }
                                i++;
                        }
                }
                data = (char *) hdr + sizeof(video_payload_hdr_t);
                len = pckt->data_len - sizeof(video_payload_hdr_t);
packet_restored:
                width = ntohs(hdr->hres);
                height = ntohs(hdr->vres);
                color_spec = get_codec_from_fcc(ntohl(hdr->fourcc));
                data_pos = ntohl(hdr->offset);
                tmp = ntohl(hdr->substream_bufnum);


                substream = tmp >> 22;

                tmp = ntohl(hdr->il_fps);
                interlacing = (enum interlacing_t) (tmp >> 29);
                fps_pt = (tmp >> 19) & 0x3ff;
                fpsd = (tmp >> 15) & 0xf;
                fd = (tmp >> 14) & 0x1;
                fi = (tmp >> 13) & 0x1;

                fps = compute_fps(fps_pt, fpsd, fd, fi);

                if(substream >= decoder->max_substreams) {
                        fprintf(stderr, "[decoder] received substream ID %d. Expecting at most %d substreams. Did you set -M option?\n", substream, decoder->max_substreams);
                        exit_uv(1);
                        return FALSE;
                }


                ll_insert(pckt_list, substream * (1<<24) + data_pos);
                
                /* Critical section 
                 * each thread *MUST* wait here if this condition is true
                 */
                if (!(decoder->received_vid_desc.width == width &&
                      decoder->received_vid_desc.height == height &&
                      decoder->received_vid_desc.color_spec == color_spec &&
                      decoder->received_vid_desc.interlacing == interlacing  &&
                      //decoder->received_vid_desc.video_type == video_type &&
                      decoder->received_vid_desc.fps == fps
                      )) {
                        decoder->received_vid_desc.width = width;
                        decoder->received_vid_desc.height = height;
                        decoder->received_vid_desc.color_spec = color_spec;
                        decoder->received_vid_desc.interlacing = interlacing;
                        decoder->received_vid_desc.fps = fps;

                        frame = reconfigure_decoder(decoder, decoder->received_vid_desc,
                                        frame);
                }

                if(!frame) {
                        return FALSE;
                }
                
                if(!decoder->postprocess) {
                        if (!decoder->merged_fb) {
                                tile = vf_get_tile(frame, substream);
                        } else {
                                tile = vf_get_tile(frame, 0);
                        }
                } else {
                        if (!decoder->merged_fb) {
                                tile = vf_get_tile(decoder->pp_frame, substream);
                        } else {
                                tile = vf_get_tile(decoder->pp_frame, 0);
                        }
                }
                
                if(decoder->decoder_type == LINE_DECODER) {
                        struct line_decoder *line_decoder = 
                                &decoder->line_decoder[substream];
                        
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
                        //int pos = (substream >> 3) & 0x7 + (substream & 0x7) * frame->grid_width;
                        memcpy(decoder->ext_recv_buffer[substream] + data_pos, (unsigned char*)(pckt->data + sizeof(video_payload_hdr_t)),
                                len);
                        decoder->total_bytes[substream] = max(decoder->total_bytes[substream], data_pos + len);
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
                int tile_width = decoder->received_vid_desc.width; // get_video_mode_tiles_x(decoder->video_mode);
                int tile_height = decoder->received_vid_desc.height; // get_video_mode_tiles_y(decoder->video_mode);
                int x, y;
                struct video_frame *output;
                if(decoder->postprocess) {
                        output = decoder->pp_frame;
                } else {
                        output = frame;
                }
                for (x = 0; x < get_video_mode_tiles_x(decoder->video_mode); ++x) {
                        for (y = 0; y < get_video_mode_tiles_y(decoder->video_mode); ++y) {
                                int pos = x + get_video_mode_tiles_x(decoder->video_mode) * y;
                                char *out;
                                if(decoder->merged_fb) {
                                        tile = vf_get_tile(output, 0);
                                        // TODO: OK when rendering directly to display FB, otherwise, do not reflect pitch (we use PP)
                                        out = tile->data + y * decoder->pitch * tile_height +
                                                vc_get_linesize(tile_width, decoder->out_codec) * x;
                                } else {
                                        tile = vf_get_tile(output, x);
                                        out = tile->data;
                                }
                                decompress_frame(decoder->ext_decoder,
                                                (unsigned char *) out,
                                                (unsigned char *) decoder->ext_recv_buffer[pos],
                                                decoder->total_bytes[pos]);
                        }
                }
        }
        
        if(decoder->postprocess) {
                int i;
                vo_postprocess(decoder->postprocess,
                               decoder->pp_frame,
                               frame,
                               decoder->pitch);
                for (i = 1; i < decoder->pp_output_frames_count; ++i) {
                        display_put_frame(decoder->display, (char *) frame);
                        frame = display_get_frame(decoder->display);
                        vo_postprocess(decoder->postprocess,
                                       NULL,
                                       frame,
                                       decoder->pitch);
                }

                /* get new postprocess frame */
                decoder->pp_frame = vo_postprocess_getf(decoder->postprocess);
        }

        if(decoder->change_il) {
                unsigned int i;
                for(i = 0; i < frame->tile_count; ++i) {
                        struct tile *tile = vf_get_tile(frame, i);
                        decoder->change_il(tile->data, tile->data, vc_get_linesize(tile->width, decoder->out_codec), tile->height);
                }
        }

cleanup:
        ll_destroy(pckt_list);
        int i = 0;
        while (xors[i]) {
                xor_restore_destroy(xors[i]);
                ++i;
        }

        return ret;
}
