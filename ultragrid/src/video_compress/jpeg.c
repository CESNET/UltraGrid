/*
 * FILE:    jpeg.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2011 CESNET z.s.p.o.
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

#include "config.h"
#include "config_unix.h"
#include "debug.h"
#include "x11_common.h"
#include "video_compress/jpeg.h"
#include "jpeg_compress/jpeg_encoder.h"
#include "jpeg_compress/jpeg_common.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>

struct compress_jpeg_state {
        struct jpeg_encoder *encoder;
        struct jpeg_encoder_parameters encoder_param;
        
        struct video_frame *out;
        int     jpeg_height;
        
        decoder_t decoder;
        char *decoded;
        unsigned int interlaced_input:1;
        unsigned int rgb:1;
        codec_t color_spec;
};

static void configure_with(struct compress_jpeg_state *s, struct video_frame *frame);

static void configure_with(struct compress_jpeg_state *s, struct video_frame *frame)
{
        int x, y;
	int h_align = 0;
        
        assert(tile_get(frame, 0, 0)->width % 4 == 0);
        s->out = vf_alloc(frame->grid_width, frame->grid_height);
        
        for (x = 0; x < frame->grid_width; ++x) {
                for (y = 0; y < frame->grid_height; ++y) {
                        if (tile_get(frame, x, y)->width != tile_get(frame, 0, 0)->width ||
                                        tile_get(frame, x, y)->width != tile_get(frame, 0, 0)->width)
                                error_with_code_msg(128,"[JPEG] Requested to compress tiles of different size!");
                                
                        tile_get(s->out, x, y)->width = tile_get(frame, 0, 0)->width;
                        tile_get(s->out, x, y)->height = tile_get(frame, 0, 0)->height;
                }
        }
        
        s->out->aux = frame->aux;
        s->out->fps = frame->fps;
        s->out->color_spec = s->color_spec;

        switch (frame->color_spec) {
                case RGB:
                        s->decoder = (decoder_t) memcpy;
                        s->rgb = TRUE;
                        break;
                case RGBA:
                        s->decoder = (decoder_t) vc_copylineRGBAtoRGB;
                        s->rgb = TRUE;
                        break;
                /* TODO: enable
                 * case R10k:
                        s->decoder = (decoder_t) vc_copyliner10k;
                        s->rgb = TRUE;
                        break;*/
                case UYVY:
                case Vuy2:
                case DVS8:
                        s->decoder = (decoder_t) memcpy;
                        s->rgb = FALSE;
                        break;
                case v210:
                        s->decoder = (decoder_t) vc_copylinev210;
                        s->rgb = FALSE;
                        break;
                case DVS10:
                        s->decoder = (decoder_t) vc_copylineDVS10;
                        s->rgb = FALSE;
                        break;
                case DPX10:        
                        s->decoder = (decoder_t) vc_copylineDPX10toRGB;
                        s->rgb = TRUE;
                        break;
                default:
                        error_with_code_msg(128, "Unknown codec: %d\n", frame->color_spec);
        }

        /* We will deinterlace the output frame */
        if(s->out->aux & AUX_INTERLACED)
                s->interlaced_input = TRUE;
        else
                s->interlaced_input = FALSE;
        s->out->aux &= ~AUX_INTERLACED;
        
        s->out->color_spec = JPEG;
        struct jpeg_image_parameters param_image;
        s->jpeg_height = (s->out->tiles[0].height + 7) / 8 * 8;
        param_image.width = s->out->tiles[0].width;
        param_image.height = s->jpeg_height;
        param_image.comp_count = 3;
        if(s->rgb) {
                param_image.color_space = JPEG_RGB;
                param_image.sampling_factor = JPEG_4_4_4;
        } else {
                param_image.color_space = JPEG_YUV;
                param_image.sampling_factor = JPEG_4_2_2;
        }
        
        s->encoder = jpeg_encoder_create(&param_image, &s->encoder_param);
        
        for (x = 0; x < frame->grid_width; ++x) {
                for (y = 0; y < frame->grid_height; ++y) {
                        tile_get(s->out, x, y)->data = (char *) malloc(s->out->tiles[0].width * s->jpeg_height * 3);
                        tile_get(s->out, x, y)->linesize = s->out->tiles[0].width * (param_image.color_space == JPEG_RGB ? 3 : 2);
                }
        }
        
        if(!s->encoder) {
                error_with_code_msg(128, "[DXT GLSL] Failed to create encoder.\n");
        }
        
        s->decoded = malloc(4 * s->out->tiles[0].width * s->jpeg_height);
}

void * jpeg_compress_init(char * opts)
{
        struct compress_jpeg_state *s;
        
        s = (struct compress_jpeg_state *) malloc(sizeof(struct compress_jpeg_state));

        s->out = NULL;
        s->decoded = NULL;
                
        if(opts && strcmp(opts, "help") == 0) {
                printf("JPEG comperssion usage:\n");
                printf("\t-c JPEG:[<quality>:<restart_interval>:[<cuda_device>]]\n");
                printf("\nCUDA devices:\n");
                jpeg_get_device_list();
                return NULL;
        }

        if(opts) {
                char *tok, *save_ptr = NULL;
                tok = strtok_r(opts, ":", &save_ptr);
                s->encoder_param.quality = atoi(tok);
                tok = strtok_r(NULL, ":", &save_ptr);
                if(!tok) {
                        fprintf(stderr, "[JPEG] Quality set but restart interval unset\n");
                        free(s);
                        return NULL;
                }
                s->encoder_param.restart_interval = atoi(tok);
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        int ret;
                        ret = jpeg_init_device(atoi(tok), TRUE);

                        if(ret != 0) {
                                fprintf(stderr, "[JPEG] initializing CUDA device %d failed.\n", atoi(tok));
                                return NULL;
                        }
                } else {
                        printf("Initializing CUDA device 0...\n");
                        int ret = jpeg_init_device(0, TRUE);
                        if(ret != 0) {
                                fprintf(stderr, "[JPEG] initializing default CUDA device failed.\n");
                                return NULL;
                        }
                }
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        fprintf(stderr, "[JPEG] WARNING: Trailing configuration parameters.\n");
                }
                        
        } else {
                jpeg_encoder_set_default_parameters(&s->encoder_param);
                printf("[JPEG] setting default encode parameters (%d, %d)\n", 
                                s->encoder_param.quality,
                                s->encoder_param.restart_interval);
        }
                
        s->encoder = NULL; /* not yet configured */

        return s;
}

struct video_frame * jpeg_compress(void *arg, struct video_frame * tx)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) arg;
        int i;
        unsigned char *line1, *line2;

        int x, y;
        
        if(!s->encoder)
                configure_with(s, tx);

        for (x = 0; x < tx->grid_width;  ++x) {
                for (y = 0; y < tx->grid_height;  ++y) {
                        struct tile *in_tile = tile_get(tx, x, y);
                        struct tile *out_tile = tile_get(s->out, x, y);
                        
                        line1 = (unsigned char *) in_tile->data;
                        line2 = (unsigned char *) s->decoded;
                        
                        for (i = 0; i < (int) in_tile->height; ++i) {
                                s->decoder(line2, line1, out_tile->linesize,
                                                0, 8, 16);
                                line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                                line2 += out_tile->linesize;
                        }
                        
                        line1 = out_tile->data + (in_tile->height - 1) * out_tile->linesize;
                        for( ; i < s->jpeg_height; ++i) {
                                memcpy(line2, line1, out_tile->linesize);
                                line2 += out_tile->linesize;
                        }
                        
                        if(s->interlaced_input)
                                vc_deinterlace((unsigned char *) s->decoded, out_tile->linesize,
                                                s->jpeg_height);
                        
                        uint8_t *compressed;
                        int size;
                        int ret;
                        ret = jpeg_encoder_encode(s->encoder, s->decoded, &compressed, &size);
                        
                        if(ret != 0)
                                return NULL;
                        
                        out_tile->data_len = size;
                        memcpy(out_tile->data, compressed, size);
                }
        }
        
        return s->out;
}

void jpeg_compress_done(void *arg)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) arg;
        
        if(s->out)
                free(s->out->tiles[0].data);
        vf_free(s->out);
        if(s->encoder)
                jpeg_encoder_destroy(s->encoder);
        
        free(s);
}
