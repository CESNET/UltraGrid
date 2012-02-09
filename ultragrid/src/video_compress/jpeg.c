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
#include "host.h"
#include "video_compress/jpeg.h"
#include "libgpujpeg/gpujpeg_encoder.h"
#include "libgpujpeg/gpujpeg_common.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>

struct compress_jpeg_state {
        struct gpujpeg_encoder *encoder;
        struct gpujpeg_parameters encoder_param;
        
        struct video_frame *out;
        int     jpeg_height;
        
        decoder_t decoder;
        char *decoded;
        unsigned int interlaced_input:1;
        unsigned int rgb:1;
        codec_t color_spec;
};

static int configure_with(struct compress_jpeg_state *s, struct video_frame *frame);

static int configure_with(struct compress_jpeg_state *s, struct video_frame *frame)
{
        unsigned int x;
        
        s->out = vf_alloc(frame->tile_count);
        
        for (x = 0; x < frame->tile_count; ++x) {
                if (vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width ||
                                vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width) {
                        fprintf(stderr,"[JPEG] Requested to compress tiles of different size!");
                        exit_uv(129);
                        return FALSE;
                }
                        
                vf_get_tile(s->out, x)->width = vf_get_tile(frame, 0)->width;
                vf_get_tile(s->out, x)->height = vf_get_tile(frame, 0)->height;
        }
        
        s->out->interlacing = PROGRESSIVE;
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
                /* TODO: enable (we need R10k -> RGB)
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
                        fprintf(stderr, "[JPEG] Unknown codec: %d\n", frame->color_spec);
                        exit_uv(128);
                        return FALSE;
        }

        /* We will deinterlace the output frame */
        if(frame->interlacing == INTERLACED_MERGED)
                s->interlaced_input = TRUE;
        else if(frame->interlacing == PROGRESSIVE)
                s->interlaced_input = FALSE;
        else {
                fprintf(stderr, "Unsupported interlacing option: %s.\n", get_interlacing_description(frame->interlacing));
                exit_uv(128);
                return FALSE;
        }

	s->encoder_param.verbose = 0;

        if(s->rgb) {
                s->encoder_param.interleaved = 0;
                s->encoder_param.restart_interval = 8;
                /* LUMA */
                s->encoder_param.sampling_factor[0].horizontal = 1;
                s->encoder_param.sampling_factor[0].vertical = 1;
                /* Cb and Cr */
                s->encoder_param.sampling_factor[1].horizontal = 1;
                s->encoder_param.sampling_factor[1].vertical = 1;
                s->encoder_param.sampling_factor[2].horizontal = 1;
                s->encoder_param.sampling_factor[2].vertical = 1;
        } else {
                s->encoder_param.interleaved = 1;
                s->encoder_param.restart_interval = 2;
                /* LUMA */
                s->encoder_param.sampling_factor[0].horizontal = 2;
                s->encoder_param.sampling_factor[0].vertical = 1;
                /* Cb and Cr */
                s->encoder_param.sampling_factor[1].horizontal = 1;
                s->encoder_param.sampling_factor[1].vertical = 1;
                s->encoder_param.sampling_factor[2].horizontal = 1;
                s->encoder_param.sampling_factor[2].vertical = 1;
        }

        
        s->out->color_spec = JPEG;
        struct gpujpeg_image_parameters param_image;
        s->jpeg_height = (s->out->tiles[0].height + 7) / 8 * 8;
        param_image.width = s->out->tiles[0].width;
        param_image.height = s->jpeg_height;
        param_image.comp_count = 3;
        if(s->rgb) {
                param_image.color_space = GPUJPEG_RGB;
                param_image.sampling_factor = GPUJPEG_4_4_4;
        } else {
                param_image.color_space = GPUJPEG_YCBCR_ITU_R;
                param_image.sampling_factor = GPUJPEG_4_2_2;
        }
        
        s->encoder = gpujpeg_encoder_create(&s->encoder_param, &param_image);
        
        for (x = 0; x < frame->tile_count; ++x) {
                        vf_get_tile(s->out, x)->data = (char *) malloc(s->out->tiles[0].width * s->jpeg_height * 3);
                        vf_get_tile(s->out, x)->linesize = s->out->tiles[0].width * (param_image.color_space == GPUJPEG_RGB ? 3 : 2);

        }
        
        if(!s->encoder) {
                fprintf(stderr, "[DXT GLSL] Failed to create encoder.\n");
                exit_uv(128);
                return FALSE;
        }
        
        s->decoded = malloc(4 * s->out->tiles[0].width * s->jpeg_height);
        return TRUE;
}

void * jpeg_compress_init(char * opts)
{
        struct compress_jpeg_state *s;
        
        s = (struct compress_jpeg_state *) malloc(sizeof(struct compress_jpeg_state));

        s->out = NULL;
        s->decoded = NULL;
                
        if(opts && strcmp(opts, "help") == 0) {
                printf("JPEG comperssion usage:\n");
                printf("\t-c JPEG[:<quality>][:<cuda_device>]]\n");
                printf("\nCUDA devices:\n");
                gpujpeg_print_devices_info();
                return NULL;
        }

        if(opts) {
                char *tok, *save_ptr = NULL;
                tok = strtok_r(opts, ":", &save_ptr);
                s->encoder_param.quality = atoi(tok);
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        int ret;
                        ret = gpujpeg_init_device(atoi(tok), TRUE);

                        if(ret != 0) {
                                fprintf(stderr, "[JPEG] initializing CUDA device %d failed.\n", atoi(tok));
                                return NULL;
                        }
                } else {
                        printf("Initializing CUDA device 0...\n");
                        int ret = gpujpeg_init_device(0, TRUE);
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
                gpujpeg_set_default_parameters(&s->encoder_param);
                printf("[JPEG] setting default encode parameters (quality: %d)\n", 
                                s->encoder_param.quality
                );
        }
                
        s->encoder = NULL; /* not yet configured */

        return s;
}

struct video_frame * jpeg_compress(void *arg, struct video_frame * tx)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) arg;
        int i;
        unsigned char *line1, *line2;

        unsigned int x;
        
        if(!s->encoder) {
                int ret;
                ret = configure_with(s, tx);
                if(!ret)
                        return NULL;
        }

        for (x = 0; x < tx->tile_count;  ++x) {
                struct tile *in_tile = vf_get_tile(tx, x);
                struct tile *out_tile = vf_get_tile(s->out, x);
                
                line1 = (unsigned char *) in_tile->data;
                line2 = (unsigned char *) s->decoded;
                
                for (i = 0; i < (int) in_tile->height; ++i) {
                        s->decoder(line2, line1, out_tile->linesize,
                                        0, 8, 16);
                        line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                        line2 += out_tile->linesize;
                }
                
                line1 = (unsigned char *) out_tile->data + (in_tile->height - 1) * out_tile->linesize;
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
                ret = gpujpeg_encoder_encode(s->encoder, (unsigned char *) s->decoded, &compressed, &size);
                
                if(ret != 0)
                        return NULL;
                
                out_tile->data_len = size;
                memcpy(out_tile->data, compressed, size);
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
                gpujpeg_encoder_destroy(s->encoder);
        
        free(s);
}
