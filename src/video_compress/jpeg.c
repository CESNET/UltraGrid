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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "video_compress.h"
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
        
        struct video_frame *out[2];
        
        decoder_t decoder;
        char *decoded;
        unsigned int rgb:1;
        codec_t color_spec;

        struct video_desc saved_desc;

        int restart_interval;
};

static int configure_with(struct compress_jpeg_state *s, struct video_frame *frame);
static void cleanup_state(struct compress_jpeg_state *s);

static int configure_with(struct compress_jpeg_state *s, struct video_frame *frame)
{
        unsigned int x;
        int frame_idx;

        s->saved_desc.width = frame->tiles[0].width;
        s->saved_desc.height = frame->tiles[0].height;
        s->saved_desc.color_spec = frame->color_spec;
        s->saved_desc.fps = frame->fps;
        s->saved_desc.interlacing = frame->interlacing;
        s->saved_desc.tile_count = frame->tile_count;
        
        for (frame_idx = 0; frame_idx < 2; frame_idx++) {
                s->out[frame_idx] = vf_alloc(frame->tile_count);
        }
        
        for (x = 0; x < frame->tile_count; ++x) {
                if (vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width ||
                                vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width) {
                        fprintf(stderr,"[JPEG] Requested to compress tiles of different size!");
                        exit_uv(129);
                        return FALSE;
                }
        }
        
        for (frame_idx = 0; frame_idx < 2; frame_idx++) {
                for (x = 0; x < frame->tile_count; ++x) {
                        vf_get_tile(s->out[frame_idx], x)->width = vf_get_tile(frame, 0)->width;
                        vf_get_tile(s->out[frame_idx], x)->height = vf_get_tile(frame, 0)->height;
                }
                s->out[frame_idx]->interlacing = frame->interlacing;
                s->out[frame_idx]->fps = frame->fps;
                s->out[frame_idx]->color_spec = s->color_spec;
                s->out[frame_idx]->color_spec = JPEG;
        }

        switch (frame->color_spec) {
                case RGB:
                        s->decoder = (decoder_t) memcpy;
                        s->rgb = TRUE;
                        break;
                case RGBA:
                        s->decoder = (decoder_t) vc_copylineRGBAtoRGB;
                        s->rgb = TRUE;
                        break;
                case BGR:
                        s->decoder = (decoder_t) vc_copylineBGRtoRGB;
                        s->rgb = TRUE;
                        break;
                /* TODO: enable (we need R10k -> RGB)
                 * case R10k:
                        s->decoder = (decoder_t) vc_copyliner10k;
                        s->rgb = TRUE;
                        break;*/
                case YUYV:
                        s->decoder = (decoder_t) vc_copylineYUYV;
                        s->rgb = FALSE;
                        break;
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

	s->encoder_param.verbose = 0;
	s->encoder_param.segment_info = 1;

        if(s->rgb) {
                s->encoder_param.interleaved = 0;
                s->encoder_param.restart_interval = s->restart_interval == -1 ? 8
                        : s->restart_interval;
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
                s->encoder_param.restart_interval = s->restart_interval == -1 ? 2
                        : s->restart_interval;
                /* LUMA */
                s->encoder_param.sampling_factor[0].horizontal = 2;
                s->encoder_param.sampling_factor[0].vertical = 1;
                /* Cb and Cr */
                s->encoder_param.sampling_factor[1].horizontal = 1;
                s->encoder_param.sampling_factor[1].vertical = 1;
                s->encoder_param.sampling_factor[2].horizontal = 1;
                s->encoder_param.sampling_factor[2].vertical = 1;
        }

        
        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);

        param_image.width = s->out[0]->tiles[0].width;
        param_image.height = s->out[0]->tiles[0].height;
        
        param_image.comp_count = 3;
        if(s->rgb) {
                param_image.color_space = GPUJPEG_RGB;
                param_image.sampling_factor = GPUJPEG_4_4_4;
        } else {
                param_image.color_space = GPUJPEG_YCBCR_BT709;
                param_image.sampling_factor = GPUJPEG_4_2_2;
        }
        
        s->encoder = gpujpeg_encoder_create(&s->encoder_param, &param_image);
        
        for (frame_idx = 0; frame_idx < 2; frame_idx++) {
                for (x = 0; x < frame->tile_count; ++x) {
                                vf_get_tile(s->out[frame_idx], x)->data = (char *) malloc(s->out[frame_idx]->tiles[0].width * s->out[frame_idx]->tiles[0].height * 3);
                                vf_get_tile(s->out[frame_idx], x)->linesize = s->out[frame_idx]->tiles[0].width * (param_image.color_space == GPUJPEG_RGB ? 3 : 2);

                }
        }
        
        if(!s->encoder) {
                fprintf(stderr, "[DXT GLSL] Failed to create encoder.\n");
                exit_uv(128);
                return FALSE;
        }
        
        s->decoded = malloc(4 * s->out[0]->tiles[0].width * s->out[0]->tiles[0].height);
        return TRUE;
}

void * jpeg_compress_init(char * opts)
{
        struct compress_jpeg_state *s;
        int frame_idx;
        
        s = (struct compress_jpeg_state *) malloc(sizeof(struct compress_jpeg_state));

        for (frame_idx = 0; frame_idx < 2; frame_idx++) {
                s->out[frame_idx] = NULL;
        }
        s->decoded = NULL;
                
        if(opts && strcmp(opts, "help") == 0) {
                printf("JPEG comperssion usage:\n");
                printf("\t-c JPEG[:<quality>[:<restart_interval>]]\n");
                return &compress_init_noerr;
        } else if(opts && strcmp(opts, "list_devices") == 0) {
                printf("CUDA devices:\n");
                gpujpeg_print_devices_info();
                return &compress_init_noerr;
        }

        s->restart_interval = -1;

        if(opts) {
                char *tok, *save_ptr = NULL;
                gpujpeg_set_default_parameters(&s->encoder_param);
                tok = strtok_r(opts, ":", &save_ptr);
                s->encoder_param.quality = atoi(tok);
                int ret;
                printf("Initializing CUDA device %d...\n", cuda_devices[0]);
                ret = gpujpeg_init_device(cuda_devices[0], TRUE);

                if(ret != 0) {
                        fprintf(stderr, "[JPEG] initializing CUDA device %d failed.\n", cuda_devices[0]);
                        return NULL;
                }
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        s->restart_interval = atoi(tok);
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

                int ret;
                printf("Initializing CUDA device %d...\n", cuda_devices[0]);
                ret = gpujpeg_init_device(cuda_devices[0], TRUE);

                if(ret != 0) {
                        fprintf(stderr, "[JPEG] initializing CUDA device %d failed.\n",
                                        cuda_devices[0]);
                        return NULL;
                }
        }
                
        s->encoder = NULL; /* not yet configured */

        return s;
}

struct video_frame * jpeg_compress(void *arg, struct video_frame * tx, int buffer_idx)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) arg;
        int i;
        unsigned char *line1, *line2;
        struct video_frame *out;

        unsigned int x;
        
        if(!s->encoder) {
                int ret;
                ret = configure_with(s, tx);
                if(!ret) {
                        return NULL;
                }
        }

        struct video_desc desc;
        desc = video_desc_from_frame(tx);

        // if format changed, reconfigure
        if(!video_desc_eq_excl_param(s->saved_desc, desc, PARAM_INTERLACING)) {
                cleanup_state(s);
                int ret;
                ret = configure_with(s, tx);
                if(!ret) {
                        return NULL;
                }
        }

        out = s->out[buffer_idx];

        for (x = 0; x < tx->tile_count;  ++x) {
                struct tile *in_tile = vf_get_tile(tx, x);
                struct tile *out_tile = vf_get_tile(out, x);
                
                line1 = (unsigned char *) in_tile->data;
                line2 = (unsigned char *) s->decoded;
                
                for (i = 0; i < (int) in_tile->height; ++i) {
                        s->decoder(line2, line1, out_tile->linesize,
                                        0, 8, 16);
                        line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                        line2 += out_tile->linesize;
                }
                
                line1 = (unsigned char *) out_tile->data + (in_tile->height - 1) * out_tile->linesize;
                for( ; i < (int) out->tiles[0].height; ++i) {
                        memcpy(line2, line1, out_tile->linesize);
                        line2 += out_tile->linesize;
                }
                
                /*if(s->interlaced_input)
                        vc_deinterlace((unsigned char *) s->decoded, out_tile->linesize,
                                        s->out->tiles[0].height);*/
                
                uint8_t *compressed;
                int size;
                int ret;


                struct gpujpeg_encoder_input encoder_input;
                gpujpeg_encoder_input_set_image(&encoder_input, (uint8_t *) s->decoded);
                ret = gpujpeg_encoder_encode(s->encoder, &encoder_input, &compressed, &size);
                
                if(ret != 0)
                        return NULL;
                
                out_tile->data_len = size;
                memcpy(out_tile->data, compressed, size);
        }
        
        return out;
}

void jpeg_compress_done(void *arg)
{
        struct compress_jpeg_state *s = (struct compress_jpeg_state *) arg;

        cleanup_state(s);
        
        free(s);
}

static void cleanup_state(struct compress_jpeg_state *s)
{
        int frame_idx;
        
        for (frame_idx = 0; frame_idx < 2; frame_idx++) {
                if(s->out[frame_idx]) {
                        int x;
                        for (x = 0; x < (int) s->out[frame_idx]->tile_count; ++x) {
                                free(s->out[frame_idx]->tiles[x].data);
                        }
                }
                vf_free(s->out[frame_idx]);
        }
        if(s->encoder)
                gpujpeg_encoder_destroy(s->encoder);
}

