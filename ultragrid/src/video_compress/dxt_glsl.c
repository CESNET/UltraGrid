/*
 * FILE:    dxt_glsl_compress.c
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
#include "video_compress/dxt_glsl.h"
#include "dxt_compress/dxt_encoder.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#ifdef HAVE_MACOSX
#include "mac_gl_common.h"
#else
#include <GL/glew.h>
#include "x11_common.h"
#include "glx_common.h"
#endif

struct video_compress {
        struct dxt_encoder *encoder;

        struct video_frame *out;
        decoder_t decoder;
        char *decoded;
        unsigned int configured:1;
        unsigned int interlaced_input:1;
        codec_t color_spec;
        
        int dxt_height;
        void *gl_context;
        int legacy:1;
};

static int configure_with(struct video_compress *s, struct video_frame *frame);

static int configure_with(struct video_compress *s, struct video_frame *frame)
{
        unsigned int x;
        enum dxt_format format;
        
        assert(vf_get_tile(frame, 0)->width % 4 == 0);
        s->out = vf_alloc(frame->tile_count);
        
        for (x = 0; x < frame->tile_count; ++x) {
                if (vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width ||
                                vf_get_tile(frame, x)->width != vf_get_tile(frame, 0)->width) {

                        fprintf(stderr,"[RTDXT] Requested to compress tiles of different size!");
                        exit_uv(128);
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
                        format = DXT_FORMAT_RGB;
                        break;
                case RGBA:
                        s->decoder = (decoder_t) memcpy;
                        format = DXT_FORMAT_RGBA;
                        break;
                case R10k:
                        s->decoder = (decoder_t) vc_copyliner10k;
                        format = DXT_FORMAT_RGBA;
                        break;
                case UYVY:
                case Vuy2:
                case DVS8:
                        s->decoder = (decoder_t) memcpy;
                        format = DXT_FORMAT_YUV422;
                        break;
                case v210:
                        s->decoder = (decoder_t) vc_copylinev210;
                        format = DXT_FORMAT_YUV422;
                        break;
                case DVS10:
                        s->decoder = (decoder_t) vc_copylineDVS10;
                        format = DXT_FORMAT_YUV422;
                        break;
                case DPX10:        
                        s->decoder = (decoder_t) vc_copylineDPX10toRGBA;
                        format = DXT_FORMAT_RGBA;
                        break;
                default:
                        fprintf(stderr, "[RTDXT] Unknown codec: %d\n", frame->color_spec);
                        exit_uv(128);
                        return FALSE;
        }

        /* We will deinterlace the output frame */
        if(frame->interlacing  == INTERLACED_MERGED)
                s->interlaced_input = TRUE;
        else
                s->interlaced_input = FALSE;

        s->dxt_height = (s->out->tiles[0].height + 3) / 4 * 4;

        if(s->out->color_spec == DXT1) {
                s->encoder = dxt_encoder_create(DXT_TYPE_DXT1, s->out->tiles[0].width, s->dxt_height, format, s->legacy);
                s->out->tiles[0].data_len = s->out->tiles[0].width * s->dxt_height / 2;
        } else if(s->out->color_spec == DXT5){
                s->encoder = dxt_encoder_create(DXT_TYPE_DXT5_YCOCG, s->out->tiles[0].width, s->dxt_height, format, s->legacy);
                s->out->tiles[0].data_len = s->out->tiles[0].width * s->dxt_height;
        }
        
        for (x = 0; x < frame->tile_count; ++x) {
                vf_get_tile(s->out, x)->linesize = s->out->tiles[0].width;
                switch(format) { 
                        case DXT_FORMAT_RGBA:
                                vf_get_tile(s->out, x)->linesize *= 4;
                                break;
                        case DXT_FORMAT_RGB:
                                vf_get_tile(s->out, x)->linesize *= 3;
                                break;
                        case DXT_FORMAT_YUV422:
                                vf_get_tile(s->out, x)->linesize *= 2;
                                break;
                        case DXT_FORMAT_YUV:
                                /* not used - just not compilator to complain */
                                abort();
                                break;
                }
                vf_get_tile(s->out, x)->data_len = s->out->tiles[0].data_len;
                vf_get_tile(s->out, x)->data = (char *) malloc(s->out->tiles[0].data_len);
        }
        
        if(!s->encoder) {
                fprintf(stderr, "[RTDXT] Failed to create encoder.\n");
                exit_uv(128);
                return FALSE;
        }
        
        s->decoded = malloc(4 * s->out->tiles[0].width * s->dxt_height);
        
        s->configured = TRUE;
        return TRUE;
}

void * dxt_glsl_compress_init(char * opts)
{
        struct video_compress *s;
        
        s = (struct video_compress *) malloc(sizeof(struct video_compress));
        s->out = NULL;
        s->decoded = NULL;

#ifndef HAVE_MACOSX
        x11_enter_thread();
        printf("Trying OpenGL 3.1 first.\n");
        s->gl_context = glx_init(MK_OPENGL_VERSION(3,1));
        s->legacy = FALSE;
        if(!s->gl_context) {
                fprintf(stderr, "[RTDXT] OpenGL 3.1 profile failed to initialize, falling back to legacy profile.\n");
                s->gl_context = glx_init(OPENGL_VERSION_UNSPECIFIED);
                s->legacy = TRUE;
        }
        glx_validate(s->gl_context);
#else
        s->gl_context = NULL;
        if(get_mac_kernel_version_major() >= 11) {
                printf("[RTDXT] Mac 10.7 or latter detected. Trying OpenGL 3.2 Core profile first.\n");
                s->gl_context = mac_gl_init(MAC_GL_PROFILE_3_2);
                if(!s->gl_context) {
                        fprintf(stderr, "[RTDXT] OpenGL 3.2 Core profile failed to initialize, falling back to legacy profile.\n");
                } else {
                        s->legacy = FALSE;
                }
        }

        if(!s->gl_context) {
                s->gl_context = mac_gl_init(MAC_GL_PROFILE_LEGACY);
                s->legacy = TRUE;
        }
#endif

        if(!s->gl_context) {
                fprintf(stderr, "[RTDXT] Error initializing GLX context");
                exit_uv(128);
                return NULL;
        }
        
        if(opts && strcmp(opts, "help") == 0) {
                printf("DXT GLSL comperssion usage:\n");
                printf("\t-c RTDXT:DXT1\n");
                printf("\t\tcompress with DXT1\n");
                printf("\t-c RTDXT:DXT5\n");
                printf("\t\tcompress with DXT5 YCoCg\n");
                return NULL;
        }
        
        if(opts) {
                if(strcasecmp(opts, "DXT5") == 0) {
                        s->color_spec = DXT5;
                } else if(strcasecmp(opts, "DXT1") == 0) {
                        s->color_spec = DXT1;
                } else {
                        fprintf(stderr, "Unknown compression : %s\n", opts);
                        return NULL;
                }
        } else {
                s->color_spec = DXT1;
        }
                
        s->configured = FALSE;

        return s;
}

struct video_frame * dxt_glsl_compress(void *arg, struct video_frame * tx)
{
        struct video_compress *s = (struct video_compress *) arg;
        int i;
        unsigned char *line1, *line2;
        
        unsigned int x;
        
        if(!s->configured) {
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
                
                /* if height % 4 != 0, copy last line to align */
                if((unsigned int) s->dxt_height != out_tile->height) {
                        int y;
                        line1 = (unsigned char *) s->decoded + out_tile->linesize * (out_tile->height - 1);
                        for (y = out_tile->height; y < s->dxt_height; ++y)
                        {
                                memcpy(line2, line1, out_tile->linesize);
                                line2 += out_tile->linesize;
                        }
                        line2 += out_tile->linesize;
                }
                
                if(s->interlaced_input)
                        vc_deinterlace((unsigned char *) s->decoded, out_tile->linesize,
                                        s->dxt_height);
                
                dxt_encoder_compress(s->encoder,
                                (unsigned char *) s->decoded,
                                (unsigned char *) out_tile->data);
        }
        
        return s->out;
}

void dxt_glsl_compress_done(void *arg)
{
        struct video_compress *s = (struct video_compress *) arg;
        
        dxt_encoder_destroy(s->encoder);

        if(s->out)
                free(s->out->tiles[0].data);
        vf_free(s->out);

        free(s->decoded);

#ifdef HAVE_MACOSX
        mac_gl_free(s->gl_context);
#else
        glx_free(s->gl_context);
#endif
        free(s);
}
