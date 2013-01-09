/*
 * FILE:    video_decompress/dxt_glsl.c
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
#include "video_decompress.h"

#include "dxt_compress/dxt_decoder.h"
#include "dxt_compress/dxt_util.h"
//#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include "video_decompress/dxt_glsl.h"
#ifdef HAVE_MACOSX
#include "mac_gl_common.h"
#else
#include "x11_common.h"
#include "glx_common.h"
#endif

struct state_decompress {
        struct dxt_decoder *decoder;

        struct video_desc desc;
        int compressed_len;
        int rshift, gshift, bshift;
        int pitch;
        codec_t out_codec;
        unsigned int configured:1;
        
        void *gl_context;
        unsigned int legacy:1;
};

static void configure_with(struct state_decompress *decompressor, struct video_desc desc)
{
        enum dxt_type type;

#ifndef HAVE_MACOSX
        printf("[RTDXT] Trying OpenGL 3.1 context first.\n");
        decompressor->gl_context = glx_init(MK_OPENGL_VERSION(3,1));
        decompressor->legacy = FALSE;
        if(!decompressor->gl_context) {
                fprintf(stderr, "[RTDXT] OpenGL 3.1 profile failed to initialize, falling back to legacy profile.\n");
                decompressor->gl_context = glx_init(OPENGL_VERSION_UNSPECIFIED);
                decompressor->legacy = TRUE;
        }
        glx_validate(decompressor->gl_context);
#else
        decompressor->gl_context = NULL;
        if(get_mac_kernel_version_major() >= 11) {
                printf("[RTDXT] Mac 10.7 or latter detected. Trying OpenGL 3.2 Core profile first.\n");
                decompressor->gl_context = mac_gl_init(MAC_GL_PROFILE_3_2);
                if(!decompressor->gl_context) {
                        fprintf(stderr, "[RTDXT] OpenGL 3.2 Core profile failed to initialize, falling back to legacy profile.\n");
                } else {
                        decompressor->legacy = FALSE;
                }
        }

        if(!decompressor->gl_context) {
                decompressor->gl_context = mac_gl_init(MAC_GL_PROFILE_LEGACY);
                decompressor->legacy = TRUE;
        }
#endif
        if(!decompressor->gl_context) {
                fprintf(stderr, "[RTDXT decompress] Failed to create GL context.");
                exit_uv(128);
                decompressor->compressed_len = 0;
                return;
        }

        if(desc.color_spec == DXT5) {
                type = DXT_TYPE_DXT5_YCOCG;
        } else if(desc.color_spec == DXT1) {
                type = DXT_TYPE_DXT1;
        } else if(desc.color_spec == DXT1_YUV) {
                type = DXT_TYPE_DXT1_YUV;
        } else {
                fprintf(stderr, "Wrong compressiong to decompress.\n");
                return;
        }
        
        decompressor->desc = desc;

        decompressor->decoder = dxt_decoder_create(type, desc.width,
                        desc.height, decompressor->out_codec == RGBA ? DXT_FORMAT_RGBA : DXT_FORMAT_YUV422, decompressor->legacy);

        assert(decompressor->decoder != NULL);
        
        decompressor->compressed_len = dxt_get_size(desc.width, desc.height, type);
        decompressor->configured = TRUE;
}

void * dxt_glsl_decompress_init(void)
{
        struct state_decompress *s;
        
        s = (struct state_decompress *) malloc(sizeof(struct state_decompress));
        s->configured = FALSE;
#ifndef HAVE_MACOSX
        x11_enter_thread();
#endif
        return s;
}

int dxt_glsl_decompress_reconfigure(void *state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress *s = (struct state_decompress *) state;
        
        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        s->out_codec = out_codec;
        if(!s->configured) {
                configure_with(s, desc);
        } else {
                dxt_decoder_destroy(s->decoder);
                configure_with(s, desc);
        }
        return s->compressed_len;
}

int dxt_glsl_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq)
{
        struct state_decompress *s = (struct state_decompress *) state;
        UNUSED(src_len);
        UNUSED(frame_seq);
        
        if(s->pitch == vc_get_linesize(s->desc.width, s->out_codec)) {
                dxt_decoder_decompress(s->decoder, (unsigned char *) buffer,
                                (unsigned char *) dst);
        } else {
                int i;
                int linesize;
                unsigned char *line_src, *line_dst;
                
                unsigned char *tmp;
                
                if(s->out_codec == UYVY)
                        linesize = s->desc.width * 2;
                else
                        linesize = s->desc.width * 4;
                tmp = malloc(linesize * s->desc.height);

                dxt_decoder_decompress(s->decoder, (unsigned char *) buffer,
                                (unsigned char *) tmp);
                line_dst = dst;
                line_src = tmp;
                for(i = 0; i < (int) s->desc.height; i++) {
                        if(s->out_codec == RGBA) {
                                vc_copylineRGBA(line_dst, line_src, linesize,
                                                s->rshift, s->gshift, s->bshift);
                        } else { /* UYVY */
                                memcpy(line_dst, line_src, linesize);
                        }
                        line_dst += s->pitch;
                        line_src += linesize;
                        
                }
                free(tmp);
        }

        return TRUE;
}

int dxt_glsl_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decompress *s = (struct state_decompress *) state;
        UNUSED(s);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = TRUE;
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

void dxt_glsl_decompress_done(void *state)
{
        struct state_decompress *s = (struct state_decompress *) state;
        
        if(s->configured) {
                dxt_decoder_destroy(s->decoder);
#ifdef HAVE_MACOSX
                mac_gl_free(s->gl_context);
#else
                glx_free(s->gl_context);
#endif
        }
        free(s);
}
