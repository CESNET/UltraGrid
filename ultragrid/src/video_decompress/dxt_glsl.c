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

#include "x11_common.h"
#include "dxt_compress/dxt_decoder.h"
//#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include "video_decompress/dxt_glsl.h"
#include <GL/glew.h>
#include "x11_common.h"

struct state_decompress {
        struct dxt_decoder *decoder;

        struct video_desc desc;
        int compressed_len;
        int rshift, gshift, bshift;
        int pitch;
        unsigned int configured:1;
        
        void *glx_context;
};

static void configure_with(struct state_decompress *decompressor, struct video_desc desc)
{
        enum dxt_type type;
        
        decompressor->glx_context = glx_init();
        if(!decompressor->glx_context) {
                error_with_code_msg(128, "Failed to create GLX context.");
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
        decompressor->decoder = dxt_decoder_create(type, desc.width, desc.height);
        
        decompressor->compressed_len = desc.width * desc.height /
                (desc.color_spec == DXT5 ? 1 : 2);
        decompressor->configured = TRUE;
}

void * dxt_glsl_decompress_init(void)
{
        struct state_decompress *s;
        
        s = (struct state_decompress *) malloc(sizeof(struct state_decompress));
        s->configured = FALSE;
        x11_enter_thread();

        return s;
}

int dxt_glsl_decompress_reconfigure(void *state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch)
{
        struct state_decompress *s = (struct state_decompress *) state;
        
        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        if(!s->configured) {
                configure_with(s, desc);
        } else {
                dxt_decoder_destroy(s->decoder);
                configure_with(s, desc);
        }
        return s->compressed_len;
}

void dxt_glsl_decompress(void *state, unsigned char *dst, unsigned char *buffer, unsigned int src_len)
{
        struct state_decompress *s = (struct state_decompress *) state;
        UNUSED(src_len);
        
        if(s->pitch == 0) {
                dxt_decoder_decompress(s->decoder, (unsigned char *) buffer,
                                (unsigned char *) dst);
        } else {
                int i;
                int linesize = s->desc.width * 4;
                char *tmp = malloc(linesize * s->desc.height);
                dxt_decoder_decompress(s->decoder, (unsigned char *) buffer,
                                (unsigned char *) tmp);
                for(i = 0; i < s->desc.height; ++i)
                        vc_copylineRGBA(dst + i * s->pitch, tmp + i * linesize, linesize,
                                        s->rshift, s->gshift, s->bshift);
                free(tmp);
        }
}

void dxt_glsl_decompress_done(void *state)
{
        struct state_decompress *s = (struct state_decompress *) state;
        
        if(s->configured) {
                dxt_decoder_destroy(s->decoder);
                glx_free(s->glx_context);
        }
        free(s);
}
