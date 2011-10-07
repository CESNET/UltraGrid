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
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
#include "dxt_glsl_compress.h"
#include "dxt_compress/dxt_encoder.h"
#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>

struct video_compress {
        struct dxt_encoder *encoder;

        struct video_frame out;
        unsigned int configured:1;
};

static void configure_with(struct video_compress *s, struct video_frame *frame);

static void configure_with(struct video_compress *s, struct video_frame *frame)
{
        codec_t tmp = s->out.color_spec;
        
        dxt_init();
        
        memcpy(&s->out, frame, sizeof(struct video_frame));
        s->out.color_spec = tmp;
        if(s->out.color_spec == DXT1) {
                s->encoder = dxt_encoder_create(COMPRESS_TYPE_DXT1, frame->width, frame->height);
                s->out.aux |= AUX_RGB;
                s->out.data_len = frame->width * frame->height / 2;
        } else if(s->out.color_spec == DXT5){
                s->encoder = dxt_encoder_create(COMPRESS_TYPE_DXT5_YCOCG, frame->width, frame->height);
                s->out.aux |= AUX_YUV; /* YCoCg */
                s->out.data_len = frame->width * frame->height;
        }
        
        s->out.data = (char *) malloc(s->out.data_len);
                
        s->configured = TRUE;
}

struct video_compress * dxt_glsl_init(char * opts)
{
        struct video_compress *s;
        
        s = (struct video_compress *) malloc(sizeof(struct video_compress));
        s->out.data = NULL;
        
        if(opts) {
                if(strcasecmp(opts, "DXT5_YCoCg") == 0) {
                        s->out.color_spec = DXT5;
                } else if(strcasecmp(opts, "DXT1") == 0) {
                        s->out.color_spec = DXT1;
                } else {
                        fprintf(stderr, "Unknown compression : %s\n", opts);
                        return NULL;
                }
        } else {
                s->out.color_spec = DXT1;
        }
                
        s->configured = FALSE;

        return s;
}

struct video_frame * dxt_glsl_compress(void *arg, struct video_frame * tx)
{
        struct video_compress *s = (struct video_compress *) arg;
        unsigned char *result;
        int size;
        
        if(!s->configured)
                configure_with(s, tx);
        dxt_encoder_compress(s->encoder, tx->data, s->out.data);
        
        return &s->out;
}

void dxt_glsl_exit(void *arg)
{
        struct video_compress *s = (struct video_compress *) arg;
        free(s->out.data);
}
