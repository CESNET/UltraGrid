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
#include "jpeg_compress/jpeg_decoder.h"
//#include "compat/platform_semaphore.h"
#include "video_codec.h"
#include <pthread.h>
#include <stdlib.h>
#include "video_decompress/jpeg.h"
#include <GL/glew.h>
#include "x11_common.h"

struct state_decompress_jpeg {
        struct jpeg_decoder *decoder;

        struct video_desc desc;
        int compressed_len;
        int rshift, gshift, bshift;
        int jpeg_height;
        int pitch;
        codec_t out_codec;
};

static void configure_with(struct state_decompress_jpeg *s, struct video_desc desc)
{
        struct jpeg_image_parameters param_image;
        s->desc = desc;
        
        param_image.width = desc.width;
        param_image.height = s->jpeg_height;
        param_image.comp_count = 3;
        if(s->out_codec == RGB) {
                param_image.color_space = JPEG_RGB;
                param_image.sampling_factor = JPEG_4_4_4;
                s->compressed_len = desc.width * desc.height * 2;
        } else {
                param_image.color_space = JPEG_YUV;
                param_image.sampling_factor = JPEG_4_2_2;
                s->compressed_len = desc.width * desc.height * 3;
        }
        
        s->decoder = jpeg_decoder_create(&param_image);
}

void * jpeg_decompress_init(void)
{
        struct state_decompress_jpeg *s;
        
        s = (struct state_decompress_jpeg *) malloc(sizeof(struct state_decompress_jpeg));
        s->decoder = NULL;

        return s;
}

int jpeg_decompress_reconfigure(void *state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;
        
        assert(out_codec == RGB || out_codec == UYVY);
        
        s->out_codec = out_codec;
        s->pitch = pitch;
        s->rshift = rshift;
        s->gshift = gshift;
        s->bshift = bshift;
        s->jpeg_height = (desc.height + 7) / 8 * 8;
        if(!s->decoder) {
                configure_with(s, desc);
        } else {
                jpeg_decoder_destroy(s->decoder);
                configure_with(s, desc);
        }
        return s->compressed_len;
}

void jpeg_decompress(void *state, unsigned char *dst, unsigned char *buffer, unsigned int src_len)
{
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;
        UNUSED(src_len);
        uint8_t *decompressed;
        int size;
        int ret;

        ret = jpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len,
                                &decompressed, &size);
        if (ret != 0) return;
        
        int i;
        int linesize;
        char *line_src, *line_dst;
        
        if(s->out_codec == RGB) {
                linesize = s->desc.width * 3;
        } else {
                linesize = s->desc.width * 2;
        }
        
        line_dst = dst;
        line_src = decompressed;
        for(i = 0; i < s->desc.height; i++) {
                if(s->out_codec == RGB) {
                        vc_copylineRGB(line_dst, line_src, linesize,
                                        s->rshift, s->gshift, s->bshift);
                } else {
                        memcpy(line_dst, line_src, linesize);
                }
                        
                line_dst += s->pitch;
                line_src += linesize;
                
        }
}

void jpeg_decompress_done(void *state)
{
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;

        if(s->decoder) {
                jpeg_decoder_destroy(s->decoder);
        }
        free(s);
}
