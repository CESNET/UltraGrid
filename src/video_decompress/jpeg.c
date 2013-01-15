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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H
#include "debug.h"
#include "host.h"
#include "video_codec.h"
#include "video_decompress.h"

#include "libgpujpeg/gpujpeg_decoder.h"
//#include "compat/platform_semaphore.h"
#include <cuda_runtime.h>
#include <pthread.h>
#include <stdlib.h>
#include "video_decompress/jpeg.h"

struct state_decompress_jpeg {
        struct gpujpeg_decoder *decoder;

        struct video_desc desc;
        int compressed_len;
        int rshift, gshift, bshift;
        int pitch;
        codec_t out_codec;
};

static int configure_with(struct state_decompress_jpeg *s, struct video_desc desc);

static int configure_with(struct state_decompress_jpeg *s, struct video_desc desc)
{
        s->desc = desc;

        s->decoder = gpujpeg_decoder_create();
        if(!s->decoder) {
                return FALSE;
        }
        if(s->out_codec == RGB) {
                s->decoder->coder.param_image.color_space = GPUJPEG_RGB;
                s->decoder->coder.param_image.sampling_factor = GPUJPEG_4_4_4;
                s->compressed_len = desc.width * desc.height * 2;
        } else {
                s->decoder->coder.param_image.color_space = GPUJPEG_YCBCR_BT709;
                s->decoder->coder.param_image.sampling_factor = GPUJPEG_4_2_2;
                s->compressed_len = desc.width * desc.height * 3;
        }

        return TRUE;
}

void * jpeg_decompress_init(void)
{
        struct state_decompress_jpeg *s;

        s = (struct state_decompress_jpeg *) malloc(sizeof(struct state_decompress_jpeg));

        s->decoder = NULL;
        s->pitch = 0;

        int ret;
        printf("Initializing CUDA device %d...\n", cuda_devices[0]);
        ret = gpujpeg_init_device(cuda_devices[0], TRUE);
        if(ret != 0) {
                fprintf(stderr, "[JPEG] initializing CUDA device %d failed.\n", cuda_devices[0]);
                free(s);
                return NULL;
        }


        return s;
}

int jpeg_decompress_reconfigure(void *state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;
        int ret;
        
        assert(out_codec == RGB || out_codec == UYVY);

        if(s->out_codec == out_codec &&
                        s->pitch == pitch &&
                        s->rshift == rshift &&
                        s->gshift == gshift &&
                        s->bshift == bshift &&
                        video_desc_eq_excl_param(s->desc, desc, PARAM_INTERLACING)) {
                ret = TRUE;
        } else {
                s->out_codec = out_codec;
                s->pitch = pitch;
                s->rshift = rshift;
                s->gshift = gshift;
                s->bshift = bshift;
                if(!s->decoder) {
                        ret = configure_with(s, desc);
                } else {
                        gpujpeg_decoder_destroy(s->decoder);
                        ret = configure_with(s, desc);
                }
        }

        if(ret) {
                return s->compressed_len;
        } else {
                return 0;
        }
}

int jpeg_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int frame_seq)
{
        UNUSED(frame_seq);
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;
        int ret;
        struct gpujpeg_decoder_output decoder_output;
        int linesize;

        if(s->out_codec == RGB) {
                linesize = s->desc.width * 3;
        } else {
                linesize = s->desc.width * 2;
        }
        
        cudaSetDevice(cuda_devices[0]);

        if((s->out_codec != RGB || (s->rshift == 0 && s->gshift == 8 && s->bshift == 16)) &&
                        s->pitch == linesize) {
                gpujpeg_decoder_output_set_default(&decoder_output);
                decoder_output.type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;
                decoder_output.data = dst;
                //int data_decompressed_size = decoder_output.data_size;
                    
                ret = gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output);
                if (ret != 0) return FALSE;
        } else {
                unsigned int i;
                unsigned char *line_src, *line_dst;
                
                gpujpeg_decoder_output_set_default(&decoder_output);
                decoder_output.type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
                //int data_decompressed_size = decoder_output.data_size;
                    
                ret = gpujpeg_decoder_decode(s->decoder, (uint8_t*) buffer, src_len, &decoder_output);

                if (ret != 0) return FALSE;
                
                line_dst = dst;
                line_src = decoder_output.data;
                for(i = 0u; i < s->desc.height; i++) {
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

        return TRUE;
}

int jpeg_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_decompress *s = (struct state_decompress *) state;
        UNUSED(s);
        int ret = FALSE;

        switch(property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if(*len >= sizeof(int)) {
                                *(int *) val = FALSE;
                                *len = sizeof(int);
                                ret = TRUE;
                        }
                        break;
                default:
                        ret = FALSE;
        }

        return ret;
}

void jpeg_decompress_done(void *state)
{
        struct state_decompress_jpeg *s = (struct state_decompress_jpeg *) state;

        if(s->decoder) {
                gpujpeg_decoder_destroy(s->decoder);
        }
        free(s);
}
