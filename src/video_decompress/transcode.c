/*
 * FILE:    video_decompress/transcode.c
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

#if ! defined BUILD_LIBRARIES && defined HAVE_JPEG || defined HAVE_RTDXT

#include "video_decompress/transcode.h"

#include <pthread.h>
#include <stdlib.h>

#include "libgpujpeg/gpujpeg_decoder.h"
#include "cuda_dxt/cuda_dxt.h"

#include "debug.h"
#include "host.h"
#include "video.h"
#include "video_codec.h"
#include "video_decompress.h"
#include "video_decompress/jpeg.h"

struct state_decompress_transcode {
        struct gpujpeg_decoder *jpeg_decoder;

        char *dxt_out_buff;

        struct video_desc desc;
};

void * transcode_decompress_init(void)
{
        struct state_decompress_transcode *s;

        s = (struct state_decompress_transcode *) malloc(sizeof(struct state_decompress_transcode));
        s->jpeg_decoder = NULL;

        return s;
}

/**
 * Reconfigureation function
 *
 * @return 0 to indicate error
 *         otherwise maximal buffer size which ins needed for image of given codec, width, and height
 */
int transcode_decompress_reconfigure(void *state, struct video_desc desc, 
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);
        struct state_decompress_transcode *s = (struct state_decompress_transcode *) state;

        s->desc = desc;

        assert(out_codec == DXT1);
        assert(pitch == (int) desc.width / 2); // default for DXT1
        
        if(s->jpeg_decoder != NULL) {
                fprintf(stderr, "Reconfiguration is not currently supported in module [%s:%d]\n",
                                __FILE__, __LINE__);
                return 0;
        }
        
        gpujpeg_init_device(cuda_device, 0);

        cudaMallocHost((void **) &s->dxt_out_buff, desc.width * desc.height / 2);

        s->jpeg_decoder = gpujpeg_decoder_create();
        if(!s->jpeg_decoder) {
                fprintf(stderr, "Creating JPEG decoder failed.\n");
                return 0;
        }
        
        return desc.width * desc.height;
}

void transcode_decompress(void *state, unsigned char *dst, unsigned char *buffer, unsigned int src_len)
{
        struct state_decompress_transcode *s = (struct state_decompress_transcode *) state;
        struct gpujpeg_decoder_output decoder_output;

        gpujpeg_decoder_output_set_cuda_buffer(&decoder_output);

        gpujpeg_decoder_decode(s->jpeg_decoder, buffer, src_len, &decoder_output);
        cuda_rgb_to_dxt1(decoder_output.data, s->dxt_out_buff, s->desc.width, s->desc.height, 0);
        if(cudaSuccess != cudaMemcpy((char*) dst, s->dxt_out_buff, s->desc.width * s->desc.height / 2, cudaMemcpyDeviceToHost)) {
                fprintf(stderr, "[transcode] unable to copy from device.");
        }
}

int transcode_decompress_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return FALSE;
}

void transcode_decompress_done(void *state)
{
        struct state_decompress_transcode *s = (struct state_decompress_transcode *) state;

        gpujpeg_decoder_destroy(s->jpeg_decoder);
}

#endif // ! defined BUILD_LIBRARIES && defined HAVE_CUDA && defined HAVE_GL

