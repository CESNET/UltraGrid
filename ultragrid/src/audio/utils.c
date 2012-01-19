/*
 * FILE:    audio/utils.c
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
 * 4. Neither the name of CESNET nor the names of its contributors may be used 
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
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
 *
 */

#include "audio/audio.h"
#include "audio/utils.h" 
#include <assert.h>
#include <string.h>
#include <limits.h>

void change_bps(char *out, int out_bps, const char *in, int in_bps, int in_len /* bytes */)
{
        int pattern = 0xffff << (out_bps * 8);
        int i;

        assert (out_bps <= 4);
        if(out_bps == 4) /* take care of 32b shifts ! */
                pattern = 0;
        for(i = 0; i < in_len / in_bps; i++) {
                *((unsigned int *) out) &= pattern;
                if(in_bps > out_bps)
                        *((int *) out) |= *((const int *) in) >> (in_bps * 8 - out_bps * 8);
                else
                        *((int *) out) |= *((const int *) in) << (out_bps * 8 - in_bps * 8);
                in += in_bps;
                out += out_bps;
        }
}

void copy_channel(char *out, const char *in, int bps, int in_len /* bytes */, int out_channel_count)
{
        int samples = in_len / bps;
        int i;
        
        assert(out_channel_count > 0);
        assert(bps > 0);
        assert(in_len >= 0);
        
        in += in_len;
        out += in_len * out_channel_count;
        for (i = samples; i > 0 ; --i) {
                int j;
                
                in -= bps;
                for  (j = out_channel_count + 0; j > 0; --j) {
                        out -= bps;
                        memmove(out, in, bps);
                }
        }
}

void audio_frame_multiply_channel(struct audio_frame *frame, int new_channel_count) {
        assert(frame->max_size >= (unsigned int) frame->data_len * new_channel_count / frame->ch_count);

        copy_channel(frame->data, frame->data, frame->bps, frame->data_len, new_channel_count);
}

void demux_channel(char *out, char *in, int bps, int in_len, int in_stream_channels, int pos_in_stream)
{
        int samples = in_len / (in_stream_channels * bps);
        int i;
        
        in += pos_in_stream * bps;

        for (i = 0; i < samples; ++i) {
                memcpy(out, in, bps);

                out += bps;
                in += in_stream_channels * bps;

        }
}

void mux_channel(char *out, char *in, int bps, int in_len, int out_stream_channels, int pos_in_stream)
{
        int samples = in_len / bps;
        int i;
        
        out += pos_in_stream * bps;

        for (i = 0; i < samples; ++i) {
                memcpy(out, in, bps);

                in += bps;
                out += out_stream_channels * bps;

        }
}

void float2int(char *out, char *in, int len)
{
        float *inf = (float *) in;
        int32_t *outi = (int32_t *) out;
        int items = len / sizeof(int32_t);

        while(items-- > 0) {
                *outi++ = *inf++ * INT_MAX;
        }
}

void int2float(char *out, char *in, int len)
{
        int32_t *ini = (int32_t *) in;
        float *outf = (float *) out;
        int items = len / sizeof(int32_t);

        while(items-- > 0) {
                *outf++ = (float) *ini++ / INT_MAX;
        }
}

void short_int2float(char *out, char *in, int in_len)
{
        int16_t *ini = (int16_t *) in;
        float *outf = (float *) out;
        int items = in_len / sizeof(int16_t);

        while(items-- > 0) {
                *outf++ = (float) *ini++ / SHRT_MAX;
        }
}
