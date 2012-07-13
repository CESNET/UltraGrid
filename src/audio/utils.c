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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


#include "audio/utils.h" 
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#ifdef WORDS_BIGENDIAN
#error "This code will not run with a big-endian machine. Please report a bug to " PACKAGE_BUGREPORT " if you reach here."
#endif // WORDS_BIGENDIAN

static inline int32_t format_from_in_bps(const char *in, int bps);
static inline void format_to_out_bps(char *out, int bps, int32_t out_value);

static inline int32_t format_from_in_bps(const char * in, int bps) {
        int32_t in_value = 0;
        memcpy(&in_value, in, bps);

        if(in_value >> (bps * 8 - 1) && bps != 4) { //negative
                in_value |= ((1<<(32 - bps * 8)) - 1) << (bps * 8);
        }

        return in_value;
}

static inline void format_to_out_bps(char *out, int bps, int32_t out_value) {
        uint32_t mask;
        if(bps == sizeof(uint32_t)) {
                mask = 0xffffffffu - 1;
        } else {
                mask = ((1 << (bps * 8)) - 1);
        }

        if(out_value > (1 << (bps * 8 - 1)) -1) {
                out_value = (1 << (bps * 8 - 1)) -1;
        }

        if(out_value < -(1 << (bps * 8 - 1))) {
                out_value = -(1 << (bps * 8 - 1));
        }

        uint32_t out_value_formatted = (1 * (0x1 & (out_value >> 31))) << (bps * 8 - 1) | (out_value & mask);

        memcpy(out, &out_value_formatted, bps);
}

void change_bps(char *out, int out_bps, const char *in, int in_bps, int in_len /* bytes */)
{
        int i;

        assert ((unsigned int) out_bps <= sizeof(int32_t));

        for(i = 0; i < in_len / in_bps; i++) {
                int32_t in_value = format_from_in_bps(in, in_bps);

                int32_t out_value;

                if(in_bps > out_bps) {
                        out_value = in_value >> (in_bps * 8 - out_bps * 8);
                } else {
                        out_value = in_value << (out_bps * 8 - in_bps * 8);
                }

                format_to_out_bps(out, out_bps, out_value);

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

        assert (bps <= 4);

        in += pos_in_stream * bps;

        for (i = 0; i < samples; ++i) {
                memcpy(out, in, bps);

                out += bps;
                in += in_stream_channels * bps;

        }
}

void mux_channel(char *out, char *in, int bps, int in_len, int out_stream_channels, int pos_in_stream, double scale)
{
        int samples = in_len / bps;
        int i;
        
        assert (bps <= 4);

        out += pos_in_stream * bps;

        if(scale == 1.0) {
                for (i = 0; i < samples; ++i) {
                        memcpy(out, in, bps);

                        in += bps;
                        out += out_stream_channels * bps;

                }
        } else {
                for (i = 0; i < samples; ++i) {
                        int32_t in_value = format_from_in_bps(in, bps);

                        in_value *= scale;

                        format_to_out_bps(out, bps, in_value);

                        in += bps;
                        out += out_stream_channels * bps;
                }
        }
}

void mux_and_mix_channel(char *out, char *in, int bps, int in_len, int out_stream_channels, int pos_in_stream, double scale)
{
        int i;

        assert (bps <= 4);

        out += pos_in_stream * bps;

        for(i = 0; i < in_len / bps; i++) {
                int32_t in_value = format_from_in_bps(in, bps);
                int32_t out_value = format_from_in_bps(out, bps);

                int32_t new_value = (double)in_value * scale + out_value;

                format_to_out_bps(out, bps, new_value);

                in += bps;
                out += out_stream_channels * bps;
        }
}

double get_avg_volume(char *data, int bps, int in_len, int stream_channels, int pos_in_stream)
{
        float average_vol = 0;
        int i;

        assert ((unsigned int) bps <= sizeof(int32_t));

        data += pos_in_stream * bps;

        for(i = 0; i < in_len / bps; i++) {
                int32_t in_value = format_from_in_bps(data, bps);

                //if(pos_in_stream) fprintf(stderr, "%d-%d ", pos_in_stream, data);

                average_vol = average_vol * (i / ((double) i + 1)) + 
                        fabs(((double) in_value / ((1 << (bps * 8 - 1)) - 1)) / (i + 1));

                data += bps * stream_channels;
        }

        return average_vol;
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

void signed2unsigned(char *out, char *in, int in_len)
{
        int8_t *inch = (int8_t *) in;
        uint8_t *outch = (uint8_t *) out;
        int items = in_len / sizeof(int8_t);

        while(items-- > 0) {
                int8_t in_value = *inch++;
                uint8_t out_value = (int) 128 + in_value;
                *outch++ = out_value;
        }
}

