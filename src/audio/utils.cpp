/**
 * @file   audio/utils.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2014 CESNET z.s.p.o.
 * All rights reserved.
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
 * 3. Neither the name of CESNET nor the names of its contributors may be
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
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


#include "audio/audio.h"
#include "audio/codec.h"
#include "audio/utils.h" 
#include "debug.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>


#ifdef WORDS_BIGENDIAN
#error "This code will not run with a big-endian machine. Please report a bug to " PACKAGE_BUGREPORT " if you reach here."
#endif // WORDS_BIGENDIAN

using namespace std;

static double get_normalized(const int8_t *in, int bps) {
        int64_t sample = 0;
        bool negative = false;

        for (int j = 0; j < bps; ++j) {
                sample = (sample | ((((const uint8_t *)in)[j]) << (uint64_t)(8ull * j)));
        }
        if ((int8_t)(in[bps - 1] < 0))
                negative = true;
        if (negative) {
                for (int i = bps; i < 8; ++i) {
                        sample = (sample |  (255ull << (8ull * i)));
                }
        }
        return (double) sample / ((1 << (bps * 8 - 1)));
}

/**
 * @brief Calculates mean and peak RMS from audio samples
 *
 * @param[in]  frame   audio frame
 * @param[in]  channel channel index to calculate RMS to
 * @param[out] peak    peak RMS
 * @returns            mean RMS
 */
double calculate_rms(audio_frame2 *frame, int channel, double *peak)
{
        assert(frame->get_codec() == AC_PCM);
        double sum = 0;
        *peak = 0;
        int sample_count = frame->get_data_len(channel) / frame->get_bps();
        const char *channel_data = frame->get_data(channel);
        for (size_t i = 0; i < frame->get_data_len(channel); i += frame->get_bps()) {
                double val = get_normalized((const int8_t *) channel_data + i, frame->get_bps());
                sum += val;
                if (fabs(val) > *peak) {
                        *peak = fabs(val);
                }
        }

        double average = sum / sample_count;

        double sumMeanSquare = 0.0;

        for (size_t i = 0; i < frame->get_data_len(channel); i += frame->get_bps()) {
                sumMeanSquare += pow(get_normalized((const int8_t *) channel_data + i, frame->get_bps())
                                - average, 2.0);
        }

        double averageMeanSquare = sumMeanSquare / sample_count;
        double rootMeanSquare = sqrt(averageMeanSquare);

        return rootMeanSquare;
}

bool audio_desc_eq(struct audio_desc a1, struct audio_desc a2) {
        return a1.bps == a2.bps &&
                a1.sample_rate == a2.sample_rate &&
                a1.ch_count == a2.ch_count &&
                a1.codec == a2.codec;
}

struct audio_desc audio_desc_from_audio_frame(struct audio_frame *frame) {
        return audio_desc { frame->bps,
                frame->sample_rate,
                frame->ch_count,
                AC_PCM
        };
}

struct audio_desc audio_desc_from_audio_channel(audio_channel *channel) {
        return audio_desc { channel->bps,
                channel->sample_rate,
                1,
                channel->codec
        };
}

/**
 * Copies desc from desc to f.
 *
 * @note
 * Doesn't clear/set other members of f, thus caller needs to do that first if needed.
 */
void audio_frame_write_desc(struct audio_frame *f, struct audio_desc desc)
{
        f->bps = desc.bps;
        f->sample_rate = desc.sample_rate;
        f->ch_count = desc.ch_count;
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
        assert((unsigned int) frame->max_size >= (unsigned int) frame->data_len * new_channel_count / frame->ch_count);

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

void remux_channel(char *out, const char *in, int bps, int in_len, int in_stream_channels, int out_stream_channels, int pos_in_stream, int pos_out_stream)
{
        int samples = in_len / (in_stream_channels * bps);
        int i;

        assert (bps <= 4);

        in += pos_in_stream * bps;
        out += pos_out_stream * bps;

        for (i = 0; i < samples; ++i) {
                memcpy(out, in, bps);

                out += bps * out_stream_channels;
                in += bps * in_stream_channels;

        }
}

void mux_channel(char *out, const char *in, int bps, int in_len, int out_stream_channels, int pos_in_stream, double scale)
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

void mux_and_mix_channel(char *out, const char *in, int bps, int in_len, int out_stream_channels, int pos_in_stream, double scale)
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

void float2int(char *out, const char *in, int len)
{
        const float *inf = (const float *)(const void *) in;
        int32_t *outi = (int32_t *)(void *) out;
        int items = len / sizeof(int32_t);

        while(items-- > 0) {
                float sample = *inf++;
                if(sample > 1.0) sample = 1.0;
                if(sample < -1.0) sample = -1.0;
                *outi++ = sample * INT_MAX;
        }
}

void int2float(char *out, const char *in, int len)
{
        const int32_t *ini = (const int32_t *)(const void *) in;
        float *outf = (float *)(void *) out;
        int items = len / sizeof(int32_t);

        while(items-- > 0) {
                *outf++ = (float) *ini++ / INT_MAX;
        }
}

void short_int2float(char *out, char *in, int in_len)
{
        int16_t *ini = (int16_t *)(void *) in;
        float *outf = (float *)(void *) out;
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

void audio_channel_demux(const audio_frame2 *frame, int index, audio_channel *channel)
{
        channel->data = frame->get_data(index);
        channel->data_len = frame->get_data_len(index);
        channel->codec = frame->get_codec();
        channel->bps = frame->get_bps();
        channel->sample_rate = frame->get_sample_rate();
}

int32_t format_from_in_bps(const char * in, int bps) {
        int32_t in_value = 0;
        memcpy(&in_value, in, bps);

        if(in_value >> (bps * 8 - 1) && bps != 4) { //negative
                in_value |= ((1<<(32 - bps * 8)) - 1) << (bps * 8);
        }

        return in_value;
}

void format_to_out_bps(char *out, int bps, int32_t out_value) {
        uint32_t mask = ((1ll << (bps * 8)) - 1);

        // clamp
        if(out_value > (1ll << (bps * 8 - 1)) -1) {
                out_value = (1ll << (bps * 8 - 1)) -1;
        }

        if(out_value < -(1ll << (bps * 8 - 1))) {
                out_value = -(1ll << (bps * 8 - 1));
        }

        uint32_t out_value_formatted = (1 * (0x1 & (out_value >> 31))) << (bps * 8 - 1) | (out_value & mask);

        memcpy(out, &out_value_formatted, bps);
}

void interleaved2noninterleaved(char *out, const char *in, int bps, int in_len, int channel_count)
{
        vector<char *> out_ch(channel_count);
        for (int i = 0; i < channel_count; ++i) {
                out_ch[i] = out + in_len / channel_count * i;
        }

        int offset = 0;
        int index = 0;
        while (offset < in_len) {
                memcpy(out_ch[index], in, bps);
                out_ch[index] += bps;
                index = (index + 1) % channel_count;
                in += bps;
                offset += bps;
        }
}

bool append_audio_frame(struct audio_frame *frame, char *data, size_t data_len) {
        bool ret = true;
        if (frame->data_len + data_len > (size_t) frame->max_size) {
                log_msg(LOG_LEVEL_WARNING, "Audio frame overrun, discarding some data.\n");
                data_len = frame->max_size - frame->data_len;
                ret = false;
        }
        memcpy(frame->data + frame->data_len, data, data_len);
        frame->data_len += data_len;

        return ret;
}

struct audio_frame *audio_frame_copy(const struct audio_frame *src, bool keep_size) {
        struct audio_frame *ret = (struct audio_frame *) malloc(sizeof(struct audio_frame));
        memcpy(ret, src, sizeof *ret);
        if (!keep_size) {
                ret->max_size = src->data_len;
        }
        ret->data = (char *) malloc(ret->max_size);
        memcpy(ret->data, src->data, src->data_len);
        return ret;
}

