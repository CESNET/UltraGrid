/**
 * @file   audio/utils.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2021 CESNET z.s.p.o.
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


#include "audio/codec.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <random>


#ifdef WORDS_BIGENDIAN
#error "This code will not run with a big-endian machine. Please report a bug to " PACKAGE_BUGREPORT " if you reach here."
#endif // WORDS_BIGENDIAN

using namespace std;

/**
 * Loads sample with BPS width and returns it cast to
 * int32_t.
 */
template<int BPS> static int32_t load_sample(const char *data);

template<> int32_t load_sample<1>(const char *data) {
        return *reinterpret_cast<const int8_t *>(data);
}

template<> int32_t load_sample<2>(const char *data) {
        return *reinterpret_cast<const int16_t *>(data);
}

template<> int32_t load_sample<3>(const char *data) {
        int32_t in_value = 0;
        memcpy(&in_value, data, 3);

        if ((in_value & 1U<<23U) != 0U) { // negative
                in_value |= 0xFF000000U;
        }

        return in_value;
}

template<> int32_t load_sample<4>(const char *data) {
        return *reinterpret_cast<const int32_t *>(data);
}

template<int BPS> static void store_sample(char *data, int32_t val);

template<> void store_sample<1>(char *data, int32_t val) {
        *reinterpret_cast<int8_t *>(data) = clamp(val, INT8_MIN, INT8_MAX);
}

template<> void store_sample<2>(char *data, int32_t val) {
        *reinterpret_cast<int16_t *>(data) =  clamp(val, INT16_MIN, INT16_MAX);
}

template<> void store_sample<3>(char *data, int32_t val) {
        val = clamp<int32_t>(val, -(1L<<24), (1L<<24) - 1);
        memcpy(data, &val, 3);
}

template<> void store_sample<4>(char *data, int32_t val) {
        *reinterpret_cast<int32_t *>(data) = val;
}

/**
 * @brief Calculates mean and peak RMS from audio samples
 *
 * @param[in]  frame   audio frame
 * @param[in]  channel channel index to calculate RMS to
 * @param[out] peak    peak RMS
 * @returns            mean RMS
 */
template<int BPS>
static double calculate_rms_helper(const char *channel_data, int sample_count, double *peak)
{
        double sum = 0;
        *peak = 0;
        for (int i = 0; i < sample_count; i += 1) {
                double val = load_sample<BPS>(channel_data + i * BPS) / static_cast<double>(1U << (BPS * CHAR_BIT - 1U));
                sum += val;
                *peak = max(fabs(val), *peak);
        }

        double average = sum / sample_count;

        double sumMeanSquare = 0.0;

        for (int i = 0; i < sample_count; i += 1) {
                sumMeanSquare += pow(load_sample<BPS>(channel_data + i * BPS) / static_cast<double>(1U << (BPS * CHAR_BIT - 1U))
                                - average, 2.0);
        }

        double averageMeanSquare = sumMeanSquare / sample_count;
        double rootMeanSquare = sqrt(averageMeanSquare);

        return rootMeanSquare;
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
        int sample_count = frame->get_data_len(channel) / frame->get_bps();
        switch (frame->get_bps()) {
                case 1:
                        return calculate_rms_helper<1>(frame->get_data(channel), sample_count, peak);
                case 2:
                        return calculate_rms_helper<2>(frame->get_data(channel), sample_count, peak);
                case 3:
                        return calculate_rms_helper<3>(frame->get_data(channel), sample_count, peak);
                case 4:
                        return calculate_rms_helper<4>(frame->get_data(channel), sample_count, peak);
                default:
                        LOG(LOG_LEVEL_FATAL) << "Wrong BPS " << frame->get_bps() << "\n";
                        abort();
        }
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

int32_t downshift_with_dither(int32_t val, int shift){
        static thread_local std::uint_fast32_t last_rand = 1;

        const int mask = (1 << shift) - 1;

        //Pseudorandom number generation, same parameters as std::minstd_rand
        last_rand = (last_rand * 48271) % 2147483647;
        int triangle_dither = last_rand & mask;
        last_rand = (last_rand * 48271) % 2147483647;
        triangle_dither -= last_rand & mask; //triangle probability distribution

        /* Prevent over/underflow when val is big.
         *
         * abs(val) could cause problems if val is INT32_MIN, but integer
         * 32-bit pcm is rare and should not contain the value INT32_MIN
         * anyway because of symmetry, as specified by AES17 and IEC 61606-3
         */
        if(INT32_MAX - abs(val) < mask)
                return val >> shift;

        return (val + triangle_dither) >> shift;
}

void change_bps(char *out, int out_bps, const char *in, int in_bps, int in_len /* bytes */)
{
        int i;

        assert ((unsigned int) out_bps <= sizeof(int32_t));

        for(i = 0; i < in_len / in_bps; i++) {
                int32_t in_value = format_from_in_bps(in, in_bps);

                int32_t out_value;

                if(in_bps > out_bps) {
                        const int downshift = in_bps * 8 - out_bps * 8;
                        out_value = downshift_with_dither(in_value, downshift);
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

template<int BPS>
static double get_avg_volume_helper(const char *data, int sample_count, int stream_channels, int pos_in_stream)
{
        int64_t vol = 0;

        data += pos_in_stream * BPS;

        for (int i = 0; i < sample_count; i++) {
                int32_t in_value = load_sample<BPS> (data + i * BPS * stream_channels);

                vol += labs(in_value);
        }

        return static_cast<double>(vol) / sample_count / ((1U << (BPS * 8U - 1U)));
}

double get_avg_volume(char *data, int bps, int sample_count, int stream_channels, int pos_in_stream) {
        switch (bps) {
                case 1:
                        return get_avg_volume_helper<1>(data, sample_count, stream_channels, pos_in_stream);
                case 2:
                        return get_avg_volume_helper<2>(data, sample_count, stream_channels, pos_in_stream);
                case 3:
                        return get_avg_volume_helper<3>(data, sample_count, stream_channels, pos_in_stream);
                case 4:
                        return get_avg_volume_helper<4>(data, sample_count, stream_channels, pos_in_stream);
                default:
                        LOG(LOG_LEVEL_FATAL) << "Wrong BPS " << bps << "\n";
                        abort();
        }
}

const float INT_MAX_FLT = nexttowardf((float) INT_MAX, INT_MAX); // max int representable as float

/**
 * Can be used in situ.
 */
void float2int(char *out, const char *in, int len)
{
        const float *inf = (const float *)(const void *) in;
        int32_t *outi = (int32_t *)(void *) out;
        int items = len / sizeof(int32_t);

        while(items-- > 0) {
                float sample = *inf++;
                if(sample > 1.0) sample = 1.0;
                if(sample < -1.0) sample = -1.0;
                *outi++ = sample * INT_MAX_FLT;
        }
}

void int2float(char *out, const char *in, int len)
{
        const int32_t *ini = (const int32_t *)(const void *) in;
        float *outf = (float *)(void *) out;
        int items = len / sizeof(int32_t);

        while(items-- > 0) {
                *outf++ = (float) *ini++ / (float) INT_MAX;
        }
}

void short_int2float(char *out, const char *in, int in_len)
{
        const auto *ini = reinterpret_cast<const int16_t *>(in);
        float *outf = (float *)(void *) out;
        int items = in_len / sizeof(int16_t);

        while(items-- > 0) {
                *outf++ = (float) *ini++ / SHRT_MAX;
        }
}

/**
 * Converts int8_t samples to uint8_t by adding 128 (standard
 * shifted zero unsigned samples).
 *
 * Works also in the opposite direction!
 *
 * Can be used in situ.
 */
void signed2unsigned(char *out, const char *in, int in_len)
{
        const auto *inch = reinterpret_cast<const int8_t *>(in);
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
        switch (bps) {
                case 1: return load_sample<1>(in);
                case 2: return load_sample<2>(in);
                case 3: return load_sample<3>(in);
                case 4: return load_sample<4>(in);
                default: abort();
        }
}

void format_to_out_bps(char *out, int bps, int32_t out_value) {

        switch (bps) {
                case 1: store_sample<1>(out, out_value); break;
                case 2: store_sample<2>(out, out_value); break;
                case 3: store_sample<3>(out, out_value); break;
                case 4: store_sample<4>(out, out_value); break;
        }
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

