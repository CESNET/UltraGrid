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


#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstring>

#include "audio/codec.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h" // ADD_TO_PARAM
#include "utils/misc.h"

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
static double calculate_rms_helper(const char *channel_data,
                int sample_count,
                double *peak,
                int sample_stride = 1)
{
        double sum = 0;
        *peak = 0;
        const int byte_stride = BPS * sample_stride;
        for (int i = 0; i < sample_count; i += 1) {
                double val = load_sample<BPS>(channel_data + i * byte_stride) / static_cast<double>(1U << (BPS * CHAR_BIT - 1U));
                sum += val;
                *peak = max(fabs(val), *peak);
        }

        double average = sum / sample_count;

        double sumMeanSquare = 0.0;

        for (int i = 0; i < sample_count; i += 1) {
                sumMeanSquare += pow(load_sample<BPS>(channel_data + i * byte_stride) / static_cast<double>(1U << (BPS * CHAR_BIT - 1U))
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

double calculate_rms(audio_frame *frame, int channel, double *peak)
{
        assert(channel < frame->ch_count);
        char *data = frame->data + channel * frame->bps;
        int sample_count = frame->data_len / frame->bps / frame->ch_count;
        switch (frame->bps) {
        case 1:
                return calculate_rms_helper<1>(data, sample_count, peak, frame->ch_count);
        case 2:
                return calculate_rms_helper<2>(data, sample_count, peak, frame->ch_count);
        case 3:
                return calculate_rms_helper<3>(data, sample_count, peak, frame->ch_count);
        case 4:
                return calculate_rms_helper<4>(data, sample_count, peak, frame->ch_count);
        default:
                LOG(LOG_LEVEL_FATAL) << "Wrong BPS " << frame->bps << "\n";
                abort();
        }
}

bool audio_desc_eq(struct audio_desc a1, struct audio_desc a2) {
        return a1.bps == a2.bps &&
                a1.sample_rate == a2.sample_rate &&
                a1.ch_count == a2.ch_count &&
                a1.codec == a2.codec;
}

struct audio_desc audio_desc_from_frame(const struct audio_frame *frame) {
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
        static thread_local uint32_t last_rand = 0;

        //Quick and dirty random number generation (ranqd1)
        //Numerical Recipes in C, page 284
        last_rand = (last_rand * 1664525) + 1013904223L;
        int triangle_dither = last_rand >> (32 - shift);
        last_rand = (last_rand * 1664525) + 1013904223L;
        triangle_dither -= last_rand >> (32 - shift); //triangle probability distribution

        /* Prevent over/underflow when val is big.
         *
         * abs(val) could cause problems if val is INT32_MIN, but integer
         * 32-bit pcm is rare and should not contain the value INT32_MIN
         * anyway because of symmetry, as specified by AES17 and IEC 61606-3
         */
        if(INT32_MAX - abs(val) < (1 << shift) - 1 + (1 << (shift - 1)))
                return val / (1 << shift);

        if(val > 0)
                val += 1 << (shift - 1);
        else
                val -= 1 << (shift - 1);

        return (val + triangle_dither) / (1 << shift);
}

#define NO_DITHER_PARAM "no-dither"
ADD_TO_PARAM(NO_DITHER_PARAM, "* " NO_DITHER_PARAM "\n"
                "  Disable audio dithering when reducing bit depth\n");

void change_bps(char *out, int out_bps, const char *in, int in_bps, int in_len /* bytes */) {
        static const bool dither = commandline_params.find(NO_DITHER_PARAM) == commandline_params.end();
        change_bps2(out, out_bps, in, in_bps, in_len, dither);
}

void change_bps2(char *out, int out_bps, const char *in, int in_bps, int in_len /* bytes */, bool dither)
{
        assert ((unsigned int) out_bps <= sizeof(int32_t));
        static_assert(-2>>1 == -1, "Implementation-defined behavior doesn't work as expected by the implementation.");

        if (in_bps == out_bps ) {
                memcpy(out, in, in_len);
                return;
        }

        if (in_bps < out_bps ) {
                for (int i = 0; i < in_len / in_bps; i++) {
                        int32_t in_value = format_from_in_bps(in, in_bps);
                        int32_t out_value = in_value << (out_bps * 8 - in_bps * 8);
                        format_to_out_bps(out, out_bps, out_value);
                        in += in_bps;
                        out += out_bps;
                }
                return;
        }

        // downsampling
        if (dither) {
                const int downshift = in_bps * 8 - out_bps * 8;
                for (int i = 0; i < in_len / in_bps; i++) {
                        int32_t in_value = format_from_in_bps(in, in_bps);
                        int32_t out_value = downshift_with_dither(in_value, downshift);
                        format_to_out_bps(out, out_bps, out_value);
                        in += in_bps;
                        out += out_bps;
                }
                return;
        }
        // no dithering
        for (int i = 0; i < in_len / in_bps; i++) {
                int32_t in_value = format_from_in_bps(in, in_bps);
                int32_t out_value = in_value >> (in_bps * 8 - out_bps * 8);
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

const char *audio_desc_to_cstring(struct audio_desc desc) {
        thread_local string str = desc;
        return str.c_str();
}

/**
 * Parses configuration string for audio format.
 *
 * Only members that are specified explicitly by the config string are changed
 * in returned audio desc, the remaining members are left untouched!
 */
int parse_audio_format(const char *str, struct audio_desc *ret) {
        if (strcmp(str, "help") == 0) {
                color_printf(TBOLD("Audio format") " syntax:\n");
                color_printf(TBOLD("\t{channels=<num>|bps=<bits_per_sample>|sample_rate=<rate>}*\n"));
                color_printf("\t\tmultiple options can be separated by a colon\n");
                return 1;
        }

        unique_ptr<char[]> arg_copy(new char[strlen(str) + 1]);
        char *arg = arg_copy.get();
        strcpy(arg, str);

        char *save_ptr = nullptr;
        char *tmp = arg;

        while (char *item = strtok_r(tmp, ",:", &save_ptr)) {
                char *endptr = nullptr;
                if (strncmp(item, "channels=", strlen("channels=")) == 0) {
                        item += strlen("channels=");
                        ret->ch_count = strtol(item, &endptr, 10);
                        if (ret->ch_count < 1 || endptr != item + strlen(item)) {
                                log_msg(LOG_LEVEL_ERROR, "Invalid number of channels %s!\n", item);
                                return -1;
                        }
                } else if (strncmp(item, "bps=", strlen("bps=")) == 0) {
                        item += strlen("bps=");
                        int bps = strtol(item, &endptr, 10);
                        if (bps % 8 != 0 || (bps != 8 && bps != 16 && bps != 24 && bps != 32) || endptr != item + strlen(item)) {
                                log_msg(LOG_LEVEL_ERROR, "Invalid bps %s!\n", item);
                                if (bps % 8 != 0) {
                                        LOG(LOG_LEVEL_WARNING) << "bps is in bits per sample but a value not divisible by 8 was given.\n";
                                }
                                log_msg(LOG_LEVEL_ERROR, "Supported values are 8, 16, 24, or 32 bits.\n");
                                return -1;

                        }
                        ret->bps = bps / 8;
                } else if (strncmp(item, "sample_rate=", strlen("sample_rate=")) == 0) {
                        const char *sample_rate_str = item + strlen("sample_rate=");
                        long long val = unit_evaluate(sample_rate_str);
                        if (val <= 0 || val > numeric_limits<decltype(ret->sample_rate)>::max()) {
                                LOG(LOG_LEVEL_ERROR) << "Invalid sample_rate " << sample_rate_str << "!\n";
                                return -1;
                        }
                        ret->sample_rate = val;
                } else {
                        LOG(LOG_LEVEL_ERROR) << "Unkonwn option \"" << item << "\" for audio format!\n";
                        LOG(LOG_LEVEL_INFO) << "Use \"help\" keyword for syntax.!\n";
                        return -1;
                }

                tmp = nullptr;
        }
        return 0;
}

