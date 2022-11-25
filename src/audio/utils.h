/**
 * @file   audio/utils.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2021 CESNET, z. s. p. o.
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

#ifndef _AUDIO_UTILS_H_
#define _AUDIO_UTILS_H_

#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
#else
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#endif

#include "audio/types.h"

#ifdef __cplusplus
double calculate_rms(audio_frame2 *frame, int channel, double *peak);
double calculate_rms(audio_frame *frame, int channel, double *peak);
void audio_channel_demux(const audio_frame2 *, int, audio_channel*);
#endif

#ifdef __cplusplus
extern "C" {
#endif

bool audio_desc_eq(struct audio_desc, struct audio_desc);
struct audio_desc audio_desc_from_audio_channel(audio_channel *);
void audio_frame_write_desc(struct audio_frame *f, struct audio_desc desc);

/**
 * Changes bps for everey sample.
 * 
 * The memory areas shouldn't (supposedly) overlap.
 */
void change_bps(char *out, int out_bps, const char *in, int in_bps, int in_len /* bytes */);
/// @brief changes bps with option to specify whether or not do the dithering
void change_bps2(char *out, int out_bps, const char *in, int in_bps, int in_len /* bytes */, bool dither);

/**
 * Bit-shifts val down by shift bits with a triangular dither noise.
 *
 * By reducing bit-depth of audio, quantization noise is introduced. If the bit
 * depth is reduced by consistent rounding (either always down, always up, or
 * always to nearest) the quantization noise is correlated with the input
 * signal and percieved as distortion. Adding a small amount of noise to the
 * input signal before reducing the bit-depth causes random rounding, which
 * makes the noise uncorrelated and uniformly spread through the whole
 * spectrum.
 */
int32_t downshift_with_dither(int32_t val, int shift);

/**
 * Makes n copies of first channel (interleaved).
 */
void audio_frame_multiply_channel(struct audio_frame *frame, int new_channel_count);

/*
 * Extracts out_channel_count of channels from interleaved stream, starting with first_chan
 */
void copy_channel(char *out, const char *in, int bps, int in_len /* bytes */, int out_channel_count);

/*
 * Multiplexes channel into interleaved stream
 */
void mux_channel(char *out, const char *in, int bps, int in_len, int out_stream_channels, int chan_pos_stream, double scale);
void demux_channel(char *out, char *in, int bps, int in_len, int in_stream_channels, int pos_in_stream);
void remux_channel(char *out, const char *in, int bps, int in_len, int in_stream_channels, int out_stream_channels, int pos_in_stream, int pos_out_stream);

void interleaved2noninterleaved(char *out, const char *in, int bps, int in_len /* bytes */, int channel_count);

/*
 * Additional function that allosw mixing channels
 *
 * @return avareage volume
 */
void mux_and_mix_channel(char *out, const char *in, int bps, int in_len, int out_stream_channels, int chan_pos_stream, double scale);
double get_avg_volume(char *data, int bps, int in_len, int stream_channels, int chan_pos_stream);

/**
 * This fuction converts from normalized float to int32_t representation
 * Input and output data may overlap.
 * @param[out] out 4-byte aligned output buffer
 * @param[in] in 4-byte aligned input buffer
 */
void float2int(char *out, const char *in, int len);
/**
 * This fuction converts from int32_t to normalized float
 * Input and output data may overlap.
 * @param[out] out 4-byte aligned output buffer
 * @param[in] in 4-byte aligned input buffer
 */
void int2float(char *out, const char *in, int len);
/**
 * This fuction converts from int16_t to normalized float
 * Input and output data may overlap.
 * @param[out] out 4-byte aligned output buffer
 * @param[in] in 4-byte aligned input buffer
 */
void short_int2float(char *out, const char *in, int in_len);

void signed2unsigned(char *out, const char *in, int in_len);

struct audio_desc audio_desc_from_frame(const struct audio_frame *frame);
const char *audio_desc_to_cstring(struct audio_desc desc);

int32_t format_from_in_bps(const char * in, int bps);
void format_to_out_bps(char *out, int bps, int32_t out_value);

/**
 * Appends data to audio frame
 *
 * @returns whether all data has been written
 */
bool append_audio_frame(struct audio_frame *frame, char *data, size_t data_len);

/**
 * Creates a deep copy of src. Both struct and audio_frame::data is malloc'd.
 *
 * @param keep_size allocate exactly same storage (audio_frame::max_size) as
 *                  original frame. If false, the storage may be shorter.
 */
struct audio_frame *audio_frame_copy(const struct audio_frame *src, bool keep_size);

int parse_audio_format(const char *str, struct audio_desc *ret);

#ifdef __cplusplus
}
#endif

#endif
