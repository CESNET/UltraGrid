/**
 * @file   audio/wav_reader.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2018 CESNET, z. s. p. o.
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

#include <stdio.h> // FILE *

#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

struct wav_metadata {
        unsigned int ch_count;
        unsigned int sample_rate;
        unsigned int bits_per_sample;
        uint16_t valid_bits;
        uint32_t channel_mask;

        unsigned int data_size;
        unsigned int data_offset; // from the beginning of file
};

#define WAV_HDR_PARSE_OK           0
#define WAV_HDR_PARSE_READ_ERROR   1
#define WAV_HDR_PARSE_WRONG_FORMAT 2
#define WAV_HDR_PARSE_NOT_PCM      3
#define WAV_HDR_PARSE_INVALID_PARAM 4

#ifdef __cplusplus
extern "C" {
#endif

/**
 * This function reads wav header
 *
 * If read successfully, it leaves read file position at the beginning of data.
 * Currently, only interleaved PCM is supported.
 *
 * @retval WAV_HDR_PARSE_OK if ok
 * @retval WAV_HDR_PARSE_READ_ERROR in case of file read error
 * @retval WAV_HDR_PARSE_WRONG_FORMAT if unsupported wav format
 * @retval WAV_HDR_PARSE_NOT_PCM non-PCM WAV detected
 */
int read_wav_header(FILE *wav_file, struct wav_metadata *metadata);

/**
 * Reads sample_count samples from wav_file according to metadata.
 */
size_t wav_read(void *buffer, size_t sample_count, FILE *wav_file, struct wav_metadata *metadata);

const char *get_wav_error(int errcode);

#ifdef __cplusplus
}
#endif

