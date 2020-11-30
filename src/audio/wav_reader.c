/**
 * @file   audio/wav_reader.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2019 CESNET, z. s. p. o.
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

#include "audio/wav_reader.h"
#include "debug.h"

#define READ_N(buf, len) \
        if (fread(buf, len, 1, wav_file) != 1) {\
                log_msg(LOG_LEVEL_ERROR, "[WAV] Read error: %s.\n", strerror(errno));\
                return WAV_HDR_PARSE_READ_ERROR;\
        }

static int read_fmt_chunk(FILE *wav_file, struct wav_metadata *metadata)
{
        uint16_t format;
        READ_N(&format, 2);
        if (format != 0x0001) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Expected format 0x0001, 0x%04d given.\n", format);
                return WAV_HDR_PARSE_NOT_PCM;
        }

        uint16_t ch_count;
        READ_N(&ch_count, 2);
        if (ch_count > 128) {
                return WAV_HDR_PARSE_INVALID_PARAM;
        }
        metadata->ch_count = ch_count;

        uint32_t sample_rate;
        READ_N(&sample_rate, sizeof(sample_rate));
        if (sample_rate > 192000) {
                return WAV_HDR_PARSE_INVALID_PARAM;
        }
        metadata->sample_rate = sample_rate;

        uint32_t avg_bytes_per_sec;
        READ_N(&avg_bytes_per_sec, sizeof(avg_bytes_per_sec));

        uint16_t block_align_offset;
        READ_N(&block_align_offset, sizeof(block_align_offset));

        uint16_t bits_per_sample;
        READ_N(&bits_per_sample, sizeof(bits_per_sample));
        if (bits_per_sample > 64) {
                return WAV_HDR_PARSE_INVALID_PARAM;
        }
        metadata->bits_per_sample = bits_per_sample;

        return WAV_HDR_PARSE_OK;
}

#define CHECK(cmd, retval) if (cmd == -1) return retval;
int read_wav_header(FILE *wav_file, struct wav_metadata *metadata)
{
        char buffer[16];
        uint32_t chunk_size;
        rewind(wav_file);

        READ_N(buffer, 4);
        if(strncmp(buffer, "RIFF", 4) != 0) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Expected RIFF chunk, %.4s given.\n", buffer);
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        // this is the length of the rest of the file, we may ignore
        READ_N(&chunk_size, 4);

        READ_N(buffer, 4);
        if (strncmp(buffer, "WAVE", 4) != 0) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Expected WAVE chunk, %.4s given.\n", buffer);
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        bool found_data_chunk = false;
        bool found_fmt_chunk = false;
        while (fread(buffer, 4, 1, wav_file) == 1) {
                READ_N(&chunk_size, 4);
                if (strncmp(buffer, "data", 4) == 0) {
                        found_data_chunk = true;
                        metadata->data_size = chunk_size;
                        metadata->data_offset = ftell(wav_file);
                        CHECK(fseek(wav_file, chunk_size, SEEK_CUR), WAV_HDR_PARSE_READ_ERROR);
                } else if (strncmp(buffer, "fmt ", 4) == 0) {
                        found_fmt_chunk = true;
                        if (chunk_size != 16) {
                                log_msg(LOG_LEVEL_ERROR, "[WAV] Expected fmt chunk size 16, %d given.\n", chunk_size);
                                return WAV_HDR_PARSE_WRONG_FORMAT;
                        }
                        int rc = read_fmt_chunk(wav_file, metadata);
                        if (rc != WAV_HDR_PARSE_OK) {
                                return rc;
                        }
                } else if (strncmp(buffer, "LIST", 4) == 0) {
                        log_msg(LOG_LEVEL_DEBUG, "[WAV] Skipping LIST chunk.\n");
                        CHECK(fseek(wav_file, chunk_size, SEEK_CUR), WAV_HDR_PARSE_READ_ERROR);
                } else if (strncmp(buffer, "JUNK", 4) == 0) {
                        log_msg(LOG_LEVEL_DEBUG, "[WAV] Skipping JUNK chunk.\n");
                        CHECK(fseek(wav_file, chunk_size, SEEK_CUR), WAV_HDR_PARSE_READ_ERROR);
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[WAV] Unknown chunk \"%4s\" found!\n", buffer);
                        return WAV_HDR_PARSE_WRONG_FORMAT;
                }
        }

        if (!found_data_chunk) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Data chunk not found!\n");
                return WAV_HDR_PARSE_READ_ERROR;
        }

        if (!found_fmt_chunk) {
                log_msg(LOG_LEVEL_ERROR, "[WAV] Fmt chunk not found!\n");
                return WAV_HDR_PARSE_READ_ERROR;
        }

        log_msg(LOG_LEVEL_VERBOSE, "[WAV] File parsed correctly - length %u bytes, offset %u.\n",
                        metadata->data_size, metadata->data_offset);
        if (fseek(wav_file, metadata->data_offset, SEEK_SET) != 0) {
                return WAV_HDR_PARSE_READ_ERROR;
        }

        return WAV_HDR_PARSE_OK;
}

const char *get_wav_error(int errcode)
{
        switch(errcode) {
                case WAV_HDR_PARSE_OK:
                        return "Wav header OK";
                case WAV_HDR_PARSE_READ_ERROR:
                        return "Premature end of WAV file";
                case WAV_HDR_PARSE_WRONG_FORMAT:
                        return "Wav header in wrong format";
                case WAV_HDR_PARSE_NOT_PCM:
                        return "Wav not in PCM";
                case WAV_HDR_PARSE_INVALID_PARAM:
                        return "Wrong/unsupported value in header";
                default:
                        log_msg(LOG_LEVEL_ERROR, "[WAV] Unknown error code %d passed!\n", errcode);
                        return "Unknown error";
        }
}

