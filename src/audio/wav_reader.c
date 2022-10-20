/**
 * @file   audio/wav_reader.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2022 CESNET, z. s. p. o.
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

#include <inttypes.h>
#include <limits.h>
#include <stdbool.h>
#include <stdint.h>

#include "audio/utils.h"
#include "audio/wav_reader.h"
#include "debug.h"
#include "utils/misc.h"

#define MOD_NAME "[WAV reader] "
#define WAV_MAX_BIT_DEPTH 64
#define WAV_MAX_CHANNELS 128

#define READ_N(buf, len) \
        if (fread(buf, len, 1, wav_file) != 1) {\
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Read error: %s.\n", ug_strerror(errno));\
                return WAV_HDR_PARSE_READ_ERROR;\
        }

#define GUID_PCM { 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71 }

static int read_fmt_chunk(FILE *wav_file, struct wav_metadata *metadata, size_t chunk_size)
{
        if (chunk_size != 16 && chunk_size != 18 && chunk_size != 40) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Expected fmt chunk size 16, 18 or 40, %zd given.\n", chunk_size);
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        uint16_t format;
        READ_N(&format, 2);
        if (format != 0x0001 && format != 0xFFFE) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Expected format 0x0001, 0x%04d given.\n", format);
                return WAV_HDR_PARSE_NOT_PCM;
        }

        uint16_t ch_count;
        READ_N(&ch_count, 2);
        if (ch_count == 0 || ch_count > WAV_MAX_CHANNELS) {
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
        if (bits_per_sample == 0 || bits_per_sample > WAV_MAX_BIT_DEPTH) {
                return WAV_HDR_PARSE_INVALID_PARAM;
        }
        metadata->bits_per_sample = bits_per_sample;

        if (chunk_size == 17) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong fmt chunk size 17!\n");
                return WAV_HDR_PARSE_READ_ERROR;
        }

        if (chunk_size >= 18) {
                uint16_t ext_size;
                READ_N(&ext_size, 2);
                if (ext_size != chunk_size - 18) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unexpected ext size %" PRIu16 ", remaining chunk size %zu.\n", ext_size, chunk_size - 18);
                        return WAV_HDR_PARSE_READ_ERROR;
                }
                if (ext_size == 22) {
                        READ_N(&metadata->valid_bits, sizeof metadata->valid_bits);
                        READ_N(&metadata->channel_mask, sizeof metadata->channel_mask);
                        char buffer[16];
                        READ_N(buffer, 16);
                        const char guid[] = GUID_PCM;
                        if (memcmp(guid, buffer, 16) != 0) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "GUID is not PCM!\n");
                                return WAV_HDR_PARSE_NOT_PCM;
                        }
                        format = 0x0001;
                } else if (ext_size != 0) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Extension size either 0 or 22 expected, %d presented.\n", ext_size);
                        return WAV_HDR_PARSE_READ_ERROR;
                }
        }

        if (format == 0xFFFE) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Subtype GUID not found!\n");
                return WAV_HDR_PARSE_READ_ERROR;
        }

        return WAV_HDR_PARSE_OK;
}

static _Bool is_member(const char *needle, const char **haystack) {
        while (*haystack != NULL) {
                if (strncmp(needle, *haystack, 4) == 0) {
                        return 1;
                }
                haystack++;
        }
        return 0;
}

static int fskip(FILE *f, long off) {
        if (_fseeki64(f, off, SEEK_CUR) == 0) {
                return 1;
        }
        while (off-- > 0) {
                if (getc(f) == EOF) {
                        return -1;
                }
        }
        return 1;
}

// Sets data_offset and data_size to metadata
static _Bool process_data_chunk(FILE *wav_file, uint32_t chunk_size, struct wav_metadata *metadata, _Bool found_ds64_chunk) {
        if (!found_ds64_chunk) { // RF64 has this chunk size always -1, value from ds64 is used instead
                metadata->data_size = chunk_size;
        }
        metadata->data_offset = _ftelli64(wav_file);

        if (metadata->data_size == UINT32_MAX) {
                // for UINT32_MAX deduce the length from file length (as FFmpeg does)
                if (metadata->data_offset == -1 || _fseeki64(wav_file, 0, SEEK_END) == -1) {
                        metadata->data_size = -1;
                } else {
                        int64_t data_end = _ftelli64(wav_file);
                        if (_fseeki64(wav_file, metadata->data_offset, SEEK_SET) != 0) {
                                ug_perror(MOD_NAME "cannot seek back to data chunk");
                                return 0;
                        }
                        if (data_end == -1) {
                               return 0;
                        }
                        metadata->data_size = data_end - metadata->data_offset;
                }
        }
        return 1;
}

static _Bool read_ds64(FILE *wav_file, uint32_t chunk_size, struct wav_metadata *metadata) {
        if (chunk_size != 28) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "ds64 chunk size must be 28 but %" PRIu32 " was presented!\n", chunk_size);
                return 0;
        }
        uint64_t tmp = 0;
        READ_N(&tmp, 8); // RF64 chunk size - unused
        READ_N(&tmp, 8);
        assert(tmp <= LLONG_MAX);
        metadata->data_size = tmp;
        READ_N(&tmp, 8); // dummy - should be ignored
        uint32_t table_count = 0;
        READ_N(&table_count, 4); // number of table entries for non-'data' chunks
        if (!fskip(wav_file, (long)(table_count * (sizeof(uint32_t) /*ID*/ + sizeof(uint64_t) /*size*/)))) { // just skip the table contents
                return 0;
        }
        return 1;
}

#define CHECK(cmd, retval) if (cmd == -1) return retval;
int read_wav_header(FILE *wav_file, struct wav_metadata *metadata)
{
        char buffer[16];
        uint32_t chunk_size;
        rewind(wav_file);

        _Bool rf64 = 0; ///< RF64 or BW64
        READ_N(buffer, 4);
        if (strncmp(buffer, "RF64", 4) == 0 || strncmp(buffer, "BW64", 4) == 0) {
                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Using %4.4s WAV file.\n", buffer);
                rf64 = 1;
        } else if (strncmp(buffer, "RIFF", 4) != 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Expected RIFF or RF64/BW64 chunk, %.4s given.\n", buffer);
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        // this is the length of the rest of the file, we may ignore
        READ_N(&chunk_size, 4);

        READ_N(buffer, 4);
        if (strncmp(buffer, "WAVE", 4) != 0) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Expected WAVE chunk, %.4s given.\n", buffer);
                return WAV_HDR_PARSE_WRONG_FORMAT;
        }

        bool found_data_chunk = false;
        bool found_ds64_chunk = false;
        bool found_fmt_chunk = false;
        while (fread(buffer, 4, 1, wav_file) == 1) {
                READ_N(&chunk_size, 4);
                if (strncmp(buffer, "data", 4) == 0) {
                        if (!process_data_chunk(wav_file, chunk_size, metadata, found_ds64_chunk)) {
                                return WAV_HDR_PARSE_READ_ERROR;
                        }
                        found_data_chunk = true;
                        break;
                }
                if (strncmp(buffer, "ds64", 4) == 0) {
                        if (!read_ds64(wav_file, chunk_size, metadata)) {
                                return WAV_HDR_PARSE_WRONG_FORMAT;
                        }
                        found_ds64_chunk = true;
                } else if (strncmp(buffer, "fmt ", 4) == 0) {
                        found_fmt_chunk = true;
                        int rc = read_fmt_chunk(wav_file, metadata, chunk_size);
                        if (rc != WAV_HDR_PARSE_OK) {
                                return rc;
                        }
                } else { // other tags
                        const char *known_tags[] = { "JUNK", "LIST", "id3 ", NULL }; // "olym" ?
                        int level = is_member(buffer, known_tags) ? LOG_LEVEL_VERBOSE : LOG_LEVEL_WARNING;
                        log_msg(level, MOD_NAME "Skipping chunk \"%4.4s\" sized %" PRIu32 " B!\n", buffer, chunk_size);
                        CHECK(fskip(wav_file, chunk_size), WAV_HDR_PARSE_READ_ERROR);
                }
        }

        if (!found_data_chunk) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Data chunk not found!\n");
                return WAV_HDR_PARSE_READ_ERROR;
        }

        if (!found_fmt_chunk) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Fmt chunk not found!\n");
                return WAV_HDR_PARSE_READ_ERROR;
        }

        if (rf64 && !found_ds64_chunk) { // buggy, but we may try to continue witn implicit data size
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Broken WAV - a RF64 file detected but no ds64 chunk found!\n");
        }

        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "File parsed correctly - length %lld bytes, data offset %lld.\n",
                        metadata->data_size, metadata->data_offset);

        return WAV_HDR_PARSE_OK;
}

size_t wav_read(void *buffer, size_t sample_count, FILE *wav_file, struct wav_metadata *metadata)
{
        size_t samples = fread(buffer, metadata->ch_count * metadata->bits_per_sample / 8, sample_count, wav_file);
        if (metadata->bits_per_sample == 8) {
                signed2unsigned(buffer, buffer, samples * metadata->ch_count * metadata->bits_per_sample / 8);
        }
        return samples;
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
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown error code %d passed!\n", errcode);
                        return "Unknown error";
        }
}

int wav_seek(FILE *wav_file, long offset, int whence, struct wav_metadata *metadata)
{
        if (whence == SEEK_CUR || whence == SEEK_END) {
                return fseek(wav_file, offset, whence);
        }
        assert(whence == SEEK_SET);
        return fseek(wav_file, metadata->data_offset + offset, SEEK_SET);
}

