/**
 * @file   audio/wav_writer.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2021 CESNET, z. s. p. o.
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

/**
 * @file
 * References:
 * [1]: http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
 * [2]: http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/Docs/multichaudP.pdf
 * [3]: https://docs.microsoft.com/en-us/windows/win32/api/mmeapi/ns-mmeapi-waveformatex
 *
 * WAV file format is taken from [1].
 *
 * Note that this page states that WAVE_FORMAT_EXTENSIBLE should be used if:
 *
 * - PCM data has more than 16 bits/sample.
 * - The number of channels is more than 2.
 * - The actual number of bits/sample is not equal to the container size.
 * - The mapping from channels to speakers needs to be specified.
 *
 * However, the document [2] doesn't mention this and there is no ambiguity for UG - we do
 * not specify channel layout, bits per sample matches container size and is never padded.
 * Resource [3] also suggests this use but strictly speaking this is documentation of system
 * API, not file format. flac(1) complains if the original format is used, FFmpeg RIFF writter
 * also follows the above rules.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "audio/utils.h"
#include "audio/wav_writer.h"

/* Chunk size: 4 + 24 + (8 + M * Nc * Ns + (0 or 1)) */
#define CK_MASTER_SIZE_OFFSET            4
#define CK_DATA_SIZE_OFFSET              40 /*  M * Nc * Ns */
#define FMT_CHUNK_SIZE                   16 /* fmt chunk size including strlen("fmt ") and sizeof(uint32_t) */
#define FMT_CHUNK_SIZE_BRUT (FMT_CHUNK_SIZE + 8) /* fmt chunk size including strlen("fmt ") and sizeof(uint32_t) */
#define DATA_CHUNK_HDR_SIZE              8  /* data header size - strlen("data") + sizeof(uint32_t) */

#define NCHANNELS_OFFSET                 22 /* Nc */
#define NSAMPLES_PER_SEC_OFFSET          24 /* F */
#define NAVG_BYTES_PER_SEC_OFFSET        28 /* F * M * Nc */
#define NBLOCK_ALIGN_OFFSET              32 /* M * Nc */
#define NBITS_PER_SAMPLE                 34 /* rounds up to 8 * M */

struct wav_writer_file {
        FILE *outfile;
        struct audio_desc fmt;
        long long samples_written;
};

#define CHECK_FWRITE(a, b, c, d) do { if (fwrite(a, b, c, d) != (c)) { \
        return false; \
} } while(0)

static bool wav_write_header_data(FILE *wav, struct audio_desc fmt) {

        uint32_t max_size = UINT32_MAX;
        CHECK_FWRITE("RIFF", 4, 1, wav);
        // chunk size - to be added
        CHECK_FWRITE(&max_size, 4, 1, wav);
        CHECK_FWRITE("WAVE", 4, 1, wav);
        CHECK_FWRITE("fmt ", 4, 1, wav);

        uint32_t chunk_size = FMT_CHUNK_SIZE;
        CHECK_FWRITE(&chunk_size, sizeof(chunk_size), 1, wav);

        uint16_t wave_format_pcm = 0x0001;
        CHECK_FWRITE(&wave_format_pcm, sizeof(wave_format_pcm), 1, wav);

        uint16_t channels = fmt.ch_count;
        CHECK_FWRITE(&channels, sizeof(channels), 1, wav);

        uint32_t sample_rate = fmt.sample_rate;
        CHECK_FWRITE(&sample_rate, sizeof(sample_rate), 1, wav);

        uint32_t avg_bytes_per_sec = fmt.sample_rate * fmt.bps * fmt.ch_count;
        CHECK_FWRITE(&avg_bytes_per_sec, sizeof(avg_bytes_per_sec), 1, wav);

        uint16_t block_align_offset = fmt.bps * fmt.ch_count;
        CHECK_FWRITE(&block_align_offset, sizeof(block_align_offset), 1, wav);

        uint16_t bits_per_sample = fmt.bps * CHAR_BIT;
        CHECK_FWRITE(&bits_per_sample, sizeof(bits_per_sample), 1, wav);

        CHECK_FWRITE("data", 4, 1, wav);

        CHECK_FWRITE(&max_size, 1, 4, wav); // data size - to be added

        return true;
}

struct wav_writer_file *wav_writer_create(const char *filename, struct audio_desc fmt) {
        FILE *wav = fopen(filename, "wb+");
        if (wav == NULL) {
                perror("[WAV writer] Output file creating error\n");
                return NULL;
        }
        if (!wav_write_header_data(wav, fmt)) {
                fclose(wav);
                return NULL;
        }
        struct wav_writer_file *out = calloc(1, sizeof *out);
        out->outfile = wav;
        out->fmt = fmt;

        return out;
}

bool wav_writer_write(struct wav_writer_file *wav, long long sample_count, const char *data)
{
        char *tmp = NULL;
        if (wav->fmt.bps == 1) {
                long long size = sample_count * wav->fmt.bps;
                tmp = malloc(size);
                signed2unsigned(tmp, data, size);
                data = tmp;
        }
        size_t res = fwrite(data, wav->fmt.bps * wav->fmt.ch_count, sample_count, wav->outfile);
        wav->samples_written += res;
        free(tmp);
        if (res != (size_t) sample_count) {
                return false;
        }
        return true;
}

bool wav_writer_close(struct wav_writer_file *wav)
{
        int padding_byte_len = 0;
        if ((wav->fmt.ch_count * wav->fmt.bps * wav->samples_written) % 2 == 1) {
                char padding_byte = '\0';
                padding_byte_len = 1;
                if (fwrite(&padding_byte, sizeof(padding_byte), 1, wav->outfile) != 1) {
                        goto error;
                }
        }

        int64_t ret = _fseeki64(wav->outfile, CK_MASTER_SIZE_OFFSET, SEEK_SET);
        if (ret != 0) {
                goto error;
        }
        long long ck_master_size = 4 + FMT_CHUNK_SIZE_BRUT + (DATA_CHUNK_HDR_SIZE + wav->fmt.bps *
                        wav->fmt.ch_count * wav->samples_written + padding_byte_len);
        if (ck_master_size > UINT32_MAX) {
                fprintf(stderr, "[WAV writer] Data size exceeding 4 GiB, resulting file may be incompatible!\n");
        }

        uint32_t val = ck_master_size < UINT32_MAX ? ck_master_size : UINT32_MAX;
        size_t res = fwrite(&val, sizeof val, 1, wav->outfile);
        if(res != 1) {
                goto error;
        }

        ret = _fseeki64(wav->outfile, CK_DATA_SIZE_OFFSET, SEEK_SET);
        if (ret != 0) {
                goto error;
        }
        long long ck_data_size = wav->fmt.bps *
                        wav->fmt.ch_count * wav->samples_written;
        val = ck_data_size < UINT32_MAX ? ck_data_size : UINT32_MAX;
        res = fwrite(&val, sizeof val, 1, wav->outfile);
        if(res != 1) {
                goto error;
        }

        fclose(wav->outfile);
        free(wav);
        return true;
error:
        fprintf(stderr, "[Audio export] Could not finalize file. Audio file may be corrupted.\n");
        fclose(wav->outfile);
        free(wav);
        return false;
}

