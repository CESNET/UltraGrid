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
 * WAV file format is taken from:
 * http://www-mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

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

FILE *wav_write_header(const char *filename, struct audio_desc fmt) {
        FILE *wav = fopen(filename, "wb+");
        if (wav == NULL) {
                fprintf(stderr, "[WAV writer] Output file creating error.\n");
                return NULL;
        }

        size_t res = 0;

        res = fwrite("RIFF", 4, 1, wav);
        if(res != 1) {
                return NULL;
        }
        // chunk size - to be added
        res = fwrite("    ", 4, 1, wav);
        if(res != 1) {
                return NULL;
        }
        res = fwrite("WAVE", 4, 1, wav);
        if(res != 1) {
                return NULL;
        }
        res = fwrite("fmt ", 4, 1, wav);
        if(res != 1) {
                return NULL;
        }

        uint32_t chunk_size = FMT_CHUNK_SIZE;
        res = fwrite(&chunk_size, sizeof(chunk_size), 1, wav);
        if(res != 1) {
                return NULL;
        }

        uint16_t wave_format_pcm = 0x0001;
        res = fwrite(&wave_format_pcm, sizeof(wave_format_pcm), 1, wav);
        if(res != 1) {
                return NULL;
        }

        uint16_t channels = fmt.ch_count;
        res = fwrite(&channels, sizeof(channels), 1, wav);
        if(res != 1) {
                return NULL;
        }

        uint32_t sample_rate = fmt.sample_rate;
        res = fwrite(&sample_rate, sizeof(sample_rate), 1, wav);
        if(res != 1) {
                return NULL;
        }

        uint32_t avg_bytes_per_sec = fmt.sample_rate * fmt.bps * fmt.ch_count;
        res = fwrite(&avg_bytes_per_sec, sizeof(avg_bytes_per_sec), 1, wav);
        if(res != 1) {
                return NULL;
        }

        uint16_t block_align_offset = fmt.bps * fmt.ch_count;
        res = fwrite(&block_align_offset, sizeof(block_align_offset), 1, wav);
        if(res != 1) {
                return NULL;
        }

        uint16_t bits_per_sample = fmt.bps * CHAR_BIT;
        res = fwrite(&bits_per_sample, sizeof(bits_per_sample), 1, wav);
        if(res != 1) {
                return NULL;
        }

        res = fwrite("data", 4, 1, wav);
        if(res != 1) {
                return NULL;
        }

        char blank[4] = "";
        res = fwrite(blank, 1, 4, wav);
        if(res != 4) {
                return NULL;
        }

        return wav;
}

bool wav_finalize(FILE *wav, int bps, int ch_count, long long total_samples)
{
        int padding_byte_len = 0;
        if ((ch_count * bps * total_samples) % 2 == 1) {
                char padding_byte = '\0';
                padding_byte_len = 1;
                if (fwrite(&padding_byte, sizeof(padding_byte), 1, wav) != 1) {
                        goto error;
                }
        }


        int ret = fseek(wav, CK_MASTER_SIZE_OFFSET, SEEK_SET);
        if (ret != 0) {
                goto error;
        }
        long long ck_master_size = 4 + FMT_CHUNK_SIZE_BRUT + (DATA_CHUNK_HDR_SIZE + bps *
                        ch_count * total_samples + padding_byte_len);
        uint32_t val = ck_master_size < UINT32_MAX ? ck_master_size : UINT32_MAX;
        size_t res = fwrite(&val, sizeof val, 1, wav);
        if(res != 1) {
                goto error;
        }

        ret = fseek(wav, CK_DATA_SIZE_OFFSET, SEEK_SET);
        if (ret != 0) {
                goto error;
        }
        long long ck_data_size = bps *
                        ch_count * total_samples;
        val = ck_data_size < UINT32_MAX ? ck_data_size : UINT32_MAX;
        res = fwrite(&val, sizeof val, 1, wav);
        if(res != 1) {
                goto error;
        }

        fclose(wav);
        return true;
error:
        fprintf(stderr, "[Audio export] Could not finalize file. Audio file may be corrupted.\n");
        fclose(wav);
        return false;
}

