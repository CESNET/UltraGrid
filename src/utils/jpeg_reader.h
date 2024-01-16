/**
 * @file   utils/jpeg_reader.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
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

#ifndef UTILS_JPEG_READER_H_
#define UTILS_JPEG_READER_H_

#ifndef __cplusplus
#include <stdbool.h>
#include <stdint.h>
#else
#include <cstdint>
#endif

enum jpeg_color_spec {
        JPEG_COLOR_SPEC_NONE = 0,
        JPEG_COLOR_SPEC_YCBCR,
        JPEG_COLOR_SPEC_RGB,
        JPEG_COLOR_SPEC_CMYK,
        JPEG_COLOR_SPEC_YCCK
};

enum {
        JPEG_MAX_COMPONENT_COUNT = 4,
};

struct jpeg_info {
        int width, height, comp_count;
        enum jpeg_color_spec color_spec;
        bool interleaved;
        int sampling_factor_v[4];
        int sampling_factor_h[4];
        int restart_interval; // content of DRI marker
        uint8_t *quantization_tables[2]; // assuming 8-bit tables
        /// mapping component -> index to table_quantization
        int comp_table_quantization_map[JPEG_MAX_COMPONENT_COUNT];
        uint8_t *data; // entropy-coded segment start
        char com[65536 - 2 + 1]; // comment (COM marker)
        uint8_t huff_lum_dc[16 + 256]; // if first byte is 255, table was not defined
        uint8_t huff_lum_ac[16 + 256];
        uint8_t huff_chm_dc[16 + 256];
        uint8_t huff_chm_ac[16 + 256];
};

struct jpeg_rtp_data {
        int width, height;
        int type; // RTP JPEG type
        int q; // either 0..100 or 255 (dynamic QT - included in quantization_tables)
        int restart_interval; // content of DRI marker
        uint8_t *quantization_tables[2]; // assuming 8-bit tables
        uint8_t *data; // entropy-coded data start
};

#ifdef __cplusplus
extern "C" {
#endif // defined __cplusplus

int jpeg_read_info(uint8_t *image, int len, struct jpeg_info *info);
bool jpeg_get_rtp_hdr_data(uint8_t *jpeg_data, int len, struct jpeg_rtp_data *hdr_data);

#ifdef __cplusplus
}
#endif // defined __cplusplus

#endif// UTILS_JPEG_READER_H_

