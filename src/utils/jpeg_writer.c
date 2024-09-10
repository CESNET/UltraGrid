/**
 * @file   utils/jpeg_writer.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * The code of gpujpeg_writer is derived from
 * [GPUJPEG](https://github.com/CESNET/GPUJPEG).
 *
 * Currently, it is intended mainly to asssemble JFIF from RFC 2435 packed
 * JPEG (abbreviated format in RTP, so that feature set is limited to that.
 * Following the properties assumed RFC are expected:
 *
 * - 3 channels, YCbCr
 * - default Huffman tables
 * - 2 quantization tables - first for first (luminance) channel, the other
 *   for chrominance
 *
 * (see also jpeg_writer_write_headers() for details)
 */
/*
 * Copyright (c) 2024 CESNET
 * Copyright (c) 2011, Silicon Genome, LLC.
 *
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "jpeg_reader.h"

#include <stdint.h>      // for uint8_t

#include "jpeg_writer.h"

enum jpeg_marker_code {
        JPEG_MARKER_SOF0 = 0xc0,
        JPEG_MARKER_SOF1 = 0xc1,
        JPEG_MARKER_SOF2 = 0xc2,
        JPEG_MARKER_SOF3 = 0xc3,

        JPEG_MARKER_SOF5 = 0xc5,
        JPEG_MARKER_SOF6 = 0xc6,
        JPEG_MARKER_SOF7 = 0xc7,

        JPEG_MARKER_JPG   = 0xc8,
        JPEG_MARKER_SOF9  = 0xc9,
        JPEG_MARKER_SOF10 = 0xca,
        JPEG_MARKER_SOF11 = 0xcb,

        JPEG_MARKER_SOF13 = 0xcd,
        JPEG_MARKER_SOF14 = 0xce,
        JPEG_MARKER_SOF15 = 0xcf,

        JPEG_MARKER_DHT = 0xc4,

        JPEG_MARKER_DAC = 0xcc,

        JPEG_MARKER_RST0 = 0xd0,
        JPEG_MARKER_RST1 = 0xd1,
        JPEG_MARKER_RST2 = 0xd2,
        JPEG_MARKER_RST3 = 0xd3,
        JPEG_MARKER_RST4 = 0xd4,
        JPEG_MARKER_RST5 = 0xd5,
        JPEG_MARKER_RST6 = 0xd6,
        JPEG_MARKER_RST7 = 0xd7,

        JPEG_MARKER_SOI = 0xd8,
        JPEG_MARKER_EOI = 0xd9,
        JPEG_MARKER_SOS = 0xda,
        JPEG_MARKER_DQT = 0xdb,
        JPEG_MARKER_DNL = 0xdc,
        JPEG_MARKER_DRI = 0xdd,
        JPEG_MARKER_DHP = 0xde,
        JPEG_MARKER_EXP = 0xdf,

        JPEG_MARKER_APP0  = 0xe0,
        JPEG_MARKER_APP1  = 0xe1,
        JPEG_MARKER_APP2  = 0xe2,
        JPEG_MARKER_APP3  = 0xe3,
        JPEG_MARKER_APP4  = 0xe4,
        JPEG_MARKER_APP5  = 0xe5,
        JPEG_MARKER_APP6  = 0xe6,
        JPEG_MARKER_APP7  = 0xe7,
        JPEG_MARKER_APP8  = 0xe8,
        JPEG_MARKER_APP9  = 0xe9,
        JPEG_MARKER_APP10 = 0xea,
        JPEG_MARKER_APP11 = 0xeb,
        JPEG_MARKER_APP12 = 0xec,
        JPEG_MARKER_APP13 = 0xed,
        JPEG_MARKER_APP14 = 0xee,
        JPEG_MARKER_APP15 = 0xef,

        JPEG_MARKER_JPG0  = 0xf0,
        JPEG_MARKER_JPG13 = 0xfd,
        JPEG_MARKER_COM   = 0xfe,

        JPEG_MARKER_TEM = 0x01,

        JPEG_MARKER_ERROR = 0x100
};

enum {
        COMP_COUNT = 3,
};

#define jpeg_writer_emit_byte(buffer, value) *(buffer)++ = (uint8_t) (value);
#define jpeg_writer_emit_2byte(buffer, value) \
        { \
                *(buffer)++ = (uint8_t) (((value) >> 8) & 0xFF); \
                *(buffer)++ = (uint8_t) ((value) & 0xFF); \
        }

#define jpeg_writer_emit_marker(buffer, marker) \
        { \
                *(buffer)++ = 0xFF; \
                *(buffer)++ = (uint8_t) (marker); \
        }

static void
jpeg_writer_write_soi(char **buffer)
{
        jpeg_writer_emit_marker(*buffer, JPEG_MARKER_SOI);
}

static void
jpeg_writer_write_app0(char **buffer)
{
        // Length of APP0 block    (2 bytes)
        // Block ID                (4 bytes - ASCII "JFIF")
        // Zero byte               (1 byte to terminate the ID string)
        // Version Major, Minor    (2 bytes - 0x01, 0x01)
        // Units                   (1 byte - 0x00 = none, 0x01 = inch, 0x02 =
        // cm) Xdpu                    (2 bytes - dots per unit horizontal) Ydpu
        // (2 bytes - dots per unit vertical) Thumbnail X size        (1 byte)
        // Thumbnail Y size        (1 byte)
        jpeg_writer_emit_marker(*buffer, JPEG_MARKER_APP0);

        // Length
        jpeg_writer_emit_2byte(*buffer, 2 + 4 + 1 + 2 + 1 + 2 + 2 + 1 + 1);

        // Identifier: ASCII "JFIF"
        jpeg_writer_emit_byte(*buffer, 'J');
        jpeg_writer_emit_byte(*buffer, 'F');
        jpeg_writer_emit_byte(*buffer, 'I');
        jpeg_writer_emit_byte(*buffer, 'F');
        jpeg_writer_emit_byte(*buffer, 0);

        // We currently emit version code 1.01 since we use no 1.02 features.
        // This may avoid complaints from some older decoders.
        // Major version
        jpeg_writer_emit_byte(*buffer, 1);
        // Minor version
        jpeg_writer_emit_byte(*buffer, 1);
        // Pixel size information
        jpeg_writer_emit_byte(*buffer, 1);
        jpeg_writer_emit_2byte(*buffer, 300);
        jpeg_writer_emit_2byte(*buffer, 300);
        // No thumbnail image
        jpeg_writer_emit_byte(*buffer, 0);
        jpeg_writer_emit_byte(*buffer, 0);
}

static void
jpeg_writer_allocate_dqt(char **buffer)
{
        enum {
                MARKER_LEN = 2,
                LENGTH_LEN = 2,
                TYPE_LEN   = 1,
        };
        *buffer += MARKER_LEN;
        *buffer += LENGTH_LEN;
        *buffer += TYPE_LEN;
        *buffer += JPEG_QUANT_SIZE;
}

static void
jpeg_writer_write_dqt(char **buffer, int type, const uint8_t *dqt)
{
        jpeg_writer_emit_marker(*buffer, JPEG_MARKER_DQT);

        // Length
        jpeg_writer_emit_2byte(*buffer, JPEG_QUANT_SIZE + 3);

        // Index: Y component = 0, Cb or Cr component = 1
        jpeg_writer_emit_byte(*buffer, type);

        // Emit table in zig-zag order
        for (int i = 0; i < JPEG_QUANT_SIZE; i++) {
                unsigned char qval = (unsigned char) ((char) (dqt[i]));
                jpeg_writer_emit_byte(*buffer, qval);
        }
}

static void
jpeg_writer_write_sof0(char **buffer, unsigned width, unsigned height,
                       enum gpujpeg_writer_subsasmpling subsampling)
{
        jpeg_writer_emit_marker(*buffer, JPEG_MARKER_SOF0);

        // Length
        jpeg_writer_emit_2byte(*buffer, 8 + 3 * COMP_COUNT);

        // Precision (bit depth)
        jpeg_writer_emit_byte(*buffer, 8);
        // Dimensions
        jpeg_writer_emit_2byte(*buffer, height);
        jpeg_writer_emit_2byte(*buffer, width);

        // Number of components
        jpeg_writer_emit_byte(*buffer, COMP_COUNT);

        // Components
        for (int comp_index = 0; comp_index < COMP_COUNT; comp_index++) {
                // // Get component
                // struct jpeg_component* component =
                // &encoder->coder.component[comp_index];

                // Component ID
                jpeg_writer_emit_byte(*buffer, comp_index + 1);

                // Sampling factors: 422 - [2,1],[1.1],[1,1]; 420 - [2,2][1...
                int sampling_factor_horizontal = comp_index == 0 ? 2 : 1;
                int sampling_factor_vertical   = 1;
                if (comp_index == 0 && subsampling == JPEG_WRITER_SUBS_420) {
                        sampling_factor_vertical = 2;
                }
                jpeg_writer_emit_byte(*buffer,
                                      (sampling_factor_horizontal << 4) +
                                          sampling_factor_vertical);

                // Quantization table index
                if (comp_index == 0) { // Y
                        jpeg_writer_emit_byte(*buffer, 0);
                } else { // luminance
                        jpeg_writer_emit_byte(*buffer, 1);
                }
        }
}

static void
jpeg_writer_write_dhts(char **buffer)
{
        for (int table = 0; table < 4; table++) {
                jpeg_writer_emit_marker(*buffer, JPEG_MARKER_DHT);

                int length = 0;
                for (int i = 0; i < 16; i++) {
                        length += default_huffman_tables[table].bits[i];
                }

                jpeg_writer_emit_2byte(*buffer, length + 2 + 1 + 16);

                int index = 0;
                switch ((enum huff_type) table) {
                case LUM_AC:
                        index = 16;
                        break;
                case LUM_DC:
                        index = 0;
                        break;
                case CHR_AC:
                        index = 17;
                        break;
                case CHR_DC:
                        index = 1;
                        break;
                }
                jpeg_writer_emit_byte(*buffer, index);

                for (int i = 0; i < 16; i++) {
                        jpeg_writer_emit_byte(
                            *buffer, default_huffman_tables[table].bits[i]);
                }

                // Varible-length
                for (int i = 0; i < length; i++) {
                        jpeg_writer_emit_byte(
                            *buffer, default_huffman_tables[table].bufval[i]);
                }
        }
}

static void
jpeg_writer_write_dri(char **buffer, unsigned restart_interval)
{
        jpeg_writer_emit_marker(*buffer, JPEG_MARKER_DRI);

        // Length
        jpeg_writer_emit_2byte(*buffer, 4);

        // Restart interval
        jpeg_writer_emit_2byte(*buffer, restart_interval);
}

static void
gpujpeg_writer_write_scan_header(char **buffer)
{
        // Begin scan header
        jpeg_writer_emit_marker(*buffer, JPEG_MARKER_SOS);

        // Length
        jpeg_writer_emit_2byte(*buffer, 6 + 2 * COMP_COUNT);

        // Component count
        jpeg_writer_emit_byte(*buffer, COMP_COUNT);

        // Components
        for (int comp_index = 0; comp_index < COMP_COUNT; comp_index++) {
                // Component ID
                jpeg_writer_emit_byte(*buffer, comp_index + 1);

                // Component DC and AC entropy coding table indexes
                if (comp_index == 0) {                        // luminance
                        jpeg_writer_emit_byte(*buffer, 0);    // (0 << 4) | 0
                } else {                                      // chrominance
                        jpeg_writer_emit_byte(*buffer, 0x11); // (1 << 4) | 1
                }
        }

        jpeg_writer_emit_byte(*buffer, 0);    // Ss
        jpeg_writer_emit_byte(*buffer, 0x3F); // Se
        jpeg_writer_emit_byte(*buffer, 0);    // Ah/Al
}

/**
 * @brief writes JPEG/JFIF headers up to the scan (including SOS header)
 *
 * Interleaved YCbCr 4:2:2 and 4:2:0 is supported as specified by RFC 2435
 * section 4.1 defined Type 0, 1, 64 and 65.
 *
 * Space for DQT marker is allocated and it  _must_ be filled later with
 * jpeg_writer_fill_dqt().
 *
 * @param      info    JPEG metadata to be written
 * @param[in]  buffer  where to write the jpeg data
 * @param[out] buffer  pointer to the input buffer after the written JPEG
 *                     headers (here the scan data should be written)
 */
void
jpeg_writer_write_headers(char **buffer, struct jpeg_writer_data *info)
{
        jpeg_writer_write_soi(buffer);
        jpeg_writer_write_app0(buffer);
        info->dqt_marker_start = *buffer;
        for (int i = 0; i < 2; i++) {
                jpeg_writer_allocate_dqt(buffer);
        }
        jpeg_writer_write_sof0(buffer, info->width, info->height,
                               info->subsampling);

        jpeg_writer_write_dhts(buffer);

        if (info->restart_interval > 0) {
                jpeg_writer_write_dri(buffer, info->restart_interval);
        }

        gpujpeg_writer_write_scan_header(buffer);
}

/**
 * @param buffer  pass jpeg_writer_data.dqt_marker_start
 */
void
jpeg_writer_fill_dqt(char *buffer, uint8_t quant_table[2][JPEG_QUANT_SIZE])
{
        for (int i = 0; i < 2; i++) {
                jpeg_writer_write_dqt(&buffer, i, quant_table[i]);
        }
}

void
jpeg_writer_write_eoi(char **buffer)
{
        jpeg_writer_emit_marker(*buffer, JPEG_MARKER_EOI);
}
