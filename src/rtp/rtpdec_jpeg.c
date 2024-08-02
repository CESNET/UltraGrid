
/**
 * @file   rtp/rtpdec_jpeg.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @todo
 * * handle precision=1 (?)
 */
/*
 * Copyright (c) 2024 CESNET
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

#include "rtp/rtpdec_jpeg.h"

#include <assert.h>             // for assert
#include <inttypes.h>           // for uint8_t, uint16_t, uint32_t
#include <stdbool.h>            // for true
#include <stdint.h>             // for PRIu8
#include <stdio.h>              // for fclose, fopen, fwrite, printf, FILE
#include <stdlib.h>             // for exit, malloc, NULL
#include <string.h>             // for memcpy

#include "debug.h"              // for LOG_LEVEL_VERBOSE, MSG, log_msg_once
#include "rtp/pbuf.h"           // for coded_data
#include "rtp/rtp.h"            // for rtp_packet
#include "rtp/rtpdec_state.h"   // for decode_data_rtsp
#include "types.h"              // for tile, video_frame, video_frame_callbacks
#include "utils/jpeg_writer.h"  // for jpeg_writer_data, JPEG_QUANT_SIZE, jpeg_wr...
#include "utils/macros.h"       // for MAX, to_fourcc
#include "video_frame.h"        // for vf_data_deleter

#define MOD_NAME "[rtpdec_jpeg] "

#define GET_BYTE(data) *(*(data))++
#define GET_2BYTE(data) \
        ntohs(*((uint16_t *) (*(data)))); \
        *(data) += sizeof(uint16_t)

enum {
        QUANT_TAB_T_FIRST_STATIC = 128,
        QUANT_TAB_T_DYN   = 255, ///< Q=255
        RTP_SZ_MULTIPLIER = 8,   ///< size in RTP hdr is /8
        RTP_TYPE_RST_BIT  = 64,  ///< indicates presence of RST markers
};

static unsigned
parse_restart_interval(unsigned char **pckt_data, int verbose_adj)
{
        const uint16_t rst_int   = GET_2BYTE(pckt_data);
        const uint16_t tmp       = GET_2BYTE(pckt_data);
        const unsigned f         = tmp >> 15;
        const unsigned l         = (tmp >> 14) & 0x1;
        const unsigned rst_count = tmp & 0x3FFFU;

        MSG(VERBOSE + verbose_adj,
            "JPEG rst int=%" PRIu16 " f=%u l=%u count=%u\n", rst_int, f, l,
            rst_count);

        return rst_int;
}

static void
parse_quant_tables(struct decode_data_rtsp *dec, unsigned char **pckt_data,
                   int q)
{
        assert (q >= QUANT_TAB_T_FIRST_STATIC);

        uint8_t  mbz       = GET_BYTE(pckt_data);
        uint8_t  precision = GET_BYTE(pckt_data);
        uint16_t length    = GET_2BYTE(pckt_data);

        MSG(VERBOSE + dec->jpeg.not_first_run,
            "JPEG quant hdr mbz=%" PRIu8 " prec=%" PRIu8 " len=%" PRIu16 "\n",
            mbz, precision, length);

        if (dec->jpeg.quantization_table_set[q] || // already set
            length == 0) { // quantization table not included in this frame
                *pckt_data += length;
                return;
        }

        assert(length == JPEG_QUANT_SIZE || length == 2 * JPEG_QUANT_SIZE);
        assert(precision == 0);

        uint8_t(*quant_table)[JPEG_QUANT_SIZE] =
            dec->jpeg.quantization_tables[q];

        memcpy(quant_table[0], *pckt_data, sizeof quant_table[0]);
        *pckt_data += sizeof quant_table[0];
        if (length == 2 * JPEG_QUANT_SIZE) {
                memcpy(quant_table[1], *pckt_data, sizeof quant_table[1]);
                *pckt_data += sizeof quant_table[1];
        } else {
                // FFmpeg uses single table if used as suggested in vcap/rtsp.c
                // but according to RFC 2 tables should be present so dup 1st
                log_msg_once(LOG_LEVEL_WARNING, to_fourcc('R', 'D', 'J', 'q'),
                             MOD_NAME
                             "Single quantization table includedd (len=64)!\n");
                memcpy(quant_table[1], quant_table[0], sizeof quant_table[1]);
        }
        dec->jpeg.quantization_table_set[q] = true;
}

static char *
create_jpeg_frame(struct video_frame *frame, unsigned char **pckt_data,
                  char **dqt_start, int verbose_adj)
{
        struct jpeg_writer_data info = { 0 };

        uint8_t type_spec = GET_BYTE(pckt_data);
        *pckt_data += 3; // skip 24 bit Fragment Offset
        uint8_t type   = GET_BYTE(pckt_data);
        uint8_t q      = GET_BYTE(pckt_data);
        uint8_t width  = GET_BYTE(pckt_data);
        uint8_t height = GET_BYTE(pckt_data);

        info.width = frame->tiles[0].width = width * RTP_SZ_MULTIPLIER;
        info.height = frame->tiles[0].height = height * RTP_SZ_MULTIPLIER;

        MSG(VERBOSE + verbose_adj,
            "JPEG type_spec=%" PRIu8 " type=%" PRIu8 " q=%" PRIu8
            " width=%d height=%d\n",
            type_spec, type, q, frame->tiles[0].width, frame->tiles[0].height);

        if ((type & RTP_TYPE_RST_BIT) != 0) {
                info.restart_interval =
                    parse_restart_interval(pckt_data, verbose_adj);
                type &= ~RTP_TYPE_RST_BIT;
        }
        assert(type == 0 || type == 1);
        info.subsampling = type;

        if (type_spec != 0) {
                log_msg_once(LOG_LEVEL_WARNING, to_fourcc('R', 'D', 'J', 't'),
                             MOD_NAME "JPEG type %d; only 0 (progressive) is "
                                      "currently supported!\n",
                             type_spec);
        }

        frame->callbacks.data_deleter = vf_data_deleter;
        frame->tiles[0].data =
            calloc(1, 1000 + 3 * info.width * info.height * 2);

        char *buffer = frame->tiles[0].data;
        jpeg_writer_write_headers(&buffer, &info);
        *dqt_start = info.dqt_marker_start;

        return buffer;
}

static void
skip_jpeg_headers(unsigned char **pckt_data)
{
        *pckt_data += sizeof(uint32_t); // skip type_spec + offset
        uint8_t type = GET_BYTE(pckt_data);
        *pckt_data += 3; // skip Q, Width, Height

        if ((type & RTP_TYPE_RST_BIT) != 0) { // skip Restart Marker header
                *pckt_data += sizeof(uint32_t);
        }
}

/*
 * Table K.1 from JPEG spec.
 */
static const int jpeg_luma_quantizer[64] = {
        16, 11, 10, 16, 24, 40, 51, 61,
        12, 12, 14, 19, 26, 58, 60, 55,
        14, 13, 16, 24, 40, 57, 69, 56,
        14, 17, 22, 29, 51, 87, 80, 62,
        18, 22, 37, 56, 68, 109, 103, 77,
        24, 35, 55, 64, 81, 104, 113, 92,
        49, 64, 78, 87, 103, 121, 120, 101,
        72, 92, 95, 98, 112, 100, 103, 99
};

/*
 * Table K.2 from JPEG spec.
 */
static const int jpeg_chroma_quantizer[64] = {
        17, 18, 24, 47, 99, 99, 99, 99,
        18, 21, 26, 66, 99, 99, 99, 99,
        24, 26, 56, 99, 99, 99, 99, 99,
        47, 66, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99,
        99, 99, 99, 99, 99, 99, 99, 99
};

/*
 * Call MakeTables with the Q factor and two u_char[64] return arrays
 */
static void
MakeTables(int q, u_char *lqt, u_char *cqt)
{
  int i;
  int factor = q;

  if (q < 1) factor = 1;
  if (q > 99) factor = 99;
  if (q < 50)
    q = 5000 / factor;
  else
    q = 200 - factor*2;

  for (i=0; i < 64; i++) {
    int lq = (jpeg_luma_quantizer[i] * q + 50) / 100;
    int cq = (jpeg_chroma_quantizer[i] * q + 50) / 100;

    /* Limit the quantizers to 1 <= q <= 255 */
    if (lq < 1) lq = 1;
    else if (lq > 255) lq = 255;
    lqt[i] = lq;

    if (cq < 1) cq = 1;
    else if (cq > 255) cq = 255;
    cqt[i] = cq;
  }
}

int
decode_frame_jpeg(struct coded_data *cdata, void *decode_data)
{
        struct decode_data_rtsp *dec_data = decode_data;

        struct video_frame *frame = dec_data->frame;
        // table with Q=255 must be always (re)set
        dec_data->jpeg.quantization_table_set[QUANT_TAB_T_DYN] = false;
        uint8_t q = -1;

        while (cdata != NULL) {
                rtp_packet *pckt = cdata->data;
                unsigned char *pckt_data = (unsigned char *) pckt->data;

                uint32_t hdr = 0;
                memcpy(&hdr, pckt_data, sizeof hdr);
                const unsigned off = ntohl(hdr) & 0xFFFFFFU;
                memcpy(&q, pckt_data + 5, sizeof q);
                if (frame->tiles[0].width == 0) {
                        char *hdr_end = create_jpeg_frame(
                            frame, &pckt_data, &dec_data->jpeg.dqt_start,
                            dec_data->jpeg.not_first_run);
                        dec_data->offset_len =
                            (int) (hdr_end - frame->tiles[0].data);
                } else {
                        skip_jpeg_headers(&pckt_data);
                }
                // for q>=128, 1st pckt contains tables
                if (off == 0 && q >= QUANT_TAB_T_FIRST_STATIC) {
                        parse_quant_tables(dec_data, &pckt_data, q);
                }

                const long payload_hdr_len =
                    pckt_data - (unsigned char *) pckt->data;
                const long     data_len  = pckt->data_len - payload_hdr_len;
                const unsigned frame_off = off + dec_data->offset_len;
                memcpy(frame->tiles[0].data + frame_off, pckt_data, data_len);

                const unsigned end_pos = frame_off + data_len;
                frame->tiles[0].data_len =
                    MAX(frame->tiles[0].data_len, end_pos);

                cdata = cdata->nxt;
        }

        if (!dec_data->jpeg.quantization_table_set[q] &&
            q < QUANT_TAB_T_FIRST_STATIC) {
                MakeTables(q, dec_data->jpeg.quantization_tables[q][0],
                           dec_data->jpeg.quantization_tables[q][1]);
                dec_data->jpeg.quantization_table_set[q] = true;
        }

        if (!dec_data->jpeg.quantization_table_set[q]) {
                MSG(WARNING,
                    "Dropping frame - missing quantization tables for Q=%d!\n",
                    q);
                return false;
        }

        jpeg_writer_fill_dqt(dec_data->jpeg.dqt_start,
                             dec_data->jpeg.quantization_tables[q]);

        char *buffer = frame->tiles[0].data + frame->tiles[0].data_len;
        jpeg_writer_write_eoi(&buffer);
        frame->tiles[0].data_len = buffer - frame->tiles[0].data;

        dec_data->jpeg.not_first_run = 1;

        return true;
}
