/**
 * @file   openapv/from_openapv_conversions.c
 * @author Juraj Zemančík    <550535@mail.muni.cz>
 */
/*
 * Copyright (c) 2026 CESNET
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

#include "from_openapv_conversions.h"

#include "../from_planar.h"
#include "../types.h"

#include <oapv/oapv.h>
#include <stdint.h>

static void yuv4444p10le_to_vuya(const struct from_planar_data d) {
        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_y = (const void *) (d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_u = (const void *) (d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_v = (const void *) (d.in_data[2] + d.in_linesize[2] * y);
                const uint16_t *src_a = (const void *) (d.in_data[3] + d.in_linesize[3] * y);
                uint8_t *dst = d.out_data + (size_t) y * d.out_pitch;

                const int width = d.width;
                for (int x = 0; x < width; ++x) {
                        *dst++ = (uint8_t) (*src_v++ >> 2); // V
                        *dst++ = (uint8_t) (*src_u++ >> 2); // U
                        *dst++ = (uint8_t) (*src_y++ >> 2); // Y
                        *dst++ = (uint8_t) (*src_a++ >> 2); // A
                }
        }
}

static void yuv444p10le_to_y416(const struct from_planar_data d) {
        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_y = (const void *) (d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_u = (const void *) (d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_v = (const void *) (d.in_data[2] + d.in_linesize[2] * y);
                uint16_t *dst = (void *) (d.out_data + (size_t) y * d.out_pitch);

                const int width = d.width;
                for (int x = 0; x < width; ++x) {
                        *dst++ = (uint16_t) (*src_u++ << 6); // Cb
                        *dst++ = (uint16_t) (*src_y++ << 6); // Y
                        *dst++ = (uint16_t) (*src_v++ << 6); // Cr
                        *dst++ = 0xFFFFU;                     // A
                }
        }
}

static void yuv4444p10le_to_y416(const struct from_planar_data d) {
        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_y = (const void *) (d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_u = (const void *) (d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_v = (const void *) (d.in_data[2] + d.in_linesize[2] * y);
                const uint16_t *src_a = (const void *) (d.in_data[3] + d.in_linesize[3] * y);
                uint16_t *dst = (void *) (d.out_data + (size_t) y * d.out_pitch);

                const int width = d.width;
                for (int x = 0; x < width; ++x) {
                        *dst++ = (uint16_t) (*src_u++ << 6); // Cb
                        *dst++ = (uint16_t) (*src_y++ << 6); // Y
                        *dst++ = (uint16_t) (*src_v++ << 6); // Cr
                        *dst++ = (uint16_t) (*src_a++ << 6); // A
                }
        }
}

static void yuv444p12le_to_y416(const struct from_planar_data d) {
        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_y = (const void *) (d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_u = (const void *) (d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_v = (const void *) (d.in_data[2] + d.in_linesize[2] * y);
                uint16_t *dst = (void *) (d.out_data + (size_t) y * d.out_pitch);

                const int width = d.width;
                for (int x = 0; x < width; ++x) {
                        *dst++ = (uint16_t) (*src_u++ << 4); // Cb
                        *dst++ = (uint16_t) (*src_y++ << 4); // Y
                        *dst++ = (uint16_t) (*src_v++ << 4); // Cr
                        *dst++ = 0xFFFFU;                     // A
                }
        }
}

static const struct from_openapv_conversion from_openapv_conversions[] = {
        { UYVY, OAPV_CS_YCBCR422_10LE,  yuv422p10le_to_uyvy   },
        { YUYV, OAPV_CS_YCBCR422_10LE,  yuv422p_to_yuyv       },
        { v210, OAPV_CS_YCBCR422_10LE,  yuv422p10le_to_v210   },
        { Y416, OAPV_CS_YCBCR444_10LE,  yuv444p10le_to_y416   },
        { VUYA, OAPV_CS_YCBCR4444_10LE, yuv4444p10le_to_vuya  },
        { Y416, OAPV_CS_YCBCR4444_10LE, yuv4444p10le_to_y416  },
        { Y416, OAPV_CS_YCBCR444_12LE,  yuv444p12le_to_y416   },
        { VIDEO_CODEC_NONE, OAPV_CS_UNKNOWN, NULL              },
};

const struct from_openapv_conversion *
get_from_openapv_conversion(codec_t dst_codec, int src_cs)
{
        for (const struct from_openapv_conversion *c = from_openapv_conversions;
             c->dst_codec != VIDEO_CODEC_NONE; ++c) {
                if (c->dst_codec == dst_codec && c->required_src_cs == src_cs) {
                        return c;
                }
        }
        return NULL;
}
