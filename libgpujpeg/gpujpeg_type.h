/**
 * Copyright (c) 2011, CESNET z.s.p.o
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

#ifndef GPUJPEG_TYPE_H
#define GPUJPEG_TYPE_H

#include <stdint.h>

/** Contants */
#define GPUJPEG_BLOCK_SIZE                  8
#define GPUJPEG_MAX_COMPONENT_COUNT         3
#define GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE   (GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE * 2)

/** Flags */
#define GPUJPEG_VERBOSE                     1
#define GPUJPEG_OPENGL_INTEROPERABILITY     2

/**
 * Color spaces for JPEG codec
 */
enum gpujpeg_color_space {
    GPUJPEG_RGB = 1,
    GPUJPEG_YCBCR_ITU_R = 2,
    GPUJPEG_YCBCR_JPEG = 3,
};

/**
 * Sampling factor for image data
 */
enum gpujpeg_sampling_factor {
    GPUJPEG_4_4_4 = ((4 << 16) | (4 << 8) | 4),
    GPUJPEG_4_2_2 = ((4 << 16) | (2 << 8) | 2),
};

/**
 * Sampling factor for color component in JPEG format
 */
struct gpujpeg_component_sampling_factor
{
    uint8_t horizontal;
    uint8_t vertical;
};

/**
 * JPEG component type
 */
enum gpujpeg_component_type {
    GPUJPEG_COMPONENT_LUMINANCE = 0,
    GPUJPEG_COMPONENT_CHROMINANCE = 1,
    GPUJPEG_COMPONENT_TYPE_COUNT = 2
};

/** 
 * JPEG huffman type 
 */
enum gpujpeg_huffman_type {
    GPUJPEG_HUFFMAN_DC = 0,
    GPUJPEG_HUFFMAN_AC = 1,
    GPUJPEG_HUFFMAN_TYPE_COUNT = 2
};

#include <stdio.h>

/** 
 * JPEG marker codes 
 */
enum gpujpeg_marker_code {		
    GPUJPEG_MARKER_SOF0  = 0xc0,
    GPUJPEG_MARKER_SOF1  = 0xc1,
    GPUJPEG_MARKER_SOF2  = 0xc2,
    GPUJPEG_MARKER_SOF3  = 0xc3,
  
    GPUJPEG_MARKER_SOF5  = 0xc5,
    GPUJPEG_MARKER_SOF6  = 0xc6,
    GPUJPEG_MARKER_SOF7  = 0xc7,
  
    GPUJPEG_MARKER_JPG   = 0xc8,
    GPUJPEG_MARKER_SOF9  = 0xc9,
    GPUJPEG_MARKER_SOF10 = 0xca,
    GPUJPEG_MARKER_SOF11 = 0xcb,
  
    GPUJPEG_MARKER_SOF13 = 0xcd,
    GPUJPEG_MARKER_SOF14 = 0xce,
    GPUJPEG_MARKER_SOF15 = 0xcf,
  
    GPUJPEG_MARKER_DHT   = 0xc4,
  
    GPUJPEG_MARKER_DAC   = 0xcc,
  
    GPUJPEG_MARKER_RST0  = 0xd0,
    GPUJPEG_MARKER_RST1  = 0xd1,
    GPUJPEG_MARKER_RST2  = 0xd2,
    GPUJPEG_MARKER_RST3  = 0xd3,
    GPUJPEG_MARKER_RST4  = 0xd4,
    GPUJPEG_MARKER_RST5  = 0xd5,
    GPUJPEG_MARKER_RST6  = 0xd6,
    GPUJPEG_MARKER_RST7  = 0xd7,
  
    GPUJPEG_MARKER_SOI   = 0xd8,
    GPUJPEG_MARKER_EOI   = 0xd9,
    GPUJPEG_MARKER_SOS   = 0xda,
    GPUJPEG_MARKER_DQT   = 0xdb,
    GPUJPEG_MARKER_DNL   = 0xdc,
    GPUJPEG_MARKER_DRI   = 0xdd,
    GPUJPEG_MARKER_DHP   = 0xde,
    GPUJPEG_MARKER_EXP   = 0xdf,
  
    GPUJPEG_MARKER_APP0  = 0xe0,
    GPUJPEG_MARKER_APP1  = 0xe1,
    GPUJPEG_MARKER_APP2  = 0xe2,
    GPUJPEG_MARKER_APP3  = 0xe3,
    GPUJPEG_MARKER_APP4  = 0xe4,
    GPUJPEG_MARKER_APP5  = 0xe5,
    GPUJPEG_MARKER_APP6  = 0xe6,
    GPUJPEG_MARKER_APP7  = 0xe7,
    GPUJPEG_MARKER_APP8  = 0xe8,
    GPUJPEG_MARKER_APP9  = 0xe9,
    GPUJPEG_MARKER_APP10 = 0xea,
    GPUJPEG_MARKER_APP11 = 0xeb,
    GPUJPEG_MARKER_APP12 = 0xec,
    GPUJPEG_MARKER_APP13 = 0xed,
    GPUJPEG_MARKER_APP14 = 0xee,
    GPUJPEG_MARKER_APP15 = 0xef,
  
    GPUJPEG_MARKER_JPG0  = 0xf0,
    GPUJPEG_MARKER_JPG13 = 0xfd,
    GPUJPEG_MARKER_COM   = 0xfe,
  
    GPUJPEG_MARKER_TEM   = 0x01,
  
    GPUJPEG_MARKER_ERROR = 0x100
};

/**
 * Get marker name from code
 * 
 * @param code
 * @return marker name
 */
static const char* 
gpujpeg_marker_name(enum gpujpeg_marker_code code)
{
    switch (code) {
        case GPUJPEG_MARKER_SOF0: return "SOF0";
        case GPUJPEG_MARKER_SOF1: return "SOF1";
        case GPUJPEG_MARKER_SOF2: return "SOF2";
        case GPUJPEG_MARKER_SOF3: return "SOF3";
        case GPUJPEG_MARKER_SOF5: return "SOF5";
        case GPUJPEG_MARKER_SOF6: return "SOF6";
        case GPUJPEG_MARKER_SOF7: return "SOF7";
        case GPUJPEG_MARKER_JPG: return "JPG";
        case GPUJPEG_MARKER_SOF9: return "SOF9";
        case GPUJPEG_MARKER_SOF10: return "SOF10";
        case GPUJPEG_MARKER_SOF11: return "SOF11";
        case GPUJPEG_MARKER_SOF13: return "SOF13";
        case GPUJPEG_MARKER_SOF14: return "SOF14";
        case GPUJPEG_MARKER_SOF15: return "SOF15";
        case GPUJPEG_MARKER_DHT: return "DHT";
        case GPUJPEG_MARKER_DAC: return "DAC";
        case GPUJPEG_MARKER_RST0: return "RST0";
        case GPUJPEG_MARKER_RST1: return "RST1";
        case GPUJPEG_MARKER_RST2: return "RST2";
        case GPUJPEG_MARKER_RST3: return "RST3";
        case GPUJPEG_MARKER_RST4: return "RST4";
        case GPUJPEG_MARKER_RST5: return "RST5";
        case GPUJPEG_MARKER_RST6: return "RST6";
        case GPUJPEG_MARKER_RST7: return "RST7";
        case GPUJPEG_MARKER_SOI: return "SOI";
        case GPUJPEG_MARKER_EOI: return "EOI";
        case GPUJPEG_MARKER_SOS: return "SOS";
        case GPUJPEG_MARKER_DQT: return "DQT";
        case GPUJPEG_MARKER_DNL: return "DNL";
        case GPUJPEG_MARKER_DRI: return "DRI";
        case GPUJPEG_MARKER_DHP: return "DHP";
        case GPUJPEG_MARKER_EXP: return "EXP";
        case GPUJPEG_MARKER_APP0: return "APP0";
        case GPUJPEG_MARKER_APP1: return "APP1";
        case GPUJPEG_MARKER_APP2: return "APP2";
        case GPUJPEG_MARKER_APP3: return "APP3";
        case GPUJPEG_MARKER_APP4: return "APP4";
        case GPUJPEG_MARKER_APP5: return "APP5";
        case GPUJPEG_MARKER_APP6: return "APP6";
        case GPUJPEG_MARKER_APP7: return "APP7";
        case GPUJPEG_MARKER_APP8: return "APP8";
        case GPUJPEG_MARKER_APP9: return "APP9";
        case GPUJPEG_MARKER_APP10: return "APP10";
        case GPUJPEG_MARKER_APP11: return "APP11";
        case GPUJPEG_MARKER_APP12: return "APP12";
        case GPUJPEG_MARKER_APP13: return "APP13";
        case GPUJPEG_MARKER_APP14: return "APP14";
        case GPUJPEG_MARKER_APP15: return "APP15";
        case GPUJPEG_MARKER_JPG0: return "JPG0";
        case GPUJPEG_MARKER_JPG13: return "JPG13";
        case GPUJPEG_MARKER_COM: return "COM";
        case GPUJPEG_MARKER_TEM: return "TEM";
        case GPUJPEG_MARKER_ERROR: return "ERROR";
        default: 
        {
            static char buffer[255];
            sprintf(buffer, "Unknown (0x%X)", code);
            return buffer;
        }
    }
}

#endif // GPUJPEG_TYPE_H
