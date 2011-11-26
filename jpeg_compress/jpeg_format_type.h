/**
 * Copyright (c) 2011, Martin Srom
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
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
 
#ifndef JPEG_FORMAT_TYPE
#define JPEG_FORMAT_TYPE

#include <stdio.h>

/** JPEG marker codes */
enum jpeg_marker_code {		
    JPEG_MARKER_SOF0  = 0xc0,
    JPEG_MARKER_SOF1  = 0xc1,
    JPEG_MARKER_SOF2  = 0xc2,
    JPEG_MARKER_SOF3  = 0xc3,
  
    JPEG_MARKER_SOF5  = 0xc5,
    JPEG_MARKER_SOF6  = 0xc6,
    JPEG_MARKER_SOF7  = 0xc7,
  
    JPEG_MARKER_JPG   = 0xc8,
    JPEG_MARKER_SOF9  = 0xc9,
    JPEG_MARKER_SOF10 = 0xca,
    JPEG_MARKER_SOF11 = 0xcb,
  
    JPEG_MARKER_SOF13 = 0xcd,
    JPEG_MARKER_SOF14 = 0xce,
    JPEG_MARKER_SOF15 = 0xcf,
  
    JPEG_MARKER_DHT   = 0xc4,
  
    JPEG_MARKER_DAC   = 0xcc,
  
    JPEG_MARKER_RST0  = 0xd0,
    JPEG_MARKER_RST1  = 0xd1,
    JPEG_MARKER_RST2  = 0xd2,
    JPEG_MARKER_RST3  = 0xd3,
    JPEG_MARKER_RST4  = 0xd4,
    JPEG_MARKER_RST5  = 0xd5,
    JPEG_MARKER_RST6  = 0xd6,
    JPEG_MARKER_RST7  = 0xd7,
  
    JPEG_MARKER_SOI   = 0xd8,
    JPEG_MARKER_EOI   = 0xd9,
    JPEG_MARKER_SOS   = 0xda,
    JPEG_MARKER_DQT   = 0xdb,
    JPEG_MARKER_DNL   = 0xdc,
    JPEG_MARKER_DRI   = 0xdd,
    JPEG_MARKER_DHP   = 0xde,
    JPEG_MARKER_EXP   = 0xdf,
  
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
  
    JPEG_MARKER_TEM   = 0x01,
  
    JPEG_MARKER_ERROR = 0x100
};

static const char* 
jpeg_marker_name(enum jpeg_marker_code code)
{
    switch (code) {
        case JPEG_MARKER_SOF0: return "SOF0";
        case JPEG_MARKER_SOF1: return "SOF1";
        case JPEG_MARKER_SOF2: return "SOF2";
        case JPEG_MARKER_SOF3: return "SOF3";
        case JPEG_MARKER_SOF5: return "SOF5";
        case JPEG_MARKER_SOF6: return "SOF6";
        case JPEG_MARKER_SOF7: return "SOF7";
        case JPEG_MARKER_JPG: return "JPG";
        case JPEG_MARKER_SOF9: return "SOF9";
        case JPEG_MARKER_SOF10: return "SOF10";
        case JPEG_MARKER_SOF11: return "SOF11";
        case JPEG_MARKER_SOF13: return "SOF13";
        case JPEG_MARKER_SOF14: return "SOF14";
        case JPEG_MARKER_SOF15: return "SOF15";
        case JPEG_MARKER_DHT: return "DHT";
        case JPEG_MARKER_DAC: return "DAC";
        case JPEG_MARKER_RST0: return "RST0";
        case JPEG_MARKER_RST1: return "RST1";
        case JPEG_MARKER_RST2: return "RST2";
        case JPEG_MARKER_RST3: return "RST3";
        case JPEG_MARKER_RST4: return "RST4";
        case JPEG_MARKER_RST5: return "RST5";
        case JPEG_MARKER_RST6: return "RST6";
        case JPEG_MARKER_RST7: return "RST7";
        case JPEG_MARKER_SOI: return "SOI";
        case JPEG_MARKER_EOI: return "EOI";
        case JPEG_MARKER_SOS: return "SOS";
        case JPEG_MARKER_DQT: return "DQT";
        case JPEG_MARKER_DNL: return "DNL";
        case JPEG_MARKER_DRI: return "DRI";
        case JPEG_MARKER_DHP: return "DHP";
        case JPEG_MARKER_EXP: return "EXP";
        case JPEG_MARKER_APP0: return "APP0";
        case JPEG_MARKER_APP1: return "APP1";
        case JPEG_MARKER_APP2: return "APP2";
        case JPEG_MARKER_APP3: return "APP3";
        case JPEG_MARKER_APP4: return "APP4";
        case JPEG_MARKER_APP5: return "APP5";
        case JPEG_MARKER_APP6: return "APP6";
        case JPEG_MARKER_APP7: return "APP7";
        case JPEG_MARKER_APP8: return "APP8";
        case JPEG_MARKER_APP9: return "APP9";
        case JPEG_MARKER_APP10: return "APP10";
        case JPEG_MARKER_APP11: return "APP11";
        case JPEG_MARKER_APP12: return "APP12";
        case JPEG_MARKER_APP13: return "APP13";
        case JPEG_MARKER_APP14: return "APP14";
        case JPEG_MARKER_APP15: return "APP15";
        case JPEG_MARKER_JPG0: return "JPG0";
        case JPEG_MARKER_JPG13: return "JPG13";
        case JPEG_MARKER_COM: return "COM";
        case JPEG_MARKER_TEM: return "TEM";
        case JPEG_MARKER_ERROR: return "ERROR";
        default: 
        {
            static char buffer[255];
            sprintf(buffer, "Unknown (0x%X)", code);
            return buffer;
        }
    }
}

#endif // JPEG_FORMAT_TYPE
