/**
 * @file   utils/jpeg_reader.c
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
/**
 * @file
 * Most of the code is taken from GPUJPEG reader.
 *
 * @todo
 * Fix correct quantization table mapping handling - eg. libavcodec uses only
 * one table (DQT) and maps in SOF0 both Y and CbCr to that. Since GPUJPEG
 * used 2 (hardcoded), the warning is only commented out. In jpeg_get_rtp_hdr_data()
 * the second (missing) table is simply taken from 1st (that is correct - but needs
 * to be checked from SOF0 mapping).
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // defined HAVE_CONFIG_H

#include "utils/jpeg_reader.h"

#include "debug.h"

u_char lum_dc_codelens[] = {
        0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
};

u_char lum_dc_symbols[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
};

u_char lum_ac_codelens[] = {
        0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d,
};

u_char lum_ac_symbols[] = {
        0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12,
        0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61, 0x07,
        0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08,
        0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52, 0xd1, 0xf0,
        0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16,
        0x17, 0x18, 0x19, 0x1a, 0x25, 0x26, 0x27, 0x28,
        0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39,
        0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49,
        0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59,
        0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69,
        0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79,
        0x7a, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89,
        0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98,
        0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7,
        0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
        0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5,
        0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3, 0xd4,
        0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2,
        0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9, 0xea,
        0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
};

u_char chm_dc_codelens[] = {
        0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
};

u_char chm_dc_symbols[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
};

u_char chm_ac_codelens[] = {
        0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77,
};

u_char chm_ac_symbols[] = {
        0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21,
        0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61, 0x71,
        0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91,
        0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33, 0x52, 0xf0,
        0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34,
        0xe1, 0x25, 0xf1, 0x17, 0x18, 0x19, 0x1a, 0x26,
        0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38,
        0x39, 0x3a, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48,
        0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58,
        0x59, 0x5a, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68,
        0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78,
        0x79, 0x7a, 0x82, 0x83, 0x84, 0x85, 0x86, 0x87,
        0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96,
        0x97, 0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5,
        0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
        0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3,
        0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2,
        0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda,
        0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8, 0xe9,
        0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8,
        0xf9, 0xfa,
};

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


static const char* jpeg_marker_name(enum jpeg_marker_code code)
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

#define read_byte(image) \
        (uint8_t)(*(image)++)

#define read_2byte(image) \
        (uint16_t)(((*(image)) << 8) + (*((image) + 1))); \
        image += 2;

static int read_marker(uint8_t** image)
{
        uint8_t byte = read_byte(*image);
        if( byte != 0xFF ) {
                return -1;
        }
        int marker = read_byte(*image);
        return marker;
}

static void skip_marker_content(uint8_t** image)
{
        int length = (int)read_2byte(*image);

        *image += length - 2;
}

static int read_sof0(struct jpeg_info *param, uint8_t** image)
{
        int length = (int)read_2byte(*image);
        if ( length < 6 ) {
                log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] SOF0 marker length should be greater than 6 but %d was presented!\n", length);
                return -1;
        }
        length -= 2;

        int precision = (int)read_byte(*image);
        if ( precision != 8 ) {
                log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] SOF0 marker precision should be 8 but %d was presented!\n", precision);
                return -1;
        }

        param->height = (int)read_2byte(*image);
        param->width = (int)read_2byte(*image);
        param->comp_count = (int)read_byte(*image);
        length -= 6;

        for ( int comp = 0; comp < param->comp_count; comp++ ) {
                int index = (int)read_byte(*image);
                if ( index != (comp + 1) ) {
                        log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] SOF0 marker component %d id should be %d but %d was presented!\n", comp, comp + 1, index);
                        return -1;
                }

                int sampling = (int)read_byte(*image);
                param->sampling_factor_h[comp] = (sampling >> 4) & 15;
                param->sampling_factor_v[comp] = (sampling) & 15;

                int table_index = (int)read_byte(*image);
                if ( comp == 0 && table_index != 0 ) {
                        log_msg(LOG_LEVEL_VERBOSE, "[JPEG] [Error] SOF0 marker component Y should have quantization table index 0 but %d was presented!\n", table_index);
                        return -1;
                }
                if ( (comp == 1 || comp == 2) && table_index != 1 ) {
                        log_msg(LOG_LEVEL_VERBOSE, "[JPEG] [Warning] SOF0 marker component Cb or Cr should have quantization table index 1 but %d was presented!\n", table_index);
                }
                length -= 3;
        }

        // Check length
        if ( length > 0 ) {
                log_msg(LOG_LEVEL_WARNING, "[JPEG] [Warning] SOF0 marker contains %d more bytes than needed!\n", length);
                *image += length;
        }

        return 0;
}

static int read_dri(struct jpeg_info *param, uint8_t** image)
{
        int length = (int)read_2byte(*image);
        if ( length != 4 ) {
                log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] DRI marker length should be 4 but %d was presented!\n", length);
                return -1;
        }

        param->restart_interval = read_2byte(*image);
        return 0;
}

static int read_dht(struct jpeg_info *param, uint8_t** image)
{
	int length = (int)read_2byte(*image);
	length -= 2;

	while (length > 0) {
		int index = read_byte(*image);
		uint8_t *table;
		switch(index) {
			case 0:
				table = param->huff_lum_dc;
				break;
			case 16:
				table = param->huff_lum_ac;
				break;
			case 1:
				table = param->huff_chm_dc;
				break;
			case 17:
				table = param->huff_chm_ac;
				break;
			default:
				log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] DHT marker index should be 0, 1, 16 or 17 but %d was presented!\n", index);
				return -1;
		}
		length -= 1;

		// Read in bits[]
		int count = 0;
		for ( int i = 0; i < 16; i++ ) {
			table[i] = read_byte(*image);
			count += table[i];
			if ( length > 0 ) {
				length--;
			} else {
				log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] DHT marker unexpected end when reading bit counts!\n");
				return -1;
			}
		}

		// Read in huffval
		for ( int i = 0; i < count; i++ ){
			table[16 + i] = read_byte(*image);
			if ( length > 0 ) {
				length--;
			} else {
				log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] DHT marker unexpected end when reading huffman values!\n");
				return -1;
			}
		}
	}
	return 0;
}


static int read_dqt(struct jpeg_info *param, uint8_t** image)
{
        int length = (int)read_2byte(*image);
        length -= 2;

        if ( (length % 65) != 0 ) {
                log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] DQT marker length should be 65 but %d was presented!\n", length);
                return -1;
        }

        for (;length > 0; length-=65) {
                int index = read_byte(*image);
                uint8_t **table;
                if( index == 0 || index == 1 ) {
                        table = &param->quantization_tables[index];
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] DQT marker index should be 0 or 1 but %d was presented!\n", index);
                        return -1;
                }
                *table = *image;

                *image += 64; // skip table content
        }
        return 0;
}

static int read_com(struct jpeg_info *param, uint8_t** image) {
        int length = (int)read_2byte(*image);
        length -= 2;

        for (int i = 0; i < length; ++i) {
                param->com[i] = read_byte(*image);
        }

        // add terminating zero (if not present)
        param->com[length] = '\0';

        return 0;
}

static int read_sos(struct jpeg_info *param, uint8_t** image)
{
        int length = (int)read_2byte(*image);
        length -= 2;

        if (length == 0) {
                log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] Wrong SOS length: %d\n", length);
                return -1;
        }
        int comp_count = (int)read_byte(*image);
        if ( comp_count != param->comp_count ) {
                param->interleaved = false;
        }
        if (length < 1 + comp_count * 2 + 3) {
                log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] Wrong SOS length: %d\n", length);
                return -1;
        }

        // Collect the component-spec parameters
        for ( int comp = 0; comp < comp_count; comp++ )
        {
                int index = (int)read_byte(*image);
                int table = (int)read_byte(*image);
                int table_dc = (table >> 4) & 15;
                int table_ac = table & 15;

                if ( index == 1 && (table_ac != 0 || table_dc != 0) ) {
                        log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] SOS marker for Y should have huffman tables 0,0 but %d,%d was presented!\n", table_dc, table_ac);
                        return -1;
                }
                if ( (index == 2 || index == 3) && (table_ac != 1 || table_dc != 1) ) {
                        log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] SOS marker for Cb or Cr should have huffman tables 1,1 but %d,%d was presented!\n", table_dc, table_ac);
                        return -1;
                }
        }

        // Collect the additional scan parameters Ss, Se, Ah/Al.
        int Ss __attribute__((unused)) = (int)read_byte(*image);
        int Se __attribute__((unused)) = (int)read_byte(*image);
        int Ax __attribute__((unused)) = (int)read_byte(*image);
        int Ah __attribute__((unused)) = (Ax >> 4) & 15;
        int Al __attribute__((unused)) = (Ax) & 15;

        // Get scan structure
        param->data = *image;

        return 0;
}

static int read_adobe_app14(struct jpeg_info *param, uint8_t** image)
{
        int length = read_2byte(*image);
        if (length != 14) { // not an Adobe APP14 marker
                return 0;
        }

        char adobe[6] = "";
        adobe[0] = read_byte(*image);
        adobe[1] = read_byte(*image);
        adobe[2] = read_byte(*image);
        adobe[3] = read_byte(*image);
        adobe[4] = read_byte(*image);
        if (strcmp(adobe, "Adobe") != 0) { // not an Adobe APP14 marker
                return 0;
        }

        int version __attribute__((unused)) = read_2byte(*image);
        int flags0 __attribute__((unused)) = read_2byte(*image);
        int flags1 __attribute__((unused)) = read_2byte(*image);
        int color_transform = read_byte(*image);

	if (color_transform == 0) {
		param->color_spec = JPEG_COLOR_SPEC_CMYK; // or RGB - will be determined later
	} else if (color_transform == 1) {
		param->color_spec = JPEG_COLOR_SPEC_YCBCR;
	} else if (color_transform == 2) {
		param->color_spec = JPEG_COLOR_SPEC_YCCK;
	} else {
		log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] Unsupported color transformation value '%d' was presented in APP14 marker!\n", color_transform);
		return -1;
	}

        return 0;
}

/**
 * This function reads values from JPEG file.
 *
 * @retval  0 ok
 * @retval -1 general error
 * @retval -2 incomplete file (required markers missing)
 */
int jpeg_read_info(uint8_t *image, int len, struct jpeg_info *info)
{
        uint8_t *image_start = image;

        // Check first SOI marker
        int marker_soi = read_marker(&image);
        if (marker_soi != JPEG_MARKER_SOI) {
                log_msg(LOG_LEVEL_ERROR, "[JPEG] [Error] JPEG data should begin with SOI marker, but marker %s was found!\n", jpeg_marker_name((enum jpeg_marker_code)marker_soi));
                return -1;
        }

        info->huff_lum_dc[0] = 255;
        info->huff_lum_ac[0] = 255;
        info->huff_chm_dc[0] = 255;
        info->huff_chm_ac[0] = 255;
        info->comp_count = 0;
        info->interleaved = true;
        info->restart_interval = 0; // if DRI is not present
        info->com[0] = '\0'; // if COM is not present
        info->color_spec = JPEG_COLOR_SPEC_YCBCR; // default
        bool marker_present[255] = { 0 };

        // currently reading up to SOS marker gives us all needed data
        while (!marker_present[JPEG_MARKER_EOI] && !marker_present[JPEG_MARKER_SOS]
                        && image < image_start + len) {
                // Read marker
                int marker = read_marker(&image);
                if ( marker == 0) { // not a marker (byte stuffing)
                        continue;
                }
                if (marker == -1) {
                        log_msg(LOG_LEVEL_WARNING, "Unknown marker!\n");
                        continue;
                }
                if (marker == 0xff) { // first byte might have been padding and the second byte marker start
                        image--;
                        continue;
                }

                marker_present[marker] = true;
                int rc;

                // Read more info according to the marker
                switch (marker)
                {
                        case JPEG_MARKER_SOF0: // Baseline
                                if ((rc = read_sof0(info, &image)) != 0) {
                                        log_msg(LOG_LEVEL_ERROR, "Error reading SOF0!\n");
                                        return rc;
                                }
                                break;

                        case JPEG_MARKER_DHT:
                                if ((rc = read_dht(info, &image)) != 0) {
                                        log_msg(LOG_LEVEL_ERROR, "Error reading DQT!\n");
                                        return rc;
                                }
                                break;

                        case JPEG_MARKER_DQT:
                                if ((rc = read_dqt(info, &image)) != 0) {
                                        log_msg(LOG_LEVEL_ERROR, "Error reading DQT!\n");
                                        return rc;
                                }
                                break;

                        case JPEG_MARKER_DRI:
                                if ((rc = read_dri(info, &image)) != 0) {
                                        log_msg(LOG_LEVEL_ERROR, "Error reading DRI!\n");
                                        return rc;
                                }
                                break;

                        case JPEG_MARKER_COM:
                                if ((rc = read_com(info, &image)) != 0) {
                                        log_msg(LOG_LEVEL_ERROR, "Error reading COM!\n");
                                        return rc;
                                }
                                break;

                        case JPEG_MARKER_APP14:
                                if ((rc = read_adobe_app14(info, &image)) != 0) {
                                        log_msg(LOG_LEVEL_ERROR, "Error reading APP14!\n");
                                        return rc;
                                }
                                break;

                        case JPEG_MARKER_SOS:
                                if ((rc = read_sos(info, &image)) != 0) {
                                        log_msg(LOG_LEVEL_ERROR, "Error reading SOS!\n");
                                        return rc;
                                }
                                break;

                        case JPEG_MARKER_EOI:
                                break;

                        // included because these fixed-sized markers do not contain lenght field
                        // thus skipping in default branch will misbehave
                        case JPEG_MARKER_RST0:
                        case JPEG_MARKER_RST1:
                        case JPEG_MARKER_RST2:
                        case JPEG_MARKER_RST3:
                        case JPEG_MARKER_RST4:
                        case JPEG_MARKER_RST5:
                        case JPEG_MARKER_RST6:
                        case JPEG_MARKER_RST7:
                                break;

                        default:
                                skip_marker_content(&image);
                                break;
                }
        }

        enum jpeg_marker_code req_markers[] = { JPEG_MARKER_SOF0, JPEG_MARKER_DQT, JPEG_MARKER_SOS };
        for (unsigned int i = 0; i < sizeof req_markers / sizeof req_markers[0]; ++i) {
                if (!marker_present[req_markers[i]]) {
                        log_msg(LOG_LEVEL_ERROR, "Marker \"%s\" missing in JPEG file!\n",
                                        jpeg_marker_name(req_markers[i]));
                        if (req_markers[i] == JPEG_MARKER_SOF0) {
                                log_msg(LOG_LEVEL_INFO, "Perhaps unsupported JPEG type.\n");
                        }
                        return -2;
                }
        }

        // APP14 doesn't distinguis between CMYK and RGB thus we deduce from comp count
        if (info->color_spec == JPEG_COLOR_SPEC_CMYK && info->comp_count == 3) {
                info->color_spec = JPEG_COLOR_SPEC_RGB;
        }

        return 0;
}

static bool check_huff_tables(struct jpeg_info *param) {
        // tables defined implicitly
        if (param->huff_lum_dc[0] == 255 || param->huff_lum_ac[0] == 255 ||
                        param->huff_chm_dc[0] == 255 || param->huff_chm_ac[0] == 255) {
                return true;
        }

	for (unsigned int i = 0; i < sizeof lum_dc_codelens; ++i) {
		if (param->huff_lum_dc[i] != lum_dc_codelens[i])
			return false;
	}
	for (unsigned int i = 0; i < sizeof lum_dc_symbols; ++i) {
		if (param->huff_lum_dc[16 + i] != lum_dc_symbols[i])
			return false;
	}

	for (unsigned int i = 0; i < sizeof lum_ac_codelens; ++i) {
		if (param->huff_lum_ac[i] != lum_ac_codelens[i])
			return false;
	}
	for (unsigned int i = 0; i < sizeof lum_ac_symbols; ++i) {
		if (param->huff_lum_ac[16 + i] != lum_ac_symbols[i])
			return false;
	}

	for (unsigned int i = 0; i < sizeof chm_dc_codelens; ++i) {
		if (param->huff_chm_dc[i] != chm_dc_codelens[i])
			return false;
	}
	for (unsigned int i = 0; i < sizeof chm_dc_symbols; ++i) {
		if (param->huff_chm_dc[16 + i] != chm_dc_symbols[i])
			return false;
	}

	for (unsigned int i = 0; i < sizeof chm_ac_codelens; ++i) {
		if (param->huff_chm_ac[i] != chm_ac_codelens[i])
			return false;
	}
	for (unsigned int i = 0; i < sizeof chm_ac_symbols; ++i) {
		if (param->huff_chm_ac[16 + i] != chm_ac_symbols[i])
			return false;
	}

        return true;
}

static bool arr_eq3(const int *a1, const int *a2) { for (int i = 0; i < 3; ++i) if (a1[i] != a2[i]) return false; return true; }

bool jpeg_get_rtp_hdr_data(uint8_t *jpeg_data, int len, struct jpeg_rtp_data *hdr_data)
{
        struct jpeg_info i;
        int rc = jpeg_read_info(jpeg_data, len, &i);

	if (rc != 0) {
                log_msg(LOG_LEVEL_ERROR, "Cannot parse JPEG!\n");
                return false;
	}
        if (i.comp_count != 3) {
                log_msg(LOG_LEVEL_ERROR, "Unsupported JPEG component count: %d!\n", i.comp_count);
		return false;
        }
        if (i.color_spec != JPEG_COLOR_SPEC_YCBCR) {
                log_msg(LOG_LEVEL_ERROR, "Unsupported JPEG color space (only YCbCr supported!\n");
		return false;
        }
        if (!i.interleaved) {
                if (strstr(i.com, "GPUJPEG")) {
                        log_msg(LOG_LEVEL_ERROR, "Non-interleaved JPEG detected, use \"-c GPUJPEG:interleaved\"!\n");
                } else {
                        log_msg(LOG_LEVEL_ERROR, "Non-interleaved JPEG detected!\n");
                }
		return false;
        }

        if (!check_huff_tables(&i)) {
                log_msg(LOG_LEVEL_ERROR, "Non-default Huffman tables detected!\n");
                return false;
        }

        hdr_data->width = i.width;
        hdr_data->height = i.height;
        hdr_data->restart_interval = i.restart_interval;
        hdr_data->data = i.data;
        const int a111[] = { 1, 1, 1 },
            a211[] = { 2, 1, 1 },
            a222[] = { 2, 2, 2 };
        if (arr_eq3(i.sampling_factor_v, a111) && arr_eq3(i.sampling_factor_h, a211)) { // 4:2:2
                hdr_data->type = 0;
        } else if (arr_eq3(i.sampling_factor_v, a211) && arr_eq3(i.sampling_factor_h, a211)) { // 4:2:0
                hdr_data->type = 1;
        } else {
                log_msg(LOG_LEVEL_ERROR, "Unsupported JPEG type - sampling horizontal: %d, %d, %d; vertical: %d, %d, %d\n", i.sampling_factor_h[0], i.sampling_factor_h[1], i.sampling_factor_h[2], i.sampling_factor_v[0], i.sampling_factor_v[1], i.sampling_factor_v[2]);
                if (arr_eq3(i.sampling_factor_v, a222) && arr_eq3(i.sampling_factor_h, a211)) {
                        fprintf(stderr, "Try \"-c libavcodec:subsampling=420\"\n");
                }
                return false;
        }
        if (i.restart_interval != 0) {
                hdr_data->type |= 0x40;
        }

        hdr_data->q = 255;
        // we try deduce quality from COM marker
        const char *q_str = "quality = ";
        if (strstr(i.com, q_str) && isdigit(*(strstr(i.com, q_str) + strlen(q_str)))) {
                hdr_data->q = atoi(strstr(i.com, q_str) + strlen(q_str));
                assert(hdr_data->q <= 127);
	}
        if (hdr_data->q == 255) {
		hdr_data->quantization_tables[0] = i.quantization_tables[0];
                if (i.quantization_tables[1] != NULL) {
                        hdr_data->quantization_tables[1] = i.quantization_tables[1];
                } else { // libavcodec uses one table only
                        hdr_data->quantization_tables[1] = hdr_data->quantization_tables[0];
                }
	}

	return true;
}

