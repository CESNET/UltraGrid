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

#ifndef JPEG_TABLE
#define JPEG_TABLE

#include "jpeg_type.h"

/** JPEG natural order from zigzag order */
static const int jpeg_order_natural[64] = {
     0,  1,  8, 16,  9,  2,  3, 10,
    17, 24, 32, 25, 18, 11,  4,  5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13,  6,  7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63
};

/** JPEG quantization table structure */
struct jpeg_table_quantization
{
    // Quantization raw table
    uint8_t table_raw[64];
    // Quantization forward/inverse table
    uint16_t table[64];
    // Quantization forward/inverse table in device memory
    uint16_t* d_table;
};

/** JPEG table for huffman encoding */
struct jpeg_table_huffman_encoder {
    // Code for each symbol 
    unsigned int code[256];	
    // Length of code for each symbol 
	char size[256];
    // If no code has been allocated for a symbol S, size[S] is 0 

	// These two fields directly represent the contents of a JPEG DHT marker
    // bits[k] = # of symbols with codes of length k bits; bits[0] is unused
    unsigned char bits[17];
    // The symbols, in order of incr code length
    unsigned char huffval[256];
};

/** JPEG table for huffman decoding */
struct jpeg_table_huffman_decoder {
    // Smallest code of length k
    int mincode[17]; 
    // Largest code of length k (-1 if none) 
	int maxcode[18];
    // Huffval[] index of 1st symbol of length k
	int valptr[17];
    // # bits, or 0 if too long
    int look_nbits[256];
    // Symbol, or unused
	unsigned char look_sym[256];
    
    // These two fields directly represent the contents of a JPEG DHT marker
    // bits[k] = # of symbols with codes of 
	unsigned char bits[17];
    // The symbols, in order of incr code length 
	unsigned char huffval[256];
};

/**
 * Init JPEG quantization table for encoder
 * 
 * @param table  Table structure
 * @param type  Type of component for table
 * @param quality  Quality (0-100)
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_table_quantization_encoder_init(struct jpeg_table_quantization* table, enum jpeg_component_type type, int quality);

/**
 * Init JPEG quantization table for decoder
 * 
 * @param table  Table structure
 * @param type  Type of component for table
 * @param quality  Quality (0-100)
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_table_quantization_decoder_init(struct jpeg_table_quantization* table, enum jpeg_component_type type, int quality);

/**
 * Compute JPEG quantization table for decoder
 * 
 * @param table  Table structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_table_quantization_decoder_compute(struct jpeg_table_quantization* table);

/**
 * Print JPEG quantization table
 * 
 * @param table  Table structure
 * @return void
 */
void
jpeg_table_quantization_print(struct jpeg_table_quantization* table);

/**
 * Initialize encoder huffman DC and AC table for component type
 * 
 * @param table  Table structure
 * @param d_table  Table structure in device memory
 * @param comp_type  Component type (luminance/chrominance)
 * @param huff_type  Huffman type (DC/AC)
 * @return void
 */
int
jpeg_table_huffman_encoder_init(struct jpeg_table_huffman_encoder* table, struct jpeg_table_huffman_encoder* d_table, enum jpeg_component_type comp_type, enum jpeg_huffman_type huff_type);

/**
 * Initialize decoder huffman DC and AC table for component type. It copies bit and values arrays to table and call compute routine.
 * 
 * @param table  Table structure
 * @param d_table  Table structure in device memory
 * @param comp_type  Component type (luminance/chrominance)
 * @param huff_type  Huffman type (DC/AC)
 * @return void
 */
int
jpeg_table_huffman_decoder_init(struct jpeg_table_huffman_decoder* table, struct jpeg_table_huffman_decoder* d_table, enum jpeg_component_type comp_type, enum jpeg_huffman_type huff_type);

/** 
 * Compute decoder huffman table from bits and values arrays (that are already set in table)
 * 
 * @param table
 * @param d_table
 * @return void
 */
void
jpeg_table_huffman_decoder_compute(struct jpeg_table_huffman_decoder* table, struct jpeg_table_huffman_decoder* d_table);

#endif // JPEG_TABLE