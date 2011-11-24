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
 
#include "jpeg_table.h"
#include "jpeg_util.h"

/** Documented at declaration */
int
jpeg_table_quantization_encoder_init(struct jpeg_table_quantization* table, enum jpeg_component_type type, int quality)
{
    // Setup raw table
    nppiSetDefaultQuantTable(table->table_raw, (int)type);
    
    // Init raw table
    nppiQuantFwdRawTableInit_JPEG_8u(table->table_raw, quality);
    
    // Setup forward table by npp
    nppiQuantFwdTableInit_JPEG_8u16u(table->table_raw, table->table);
        
    // Copy tables to device memory
    if ( cudaSuccess != cudaMemcpy(table->d_table, table->table, 64 * sizeof(uint16_t), cudaMemcpyHostToDevice) )
        return -1;
    
    return 0;
}

/** Documented at declaration */
int
jpeg_table_quantization_decoder_init(struct jpeg_table_quantization* table, enum jpeg_component_type type, int quality)
{
    // Setup raw table
    nppiSetDefaultQuantTable(table->table_raw, (int)type);
    
    // Init raw table
    nppiQuantFwdRawTableInit_JPEG_8u(table->table_raw, quality);
    
    // Setup inverse table by npp
    nppiQuantInvTableInit_JPEG_8u16u(table->table_raw, table->table);
    
    // Copy tables to device memory
    if ( cudaSuccess != cudaMemcpy(table->d_table, table->table, 64 * sizeof(uint16_t), cudaMemcpyHostToDevice) )
        return -1;
        
    return 0;
}

int
jpeg_table_quantization_decoder_compute(struct jpeg_table_quantization* table)
{
    // Setup inverse table by npp
    nppiQuantInvTableInit_JPEG_8u16u(table->table_raw, table->table);
    
    // Copy tables to device memory
    if ( cudaSuccess != cudaMemcpy(table->d_table, table->table, 64 * sizeof(uint16_t), cudaMemcpyHostToDevice) )
        return -1;
        
    return 0;
}

/** Documented at declaration */
void
jpeg_table_quantization_print(struct jpeg_table_quantization* table)
{
    puts("Raw Table (with quality):");
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%4u", table->table_raw[i * 8 + j]);
        }
        puts("");
    }
    
    puts("Forward/Inverse Table:");
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            printf("%6u", table->table[i * 8 + j]);
        }
        puts("");
    }
}

/** Huffman Table  DC for Y component */
static unsigned char jpeg_table_huffman_y_dc_bits[17] = {
    0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0 
};
static unsigned char jpeg_table_huffman_y_dc_value[] = { 
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
};
/** Huffman Table  DC for Cb or Cr component */
static unsigned char jpeg_table_huffman_cbcr_dc_bits[17] = { 
    0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0 
};
static unsigned char jpeg_table_huffman_cbcr_dc_value[] = { 
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 
};
/** Huffman Table  AC for Y component */
static unsigned char jpeg_table_huffman_y_ac_bits[17] = { 
    0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d 
};
static unsigned char jpeg_table_huffman_y_ac_value[] = { 
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
    0xf9, 0xfa 
};
/** Huffman Table AC for Cb or Cr component */
static unsigned char jpeg_table_huffman_cbcr_ac_bits[17] = { 
    0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77 
};
static unsigned char jpeg_table_huffman_cbcr_ac_value[] = { 
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
    0xf9, 0xfa 
};

/** 
 * Compute encoder huffman table from bits and values arrays (that are already set in table)
 * 
 * @param table  Table structure
 * @return void
 */
void
jpeg_table_huffman_encoder_compute(struct jpeg_table_huffman_encoder* table)
{
	char huffsize[257];
	unsigned int huffcode[257];

	// Figure C.1: make table of Huffman code length for each symbol
	// Note that this is in code-length order
	int p = 0;
	for ( int l = 1; l <= 16; l++ ) {
		for ( int i = 1; i <= (int) table->bits[l]; i++ )
			huffsize[p++] = (char) l;
	}
	huffsize[p] = 0;
	int lastp = p;

	// Figure C.2: generate the codes themselves
	// Note that this is in code-length order
	unsigned int code = 0;
	int si = huffsize[0];
	p = 0;
	while ( huffsize[p] ) {
		while ( ((int) huffsize[p]) == si ) {
			huffcode[p++] = code;
			code++;
		}
		code <<= 1;
		si++;
	}

	// Figure C.3: generate encoding tables
	// These are code and size indexed by symbol value

	// Set any codeless symbols to have code length 0;
	// this allows EmitBits to detect any attempt to emit such symbols.
	memset(table->size, 0, sizeof(table->size));

	for (p = 0; p < lastp; p++) {
		table->code[table->huffval[p]] = huffcode[p];
		table->size[table->huffval[p]] = huffsize[p];
	}
}

/** Documented at declaration */
int
jpeg_table_huffman_encoder_init(struct jpeg_table_huffman_encoder* table, struct jpeg_table_huffman_encoder* d_table, enum jpeg_component_type comp_type, enum jpeg_huffman_type huff_type)
{
    assert(comp_type == JPEG_COMPONENT_LUMINANCE || comp_type == JPEG_COMPONENT_CHROMINANCE);
    assert(huff_type == JPEG_HUFFMAN_DC || huff_type == JPEG_HUFFMAN_AC);
    if ( comp_type == JPEG_COMPONENT_LUMINANCE ) {
        if ( huff_type == JPEG_HUFFMAN_DC ) {
            memcpy(table->bits, jpeg_table_huffman_y_dc_bits, sizeof(table->bits));
            memcpy(table->huffval, jpeg_table_huffman_y_dc_value, sizeof(table->huffval));
        } else {
            memcpy(table->bits, jpeg_table_huffman_y_ac_bits, sizeof(table->bits));
            memcpy(table->huffval, jpeg_table_huffman_y_ac_value, sizeof(table->huffval));
        }        
    } else if ( comp_type == JPEG_COMPONENT_CHROMINANCE ) {
        if ( huff_type == JPEG_HUFFMAN_DC ) {
            memcpy(table->bits, jpeg_table_huffman_cbcr_dc_bits, sizeof(table->bits));
            memcpy(table->huffval, jpeg_table_huffman_cbcr_dc_value, sizeof(table->huffval));
        } else {
            memcpy(table->bits, jpeg_table_huffman_cbcr_ac_bits, sizeof(table->bits));
            memcpy(table->huffval, jpeg_table_huffman_cbcr_ac_value, sizeof(table->huffval));
        }
    }
    jpeg_table_huffman_encoder_compute(table);
    
    // Copy table to device memory
    if ( cudaSuccess != cudaMemcpy(d_table, table, sizeof(struct jpeg_table_huffman_encoder), cudaMemcpyHostToDevice) )
        return -1;
        
    return 0;
}

/** Documented at declaration */
int
jpeg_table_huffman_decoder_init(struct jpeg_table_huffman_decoder* table, struct jpeg_table_huffman_decoder* d_table, enum jpeg_component_type comp_type, enum jpeg_huffman_type huff_type)
{
    assert(comp_type == JPEG_COMPONENT_LUMINANCE || comp_type == JPEG_COMPONENT_CHROMINANCE);
    assert(huff_type == JPEG_HUFFMAN_DC || huff_type == JPEG_HUFFMAN_AC);
    if ( comp_type == JPEG_COMPONENT_LUMINANCE ) {
        if ( huff_type == JPEG_HUFFMAN_DC ) {
            memcpy(table->bits, jpeg_table_huffman_y_dc_bits, sizeof(table->bits));
            memcpy(table->huffval, jpeg_table_huffman_y_dc_value, sizeof(table->huffval));
        } else {
            memcpy(table->bits, jpeg_table_huffman_y_ac_bits, sizeof(table->bits));
            memcpy(table->huffval, jpeg_table_huffman_y_ac_value, sizeof(table->huffval));
        }        
    } else if ( comp_type == JPEG_COMPONENT_CHROMINANCE ) {
        if ( huff_type == JPEG_HUFFMAN_DC ) {
            memcpy(table->bits, jpeg_table_huffman_cbcr_dc_bits, sizeof(table->bits));
            memcpy(table->huffval, jpeg_table_huffman_cbcr_dc_value, sizeof(table->huffval));
        } else {
            memcpy(table->bits, jpeg_table_huffman_cbcr_ac_bits, sizeof(table->bits));
            memcpy(table->huffval, jpeg_table_huffman_cbcr_ac_value, sizeof(table->huffval));
        }
    }
    jpeg_table_huffman_decoder_compute(table, d_table);
        
    return 0;
}

/** Documented at declaration */
void
jpeg_table_huffman_decoder_compute(struct jpeg_table_huffman_decoder* table, struct jpeg_table_huffman_decoder* d_table)
{
	// Figure C.1: make table of Huffman code length for each symbol
	// Note that this is in code-length order.
    char huffsize[257];
	int p = 0;
	for ( int l = 1; l <= 16; l++ ) {
		for ( int i = 1; i <= (int) table->bits[l]; i++ )
			huffsize[p++] = (char) l;
	}
	huffsize[p] = 0;

	// Figure C.2: generate the codes themselves
	// Note that this is in code-length order.
    unsigned int huffcode[257];
	unsigned int code = 0;
	int si = huffsize[0];
	p = 0;
	while ( huffsize[p] ) {
		while ( ((int) huffsize[p]) == si ) {
			huffcode[p++] = code;
			code++;
		}
		code <<= 1;
		si++;
	}

	// Figure F.15: generate decoding tables for bit-sequential decoding
	p = 0;
	for ( int l = 1; l <= 16; l++ ) {
		if ( table->bits[l] ) {
			table->valptr[l] = p; // huffval[] index of 1st symbol of code length l
			table->mincode[l] = huffcode[p]; // minimum code of length l
			p += table->bits[l];
			table->maxcode[l] = huffcode[p-1]; // maximum code of length l
		} else {
			table->maxcode[l] = -1;	// -1 if no codes of this length
		}
	}
    // Ensures jpeg_huff_decode terminates
	table->maxcode[17] = 0xFFFFFL;

	// Compute lookahead tables to speed up decoding.
    //First we set all the table entries to 0, indicating "too long";
    //then we iterate through the Huffman codes that are short enough and
    //fill in all the entries that correspond to bit sequences starting
    //with that code.
	memset(table->look_nbits, 0, sizeof(int) * 256);

	int HUFF_LOOKAHEAD = 8;
	p = 0;
	for ( int l = 1; l <= HUFF_LOOKAHEAD; l++ ) {
		for ( int i = 1; i <= (int) table->bits[l]; i++, p++ ) {
			// l = current code's length, 
			// p = its index in huffcode[] & huffval[]. Generate left-justified
			// code followed by all possible bit sequences
			int lookbits = huffcode[p] << (HUFF_LOOKAHEAD - l);
			for ( int ctr = 1 << (HUFF_LOOKAHEAD - l); ctr > 0; ctr-- ) 
			{
				table->look_nbits[lookbits] = l;
				table->look_sym[lookbits] = table->huffval[p];
				lookbits++;
			}
		}
	}
    
    // Copy table to device memory
    cudaMemcpy(d_table, table, sizeof(struct jpeg_table_huffman_decoder), cudaMemcpyHostToDevice);
}