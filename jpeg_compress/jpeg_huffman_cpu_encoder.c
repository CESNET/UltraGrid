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
 
#include "jpeg_huffman_cpu_encoder.h"
#include "jpeg_format_type.h"
#include "jpeg_util.h"

#ifdef _DEBUG
#define inline
#endif

/** Huffman encoder structure */
struct jpeg_huffman_cpu_encoder
{
    // Huffman table DC
    struct jpeg_table_huffman_encoder* table_dc;
    // Huffman table AC
    struct jpeg_table_huffman_encoder* table_ac;
    // The value (in 4 byte buffer) to be written out
    int put_value;
    // The size (in bits) to be written out
    int put_bits;
    // JPEG writer structure
    struct jpeg_writer* writer;
    // DC differentize for component
    int dc;
    // Restart interval
    int restart_interval;
    // Block count
    int block_count;
};

/**
 * Output bits to the file. Only the right 24 bits of put_buffer are used; 
 * the valid bits are left-justified in this part.  At most 16 bits can be 
 * passed to EmitBits in one call, and we never retain more than 7 bits 
 * in put_buffer between calls, so 24 bits are sufficient.
 * 
 * @param coder  Huffman coder structure
 * @param code  Huffman code
 * @param size  Size in bits of the Huffman code
 * @return void
 */
inline int
jpeg_huffman_cpu_encoder_emit_bits(struct jpeg_huffman_cpu_encoder* coder, unsigned int code, int size)
{
    // This routine is heavily used, so it's worth coding tightly
    int put_buffer = (int)code;
    int put_bits = coder->put_bits;
    // If size is 0, caller used an invalid Huffman table entry
    if ( size == 0 )
        return -1;
    // Mask off any extra bits in code
    put_buffer &= (((int)1) << size) - 1; 
    // New number of bits in buffer
    put_bits += size;                    
    // Align incoming bits
    put_buffer <<= 24 - put_bits;        
    // And merge with old buffer contents
    put_buffer |= coder->put_value;    
    // If there are more than 8 bits, write it out
    unsigned char uc;
    while ( put_bits >= 8 ) {
        // Write one byte out
        uc = (unsigned char) ((put_buffer >> 16) & 0xFF);
        jpeg_writer_emit_byte(coder->writer, uc);
        // If need to stuff a zero byte
        if ( uc == 0xFF ) {  
            // Write zero byte out
            jpeg_writer_emit_byte(coder->writer, 0);
        }
        put_buffer <<= 8;
        put_bits -= 8;
    }
    // update state variables
    coder->put_value = put_buffer; 
    coder->put_bits = put_bits;
    return 0;
}

/**
 * Emit left bits
 * 
 * @param coder  Huffman coder structure
 * @return void
 */
inline void
jpeg_huffman_cpu_encoder_emit_left_bits(struct jpeg_huffman_cpu_encoder* coder)
{
    // Fill 7 bits with ones
    if ( jpeg_huffman_cpu_encoder_emit_bits(coder, 0x7F, 7) != 0 )
        return;
    
    //unsigned char uc = (unsigned char) ((coder->put_value >> 16) & 0xFF);
    // Write one byte out
    //jpeg_writer_emit_byte(coder->writer, uc);
    
    coder->put_value = 0; 
    coder->put_bits = 0;
}

/**
 * Encode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_huffman_cpu_encoder_encode_block(struct jpeg_huffman_cpu_encoder* coder, int16_t* data)
{        
    int16_t* block = data;
    
    /*printf("Encode block\n");
    for ( int y = 0; y < 8; y++) {
        for ( int x = 0; x < 8; x++ ) {
            printf("%4d ", block[y * 8 + x]);
        }
        printf("\n");
    }*/

    // Encode the DC coefficient difference per section F.1.2.1
    int temp = block[0] - coder->dc;
    coder->dc = block[0];

    int temp2 = temp;
    if ( temp < 0 ) {
        // Temp is abs value of input
        temp = -temp;
        // For a negative input, want temp2 = bitwise complement of abs(input)
        // This code assumes we are on a two's complement machine
        temp2--;
    }

    // Find the number of bits needed for the magnitude of the coefficient
    int nbits = 0;
    while ( temp ) {
        nbits++;
        temp >>= 1;
    }

    //    Write category number
    if ( jpeg_huffman_cpu_encoder_emit_bits(coder, coder->table_dc->code[nbits], coder->table_dc->size[nbits]) != 0 ) {
        fprintf(stderr, "Fail emit bits %d [code: %d, size: %d]!\n", nbits, coder->table_dc->code[nbits], coder->table_dc->size[nbits]);
        return -1;
    }

    //    Write category offset (EmitBits rejects calls with size 0)
    if ( nbits ) {
        if ( jpeg_huffman_cpu_encoder_emit_bits(coder, (unsigned int) temp2, nbits) != 0 )
            return -1;
    }
    
    // Encode the AC coefficients per section F.1.2.2 (r = run length of zeros)
    int r = 0;
    for ( int k = 1; k < 64; k++ ) 
    {
        if ( (temp = block[jpeg_order_natural[k]]) == 0 ) {
            r++;
        } 
        else {
            // If run length > 15, must emit special run-length-16 codes (0xF0)
            while ( r > 15 ) {
                if ( jpeg_huffman_cpu_encoder_emit_bits(coder, coder->table_ac->code[0xF0], coder->table_ac->size[0xF0]) != 0 )
                    return -1;
                r -= 16;
            }

            temp2 = temp;
            if ( temp < 0 ) {
                // temp is abs value of input
                temp = -temp;        
                // This code assumes we are on a two's complement machine
                temp2--;
            }

            // Find the number of bits needed for the magnitude of the coefficient
            // there must be at least one 1 bit
            nbits = 1;
            while ( (temp >>= 1) )
                nbits++;

            // Emit Huffman symbol for run length / number of bits
            int i = (r << 4) + nbits;
            if ( jpeg_huffman_cpu_encoder_emit_bits(coder, coder->table_ac->code[i], coder->table_ac->size[i]) != 0 )
                return -1;

            // Write Category offset
            if ( jpeg_huffman_cpu_encoder_emit_bits(coder, (unsigned int) temp2, nbits) != 0 )
                return -1;

            r = 0;
        }
    }

    // If all the left coefs were zero, emit an end-of-block code
    if ( r > 0 ) {
        if ( jpeg_huffman_cpu_encoder_emit_bits(coder, coder->table_ac->code[0], coder->table_ac->size[0]) != 0 )
            return -1;
    }

    return 0;
}

/** Documented at declaration */
int
jpeg_huffman_cpu_encoder_encode(struct jpeg_encoder* encoder, enum jpeg_component_type type, int16_t* data)
{    
    int block_cx = (encoder->param_image.width + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE;
    int block_cy = (encoder->param_image.height + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE;
    
    // Initialize huffman coder
    struct jpeg_huffman_cpu_encoder coder;
    coder.table_dc = &encoder->table_huffman[type][JPEG_HUFFMAN_DC];
    coder.table_ac = &encoder->table_huffman[type][JPEG_HUFFMAN_AC];
    coder.put_value = 0;
    coder.put_bits = 0;
    coder.dc = 0;
    coder.writer = encoder->writer;
    coder.restart_interval = encoder->param.restart_interval;
    coder.block_count = 0;
    
    //uint8_t* buffer = encoder->writer->buffer_current;
    
    // Encode all blocks
    for ( int block_y = 0; block_y < block_cy; block_y++ ) {
        for ( int block_x = 0; block_x < block_cx; block_x++ ) {
            // Process restart interval
            if ( coder.restart_interval != 0 ) {
                if ( coder.block_count > 0 && (coder.block_count % coder.restart_interval) == 0 ) {
                    // Emit left bits
                    if ( coder.put_bits > 0 )
                        jpeg_huffman_cpu_encoder_emit_left_bits(&coder);
                    // Restart huffman coder
                    coder.put_value = 0;
                    coder.put_bits = 0;
                    coder.dc = 0;
                    // Output restart marker
                    int restart_marker = JPEG_MARKER_RST0 + (((coder.block_count - coder.restart_interval) / coder.restart_interval) & 0x7);
                    jpeg_writer_emit_marker(encoder->writer, restart_marker);
                    //printf("byte count %d\n", (int)encoder->writer->buffer_current - (int)buffer);
                    //buffer = encoder->writer->buffer_current;
                }
            }
            uint8_t* buffer = encoder->writer->buffer_current;
            
            // Encoder block
            int data_index = (block_y * block_cx + block_x) * JPEG_BLOCK_SIZE * JPEG_BLOCK_SIZE;
            if ( jpeg_huffman_cpu_encoder_encode_block(&coder, &data[data_index]) != 0 ) {
                fprintf(stderr, "Huffman encoder failed at block [%d, %d]!\n", block_y, block_x);
                return -1;
            }
            coder.block_count++;
        }
    }
    
    // Emit left
    if ( coder.put_bits > 0 )
        jpeg_huffman_cpu_encoder_emit_left_bits(&coder);
        
    //printf("byte count %d\n", (int)encoder->writer->buffer_current - (int)buffer);
    
    return 0;
}