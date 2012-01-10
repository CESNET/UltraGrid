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
 
#include "gpujpeg_huffman_cpu_encoder.h"
#include "gpujpeg_util.h"

#ifdef _DEBUG
#define inline
#endif

/** Huffman encoder structure */
struct gpujpeg_huffman_cpu_encoder
{
    // Color components
    struct gpujpeg_component* component;
    
    // JPEG writer structure
    struct gpujpeg_writer* writer;
    
    // Huffman table DC
    struct gpujpeg_table_huffman_encoder* table_dc[GPUJPEG_COMPONENT_TYPE_COUNT];
    // Huffman table AC
    struct gpujpeg_table_huffman_encoder* table_ac[GPUJPEG_COMPONENT_TYPE_COUNT];
    
    // The value (in 4 byte buffer) to be written out
    int put_value;
    // The size (in bits) to be written out
    int put_bits;
    // DC differentize for component
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    // Current scan index
    int scan_index;
    // Component count (1 means non-interleaving, > 1 means interleaving)
    int comp_count;
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
gpujpeg_huffman_cpu_encoder_emit_bits(struct gpujpeg_huffman_cpu_encoder* coder, unsigned int code, int size)
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
        gpujpeg_writer_emit_byte(coder->writer, uc);
        // If need to stuff a zero byte
        if ( uc == 0xFF ) {  
            // Write zero byte out
            gpujpeg_writer_emit_byte(coder->writer, 0);
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
gpujpeg_huffman_cpu_encoder_emit_left_bits(struct gpujpeg_huffman_cpu_encoder* coder)
{
    // Fill 7 bits with ones
    if ( gpujpeg_huffman_cpu_encoder_emit_bits(coder, 0x7F, 7) != 0 )
        return;
    
    //unsigned char uc = (unsigned char) ((coder->put_value >> 16) & 0xFF);
    // Write one byte out
    //gpujpeg_writer_emit_byte(coder->writer, uc);
    
    coder->put_value = 0; 
    coder->put_bits = 0;
}

/**
 * Encode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_huffman_cpu_encoder_encode_block(struct gpujpeg_huffman_cpu_encoder* coder, int16_t* block, int* dc, struct gpujpeg_table_huffman_encoder* table_dc, struct gpujpeg_table_huffman_encoder* table_ac)
{
    /*printf("Encode block\n");
    for ( int y = 0; y < 8; y++) {
        for ( int x = 0; x < 8; x++ ) {
            printf("%4d ", block[y * 8 + x]);
        }
        printf("\n");
    }*/

    // Encode the DC coefficient difference per section F.1.2.1
    int temp = block[0] - *dc;
    *dc = block[0];

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
    if ( gpujpeg_huffman_cpu_encoder_emit_bits(coder, table_dc->code[nbits], table_dc->size[nbits]) != 0 ) {
        fprintf(stderr, "Fail emit bits %d [code: %d, size: %d]!\n", nbits, table_dc->code[nbits], table_dc->size[nbits]);
        return -1;
    }

    //    Write category offset (EmitBits rejects calls with size 0)
    if ( nbits ) {
        if ( gpujpeg_huffman_cpu_encoder_emit_bits(coder, (unsigned int) temp2, nbits) != 0 )
            return -1;
    }
    
    // Encode the AC coefficients per section F.1.2.2 (r = run length of zeros)
    int r = 0;
    for ( int k = 1; k < 64; k++ ) 
    {
        if ( (temp = block[gpujpeg_order_natural[k]]) == 0 ) {
            r++;
        } 
        else {
            // If run length > 15, must emit special run-length-16 codes (0xF0)
            while ( r > 15 ) {
                if ( gpujpeg_huffman_cpu_encoder_emit_bits(coder, table_ac->code[0xF0], table_ac->size[0xF0]) != 0 )
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
            if ( gpujpeg_huffman_cpu_encoder_emit_bits(coder, table_ac->code[i], table_ac->size[i]) != 0 )
                return -1;

            // Write Category offset
            if ( gpujpeg_huffman_cpu_encoder_emit_bits(coder, (unsigned int) temp2, nbits) != 0 )
                return -1;

            r = 0;
        }
    }

    // If all the left coefs were zero, emit an end-of-block code
    if ( r > 0 ) {
        if ( gpujpeg_huffman_cpu_encoder_emit_bits(coder, table_ac->code[0], table_ac->size[0]) != 0 )
            return -1;
    }

    return 0;
}

/**
 * Encode one MCU
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_huffman_cpu_encoder_encode_mcu(struct gpujpeg_huffman_cpu_encoder* coder, int segment_index, int mcu_index)
{
    // Non-interleaving mode
    if ( coder->comp_count == 1 ) {
        // Get component for current scan
        struct gpujpeg_component* component = &coder->component[coder->scan_index];
 
        // Get component data for MCU
        int16_t* block = &component->data_quantized[(segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size];
        
        // Get coder parameters
        int* dc = &coder->dc[coder->scan_index];
        struct gpujpeg_table_huffman_encoder* table_dc = coder->table_dc[component->type];
        struct gpujpeg_table_huffman_encoder* table_ac = coder->table_ac[component->type];
        
        // Encode 8x8 block
        if ( gpujpeg_huffman_cpu_encoder_encode_block(coder, block, dc, table_dc, table_ac) != 0 )
            return -1;
    } 
    // Interleaving mode
    else {
        assert(coder->scan_index == 0);
        for ( int comp = 0; comp < coder->comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];

            // Prepare mcu indexes
            int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
            int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
            // Compute base data index
            int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
            
            // For all vertical 8x8 blocks
            for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                // Compute base row data index
                assert((component->mcu_count_x * component->mcu_size_x) == component->data_width);
                int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                // For all horizontal 8x8 blocks
                for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                    // Compute 8x8 block data index
                    int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                    
                    // Get component data for MCU
                    int16_t* block = &component->data_quantized[data_index];
                    
                    // Get coder parameters
                    int* dc = &coder->dc[comp];
                    struct gpujpeg_table_huffman_encoder* table_dc = coder->table_dc[component->type];
                    struct gpujpeg_table_huffman_encoder* table_ac = coder->table_ac[component->type];
                    
                    // Encode 8x8 block
                    if ( gpujpeg_huffman_cpu_encoder_encode_block(coder, block, dc, table_dc, table_ac) != 0 )
                        return -1;
                }
            }
        }
    }

    return 0;
}

/** Documented at declaration */
int
gpujpeg_huffman_cpu_encoder_encode(struct gpujpeg_encoder* encoder)
{
    // Init huffman ecoder
    struct gpujpeg_huffman_cpu_encoder coder;
    coder.writer = encoder->writer;
    coder.component = encoder->coder.component;
    
    // Set huffman tables
    for ( int type = 0; type < GPUJPEG_COMPONENT_TYPE_COUNT; type++ ) {
        coder.table_dc[type] = &encoder->table_huffman[type][GPUJPEG_HUFFMAN_DC];
        coder.table_ac[type] = &encoder->table_huffman[type][GPUJPEG_HUFFMAN_AC];
    }
    
    // Set mcu component count
    if ( encoder->coder.param.interleaved == 1 )
        coder.comp_count = encoder->coder.param_image.comp_count;
    else
        coder.comp_count = 1;
    assert(coder.comp_count >= 1 && coder.comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    
    // Ensure that before first scan the emit_left_bits will not be invoked
    coder.put_bits = 0;
    // Perform scan init also for first scan
    coder.scan_index = -1; 
    
    // Encode all segments
    for ( int segment_index = 0; segment_index < encoder->coder.segment_count; segment_index++ ) {
        struct gpujpeg_segment* segment = &encoder->coder.segment[segment_index];
        
        //printf("segment %d, %d\n", segment_index, segment->scan_index);
        
        // Init scan if changed
        if ( coder.scan_index != segment->scan_index ) {
            // Emit left from previous scan
            if ( coder.put_bits > 0 )
                gpujpeg_huffman_cpu_encoder_emit_left_bits(&coder);
        
            // Write scan header
            gpujpeg_writer_write_scan_header(encoder, segment->scan_index);
            
            // Initialize huffman coder
            coder.put_value = 0;
            coder.put_bits = 0;
            for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
                coder.dc[comp] = 0;
            
            // Set current scan index
            coder.scan_index = segment->scan_index;
        }
        
        // Encode segment MCUs
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            if ( gpujpeg_huffman_cpu_encoder_encode_mcu(&coder, segment->scan_segment_index, mcu_index) != 0 ) {
                fprintf(stderr, "Huffman encoder failed at block [%d, %d]!\n", segment_index, mcu_index);
                return -1;
            }
        }
        
        // Output restart marker, if segment is not last in current scan
        if ( (segment_index + 1) < encoder->coder.segment_count && encoder->coder.segment[segment_index + 1].scan_index == coder.scan_index ) {
            // Emit left bits
            if ( coder.put_bits > 0 )
                gpujpeg_huffman_cpu_encoder_emit_left_bits(&coder);
            // Restart huffman coder
            coder.put_value = 0;
            coder.put_bits = 0;
            for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
                coder.dc[comp] = 0;
            // Output restart marker
            int restart_marker = GPUJPEG_MARKER_RST0 + (segment->scan_segment_index & 0x7);
            gpujpeg_writer_emit_marker(encoder->writer, restart_marker);
        }
    }
    
    // Emit left
    if ( coder.put_bits > 0 )
        gpujpeg_huffman_cpu_encoder_emit_left_bits(&coder);
    
    return 0;
}
