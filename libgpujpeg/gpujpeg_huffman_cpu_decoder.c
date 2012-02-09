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
 
#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_util.h"

#ifdef _DEBUG
#define inline
#endif

/** Huffman encoder structure */
struct gpujpeg_huffman_cpu_decoder
{
    // Color components
    struct gpujpeg_component* component;
    
    // Huffman table DC
    struct gpujpeg_table_huffman_decoder* table_dc[GPUJPEG_COMPONENT_TYPE_COUNT];
    // Huffman table AC
    struct gpujpeg_table_huffman_decoder* table_ac[GPUJPEG_COMPONENT_TYPE_COUNT];
    
    // Get bits
    int get_bits;
    // Get buffer
    int get_buff;
    // DC differentize for component
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    
    // Coding component count
    int comp_count;
    // Current scan index
    int scan_index;
    
    // Compressed data
    uint8_t* data;
    // Compressed data size
    int data_size;
};

/**
 * Fill more bit to current get buffer
 * 
 * @param coder
 * @return void
 */
void
gpujpeg_huffman_cpu_decoder_decode_fill_bit_buffer(struct gpujpeg_huffman_cpu_decoder* coder)
{
    while ( coder->get_bits < 25 ) {
        //Are there some data?
        if( coder->data_size > 0 ) { 
            // Attempt to read a byte
            //printf("read byte %X 0x%X\n", (int)coder->data, (unsigned char)*coder->data);
            unsigned char uc = *coder->data++;
            coder->data_size--;            

            // If it's 0xFF, check and discard stuffed zero byte
            if ( uc == 0xFF ) {
                do {
                    //printf("read byte %X 0x%X\n", (int)coder->data, (unsigned char)*coder->data);
                    uc = *coder->data++;
                    coder->data_size--;
                } while ( uc == 0xFF );

                if ( uc == 0 ) {
                    // Found FF/00, which represents an FF data byte
                    uc = 0xFF;
                } else {                
                    // There should be enough bits still left in the data segment;
                    // if so, just break out of the outer while loop.
                    //if (m_nGetBits >= nbits)
                    if ( coder->get_bits >= 0 )
                        break;
                }
            }

            coder->get_buff = (coder->get_buff << 8) | ((int) uc);
            coder->get_bits += 8;            
        }
        else
            break;
    }
}

/**
 * Get bits
 * 
 * @param coder  Decoder structure
 * @param nbits  Number of bits to get
 * @return bits
 */
inline int
gpujpeg_huffman_cpu_decoder_get_bits(struct gpujpeg_huffman_cpu_decoder* coder, int nbits) 
{
    //we should read nbits bits to get next data
    if( coder->get_bits < nbits )
        gpujpeg_huffman_cpu_decoder_decode_fill_bit_buffer(coder);
    coder->get_bits -= nbits;
    return (int)(coder->get_buff >> coder->get_bits) & ((1 << nbits) - 1);
}


/**
 * Special Huffman decode:
 * (1) For codes with length > 8
 * (2) For codes with length < 8 while data is finished
 * 
 * @return int
 */
int
gpujpeg_huffman_cpu_decoder_decode_special_decode(struct gpujpeg_huffman_cpu_decoder* coder, struct gpujpeg_table_huffman_decoder* table, int min_bits)
{
    // HUFF_DECODE has determined that the code is at least min_bits
    // bits long, so fetch that many bits in one swoop.
    int code = gpujpeg_huffman_cpu_decoder_get_bits(coder, min_bits);

    // Collect the rest of the Huffman code one bit at a time.
    // This is per Figure F.16 in the JPEG spec.
    int l = min_bits;
    while ( code > table->maxcode[l] ) {
        code <<= 1;
        code |= gpujpeg_huffman_cpu_decoder_get_bits(coder, 1);
        l++;
    }

    // With garbage input we may reach the sentinel value l = 17.
    if ( l > 16 ) {
        // Fake a zero as the safest result
        return 0;
    }
    
    return table->huffval[table->valptr[l] + (int)(code - table->mincode[l])];
}

/**
 * To find dc or ac value according to category and category offset
 * 
 * @return int
 */
inline int
gpujpeg_huffman_cpu_decoder_value_from_category(int category, int offset)
{
    // Method 1: 
    // On some machines, a shift and add will be faster than a table lookup.
    // #define HUFF_EXTEND(x,s) \
    // ((x)< (1<<((s)-1)) ? (x) + (((-1)<<(s)) + 1) : (x)) 

    // Method 2: Table lookup
    // If (offset < half[category]), then value is below zero
    // Otherwise, value is above zero, and just the offset 
    // entry n is 2**(n-1)
    static const int half[16] =    { 
        0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 
        0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000
    };

    //start[i] is the starting value in this category; surely it is below zero
    // entry n is (-1 << n) + 1
    static const int start[16] = { 
        0, ((-1)<<1) + 1, ((-1)<<2) + 1, ((-1)<<3) + 1, ((-1)<<4) + 1,
        ((-1)<<5) + 1, ((-1)<<6) + 1, ((-1)<<7) + 1, ((-1)<<8) + 1,
        ((-1)<<9) + 1, ((-1)<<10) + 1, ((-1)<<11) + 1, ((-1)<<12) + 1,
        ((-1)<<13) + 1, ((-1)<<14) + 1, ((-1)<<15) + 1 
    };    

    return (offset < half[category]) ? (offset + start[category]) : offset;    
}

/**
 * Get category number for dc, or (0 run length, ac category) for ac.
 * The max length for Huffman codes is 15 bits; so we use 32 bits buffer    
 * m_nGetBuff, with the validated length is m_nGetBits.
 * Usually, more than 95% of the Huffman codes will be 8 or fewer bits long
 * To speed up, we should pay more attention on the codes whose length <= 8
 * 
 * @return int
 */
inline int
gpujpeg_huffman_cpu_decoder_get_category(struct gpujpeg_huffman_cpu_decoder* coder, struct gpujpeg_table_huffman_decoder* table)
{
    // If left bits < 8, we should get more data
    if ( coder->get_bits < 8 )
        gpujpeg_huffman_cpu_decoder_decode_fill_bit_buffer(coder);

    // Call special process if data finished; min bits is 1
    if( coder->get_bits < 8 )
        return gpujpeg_huffman_cpu_decoder_decode_special_decode(coder, table, 1);

    // Peek the first valid byte    
    int look = ((coder->get_buff >> (coder->get_bits - 8)) & 0xFF);
    int nb = table->look_nbits[look];

    if ( nb ) { 
        coder->get_bits -= nb;
        return table->look_sym[look]; 
    } else {
        //Decode long codes with length >= 9
        return gpujpeg_huffman_cpu_decoder_decode_special_decode(coder, table, 9);
    }
}

/**
 * Decode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_huffman_cpu_decoder_decode_block(struct gpujpeg_huffman_cpu_decoder* coder, int16_t* data, int* dc, struct gpujpeg_table_huffman_decoder* table_dc, struct gpujpeg_table_huffman_decoder* table_ac)
{    
    // Zero block output
    memset(data, 0, sizeof(int16_t) * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE);

    // Section F.2.2.1: decode the DC coefficient difference
    // get dc category number, s
    int s = gpujpeg_huffman_cpu_decoder_get_category(coder, table_dc);
    if ( s ) {
        // Get offset in this dc category
        int r = gpujpeg_huffman_cpu_decoder_get_bits(coder, s);
        // Get dc difference value
        s = gpujpeg_huffman_cpu_decoder_value_from_category(s, r);
    }

    // Convert DC difference to actual value, update last_dc_val
    s += *dc;
    *dc = s;

    // Output the DC coefficient (assumes gpujpeg_natural_order[0] = 0)
    data[0] = s;
    
    // Section F.2.2.2: decode the AC coefficients
    // Since zeroes are skipped, output area must be cleared beforehand
    for ( int k = 1; k < 64; k++ ) {
        // s: (run, category)
        int s = gpujpeg_huffman_cpu_decoder_get_category(coder, table_ac);
        // r: run length for ac zero, 0 <= r < 16
        int r = s >> 4;
        // s: category for this non-zero ac
        s &= 15;
        if ( s ) {
            //    k: position for next non-zero ac
            k += r;
            //    r: offset in this ac category
            r = gpujpeg_huffman_cpu_decoder_get_bits(coder, s);
            //    s: ac value
            s = gpujpeg_huffman_cpu_decoder_value_from_category(s, r);

            data[gpujpeg_order_natural[k]] = s;
        } else {
            // s = 0, means ac value is 0 ? Only if r = 15.  
            //means all the left ac are zero
            if ( r != 15 )
                break;
            k += 15;
        }
    }
    
    /*printf("CPU Decode Block\n");
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            printf("%4d ", data[y * 8 + x]);
        }
        printf("\n");
    }*/
    
    return 0;
}

/**
 * Decode one MCU
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_huffman_cpu_decoder_decode_mcu(struct gpujpeg_huffman_cpu_decoder* coder, int segment_index, int mcu_index)
{
    // Non-interleaving mode
    if ( coder->comp_count == 1 ) {
        // Get component for current scan
        struct gpujpeg_component* component = &coder->component[coder->scan_index];
 
        // Get component data for MCU
        int16_t* block = &component->data_quantized[(segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size];
        
        // Get coder parameters
        int* dc = &coder->dc[coder->scan_index];
        struct gpujpeg_table_huffman_decoder* table_dc = coder->table_dc[component->type];
        struct gpujpeg_table_huffman_decoder* table_ac = coder->table_ac[component->type];
        
        // Encode 8x8 block
        if ( gpujpeg_huffman_cpu_decoder_decode_block(coder, block, dc, table_dc, table_ac) != 0 )
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
                    struct gpujpeg_table_huffman_decoder* table_dc = coder->table_dc[component->type];
                    struct gpujpeg_table_huffman_decoder* table_ac = coder->table_ac[component->type];
                    
                    // Encode 8x8 block
                    if ( gpujpeg_huffman_cpu_decoder_decode_block(coder, block, dc, table_dc, table_ac) != 0 )
                        return -1;
                }
            }
        }
    }
    return 0;
}

/** Documented at declaration */
int
gpujpeg_huffman_cpu_decoder_decode(struct gpujpeg_decoder* decoder)
{
    int block_cx = (decoder->coder.param_image.width + GPUJPEG_BLOCK_SIZE - 1) / GPUJPEG_BLOCK_SIZE;
    int block_cy = (decoder->coder.param_image.height + GPUJPEG_BLOCK_SIZE - 1) / GPUJPEG_BLOCK_SIZE;
    
    // Initialize huffman coder
    struct gpujpeg_huffman_cpu_decoder coder;
    coder.component = decoder->coder.component;
    coder.scan_index = -1;
    
    // Set huffman tables
    for ( int type = 0; type < GPUJPEG_COMPONENT_TYPE_COUNT; type++ ) {
        coder.table_dc[type] = &decoder->table_huffman[type][GPUJPEG_HUFFMAN_DC];
        coder.table_ac[type] = &decoder->table_huffman[type][GPUJPEG_HUFFMAN_AC];
    }
    
    // Set mcu component count
    if ( decoder->coder.param.interleaved == 1 )
        coder.comp_count = decoder->coder.param_image.comp_count;
    else
        coder.comp_count = 1;
    assert(coder.comp_count >= 1 && coder.comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    
    // Decode all segments
    for ( int segment_index = 0; segment_index < decoder->segment_count; segment_index++ ) {
        // Get segment structure
        struct gpujpeg_segment* segment = &decoder->coder.segment[segment_index];
        
        // Change current scan index
        if ( coder.scan_index != segment->scan_index ) {
            coder.scan_index = segment->scan_index;
        }
        
        // Initialize huffman coder
        coder.get_buff = 0;
        coder.get_bits = 0;
        for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
            coder.dc[comp] = 0;
        coder.data = &decoder->coder.data_compressed[segment->data_compressed_index];
        coder.data_size = segment->data_compressed_size;
        
        // Decode segment MCUs
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            if ( gpujpeg_huffman_cpu_decoder_decode_mcu(&coder, segment->scan_segment_index, mcu_index) != 0 ) {
                fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed at block [%d, %d]!\n", segment_index, mcu_index);
                return -1;
            }
        }
    }
    
    return 0;
}
