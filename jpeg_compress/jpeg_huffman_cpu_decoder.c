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
 
#include "jpeg_huffman_cpu_decoder.h"
#include "jpeg_util.h"

#ifdef _DEBUG
#define inline
#endif

/** Huffman encoder structure */
struct jpeg_huffman_cpu_decoder
{
    // Scan data for all scans
    uint8_t* data_scan;
    // Size for data for all scans
    int data_scan_size;
    // Indexes into scan data buffer for all segments
    int* data_scan_index;
    // Total segment count for all scans
    int segment_count;
    // Huffman table DC
    struct jpeg_table_huffman_decoder* table_dc;
    // Huffman table AC
    struct jpeg_table_huffman_decoder* table_ac;
    // DC differentize for component
    int dc;
    // Get bits
    int get_bits;
    // Get buffer
    int get_buff;
    // Compressed data
    uint8_t* data;
    // Compressed data size
    int data_size;
    // Restart interval
    int restart_interval;
    // Restart interval position
    int restart_position;
    // Current segment index
    int segment_index;
};

/**
 * Fill more bit to current get buffer
 * 
 * @param coder
 * @return void
 */
void
jpeg_huffman_cpu_decoder_decode_fill_bit_buffer(struct jpeg_huffman_cpu_decoder* coder)
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
jpeg_huffman_cpu_decoder_get_bits(struct jpeg_huffman_cpu_decoder* coder, int nbits) 
{
    //we should read nbits bits to get next data
    if( coder->get_bits < nbits )
        jpeg_huffman_cpu_decoder_decode_fill_bit_buffer(coder);
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
jpeg_huffman_cpu_decoder_decode_special_decode(struct jpeg_huffman_cpu_decoder* coder, struct jpeg_table_huffman_decoder* table, int min_bits)
{
    // HUFF_DECODE has determined that the code is at least min_bits
    // bits long, so fetch that many bits in one swoop.
    int code = jpeg_huffman_cpu_decoder_get_bits(coder, min_bits);

    // Collect the rest of the Huffman code one bit at a time.
    // This is per Figure F.16 in the JPEG spec.
    int l = min_bits;
    while ( code > table->maxcode[l] ) {
        code <<= 1;
        code |= jpeg_huffman_cpu_decoder_get_bits(coder, 1);
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
jpeg_huffman_cpu_decoder_value_from_category(int category, int offset)
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
jpeg_huffman_cpu_decoder_get_category(struct jpeg_huffman_cpu_decoder* coder, struct jpeg_table_huffman_decoder* table)
{
    // If left bits < 8, we should get more data
    if ( coder->get_bits < 8 )
        jpeg_huffman_cpu_decoder_decode_fill_bit_buffer(coder);

    // Call special process if data finished; min bits is 1
    if( coder->get_bits < 8 )
        return jpeg_huffman_cpu_decoder_decode_special_decode(coder, table, 1);

    // Peek the first valid byte    
    int look = ((coder->get_buff >> (coder->get_bits - 8)) & 0xFF);
    int nb = table->look_nbits[look];

    if ( nb ) { 
        coder->get_bits -= nb;
        return table->look_sym[look]; 
    } else {
        //Decode long codes with length >= 9
        return jpeg_huffman_cpu_decoder_decode_special_decode(coder, table, 9);
    }
}

/**
 * Decode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
jpeg_huffman_cpu_decoder_decode_block(struct jpeg_huffman_cpu_decoder* coder, int16_t* data)
{    
    // Restart coder
    if ( coder->restart_interval > 0 && coder->restart_position == 0 ) {
        coder->get_buff = 0;
        coder->get_bits = 0;
        coder->dc = 0;
        coder->restart_position = coder->restart_interval;
        coder->segment_index++;
        
        // Set coder data for next segment
        int data_index = coder->data_scan_index[coder->segment_index];
        coder->data = &coder->data_scan[data_index];
        if ( (coder->segment_index + 1) >= coder->segment_count )
            coder->data_size = coder->data_scan_size - data_index;
        else
            coder->data_size = coder->data_scan_index[coder->segment_index + 1] - data_index;
    }
    
    // Zero block output
    memset(data, 0, sizeof(int16_t) * JPEG_BLOCK_SIZE * JPEG_BLOCK_SIZE);

    // Section F.2.2.1: decode the DC coefficient difference
    // get dc category number, s
    int s = jpeg_huffman_cpu_decoder_get_category(coder, coder->table_dc);
    if ( s ) {
        // Get offset in this dc category
        int r = jpeg_huffman_cpu_decoder_get_bits(coder, s);
        // Get dc difference value
        s = jpeg_huffman_cpu_decoder_value_from_category(s, r);
    }

    // Convert DC difference to actual value, update last_dc_val
    s += coder->dc;
    coder->dc = s;

    // Output the DC coefficient (assumes jpeg_natural_order[0] = 0)
    data[0] = s;
    
    // Section F.2.2.2: decode the AC coefficients
    // Since zeroes are skipped, output area must be cleared beforehand
    for ( int k = 1; k < 64; k++ ) {
        // s: (run, category)
        int s = jpeg_huffman_cpu_decoder_get_category(coder, coder->table_ac);
        // r: run length for ac zero, 0 <= r < 16
        int r = s >> 4;
        // s: category for this non-zero ac
        s &= 15;
        if ( s ) {
            //    k: position for next non-zero ac
            k += r;
            //    r: offset in this ac category
            r = jpeg_huffman_cpu_decoder_get_bits(coder, s);
            //    s: ac value
            s = jpeg_huffman_cpu_decoder_value_from_category(s, r);

            data[jpeg_order_natural[k]] = s;
        } else {
            // s = 0, means ac value is 0 ? Only if r = 15.  
            //means all the left ac are zero
            if ( r != 15 )
                break;
            k += 15;
        }
    }
    
    coder->restart_position--;
    
    /*printf("CPU Decode Block\n");
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            printf("%4d ", data[y * 8 + x]);
        }
        printf("\n");
    }*/
    
    return 0;
}

/** Documented at declaration */
int
jpeg_huffman_cpu_decoder_decode(struct jpeg_decoder* decoder, enum jpeg_component_type type, struct jpeg_decoder_scan* scan, int16_t* data_decompressed)
{    
    int block_cx = (decoder->param_image.width + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE;
    int block_cy = (decoder->param_image.height + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE;
    
    // Initialize huffman coder
    struct jpeg_huffman_cpu_decoder coder;
    coder.data_scan = decoder->data_scan;
    coder.data_scan_size = decoder->data_scan_size;
    coder.data_scan_index = decoder->data_scan_index;
    coder.segment_count = decoder->segment_count;
    coder.table_dc = &decoder->table_huffman[type][JPEG_HUFFMAN_DC];
    coder.table_ac = &decoder->table_huffman[type][JPEG_HUFFMAN_AC];
    coder.get_buff = 0;
    coder.get_bits = 0;
    coder.dc = 0;
    coder.restart_interval = decoder->restart_interval;
    coder.restart_position = decoder->restart_interval;
    coder.segment_index = scan->segment_index;
    
    // Set coder data
    int data_index = coder.data_scan_index[coder.segment_index];
    coder.data = &coder.data_scan[data_index];
    if ( (coder.segment_index + 1) >= coder.segment_count )
        coder.data_size = coder.data_scan_size - data_index;
    else
        coder.data_size = coder.data_scan_index[coder.segment_index + 1] - data_index;
        
    //printf("start %d, size %d\n", coder.data, coder.data_size);
    
    // Decode all blocks
    for ( int block_y = 0; block_y < block_cy; block_y++ ) {
        for ( int block_x = 0; block_x < block_cx; block_x++ ) {
            int data_index = (block_y * block_cx + block_x) * JPEG_BLOCK_SIZE * JPEG_BLOCK_SIZE;
            if ( jpeg_huffman_cpu_decoder_decode_block(&coder, &data_decompressed[data_index]) != 0 ) {
                fprintf(stderr, "Huffman decoder failed at block [%d, %d]!\n", block_y, block_x);
                return -1;
            }
        }
    }
    
    return 0;
}