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

#include "gpujpeg_huffman_gpu_decoder.h"
#include <libgpujpeg/gpujpeg_util.h>

/**
 * Entry of pre-built Huffman fast-decoding table.
 */
struct gpujpeg_table_huffman_decoder_entry {
    
    int value_nbits;
};

/**
 * 4 pre-built tables for faster Huffman decoding (codewords up-to 16 bit length):
 *   0x00000 to 0x0ffff: luminance DC table
 *   0x10000 to 0x1ffff: luminance AC table
 *   0x20000 to 0x2ffff: chrominance DC table
 *   0x30000 to 0x3ffff: chrominance AC table
 * 
 * Each entry consists of:
 *   - Number of bits of code corresponding to this entry (0 - 16, both inclusive) - bits 4 to 8
 *   - Number of run-length coded zeros before currently decoded coefficient + 1 (1 - 64, both inclusive) - bits 9 to 15
 *   - Number of bits representing the value of currently decoded coefficient (0 - 15, both inclusive) - bits 0 to 3
 * bit #:    15                      9   8               4   3           0
 *         +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 * value:  |      RLE zero count       |   code bit size   | value bit size|
 *         +---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+---+
 */
__device__ uint16_t gpujpeg_huffman_gpu_decoder_tables_full[4 * (1 << 16)];

/** Number of code bits to be checked first (with high chance for the code to fit into this number of bits). */
#define QUICK_CHECK_BITS 10
#define QUICK_TABLE_ITEMS (4 * (1 << QUICK_CHECK_BITS))
// TODO: try to tweak QUICK table size and memory space

/** Table with same format as the full table, except that all-zero-entry means that the full table should be consulted. */
__device__ uint16_t gpujpeg_huffman_gpu_decoder_tables_quick[QUICK_TABLE_ITEMS];

/** Same table as above, but copied into constant memory */
__constant__ uint16_t gpujpeg_huffman_gpu_decoder_tables_quick_const[QUICK_TABLE_ITEMS];

/** Natural order in constant memory */
__constant__ int gpujpeg_huffman_gpu_decoder_order_natural[GPUJPEG_ORDER_NATURAL_SIZE];

// /**
//  * Fill more bit to current get buffer
//  * 
//  * @param get_bits
//  * @param get_buff
//  * @param data
//  * @param data_size
//  * @return void
//  */
// __device__ inline void
// gpujpeg_huffman_gpu_decoder_decode_fill_bit_buffer(int & get_bits, int & get_buff, uint8_t* & data, int & data_size)
// {
//     while ( get_bits < 25 ) {
//         //Are there some data?
//         if( data_size > 0 ) { 
//             // Attempt to read a byte
//             //printf("read byte %X 0x%X\n", (int)data, (unsigned char)*data);
//             unsigned char uc = *data++;
//             data_size--;            
// 
//             // If it's 0xFF, check and discard stuffed zero byte
//             if ( uc == 0xFF ) {
//                 while ( uc == 0xFF ) {
//                     //printf("read byte %X 0x%X\n", (int)data, (unsigned char)*data);
//                     uc = *data++;
//                     data_size--;
//                 }
// 
//                 if ( uc == 0 ) {
//                     // Found FF/00, which represents an FF data byte
//                     uc = 0xFF;
//                 } else {                
//                     // There should be enough bits still left in the data segment;
//                     // if so, just break out of the outer while loop.
//                     //if (m_nGetBits >= nbits)
//                     if ( get_bits >= 0 )
//                         break;
//                 }
//             }
// 
//             get_buff = (get_buff << 8) | ((int) uc);
//             get_bits += 8;            
//         }
//         else
//             break;
//     }
// }

/**
 * Loads at least specified number of bits into the register
 */
__device__ inline void
gpujpeg_huffman_gpu_decoder_load_bits(
                const unsigned int required_bit_count, unsigned int & r_bit,
                unsigned int & r_bit_count, uint4 * const s_byte, unsigned int & s_byte_idx
) {
    // Add bytes until have enough
    while(r_bit_count < required_bit_count) {
        // Load byte value and posibly skip next stuffed byte if loaded byte's value is 0xFF
        const uint8_t byte_value = ((const uint8_t*)s_byte)[s_byte_idx++];
        if((uint8_t)0xFF == byte_value) {
            s_byte_idx++;
        }
        
        // Add newly loaded byte to the buffer, updating bit count
        r_bit = (r_bit << 8) + byte_value;
        r_bit_count += 8;
    }
}


/**
 * Get bits
 * 
 * @param nbits  Number of bits to get
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return bits
 */
__device__ inline unsigned int
gpujpeg_huffman_gpu_decoder_get_bits(
                const unsigned int nbits, unsigned int & r_bit, unsigned int & r_bit_count, 
                uint4 * const s_byte, unsigned int & s_byte_idx)
{
    // load bits into the register if haven't got enough
    gpujpeg_huffman_gpu_decoder_load_bits(nbits, r_bit, r_bit_count, s_byte, s_byte_idx);
    
    // update remaining bit count
    r_bit_count -= nbits;
    
    // return bits 
    return (r_bit >> r_bit_count) & ((1 << nbits) - 1);
}

/**
 * Gets bits without removing them from the buffer.
 */
__device__ inline unsigned int
gpujpeg_huffman_gpu_decoder_peek_bits(
                const unsigned int nbits, unsigned int & r_bit, unsigned int & r_bit_count,
                uint4 * const s_byte, unsigned int & s_byte_idx)
{
    // load bits into the register if haven't got enough
    gpujpeg_huffman_gpu_decoder_load_bits(nbits, r_bit, r_bit_count, s_byte, s_byte_idx);
    
    // return bits 
    return (r_bit >> (r_bit_count - nbits)) & ((1 << nbits) - 1);
}

/**
 * Removes some bits from the buffer (assumes that they are there)
 */
__device__ inline void
gpujpeg_huffman_gpu_decoder_discard_bits(const unsigned int nb, unsigned int, unsigned int & r_bit_count) {
    r_bit_count -= nb;
}

/**
 * Special Huffman decode:
 * (1) For codes with length > 8
 * (2) For codes with length < 8 while data is finished
 * 
 * @param table
 * @param min_bits
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return int
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_decode_special_decode(
                const struct gpujpeg_table_huffman_decoder* const table, int min_bits, unsigned int & r_bit,
                unsigned int & r_bit_count, uint4 * const s_byte, unsigned int & s_byte_idx)
{
    // HUFF_DECODE has determined that the code is at least min_bits
    // bits long, so fetch that many bits in one swoop.
    int code = gpujpeg_huffman_gpu_decoder_get_bits(min_bits, r_bit, r_bit_count, s_byte, s_byte_idx);

    // Collect the rest of the Huffman code one bit at a time.
    // This is per Figure F.16 in the JPEG spec.
    int l = min_bits;
    while ( code > table->maxcode[l] ) {
        code <<= 1;
        code |= gpujpeg_huffman_gpu_decoder_get_bits(1, r_bit, r_bit_count, s_byte, s_byte_idx);
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
 * To find dc or ac value according to code and its bit length s
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_value_from_category(int nbits, int code)
{
    // TODO: try to replace with __constant__ table lookup
    return code < ((1 << nbits) >> 1) ? (code + ((-1) << nbits) + 1) : code;
    
//     // Method 1: 
//     // On some machines, a shift and add will be faster than a table lookup.
//     // #define HUFF_EXTEND(x,s) \
//     // ((x)< (1<<((s)-1)) ? (x) + (((-1)<<(s)) + 1) : (x)) 
// 
//     // Method 2: Table lookup
//     // If (offset < half[category]), then value is below zero
//     // Otherwise, value is above zero, and just the offset 
//     // entry n is 2**(n-1)
//     const int half[16] =    { 
//         0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 
//         0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000
//     };
// 
//     //start[i] is the starting value in this category; surely it is below zero
//     // entry n is (-1 << n) + 1
//     const int start[16] = { 
//         0, ((-1)<<1) + 1, ((-1)<<2) + 1, ((-1)<<3) + 1, ((-1)<<4) + 1,
//         ((-1)<<5) + 1, ((-1)<<6) + 1, ((-1)<<7) + 1, ((-1)<<8) + 1,
//         ((-1)<<9) + 1, ((-1)<<10) + 1, ((-1)<<11) + 1, ((-1)<<12) + 1,
//         ((-1)<<13) + 1, ((-1)<<14) + 1, ((-1)<<15) + 1 
//     };    
// 
//     return (code < half[nbits]) ? (code + start[nbits]) : code;    
}

/**
 * Decodes next coefficient, updating its output index
 * 
 * @param table
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return int
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_get_coefficient(
                unsigned int & r_bit, unsigned int & r_bit_count, uint4* const s_byte,
                unsigned int & s_byte_idx, const unsigned int table_offset, unsigned int & coefficient_idx)
{
    // Peek next 16 bits and use them as an index into decoder table to find all the info.
    const unsigned int table_idx = table_offset + gpujpeg_huffman_gpu_decoder_peek_bits(16, r_bit, r_bit_count, s_byte, s_byte_idx);
    
    // Try the quick table first (use the full table only if not succeded with the quick table)
    unsigned int packed_info = gpujpeg_huffman_gpu_decoder_tables_quick_const[table_idx >> (16 - QUICK_CHECK_BITS)];
    if(0 == packed_info) {
        packed_info = gpujpeg_huffman_gpu_decoder_tables_full[table_idx];
    }
    
    // remove the right number of bits from the bit buffer
    gpujpeg_huffman_gpu_decoder_discard_bits((packed_info >> 4) & 0x1F, r_bit, r_bit_count);
    
    // update coefficient index by skipping run-length encoded zeros
    coefficient_idx += packed_info >> 9;
    
    // read coefficient bits and decode the coefficient from them
    const unsigned int value_nbits = packed_info & 0xF;
    const unsigned int value_code = gpujpeg_huffman_gpu_decoder_get_bits(value_nbits, r_bit, r_bit_count, s_byte, s_byte_idx);
    
    // return deocded coefficient
    return gpujpeg_huffman_gpu_decoder_value_from_category(value_nbits, value_code);
}

/**
 * Decode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ inline int
gpujpeg_huffman_gpu_decoder_decode_block(
    int & dc, int16_t* const data_output, const unsigned int table_offset,
    unsigned int & r_bit, unsigned int & r_bit_count, uint4* const s_byte,
    unsigned int & s_byte_idx, const uint4* & d_byte, unsigned int & d_byte_chunk_count)
{
    // TODO: try unified decoding of DC/AC coefficients
    
    // Index of next coefficient to be decoded (in ZIG-ZAG order)
    unsigned int coefficient_idx = 0;
    
    // Section F.2.2.1: decode the DC coefficient difference
    // Get the coefficient value (using DC coding table)
    int dc_coefficient_value = gpujpeg_huffman_gpu_decoder_get_coefficient(r_bit, r_bit_count, s_byte, s_byte_idx, table_offset, coefficient_idx);

    // Convert DC difference to actual value, update last_dc_val
    dc = dc_coefficient_value += dc;

    // Output the DC coefficient (assumes gpujpeg_natural_order[0] = 0)
    // TODO: try to skip saving of zero coefficients
    data_output[0] = dc_coefficient_value;
    
    // TODO: error check: coefficient_idx must still be 0 in valid codestreams
    coefficient_idx = 1;
    
    // Section F.2.2.2: decode the AC coefficients
    // Since zeroes are skipped, output area must be cleared beforehand
    do {
        // Possibly load more bytes into shared buffer from global memory
        if(s_byte_idx >= 16) {
            // Move remaining bytes to begin and update index of next byte
            s_byte[0] = s_byte[1];
            s_byte_idx -= 16;
            
            // Load another byte chunk from global memory only if there is one
            if(d_byte_chunk_count) {
                s_byte[1] = *(d_byte++);
                d_byte_chunk_count--;
            }
        }
        
        // decode next coefficient, updating its destination index
        const int coefficient_value = gpujpeg_huffman_gpu_decoder_get_coefficient(r_bit, r_bit_count, s_byte, s_byte_idx, table_offset + 0x10000, coefficient_idx);
        
        // stop with this block if have all coefficients
        if(coefficient_idx > 64) {
            break;
        }
        
        // save the coefficient   TODO: try to ommit saving 0 coefficients
        data_output[gpujpeg_huffman_gpu_decoder_order_natural[coefficient_idx - 1]] = coefficient_value;
    } while(coefficient_idx < 64);
    
    return 0;
}


/**
 * Huffman decoder kernel
 * 
 * @return void
 */
template <bool SINGLE_COMP, int THREADS_PER_TBLOCK>
__global__ void
#if __CUDA_ARCH__ < 200
__launch_bounds__(THREADS_PER_TBLOCK, 2)
#else
__launch_bounds__(THREADS_PER_TBLOCK, 6)
#endif
gpujpeg_huffman_decoder_decode_kernel(
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int comp_count,
    int segment_count, 
    uint8_t* d_data_compressed,
    const uint64_t* d_block_list,
    int16_t* d_data_quantized
) {
    int segment_index = blockIdx.x * THREADS_PER_TBLOCK + threadIdx.x;
    if ( segment_index >= segment_count )
        return;
    
    struct gpujpeg_segment* segment = &d_segment[segment_index];
    
    // Byte buffers in shared memory
    __shared__ uint4 s_byte_all[2 * THREADS_PER_TBLOCK]; // 32 bytes per thread
    uint4 * const s_byte = s_byte_all + 2 * threadIdx.x;
    
    // Last DC coefficient values   TODO: try to move into shared memory
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
        dc[comp] = 0;
        
    // Get aligned compressed data chunk pointer and load first 2 chunks of the data
    const unsigned int d_byte_begin_idx = segment->data_compressed_index;
    const unsigned int d_byte_begin_idx_aligned = d_byte_begin_idx & ~15; // loading 16byte chunks
    const uint4* d_byte = (uint4*)(d_data_compressed + d_byte_begin_idx_aligned);
    
    // Get number of remaining global memory byte chunks (not to read bytes out of buffer)
    const unsigned int d_byte_end_idx_aligned = (d_byte_begin_idx + segment->data_compressed_size + 15) & ~15;
    unsigned int d_byte_chunk_count = (d_byte_end_idx_aligned - d_byte_begin_idx_aligned) / 16;
    
    // Load first 2 chunks of compressed data into the shared memory buffer and remember index of first code byte (skipping bytes read due to alignment)
    s_byte[0] = d_byte[0];
    s_byte[1] = d_byte[1];
    d_byte += 2;
    d_byte_chunk_count = max(d_byte_chunk_count, 2) - 2;
    unsigned int s_byte_idx = d_byte_begin_idx - d_byte_begin_idx_aligned;
    
    // bits loaded into the register and their count
    unsigned int r_bit_count = 0;
    unsigned int r_bit = 0; // LSB-aligned
    
    // Non-interleaving mode
    if ( SINGLE_COMP ) {
        // Get component for current scan
        const struct gpujpeg_component* const component = d_component + segment->scan_index; 
        
        // Get huffman tables offset
        const unsigned int table_offset = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00000 : 0x20000;
        
        // Size of MCUs in this segment's component
        const int component_mcu_size = component->mcu_size;
        
        // Pointer to first MCU's output block
        int16_t* block = component->d_data_quantized + segment->scan_segment_index * component->segment_mcu_count * component_mcu_size;
        
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            // Encode 8x8 block
            if ( gpujpeg_huffman_gpu_decoder_decode_block(dc[0], block, table_offset, r_bit, r_bit_count, s_byte, s_byte_idx, d_byte, d_byte_chunk_count) != 0 )
                break;
            
            // advance to next block
            block += component_mcu_size;
        } 
    }
    // Interleaving mode
    else {
        // Pointer to segment's list of 8x8 blocks and their count
        const uint64_t* packed_block_info_ptr = d_block_list + segment->block_index_list_begin;
        
        // Encode all blocks
        for(int block_count = segment->block_count; block_count--;) {
            // Get pointer to next block input data and info about its color type
            const uint64_t packed_block_info = *(packed_block_info_ptr++);
            
            // Get coder parameters
            const int last_dc_idx = packed_block_info & 0x7f;
            
            // Get offset to right part of huffman table
            const unsigned int huffman_table_offset = packed_block_info & 0x80 ? 0x20000 : 0x00000;
            
            // Source data pointer
            int16_t* block = d_data_quantized + (packed_block_info >> 8);
            
            // Encode 8x8 block
            gpujpeg_huffman_gpu_decoder_decode_block(dc[last_dc_idx], block, huffman_table_offset, r_bit, r_bit_count, s_byte, s_byte_idx, d_byte, d_byte_chunk_count);
        }
        
        
//         // Encode MCUs in segment
//         for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
//             
//             
//             
//             
//             
//             
//             
//             
//             //assert(segment->scan_index == 0);
//             for ( int comp = 0; comp < comp_count; comp++ ) {
//                 struct gpujpeg_component* component = &d_component[comp];
// 
//                 // Prepare mcu indexes
//                 int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
//                 int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
//                 // Compute base data index
//                 int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
//                 
//                 // For all vertical 8x8 blocks
//                 for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
//                     // Compute base row data index
//                     int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
//                     // For all horizontal 8x8 blocks
//                     for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
//                         // Compute 8x8 block data index
//                         int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
//                         
//                         // Get component data for MCU
//                         int16_t* block = &component->d_data_quantized[data_index];
//                         
//                         // Get coder parameters
//                         int & component_dc = dc[comp];
//             
//                         // Get huffman tables offset
//                         const unsigned int table_offset = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00000 : 0x20000;
//                         
//                         // Encode 8x8 block
//                         gpujpeg_huffman_gpu_decoder_decode_block(component_dc, block, table_offset, r_bit, r_bit_count, s_byte, s_byte_idx, d_byte, d_byte_chunk_count);
//                     }
//                 }
//             }
//         }
    }
}

/**
 * Setup of one Huffman table entry for fast decoding.
 * @param bits  bits to extract one codeword from (first bit is bit #15, then #14, ... last is #0)
 * @param d_table_src  source (slow-decoding) table pointer
 * @param d_table_dest  destination (fast-decoding) table pointer
 */
__device__ void
gpujpeg_huffman_gpu_decoder_table_setup(
    const int bits, 
    const struct gpujpeg_table_huffman_decoder* const d_table_src,
    const int table_idx
) {
    // Decode one codeword from given bits to get following:
    //  - minimal number of bits actually needed to decode the codeword (up to 16 bits, 0 for invalid ones)
    //  - category ID represented by the codeword, consisting from:
    //      - number of run-length-coded preceding zeros (up to 16, or 63 for both special end-of block symbol or invalid codewords)
    //      - bit-size of the actual value of coefficient (up to 16, 0 for invalid ones)
    int code_nbits = 1, category_id = 0;
    
    // First, decode codeword length (This is per Figure F.16 in the JPEG spec.)
    int code_value = bits >> 15; // only single bit initially
    while ( code_value > d_table_src->maxcode[code_nbits] ) {
        code_value = bits >> (16 - ++code_nbits); // not enough to decide => try more bits
    }
    
    // With garbage input we may reach the sentinel value l = 17.
    if ( code_nbits > 16 ) {
        code_nbits = 0;
        // category ID remains 0 for invalid symbols from garbage input
    } else {
        category_id = d_table_src->huffval[d_table_src->valptr[code_nbits] + code_value - d_table_src->mincode[code_nbits]];
    }
    
    // decompose category number into 1 + number of run-length coded zeros and length of the value
    // (special category #0 contains all invalid codes and special end-of-block code -- all of those codes 
    // should terminate block decoding => use 64 run-length zeros and 0 value bits for such symbols)
    const int value_nbits = 0xF & category_id;
    const int rle_zero_count = category_id ? min(1 + (category_id >> 4), 64) : 64;
    
    // save all the info into the right place in the destination table
    const int packed_info = (rle_zero_count << 9) + (code_nbits << 4) + value_nbits;
    gpujpeg_huffman_gpu_decoder_tables_full[(table_idx << 16) + bits] = packed_info;
    
    // some threads also save entries into the quick table
    const int dest_idx_quick = bits >> (16 - QUICK_CHECK_BITS);
    if(bits == (dest_idx_quick << (16 - QUICK_CHECK_BITS))) {
        // save info also into the quick table if number of required bits is less than quick 
        // check bit count, otherwise put 0 there to indicate that full table lookup consultation is needed
        gpujpeg_huffman_gpu_decoder_tables_quick[(table_idx << QUICK_CHECK_BITS) + dest_idx_quick] = code_nbits <= QUICK_CHECK_BITS ? packed_info : 0;
    }
}

/**
 * Huffman decoder table setup kernel
 * (Based on the original table, this kernel prepares another table, which is more suitable for fast decoding.)
 */
__global__ void
gpujpeg_huffman_decoder_table_kernel(
                const struct gpujpeg_table_huffman_decoder* const d_table_y_dc,
                const struct gpujpeg_table_huffman_decoder* const d_table_y_ac,
                const struct gpujpeg_table_huffman_decoder* const d_table_cbcr_dc,
                const struct gpujpeg_table_huffman_decoder* const d_table_cbcr_ac
) {
    // Each thread uses all 4 Huffman tables to "decode" one symbol from its unique 16bits.
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    gpujpeg_huffman_gpu_decoder_table_setup(idx, d_table_y_dc, 0);
    gpujpeg_huffman_gpu_decoder_table_setup(idx, d_table_y_ac, 1);
    gpujpeg_huffman_gpu_decoder_table_setup(idx, d_table_cbcr_dc, 2);
    gpujpeg_huffman_gpu_decoder_table_setup(idx, d_table_cbcr_ac, 3);
}

/** Documented at declaration */
int
gpujpeg_huffman_gpu_decoder_init()
{
    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        (const char*)gpujpeg_huffman_gpu_decoder_order_natural,
        gpujpeg_order_natural, 
        GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );
    gpujpeg_cuda_check_error("Huffman decoder init");
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_huffman_gpu_decoder_decode(struct gpujpeg_decoder* decoder)
{    
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    assert(coder->param.restart_interval > 0);
    
    int comp_count = 1;
    if ( coder->param.interleaved == 1 )
        comp_count = coder->param_image.comp_count;
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    
    // Number of decoder kernel threads per each threadblock
    enum { THREADS_PER_TBLOCK = 192 };
    
    // Configure more Shared memory for both kernels
    cudaFuncSetCacheConfig(gpujpeg_huffman_decoder_table_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_decoder_decode_kernel<true, THREADS_PER_TBLOCK>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_decoder_decode_kernel<false, THREADS_PER_TBLOCK>, cudaFuncCachePreferShared);
    
    // Setup GPU tables (one thread for each of 65536 entries)
    gpujpeg_huffman_decoder_table_kernel<<<256, 256>>>(
        decoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC],
        decoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC],
        decoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC],
        decoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC]
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Huffman decoder table setup failed");

    // Get pointer to quick decoding table in device memory
    void * d_src_ptr = 0;
    cudaGetSymbolAddress(&d_src_ptr, gpujpeg_huffman_gpu_decoder_tables_quick);
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Huffman decoder table address lookup failed");
    
    // Copy quick decoding table into constant memory
    cudaMemcpyToSymbol(
        gpujpeg_huffman_gpu_decoder_tables_quick_const,
        d_src_ptr,
        sizeof(*gpujpeg_huffman_gpu_decoder_tables_quick) * QUICK_TABLE_ITEMS,
        0,
        cudaMemcpyDeviceToDevice
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Huffman decoder table copy failed");
    
    // Run decoding kernel
    dim3 thread(THREADS_PER_TBLOCK);
    dim3 grid(gpujpeg_div_and_round_up(decoder->segment_count, THREADS_PER_TBLOCK));
    if(comp_count == 1) {
        gpujpeg_huffman_decoder_decode_kernel<true, THREADS_PER_TBLOCK><<<grid, thread>>>(
            coder->d_component, 
            coder->d_segment, 
            comp_count,
            decoder->segment_count,
            coder->d_data_compressed,
            coder->d_block_list,
            coder->d_data_quantized
        );
    } else {
        gpujpeg_huffman_decoder_decode_kernel<false, THREADS_PER_TBLOCK><<<grid, thread>>>(
            coder->d_component, 
            coder->d_segment, 
            comp_count,
            decoder->segment_count,
            coder->d_data_compressed,
            coder->d_block_list,
            coder->d_data_quantized
        );
    }
    cudaError cuerr = cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Huffman decoding failed");
    
    return 0;
}
