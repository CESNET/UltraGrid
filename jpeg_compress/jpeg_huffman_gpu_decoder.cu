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
 
#include "jpeg_huffman_gpu_decoder.h"
#include "jpeg_format_type.h"
#include "jpeg_util.h"

/** Natural order in constant memory */
__constant__ int jpeg_huffman_gpu_decoder_order_natural[64];

/**
 * Fill more bit to current get buffer
 * 
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return void
 */
__device__ inline void
jpeg_huffman_gpu_decoder_decode_fill_bit_buffer(int & get_bits, int & get_buff, uint8_t* & data, int & data_size)
{
    while ( get_bits < 25 ) {
        //Are there some data?
        if( data_size > 0 ) { 
            // Attempt to read a byte
            //printf("read byte %X 0x%X\n", (int)data, (unsigned char)*data);
            unsigned char uc = *data++;
            data_size--;            

            // If it's 0xFF, check and discard stuffed zero byte
            if ( uc == 0xFF ) {
                do {
                    //printf("read byte %X 0x%X\n", (int)data, (unsigned char)*data);
                    uc = *data++;
                    data_size--;
                } while ( uc == 0xFF );

                if ( uc == 0 ) {
                    // Found FF/00, which represents an FF data byte
                    uc = 0xFF;
                } else {                
                    // There should be enough bits still left in the data segment;
                    // if so, just break out of the outer while loop.
                    //if (m_nGetBits >= nbits)
                    if ( get_bits >= 0 )
                        break;
                }
            }

            get_buff = (get_buff << 8) | ((int) uc);
            get_bits += 8;            
        }
        else
            break;
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
__device__ inline int
jpeg_huffman_gpu_decoder_get_bits(int nbits, int & get_bits, int & get_buff, uint8_t* & data, int & data_size)
{
    //we should read nbits bits to get next data
    if( get_bits < nbits )
        jpeg_huffman_gpu_decoder_decode_fill_bit_buffer(get_bits, get_buff, data, data_size);
    get_bits -= nbits;
    return (int)(get_buff >> get_bits) & ((1 << nbits) - 1);
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
jpeg_huffman_gpu_decoder_decode_special_decode(struct jpeg_table_huffman_decoder* table, int min_bits, int & get_bits, int & get_buff, uint8_t* & data, int & data_size)
{
    // HUFF_DECODE has determined that the code is at least min_bits
    // bits long, so fetch that many bits in one swoop.
    int code = jpeg_huffman_gpu_decoder_get_bits(min_bits, get_bits, get_buff, data, data_size);

    // Collect the rest of the Huffman code one bit at a time.
    // This is per Figure F.16 in the JPEG spec.
    int l = min_bits;
    while ( code > table->maxcode[l] ) {
        code <<= 1;
        code |= jpeg_huffman_gpu_decoder_get_bits(1, get_bits, get_buff, data, data_size);
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
 * @param category
 * @param offset
 * @return int
 */
__device__ inline int
jpeg_huffman_gpu_decoder_value_from_category(int category, int offset)
{
    // Method 1: 
    // On some machines, a shift and add will be faster than a table lookup.
    // #define HUFF_EXTEND(x,s) \
    // ((x)< (1<<((s)-1)) ? (x) + (((-1)<<(s)) + 1) : (x)) 

    // Method 2: Table lookup
    // If (offset < half[category]), then value is below zero
    // Otherwise, value is above zero, and just the offset 
    // entry n is 2**(n-1)
    const int half[16] =    { 
        0x0000, 0x0001, 0x0002, 0x0004, 0x0008, 0x0010, 0x0020, 0x0040, 
        0x0080, 0x0100, 0x0200, 0x0400, 0x0800, 0x1000, 0x2000, 0x4000
    };

    //start[i] is the starting value in this category; surely it is below zero
    // entry n is (-1 << n) + 1
    const int start[16] = { 
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
 * @param table
 * @param get_bits
 * @param get_buff
 * @param data
 * @param data_size
 * @return int
 */
__device__ inline int
jpeg_huffman_gpu_decoder_get_category(int & get_bits, int & get_buff, uint8_t* & data, int & data_size, struct jpeg_table_huffman_decoder* table)
{
    // If left bits < 8, we should get more data
    if ( get_bits < 8 )
        jpeg_huffman_gpu_decoder_decode_fill_bit_buffer(get_bits, get_buff, data, data_size);

    // Call special process if data finished; min bits is 1
    if( get_bits < 8 )
        return jpeg_huffman_gpu_decoder_decode_special_decode(table, 1, get_bits, get_buff, data, data_size);

    // Peek the first valid byte    
    int look = ((get_buff >> (get_bits - 8)) & 0xFF);
    int nb = table->look_nbits[look];

    if ( nb ) { 
        get_bits -= nb;
        return table->look_sym[look]; 
    } else {
        //Decode long codes with length >= 9
        return jpeg_huffman_gpu_decoder_decode_special_decode(table, 9, get_bits, get_buff, data, data_size);
    }
}

/**
 * Decode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ inline int
jpeg_huffman_gpu_decoder_decode_block(int & dc, int & get_bits, int & get_buff, uint8_t* & data, int & data_size, int16_t* data_output, 
                                      struct jpeg_table_huffman_decoder* table_dc, struct jpeg_table_huffman_decoder* table_ac)
{
    // Section F.2.2.1: decode the DC coefficient difference
    // get dc category number, s
    int s = jpeg_huffman_gpu_decoder_get_category(get_bits, get_buff, data, data_size, table_dc);
    if ( s ) {
        // Get offset in this dc category
        int r = jpeg_huffman_gpu_decoder_get_bits(s, get_bits, get_buff, data, data_size);
        // Get dc difference value
        s = jpeg_huffman_gpu_decoder_value_from_category(s, r);
    }

    // Convert DC difference to actual value, update last_dc_val
    s += dc;
    dc = s;

    // Output the DC coefficient (assumes jpeg_natural_order[0] = 0)
    data_output[0] = s;
    
    // Section F.2.2.2: decode the AC coefficients
    // Since zeroes are skipped, output area must be cleared beforehand
    for ( int k = 1; k < 64; k++ ) {
        // s: (run, category)
        int s = jpeg_huffman_gpu_decoder_get_category(get_bits, get_buff, data, data_size, table_ac);
        // r: run length for ac zero, 0 <= r < 16
        int r = s >> 4;
        // s: category for this non-zero ac
        s &= 15;
        if ( s ) {
            //    k: position for next non-zero ac
            k += r;
            //    r: offset in this ac category
            r = jpeg_huffman_gpu_decoder_get_bits(s, get_bits, get_buff, data, data_size);
            //    s: ac value
            s = jpeg_huffman_gpu_decoder_value_from_category(s, r);

            data_output[jpeg_huffman_gpu_decoder_order_natural[k]] = s;            
        } else {
            // s = 0, means ac value is 0 ? Only if r = 15.  
            //means all the left ac are zero
            if ( r != 15 )
                break;
            k += 15;
        }
    }
    
    /*printf("GPU Decode Block\n");
    for ( int y = 0; y < 8; y++ ) {
        for ( int x = 0; x < 8; x++ ) {
            printf("%4d ", data_output[y * 8 + x]);
        }
        printf("\n");
    }*/
    
    return 0;
}

/**
 * Huffman decoder kernel
 * 
 * @return void
 */
__global__ void
jpeg_huffman_decoder_decode_kernel(
    int restart_interval,
    int comp_block_count,
    int comp_segment_count,
    int segment_count,    
    uint8_t* d_data_scan,
    int data_scan_size,
    int* d_data_scan_index,
    int16_t* d_data_decompressed,
    struct jpeg_table_huffman_decoder* d_table_y_dc,
    struct jpeg_table_huffman_decoder* d_table_y_ac,
    struct jpeg_table_huffman_decoder* d_table_cbcr_dc,
    struct jpeg_table_huffman_decoder* d_table_cbcr_ac
)
{
    int comp_index = blockIdx.y;
    int comp_segment_index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( comp_segment_index >= comp_segment_count )
        return;
    int segment_index = comp_index * comp_segment_count + comp_segment_index;
    if ( segment_index >= segment_count )
        return;
    
    // Get huffman tables
    struct jpeg_table_huffman_decoder* d_table_dc = NULL;
    struct jpeg_table_huffman_decoder* d_table_ac = NULL;
    if ( comp_index == 0 ) {
        d_table_dc = d_table_y_dc;
        d_table_ac = d_table_y_ac;
    } else {
        d_table_dc = d_table_cbcr_dc;
        d_table_ac = d_table_cbcr_ac;
    }
    
    // Start coder
    int get_buff = 0;
    int get_bits = 0;
    int dc = 0;
    
    // Prepare data pointer and its size
    int data_index = d_data_scan_index[segment_index];
    uint8_t* data = &d_data_scan[data_index];
    int data_size = 0;
    if ( (segment_index + 1) >= segment_count )
        data_size = data_scan_size - data_index;
    else
        data_size = d_data_scan_index[segment_index + 1] - data_index;
    
    // Encode blocks in restart segment
    int comp_block_index = comp_segment_index * restart_interval;
    for ( int block = 0; block < restart_interval; block++ ) {
        // Skip blocks out of memory
        if ( comp_block_index >= comp_block_count )
            break;
        // Decode block
        int data_index = (comp_block_count * comp_index + comp_block_index) * JPEG_BLOCK_SIZE * JPEG_BLOCK_SIZE;
        jpeg_huffman_gpu_decoder_decode_block(
            dc,
            get_bits,
            get_buff,
            data,
            data_size,
            &d_data_decompressed[data_index],
            d_table_dc,
            d_table_ac
        );
        comp_block_index++;
    }
}

/** Documented at declaration */
int
jpeg_huffman_gpu_decoder_init()
{
    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        "jpeg_huffman_gpu_decoder_order_natural",
        jpeg_order_natural, 
        64 * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );
    cudaCheckError("Huffman decoder init");
    
    return 0;
}

/** Documented at declaration */
int
jpeg_huffman_gpu_decoder_decode(struct jpeg_decoder* decoder)
{    
    assert(decoder->restart_interval > 0);
    
    int comp_block_cx = (decoder->param_image.width + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE;
    int comp_block_cy = (decoder->param_image.height + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE;
    int comp_block_count = comp_block_cx * comp_block_cy;
    int comp_segment_count = divAndRoundUp(comp_block_count, decoder->restart_interval);
    
    // Run kernel
    dim3 thread(32);
    dim3 grid(divAndRoundUp(comp_segment_count, thread.x), decoder->param_image.comp_count);
    jpeg_huffman_decoder_decode_kernel<<<grid, thread>>>(
        decoder->restart_interval,
        comp_block_count, 
        comp_segment_count,
        decoder->segment_count,
        decoder->d_data_scan,
        decoder->data_scan_size,
        decoder->d_data_scan_index,
        decoder->d_data_quantized,
        decoder->d_table_huffman[JPEG_COMPONENT_LUMINANCE][JPEG_HUFFMAN_DC],
        decoder->d_table_huffman[JPEG_COMPONENT_LUMINANCE][JPEG_HUFFMAN_AC],
        decoder->d_table_huffman[JPEG_COMPONENT_CHROMINANCE][JPEG_HUFFMAN_DC],
        decoder->d_table_huffman[JPEG_COMPONENT_CHROMINANCE][JPEG_HUFFMAN_AC]
    );
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Huffman decoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
    
    return 0;
}