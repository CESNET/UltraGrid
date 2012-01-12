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
 
#include "gpujpeg_huffman_gpu_encoder.h"
#include "gpujpeg_util.h"

#ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
/** Allocate huffman tables in constant memory */
__constant__ struct gpujpeg_table_huffman_encoder gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
/** Pass huffman tables to encoder */
extern struct gpujpeg_table_huffman_encoder (*gpujpeg_encoder_table_huffman)[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT] = &gpujpeg_huffman_gpu_encoder_table_huffman;
#endif

/** Natural order in constant memory */
__constant__ int gpujpeg_huffman_gpu_encoder_order_natural[64];

/**
 * Write one byte to compressed data
 * 
 * @param data_compressed  Data compressed
 * @param value  Byte value to write
 * @return void
 */
#define gpujpeg_huffman_gpu_encoder_emit_byte(data_compressed, value) { \
    *data_compressed = (uint8_t)(value); \
    data_compressed++; }
    
/**
 * Write two bytes to compressed data
 * 
 * @param data_compressed  Data compressed
 * @param value  Two-byte value to write
 * @return void
 */
#define gpujpeg_huffman_gpu_encoder_emit_2byte(data_compressed, value) { \
    *data_compressed = (uint8_t)(((value) >> 8) & 0xFF); \
    data_compressed++; \
    *data_compressed = (uint8_t)((value) & 0xFF); \
    data_compressed++; }
    
/**
 * Write marker to compressed data
 * 
 * @param data_compressed  Data compressed
 * @oaran marker  Marker to write (JPEG_MARKER_...)
 * @return void
 */
#define gpujpeg_huffman_gpu_encoder_marker(data_compressed, marker) { \
    *data_compressed = 0xFF;\
    data_compressed++; \
    *data_compressed = (uint8_t)(marker); \
    data_compressed++; }

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
__device__ inline int
gpujpeg_huffman_gpu_encoder_emit_bits(unsigned int code, int size, int & put_value, int & put_bits, uint8_t* & data_compressed)
{
	// This routine is heavily used, so it's worth coding tightly
	int _put_buffer = (int)code;
	int _put_bits = put_bits;
	// If size is 0, caller used an invalid Huffman table entry
	if ( size == 0 )
		return -1;
    // Mask off any extra bits in code
	_put_buffer &= (((int)1) << size) - 1; 
    // New number of bits in buffer
	_put_bits += size;					
    // Align incoming bits
	_put_buffer <<= 24 - _put_bits;		
    // And merge with old buffer contents
	_put_buffer |= put_value;    
	// If there are more than 8 bits, write it out
	unsigned char uc;
	while ( _put_bits >= 8 ) {
		// Write one byte out
		uc = (unsigned char) ((_put_buffer >> 16) & 0xFF);
		gpujpeg_huffman_gpu_encoder_emit_byte(data_compressed, uc);
        // If need to stuff a zero byte
		if ( uc == 0xFF ) {  
            // Write zero byte out
			gpujpeg_huffman_gpu_encoder_emit_byte(data_compressed, 0);
		}
		_put_buffer <<= 8;
		_put_bits -= 8;
	}
    // update state variables
	put_value = _put_buffer; 
	put_bits = _put_bits;
	return 0;
}

/**
 * Emit left bits
 * 
 * @param coder  Huffman coder structure
 * @return void
 */
__device__ inline void
gpujpeg_huffman_gpu_encoder_emit_left_bits(int & put_value, int & put_bits, uint8_t* & data_compressed)
{
    // Fill 7 bits with ones
	if ( gpujpeg_huffman_gpu_encoder_emit_bits(0x7F, 7, put_value, put_bits, data_compressed) != 0 )
		return;
	
	//unsigned char uc = (unsigned char) ((put_value >> 16) & 0xFF);
    // Write one byte out
	//gpujpeg_huffman_gpu_encoder_emit_byte(data_compressed, uc);
    
	put_value = 0; 
	put_bits = 0;
}

/**
 * Encode one 8x8 block
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ int
gpujpeg_huffman_gpu_encoder_encode_block(int & put_value, int & put_bits, int & dc, int16_t* data, uint8_t* & data_compressed, 
    struct gpujpeg_table_huffman_encoder* d_table_dc, struct gpujpeg_table_huffman_encoder* d_table_ac)
{
	// Encode the DC coefficient difference per section F.1.2.1
	int temp = data[0] - dc;
	dc = data[0];

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

	//	Write category number
	if ( gpujpeg_huffman_gpu_encoder_emit_bits(d_table_dc->code[nbits], d_table_dc->size[nbits], put_value, put_bits, data_compressed) != 0 ) {
		return -1;
    }

	//	Write category offset (EmitBits rejects calls with size 0)
	if ( nbits ) {
		if ( gpujpeg_huffman_gpu_encoder_emit_bits((unsigned int) temp2, nbits, put_value, put_bits, data_compressed) != 0 )
			return -1;
	}
    
	// Encode the AC coefficients per section F.1.2.2 (r = run length of zeros)
	int r = 0;
	for ( int k = 1; k < 64; k++ ) 
	{
		if ( (temp = data[gpujpeg_huffman_gpu_encoder_order_natural[k]]) == 0 ) {
			r++;
		}
		else {
			// If run length > 15, must emit special run-length-16 codes (0xF0)
			while ( r > 15 ) {
				if ( gpujpeg_huffman_gpu_encoder_emit_bits(d_table_ac->code[0xF0], d_table_ac->size[0xF0], put_value, put_bits, data_compressed) != 0 )
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
			if ( gpujpeg_huffman_gpu_encoder_emit_bits(d_table_ac->code[i], d_table_ac->size[i], put_value, put_bits, data_compressed) != 0 )
				return -1;

			// Write Category offset
			if ( gpujpeg_huffman_gpu_encoder_emit_bits((unsigned int) temp2, nbits, put_value, put_bits, data_compressed) != 0 )
				return -1;

			r = 0;
		}
	}

	// If all the left coefs were zero, emit an end-of-block code
	if ( r > 0 ) {
		if ( gpujpeg_huffman_gpu_encoder_emit_bits(d_table_ac->code[0], d_table_ac->size[0], put_value, put_bits, data_compressed) != 0 )
			return -1;
	}

	return 0;
}

/**
 * Huffman encoder kernel
 * 
 * @return void
 */
__global__ void
gpujpeg_huffman_encoder_encode_kernel(
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int comp_count,
    int segment_count, 
    uint8_t* d_data_compressed
#ifndef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
    ,struct gpujpeg_table_huffman_encoder* d_table_y_dc
    ,struct gpujpeg_table_huffman_encoder* d_table_y_ac
    ,struct gpujpeg_table_huffman_encoder* d_table_cbcr_dc
    ,struct gpujpeg_table_huffman_encoder* d_table_cbcr_ac
#endif
)
{    
#ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
    // Get huffman tables from constant memory
    struct gpujpeg_table_huffman_encoder* d_table_y_dc = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC];
    struct gpujpeg_table_huffman_encoder* d_table_y_ac = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC];
    struct gpujpeg_table_huffman_encoder* d_table_cbcr_dc = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC];
    struct gpujpeg_table_huffman_encoder* d_table_cbcr_ac = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC];
#endif
    
    int segment_index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( segment_index >= segment_count )
        return;
    
    struct gpujpeg_segment* segment = &d_segment[segment_index];
    
    // Initialize huffman coder
    int put_value = 0;
    int put_bits = 0;
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
        dc[comp] = 0;
    
    // Prepare data pointers
    uint8_t* data_compressed = &d_data_compressed[segment->data_compressed_index];
    uint8_t* data_compressed_start = data_compressed;
    
    // Non-interleaving mode
    if ( comp_count == 1 ) {
        int segment_index = segment->scan_segment_index;
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            // Get component for current scan
            struct gpujpeg_component* component = &d_component[segment->scan_index];
     
            // Get component data for MCU
            int16_t* block = &component->d_data_quantized[(segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size];
            
            // Get coder parameters
            int & component_dc = dc[segment->scan_index];
            
            // Get huffman tables
            struct gpujpeg_table_huffman_encoder* d_table_dc = NULL;
            struct gpujpeg_table_huffman_encoder* d_table_ac = NULL;
            if ( component->type == GPUJPEG_COMPONENT_LUMINANCE ) {
                d_table_dc = d_table_y_dc;
                d_table_ac = d_table_y_ac;
            } else {
                d_table_dc = d_table_cbcr_dc;
                d_table_ac = d_table_cbcr_ac;
            }
            
            // Encode 8x8 block
            if ( gpujpeg_huffman_gpu_encoder_encode_block(put_value, put_bits, component_dc, block, data_compressed, d_table_dc, d_table_ac) != 0 )
                break;
        } 
    }
    // Interleaving mode
    else {
        int segment_index = segment->scan_segment_index;
        // Encode MCUs in segment
        for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
            //assert(segment->scan_index == 0);
            for ( int comp = 0; comp < comp_count; comp++ ) {
                struct gpujpeg_component* component = &d_component[comp];

                // Prepare mcu indexes
                int mcu_index_x = (segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
                int mcu_index_y = (segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
                // Compute base data index
                int data_index_base = mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                
                // For all vertical 8x8 blocks
                for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                    // Compute base row data index
                    int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                    // For all horizontal 8x8 blocks
                    for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                        // Compute 8x8 block data index
                        int data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                        
                        // Get component data for MCU
                        int16_t* block = &component->d_data_quantized[data_index];
                        
                        // Get coder parameters
                        int & component_dc = dc[comp];
            
                        // Get huffman tables
                        struct gpujpeg_table_huffman_encoder* d_table_dc = NULL;
                        struct gpujpeg_table_huffman_encoder* d_table_ac = NULL;
                        if ( component->type == GPUJPEG_COMPONENT_LUMINANCE ) {
                            d_table_dc = d_table_y_dc;
                            d_table_ac = d_table_y_ac;
                        } else {
                            d_table_dc = d_table_cbcr_dc;
                            d_table_ac = d_table_cbcr_ac;
                        }
                        
                        // Encode 8x8 block
                        gpujpeg_huffman_gpu_encoder_encode_block(put_value, put_bits, component_dc, block, data_compressed, d_table_dc, d_table_ac);
                    }
                }
            }
        }
    }
    
    // Emit left bits
    if ( put_bits > 0 )
        gpujpeg_huffman_gpu_encoder_emit_left_bits(put_value, put_bits, data_compressed);
                        
    // Output restart marker
    int restart_marker = GPUJPEG_MARKER_RST0 + (segment->scan_segment_index & 0x7);
    gpujpeg_huffman_gpu_encoder_marker(data_compressed, restart_marker);
                
    // Set compressed size
    segment->data_compressed_size = data_compressed - data_compressed_start;
}

/** Documented at declaration */
int
gpujpeg_huffman_gpu_encoder_init()
{
    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        gpujpeg_huffman_gpu_encoder_order_natural,
        gpujpeg_order_natural, 
        64 * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );
    gpujpeg_cuda_check_error("Huffman encoder init");
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_huffman_gpu_encoder_encode(struct gpujpeg_encoder* encoder)
{    
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    
    assert(coder->param.restart_interval > 0);
    
    int comp_count = 1;
    if ( coder->param.interleaved == 1 )
        comp_count = coder->param_image.comp_count;
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
            
    // Run kernel
    dim3 thread(32);
    dim3 grid(gpujpeg_div_and_round_up(coder->segment_count, thread.x));
    gpujpeg_huffman_encoder_encode_kernel<<<grid, thread>>>(
        coder->d_component, 
        coder->d_segment, 
        comp_count,
        coder->segment_count, 
        coder->d_data_compressed
    #ifndef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC]
        ,encoder->d_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC]
    #endif
    );
    cudaError cuerr = cudaThreadSynchronize();
    if ( cuerr != cudaSuccess ) {
        fprintf(stderr, "Huffman encoding failed: %s!\n", cudaGetErrorString(cuerr));
        return -1;
    }
    
    return 0;
}
