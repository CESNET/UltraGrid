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
#include <libgpujpeg/gpujpeg_util.h>

#define WARPS_NUM 8



/** Natural order in constant memory */
__constant__ int gpujpeg_huffman_gpu_encoder_order_natural[GPUJPEG_ORDER_NATURAL_SIZE];

/** Size of occupied part of output buffer */
__device__ unsigned int gpujpeg_huffman_output_byte_count;

/**
 * Huffman coding tables in constant memory - each has 257 items (256 + 1 extra)
 * There are are 4 of them - one after another, in following order:
 *    - luminance (Y) AC
 *    - luminance (Y) DC
 *    - chroma (cb/cr) AC
 *    - chroma (cb/cr) DC
 */
__device__ uint32_t gpujpeg_huffman_gpu_lut[(256 + 1) * 4];

/**
 * Value decomposition in constant memory (input range from -4096 to 4095  ... both inclusive)
 * Mapping from coefficient value into the code for the value ind its bit size.
 */
__device__ unsigned int gpujpeg_huffman_value_decomposition[8 * 1024];

/**
 * Initializes coefficient decomposition table in global memory.  (CC >= 2.0)
 * Output table is a mapping from some value into its code and bit size.
 */
__global__ static void
gpujpeg_huffman_gpu_encoder_value_decomposition_init_kernel() {
    // fetch some value
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    const int value = tid - 4096;
    
    // decompose it
    unsigned int value_code = value;
    int absolute = value;
    if ( value < 0 ) {
        // valu eis now absolute value of input
        absolute = -absolute;
        // For a negative input, want temp2 = bitwise complement of abs(input)
        // This code assumes we are on a two's complement machine
        value_code--;
    }

    // Find the number of bits needed for the magnitude of the coefficient
    unsigned int value_nbits = 0;
    while ( absolute ) {
        value_nbits++;
        absolute >>= 1;
    }
    
    // save result packed into unsigned int (value bits are left aligned in MSBs and size is right aligned in LSBs)
    gpujpeg_huffman_value_decomposition[tid] = value_nbits | (value_code << (32 - value_nbits));
}

#if __CUDA_ARCH__ >= 200
/**
 * Adds up to 32 bits at once into ouptut buffer, applying byte stuffing.
 * Codeword value must be aligned to left (most significant bits). (CC >= 2.0)
 */
__device__ static void 
gpujpeg_huffman_gpu_encoder_emit_bits(unsigned int & remaining_bits, int & byte_count, int & bit_count, uint8_t * const out_ptr, const unsigned int packed_code_word)
{
    // decompose packed codeword into the msb-aligned value and bit-length of the value
    const unsigned int code_word = packed_code_word & ~31;
    const unsigned int code_bit_size = packed_code_word & 31;
    
    // concatenate with remaining bits
    remaining_bits |= code_word >> bit_count;
    bit_count += code_bit_size;
    
    // flush some bytes if have more than 8 bits
    if (bit_count >= 8) {
        do {
            const unsigned int out_byte = remaining_bits >> 24;
            out_ptr[byte_count++] = out_byte;
            if(0xff == out_byte) {
                // keep zero byte after each 0xFF (buffer is expected to be zeroed)
                out_ptr[byte_count++] = 0;
            }
            
            remaining_bits <<= 8;
            bit_count -= 8;
        } while (bit_count >= 8);
        
        // keep only remaining bits in the buffer
        remaining_bits = code_word << (code_bit_size - bit_count);
        remaining_bits &= 0xfffffffe << (31 - bit_count);
    }
}

/**
 * Given some huffman table offset, RLE zero count and coefficient value, 
 * this returns huffman codeword for the value (packed in 27 MSBs) 
 * together with its bit size (in 5 LSBs).  (CC >= 2.0)
 */
__device__ static unsigned int
gpujpeg_huffman_gpu_encode_value(const int preceding_zero_count, const int coefficient,
                                 const int huffman_lut_offset)
{
    // value bits are in MSBs (left aligned) and bit size of the value is in LSBs (right aligned)
    const unsigned int packed_value = gpujpeg_huffman_value_decomposition[4096 + coefficient];
    
    // decompose value info into upshifted value and value's bit size
    const int value_nbits = packed_value & 0xf;
    const unsigned int value_code = packed_value & ~0xf;
    
    // find prefix of the codeword and size of the prefix
    const int huffman_lut_idx = huffman_lut_offset + preceding_zero_count * 16 + value_nbits;
    const unsigned int packed_prefix = gpujpeg_huffman_gpu_lut[huffman_lut_idx];
    const unsigned int prefix_nbits = packed_prefix & 31;
    
    // compose packed codeword with its size
    return (packed_prefix + value_nbits) | (value_code >> prefix_nbits);
}

/**
 * Flush remaining codewords from buffer in shared memory to global memory output buffer.  (CC >= 2.0)
 */
__device__ static void
gpujpeg_huffman_gpu_encoder_flush_codewords(unsigned int * const s_out, unsigned int * &data_compressed, int & remaining_codewords, const int tid) {
    // this works for up to 4 * 32 remaining codewords
    if(remaining_codewords) {
        // pad remaining codewords with extra zero-sized codewords, not to have to use special case in serialization kernel, which saves 4 codewords at once
        s_out[remaining_codewords + tid] = 0;
        
        // save all remaining codewords at once (together with some zero sized padding codewords)
        *((uint4*)data_compressed) = ((uint4*)s_out)[tid];
        
        // update codeword counter
        data_compressed += remaining_codewords;
        remaining_codewords = 0;
    }
}

/**
 * Encode one 8x8 block  (CC >= 2.0)
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ static int
gpujpeg_huffman_gpu_encoder_encode_block(const int16_t * block, unsigned int * &data_compressed, unsigned int * const s_out,
                int & remaining_codewords, const int last_dc_idx, int tid, const int huffman_lut_offset)
{
    // each thread loads a pair of values (pair after zigzag reordering)
    const int load_idx = tid * 2;
    int in_even = block[gpujpeg_huffman_gpu_encoder_order_natural[load_idx]];
    const int in_odd = block[gpujpeg_huffman_gpu_encoder_order_natural[load_idx + 1]];
    
    // compute preceding zero count for even coefficient (actually compute the count multiplied by 16)
    const unsigned int nonzero_mask = (1 << tid) - 1;
    const unsigned int nonzero_bitmap_0 = 1 | __ballot(in_even);  // DC is always treated as nonzero
    const unsigned int nonzero_bitmap_1 = __ballot(in_odd);
    const unsigned int nonzero_bitmap_pairs = nonzero_bitmap_0 | nonzero_bitmap_1;
    
    const int zero_pair_count = __clz(nonzero_bitmap_pairs & nonzero_mask);
    int zeros_before_even = 2 * (zero_pair_count + tid - 32);
    if((0x80000000 >> zero_pair_count) > (nonzero_bitmap_1 & nonzero_mask)) {
        zeros_before_even += 1;
    }
    
    // true if any nonzero pixel follows thread's odd pixel
    const bool nonzero_follows = nonzero_bitmap_pairs & ~nonzero_mask;
    
    // count of consecutive zeros before odd value (either one more than 
    // even if even is zero or none if even value itself is nonzero)
    // (the count is actually multiplied by 16)
    int zeros_before_odd = in_even || !tid ? 0 : zeros_before_even + 1;
    
    // clear zero counts if no nonzero pixel follows (so that no 16-zero symbols will be emited)
    // otherwise only trim extra bits from the counts of following zeros
    const int zero_count_mask = nonzero_follows ? 0xF : 0;
    zeros_before_even &= zero_count_mask;
    zeros_before_odd &= zero_count_mask;
    
    // pointer to LUT for encoding thread's even value 
    // (only thread #0 uses DC table, others use AC table)
    int even_lut_offset = huffman_lut_offset;
    
    // first thread handles special DC coefficient
    if(0 == tid) {
        // first thread uses DC part of the table for its even value
        even_lut_offset += 256 + 1;
        
        // update last DC coefficient (saved at the special place at the end of the shared bufer)
        const int original_in_even = in_even;
        in_even -= ((int*)s_out)[last_dc_idx];
        ((int*)s_out)[last_dc_idx] = original_in_even;
    }
    
    // last thread handles special block-termination symbol
    if(0 == ((tid ^ 31) | in_odd)) {
        // this causes selection of huffman symbol at index 256 (which contains the termination symbol)
        zeros_before_odd = 16;
    }
    
    // each thread gets codeword for its two pixels
    unsigned int even_code = gpujpeg_huffman_gpu_encode_value(zeros_before_even, in_even, even_lut_offset);
    unsigned int odd_code = gpujpeg_huffman_gpu_encode_value(zeros_before_odd, in_odd, huffman_lut_offset);
            
    // concatenate both codewords into one if they are short enough
    const unsigned int even_code_size = even_code & 31;
    const unsigned int odd_code_size = odd_code & 31;
    const unsigned int total_size = even_code_size + odd_code_size;
    if(total_size <= 27) {
        even_code = total_size | ((odd_code & ~31) >> even_code_size) | (even_code & ~31);
        odd_code = 0;
    }
    
    // each thread get number of preceding nonzero codewords and total number of nonzero codewords in this block
    const unsigned int even_codeword_presence = __ballot(even_code);
    const unsigned int odd_codeword_presence = __ballot(odd_code);
    const int codeword_offset = __popc(nonzero_mask & even_codeword_presence)
                              + __popc(nonzero_mask & odd_codeword_presence);
    
    // each thread saves its values into temporary shared buffer
    if(even_code) {
        s_out[remaining_codewords + codeword_offset] = even_code;
        if(odd_code) {
            s_out[remaining_codewords + codeword_offset + 1] = odd_code;
        }
    }
    
    // advance count of codewords in shared memory buffer
    remaining_codewords += __popc(odd_codeword_presence) + __popc(even_codeword_presence);
    
    // flush some codewords to global memory if there are too many of them in shared buffer
    const int flush_count = 32 * 4; // = half of the buffer
    if(remaining_codewords > flush_count) {
        // move first half of the buffer into output buffer in global memory and update output pointer
        *((uint4*)data_compressed) = ((uint4*)s_out)[tid];
        data_compressed += flush_count;
        
        // shift remaining codewords to begin of the buffer and update their count
        ((uint4*)s_out)[tid] = ((uint4*)s_out)[flush_count / 4 + tid];  // 4 for 4 uints in uint4
        remaining_codewords -= flush_count;
    }
        
    // nothing to fail here
    return 0;
}
#endif // #if __CUDA_ARCH__ >= 200

/**
 * Huffman encoder kernel (For compute capability >= 2.0)
 * 
 * @return void
 */
template <bool CONTINUOUS_BLOCK_LIST>
#if __CUDA_ARCH__ >= 200
__launch_bounds__(WARPS_NUM * 32, 1024 / (WARPS_NUM * 32))
#endif 
__global__ static void
gpujpeg_huffman_encoder_encode_kernel_warp(
    struct gpujpeg_segment* d_segment,
    int segment_count, 
    uint8_t* d_data_compressed,
    const uint64_t* const d_block_list,
    int16_t* const d_data_quantized,
    struct gpujpeg_component* const d_component,
    const int comp_count
) {    
#if __CUDA_ARCH__ >= 200
    int warpidx = threadIdx.x >> 5;
    int tid = threadIdx.x & 31;

    __shared__ uint4 s_out_all[(64 + 1) * WARPS_NUM];
    unsigned int * s_out = (unsigned int*)(s_out_all + warpidx * (64 + 1));
    
    // Number of remaining codewords in shared buffer
    int remaining_codewords = 0;
    
    // Select Segment
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int segment_index = block_idx * WARPS_NUM + warpidx;
    
    // first thread initializes compact output size for next kernel
    if(0 == tid && 0 == warpidx && 0 == block_idx) {
        gpujpeg_huffman_output_byte_count = 0;
    }
    
    // stop if out of segment bounds
    if ( segment_index >= segment_count )
        return;
    struct gpujpeg_segment* segment = &d_segment[segment_index];
    
    // Initialize last DC coefficients
    if(tid < 3) {
        s_out[256 + tid] = 0;
    }
    
    // Prepare data pointers
    unsigned int * data_compressed = (unsigned int*)(d_data_compressed + segment->data_temp_index);
    unsigned int * data_compressed_start = data_compressed;
    
    // Pre-add thread ID to output pointer (it's allways used only with it)
    data_compressed += (tid * 4);
    
    // Encode all block in segment
    if(CONTINUOUS_BLOCK_LIST) {
        // Get component for current scan
        const struct gpujpeg_component* component = &d_component[segment->scan_index];
        
        // mcu size of the component
        const int comp_mcu_size = component->mcu_size;
        
        // Get component data for MCU (first block)
        const int16_t* block = component->d_data_quantized + (segment->scan_segment_index * component->segment_mcu_count) * comp_mcu_size;
        
        // Get huffman table offset
        const int huffman_table_offset = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0 : (256 + 1) * 2; // possibly skips luminance tables
        
        // Encode MCUs in segment
        for (int block_count = segment->mcu_count; block_count--;) {
            // Encode 8x8 block
            gpujpeg_huffman_gpu_encoder_encode_block(block, data_compressed, s_out, remaining_codewords, 256, tid, huffman_table_offset);
            
            // Advance to next block
            block += comp_mcu_size;
        }
    } else {
        // Pointer to segment's list of 8x8 blocks and their count
        const uint64_t* packed_block_info_ptr = d_block_list + segment->block_index_list_begin;
        
        // Encode all blocks
        for(int block_count = segment->block_count; block_count--;) {
            // Get pointer to next block input data and info about its color type
            const uint64_t packed_block_info = *(packed_block_info_ptr++);
            
            // Get coder parameters
            const int last_dc_idx = 256 + (packed_block_info & 0x7f);
            
            // Get offset to right part of huffman table
            const int huffman_table_offset = packed_block_info & 0x80 ? (256 + 1) * 2 : 0; // possibly skips luminance tables
            
            // Source data pointer
            int16_t* block = &d_data_quantized[packed_block_info >> 8];
                        
            // Encode 8x8 block
            gpujpeg_huffman_gpu_encoder_encode_block(block, data_compressed, s_out, remaining_codewords, last_dc_idx, tid, huffman_table_offset);
        }
    }

    // flush remaining codewords
    gpujpeg_huffman_gpu_encoder_flush_codewords(s_out, data_compressed, remaining_codewords, tid);
    
    // Set number of codewords.
    if (tid == 0 ) {
        segment->data_compressed_size = data_compressed - data_compressed_start;
    }
#endif // #if __CUDA_ARCH__ >= 200
}

#define SERIALIZATION_THREADS_PER_TBLOCK 192

/**
 * Codeword serialization kernel (CC >= 2.0).
 * 
 * @return void
 */
#if __CUDA_ARCH__ >= 200
__launch_bounds__(SERIALIZATION_THREADS_PER_TBLOCK, 1536 / SERIALIZATION_THREADS_PER_TBLOCK)
#endif
__global__ static void
gpujpeg_huffman_encoder_serialization_kernel(
    struct gpujpeg_segment* d_segment,
    int segment_count, 
    const uint8_t* const d_src,
    uint8_t* const d_dest
) {    
#if __CUDA_ARCH__ >= 200
    // Temp buffer for all threads of the threadblock
    __shared__ uint4 s_temp_all[2 * SERIALIZATION_THREADS_PER_TBLOCK];

    // Thread's 32 bytes in shared memory for output composition
    uint4 * const s_temp = s_temp_all + threadIdx.x * 2;
    
    // Select Segment
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    int segment_index = block_idx * SERIALIZATION_THREADS_PER_TBLOCK + threadIdx.x;
    if ( segment_index >= segment_count )
        return;
    
    // Thread's segment
    struct gpujpeg_segment* const segment = &d_segment[segment_index];
    
    // Input and output pointers
    const int data_offset = segment->data_temp_index;
    uint4 * const d_dest_stream_start = (uint4*)(d_dest + data_offset);
    uint4 * d_dest_stream = d_dest_stream_start;
    const uint4 * d_src_codewords = (uint4*)(d_src + data_offset);
    
    // number of bytes in the temp buffer, remaining bits and their count
    int byte_count = 0, bit_count = 0;
    unsigned int remaining_bits = 0;
    
    // "data_compressed_size" is now initialized to number of codewords to be serialized
    for(int cword_tuple_count = (segment->data_compressed_size + 3) >> 2; cword_tuple_count--; ) // reading 4 codewords at once
    {
        // read 4 codewords and advance input pointer to next ones
        const uint4 cwords = *(d_src_codewords++);
        
        // encode first pair of codewords
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, cwords.x);
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, cwords.y);
        
        // possibly flush output if have at least 16 bytes
        if(byte_count >= 16) {
            // write 16 bytes into destination buffer
            *(d_dest_stream++) = s_temp[0];
            
            // move remaining bytes to first half of the buffer
            s_temp[0] = s_temp[1];
            
            // update number of remaining bits
            byte_count -= 16;
        }
        
        // encode other two codewords
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, cwords.z);
        gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, cwords.w);
        
        // possibly flush output if have at least 16 bytes
        if(byte_count >= 16) {
            // write 16 bytes into destination buffer
            *(d_dest_stream++) = s_temp[0];
            
            // move remaining bytes to first half of the buffer
            s_temp[0] = s_temp[1];
            
            // update number of remaining bits
            byte_count -= 16;
        }
    }
    
    // Emit left bits
    gpujpeg_huffman_gpu_encoder_emit_bits(remaining_bits, byte_count, bit_count, (uint8_t*)s_temp, 0xfe000007);

    // Terminate codestream with restart marker
    ((uint8_t*)s_temp)[byte_count + 0] = 0xFF;
    ((uint8_t*)s_temp)[byte_count + 1] = GPUJPEG_MARKER_RST0 + (segment->scan_segment_index % 8);
    
    // flush remaining bytes
    d_dest_stream[0] = s_temp[0];
    d_dest_stream[1] = s_temp[1];
    
    // Set compressed size
    segment->data_compressed_size = (d_dest_stream - d_dest_stream_start) * 16 + byte_count + 2;
#endif // #if __CUDA_ARCH__ >= 200
}


/**
 * Huffman coder compact output allocation kernel - serially reserves 
 * some space for compressed output of segments in output buffer.
 * (For CC 1.0 - a workaround for missing atomic operations.)
 * 
 * Only single threadblock with 512 threads is launched.
 */
__global__ static void
gpujpeg_huffman_encoder_allocation_kernel (
    struct gpujpeg_segment* const d_segment,
    const int segment_count
) {
    // offsets of segments
    __shared__ unsigned int s_segment_offsets[512];
    
    // cumulative sum of bytes of all segments
    unsigned int total_byte_count = 0;
    
    // iterate over all segments
    const unsigned int segment_idx_end = (segment_count + 511) & ~511;
    for(unsigned int segment_idx = threadIdx.x; segment_idx < segment_idx_end; segment_idx += 512) {
        // all threads load byte sizes of their segments (rounded up to next multiple of 16 B) into the shared array 
        s_segment_offsets[threadIdx.x] = segment_idx < segment_count
                ? (d_segment[segment_idx].data_compressed_size + 15) & ~15
                : 0;
        
        // first thread runs a sort of serial prefix sum over the segment sizes to get their offsets
        __syncthreads();
        if(0 == threadIdx.x) {
            #pragma unroll 4
            for(int i = 0; i < 512; i++) {
                const unsigned int segment_size = s_segment_offsets[i];
                s_segment_offsets[i] = total_byte_count;
                total_byte_count += segment_size;
            }
        }
        __syncthreads();
        
        // all threads write offsets back into corresponding segment structures
        if(segment_idx < segment_count) {
            d_segment[segment_idx].data_compressed_index = s_segment_offsets[threadIdx.x];
        }
    }
    
    // first thread finally saves the total sum of bytes needed for compressed data
    if(threadIdx.x == 0) {
        gpujpeg_huffman_output_byte_count = total_byte_count;
    }
}


/**
 * Huffman coder output compaction kernel.
 * 
 * @return void
 */
__global__ static void
gpujpeg_huffman_encoder_compaction_kernel (
    struct gpujpeg_segment* const d_segment,
    const int segment_count, 
    const uint8_t* const d_src,
    uint8_t* const d_dest
) {
    // get some segment (size of threadblocks is 32 x N, so threadIdx.y is warp index)
    const int block_idx = blockIdx.x + blockIdx.y * gridDim.x;
    const int segment_idx = threadIdx.y + block_idx * blockDim.y;
    if(segment_idx >= segment_count) {
        return;
    }
    
    // temp variables for all warps
    __shared__ uint4* volatile s_out_ptrs[WARPS_NUM];
    
    // get info about the segment
    const unsigned int segment_byte_count = (d_segment[segment_idx].data_compressed_size + 15) & ~15;  // number of bytes rounded up to multiple of 16
    const unsigned int segment_in_offset = d_segment[segment_idx].data_temp_index;  // this should be aligned at least to 16byte boundary
    
    // first thread of each warp reserves space in output buffer
    if(0 == threadIdx.x) {
        // Either load precomputed output offset (for CC 1.0) or compute it now (for CCs with atomic operations)
        #if __CUDA_ARCH__ == 100
        const unsigned int segment_out_offset = d_segment[segment_idx].data_compressed_index;
        #else
        const unsigned int segment_out_offset = atomicAdd(&gpujpeg_huffman_output_byte_count, segment_byte_count);
        d_segment[segment_idx].data_compressed_index = segment_out_offset;
        #endif
        s_out_ptrs[threadIdx.y] = (uint4*)(d_dest + segment_out_offset);
    }
    
    // all threads read output buffer offset for their segment and prepare input and output pointers and number of copy iterations
    const uint4 * d_in = threadIdx.x + (uint4*)(d_src + segment_in_offset);
    uint4 * d_out = threadIdx.x + s_out_ptrs[threadIdx.y];
    unsigned int copy_iterations = segment_byte_count / 512; // 512 is number of bytes copied in each iteration (32 threads * 16 bytes per thread)
    
    // copy the data!
    while(copy_iterations--) {
        *d_out = *d_in;
        d_out += 32;
        d_in += 32;
    }
    
    // copy remaining bytes (less than 512 bytes)
    if((threadIdx.x * 16) < (segment_byte_count & 511)) {
        *d_out = *d_in;
    }
}

// Threadblock size for CC 1.x kernel
#define THREAD_BLOCK_SIZE 48

#ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
__constant__
#endif
/** Allocate huffman tables in constant memory */
__device__ struct gpujpeg_table_huffman_encoder gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];

/**
 * Write one byte to compressed data (CC 1.x)
 * 
 * @param data_compressed  Data compressed
 * @param value  Byte value to write
 * @return void
 */
#define gpujpeg_huffman_gpu_encoder_emit_byte(data_compressed, value) { \
    *data_compressed = (uint8_t)(value); \
    data_compressed++; }
    
/**
 * Write two bytes to compressed data (CC 1.x)
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
 * Write marker to compressed data (CC 1.x)
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
 * in put_buffer between calls, so 24 bits are sufficient. Version for CC 1.x
 * 
 * @param coder  Huffman coder structure
 * @param code  Huffman code
 * @param size  Size in bits of the Huffman code
 * @return void
 */
__device__ static int
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
 * Emit left bits (CC 1.x)
 * 
 * @param coder  Huffman coder structure
 * @return void
 */
__device__ static void
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
 * Encode one 8x8 block (for CC 1.x)
 *
 * @return 0 if succeeds, otherwise nonzero
 */
__device__ static int
gpujpeg_huffman_gpu_encoder_encode_block(int & put_value, int & put_bits, int & dc, int16_t* data, uint8_t* & data_compressed, 
    struct gpujpeg_table_huffman_encoder* d_table_dc, struct gpujpeg_table_huffman_encoder* d_table_ac)
{
    typedef uint64_t loading_t;
    const int loading_iteration_count = 64 * 2 / sizeof(loading_t);
    
    // Load block to shared memory
    __shared__ int16_t s_data[64 * THREAD_BLOCK_SIZE];
    for ( int i = 0; i < loading_iteration_count; i++ ) {
        ((loading_t*)s_data)[loading_iteration_count * threadIdx.x + i] = ((loading_t*)data)[i];
    }
    int data_start = 64 * threadIdx.x;

    // Encode the DC coefficient difference per section F.1.2.1
    int temp = s_data[data_start + 0] - dc;
    dc = s_data[data_start + 0];
    
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

    // Write category number
    if ( gpujpeg_huffman_gpu_encoder_emit_bits(d_table_dc->code[nbits], d_table_dc->size[nbits], put_value, put_bits, data_compressed) != 0 ) {
        return -1;
    }

    // Write category offset (EmitBits rejects calls with size 0)
    if ( nbits ) {
        if ( gpujpeg_huffman_gpu_encoder_emit_bits((unsigned int) temp2, nbits, put_value, put_bits, data_compressed) != 0 )
            return -1;
    }
    
    // Encode the AC coefficients per section F.1.2.2 (r = run length of zeros)
    int r = 0;
    for ( int k = 1; k < 64; k++ ) 
    {
        temp = s_data[data_start + gpujpeg_huffman_gpu_encoder_order_natural[k]];
        if ( temp == 0 ) {
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
 * Huffman encoder kernel (for CC 1.x)
 * 
 * @return void
 */
__global__ static void
gpujpeg_huffman_encoder_encode_kernel(
    struct gpujpeg_component* d_component,
    struct gpujpeg_segment* d_segment,
    int comp_count,
    int segment_count, 
    uint8_t* d_data_compressed
)
{   
    int segment_index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( segment_index >= segment_count )
        return;
    
    struct gpujpeg_segment* segment = &d_segment[segment_index];
    
    // first thread initializes compact output size for next kernel
    if(0 == segment_index) {
        gpujpeg_huffman_output_byte_count = 0;
    }
    
    // Initialize huffman coder
    int put_value = 0;
    int put_bits = 0;
    int dc[GPUJPEG_MAX_COMPONENT_COUNT];
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ )
        dc[comp] = 0;
    
    // Prepare data pointers
    uint8_t* data_compressed = &d_data_compressed[segment->data_temp_index];
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
                d_table_dc = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC];
                d_table_ac = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC];
            } else {
                d_table_dc = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC];
                d_table_ac = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC];
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
                            d_table_dc = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC];
                            d_table_ac = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC];
                        } else {
                            d_table_dc = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC];
                            d_table_ac = &gpujpeg_huffman_gpu_encoder_table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC];
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
    int restart_marker = GPUJPEG_MARKER_RST0 + (segment->scan_segment_index % 8);
    gpujpeg_huffman_gpu_encoder_marker(data_compressed, restart_marker);
                
    // Set compressed size
    segment->data_compressed_size = data_compressed - data_compressed_start;
}

/** Adds packed coefficients into the GPU version of Huffman lookup table. */
void
gpujpeg_huffman_gpu_add_packed_table(uint32_t * const dest, const struct gpujpeg_table_huffman_encoder * const src, const bool is_ac) {
    // make a upshifted copy of the table for GPU encoding
    for ( int i = 0; i <= 256; i++ ) {
        const int size = src->size[i & 0xFF];
        dest[i] = (src->code[i & 0xFF] << (32 - size)) | size;
    }
    
    // reserve first index in GPU version of AC table for special purposes
    if ( is_ac ) {
        dest[0] = 0;
    }
}

/** Documented at declaration */
int
gpujpeg_huffman_gpu_encoder_init(const struct gpujpeg_encoder * encoder)
{
    
    // Initialize decomposition lookup table
    cudaFuncSetCacheConfig(gpujpeg_huffman_gpu_encoder_value_decomposition_init_kernel, cudaFuncCachePreferShared);
    gpujpeg_huffman_gpu_encoder_value_decomposition_init_kernel<<<32, 256>>>();  // 8192 threads total
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Decomposition LUT initialization failed");
    
    // compose GPU version of the huffman LUT and copy it into GPU memory (for CC >= 2.0)
    uint32_t gpujpeg_huffman_cpu_lut[(256 + 1) * 4];
    gpujpeg_huffman_gpu_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 0, &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_AC], true);
    gpujpeg_huffman_gpu_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 1, &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC], false);
    gpujpeg_huffman_gpu_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 2, &encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_AC], true);
    gpujpeg_huffman_gpu_add_packed_table(gpujpeg_huffman_cpu_lut + 257 * 3, &encoder->table_huffman[GPUJPEG_COMPONENT_CHROMINANCE][GPUJPEG_HUFFMAN_DC], false);
    cudaMemcpyToSymbol(
        gpujpeg_huffman_gpu_lut,
        gpujpeg_huffman_cpu_lut,
        (256 + 1) * 4 * sizeof(*gpujpeg_huffman_gpu_lut),
        0,
        cudaMemcpyHostToDevice
    );
    gpujpeg_cuda_check_error("Huffman encoder init (Huffman LUT copy)");
    
    // Copy original Huffman coding tables to GPU memory (for CC 1.x)
    cudaMemcpyToSymbol(
        gpujpeg_huffman_gpu_encoder_table_huffman,
        &encoder->table_huffman[GPUJPEG_COMPONENT_LUMINANCE][GPUJPEG_HUFFMAN_DC],
        sizeof(gpujpeg_huffman_gpu_encoder_table_huffman),
        0,
        cudaMemcpyHostToDevice
    );
    gpujpeg_cuda_check_error("Huffman encoder init (Huffman coding table)");
    
    // Copy natural order to constant device memory
    cudaMemcpyToSymbol(
        (const char*)gpujpeg_huffman_gpu_encoder_order_natural,
        gpujpeg_order_natural, 
        GPUJPEG_ORDER_NATURAL_SIZE * sizeof(int),
        0,
        cudaMemcpyHostToDevice
    );
    gpujpeg_cuda_check_error("Huffman encoder init (natural order copy)");
    
    // Configure more shared memory for all kernels
    cudaFuncSetCacheConfig(gpujpeg_huffman_encoder_encode_kernel_warp<true>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_encoder_encode_kernel_warp<false>, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_encoder_serialization_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_encoder_compaction_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_encoder_encode_kernel, cudaFuncCachePreferShared);
    cudaFuncSetCacheConfig(gpujpeg_huffman_encoder_allocation_kernel, cudaFuncCachePreferShared);
    
    return 0;
}

/** 
 * Get grid size for specified count of threadblocks. (Grid size is limited 
 * to 65536 in both directions, so if we need more threadblocks, we must use 
 * both x and y coordinates.)
 */
dim3
gpujpeg_huffman_gpu_encoder_grid_size(int tblock_count)
{
    dim3 size(tblock_count);
    while(size.x > 0xffff) {
        size.x = (size.x + 1) >> 1;
        size.y <<= 1;
    }
    return size;
}

/** Documented at declaration */
int
gpujpeg_huffman_gpu_encoder_encode(struct gpujpeg_encoder* encoder, unsigned int * output_byte_count)
{   
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    
    assert(coder->param.restart_interval > 0);
    
    // Select encoder kernel which either expects continuos segments of blocks or uses block lists
    int comp_count = 1;
    if ( coder->param.interleaved == 1 )
        comp_count = coder->param_image.comp_count;
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    
    // Select encoder kernel based on compute capability
    if ( encoder->coder.cuda_cc_major < 2 ) {
        // Run kernel
        dim3 thread(THREAD_BLOCK_SIZE);
        dim3 grid(gpujpeg_div_and_round_up(coder->segment_count, thread.x));
        gpujpeg_huffman_encoder_encode_kernel<<<grid, thread>>>(
            coder->d_component, 
            coder->d_segment, 
            comp_count,
            coder->segment_count, 
            coder->d_temp_huffman
        );
        cudaThreadSynchronize();
        gpujpeg_cuda_check_error("Huffman encoding failed");
    } else {
        // Run encoder kernel
        dim3 thread(32 * WARPS_NUM);
        dim3 grid = gpujpeg_huffman_gpu_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, (thread.x / 32)));
        if(comp_count == 1) {
            gpujpeg_huffman_encoder_encode_kernel_warp<true><<<grid, thread>>>(
                coder->d_segment, 
                coder->segment_count, 
                coder->d_data_compressed,
                coder->d_block_list,
                coder->d_data_quantized,
                coder->d_component,
                comp_count
            );
        } else { 
            gpujpeg_huffman_encoder_encode_kernel_warp<false><<<grid, thread>>>(
                coder->d_segment, 
                coder->segment_count, 
                coder->d_data_compressed,
                coder->d_block_list,
                coder->d_data_quantized,
                coder->d_component,
                comp_count
            );
        }
        cudaThreadSynchronize();
        gpujpeg_cuda_check_error("Huffman encoding failed");
        
        // Run codeword serialization kernel
        const int num_serialization_tblocks = gpujpeg_div_and_round_up(coder->segment_count, SERIALIZATION_THREADS_PER_TBLOCK);
        const dim3 serialization_grid = gpujpeg_huffman_gpu_encoder_grid_size(num_serialization_tblocks);
        gpujpeg_huffman_encoder_serialization_kernel<<<num_serialization_tblocks, SERIALIZATION_THREADS_PER_TBLOCK>>>(
            coder->d_segment, 
            coder->segment_count, 
            coder->d_data_compressed,
            coder->d_temp_huffman
        );
        cudaThreadSynchronize();
        gpujpeg_cuda_check_error("Codeword serialization failed");
    }
    
    // No atomic operations in CC 1.0 => run output size computation kernel to allocate the output buffer space
    if ( encoder->coder.cuda_cc_major == 1 && encoder->coder.cuda_cc_minor == 0 ) {
        gpujpeg_huffman_encoder_allocation_kernel<<<1, 512>>>(coder->d_segment, coder->segment_count);
        cudaThreadSynchronize();
        gpujpeg_cuda_check_error("Huffman encoder output allocation failed");
    }
    
    // Run output compaction kernel (one warp per segment)    
    const dim3 compaction_thread(32, WARPS_NUM);
    const dim3 compaction_grid = gpujpeg_huffman_gpu_encoder_grid_size(gpujpeg_div_and_round_up(coder->segment_count, WARPS_NUM));
    gpujpeg_huffman_encoder_compaction_kernel<<<compaction_grid, compaction_thread>>>(
        coder->d_segment,
        coder->segment_count,
        coder->d_temp_huffman,
        coder->d_data_compressed
    );
    cudaThreadSynchronize();
    gpujpeg_cuda_check_error("Huffman output compaction failed");
    
    // Read and return number of occupied bytes
    cudaMemcpyFromSymbol(output_byte_count, gpujpeg_huffman_output_byte_count, sizeof(unsigned int), 0, cudaMemcpyDeviceToHost);
    gpujpeg_cuda_check_error("Huffman output size getting failed");
    
    // indicate success
    return 0;
}
