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
 
#include "jpeg_encoder.h"
#include "jpeg_preprocessor.h"
#include "jpeg_huffman_cpu_encoder.h"
#include "jpeg_huffman_gpu_encoder.h"
#include "jpeg_format_type.h"
#include "jpeg_util.h"

#define JPEG_ENCODER_MAX_BLOCK_COMPRESSED_SIZE (JPEG_BLOCK_SIZE * JPEG_BLOCK_SIZE * 2)

/** Documented at declaration */
void
jpeg_encoder_set_default_parameters(struct jpeg_encoder_parameters* param)
{
    param->quality = 75;
    param->restart_interval = 8;
}

/** Documented at declaration */
struct jpeg_encoder*
jpeg_encoder_create(struct jpeg_image_parameters* param_image, struct jpeg_encoder_parameters* param)
{
    assert(param_image->comp_count == 3);
    
    struct jpeg_encoder* encoder = malloc(sizeof(struct jpeg_encoder));
    if ( encoder == NULL )
        return NULL;
        
    // Set parameters
    memset(encoder, 0, sizeof(struct jpeg_encoder));
    encoder->param_image = *param_image;
    encoder->param = *param;
    
    int result = 1;
    
    // Create writer
    encoder->writer = jpeg_writer_create(encoder);
    if ( encoder->writer == NULL )
        result = 0;
    
    // Allocate data buffers
    int data_size = encoder->param_image.width * encoder->param_image.width * encoder->param_image.comp_count;
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_data_source, data_size * sizeof(uint8_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_data, data_size * sizeof(uint8_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMallocHost((void**)&encoder->data_quantized, data_size * sizeof(int16_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMalloc((void**)&encoder->d_data_quantized, data_size * sizeof(int16_t)) ) 
        result = 0;
	cudaCheckError("Encoder data allocation");

    // Calculate segments count
    if ( encoder->param.restart_interval != 0 ) {
        int block_count = ((encoder->param_image.width + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE) * ((encoder->param_image.height + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE);
        encoder->segment_count_per_comp = (block_count / encoder->param.restart_interval + 1);
        encoder->segment_count = encoder->param_image.comp_count * encoder->segment_count_per_comp;
        
        // Allocate segments
        cudaMallocHost((void**)&encoder->segments, encoder->segment_count * sizeof(struct jpeg_encoder_segment));
        if ( encoder->segments == NULL )
            result = 0;
        // Allocate segments in device memory
        if ( cudaSuccess != cudaMalloc((void**)&encoder->d_segments, encoder->segment_count * sizeof(struct jpeg_encoder_segment)) )
            result = 0;
        if ( result == 1 ) {
            // Prepare segments for encoding
            for ( int index = 0; index < encoder->segment_count; index++ ) {
                encoder->segments[index].data_compressed_index = index * encoder->param.restart_interval * JPEG_ENCODER_MAX_BLOCK_COMPRESSED_SIZE;
                encoder->segments[index].data_compressed_size = 0;
            }
            // Copy segments to device memory
            if ( cudaSuccess != cudaMemcpy(encoder->d_segments, encoder->segments, encoder->segment_count * sizeof(struct jpeg_encoder_segment), cudaMemcpyHostToDevice) )
                result = 0;
        } 
        
        // Allocate compressed data
        if ( cudaSuccess != cudaMallocHost((void**)&encoder->data_compressed, encoder->segment_count * encoder->param.restart_interval * JPEG_ENCODER_MAX_BLOCK_COMPRESSED_SIZE * sizeof(uint8_t)) ) 
            result = 0;   
        if ( cudaSuccess != cudaMalloc((void**)&encoder->d_data_compressed, encoder->segment_count * encoder->param.restart_interval * JPEG_ENCODER_MAX_BLOCK_COMPRESSED_SIZE * sizeof(uint8_t)) ) 
            result = 0;   
    }
	cudaCheckError("Encoder segment allocation");
     
    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&encoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) 
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < JPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( cudaSuccess != cudaMalloc((void**)&encoder->d_table_huffman[comp_type][huff_type], sizeof(struct jpeg_table_huffman_encoder)) )
                result = 0;
        }
    }
	cudaCheckError("Encoder table allocation");
    
    // Init quantization tables for encoder
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( jpeg_table_quantization_encoder_init(&encoder->table_quantization[comp_type], comp_type, encoder->param.quality) != 0 )
            result = 0;
    }
    // Init huffman tables for encoder
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < JPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( jpeg_table_huffman_encoder_init(&encoder->table_huffman[comp_type][huff_type], encoder->d_table_huffman[comp_type][huff_type], comp_type, huff_type) != 0 )
                result = 0;
        }
    }
	cudaCheckError("Encoder table init");
    
    // Init huffman encoder
    if ( jpeg_huffman_gpu_encoder_init() != 0 )
        result = 0;
    
    if ( result == 0 ) {
        jpeg_encoder_destroy(encoder);
        return NULL;
    }
    
    return encoder;
}

void
jpeg_encoder_print8(struct jpeg_encoder* encoder, uint8_t* d_data)
{
    int data_size = encoder->param_image.width * encoder->param_image.height;
    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(uint8_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < encoder->param_image.height; y++ ) {
        for ( int x = 0; x < encoder->param_image.width; x++ ) {
            printf("%3u ", data[y * encoder->param_image.width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

void
jpeg_encoder_print16(struct jpeg_encoder* encoder, int16_t* d_data)
{
    int data_size = encoder->param_image.width * encoder->param_image.height;
    int16_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(int16_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < encoder->param_image.height; y++ ) {
        for ( int x = 0; x < encoder->param_image.width; x++ ) {
            printf("%3d ", data[y * encoder->param_image.width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/** Documented at declaration */
int
jpeg_encoder_encode(struct jpeg_encoder* encoder, uint8_t* image, uint8_t** image_compressed, int* image_compressed_size)
{
    int data_size = encoder->param_image.width * encoder->param_image.height * encoder->param_image.comp_count;
    
    //TIMER_INIT();
    //TIMER_START();
    
    // Copy image to device memory
    if ( cudaSuccess != cudaMemcpy(encoder->d_data_source, image, data_size * sizeof(uint8_t), cudaMemcpyHostToDevice) )
        return -1;
    
    //jpeg_table_print(encoder->table[JPEG_COMPONENT_LUMINANCE]);
    //jpeg_table_print(encoder->table[JPEG_COMPONENT_CHROMINANCE]);
    
    // Preprocessing
    if ( jpeg_preprocessor_encode(encoder) != 0 )
        return -1;
        
    //TIMER_STOP_PRINT("-Preprocessing:     ");
    //TIMER_START();
        
    // Perform DCT and quantization
    for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
        uint8_t* d_data_comp = &encoder->d_data[comp * encoder->param_image.width * encoder->param_image.height];
        int16_t* d_data_quantized_comp = &encoder->d_data_quantized[comp * encoder->param_image.width * encoder->param_image.height];
        
        // Determine table type
        enum jpeg_component_type type = (comp == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
        
        //jpeg_encoder_print8(encoder, d_data_comp);
        
        //Perform forward DCT
        NppiSize fwd_roi;
        fwd_roi.width = encoder->param_image.width;
        fwd_roi.height = encoder->param_image.height;
        NppStatus status = nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R(
            d_data_comp, 
            encoder->param_image.width * sizeof(uint8_t), 
            d_data_quantized_comp, 
            encoder->param_image.width * JPEG_BLOCK_SIZE * sizeof(int16_t), 
            encoder->table_quantization[type].d_table, 
            fwd_roi
        );
        if ( status != 0 ) {
            fprintf(stderr, "Forward DCT failed for component at index %d [error %d]!\n", comp, status);		
            return -1;
        }
        
        //jpeg_encoder_print16(encoder, d_data_quantized_comp);
    }
    
    // Initialize writer output buffer current position
    encoder->writer->buffer_current = encoder->writer->buffer;
    
    // Write header
    jpeg_writer_write_header(encoder);
    
    //TIMER_STOP_PRINT("-DCT & Quantization:");
    //TIMER_START();
    
    // Perform huffman coding on CPU (when restart interval is not set)
    if ( encoder->param.restart_interval == 0 ) {
        // Copy quantized data from device memory to cpu memory
        cudaMemcpy(encoder->data_quantized, encoder->d_data_quantized, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
        
        // Perform huffman coding for all components
        for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
            // Get data buffer for component
            int16_t* data_comp = &encoder->data_quantized[comp * encoder->param_image.width * encoder->param_image.height];
            int16_t* d_data_comp = &encoder->d_data_quantized[comp * encoder->param_image.width * encoder->param_image.height];
            // Determine table type
            enum jpeg_component_type type = (comp == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
            // Write scan header
            jpeg_writer_write_scan_header(encoder, comp, type);
            // Perform huffman coding
            if ( jpeg_huffman_cpu_encoder_encode(encoder, type, data_comp) != 0 ) {
                fprintf(stderr, "Huffman encoder on CPU failed for component at index %d!\n", comp);
                return -1;
            }
        }
    }
    // Perform huffman coding on GPU (when restart interval is set)
    else {
        // Perform huffman coding
        if ( jpeg_huffman_gpu_encoder_encode(encoder) != 0 ) {
            fprintf(stderr, "Huffman encoder on GPU failed!\n");
            return -1;
        }
        
        // Copy compressed data from device memory to cpu memory
        if ( cudaSuccess != cudaMemcpy(encoder->data_compressed, encoder->d_data_compressed, encoder->segment_count * encoder->param.restart_interval * JPEG_ENCODER_MAX_BLOCK_COMPRESSED_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost) != 0 )
            return -1;
        // Copy segments to device memory
        if ( cudaSuccess != cudaMemcpy(encoder->segments, encoder->d_segments, encoder->segment_count * sizeof(struct jpeg_encoder_segment), cudaMemcpyDeviceToHost) )
            return -1;
        
        // Write huffman coder results
        for ( int comp = 0; comp < encoder->param_image.comp_count; comp++ ) {
            // Determine table type
            enum jpeg_component_type type = (comp == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
            // Write scan header
            jpeg_writer_write_scan_header(encoder, comp, type);
            // Write scan data
            for ( int index = 0; index < encoder->segment_count_per_comp; index++ ) {
                int segment_index = (comp * encoder->segment_count_per_comp + index);
                struct jpeg_encoder_segment* segment = &encoder->segments[segment_index];
                
                // Copy compressed data to writer
                memcpy(
                    encoder->writer->buffer_current, 
                    &encoder->data_compressed[segment->data_compressed_index],
                    segment->data_compressed_size
                );
                encoder->writer->buffer_current += segment->data_compressed_size;
                //printf("Compressed data %d bytes\n", segment->data_compressed_size);
            }
        }
    }
    jpeg_writer_emit_marker(encoder->writer, JPEG_MARKER_EOI);
    
    //TIMER_STOP_PRINT("-Huffman Encoder:   ");
    
    // Set compressed image
    *image_compressed = encoder->writer->buffer;
    *image_compressed_size = encoder->writer->buffer_current - encoder->writer->buffer;
    
    return 0;
}

/** Documented at declaration */
int
jpeg_encoder_destroy(struct jpeg_encoder* encoder)
{
    assert(encoder != NULL);
    
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( encoder->table_quantization[comp_type].d_table != NULL )
            cudaFree(encoder->table_quantization[comp_type].d_table);
    }
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < JPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( encoder->d_table_huffman[comp_type][huff_type] != NULL )
                cudaFree(encoder->d_table_huffman[comp_type][huff_type]);
        }
    }
    
    if ( encoder->writer != NULL )
        jpeg_writer_destroy(encoder->writer);
    
    if ( encoder->d_data_source != NULL )
        cudaFree(encoder->d_data_source);
    if ( encoder->d_data != NULL )
        cudaFree(encoder->d_data);
    if ( encoder->data_quantized != NULL )
        cudaFreeHost(encoder->data_quantized);    
    if ( encoder->d_data_quantized != NULL )
        cudaFree(encoder->d_data_quantized);    
    if ( encoder->data_compressed != NULL )
        cudaFreeHost(encoder->data_compressed);    
    if ( encoder->d_data_compressed != NULL )
        cudaFree(encoder->d_data_compressed);    
    if ( encoder->segments != NULL )
        cudaFreeHost(encoder->segments);  
    if ( encoder->d_segments != NULL )
        cudaFree(encoder->d_segments);    
    
    free(encoder);
    
    return 0;
}
