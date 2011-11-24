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
 
#include "jpeg_decoder.h"
#include "jpeg_preprocessor.h"
#include "jpeg_huffman_cpu_decoder.h"
#include "jpeg_huffman_gpu_decoder.h"
#include "jpeg_util.h"

/** Documented at declaration */
struct jpeg_decoder*
jpeg_decoder_create(struct jpeg_image_parameters* param_image)
{
    struct jpeg_decoder* decoder = malloc(sizeof(struct jpeg_decoder));
    if ( decoder == NULL )
        return NULL;
        
    // Set parameters
    decoder->param_image = *param_image;
    decoder->param_image.width = 0;
    decoder->param_image.height = 0;
    decoder->param_image.comp_count = 0;
    decoder->restart_interval = 0;
    decoder->data_quantized = NULL;
    decoder->d_data_quantized = NULL;
    decoder->d_data = NULL;
    decoder->data_target = NULL;
    decoder->d_data_target = NULL;
    
    int result = 1;
    
    // Create reader
    decoder->reader = jpeg_reader_create();
    if ( decoder->reader == NULL )
        result = 0;
    
    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&decoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) 
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < JPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( cudaSuccess != cudaMalloc((void**)&decoder->d_table_huffman[comp_type][huff_type], sizeof(struct jpeg_table_huffman_decoder)) )
                result = 0;
        }
    }
    cudaCheckError("Decoder table allocation");
    
    // Init decoder
    if ( param_image->width != 0 && param_image->height != 0 ) {
        if ( jpeg_decoder_init(decoder, param_image->width, param_image->height, param_image->comp_count) != 0 )
            result = 0;
    }
    
    // Init huffman encoder
    if ( jpeg_huffman_gpu_decoder_init() != 0 )
        result = 0;
    
    if ( result == 0 ) {
        jpeg_decoder_destroy(decoder);
        return NULL;
    }
    
    return decoder;
}

/** Documented at declaration */
int
jpeg_decoder_init(struct jpeg_decoder* decoder, int width, int height, int comp_count)
{
    assert(comp_count == 3);
    
    // No reinialization needed
    if ( decoder->param_image.width == width && decoder->param_image.height == height && decoder->param_image.comp_count == comp_count ) {
        return 0;
    }
    
    // For now we can't reinitialize decoder, we can only do first initialization
    if ( decoder->param_image.width != 0 || decoder->param_image.height != 0 || decoder->param_image.comp_count != 0 ) {
        fprintf(stderr, "Can't reinitialize decoder, implement if needed!\n");
        return -1;
    }
    
    decoder->param_image.width = width;
    decoder->param_image.height = height;
    decoder->param_image.comp_count = comp_count;
    
    // Allocate scan data (we need more data ie twice, restart_interval could be 1 so a lot of data)
    // and indexes to data for each segment
    int data_scan_size = decoder->param_image.comp_count * decoder->param_image.width * decoder->param_image.height * 2;
    int max_segment_count = decoder->param_image.comp_count * ((decoder->param_image.width + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE) * ((decoder->param_image.height + JPEG_BLOCK_SIZE - 1) / JPEG_BLOCK_SIZE);
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_scan, data_scan_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_scan, data_scan_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_scan_index, max_segment_count * sizeof(int)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_scan_index, max_segment_count * sizeof(int)) ) 
        return -1;
    cudaCheckError("Decoder scan allocation");
    
    // Allocate buffers
    int data_size = decoder->param_image.width * decoder->param_image.width * decoder->param_image.comp_count;
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_quantized, data_size * sizeof(int16_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_quantized, data_size * sizeof(int16_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data, data_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMallocHost((void**)&decoder->data_target, data_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&decoder->d_data_target, data_size * sizeof(uint8_t)) ) 
        return -1;
    cudaCheckError("Decoder data allocation");
    
    return 0;
}

void
jpeg_decoder_print8(struct jpeg_decoder* decoder, uint8_t* d_data)
{
    int data_size = decoder->param_image.width * decoder->param_image.height;
    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(uint8_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < decoder->param_image.height; y++ ) {
        for ( int x = 0; x < decoder->param_image.width; x++ ) {
            printf("%3u ", data[y * decoder->param_image.width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

void
jpeg_decoder_print16(struct jpeg_decoder* decoder, int16_t* d_data)
{
    int data_size = decoder->param_image.width * decoder->param_image.height;
    int16_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(int16_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < decoder->param_image.height; y++ ) {
        for ( int x = 0; x < decoder->param_image.width; x++ ) {
            printf("%3d ", data[y * decoder->param_image.width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/** Documented at declaration */
int
jpeg_decoder_decode(struct jpeg_decoder* decoder, uint8_t* image, int image_size, uint8_t** image_decompressed, int* image_decompressed_size)
{    
    //TIMER_INIT();
    //TIMER_START();
    
    // Read JPEG image data
    if ( jpeg_reader_read_image(decoder, image, image_size) != 0 ) {
        fprintf(stderr, "Decoder failed when decoding image data!\n");
        return -1;
    }
    
    int data_size = decoder->param_image.width * decoder->param_image.height * decoder->param_image.comp_count;
    
    //TIMER_STOP_PRINT("-Stream Reader:     ");
    //TIMER_START();
    
    // Perform huffman decoding on CPU (when restart interval is not set)
    if ( decoder->restart_interval == 0 ) {
        // Perform huffman decoding for all components
        for ( int index = 0; index < decoder->scan_count; index++ ) {
            // Get scan and data buffer
            struct jpeg_decoder_scan* scan = &decoder->scan[index];
            int16_t* data_quantized_comp = &decoder->data_quantized[index * decoder->param_image.width * decoder->param_image.height];
            // Determine table type
            enum jpeg_component_type type = (index == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
            // Huffman decode
            if ( jpeg_huffman_cpu_decoder_decode(decoder, type, scan, data_quantized_comp) != 0 ) {
                fprintf(stderr, "Huffman decoder failed for scan at index %d!\n", index);
                return -1;
            }
        }
        
        // Copy quantized data to device memory from cpu memory    
        cudaMemcpy(decoder->d_data_quantized, decoder->data_quantized, data_size * sizeof(int16_t), cudaMemcpyHostToDevice);
    }
    // Perform huffman decoding on GPU (when restart interval is set)
    else {
        cudaMemset(decoder->d_data_quantized, 0, decoder->param_image.comp_count * decoder->param_image.width * decoder->param_image.height * sizeof(int16_t));
        
        // Copy scan data to device memory
        cudaMemcpy(decoder->d_data_scan, decoder->data_scan, decoder->data_scan_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        cudaCheckError("Decoder copy scan data");
        // Copy scan data to device memory
        cudaMemcpy(decoder->d_data_scan_index, decoder->data_scan_index, decoder->segment_count * sizeof(int), cudaMemcpyHostToDevice);
        cudaCheckError("Decoder copy scan data index");
        
        // Zero output memory
        cudaMemset(decoder->d_data_quantized, 0, decoder->param_image.comp_count * decoder->param_image.width * decoder->param_image.height * sizeof(int16_t));
        
        // Perform huffman decoding
        if ( jpeg_huffman_gpu_decoder_decode(decoder) != 0 ) {
            fprintf(stderr, "Huffman decoder on GPU failed!\n");
            return -1;
        }
    }
    
    //TIMER_STOP_PRINT("-Huffman Decoder:   ");
    //TIMER_START();
    
    // Perform IDCT and dequantization
    for ( int comp = 0; comp < decoder->param_image.comp_count; comp++ ) {
        uint8_t* d_data_comp = &decoder->d_data[comp * decoder->param_image.width * decoder->param_image.height];
        int16_t* d_data_quantized_comp = &decoder->d_data_quantized[comp * decoder->param_image.width * decoder->param_image.height];
        
        // Determine table type
        enum jpeg_component_type type = (comp == 0) ? JPEG_COMPONENT_LUMINANCE : JPEG_COMPONENT_CHROMINANCE;
        
        //jpeg_decoder_print16(decoder, d_data_quantized_comp);
        
        cudaMemset(d_data_comp, 0, decoder->param_image.width * decoder->param_image.height * sizeof(uint8_t));
        
        //Perform inverse DCT
        NppiSize inv_roi;
        inv_roi.width = decoder->param_image.width * JPEG_BLOCK_SIZE;
        inv_roi.height = decoder->param_image.height / JPEG_BLOCK_SIZE;
        assert(JPEG_BLOCK_SIZE == 8);
        NppStatus status = nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(
            d_data_quantized_comp, 
            decoder->param_image.width * JPEG_BLOCK_SIZE * sizeof(int16_t), 
            d_data_comp, 
            decoder->param_image.width * sizeof(uint8_t), 
            decoder->table_quantization[type].d_table, 
            inv_roi
        );
        if ( status != 0 )
            printf("Error %d\n", status);
        //jpeg_decoder_print8(decoder, d_data_comp);
    }
    
    //TIMER_STOP_PRINT("-DCT & Quantization:");
    //TIMER_START();
    
    // Preprocessing
    if ( jpeg_preprocessor_decode(decoder) != 0 )
        return -1;
        
    //TIMER_STOP_PRINT("-Postprocessing:    ");
    
    cudaMemcpy(decoder->data_target, decoder->d_data_target, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    // Set decompressed image
    *image_decompressed = decoder->data_target;
    *image_decompressed_size = data_size * sizeof(uint8_t);
    
    return 0;
}

/** Documented at declaration */
int
jpeg_decoder_destroy(struct jpeg_decoder* decoder)
{    
    assert(decoder != NULL);
    
    for ( int comp_type = 0; comp_type < JPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( decoder->table_quantization[comp_type].d_table != NULL )
            cudaFree(decoder->table_quantization[comp_type].d_table);
    }
    
    if ( decoder->reader != NULL )
        jpeg_reader_destroy(decoder->reader);
    
    if ( decoder->data_scan != NULL )
        cudaFreeHost(decoder->data_scan);
    if ( decoder->data_scan != NULL )
        cudaFree(decoder->d_data_scan);
    if ( decoder->data_scan_index != NULL )
        cudaFreeHost(decoder->data_scan_index);
    if ( decoder->data_scan_index != NULL )
        cudaFree(decoder->d_data_scan_index);
    if ( decoder->data_quantized != NULL )
        cudaFreeHost(decoder->data_quantized);
    if ( decoder->d_data_quantized != NULL )
        cudaFree(decoder->d_data_quantized);
    if ( decoder->d_data != NULL )
        cudaFree(decoder->d_data);
    if ( decoder->data_target != NULL )
        cudaFreeHost(decoder->data_target);
    if ( decoder->d_data_target != NULL )
        cudaFree(decoder->d_data_target);
    
    free(decoder);
    
    return 0;
}
