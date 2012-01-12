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
 
#include "gpujpeg_decoder.h"
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_huffman_gpu_decoder.h"
#include "gpujpeg_util.h"
#include <npp.h>

#ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
/** Huffman tables in constant memory */
struct gpujpeg_table_huffman_decoder (*gpujpeg_decoder_table_huffman)[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
#endif

/** Documented at declaration */
void
gpujpeg_decoder_output_set_default(struct gpujpeg_decoder_output* output)
{
    output->type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture_pbo_resource = NULL;
    output->texture_callback_param = NULL;
    output->texture_callback_attach_opengl = NULL;
    output->texture_callback_detach_opengl = NULL;
}

/** Documented at declaration */
struct gpujpeg_decoder*
gpujpeg_decoder_create()
{    
    struct gpujpeg_decoder* decoder = malloc(sizeof(struct gpujpeg_decoder));
    if ( decoder == NULL )
        return NULL;
        
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    // Set parameters
    memset(decoder, 0, sizeof(struct gpujpeg_decoder));
    gpujpeg_set_default_parameters(&coder->param);
    gpujpeg_image_set_default_parameters(&coder->param_image);
    coder->param_image.comp_count = 0;
    coder->param_image.width = 0;
    coder->param_image.height = 0;
    coder->param.restart_interval = 0;
    
    int result = 1;
    
    // Create reader
    decoder->reader = gpujpeg_reader_create();
    if ( decoder->reader == NULL )
        result = 0;
    
    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&decoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) 
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( cudaSuccess != cudaMalloc((void**)&decoder->d_table_huffman[comp_type][huff_type], sizeof(struct gpujpeg_table_huffman_decoder)) )
                result = 0;
        }
    }
    gpujpeg_cuda_check_error("Decoder table allocation");
    
    // Init huffman encoder
    if ( gpujpeg_huffman_gpu_decoder_init() != 0 )
        result = 0;
    
    if ( result == 0 ) {
        gpujpeg_decoder_destroy(decoder);
        return NULL;
    }
    
    return decoder;
}

/** Documented at declaration */
int
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image)
{
    assert(param_image->comp_count == 3);
    
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    // Check if (re)inialization is needed
    int change = 0;
    change |= coder->param_image.width != param_image->width;
    change |= coder->param_image.height != param_image->height;
    change |= coder->param_image.comp_count != param_image->comp_count;
    change |= coder->param.restart_interval != param->restart_interval;
    change |= coder->param.interleaved != param->interleaved;
    for ( int comp = 0; comp < param_image->comp_count; comp++ ) {
        change |= coder->param.sampling_factor[comp].horizontal != param->sampling_factor[comp].horizontal;
        change |= coder->param.sampling_factor[comp].vertical != param->sampling_factor[comp].vertical;
    }
    if ( change == 0 )
        return 0;
    
    // For now we can't reinitialize decoder, we can only do first initialization
    if ( coder->param_image.width != 0 || coder->param_image.height != 0 || coder->param_image.comp_count != 0 ) {
        fprintf(stderr, "Can't reinitialize decoder, implement if needed!\n");
        return -1;
    }
    
    coder->param = *param;
    coder->param_image = *param_image;
    
    // Initialize coder
    if ( gpujpeg_coder_init(coder) != 0 )
        return -1;
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size, struct gpujpeg_decoder_output* output)
{
    //GPUJPEG_TIMER_INIT();
    //GPUJPEG_TIMER_START();
    
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
    // Set custom output buffer
    if ( output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER ) {
        assert(output->data != NULL);
        coder->data_raw = output->data;
    }
    
    // Read JPEG image data
    if ( gpujpeg_reader_read_image(decoder, image, image_size) != 0 ) {
        fprintf(stderr, "Decoder failed when decoding image data!\n");
        return -1;
    }
    
    //GPUJPEG_TIMER_STOP_PRINT("-Stream Reader:     ");
    //GPUJPEG_TIMER_START();
    
    // Perform huffman decoding on CPU (when restart interval is not set)
    if ( coder->param.restart_interval == 0 ) {
        if ( gpujpeg_huffman_cpu_decoder_decode(decoder) != 0 ) {
            fprintf(stderr, "Huffman decoder failed!\n", index);
            return -1;
        }
        // Copy quantized data to device memory from cpu memory
        cudaMemcpy(coder->d_data_quantized, coder->data_quantized, coder->data_size * sizeof(int16_t), cudaMemcpyHostToDevice);
    }
    // Perform huffman decoding on GPU (when restart interval is set)
    else {
    #ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
        // Copy huffman tables to constant memory
        for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
            for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
                int index = (comp_type * GPUJPEG_HUFFMAN_TYPE_COUNT + huff_type);
                cudaMemcpyToSymbol(
                    (char*)gpujpeg_decoder_table_huffman, 
                    &decoder->table_huffman[comp_type][huff_type], 
                    sizeof(struct gpujpeg_table_huffman_decoder), 
                    index * sizeof(struct gpujpeg_table_huffman_decoder), 
                    cudaMemcpyHostToDevice
                );
            }
        }
        gpujpeg_cuda_check_error("Decoder copy huffman tables to constant memory");
    #endif
    
        // Reset huffman output
        cudaMemset(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t));
        
        // Copy scan data to device memory
        cudaMemcpy(coder->d_data_compressed, coder->data_compressed, decoder->data_compressed_size * sizeof(uint8_t), cudaMemcpyHostToDevice);
        gpujpeg_cuda_check_error("Decoder copy compressed data");
        
        // Copy segments to device memory
        cudaMemcpy(coder->d_segment, coder->segment, decoder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice);
        gpujpeg_cuda_check_error("Decoder copy compressed data");
        
        // Zero output memory
        cudaMemset(coder->d_data_quantized, 0, coder->data_size * sizeof(int16_t));
        
        // Perform huffman decoding
        if ( gpujpeg_huffman_gpu_decoder_decode(decoder) != 0 ) {
            fprintf(stderr, "Huffman decoder on GPU failed!\n");
            return -1;
        }
    }
    
    //GPUJPEG_TIMER_STOP_PRINT("-Huffman Decoder:   ");
    //GPUJPEG_TIMER_START();
    
    // Perform IDCT and dequantization
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];
                
        // Determine table type
        enum gpujpeg_component_type type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;
        
        //gpujpeg_component_print16(component, component->d_data_quantized);
        
        cudaMemset(component->d_data, 0, component->data_size * sizeof(uint8_t));
        
        //Perform inverse DCT
        NppiSize inv_roi;
        inv_roi.width = component->data_width * GPUJPEG_BLOCK_SIZE;
        inv_roi.height = component->data_height / GPUJPEG_BLOCK_SIZE;
        assert(GPUJPEG_BLOCK_SIZE == 8);
        NppStatus status = nppiDCTQuantInv8x8LS_JPEG_16s8u_C1R(
            component->d_data_quantized, 
            component->data_width * GPUJPEG_BLOCK_SIZE * sizeof(int16_t), 
            component->d_data, 
            component->data_width * sizeof(uint8_t), 
            decoder->table_quantization[type].d_table, 
            inv_roi
        );
        if ( status != 0 )
            printf("Error %d\n", status);
            
        //gpujpeg_component_print8(component, component->d_data);
    }
    
    //GPUJPEG_TIMER_STOP_PRINT("-DCT & Quantization:");
    //GPUJPEG_TIMER_START();
    
    // Preprocessing
    if ( gpujpeg_preprocessor_decode(decoder) != 0 )
        return -1;
        
    //GPUJPEG_TIMER_STOP_PRINT("-Postprocessing:    ");
    
    // Set decompressed image size
    output->data_size = coder->data_raw_size * sizeof(uint8_t);
    
    // Set decompressed image
    if ( output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER ) {
        
        // Copy decompressed image to host memory
        cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        
        // Set output to internal buffer
        output->data = coder->data_raw;
        
    } else if ( output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER ) {
        
        // Copy decompressed image to host memory
        cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        
        // Do nothing more because coder->data_raw is already same as output->data
        
    } else if ( output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE ) {
        assert(output->texture_pbo_resource != NULL);
        assert((output->texture_callback_attach_opengl == NULL && output->texture_callback_detach_opengl == NULL) || 
               (output->texture_callback_attach_opengl != NULL && output->texture_callback_detach_opengl != NULL));
        
        // Attach OpenGL context by callback
        if ( output->texture_callback_attach_opengl != NULL )
            output->texture_callback_attach_opengl(output->texture_callback_param);
        
        // Map pixel buffer object to cuda
        cudaGraphicsMapResources(1, &output->texture_pbo_resource, 0);
        gpujpeg_cuda_check_error("Decoder map texture PBO resource");
         
        // Get device data pointer to pixel buffer object data
        uint8_t* d_data = NULL;
        size_t data_size; 
        cudaGraphicsResourceGetMappedPointer((void **)&d_data, &data_size, output->texture_pbo_resource);
        gpujpeg_cuda_check_error("Decoder get device pointer for texture PBO resource");
        assert(data_size == (coder->data_raw_size));
            
        // Copy decompressed image to texture pixel buffer object device data
        cudaMemcpy(d_data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);
            
        // Unmap pbo
        cudaGraphicsUnmapResources(1, &output->texture_pbo_resource, 0);
        gpujpeg_cuda_check_error("Decoder unmap texture PBO resource");
        
        // Dettach OpenGL context by callback
        if ( output->texture_callback_detach_opengl != NULL )
            output->texture_callback_detach_opengl(output->texture_callback_param);
    } else {
        // Unknown output type
        assert(0);
    }
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_decoder_destroy(struct gpujpeg_decoder* decoder)
{    
    assert(decoder != NULL);
    
    if ( gpujpeg_coder_deinit(&decoder->coder) != 0 )
        return -1;
    
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( decoder->table_quantization[comp_type].d_table != NULL )
            cudaFree(decoder->table_quantization[comp_type].d_table);
    }
    
    if ( decoder->reader != NULL )
        gpujpeg_reader_destroy(decoder->reader);
    
    free(decoder);
    
    return 0;
}
