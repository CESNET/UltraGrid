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
 
#include "gpujpeg_encoder.h"
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_dct_cpu.h"
#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_huffman_cpu_encoder.h"
#include "gpujpeg_huffman_gpu_encoder.h"
#include "gpujpeg_util.h"
#include <npp.h>

#ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
/** Huffman tables in constant memory */
struct gpujpeg_table_huffman_encoder (*gpujpeg_encoder_table_huffman)[GPUJPEG_COMPONENT_TYPE_COUNT][GPUJPEG_HUFFMAN_TYPE_COUNT];
#endif

/** Documented at declaration */
void
gpujpeg_encoder_input_set_image(struct gpujpeg_encoder_input* input, uint8_t* image)
{
    input->type = GPUJPEG_ENCODER_INPUT_IMAGE;
    input->image = image;
    input->texture = NULL;
}

/** Documented at declaration */
void
gpujpeg_encoder_input_set_texture(struct gpujpeg_encoder_input* input, struct gpujpeg_opengl_texture* texture)
{
    input->type = GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE;
    input->image = NULL;
    input->texture = texture;
}

/** Documented at declaration */
struct gpujpeg_encoder*
gpujpeg_encoder_create(struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image)
{
    assert(param_image->comp_count == 3);
    assert(param_image->comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    assert(param->quality >= 0 && param->quality <= 100);
    assert(param->restart_interval >= 0);
    assert(param->interleaved == 0 || param->interleaved == 1);
    
    struct gpujpeg_encoder* encoder = malloc(sizeof(struct gpujpeg_encoder));
    if ( encoder == NULL )
        return NULL;
        
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
        
    // Set parameters
    memset(encoder, 0, sizeof(struct gpujpeg_encoder));
    coder->param_image = *param_image;
    coder->param = *param;
    
    int result = 1;
    
    // Create writer
    encoder->writer = gpujpeg_writer_create(encoder);
    if ( encoder->writer == NULL )
        result = 0;
    
    // Initialize coder
    if ( gpujpeg_coder_init(coder) != 0 )
        result = 0;
        
    // Init preprocessor
    if ( gpujpeg_preprocessor_encoder_init(&encoder->coder) != 0 ) {
        fprintf(stderr, "Failed to init preprocessor!");
        result = 0;
    }
     
    // Allocate quantization tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( cudaSuccess != cudaMalloc((void**)&encoder->table_quantization[comp_type].d_table, 64 * sizeof(uint16_t)) ) 
            result = 0;
    }
    // Allocate huffman tables in device memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( cudaSuccess != cudaMalloc((void**)&encoder->d_table_huffman[comp_type][huff_type], sizeof(struct gpujpeg_table_huffman_encoder)) )
                result = 0;
        }
    }
    gpujpeg_cuda_check_error("Encoder table allocation");
    
    // Init quantization tables for encoder
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( gpujpeg_table_quantization_encoder_init(&encoder->table_quantization[comp_type], (enum gpujpeg_component_type)comp_type, coder->param.quality) != 0 )
            result = 0;
    }
    // Init huffman tables for encoder
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( gpujpeg_table_huffman_encoder_init(&encoder->table_huffman[comp_type][huff_type], encoder->d_table_huffman[comp_type][huff_type], (enum gpujpeg_component_type)comp_type, (enum gpujpeg_huffman_type)huff_type) != 0 )
                result = 0;
        }
    }
    gpujpeg_cuda_check_error("Encoder table init");
    
#ifdef GPUJPEG_HUFFMAN_CODER_TABLES_IN_CONSTANT
    // Copy huffman tables to constant memory
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            int index = (comp_type * GPUJPEG_HUFFMAN_TYPE_COUNT + huff_type);
            cudaMemcpyToSymbol(
                (char*)gpujpeg_encoder_table_huffman, 
                &encoder->table_huffman[comp_type][huff_type], 
                sizeof(struct gpujpeg_table_huffman_encoder), 
                index * sizeof(struct gpujpeg_table_huffman_encoder), 
                cudaMemcpyHostToDevice
            );
        }
    }
    gpujpeg_cuda_check_error("Encoder copy huffman tables to constant memory");
#endif
    
    // Init huffman encoder
    if ( gpujpeg_huffman_gpu_encoder_init() != 0 )
        result = 0;
    
    if ( result == 0 ) {
        gpujpeg_encoder_destroy(encoder);
        return NULL;
    }
    
    // Timers
    GPUJPEG_CUSTOM_TIMER_CREATE(encoder->def);
    GPUJPEG_CUSTOM_TIMER_CREATE(encoder->in_gpu);

    return encoder;
}

/** Documented at declaration */
int
gpujpeg_encoder_encode(struct gpujpeg_encoder* encoder, struct gpujpeg_encoder_input* input, uint8_t** image_compressed, int* image_compressed_size)
{
    // Get coder
    struct gpujpeg_coder* coder = &encoder->coder;
    
    // Reset durations
    coder->duration_memory_to = 0.0;
    coder->duration_memory_from = 0.0;
    coder->duration_memory_map = 0.0;
    coder->duration_memory_unmap = 0.0;
    coder->duration_preprocessor = 0.0;
    coder->duration_dct_quantization = 0.0;
    coder->duration_huffman_coder = 0.0;
    coder->duration_stream = 0.0;
    coder->duration_in_gpu = 0.0;
    
    // Load input image
    if ( input->type == GPUJPEG_ENCODER_INPUT_IMAGE ) {
        GPUJPEG_CUSTOM_TIMER_START(encoder->def);

        // Copy image to device memory
        if ( cudaSuccess != cudaMemcpy(coder->d_data_raw, input->image, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyHostToDevice) )
            return -1;

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_memory_to = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
    } else
    if ( input->type == GPUJPEG_ENCODER_INPUT_OPENGL_TEXTURE ) {
        assert(input->texture != NULL);

        GPUJPEG_CUSTOM_TIMER_START(encoder->def);

        // Map texture to CUDA
        int data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(input->texture, &data_size);
        assert(data_size == (coder->data_raw_size));

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);

        GPUJPEG_CUSTOM_TIMER_START(encoder->def);

        // Copy image data from texture pixel buffer object to device data
        cudaMemcpy(coder->d_data_raw, d_data, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_memory_to = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);

        GPUJPEG_CUSTOM_TIMER_START(encoder->def);

        // Unmap texture from CUDA
        gpujpeg_opengl_texture_unmap(input->texture);

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_memory_unmap = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
    } else {
        // Unknown output type
        assert(0);
    }

    //gpujpeg_table_print(encoder->table[JPEG_COMPONENT_LUMINANCE]);
    //gpujpeg_table_print(encoder->table[JPEG_COMPONENT_CHROMINANCE]);
    
    GPUJPEG_CUSTOM_TIMER_START(encoder->in_gpu);
    GPUJPEG_CUSTOM_TIMER_START(encoder->def);

    // Preprocessing
    if ( gpujpeg_preprocessor_encode(&encoder->coder) != 0 )
        return -1;
        
    GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
    coder->duration_preprocessor = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
    GPUJPEG_CUSTOM_TIMER_START(encoder->def);
        
#ifdef GPUJPEG_DCT_FROM_NPP
    // Perform DCT and quantization (implementation from NPP)
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];
                
        // Determine table type
        enum gpujpeg_component_type type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;
        
        //gpujpeg_component_print8(&coder->component[comp], coder->component[comp].d_data);
        
        //Perform forward DCT
        NppiSize fwd_roi;
        fwd_roi.width = component->data_width;
        fwd_roi.height = component->data_height;
        NppStatus status = nppiDCTQuantFwd8x8LS_JPEG_8u16s_C1R(
            component->d_data, 
            component->data_width * sizeof(uint8_t), 
            component->d_data_quantized, 
            component->data_width * GPUJPEG_BLOCK_SIZE * sizeof(int16_t), 
            encoder->table_quantization[type].d_table, 
            fwd_roi
        );
        if ( status != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Forward DCT failed for component at index %d [error %d]!\n", comp, status);
            return -1;
        }

        //gpujpeg_component_print16(&coder->component[comp], coder->component[comp].d_data_quantized);
    }
#else
    // Perform DCT and quantization (own CUDA implementation)
    gpujpeg_dct_gpu(encoder);
#endif

    // If restart interval is 0 then the GPU processing is in the end (even huffman coder will be performed on CPU)
    if ( coder->param.restart_interval == 0 ) {
        GPUJPEG_CUSTOM_TIMER_STOP(encoder->in_gpu);
        coder->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->in_gpu);
    }

    // Initialize writer output buffer current position
    encoder->writer->buffer_current = encoder->writer->buffer;
    
    // Write header
    gpujpeg_writer_write_header(encoder);
    
    GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
    coder->duration_dct_quantization = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
    GPUJPEG_CUSTOM_TIMER_START(encoder->def);

    // Perform huffman coding on CPU (when restart interval is not set)
    if ( coder->param.restart_interval == 0 ) {
        // Copy quantized data from device memory to cpu memory
        cudaMemcpy(coder->data_quantized, coder->d_data_quantized, coder->data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
        
        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
        GPUJPEG_CUSTOM_TIMER_START(encoder->def);

        // Perform huffman coding
        if ( gpujpeg_huffman_cpu_encoder_encode(encoder) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman encoder on CPU failed!\n");
            return -1;
        }

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_huffman_coder = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
    }
    // Perform huffman coding on GPU (when restart interval is set)
    else {    
        // Perform huffman coding
        if ( gpujpeg_huffman_gpu_encoder_encode(encoder) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman encoder on GPU failed!\n");
            return -1;
        }
        
        GPUJPEG_CUSTOM_TIMER_STOP(encoder->in_gpu);
        coder->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->in_gpu);

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_huffman_coder = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
        GPUJPEG_CUSTOM_TIMER_START(encoder->def);

        // Copy compressed data from device memory to cpu memory
        if ( cudaSuccess != cudaMemcpy(coder->data_compressed, coder->d_data_compressed, coder->data_compressed_size * sizeof(uint8_t), cudaMemcpyDeviceToHost) != 0 )
            return -1;
        // Copy segments from device memory
        if ( cudaSuccess != cudaMemcpy(coder->segment, coder->d_segment, coder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyDeviceToHost) )
            return -1;

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
        GPUJPEG_CUSTOM_TIMER_START(encoder->def);
            
        if ( coder->param.interleaved == 1 ) {
            // Write scan header (only one scan is written, that contains all color components data)
            gpujpeg_writer_write_scan_header(encoder, 0);

            // Write scan data
            for ( int segment_index = 0; segment_index < coder->segment_count; segment_index++ ) {
                struct gpujpeg_segment* segment = &coder->segment[segment_index];

                gpujpeg_writer_write_segment_info(encoder);

                // Copy compressed data to writer
                memcpy(
                    encoder->writer->buffer_current, 
                    &coder->data_compressed[segment->data_compressed_index],
                    segment->data_compressed_size
                );
                encoder->writer->buffer_current += segment->data_compressed_size;
                //printf("Compressed data %d bytes\n", segment->data_compressed_size);
            }
            // Remove last restart marker in scan (is not needed)
            encoder->writer->buffer_current -= 2;

            gpujpeg_writer_write_segment_info(encoder);
        } else {
            // Write huffman coder results as one scan for each color component
            int segment_index = 0;
            for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
                // Write scan header
                gpujpeg_writer_write_scan_header(encoder, comp);
                // Write scan data
                for ( int index = 0; index < coder->component[comp].segment_count; index++ ) {
                    struct gpujpeg_segment* segment = &coder->segment[segment_index];

                    gpujpeg_writer_write_segment_info(encoder);
                
                    // Copy compressed data to writer
                    memcpy(
                        encoder->writer->buffer_current, 
                        &coder->data_compressed[segment->data_compressed_index],
                        segment->data_compressed_size
                    );
                    encoder->writer->buffer_current += segment->data_compressed_size;
                    //printf("Compressed data %d bytes\n", segment->data_compressed_size);

                    segment_index++;
                }
                // Remove last restart marker in scan (is not needed)
                encoder->writer->buffer_current -= 2;

                gpujpeg_writer_write_segment_info(encoder);
            }
        }

        GPUJPEG_CUSTOM_TIMER_STOP(encoder->def);
        coder->duration_stream = GPUJPEG_CUSTOM_TIMER_DURATION(encoder->def);
    }
    gpujpeg_writer_emit_marker(encoder->writer, GPUJPEG_MARKER_EOI);
    
    // Set compressed image
    *image_compressed = encoder->writer->buffer;
    *image_compressed_size = encoder->writer->buffer_current - encoder->writer->buffer;
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_encoder_destroy(struct gpujpeg_encoder* encoder)
{
    assert(encoder != NULL);
    
    GPUJPEG_CUSTOM_TIMER_DESTROY(encoder->def);
    GPUJPEG_CUSTOM_TIMER_DESTROY(encoder->in_gpu);

    if ( gpujpeg_coder_deinit(&encoder->coder) != 0 )
        return -1;
    
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        if ( encoder->table_quantization[comp_type].d_table != NULL )
            cudaFree(encoder->table_quantization[comp_type].d_table);
    }
    for ( int comp_type = 0; comp_type < GPUJPEG_COMPONENT_TYPE_COUNT; comp_type++ ) {
        for ( int huff_type = 0; huff_type < GPUJPEG_HUFFMAN_TYPE_COUNT; huff_type++ ) {
            if ( encoder->d_table_huffman[comp_type][huff_type] != NULL )
                cudaFree(encoder->d_table_huffman[comp_type][huff_type]);
        }
    }
    
    if ( encoder->writer != NULL )
        gpujpeg_writer_destroy(encoder->writer);

    free(encoder);
    
    return 0;
}
