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
 
#include <libgpujpeg/gpujpeg_decoder.h>
#include "gpujpeg_preprocessor.h"
#include "gpujpeg_dct_cpu.h"
#include "gpujpeg_dct_gpu.h"
#include "gpujpeg_huffman_cpu_decoder.h"
#include "gpujpeg_huffman_gpu_decoder.h"
#include <libgpujpeg/gpujpeg_util.h>

/** Documented at declaration */
void
gpujpeg_decoder_output_set_default(struct gpujpeg_decoder_output* output)
{
    output->type = GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER;
    output->data = NULL;
    output->data_size = 0;
    output->texture = NULL;
}

/** Documented at declaration */
void
gpujpeg_decoder_output_set_custom(struct gpujpeg_decoder_output* output, uint8_t* custom_buffer)
{
    output->type = GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER;
    output->data = custom_buffer;
    output->data_size = 0;
}

/** Documented at declaration */
void
gpujpeg_decoder_output_set_texture(struct gpujpeg_decoder_output* output, struct gpujpeg_opengl_texture* texture)
{
    output->type = GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE;
    output->data = NULL;
    output->data_size = 0;
    output->texture = texture;
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
    
    // Timers
    GPUJPEG_CUSTOM_TIMER_CREATE(decoder->def);
    GPUJPEG_CUSTOM_TIMER_CREATE(decoder->in_gpu);

    return decoder;
}

/** Documented at declaration */
int
gpujpeg_decoder_init(struct gpujpeg_decoder* decoder, struct gpujpeg_parameters* param, struct gpujpeg_image_parameters* param_image)
{
    assert(param_image->comp_count == 1 || param_image->comp_count == 3);
    
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
        fprintf(stderr, "[GPUJPEG] [Error] Can't reinitialize decoder, implement if needed!\n");
        return -1;
    }
    
    coder->param = *param;
    coder->param_image = *param_image;
    
    // Initialize coder
    if ( gpujpeg_coder_init(coder) != 0 )
        return -1;
        
    // Init postprocessor
    if ( gpujpeg_preprocessor_decoder_init(&decoder->coder) != 0 ) {
        fprintf(stderr, "Failed to init postprocessor!");
        return -1;
    }
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_decoder_decode(struct gpujpeg_decoder* decoder, uint8_t* image, int image_size, struct gpujpeg_decoder_output* output)
{
    // Get coder
    struct gpujpeg_coder* coder = &decoder->coder;
    
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

    GPUJPEG_CUSTOM_TIMER_START(decoder->def);
    
    // Read JPEG image data
    if ( gpujpeg_reader_read_image(decoder, image, image_size) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Decoder failed when decoding image data!\n");
        return -1;
    }
    
    GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
    coder->duration_stream = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    GPUJPEG_CUSTOM_TIMER_START(decoder->def);
    
    // Perform huffman decoding on CPU (when restart interval is not set)
    if ( coder->param.restart_interval == 0 ) {
        if ( gpujpeg_huffman_cpu_decoder_decode(decoder) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder failed!\n");
            return -1;
        }
        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_huffman_coder = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Copy quantized data to device memory from cpu memory
        cudaMemcpy(coder->d_data_quantized, coder->data_quantized, coder->data_size * sizeof(int16_t), cudaMemcpyHostToDevice);

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_to = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);
    }
    // Perform huffman decoding on GPU (when restart interval is set)
    else {
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
        
        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_to = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        GPUJPEG_CUSTOM_TIMER_START(decoder->in_gpu);

        // Perform huffman decoding
        if ( gpujpeg_huffman_gpu_decoder_decode(decoder) != 0 ) {
            fprintf(stderr, "[GPUJPEG] [Error] Huffman decoder on GPU failed!\n");
            return -1;
        }

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_huffman_coder = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);
    }
    
    // Perform IDCT and dequantization (own CUDA implementation)
    gpujpeg_idct_gpu(decoder);
    
    GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
    coder->duration_dct_quantization = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    GPUJPEG_CUSTOM_TIMER_START(decoder->def);
    
    // Preprocessing
    if ( gpujpeg_preprocessor_decode(&decoder->coder) != 0 )
        return -1;

    GPUJPEG_CUSTOM_TIMER_STOP(decoder->in_gpu);
    coder->duration_in_gpu = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->in_gpu);

    GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
    coder->duration_preprocessor = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    
    // Set decompressed image size
    output->data_size = coder->data_raw_size * sizeof(uint8_t);
    
    // Set decompressed image
    if ( output->type == GPUJPEG_DECODER_OUTPUT_INTERNAL_BUFFER ) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Copy decompressed image to host memory
        cudaMemcpy(coder->data_raw, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        
        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

        // Set output to internal buffer
        output->data = coder->data_raw;
    } else if ( output->type == GPUJPEG_DECODER_OUTPUT_CUSTOM_BUFFER ) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);
        
        assert(output->data != NULL);

        // Copy decompressed image to host memory
        cudaMemcpy(output->data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
        
        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
    } else if ( output->type == GPUJPEG_DECODER_OUTPUT_OPENGL_TEXTURE ) {
        GPUJPEG_CUSTOM_TIMER_START(decoder->def);

        // Map OpenGL texture
        int data_size = 0;
        uint8_t* d_data = gpujpeg_opengl_texture_map(output->texture, &data_size);
        assert(data_size == coder->data_raw_size);

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_map = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

        GPUJPEG_CUSTOM_TIMER_START(decoder->def);
            
        // Copy decompressed image to texture pixel buffer object device data
        cudaMemcpy(d_data, coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToDevice);

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_from = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);

        GPUJPEG_CUSTOM_TIMER_START(decoder->def);
            
        // Unmap OpenGL texture
        gpujpeg_opengl_texture_unmap(output->texture);

        GPUJPEG_CUSTOM_TIMER_STOP(decoder->def);
        coder->duration_memory_unmap = GPUJPEG_CUSTOM_TIMER_DURATION(decoder->def);
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
    
    GPUJPEG_CUSTOM_TIMER_DESTROY(decoder->def);
    GPUJPEG_CUSTOM_TIMER_DESTROY(decoder->in_gpu);

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
