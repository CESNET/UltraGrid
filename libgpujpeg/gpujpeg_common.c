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
 
#include "gpujpeg_common.h"
#include "gpujpeg_util.h"
#include <npp.h>
#include <cuda_gl_interop.h>

/** Documented at declaration */
struct gpujpeg_devices_info
gpujpeg_get_devices_info()
{
    struct gpujpeg_devices_info devices_info;
    
    if ( cudaGetDeviceCount(&devices_info.device_count) != cudaSuccess ) {
        fprintf(stderr, "[GPUJPEG] [Error] CUDA Driver and Runtime version may be mismatched.\n");
        exit(-1);
    }
    
    if ( devices_info.device_count > GPUJPEG_MAX_DEVICE_COUNT ) {
        fprintf(stderr, "[GPUJPEG] [Warning] There are available more CUDA devices (%d) than maximum count (%d).\n",
            devices_info.device_count, GPUJPEG_MAX_DEVICE_COUNT);
        fprintf(stderr, "[GPUJPEG] [Warning] Using maximum count (%d).\n", GPUJPEG_MAX_DEVICE_COUNT);
        devices_info.device_count = GPUJPEG_MAX_DEVICE_COUNT;
    }

    for ( int device_id = 0; device_id < devices_info.device_count; device_id++ ) {
        struct cudaDeviceProp device_properties;
        cudaGetDeviceProperties(&device_properties, device_id);
        
        struct gpujpeg_device_info* device_info = &devices_info.device[device_id];
        
        device_info->id = device_id;
        strncpy(device_info->name, device_properties.name, 255);
        device_info->cc_major = device_properties.major;
        device_info->cc_minor = device_properties.minor;
        device_info->global_memory = device_properties.totalGlobalMem;
        device_info->constant_memory = device_properties.totalConstMem;
        device_info->shared_memory = device_properties.sharedMemPerBlock;
        device_info->register_count = device_properties.regsPerBlock;
#if CUDART_VERSION >= 2000
        device_info->multiprocessor_count = device_properties.multiProcessorCount;
#endif
    }
    
    return devices_info;
}

/** Documented at declaration */
void
gpujpeg_print_devices_info()
{
    struct gpujpeg_devices_info devices_info = gpujpeg_get_devices_info();
    if ( devices_info.device_count == 0 ) {
        printf("There is no device supporting CUDA.\n");
        return;
    } else if ( devices_info.device_count == 1 ) {
        printf("There is 1 device supporting CUDA:\n");
    } else {
        printf("There are %d devices supporting CUDA:\n", devices_info.device_count);
    }
    
    for ( int device_id = 0; device_id < devices_info.device_count; device_id++ ) {
        struct gpujpeg_device_info* device_info = &devices_info.device[device_id];
        printf("\nDevice %d: \"%s\"\n", device_info->id, device_info->name);
        printf("  Compute capability: %d.%d\n", device_info->cc_major, device_info->cc_minor);
        printf("  Total amount of global memory: %ld kB\n", device_info->global_memory / 1024);
        printf("  Total amount of constant memory: %d kB\n", device_info->constant_memory / 1024); 
        printf("  Total amount of shared memory per block: %d kB\n", device_info->shared_memory / 1024);
        printf("  Total number of registers available per block: %d\n", device_info->register_count);
        printf("  Multiprocessors: %d\n", device_info->multiprocessor_count);
    }
}

/** Documented at declaration */
int
gpujpeg_init_device(int device_id, int flags)
{
    int dev_count;
    cudaGetDeviceCount(&dev_count);
    if ( dev_count == 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] No CUDA enabled device\n");
        return -1;
    }

    if ( device_id < 0 || device_id >= dev_count ) {
        fprintf(stderr, "[GPUJPEG] [Error] Selected device %d is out of bound. Devices on your system are in range %d - %d\n",
            device_id, 0, dev_count - 1);
        return -1;
    }

    struct cudaDeviceProp devProp;
    if ( cudaSuccess != cudaGetDeviceProperties(&devProp, device_id) ) {
        fprintf(stderr,
            "[GPUJPEG] [Error] Can't get CUDA device properties!\n"
            "[GPUJPEG] [Error] Do you have proper driver for CUDA installed?\n"
        );
        return -1;
    }

    if ( devProp.major < 1 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Device %d does not support CUDA\n", device_id);
        return -1;
    }
    
    if ( flags & GPUJPEG_OPENGL_INTEROPERABILITY ) {
        cudaGLSetGLDevice(device_id);
        gpujpeg_cuda_check_error("Enabling OpenGL interoperability");
    }

    if ( flags & GPUJPEG_VERBOSE ) {
        int cuda_driver_version = 0;
        cudaDriverGetVersion(&cuda_driver_version);
        printf("CUDA driver version:   %d.%d\n", cuda_driver_version / 1000, (cuda_driver_version % 100) / 10);
        
        int cuda_runtime_version = 0;
        cudaRuntimeGetVersion(&cuda_runtime_version);
        printf("CUDA runtime version:  %d.%d\n", cuda_runtime_version / 1000, (cuda_runtime_version % 100) / 10);
        
        const NppLibraryVersion* npp_version = nppGetLibVersion();
        printf("NPP version:           %d.%d\n", npp_version->major, npp_version->minor);
        
        printf("Using Device #%d:       %s (c.c. %d.%d)\n", device_id, devProp.name, devProp.major, devProp.minor);
    }
    
    cudaSetDevice(device_id);

    return 0;
}

/** Documented at declaration */
void
gpujpeg_set_default_parameters(struct gpujpeg_parameters* param)
{
    param->verbose = 0;
    param->quality = 75;
    param->restart_interval = 8;
    param->interleaved = 0;
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ ) {
        param->sampling_factor[comp].horizontal = 1;
        param->sampling_factor[comp].vertical = 1;
    }
}

/** Documented at declaration */
void
gpujpeg_parameters_chroma_subsampling(struct gpujpeg_parameters* param)
{
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ ) {
        if ( comp == 0 ) {
            param->sampling_factor[comp].horizontal = 2;
            param->sampling_factor[comp].vertical = 2;
        } else {
            param->sampling_factor[comp].horizontal = 1;
            param->sampling_factor[comp].vertical = 1;
        }
    }
}

/** Documented at declaration */
void
gpujpeg_image_set_default_parameters(struct gpujpeg_image_parameters* param)
{
    param->width = 0;
    param->height = 0;
    param->comp_count = 3;
    param->color_space = GPUJPEG_RGB;
    param->sampling_factor = GPUJPEG_4_4_4;
}

/** Documented at declaration */
enum gpujpeg_image_file_format
gpujpeg_image_get_file_format(const char* filename)
{
    static const char *extension[] = { "raw", "rgb", "yuv", "jpg" };
    static const enum gpujpeg_image_file_format format[] = { GPUJPEG_IMAGE_FILE_RAW, GPUJPEG_IMAGE_FILE_RGB, GPUJPEG_IMAGE_FILE_YUV, GPUJPEG_IMAGE_FILE_JPEG };
        
    char * ext = strrchr(filename, '.');
    if ( ext == NULL )
        return -1;
    ext++;
    for ( int i = 0; i < sizeof(format) / sizeof(*format); i++ ) {
        if ( strncasecmp(ext, extension[i], 3) == 0 ) {
            return format[i];
        }
    }
    return GPUJPEG_IMAGE_FILE_UNKNOWN;
}

/** Documented at declaration */
void
gpujpeg_component_print8(struct gpujpeg_component* component, uint8_t* d_data)
{
    int data_size = component->data_width * component->data_height;
    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(uint8_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < component->data_height; y++ ) {
        for ( int x = 0; x < component->data_width; x++ ) {
            printf("%3u ", data[y * component->data_width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/** Documented at declaration */
void
gpujpeg_component_print16(struct gpujpeg_component* component, int16_t* d_data)
{
    int data_size = component->data_width * component->data_height;
    int16_t* data = NULL;
    cudaMallocHost((void**)&data, data_size * sizeof(int16_t)); 
    cudaMemcpy(data, d_data, data_size * sizeof(int16_t), cudaMemcpyDeviceToHost);
    
    printf("Print Data\n");
    for ( int y = 0; y < component->data_height; y++ ) {
        for ( int x = 0; x < component->data_width; x++ ) {
            printf("%3d ", data[y * component->data_width + x]);
        }
        printf("\n");
    }
    cudaFreeHost(data);
}

/** Documented at declaration */
int
gpujpeg_coder_init(struct gpujpeg_coder* coder)
{
    int result = 1;
    
    coder->preprocessor = NULL;
    
    // Allocate color components
    cudaMallocHost((void**)&coder->component, coder->param_image.comp_count * sizeof(struct gpujpeg_component));
    if ( coder->component == NULL )
        result = 0;
    // Allocate color components in device memory
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_component, coder->param_image.comp_count * sizeof(struct gpujpeg_component)) )
        result = 0;
    gpujpeg_cuda_check_error("Coder color component allocation");
        
    // Initialize sampling factors and compute maximum sampling factor to coder->sampling_factor
    coder->sampling_factor.horizontal = 0;
    coder->sampling_factor.vertical = 0;
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        assert(coder->param.sampling_factor[comp].horizontal >= 1 && coder->param.sampling_factor[comp].horizontal <= 15);
        assert(coder->param.sampling_factor[comp].vertical >= 1 && coder->param.sampling_factor[comp].vertical <= 15);
        coder->component[comp].sampling_factor = coder->param.sampling_factor[comp];
        if ( coder->component[comp].sampling_factor.horizontal > coder->sampling_factor.horizontal )
            coder->sampling_factor.horizontal = coder->component[comp].sampling_factor.horizontal;
        if ( coder->component[comp].sampling_factor.vertical > coder->sampling_factor.vertical )
            coder->sampling_factor.vertical = coder->component[comp].sampling_factor.vertical;
    }
    
    // Calculate data size
    coder->data_raw_size = gpujpeg_image_calculate_size(&coder->param_image);
    coder->data_size = 0;
    
    // Initialize color components
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        // Get component
        struct gpujpeg_component* component = &coder->component[comp];
        
        // Set type
        component->type = (comp == 0) ? GPUJPEG_COMPONENT_LUMINANCE : GPUJPEG_COMPONENT_CHROMINANCE;
        
        // Set proper color component sizes in pixels based on sampling factors
        int samp_factor_h = component->sampling_factor.horizontal;
        int samp_factor_v = component->sampling_factor.vertical;
        component->width = (coder->param_image.width * samp_factor_h) / coder->sampling_factor.horizontal;
        component->height = (coder->param_image.height * samp_factor_v) / coder->sampling_factor.vertical;
        
        // Compute component MCU size
        component->mcu_size_x = GPUJPEG_BLOCK_SIZE;
        component->mcu_size_y = GPUJPEG_BLOCK_SIZE;
        if ( coder->param.interleaved == 1 ) {
            component->mcu_compressed_size = GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE * samp_factor_h * samp_factor_v;
            component->mcu_size_x *= samp_factor_h;
            component->mcu_size_y *= samp_factor_v;
        } else {
            component->mcu_compressed_size = GPUJPEG_MAX_BLOCK_COMPRESSED_SIZE;
        }
        component->mcu_size = component->mcu_size_x * component->mcu_size_y;
        
        // Compute allocated data size
        component->data_width = gpujpeg_div_and_round_up(component->width, component->mcu_size_x) * component->mcu_size_x;
        component->data_height = gpujpeg_div_and_round_up(component->height, component->mcu_size_y) * component->mcu_size_y;
        component->data_size = component->data_width * component->data_height;
        // Increase total data size
        coder->data_size += component->data_size;
        
        // Compute component MCU count
        component->mcu_count_x = gpujpeg_div_and_round_up(component->data_width, component->mcu_size_x);
        component->mcu_count_y = gpujpeg_div_and_round_up(component->data_height, component->mcu_size_y);
        component->mcu_count = component->mcu_count_x * component->mcu_count_y;
        
        // Compute MCU count per segment
        component->segment_mcu_count = coder->param.restart_interval;
        if ( component->segment_mcu_count == 0 ) {
            // If restart interval is disabled, restart interval is equal MCU count
            component->segment_mcu_count = component->mcu_count;
        }
        
        // Calculate segment count
        component->segment_count = gpujpeg_div_and_round_up(component->mcu_count, component->segment_mcu_count);
        
        //printf("Subsampling %dx%d, Resolution %d, %d, mcu size %d, mcu count %d\n",
        //    coder->param.sampling_factor[comp].horizontal, coder->param.sampling_factor[comp].vertical,
        //    component->data_width, component->data_height,
        //    component->mcu_compressed_size, component->mcu_count
        //);
    }
    
    // Maximum component data size for allocated buffers
    coder->data_width = gpujpeg_div_and_round_up(coder->param_image.width, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;
    coder->data_height = gpujpeg_div_and_round_up(coder->param_image.height, GPUJPEG_BLOCK_SIZE) * GPUJPEG_BLOCK_SIZE;
    
    // Allocate data buffers for all color components
    if ( cudaSuccess != cudaMallocHost((void**)&coder->data_raw, coder->data_raw_size * sizeof(uint8_t)) ) 
        return -1;
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_data_raw, coder->data_raw_size * sizeof(uint8_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_data, coder->data_size * sizeof(uint8_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMallocHost((void**)&coder->data_quantized, coder->data_size * sizeof(int16_t)) ) 
        result = 0;
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_data_quantized, coder->data_size * sizeof(int16_t)) ) 
        result = 0;
    gpujpeg_cuda_check_error("Coder data allocation");
    
    // Set data buffer to color components
    uint8_t* d_comp_data = coder->d_data;
    int16_t* d_comp_data_quantized = coder->d_data_quantized;
    int16_t* comp_data_quantized = coder->data_quantized;
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        struct gpujpeg_component* component = &coder->component[comp];
        component->d_data = d_comp_data;
        component->d_data_quantized = d_comp_data_quantized;
        component->data_quantized = comp_data_quantized;
        d_comp_data += component->data_width * component->data_height;
        d_comp_data_quantized += component->data_width * component->data_height;
        comp_data_quantized += component->data_width * component->data_height;
    }
    
    // Copy components to device memory
    if ( cudaSuccess != cudaMemcpy(coder->d_component, coder->component, coder->param_image.comp_count * sizeof(struct gpujpeg_component), cudaMemcpyHostToDevice) )
        result = 0;
    gpujpeg_cuda_check_error("Coder component copy");
    
    // Compute MCU size, MCU count, segment count and compressed data allocation size
    coder->mcu_count = 0;
    coder->mcu_size = 0;
    coder->mcu_compressed_size = 0;
    coder->segment_count = 0;
    coder->data_compressed_size = 0;
    if ( coder->param.interleaved == 1 ) {
        assert(coder->param_image.comp_count > 0);
        coder->mcu_count = coder->component[0].mcu_count;
        coder->segment_count = coder->component[0].segment_count;
        coder->segment_mcu_count = coder->component[0].segment_mcu_count;
        for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];
            assert(coder->mcu_count == component->mcu_count);
            assert(coder->segment_mcu_count == component->segment_mcu_count);
            coder->mcu_size += component->mcu_size;
            coder->mcu_compressed_size += component->mcu_compressed_size;
        }
    } else {
        assert(coder->param_image.comp_count > 0);
        coder->mcu_size = coder->component[0].mcu_size;
        coder->mcu_compressed_size = coder->component[0].mcu_compressed_size;
        coder->segment_mcu_count = 0;
        for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
            struct gpujpeg_component* component = &coder->component[comp];
            assert(coder->mcu_size == component->mcu_size);
            assert(coder->mcu_compressed_size == component->mcu_compressed_size);
            coder->mcu_count += component->mcu_count;
            coder->segment_count += component->segment_count;
        }
    }
    //printf("mcu size %d -> %d, mcu count %d, segment mcu count %d\n", coder->mcu_size, coder->mcu_compressed_size, coder->mcu_count, coder->segment_mcu_count);

    // Allocate segments
    cudaMallocHost((void**)&coder->segment, coder->segment_count * sizeof(struct gpujpeg_segment));
    if ( coder->segment == NULL )
        result = 0;
    // Allocate segments in device memory
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_segment, coder->segment_count * sizeof(struct gpujpeg_segment)) )
        result = 0;
    gpujpeg_cuda_check_error("Coder segment allocation");
    
    // Prepare segments
    if ( result == 1 ) {            
        // While preparing segments compute input size and compressed size
        int data_index = 0;
        int data_compressed_index = 0;
        
        // Prepare segments based on (non-)interleaved mode
        if ( coder->param.interleaved == 1 ) {
            // Prepare segments for encoding (only one scan for all color components)
            int mcu_index = 0;
            for ( int index = 0; index < coder->segment_count; index++ ) {
                // Prepare segment MCU count
                int mcu_count = coder->segment_mcu_count;
                if ( (mcu_index + mcu_count) >= coder->mcu_count )
                    mcu_count = coder->mcu_count - mcu_index;
                // Set parameters for segment
                coder->segment[index].scan_index = 0;
                coder->segment[index].scan_segment_index = index;
                coder->segment[index].mcu_count = mcu_count;
                coder->segment[index].data_compressed_index = data_compressed_index;
                coder->segment[index].data_compressed_size = 0;
                // Increase parameters for next segment
                data_index += mcu_count * coder->mcu_size;
                data_compressed_index += mcu_count * coder->mcu_compressed_size;
                mcu_index += mcu_count;
            }
        } else {
            // Prepare segments for encoding (one scan for each color component)
            int index = 0;
            for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
                // Get component
                struct gpujpeg_component* component = &coder->component[comp];
                // Prepare component segments
                int mcu_index = 0;
                for ( int segment = 0; segment < component->segment_count; segment++ ) {
                    // Prepare segment MCU count
                    int mcu_count = component->segment_mcu_count;
                    if ( (mcu_index + mcu_count) >= component->mcu_count )
                        mcu_count = component->mcu_count - mcu_index;
                    // Set parameters for segment
                    coder->segment[index].scan_index = comp;
                    coder->segment[index].scan_segment_index = segment;
                    coder->segment[index].mcu_count = mcu_count;
                    coder->segment[index].data_compressed_index = data_compressed_index;
                    coder->segment[index].data_compressed_size = 0;
                    // Increase parameters for next segment
                    data_index += mcu_count * component->mcu_size;
                    data_compressed_index += mcu_count * component->mcu_compressed_size;
                    mcu_index += mcu_count;
                    index++;
                }
            }
        }
        
        // Check data size
        //printf("%d == %d\n", coder->data_size, data_index);
        assert(coder->data_size == data_index);
            
        // Set compressed size
        coder->data_compressed_size = data_compressed_index;
    }
    //printf("Compressed size %d (segments %d)\n", coder->data_compressed_size, coder->segment_count);
        
    // Copy segments to device memory
    if ( cudaSuccess != cudaMemcpy(coder->d_segment, coder->segment, coder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice) )
        result = 0;
        
    // Allocate compressed data
    int max_compressed_data_size = coder->data_compressed_size;
    max_compressed_data_size += GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
    max_compressed_data_size *= 2;
    if ( cudaSuccess != cudaMallocHost((void**)&coder->data_compressed, max_compressed_data_size * sizeof(uint8_t)) ) 
        result = 0;   
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_data_compressed, max_compressed_data_size * sizeof(uint8_t)) ) 
        result = 0;   
    gpujpeg_cuda_check_error("Coder data compressed allocation");
     
    return 0;
}

/** Documented at declaration */
int
gpujpeg_coder_deinit(struct gpujpeg_coder* coder)
{
    if ( coder->data_raw != NULL )
        cudaFreeHost(coder->data_raw);
    if ( coder->d_data_raw != NULL )
        cudaFree(coder->d_data_raw);
    if ( coder->d_data != NULL )
        cudaFree(coder->d_data);
    if ( coder->data_quantized != NULL )
        cudaFreeHost(coder->data_quantized);    
    if ( coder->d_data_quantized != NULL )
        cudaFree(coder->d_data_quantized);    
    if ( coder->data_compressed != NULL )
        cudaFreeHost(coder->data_compressed);    
    if ( coder->d_data_compressed != NULL )
        cudaFree(coder->d_data_compressed);    
    if ( coder->segment != NULL )
        cudaFreeHost(coder->segment); 
    if ( coder->d_segment != NULL )
        cudaFree(coder->d_segment);
    return 0;
}

/** Documented at declaration */
int
gpujpeg_image_calculate_size(struct gpujpeg_image_parameters* param)
{
    assert(param->comp_count == 3);
    
    int image_size = 0;
    if ( param->sampling_factor == GPUJPEG_4_4_4 ) {
        image_size = param->width * param->height * param->comp_count;
    } else if ( param->sampling_factor == GPUJPEG_4_2_2 ) {
        int width = gpujpeg_div_and_round_up(param->width, 2) * 2 + 0;
        image_size = (width * param->height) * 2;
    } else {
        assert(0);
    }
    return image_size;
}

/** Documented at declaration */
int
gpujpeg_image_load_from_file(const char* filename, uint8_t** image, int* image_size)
{
    FILE* file;
    file = fopen(filename, "rb");
    if ( !file ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed open %s for reading!\n", filename);
        return -1;
    }

    if ( *image_size == 0 ) {
        fseek(file, 0, SEEK_END);
        *image_size = ftell(file);
        rewind(file);
    }
    
    uint8_t* data = NULL;
    cudaMallocHost((void**)&data, *image_size * sizeof(uint8_t));
    if ( *image_size != fread(data, sizeof(uint8_t), *image_size, file) ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to load image data [%d bytes] from file %s!\n", *image_size, filename);
        return -1;
    }
    fclose(file);
    
    *image = data;
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_image_save_to_file(const char* filename, uint8_t* image, int image_size)
{
    FILE* file;
    file = fopen(filename, "wb");
    if ( !file ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed open %s for writing!\n", filename);
        return -1;
    }
    
    if ( image_size != fwrite(image, sizeof(uint8_t), image_size, file) ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to write image data [%d bytes] to file %s!\n", image_size, filename);
        return -1;
    }
    fclose(file);
    
    return 0;
}

/** Documented at declaration */
int
gpujpeg_image_destroy(uint8_t* image)
{
    cudaFreeHost(image);

    return 0;
}
