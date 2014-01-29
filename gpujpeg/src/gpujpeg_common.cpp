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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif /* HAVE_CONFIG_H */
 
#include <algorithm>
#include <ctype.h>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_util.h>
#include "gpujpeg_preprocessor.h"
#include <math.h>
#ifdef GPUJPEG_USE_OPENGL
    #define GL_GLEXT_PROTOTYPES
    #include <GL/gl.h>
    #include <GL/glx.h>
    #include <cuda_gl_interop.h>
#endif

// rounds number of segment bytes up to next multiple of 128
#define SEGMENT_ALIGN(b) (((b) + 127) & ~127)

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
        printf("\nDevice #%d: \"%s\"\n", device_info->id, device_info->name);
        printf("  Compute capability: %d.%d\n", device_info->cc_major, device_info->cc_minor);
        printf("  Total amount of global memory: %lu kB\n", device_info->global_memory / 1024);
        printf("  Total amount of constant memory: %lu kB\n", device_info->constant_memory / 1024); 
        printf("  Total amount of shared memory per block: %lu kB\n", device_info->shared_memory / 1024);
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
    
#ifdef GPUJPEG_USE_OPENGL
    if ( flags & GPUJPEG_OPENGL_INTEROPERABILITY ) {
        cudaGLSetGLDevice(device_id);
        gpujpeg_cuda_check_error("Enabling OpenGL interoperability");
    }
#endif

    if ( flags & GPUJPEG_VERBOSE ) {
        int cuda_driver_version = 0;
        cudaDriverGetVersion(&cuda_driver_version);
        printf("CUDA driver version:   %d.%d\n", cuda_driver_version / 1000, (cuda_driver_version % 100) / 10);
        
        int cuda_runtime_version = 0;
        cudaRuntimeGetVersion(&cuda_runtime_version);
        printf("CUDA runtime version:  %d.%d\n", cuda_runtime_version / 1000, (cuda_runtime_version % 100) / 10);
        
        printf("Using Device #%d:       %s (c.c. %d.%d)\n", device_id, devProp.name, devProp.major, devProp.minor);
    }
    
    cudaSetDevice(device_id);
    gpujpeg_cuda_check_error("Set CUDA device");

    // Test by simple copying that the device is ready
    uint8_t data[] = {8};
    uint8_t* d_data = NULL;
    cudaMalloc((void**)&d_data, 1);
    cudaMemcpy(d_data, data, 1, cudaMemcpyHostToDevice);
    cudaFree(d_data);
    cudaError_t error = cudaGetLastError();
    if ( cudaSuccess != error ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to initialize CUDA device.\n");
        if ( flags & GPUJPEG_OPENGL_INTEROPERABILITY )
            fprintf(stderr, "[GPUJPEG] [Info]  OpenGL interoperability is used, is OpenGL context available?\n");
        return -1;
    }

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
    param->segment_info = 0;
    for ( int comp = 0; comp < GPUJPEG_MAX_COMPONENT_COUNT; comp++ ) {
        param->sampling_factor[comp].horizontal = 1;
        param->sampling_factor[comp].vertical = 1;
    }
    param->color_space_internal = GPUJPEG_YCBCR_BT601_256LVLS;
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
    static const char *extension[] = { "raw", "rgb", "yuv", "r", "jpg" };
    static const enum gpujpeg_image_file_format format[] = {
        GPUJPEG_IMAGE_FILE_RAW,
        GPUJPEG_IMAGE_FILE_RGB,
        GPUJPEG_IMAGE_FILE_YUV,
        GPUJPEG_IMAGE_FILE_GRAY,
        GPUJPEG_IMAGE_FILE_JPEG
    };
        
    const char * ext = strrchr(filename, '.');
    if ( ext == NULL )
        return GPUJPEG_IMAGE_FILE_UNKNOWN;
    ext++;
    char ext_lc[3];
    strncpy(ext_lc, ext, 3);
    std::transform(ext_lc, ext_lc + sizeof(ext_lc), ext_lc, ::tolower);
    for ( int i = 0; i < sizeof(format) / sizeof(*format); i++ ) {
        if ( strncmp(ext_lc, extension[i], 3) == 0 ) {
            return format[i];
        }
    }
    return GPUJPEG_IMAGE_FILE_UNKNOWN;
}

void gpujpeg_set_device(int index)
{
    cudaSetDevice(index);
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
    
    // Get info about the device
    struct cudaDeviceProp device_properties;
    int device_idx;
    cudaGetDevice(&device_idx);
    cudaGetDeviceProperties(&device_properties, device_idx);
    gpujpeg_cuda_check_error("Device info getting");
    coder->cuda_cc_major = device_properties.major;
    coder->cuda_cc_minor = device_properties.minor;
    
    if ( device_properties.major < 2 ) {
            fprintf(stderr, "GPUJPEG coder is currently broken on "
                            "cards with cc < 2.0\n");
            return 1;
    }

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
                coder->segment[index].data_temp_index = data_compressed_index;
                coder->segment[index].data_compressed_size = 0;
                // Increase parameters for next segment
                data_index += mcu_count * coder->mcu_size;
                data_compressed_index += SEGMENT_ALIGN(mcu_count * coder->mcu_compressed_size);
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
                    coder->segment[index].data_temp_index = data_compressed_index;
                    coder->segment[index].data_compressed_size = 0;
                    // Increase parameters for next segment
                    data_index += mcu_count * component->mcu_size;
                    data_compressed_index += SEGMENT_ALIGN(mcu_count * component->mcu_compressed_size);
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
        
    // Print allocation info
    if ( coder->param.verbose ) {
        int structures_size = 0;
        structures_size += coder->segment_count * sizeof(struct gpujpeg_segment);
        structures_size += coder->param_image.comp_count * sizeof(struct gpujpeg_component);
        int total_size = 0;
        total_size += structures_size;
        total_size += coder->data_raw_size;
        total_size += coder->data_size;
        total_size += coder->data_size * 2;
        total_size += coder->data_compressed_size;  // for Huffman coding output
        total_size += coder->data_compressed_size;  // for Hiffman coding temp buffer

        printf("\nAllocation Info:\n");
        printf("    Segment Count:            %d\n", coder->segment_count);
        printf("    Allocated Data Size:      %dx%d\n", coder->data_width, coder->data_height);
        printf("    Raw Buffer Size:          %0.1f MB\n", (double)coder->data_raw_size / (1024.0 * 1024.0));
        printf("    Preprocessor Buffer Size: %0.1f MB\n", (double)coder->data_size / (1024.0 * 1024.0));
        printf("    DCT Buffer Size:          %0.1f MB\n", (double)2 * coder->data_size / (1024.0 * 1024.0));
        printf("    Compressed Buffer Size:   %0.1f MB\n", (double)coder->data_compressed_size / (1024.0 * 1024.0));
        printf("    Huffman Temp buffer Size: %0.1f MB\n", (double)coder->data_compressed_size / (1024.0 * 1024.0));
        printf("    Structures Size:          %0.1f kB\n", (double)structures_size / (1024.0));
        printf("    Total GPU Memory Size:    %0.1f MB\n", (double)total_size / (1024.0 * 1024.0));
        printf("");
    }

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
    unsigned int data_quantized_index = 0;
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ ) {
        struct gpujpeg_component* component = &coder->component[comp];
        component->d_data = d_comp_data;
        component->d_data_quantized = d_comp_data_quantized;
        component->data_quantized_index = data_quantized_index;
        component->data_quantized = comp_data_quantized;
        d_comp_data += component->data_width * component->data_height;
        d_comp_data_quantized += component->data_width * component->data_height;
        comp_data_quantized += component->data_width * component->data_height;
        data_quantized_index += component->data_width * component->data_height;
     }

    // Copy components to device memory
    if ( cudaSuccess != cudaMemcpy(coder->d_component, coder->component, coder->param_image.comp_count * sizeof(struct gpujpeg_component), cudaMemcpyHostToDevice) )
        result = 0;
    gpujpeg_cuda_check_error("Coder component copy");
        
    // Allocate compressed data
    int max_compressed_data_size = coder->data_compressed_size;
    max_compressed_data_size += GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
    //max_compressed_data_size *= 2;
    if ( cudaSuccess != cudaMallocHost((void**)&coder->data_compressed, max_compressed_data_size * sizeof(uint8_t)) ) 
        result = 0;   
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_data_compressed, max_compressed_data_size * sizeof(uint8_t)) ) 
        result = 0;   
    gpujpeg_cuda_check_error("Coder data compressed allocation");
    
    // Allocate Huffman coder temporary buffer
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_temp_huffman, max_compressed_data_size * sizeof(uint8_t)) ) 
        result = 0;   
    gpujpeg_cuda_check_error("Huffman temp buffer allocation");
     
    // Initialize block lists in host memory
    coder->block_count = 0;
    for ( int comp = 0; comp < coder->param_image.comp_count; comp++ )
        coder->block_count += (coder->component[comp].data_width * coder->component[comp].data_height) / (8 * 8);
    if ( cudaSuccess != cudaMallocHost((void**)&coder->block_list, coder->block_count * sizeof(*coder->block_list)) ) 
        result = 0;   
    if ( cudaSuccess != cudaMalloc((void**)&coder->d_block_list, coder->block_count * sizeof(*coder->d_block_list)) ) 
        result = 0;
    if( result == 0 )
        return 0;
    int block_idx = 0;
    int comp_count = 1;
    if ( coder->param.interleaved == 1 )
        comp_count = coder->param_image.comp_count;
    assert(comp_count >= 1 && comp_count <= GPUJPEG_MAX_COMPONENT_COUNT);
    for( int segment_idx = 0; segment_idx < coder->segment_count; segment_idx++ ) {
        struct gpujpeg_segment* const segment = &coder->segment[segment_idx];
        segment->block_index_list_begin = block_idx;
        
        // Non-interleaving mode
        if ( comp_count == 1 ) {
            // Inspect MCUs in segment
            for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
                // Component for the scan
                struct gpujpeg_component* component = &coder->component[segment->scan_index];
         
                // Offset of component data for MCU
                uint64_t data_index = component->data_quantized_index + (segment->scan_segment_index * component->segment_mcu_count + mcu_index) * component->mcu_size;
                uint64_t component_type = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00 : 0x80;
                uint64_t dc_index = segment->scan_index;
                coder->block_list[block_idx++] = dc_index | component_type | (data_index << 8);
            } 
        }
        // Interleaving mode
        else {
            // Encode MCUs in segment
            for ( int mcu_index = 0; mcu_index < segment->mcu_count; mcu_index++ ) {
                //assert(segment->scan_index == 0);
                for ( int comp = 0; comp < comp_count; comp++ ) {
                    struct gpujpeg_component* component = &coder->component[comp];

                    // Prepare mcu indexes
                    int mcu_index_x = (segment->scan_segment_index * component->segment_mcu_count + mcu_index) % component->mcu_count_x;
                    int mcu_index_y = (segment->scan_segment_index * component->segment_mcu_count + mcu_index) / component->mcu_count_x;
                    // Compute base data index
                    int data_index_base = component->data_quantized_index + mcu_index_y * (component->mcu_size * component->mcu_count_x) + mcu_index_x * (component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                    
                    // For all vertical 8x8 blocks
                    for ( int y = 0; y < component->sampling_factor.vertical; y++ ) {
                        // Compute base row data index
                        int data_index_row = data_index_base + y * (component->mcu_count_x * component->mcu_size_x * GPUJPEG_BLOCK_SIZE);
                        // For all horizontal 8x8 blocks
                        for ( int x = 0; x < component->sampling_factor.horizontal; x++ ) {
                            // Compute 8x8 block data index
                            uint64_t data_index = data_index_row + x * GPUJPEG_BLOCK_SIZE * GPUJPEG_BLOCK_SIZE;
                            uint64_t component_type = component->type == GPUJPEG_COMPONENT_LUMINANCE ? 0x00 : 0x80;
                            uint64_t dc_index = comp;
                            coder->block_list[block_idx++] = dc_index | component_type | (data_index << 8);
                        }
                    }
                }
            }
        }
        segment->block_count = block_idx - segment->block_index_list_begin;
    }
    assert(block_idx == coder->block_count);
    
    // Copy block lists to device memory
    if ( cudaSuccess != cudaMemcpy(coder->d_block_list, coder->block_list, coder->block_count * sizeof(*coder->d_block_list), cudaMemcpyHostToDevice) )
        result = 0;
    
    // Copy segments to device memory
    if ( cudaSuccess != cudaMemcpy(coder->d_segment, coder->segment, coder->segment_count * sizeof(struct gpujpeg_segment), cudaMemcpyHostToDevice) )
        result = 0;
    
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
    if ( coder->d_temp_huffman != NULL )
        cudaFree(coder->d_temp_huffman);
    if ( coder->block_list != NULL )
        cudaFreeHost(coder->block_list); 
    if ( coder->d_block_list != NULL )
        cudaFree(coder->d_block_list); 
    return 0;
}

/** Documented at declaration */
int
gpujpeg_image_calculate_size(struct gpujpeg_image_parameters* param)
{
    int image_size = 0;
    if ( param->sampling_factor == GPUJPEG_4_4_4 ) {
        image_size = param->width * param->height * param->comp_count;
    } else if ( param->sampling_factor == GPUJPEG_4_2_2 ) {
        assert(param->comp_count == 3);
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

/** Documented at declaration */
void
gpujpeg_image_range_info(const char* filename, int width, int height, enum gpujpeg_sampling_factor sampling_factor)
{
    // Load image
    int image_size = 0;
    uint8_t* image = NULL;
    if ( gpujpeg_image_load_from_file(filename, &image, &image_size) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to load image [%s]!\n", filename);
        return;
    }

    int c_min[3] = {256, 256, 256};
    int c_max[3] = {0, 0, 0};

    if ( sampling_factor == GPUJPEG_4_4_4 ) {
        uint8_t* in_ptr = image;
        for ( int i = 0; i < width * height; i++ ) {
            for ( int c = 0; c < 3; c++ ) {
                if ( in_ptr[c] < c_min[c] )
                    c_min[c] = in_ptr[c];
                if ( in_ptr[c] > c_max[c] )
                    c_max[c] = in_ptr[c];
            }
            in_ptr += 3;
        }
    } else if ( sampling_factor == GPUJPEG_4_2_2 ) {
        uint8_t* in_ptr = image;
        for ( int i = 0; i < width * height; i++ ) {
            if ( in_ptr[1] < c_min[0] )
                c_min[0] = in_ptr[1];
            if ( in_ptr[1] > c_max[0] )
                c_max[0] = in_ptr[1];
            if ( i % 2 == 1 ) {
                if ( in_ptr[0] < c_min[1] )
                    c_min[1] = in_ptr[0];
                if ( in_ptr[0] > c_max[1] )
                    c_max[1] = in_ptr[0];
            } else {
                if ( in_ptr[0] < c_min[2] )
                    c_min[2] = in_ptr[0];
                if ( in_ptr[0] > c_max[2] )
                    c_max[2] = in_ptr[0];
            }

            in_ptr += 2;
        }
    } else {
        assert(0);
    }

    printf("Image Samples Range:\n");
    for ( int c = 0; c < 3; c++ ) {
        printf("Component %d: %d - %d\n", c + 1, c_min[c], c_max[c]);
    }

    // Destroy image
    gpujpeg_image_destroy(image);
}

/** Documented at declaration */
void
gpujpeg_image_convert(const char* input, const char* output, struct gpujpeg_image_parameters param_image_from,
        struct gpujpeg_image_parameters param_image_to)
{
    assert(param_image_from.width == param_image_to.width);
    assert(param_image_from.height == param_image_to.height);
    assert(param_image_from.comp_count == param_image_to.comp_count);

    // Load image
    int image_size = gpujpeg_image_calculate_size(&param_image_from);
    uint8_t* image = NULL;
    if ( gpujpeg_image_load_from_file(input, &image, &image_size) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to load image [%s]!\n", input);
        return;
    }

    struct gpujpeg_coder coder;
    gpujpeg_set_default_parameters(&coder.param);
    coder.param.color_space_internal = GPUJPEG_RGB;

    // Initialize coder and preprocessor
    coder.param_image = param_image_from;
    assert(gpujpeg_coder_init(&coder) == 0);
    assert(gpujpeg_preprocessor_encoder_init(&coder) == 0);
    // Perform preprocessor
    assert(cudaMemcpy(coder.d_data_raw, image, coder.data_raw_size * sizeof(uint8_t), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(gpujpeg_preprocessor_encode(&coder) == 0);
    // Save preprocessor result
    uint8_t* buffer = NULL;
    assert(cudaMallocHost((void**)&buffer, coder.data_size * sizeof(uint8_t)) == cudaSuccess);
    assert(buffer != NULL);
    assert(cudaMemcpy(buffer, coder.d_data, coder.data_size * sizeof(uint8_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    // Deinitialize decoder
    gpujpeg_coder_deinit(&coder);

    // Initialize coder and postprocessor
    coder.param_image = param_image_to;
    assert(gpujpeg_coder_init(&coder) == 0);
    assert(gpujpeg_preprocessor_decoder_init(&coder) == 0);
    // Perform postprocessor
    assert(cudaMemcpy(coder.d_data, buffer, coder.data_size * sizeof(uint8_t), cudaMemcpyHostToDevice) == cudaSuccess);
    assert(gpujpeg_preprocessor_decode(&coder) == 0);
    // Save preprocessor result
    assert(cudaMemcpy(coder.data_raw, coder.d_data_raw, coder.data_raw_size * sizeof(uint8_t), cudaMemcpyDeviceToHost) == cudaSuccess);
    if ( gpujpeg_image_save_to_file(output, coder.data_raw, coder.data_raw_size) != 0 ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to save image [%s]!\n", output);
        return;
    }
    // Deinitialize decoder
    gpujpeg_coder_deinit(&coder);
}

/** Documented at declaration */
int
gpujpeg_opengl_init()
{
#ifdef GPUJPEG_USE_OPENGL
    // Open display
    Display* glx_display = XOpenDisplay(0);
    if ( glx_display == NULL ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to open X display!\n");
        return -1;
    }

    // Choose visual
    static int attributes[] = {
        GLX_RGBA,
        GLX_DOUBLEBUFFER,
        GLX_RED_SIZE,   1,
        GLX_GREEN_SIZE, 1,
        GLX_BLUE_SIZE,  1,
        None
    };
    XVisualInfo* visual = glXChooseVisual(glx_display, DefaultScreen(glx_display), attributes);
    if ( visual == NULL ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to choose visual!\n");
        return -1;
    }

    // Create OpenGL context
    GLXContext glx_context = glXCreateContext(glx_display, visual, 0, GL_TRUE);
    if ( glx_context == NULL ) {
        fprintf(stderr, "[GPUJPEG] [Error] Failed to create OpenGL context!\n");
        return -1;
    }

    // Create window
    Colormap colormap = XCreateColormap(glx_display, RootWindow(glx_display, visual->screen), visual->visual, AllocNone);
    XSetWindowAttributes swa;
    swa.colormap = colormap;
    swa.border_pixel = 0;
    Window glx_window = XCreateWindow(
        glx_display,
        RootWindow(glx_display, visual->screen),
        0, 0, 640, 480,
        0, visual->depth, InputOutput, visual->visual,
        CWBorderPixel | CWColormap | CWEventMask,
        &swa
    );
    // Do not map window to display to keep it hidden
    //XMapWindow(glx_display, glx_window);

    glXMakeCurrent(glx_display, glx_window, glx_context);
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif
    return 0;
}

/** Documented at declaration */
int
gpujpeg_opengl_texture_create(int width, int height, uint8_t* data)
{
    int texture_id = 0;

#ifdef GPUJPEG_USE_OPENGL
    glGenTextures(1, &texture_id);
    glBindTexture(GL_TEXTURE_2D, texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif

    return texture_id;
}

/** Documented at declaration */
int
gpujpeg_opengl_texture_set_data(int texture_id, uint8_t* data)
{
#ifdef GPUJPEG_USE_OPENGL
    glBindTexture(GL_TEXTURE_2D, texture_id);

    int width = 0;
    int height = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
    assert(width != 0 && height != 0);

    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif
    return 0;
}

/** Documented at declaration */
int
gpujpeg_opengl_texture_get_data(int texture_id, uint8_t* data, int* data_size)
{
#ifdef GPUJPEG_USE_OPENGL
    glBindTexture(GL_TEXTURE_2D, texture_id);

    int width = 0;
    int height = 0;
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &height);
    assert(width != 0 && height != 0);

    glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    if ( data_size != NULL )
        *data_size = width * height * 3;

    glBindTexture(GL_TEXTURE_2D, 0);
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif
    return 0;
}

/** Documented at declaration */
void
gpujpeg_opengl_texture_destroy(int texture_id)
{
#ifdef GPUJPEG_USE_OPENGL
    glDeleteTextures(1, &texture_id);
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif
}

/** Documented at declaration */
struct gpujpeg_opengl_texture*
gpujpeg_opengl_texture_register(int texture_id, enum gpujpeg_opengl_texture_type texture_type)
{
    struct gpujpeg_opengl_texture* texture = NULL;
    cudaMallocHost((void**)&texture, sizeof(struct gpujpeg_opengl_texture));
    assert(texture != NULL);

    texture->texture_id = texture_id;
    texture->texture_type = texture_type;
    texture->texture_width = 0;
    texture->texture_height = 0;
    texture->texture_pbo_id = 0;
    texture->texture_pbo_type = 0;
    texture->texture_pbo_resource = 0;
    texture->texture_callback_param = NULL;
    texture->texture_callback_attach_opengl = NULL;
    texture->texture_callback_detach_opengl = NULL;

#ifdef GPUJPEG_USE_OPENGL
    glBindTexture(GL_TEXTURE_2D, texture->texture_id);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &texture->texture_width);
    glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &texture->texture_height);
    glBindTexture(GL_TEXTURE_2D, 0);
    assert(texture->texture_width != 0 && texture->texture_height != 0);

    // Select PBO type
    if ( texture->texture_type == GPUJPEG_OPENGL_TEXTURE_READ ) {
        texture->texture_pbo_type = GL_PIXEL_PACK_BUFFER;
    } else if ( texture->texture_type == GPUJPEG_OPENGL_TEXTURE_WRITE ) {
        texture->texture_pbo_type = GL_PIXEL_UNPACK_BUFFER;
    } else {
        assert(0);
    }

    // Create PBO
    glGenBuffers(1, &texture->texture_pbo_id);
    glBindBuffer(texture->texture_pbo_type, texture->texture_pbo_id);
    glBufferData(texture->texture_pbo_type, texture->texture_width * texture->texture_height * 3 * sizeof(uint8_t), NULL, GL_DYNAMIC_DRAW);
    glBindBuffer(texture->texture_pbo_type, 0);

    // Create CUDA PBO Resource
    cudaGraphicsGLRegisterBuffer(&texture->texture_pbo_resource, texture->texture_pbo_id, cudaGraphicsMapFlagsNone);
    gpujpeg_cuda_check_error("Register OpenGL buffer");
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif

    return texture;
}

/** Documented at declaration */
void
gpujpeg_opengl_texture_unregister(struct gpujpeg_opengl_texture* texture)
{
#ifdef GPUJPEG_USE_OPENGL
    if ( texture->texture_pbo_id != 0 ) {
        glDeleteBuffers(1, &texture->texture_pbo_id);
    }
    if ( texture->texture_pbo_resource != NULL ) {
        cudaGraphicsUnregisterResource(texture->texture_pbo_resource);
    }
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif

    assert(texture != NULL);
    cudaFreeHost(texture);
}

/** Documented at declaration */
uint8_t*
gpujpeg_opengl_texture_map(struct gpujpeg_opengl_texture* texture, int* data_size)
{
    assert(texture->texture_pbo_resource != NULL);
    assert((texture->texture_callback_attach_opengl == NULL && texture->texture_callback_detach_opengl == NULL) ||
           (texture->texture_callback_attach_opengl != NULL && texture->texture_callback_detach_opengl != NULL));

    // Attach OpenGL context by callback
    if ( texture->texture_callback_attach_opengl != NULL )
        texture->texture_callback_attach_opengl(texture->texture_callback_param);

    uint8_t* d_data = NULL;

#ifdef GPUJPEG_USE_OPENGL
    if ( texture->texture_type == GPUJPEG_OPENGL_TEXTURE_READ ) {
        assert(texture->texture_pbo_type == GL_PIXEL_PACK_BUFFER);

        glBindTexture(GL_TEXTURE_2D, texture->texture_id);
        glBindBuffer(GL_PIXEL_PACK_BUFFER, texture->texture_pbo_id);

        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGB, GL_UNSIGNED_BYTE, 0);

        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif

    // Map pixel buffer object to cuda
    cudaGraphicsMapResources(1, &texture->texture_pbo_resource, 0);
    gpujpeg_cuda_check_error("Encoder map texture PBO resource");

    // Get device data pointer to pixel buffer object data
    size_t d_data_size;
    cudaGraphicsResourceGetMappedPointer((void **)&d_data, &d_data_size, texture->texture_pbo_resource);
    gpujpeg_cuda_check_error("Encoder get device pointer for texture PBO resource");
    if ( data_size != NULL )
        *data_size = d_data_size;

    return d_data;
}

/** Documented at declaration */
void
gpujpeg_opengl_texture_unmap(struct gpujpeg_opengl_texture* texture)
{
    // Unmap pbo
    cudaGraphicsUnmapResources(1, &texture->texture_pbo_resource, 0);
    gpujpeg_cuda_check_error("Encoder unmap texture PBO resource");

#ifdef GPUJPEG_USE_OPENGL
    if ( texture->texture_type == GPUJPEG_OPENGL_TEXTURE_WRITE ) {
        assert(texture->texture_pbo_type == GL_PIXEL_UNPACK_BUFFER);

        glBindTexture(GL_TEXTURE_2D, texture->texture_id);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, texture->texture_pbo_id);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, texture->texture_width, texture->texture_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        glBindTexture(GL_TEXTURE_2D, 0);
        glFinish();
    }
#else
    GPUJPEG_EXIT_MISSING_OPENGL();
#endif

    // Dettach OpenGL context by callback
    if ( texture->texture_callback_detach_opengl != NULL )
        texture->texture_callback_detach_opengl(texture->texture_callback_param);
}
