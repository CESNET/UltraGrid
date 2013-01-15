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
 
#ifndef GPUJPEG_COMMON_H
#define GPUJPEG_COMMON_H

#include <stdint.h>
#include <libgpujpeg/gpujpeg_type.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Marker used as segment info */
#define GPUJPEG_MARKER_SEGMENT_INFO GPUJPEG_MARKER_APP13

/** Maximum number of devices for get device info */
#define GPUJPEG_MAX_DEVICE_COUNT 10

/** Device info for one device */
struct gpujpeg_device_info
{
    // Device id
    int id;
    // Device name
    char name[255];
    // Compute capability major version
    int cc_major;
    // Compute capability minor version
    int cc_minor;
    // Amount of global memory
    long global_memory;
    // Amount of constant memory
    long constant_memory;
    // Amount of shared memory
    long shared_memory;
    // Number of registers per block
    int register_count;
    // Number of multiprocessors
    int multiprocessor_count;
};

/** Device info for all devices */
struct gpujpeg_devices_info
{
    // Number of devices
    int device_count;
    // Device info for each
    struct gpujpeg_device_info device[GPUJPEG_MAX_DEVICE_COUNT];
};

/**
 * Get information about available devices
 * 
 * @return devices info
 */
struct gpujpeg_devices_info
gpujpeg_get_devices_info();

/**
 * Print information about available devices
 * 
 * @return void
 */
void
gpujpeg_print_devices_info();

/**
 * Init CUDA device
 * 
 * @param device_id  CUDA device id (starting at 0)
 * @param flags  Flags, e.g. if device info should be printed out (GPUJPEG_VERBOSE) or 
 *               enable OpenGL interoperability (GPUJPEG_OPENGL_INTEROPERABILITY)
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_init_device(int device_id, int flags);

/**
 * JPEG parameters. This structure should not be initialized only be hand,
 * but at first gpujpeg_set_default_parameters should be call and then
 * some parameters could be changed.
 */
struct gpujpeg_parameters
{
    // Verbose output - show more information, collects duration of each phase, etc.
    int verbose;
    
    // Encoder quality level (0-100)
    int quality;
    
    // Restart interval (0 means that restart interval is disabled and CPU huffman coder is used)
    int restart_interval;
    
    // Flag which determines if interleaved format of JPEG stream should be used, "1" = only
    // one scan which includes all color components (e.g. Y Cb Cr Y Cb Cr ...),
    // or "0" = one scan for each color component (e.g. Y Y Y ..., Cb Cb Cb ..., Cr Cr Cr ...)
    int interleaved;
    
    // Use segment info in stream for fast decoding. The segment info is placed into special
    // application headers and contains indexes to beginnings of segments in the stream, so
    // the decoder don't have to parse the stream byte-by-byte but he can only read the
    // segment info and start decoding. The segment info is presented for each scan, and thus
    // the best result is achieved when it is used in combination with "interleaved = 1" settings.
    int segment_info;

    // Sampling factors for each color component
    struct gpujpeg_component_sampling_factor sampling_factor[GPUJPEG_MAX_COMPONENT_COUNT];

    // Color space that is used inside JPEG stream = that is carried in JPEG format = to
    // which are input data converted (default value is JPEG YCbCr)
    enum gpujpeg_color_space color_space_internal;
};

/**
 * Set default parameters for JPEG coder
 * 
 * @param param  Parameters for JPEG coder
 * @return void
 */
void
gpujpeg_set_default_parameters(struct gpujpeg_parameters* param);

/**
 * Set parameters for using chroma subsampling
 * 
 * @param param  Parameters for coder
 * @return void
 */
void
gpujpeg_parameters_chroma_subsampling(struct gpujpeg_parameters* param);

/**
 * Image parameters. This structure should not be initialized only be hand,
 * but at first gpujpeg_image_set_default_parameters should be call and then
 * some parameters could be changed.
 *
 */
struct gpujpeg_image_parameters {
    // Image data width
    int width;
    // Image data height
    int height;
    // Image data component count
    int comp_count;
    // Image data color space
    enum gpujpeg_color_space color_space;
    // Image data sampling factor
    enum gpujpeg_sampling_factor sampling_factor;
};

/**
 * Set default parameters for JPEG image
 * 
 * @param param  Parameters for image
 * @return void
 */
void
gpujpeg_image_set_default_parameters(struct gpujpeg_image_parameters* param);

/** Image file formats */
enum gpujpeg_image_file_format {
    // Unknown image file format
    GPUJPEG_IMAGE_FILE_UNKNOWN = 0,
    // Raw file format
    GPUJPEG_IMAGE_FILE_RAW = 1,
    // JPEG file format
    GPUJPEG_IMAGE_FILE_JPEG = 2,
    // RGB file format, simple data format without header [R G B] [R G B] ...
    GPUJPEG_IMAGE_FILE_RGB = 1 | 4,
    // YUV file format, simple data format without header [Y U V] [Y U V] ...
    GPUJPEG_IMAGE_FILE_YUV = 1 | 8,
    // Gray file format
    GPUJPEG_IMAGE_FILE_GRAY = 1 | 16
};

/**
 * Get image file format from filename
 *
 * @param filename  Filename of image file
 * @return image_file_format or GPUJPEG_IMAGE_FILE_UNKNOWN if type cannot be determined
 */
enum gpujpeg_image_file_format
gpujpeg_image_get_file_format(const char* filename);

/** 
 * JPEG segment structure. Segment is data in scan generated by huffman coder 
 * for N consecutive MCUs, where N is restart interval (e.g. data for MCUs between 
 * restart markers)
 */
struct gpujpeg_segment
{
    // Scan index (in which segment belongs)
    int scan_index;
    // Segment index in the scan (position of segment in scan starting at 0)
    int scan_segment_index;
    // MCU count in segment
    int mcu_count;
    
    // Data compressed index (output/input data from/to segment for encoder/decoder)
    int data_compressed_index;
    // Date temp index (temporary data of segment in CC 2.0 encoder)
    int data_temp_index;
    // Data compressed size (output/input data from/to segment for encoder/decoder)
    int data_compressed_size;
    
    // Offset of first block index
    int block_index_list_begin;
    // Number of blocks of the segment
    int block_count;
};

/**
 * JPEG color component structure
 */
struct gpujpeg_component
{
    // Component type (luminance or chrominance)
    enum gpujpeg_component_type type;
    
    // Component sampling factor (horizontal and vertical)
    struct gpujpeg_component_sampling_factor sampling_factor;
    
    // Real component width
    int width;
    // Real component height
    int height;
    
    // Allocated data width for component (rounded to 8 for 8x8 blocks)
    int data_width;
    // Allocated data height for component (rounded to 8 for 8x8 blocks)
    int data_height;
    // Allocated data size for component
    int data_size;
    
    // MCU size for component (minimun coded unit size)
    int mcu_size;
    // MCU size in component x-axis
    int mcu_size_x;
    // MCU size in component y-axis
    int mcu_size_y;
    
    // MCU maximum compressed size for component
    int mcu_compressed_size;
    
    // MCU count for component (for interleaved mode the same value as [en|de]coder->mcu_count)
    int mcu_count;
    // MCU count in component x-axis
    int mcu_count_x;
    // MCU count in component y-axis
    int mcu_count_y;
    // Segment count in component
    int segment_count;
    // MCU count per segment in component (the last segment can contain less MCUs, but all other must contain this count)
    int segment_mcu_count;
    
    // Preprocessor data in device memory (output/input for encoder/decoder)
    uint8_t* d_data;
    
    // DCT and quantizer data in host memory (output/input for encoder/decoder)
    int16_t* data_quantized;
    // DCT and quantizer data in device memory (output/input for encoder/decoder)
    int16_t* d_data_quantized;
    // Index of DCT and quantizer data in device and host buffers
    unsigned int data_quantized_index;
};

/**
 * Print component data
 * 
 * @param component
 * @param d_data
 */
void
gpujpeg_component_print8(struct gpujpeg_component* component, uint8_t* d_data);

/**
 * Print component data
 * 
 * @param component
 * @param d_data
 */
void
gpujpeg_component_print16(struct gpujpeg_component* component, int16_t* d_data);

/**
 * JPEG coder structure
 */
struct gpujpeg_coder
{  
    // Parameters (quality, restart_interval, etc.)
    struct gpujpeg_parameters param;
    
    // Parameters for image data (width, height, comp_count, etc.)
    struct gpujpeg_image_parameters param_image;
    
    // Color components
    struct gpujpeg_component* component;
    // Color components in device memory
    struct gpujpeg_component* d_component;
    
    // Segments for all components
    struct gpujpeg_segment* segment;
    // Segments in device memory for all components
    struct gpujpeg_segment* d_segment;
    
    // Preprocessor data (kernel function pointer)
    void* preprocessor;

    // Maximum sampling factor from components
    struct gpujpeg_component_sampling_factor sampling_factor;
    // MCU size (for all components)
    int mcu_size;
    // MCU compressed size (for all components)
    int mcu_compressed_size;
    // MCU count (for all components)
    int mcu_count;  
    // Segment total count for all components
    int segment_count;
    // MCU count per segment (the last segment can contain less MCUs, but all other must contain this count)
    int segment_mcu_count;
    
    // Allocated data width
    int data_width;
    // Allocated data height
    int data_height;
    // Raw image data coefficient count
    int data_raw_size;
    // Allocated data coefficient count for all components
    int data_size;
    // Compressed allocated data size
    int data_compressed_size;
    
    // Number od 8x8 blocks in all components
    int block_count;
    
    // List of block indices in host memory (lower 7 bits are index of component, 
    // 8th bit is 0 for luminance block or 1 for chroma block and bits from 9 
    // above are base index of the block in quantized buffer data)
    uint64_t* block_list;
    // List of block indices in device memory (same format as host-memory block list)
    uint64_t* d_block_list;
    
    
    // Raw image data in host memory (loaded from file for encoder, saved to file for decoder)
    uint8_t* data_raw;
    // Raw image data in device memory (loaded from file for encoder, saved to file for decoder)
    uint8_t* d_data_raw;
    
    // Preprocessor data in device memory (output/input for encoder/decoder)
    uint8_t* d_data;
    
    // DCT and quantizer data in host memory (output/input for encoder/decoder)
    int16_t* data_quantized;
    // DCT and quantizer data in device memory (output/input for encoder/decoder)
    int16_t* d_data_quantized;
    
    // Huffman coder data in host memory (output/input for encoder/decoder)
    uint8_t* data_compressed;
    // Huffman coder data in device memory (output/input for encoder/decoder)
    uint8_t* d_data_compressed;
    // Huffman coder temporary data (in device memory only)
    uint8_t* d_temp_huffman;
    
    // CUDA Compute capability (major and minor version)
    int cuda_cc_major;
    int cuda_cc_minor;

    // Operation durations
    float duration_memory_to;
    float duration_memory_from;
    float duration_memory_map;
    float duration_memory_unmap;
    float duration_preprocessor;
    float duration_dct_quantization;
    float duration_huffman_coder;
    float duration_stream;
    float duration_in_gpu;
};

/**
 * Initialize JPEG coder (allocate buffers and initialize structures)
 * 
 * @param codec  Codec structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_coder_init(struct gpujpeg_coder* coder);

/**
 * Deinitialize JPEG coder (free buffers)
 * 
 * @param codec  Codec structure
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_coder_deinit(struct gpujpeg_coder* coder);

/**
 * Calculate size for image by parameters
 * 
 * @param param  Image parameters
 * @return calculate size
 */
int
gpujpeg_image_calculate_size(struct gpujpeg_image_parameters* param);

/**
 * Load RGB image from file
 * 
 * @param filaname  Image filename
 * @param image  Image data buffer
 * @param image_size  Image data buffer size (can be specified for verification or 0 for retrieval)
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_image_load_from_file(const char* filename, uint8_t** image, int* image_size);

/**
 * Save RGB image to file
 * 
 * @param filaname  Image filename
 * @param image  Image data buffer
 * @param image_size  Image data buffer size
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_image_save_to_file(const char* filename, uint8_t* image, int image_size);

/**
 * Destroy DXT image
 * 
 * @param image  Image data buffer
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_image_destroy(uint8_t* image);

/**
 * Print range info for image samples
 *
 * @param filename
 * @param width
 * @param height
 * @param sampling_factor
 */
void
gpujpeg_image_range_info(const char* filename, int width, int height, enum gpujpeg_sampling_factor sampling_factor);

/**
 * Convert image
 *
 * @param input
 * @param filename
 * @param param_image_from
 * @param param_image_to
 */
void
gpujpeg_image_convert(const char* input, const char* output, struct gpujpeg_image_parameters param_image_from,
        struct gpujpeg_image_parameters param_image_to);

/**
 * Init OpenGL context
 *
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_opengl_init();

/**
 * Create OpenGL texture
 *
 * @param width
 * @param height
 * @param data
 * @return nonzero texture id if succeeds, otherwise 0
 */
int
gpujpeg_opengl_texture_create(int width, int height, uint8_t* data);

/**
 * Set data to OpenGL texture
 *
 * @param texture_id
 * @param data
 * @return 0 if succeeds, otherwise nonzero
 */
int
gpujpeg_opengl_texture_set_data(int texture_id, uint8_t* data);

/**
 * Get data from OpenGL texture
 *
 * @param texture_id
 * @param data
 * @param data_size
 * @return 0 data if succeeds, otherwise nonzero
 */
int
gpujpeg_opengl_texture_get_data(int texture_id, uint8_t* data, int* data_size);

/**
 * Destroy OpenGL texture
 *
 * @param texture_id
 */
void
gpujpeg_opengl_texture_destroy(int texture_id);

/**
 * Registered OpenGL texture type
 */
enum gpujpeg_opengl_texture_type
{
    GPUJPEG_OPENGL_TEXTURE_READ = 1,
    GPUJPEG_OPENGL_TEXTURE_WRITE = 2
};

/**
 * Represents OpenGL texture that is registered to CUDA,
 * thus the device pointer can be acquired.
 */
struct gpujpeg_opengl_texture
{
    // Texture id
    int texture_id;
    // Texture type
    enum gpujpeg_opengl_texture_type texture_type;
    // Texture width
    int texture_width;
    // Texture height
    int texture_height;
    // Texture pixel buffer object type
    int texture_pbo_type;
    // Texture pixel buffer object id
    int texture_pbo_id;
    // Texture PBO resource for CUDA
    struct cudaGraphicsResource* texture_pbo_resource;

    // Texture callbacks parameter
    void * texture_callback_param;
    // Texture callback for attaching OpenGL context (by default not used)
    void (*texture_callback_attach_opengl)(void* param);
    // Texture callback for detaching OpenGL context (by default not used)
    void (*texture_callback_detach_opengl)(void* param);
    // If you develop multi-threaded application where one thread use CUDA
    // for JPEG decoding and other thread use OpenGL for displaying results
    // from JPEG decoder, when an image is decoded you must detach OpenGL context
    // from displaying thread and attach it to compressing thread (inside
    // code of texture_callback_attach_opengl which is automatically invoked
    // by decoder), decoder then is able to copy data from GPU memory used
    // for compressing to GPU memory used by OpenGL texture for displaying,
    // then decoder call the second callback and you have to detach OpenGL context
    // from compressing thread and attach it to displaying thread (inside code of
    // texture_callback_detach_opengl).
    //
    // If you develop single-thread application where the only thread use CUDA
    // for compressing and OpenGL for displaying you don't have to implement
    // these callbacks because OpenGL context is already attached to thread
    // that use CUDA for JPEG decoding.
};

/**
 * Register OpenGL texture to CUDA
 *
 * @param texture_id
 * @return allocated registred texture structure
 */
struct gpujpeg_opengl_texture*
gpujpeg_opengl_texture_register(int texture_id, enum gpujpeg_opengl_texture_type texture_type);

/**
 * Unregister OpenGL texture from CUDA. Deallocated given
 * structure.
 *
 * @param texture
 */
void
gpujpeg_opengl_texture_unregister(struct gpujpeg_opengl_texture* texture);

/**
 * Map registered OpenGL texture to CUDA and return
 * device pointer to the texture data
 *
 * @param texture
 * @param data_size  Data size in returned buffer
 * @param copy_from_texture  Specifies whether memory copy from texture
 *                           should be performed
 */
uint8_t*
gpujpeg_opengl_texture_map(struct gpujpeg_opengl_texture* texture, int* data_size);

/**
 * Unmap registered OpenGL texture from CUDA and the device
 * pointer is no longer useable.
 *
 * @param texture
 * @param copy_to_texture  Specifies whether memoryc copy to texture
 *                         should be performed
 */
void
gpujpeg_opengl_texture_unmap(struct gpujpeg_opengl_texture* texture);

/**
 * Declare timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DECLARE(name) \
    cudaEvent_t name ## _start__; \
    cudaEvent_t name ## _stop__; \
    float name ## _elapsedTime__; \

/**
 * Create timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_CREATE(name) \
    cudaEventCreate(&name ## _start__); \
    cudaEventCreate(&name ## _stop__); \

/**
 * Start timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_START(name) \
    cudaEventRecord(name ## _start__, 0) \

/**
 * Stop timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_STOP(name) \
    cudaEventRecord(name ## _stop__, 0); \
    cudaEventSynchronize(name ## _stop__); \
    cudaEventElapsedTime(&name ## _elapsedTime__, name ## _start__, name ## _stop__) \

/**
 * Get duration for timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DURATION(name) name ## _elapsedTime__

/**
 * Stop timer and print result
 *
 * @param name
 * @param text
 */
#define GPUJPEG_CUSTOM_TIMER_STOP_PRINT(name, text) \
    GPUJPEG_CUSTOM_TIMER_STOP(name); \
    printf("%s %f ms\n", text, name ## _elapsedTime__) \

/**
 * Destroy timer
 *
 * @param name
 */
#define GPUJPEG_CUSTOM_TIMER_DESTROY(name) \
    cudaEventDestroy(name ## _start__); \
    cudaEventDestroy(name ## _stop__); \

/**
 * Default timer implementation
 */
#define GPUJPEG_TIMER_INIT() \
    GPUJPEG_CUSTOM_TIMER_DECLARE(def) \
    GPUJPEG_CUSTOM_TIMER_CREATE(def)
#define GPUJPEG_TIMER_START() GPUJPEG_CUSTOM_TIMER_START(def)
#define GPUJPEG_TIMER_STOP() GPUJPEG_CUSTOM_TIMER_STOP(def)
#define GPUJPEG_TIMER_DURATION() GPUJPEG_CUSTOM_TIMER_DURATION(def)
#define GPUJPEG_TIMER_STOP_PRINT(text) GPUJPEG_CUSTOM_TIMER_STOP_PRINT(def, text)
#define GPUJPEG_TIMER_DEINIT() GPUJPEG_CUSTOM_TIMER_DESTROY(def)

#ifdef __cplusplus
}
#endif

#endif // GPUJPEG_COMMON_H
