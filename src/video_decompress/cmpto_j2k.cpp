/**
 * @file   video_decompress/cmpto_j2k.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2023 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
/**
 * @file
 * Some of the concepts are similar to encoder (eg. keeping limited number of
 * frames in decoder) so please refer to that file.
 *
 * Problematic part of following code is that UltraGrid decompress API is
 * synchronous only while the CMPTO J2K decoder is inherently asynchronous.
 * Threrefore the integration works in following fashion:
 * - there is a thread that waits for completed (decompressed) frames,
 *   if there is any, it put it in queue (or drop if full)
 * - when a new frame arives, j2k_decompress() passes it to decoder
 *   (which is asynchronous, thus non-blocking)
 * - then queue (filled by thread in first point) is checked - if it is
 *   non-empty, frame is copied to framebufffer. If not false is returned.
 *
 * @todo
 * Reconfiguration isn't entirely correct - on reconfigure, all frames
 * should be dropped and not copied to framebuffer. However this is usually
 * not an issue because dynamic video change is rare (except switching to
 * another stream, which, however, creates a new decoder).
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include <cmpto_j2k_dec.h>

#include <algorithm>
#include <mutex>
#include <queue>
#include <utility>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/macros.h"
#include "utils/misc.h"
#include "video.h"
#include "video_decompress.h"

constexpr const char *MOD_NAME = "[Cmpto J2K dec.]";

#define CHECK_OK(cmd, err_msg, action_fail) do { \
        int j2k_error = cmd; \
        if (j2k_error != CMPTO_OK) {\
                log_msg(LOG_LEVEL_ERROR, "%s %s: %s\n", \
                        MOD_NAME, err_msg, cmpto_j2k_dec_get_last_error()); \
                action_fail;\
        } \
} while (0)

#define NOOP ((void) 0)

// General Parameter Defaults
constexpr int DEFAULT_MAX_QUEUE_SIZE         = 2;                         // maximal size of queue for decompressed frames
constexpr int DEFAULT_MAX_IN_FRAMES          = 4;                         // maximal number of concurrently decompressed frames

// CPU-specific Defaults
constexpr int DEFAULT_THREAD_COUNT           = CMPTO_J2K_DEC_CPU_DEFAULT; // Number of threads equal to all cores
constexpr int MIN_CPU_THREAD_COUNT           = CMPTO_J2K_DEC_CPU_NONE;    // No threads will be created
constexpr size_t DEFAULT_CPU_MEM_LIMIT       = 0;                         // Should always be 0. Not implemented as of v2.8.1
constexpr unsigned int DEFAULT_CPU_IMG_LIMIT = 0;                         // 0 for default, thread_count for max
constexpr unsigned int MIN_CPU_IMG_LIMIT     = 0;                         // Min number of images encoded by the CPU at once

#ifdef HAVE_CUDA
// CUDA-specific Defaults
constexpr int64_t DEFAULT_CUDA_MEM_LIMIT     = 1000000000;
constexpr int     DEFAULT_CUDA_TILE_LIMIT    = 2;
#endif // HAVE_CUDA

using std::lock_guard;
using std::mutex;

/*
 * Function Predeclarations
 */ 
static void *decompress_j2k_worker(void *args);
static void rg48_to_r12l(unsigned char *dst_buffer, unsigned char *src_buffer, unsigned int width, unsigned int height);

/*
 * Platform to use for J2K Decompression
 */ 
enum j2k_decompress_platform {
        NONE = 0,
        CPU = 1,
#ifdef HAVE_CUDA
        CUDA = 2,
#endif // HAVE_CUDA
};

/**
 * @brief Struct to hold UG and CMPTO Codec information
 */
struct Codec {
        codec_t ug_codec;
        enum cmpto_sample_format_type cmpto_sf;
        void (*convert)(unsigned char *dst_buffer, unsigned char *src_buffer, unsigned int width, unsigned int height);
};

// Supported UG/CMPTO Decompress Codecs
constexpr auto codecs = std::array{
        Codec{UYVY, CMPTO_422_U8_P1020, nullptr},
        Codec{v210, CMPTO_422_U10_V210, nullptr},
        Codec{RGB, CMPTO_444_U8_P012, nullptr},
        Codec{BGR, CMPTO_444_U8_P210, nullptr},
        Codec{RGBA, CMPTO_444_U8_P012Z, nullptr},
        Codec{R10k, CMPTO_444_U10U10U10_MSB32BE_P210, nullptr},
        Codec{R12L, CMPTO_444_U12_MSB16LE_P012, rg48_to_r12l},
};

/*
 * Exceptions for state_video_decompress_j2k construction
 */
/// @brief UnableToCreateJ2KDecoderCTX Exception
struct UnableToCreateJ2KDecoderCTX : public std::exception {
        UnableToCreateJ2KDecoderCTX() = default;
};

/*
 * Command Line Parameters for state_video_decompress_j2k
 */
#ifdef HAVE_CUDA
// CUDA-specific Command Line Parameters
ADD_TO_PARAM("j2k-dec-use-cuda", "* j2k-dec-use-cuda\n"
                                "  use CUDA to decode images\n");
ADD_TO_PARAM("j2k-dec-mem-limit", "* j2k-dec-mem-limit=<limit>\n"
                                "  J2K max memory usage in bytes.\n");
ADD_TO_PARAM("j2k-dec-tile-limit", "* j2k-dec-tile-limit=<limit>\n"
                                "  number of tiles decoded at moment (less to reduce latency, more to increase performance, 0 unlimited)\n");
// Option to use CPU for image decompression only required if CUDA is also compiled.
// Otherwise, CPU will be the default, with no need to explicity specify.
ADD_TO_PARAM("j2k-dec-use-cpu", "* j2k-dec-use-cpu\n"
                                "  use the CPU to decode images\n");
#endif // HAVE_CUDA
// CPU-specific Command Line Parameters
ADD_TO_PARAM("j2k-dec-cpu-thread-count", "* j2k-dec-cpu-thread-count=<threads>\n"
                                "  number of threads to use on the CPU (0 means number of threads equal to all cores)\n");
ADD_TO_PARAM("j2k-dec-img-limit", "* j2k-dec-img-limit=<limit>\n"
                                "  number of images which can be decoded at one moment (0 means default, thread-count is maximum limit)\n");
// General Command Line Parameters
ADD_TO_PARAM("j2k-dec-queue-len", "* j2k-dec-queue-len=<len>\n"
                                "  max queue len\n");
ADD_TO_PARAM("j2k-dec-encoder-queue", "* j2k-dec-encoder-queue=<len>\n"
                                "  max number of frames held by encoder\n");


/**
 * @brief  state_video_decompress_j2k Class
 */
struct state_video_decompress_j2k {
        state_video_decompress_j2k();

        cmpto_j2k_dec_ctx *decoder{};
        cmpto_j2k_dec_cfg *settings{};

        struct video_desc desc{};
        codec_t out_codec{};

        mutex lock;
        std::queue<std::pair<char *, size_t>> decompressed_frames; ///< buffer, length
        int pitch;
        pthread_t thread_id{};
        unsigned int in_frames{}; ///< actual number of decompressed frames
        unsigned long long int dropped{}; ///< number of dropped frames because queue was full

#ifdef HAVE_CUDA
        // CUDA Defaults
        unsigned int    cuda_mem_limit      = DEFAULT_CUDA_MEM_LIMIT;
        unsigned int    cuda_tile_limit     = DEFAULT_CUDA_TILE_LIMIT;

        // Default Decompression Platform to Use
        j2k_decompress_platform platform       = j2k_decompress_platform::CUDA;
#else
        // Default Decompression Platform to Use
        j2k_decompress_platform platform       = j2k_decompress_platform::CPU;
#endif

        // CPU Defaults
        unsigned int cpu_img_limit          = DEFAULT_CPU_IMG_LIMIT;
        const size_t cpu_mem_limit          = DEFAULT_CPU_MEM_LIMIT;    // Should always be 0. Not yet implemented as of Cmpto v2.8.1
        signed int   cpu_thread_count       = DEFAULT_THREAD_COUNT;

        // General Defaults
        unsigned int max_in_frames          = DEFAULT_MAX_IN_FRAMES;    // maximal frames that can be "in progress"
        unsigned int max_queue_size         = DEFAULT_MAX_QUEUE_SIZE;   // maximal length of @ref decompressed_frames

        void (*convert)(unsigned char *dst_buffer,
                unsigned char *src_buffer,
                unsigned int width, unsigned int height) { nullptr };

 private:
        void parse_params();
        bool initialize_j2k_dec_ctx();
};

/**
 * @brief Default state_video_decompress_j2k Constructor
 * @throw UnableToCreateJ2KDecoderCTX if unable to create J2K CTX
 */
state_video_decompress_j2k::state_video_decompress_j2k() {
        parse_params();

        if (!initialize_j2k_dec_ctx()) {
                throw UnableToCreateJ2KDecoderCTX();
        }
}

/**
 * @fn parse_params
 * @brief Parse Command Line Parameters and Initialize Struct Members
 */
void state_video_decompress_j2k::parse_params() {
#ifdef HAVE_CUDA
        if (get_commandline_param("j2k-dec-use-cuda")) {
                platform = j2k_decompress_platform::CUDA;
        }

        if (get_commandline_param("j2k-dec-mem-limit")) {
                cuda_mem_limit = unit_evaluate(get_commandline_param("j2k-dec-mem-limit"), nullptr);
        }

        if (get_commandline_param("j2k-dec-tile-limit")) {
                cuda_tile_limit = atoi(get_commandline_param("j2k-dec-tile-limit"));
        }

        if (get_commandline_param("j2k-dec-use-cpu")) {
                platform = j2k_decompress_platform::CPU;
        }
#endif // HAVE_CUDA

        if (get_commandline_param("j2k-dec-cpu-thread-count")) {
                cpu_thread_count = atoi(get_commandline_param("j2k-dec-cpu-thread-count"));

                // Confirm cpu_thread_count between MIN_CPU_THREAD_COUNT + 1 (0)
                if (cpu_thread_count <= MIN_CPU_THREAD_COUNT) {
                        // Implementing this requires the creation of executor threads.
                        log_msg(LOG_LEVEL_INFO, "%s j2k-dec-cpu-thread-count must be 0 or higher. Setting to min allowed 0\n",
                                MOD_NAME);
                        cpu_thread_count = 0;
                }
        }

        if (get_commandline_param("j2k-dec-queue-len")) {
                max_queue_size = atoi(get_commandline_param("j2k-dec-queue-len"));
        }

        if (get_commandline_param("j2k-dec-img-limit")) {
                cpu_img_limit = atoi(get_commandline_param("j2k-dec-img-limit"));

                // Confirm cpu_img_limit between MIN_CPU_IMG_LIMIT
                if (cpu_img_limit < MIN_CPU_IMG_LIMIT) {
                        log_msg(LOG_LEVEL_INFO, "%s j2k-dec-img-limit below min allowed of %i. Setting to min allowed %i\n",
                                MOD_NAME,
                                MIN_CPU_IMG_LIMIT,
                                MIN_CPU_IMG_LIMIT);
                        cpu_img_limit = MIN_CPU_IMG_LIMIT;
                }
        }

        if (get_commandline_param("j2k-dec-encoder-queue")) {
                max_in_frames = atoi(get_commandline_param("j2k-dec-encoder-queue"));
        }

        const auto *version = cmpto_j2k_dec_get_version();
        log_msg(LOG_LEVEL_INFO, "%s Using codec version: %s\n",
                MOD_NAME,
                (version == nullptr ? "(unknown)" : version->name));
}


/**
 * @fn initialize_j2k_dec_ctx
 * @brief Create cmpto_j2k_dec_ctx_cfg based on requested platform and command line arguments
 * @return true if cmpto_j2k_dec_ctx_cfg successfully created
 * @return false if unable to create cmpto_j2k_dec_ctx_cfg
 */
[[nodiscard]]
bool state_video_decompress_j2k::initialize_j2k_dec_ctx() {
        struct cmpto_j2k_dec_ctx_cfg *dec_ctx_cfg;
        CHECK_OK(cmpto_j2k_dec_ctx_cfg_create(&dec_ctx_cfg), "Error creating dec cfg", return false);

#ifdef HAVE_CUDA
        if (j2k_decompress_platform::CUDA == platform) {
                log_msg(LOG_LEVEL_INFO, "%s Configuring for CUDA Decoding\n", MOD_NAME);
                for (unsigned int i = 0; i < cuda_devices_count; ++i) {
                        CHECK_OK(cmpto_j2k_dec_ctx_cfg_add_cuda_device(dec_ctx_cfg, cuda_devices[i], cuda_mem_limit, cuda_tile_limit),
                                        "Error setting CUDA device", return false);
                        log_msg(LOG_LEVEL_INFO, "%s Using CUDA Device %s\n", MOD_NAME, std::to_string(cuda_devices[i]).c_str());
                }
        }
#endif // HAVE_CUDA
        if (j2k_decompress_platform::CPU == platform) {
                log_msg(LOG_LEVEL_INFO, "%s Configuring for CPU Decoding\n", MOD_NAME);

                // Confirm that cpu_thread_count != 0 (unlimited). If it does, cpu_img_limit can exceed thread_count
                if (cpu_thread_count != DEFAULT_THREAD_COUNT && cpu_img_limit > static_cast<unsigned>(cpu_thread_count)) {
                        log_msg(LOG_LEVEL_INFO, "%s j2k-dec-img-limit set to %i. Lowering to match to match j2k-dec-cpu-thread-count (%i)\n",
                                MOD_NAME,
                                cpu_img_limit,
                                cpu_thread_count);
                        cpu_img_limit = cpu_thread_count;
                }

                CHECK_OK(cmpto_j2k_dec_ctx_cfg_add_cpu(
                        dec_ctx_cfg,
                        cpu_thread_count,
                        cpu_mem_limit,
                        cpu_img_limit),
                        "Error configuring the CPU",
                        return false);

                log_msg(LOG_LEVEL_INFO, "%s Using %s threads on the CPU. Image Limit set to %i.\n",
                        MOD_NAME,
                        (cpu_thread_count == 0 ? "all available" : std::to_string(cpu_thread_count).c_str()),
                        cpu_img_limit);
        }

        CHECK_OK(cmpto_j2k_dec_ctx_create(dec_ctx_cfg, &this->decoder),
                "Error initializing context",
                return false);

        CHECK_OK(cmpto_j2k_dec_ctx_cfg_destroy(dec_ctx_cfg),
                "Destroy cfg",
                NOOP);

        CHECK_OK(cmpto_j2k_dec_cfg_create(this->decoder, &this->settings), 
                "Error creating configuration", {
                        cmpto_j2k_dec_cfg_destroy(this->settings);
                        cmpto_j2k_dec_ctx_destroy(this->decoder);
                        return false;
                });

        auto ret = pthread_create(&this->thread_id, NULL, decompress_j2k_worker, static_cast<void *>(this));
        assert(ret == 0 && "Unable to create thread");

        return true;
}


static void rg48_to_r12l(unsigned char *dst_buffer,
                unsigned char *src_buffer,
                unsigned int width, unsigned int height) {
        int src_pitch = vc_get_linesize(width, RG48);
        int dst_len   = vc_get_linesize(width, R12L);
        decoder_t vc_copylineRG48toR12L = get_decoder_from_to(RG48, R12L);

        for (unsigned i = 0; i < height; i++) {
                vc_copylineRG48toR12L(dst_buffer, src_buffer, dst_len, 0, 0, 0);
                src_buffer += src_pitch;
                dst_buffer += dst_len;
        }
}

static void print_dropped(unsigned long long int dropped, const j2k_decompress_platform& platform) {
        if (dropped % 10 == 1) {
                log_msg(LOG_LEVEL_WARNING, "%s Some frames (%llu) dropped.\n", MOD_NAME, dropped);

                if (j2k_decompress_platform::CPU == platform) {
                        log_msg_once(LOG_LEVEL_INFO, to_fourcc('J', '2', 'D', 'W'), "%s You may try to increase "
                                "image limit to increase the number of images decoded at one moment by adding parameter: --param j2k-dec-img-limit=#\n",
                                MOD_NAME);
                }
#ifdef HAVE_CUDA
                if (j2k_decompress_platform::CUDA == platform) {
                        log_msg_once(LOG_LEVEL_INFO, to_fourcc('J', '2', 'D', 'W'), "%s You may try to increase "
                                "tile limit to increase the throughput by adding parameter: --param j2k-dec-tile-limit=4\n",
                                MOD_NAME);
                }
#endif // HAVE_CUDA
        }
}

/**
 * This function just runs in thread and gets decompressed images from decoder
 * putting them to queue (or dropping if full).
 */
static void *decompress_j2k_worker(void *args) {
        auto *s = static_cast<state_video_decompress_j2k*>(args);

        while (true) {
next_image:
                struct cmpto_j2k_dec_img *img;
                int decoded_img_status;
                CHECK_OK(cmpto_j2k_dec_ctx_get_decoded_img(s->decoder, 1, &img, &decoded_img_status),
                        "Decode image", goto next_image);

                {
                        lock_guard<mutex> lk(s->lock);
                        if (s->in_frames) s->in_frames--;
                }

                if (img == NULL) { // decoder stopped (poison pill)
                        break;
                }

                if (decoded_img_status != CMPTO_J2K_DEC_IMG_OK) {
                        const char * decoding_error = "";
                        CHECK_OK(cmpto_j2k_dec_img_get_error(img, &decoding_error), "get error status",
                                decoding_error = "(failed)");
                        log_msg(LOG_LEVEL_ERROR, "Image decoding failed: %s\n", decoding_error);
                        continue;
                }

                void *dec_data;
                size_t len;
                CHECK_OK(cmpto_j2k_dec_img_get_samples(img, &dec_data, &len),
                                "Error getting samples", cmpto_j2k_dec_img_destroy(img); goto next_image);

                char *buffer = (char *) malloc(len);
                if (s->convert) {
                        s->convert((unsigned char*) buffer, (unsigned char*) dec_data, s->desc.width, s->desc.height);
                        len = vc_get_linesize(s->desc.width, s->out_codec) * s->desc.height;
                } else {
                        memcpy(buffer, dec_data, len);
                }

                CHECK_OK(cmpto_j2k_dec_img_destroy(img),
                                "Unable to to return processed image", NOOP);
                lock_guard<mutex> lk(s->lock);
                while (s->decompressed_frames.size() >= s->max_queue_size) {
                        print_dropped(s->dropped++, s->platform);
                        auto decoded = s->decompressed_frames.front();
                        s->decompressed_frames.pop();
                        free(decoded.first);
                }
                s->decompressed_frames.push({buffer,len});
        }

        return NULL;
}


/**
 * @brief Initialize a new instance of state_video_decompress_j2k
 * @return Null or Pointer to state_video_decompress_j2k
 */
static void * j2k_decompress_init(void) {
        try {
                auto *s = new state_video_decompress_j2k();
                return s;
        } catch (...) {
                return NULL;
        }
}

static int j2k_decompress_reconfigure(void *state, struct video_desc desc,
                int rshift, int gshift, int bshift, int pitch, codec_t out_codec) {
        auto *s = static_cast<state_video_decompress_j2k*>(state);

        if (out_codec == VIDEO_CODEC_NONE) { // probe format
                s->out_codec = VIDEO_CODEC_NONE;
                s->desc = desc;
                return true;
        }

        if (out_codec == R12L) {
                log_msg(LOG_LEVEL_NOTICE, "%s Decoding to 12-bit RGB.\n", MOD_NAME);
        }

        enum cmpto_sample_format_type cmpto_sf = (cmpto_sample_format_type) 0;

        auto matches = [&](const Codec& codec) { return codec.ug_codec == out_codec; };

        if (const auto& codec = std::find_if(codecs.begin(), codecs.end(), matches) ; codec != codecs.end()) {
                cmpto_sf = codec->cmpto_sf;
                s->convert = codec->convert;
        }

        if (!cmpto_sf) {
                log_msg(LOG_LEVEL_ERROR, "%s Unsupported output codec: %s", 
                        MOD_NAME,
                        get_codec_name(out_codec));
                abort();
        }

        if (out_codec != RGBA || (rshift == 0 && gshift == 8 && bshift == 16)) {
                CHECK_OK(cmpto_j2k_dec_cfg_set_samples_format_type(s->settings, cmpto_sf),
                                "Error setting sample format type", return false);
        } else { // RGBA with non-standard shift
                if (rshift % 8 != 0 || gshift % 8 != 0 || bshift % 8 != 0) {
                        log_msg(LOG_LEVEL_ERROR, "%s Component shifts not aligned to a byte boundary is not supported.\n",
                                MOD_NAME);
                        return false;
                }
                cmpto_j2k_dec_comp_format fmt[3] = {};
                const int shifts[3] = { rshift, gshift, bshift };
                for (int i = 0; i < 3; ++i) {
                        fmt[i].comp_index = i;
                        fmt[i].data_type = CMPTO_INT8;
                        fmt[i].offset = shifts[i] / 8;
                        fmt[i].stride_x = get_bpp(out_codec);
                        fmt[i].stride_y = vc_get_linesize(desc.width, out_codec);
                        fmt[i].bit_depth = get_bits_per_component(out_codec);
                        fmt[i].bit_shift = 0;
                        fmt[i].is_or_combined = 0;
                        fmt[i].is_signed = 0;
                        fmt[i].sampling_factor_x = 1;
                        fmt[i].sampling_factor_y = 1;
                }

                CHECK_OK(cmpto_j2k_dec_cfg_set_samples_format(s->settings, fmt, 3),
                                "Error setting sample format", return false);
        }

        s->desc = desc;
        s->out_codec = out_codec;
        s->pitch = pitch;

        return true;
}

/**
 * Callback called by the codec when codestream is no longer required.
 */
static void release_cstream(void * custom_data, size_t custom_data_size, const void * codestream, size_t codestream_size) {
        (void) custom_data; (void) custom_data_size; (void) codestream_size;
        free(const_cast<void *>(codestream));
}

static decompress_status j2k_probe_internal_codec(codec_t in_codec, unsigned char *buffer, size_t len, struct pixfmt_desc *internal_prop) {
        struct cmpto_j2k_dec_img_info info;
        struct cmpto_j2k_dec_comp_info comp_info[3];
        if (cmpto_j2k_dec_cstream_get_img_info(buffer, len, &info) != CMPTO_OK ||
                        cmpto_j2k_dec_cstream_get_comp_info(buffer, len, 0, &comp_info[0]) != CMPTO_OK) {
                log_msg(LOG_LEVEL_ERROR, "%s Failed to get image or first component info.\n", MOD_NAME);
                return DECODER_NO_FRAME;
        }

        internal_prop->depth = comp_info[0].bit_depth;
        internal_prop->rgb = in_codec == J2KR;
        if (info.comp_count == 3) {
                if (cmpto_j2k_dec_cstream_get_comp_info(buffer, len, 1, &comp_info[1]) != CMPTO_OK ||
                                cmpto_j2k_dec_cstream_get_comp_info(buffer, len, 2, &comp_info[2]) != CMPTO_OK) {
                        log_msg(LOG_LEVEL_ERROR, "%s Failed to get component 1 or 2 info.\n", MOD_NAME);
                        return DECODER_NO_FRAME;
                }
                if (comp_info[0].sampling_factor_x == 1 && comp_info[0].sampling_factor_y == 1 &&
                                comp_info[1].sampling_factor_x == comp_info[2].sampling_factor_x &&
                                comp_info[1].sampling_factor_y == comp_info[2].sampling_factor_y) {
                        int a = 4 / comp_info[1].sampling_factor_x;
                        internal_prop->subsampling = 4000 + a * 100;
                        if (comp_info[1].sampling_factor_y == 1) {
                                internal_prop->subsampling += a * 10;
                        }
                }
        }

        int msg_level = internal_prop->subsampling == 0 ? LOG_LEVEL_WARNING /* bogus? */ : LOG_LEVEL_VERBOSE;
        log_msg(msg_level, "%s stream properties: %s\n", MOD_NAME, get_pixdesc_desc(*internal_prop));

        return DECODER_GOT_CODEC;
}

/**
 * Main decompress function - passes frame to the codec and checks if there are
 * some decoded frames. If so, copies that to framebuffer. In the opposite case
 * it just returns false.
 */
static decompress_status j2k_decompress(void *state, unsigned char *dst, unsigned char *buffer,
                unsigned int src_len, int /* frame_seq */, struct video_frame_callbacks * /* callbacks */, struct pixfmt_desc *internal_prop) {
        auto *s = static_cast<state_video_decompress_j2k*>(state);
        struct cmpto_j2k_dec_img *img;
        std::pair<char *, size_t> decoded;
        void *tmp;

        if (s->out_codec == VIDEO_CODEC_NONE) {
                return j2k_probe_internal_codec(s->desc.color_spec, buffer, src_len, internal_prop);
        }

        if (s->in_frames >= s->max_in_frames + 1) {
                print_dropped(s->dropped++, s->platform);
                goto return_previous;
        }

        CHECK_OK(cmpto_j2k_dec_img_create(s->decoder, &img),
                        "Could not create frame", goto return_previous);

        tmp = malloc(src_len);
        memcpy(tmp, buffer, src_len);
        CHECK_OK(cmpto_j2k_dec_img_set_cstream(img, tmp, src_len, &release_cstream),
                        "Error setting cstream", cmpto_j2k_dec_img_destroy(img); goto return_previous);

        CHECK_OK(cmpto_j2k_dec_img_decode(img, s->settings), "Decode image",
                        cmpto_j2k_dec_img_destroy(img); goto return_previous);
        {
                lock_guard<mutex> lk(s->lock);
                s->in_frames++;
        }

return_previous:
        std::unique_lock<mutex> lk(s->lock);
        if (s->decompressed_frames.size() == 0) {
                return DECODER_NO_FRAME;
        }
        decoded = s->decompressed_frames.front();
        s->decompressed_frames.pop();
        lk.unlock();

        size_t linesize = vc_get_linesize(s->desc.width, s->out_codec);
        size_t frame_size = linesize * s->desc.height;
        if ((decoded.second + 3) / 4 * 4 != frame_size) { // for "RGBA with non-standard shift" (search) it would be (frame_size - 1)
                log_msg(LOG_LEVEL_WARNING, "%s Incorrect decoded size (%lu vs. %lu)\n",
                        MOD_NAME,
                        frame_size,
                        decoded.second);
        }

        for (size_t i = 0; i < s->desc.height; ++i) {
                memcpy(dst + i * s->pitch, decoded.first + i * linesize, std::min(linesize, decoded.second - std::min(decoded.second, i * linesize)));
        }

        free(decoded.first);

        return DECODER_GOT_FRAME;
}

static int j2k_decompress_get_property(void *state, int property, void *val, size_t *len) {
        UNUSED(state);
        int ret = false;

        switch (property) {
                case DECOMPRESS_PROPERTY_ACCEPTS_CORRUPTED_FRAME:
                        if (*len >= sizeof(int)) {
                                *(int *) val = false;
                                *len = sizeof(int);
                                ret = true;
                        }
                        break;
                default:
                        ret = false;
        }

        return ret;
}

static void j2k_decompress_done(void *state) {
        auto *s = static_cast<state_video_decompress_j2k*>(state);

        cmpto_j2k_dec_ctx_stop(s->decoder);
        pthread_join(s->thread_id, NULL);
        log_msg(LOG_LEVEL_VERBOSE, "%s Decoder stopped.\n", MOD_NAME);

        cmpto_j2k_dec_cfg_destroy(s->settings);
        cmpto_j2k_dec_ctx_destroy(s->decoder);

        while (s->decompressed_frames.size() > 0) {
                auto decoded = s->decompressed_frames.front();
                s->decompressed_frames.pop();
                free(decoded.first);
        }

        delete s;
}

static int j2k_decompress_get_priority(codec_t compression, struct pixfmt_desc internal, codec_t ugc) {
        if (compression != J2K && compression != J2KR) {
                return -1;
        }
        switch (ugc) {
                case VIDEO_CODEC_NONE:
                        return 50; // probe
                case UYVY:
                case v210:
                case RGB:
                case BGR:
                case RGBA:
                case R10k:
                case R12L:
                        break;
                default:
                        return -1;
        }
        if (ugc == VIDEO_CODEC_NONE) {
                return 50; // probe
        }
        if (internal.depth == 0) { // fallback - internal undefined
                return 800;
        }
        return internal.rgb == codec_is_a_rgb(ugc) ? 300 : -1;
}

static const struct video_decompress_info j2k_decompress_info = {
        j2k_decompress_init,
        j2k_decompress_reconfigure,
        j2k_decompress,
        j2k_decompress_get_property,
        j2k_decompress_done,
        j2k_decompress_get_priority,
};

REGISTER_MODULE(j2k, &j2k_decompress_info, LIBRARY_CLASS_VIDEO_DECOMPRESS, VIDEO_DECOMPRESS_ABI_VERSION);
