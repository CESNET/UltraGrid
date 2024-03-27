/**
 * @file   video_compress/j2k.cpp
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
 * Main idea behind the code below is to control how many frames the encoder
 * holds. The codec itself doesn't have a limit, thus without that it is
 * possible to run out of memory. This is possible even in the case when
 * the GPU is powerful enough due to the fact that CUDA registers the new
 * buffers which is very slow and because of that the frames cumulate before
 * the GPU encoder.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <cmpto_j2k_enc.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <climits>
#include <condition_variable>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef HAVE_CUDA
#include "cuda_wrapper.h"
#endif // HAVE_CUDA
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "module.h"
#include "utils/string.h" // replace_all
#include "tv.h"
#include "utils/color_out.h"
#include "utils/misc.h"
#include "utils/video_frame_pool.h"
#include "video.h"
#include "video_compress.h"

constexpr const char *MOD_NAME = "[Cmpto J2K enc.]";

#define ASSIGN_CHECK_VAL(var, str, minval) \
        do { \
                long long val = unit_evaluate(str, nullptr); \
                if (val < (minval) || val > UINT_MAX) { \
                        LOG(LOG_LEVEL_ERROR) \
                            << MOD_NAME << " Wrong value " << (str) \
                            << " for " #var "! Value must be >= " << (minval) \
                            << ".\n"; \
                        throw InvalidArgument(); \
                } \
                (var) = val; \
        } while (0)

#define CHECK_OK(cmd, err_msg, action_fail) do { \
        int j2k_error = cmd; \
        if (j2k_error != CMPTO_OK) {\
                log_msg(LOG_LEVEL_ERROR, "%s %s: %s\n", \
                        MOD_NAME, err_msg, cmpto_j2k_enc_get_last_error()); \
                action_fail;\
        } \
} while (0)

#define NOOP ((void) 0)

// Default CPU Settings
#define DEFAULT_CPU_THREAD_COUNT CMPTO_J2K_ENC_CPU_DEFAULT
#define MIN_CPU_THREAD_COUNT     CMPTO_J2K_ENC_CPU_NONE
#define DEFAULT_CPU_MEM_LIMIT    0      // DEFAULT_CPU_MEM_LIMIT should always be 0
#define DEFAULT_CPU_POOL_SIZE    8
#define DEFAULT_IMG_LIMIT        0      // Default number of images to be decoded by CPU (0 = CMPTO Default)
#define MIN_CPU_IMG_LIMIT        0      // Min number of images decoded by the CPU at once

// Default CUDA Settings
#define DEFAULT_CUDA_POOL_SIZE   4
#define DEFAULT_CUDA_TILE_LIMIT  1
#define DEFAULT_CUDA_MEM_LIMIT   1000000000ULL

// Default General Settings
#define DEFAULT_QUALITY          0.7
#ifdef HAVE_CUDA
#define DEFAULT_POOL_SIZE        DEFAULT_CUDA_POOL_SIZE
#else
#define DEFAULT_POOL_SIZE        DEFAULT_CPU_POOL_SIZE
#endif

using std::mutex;
using std::shared_ptr;

#ifdef HAVE_CUDA
struct cmpto_j2k_enc_cuda_host_buffer_data_allocator
    : public video_frame_pool_allocator {
        void *allocate(size_t size) override {
                void *ptr = nullptr;
                if (CUDA_WRAPPER_SUCCESS !=
                    cuda_wrapper_malloc_host(&ptr, size)) {
                        log_msg(LOG_LEVEL_ERROR, "Cannot allocate host buffer: %s\n",
                            cuda_wrapper_last_error_string());
                        return nullptr;
                }
                return ptr;
        }

        void deallocate(void *ptr) override { cuda_wrapper_free(ptr); }

        [[nodiscard]] video_frame_pool_allocator *clone() const override {
                return new cmpto_j2k_enc_cuda_host_buffer_data_allocator(*this);
        }
};
using allocator      = cmpto_j2k_enc_cuda_host_buffer_data_allocator;
using cuda_allocator = cmpto_j2k_enc_cuda_host_buffer_data_allocator;
#else
using allocator      = default_data_allocator;
#endif
using cpu_allocator  = default_data_allocator;

// Pre Declarations
static void j2k_compressed_frame_dispose(struct video_frame *frame);
static void j2k_compress_done(struct module *mod);
static void R12L_to_RG48(video_frame *dst, video_frame *src);

/** 
 * @brief Platforms available for J2K Compression
 */
enum j2k_compress_platform {
        NONE = 0,
        CPU = 1,
#ifdef HAVE_CUDA
        CUDA = 2,
#endif // HAVE_CUDA
};

/** 
 * @brief Struct to hold Platform Name and j2k_compress_platform Type
 */
struct j2k_compress_platform_info_t {
        const char* name;
        j2k_compress_platform platform;
};

// Supported Platforms for Compressing J2K
constexpr auto compress_platforms = std::array {
    j2k_compress_platform_info_t{"none", j2k_compress_platform::NONE},
    j2k_compress_platform_info_t{"cpu", j2k_compress_platform::CPU},
#ifdef HAVE_CUDA
    j2k_compress_platform_info_t{"cuda", j2k_compress_platform::CUDA}
#endif
};

/**
 * @fn get_platform_from_name
 * @brief Search for j2k_compress_platform from friendly name
 * @param name Friendly name of platform to search for
 * @return j2k_compress_platform that corresponds to name. If no match, return j2k_compress_platform::NONE
 */
[[nodiscard]][[maybe_unused]]
static j2k_compress_platform get_platform_from_name(std::string name) {
    std::transform(name.cbegin(), name.cend(), name.begin(), [](unsigned char c) { return std::tolower(c); });

    auto matches = [&name](const auto& p) { return name.compare(p.name) == 0; };

    if (const auto& it = std::find_if(compress_platforms.begin(), compress_platforms.end(), matches) ; it != compress_platforms.end()) {
        return it->platform;
    }

    return j2k_compress_platform::NONE;
}

/**
 * @brief Struct to hold UG and CMPTO Codec information
 */
struct Codec {
        codec_t ug_codec;
        enum cmpto_sample_format_type cmpto_sf;
        codec_t convert_codec;
        void (*convertFunc)(video_frame *dst, video_frame *src);
};

// Supported UG/CMPTO Compress Codecs
constexpr auto codecs = std::array{
        Codec{UYVY, CMPTO_422_U8_P1020, VIDEO_CODEC_NONE, nullptr},
        Codec{v210, CMPTO_422_U10_V210, VIDEO_CODEC_NONE, nullptr},
        Codec{RGB, CMPTO_444_U8_P012, VIDEO_CODEC_NONE, nullptr},
        Codec{RGBA, CMPTO_444_U8_P012Z, VIDEO_CODEC_NONE, nullptr},
        Codec{R10k, CMPTO_444_U10U10U10_MSB32BE_P210, VIDEO_CODEC_NONE, nullptr},
        Codec{R12L, CMPTO_444_U12_MSB16LE_P012, RG48, R12L_to_RG48},
};

/**
 * Exceptions for state_video_compress_j2k construction
 */

/// @brief HelpRequested Exception
struct HelpRequested : public std::exception {
        HelpRequested() = default;
};

/// @brief InvalidArgument Exception
struct InvalidArgument : public std::exception {
        InvalidArgument() = default;
};

/// @brief UnableToCreateJ2KEncoderCTX Exception
struct UnableToCreateJ2KEncoderCTX : public std::exception {
        UnableToCreateJ2KEncoderCTX() = default;
};

/// @brief Struct for options for J2K Compression Usage
struct opts {
        const char *label;
        const char *key;
        const char *description;
        const char *opt_str;
        const bool is_boolean;
};

#ifdef HAVE_CUDA
constexpr opts cuda_opts[2] = {
        {"Mem limit", "mem_limit", "CUDA device memory limit (in bytes), default: " TOSTRING(DEFAULT_CUDA_MEM_LIMIT), ":mem_limit=", false},
        {"Tile limit", "tile_limit", "Number of tiles encoded at one moment by GPU (less to reduce latency, more to increase performance, 0 means infinity). default: " TOSTRING(DEFAULT_CUDA_TILE_LIMIT), ":tile_limit=", false},
};
constexpr opts platform_opts[1] = {
        {"Plaform", "platform", "Platform device for the encoder to use, default: cuda", ":platform=", false},
};
#endif // HAVE_CUDA

constexpr opts cpu_opts[2] = {
        {"Thread count", "thread_count", "Number of threads to use on the CPU. 0 is all available. default: " TOSTRING(DEFAULT_CPU_THREAD_COUNT), ":thread_count=", false},
        {"Image limit", "img_limit", "Number of images which can be encoded at one moment by CPU. Maximum allowed limit is thread_count. 0 is default limit. default: " TOSTRING(DEFAULT_IMG_LIMIT), ":img_limit=", false},
};

constexpr opts general_opts[5] = {
        {"Bitrate", "quality", "Target bitrate", ":rate=", false},
        {"Quality", "quant_coeff", "Quality in range [0-1]. default: " TOSTRING(DEFAULT_QUALITY), ":quality=", false},
#ifdef HAVE_CUDA
        {"Pool size", "pool_size", "Total number of frames encoder can hold at one moment. Should be greater than tile_limit or img_limit. default: " TOSTRING(DEFAULT_POOL_SIZE), ":pool_size=", false},
#else
        {"Pool size", "pool_size", "Total number of frames encoder can hold at one moment. Should be greater than img_limit. default: " TOSTRING(DEFAULT_POOL_SIZE) , ":pool_size=", false},
#endif
        {"Use MCT", "mct", "Use MCT", ":mct", true},
        {"Lossless compression", "lossless", "Enable lossless compression. default: disabled", ":lossless", true}
};

/**
 * @fn usage
 * @brief Display J2K Compression Usage Information
 */
static void usage() {
        col() << "J2K compress platform support:\n";
        col() << "\tCPU .... yes\n";
#ifdef HAVE_CUDA
        col() << "\tCUDA ... yes\n";
#else
        col() << "\tCUDA ... no\n";
#endif

        col() << "J2K compress usage:\n";

        auto show_syntax = [](const auto& options) {
                for (const auto& opt : options) {
                        assert(strlen(opt.opt_str) >= 2);
                        col() << "[" << opt.opt_str;
                        if (!opt.is_boolean) {
                                col() << "<" << opt.opt_str[1] << ">"; // :quality -> <q> (first letter used as ":quality=<q>")
                        }
                        col() << "]";
                }
        };

        auto show_arguments = [](const auto& options) {
                for (const auto& opt : options) {
                        assert(strlen(opt.opt_str) >= 2);
                        if (opt.is_boolean) {
                                col() << TBOLD("\t" << opt.opt_str + 1 <<);
                        } else {
                                col() << TBOLD("\t<" << opt.opt_str[1] << ">");
                        }
                        col() << " - " << opt.description << "\n";
                }
        };

#ifdef HAVE_CUDA
        // CPU and CUDA Platforms Supported. Show platform= options
        col() << TERM_BOLD << TRED("\t-c cmpto_j2k:platform=cuda");
        show_syntax(cuda_opts);
        show_syntax(general_opts);
        col() << " [--cuda-device <c_index>]\n" << TERM_RESET;
        col() << TERM_BOLD << TRED("\t-c cmpto_j2k:platform=cpu");
        show_syntax(cpu_opts);
        show_syntax(general_opts);
#else // HAVE_CUDA
        // Only CPU Platform Supported. No option to switch platform from default.
        col() << TERM_BOLD << TRED("\t-c cmpto_j2k");
        show_syntax(cpu_opts);
        show_syntax(general_opts);
#endif
        col() << "\n" << TERM_RESET;
        col() << "where:\n";
#ifdef HAVE_CUDA
        show_arguments(platform_opts);
        show_arguments(cuda_opts);
        col() << TBOLD("\t<c_index>") << " - CUDA device(s) to use (comma separated)\n";
#endif // HAVE_CUDA
        show_arguments(cpu_opts);
        show_arguments(general_opts);
}


/**
 * @brief state_video_compress_j2k Class
 */
struct state_video_compress_j2k {
        explicit state_video_compress_j2k(struct module *parent);
        state_video_compress_j2k(struct module *parent, const char* opts);

        module                            module_data{};
        struct cmpto_j2k_enc_ctx          *context{};
        struct cmpto_j2k_enc_cfg          *enc_settings{};
        std::unique_ptr<video_frame_pool> pool;
        unsigned int                      in_frames{};          ///< number of currently encoding frames
        mutex                             lock;
        std::condition_variable           frame_popped;
        video_desc                        saved_desc{};         ///< for pool reconfiguration
        video_desc                        precompress_desc{};
        video_desc                        compressed_desc{};

        void (*convertFunc)(video_frame *dst, video_frame *src) { nullptr };

        // Generic Parameters
        double        quality              = DEFAULT_QUALITY;   // default image quality
        long long int rate                 = 0;                 // bitrate in bits per second
        int           mct                  = -1;                // force use of mct - -1 means default
        bool          lossless             = false;             // lossless encoding

        // CPU Parameters
        int           cpu_thread_count     = DEFAULT_CPU_THREAD_COUNT;
        unsigned int  cpu_img_limit        = DEFAULT_IMG_LIMIT;

        // CUDA Parameters
        unsigned long long cuda_mem_limit  = DEFAULT_CUDA_MEM_LIMIT;
        unsigned int       cuda_tile_limit = DEFAULT_CUDA_TILE_LIMIT;

        // Platform to use by default
#ifdef HAVE_CUDA
        j2k_compress_platform platform      = j2k_compress_platform::CUDA;
        unsigned int         max_in_frames = DEFAULT_CUDA_POOL_SIZE; ///< max number of frames between push and pop
#else
        j2k_compress_platform platform      = j2k_compress_platform::CPU;
        unsigned int         max_in_frames = DEFAULT_CPU_POOL_SIZE;  ///< max number of frames between push and pop
#endif

 private:
        void parse_fmt(const char* opts);
        bool initialize_j2k_enc_ctx();

        // CPU Parameter
        const size_t  cpu_mem_limit = 0;                // Not yet implemented as of v2.8.1. Must be 0.
};


/**
 * @brief state_video_compress_j2k default constructor to create from module
 * @param parent Base Module Struct
*/
state_video_compress_j2k::state_video_compress_j2k(struct module *parent)
        : pool(std::make_unique<video_frame_pool>(DEFAULT_POOL_SIZE, allocator())) {
        module_init_default(&module_data);
        module_data.cls         = MODULE_CLASS_DATA;
        module_data.priv_data   = this;
        module_data.deleter     = j2k_compress_done;
        module_register(&module_data, parent);
}

/**
 * @brief state_video_compress_j2k constructor to create from opts
 * @param parent Base Module Struct
 * @param opts Configuration options to construct class
 * @throw HelpRequested if help requested
 * @throw InvalidArgument if argument provided isn't known
 * @throw UnableToCreateJ2KEncoderCTX if failure to create J2K CTX
*/
state_video_compress_j2k::state_video_compress_j2k(struct module *parent, const char* opts) {
        try {
                parse_fmt(opts);
        } catch (...) {
                throw;
        }

        if (!initialize_j2k_enc_ctx()) {
                throw UnableToCreateJ2KEncoderCTX();
        }

        module_init_default(&module_data);
        module_data.cls         = MODULE_CLASS_DATA;
        module_data.priv_data   = this;
        module_data.deleter     = j2k_compress_done;
        module_register(&module_data, parent);
}

/// CUDA opt Syntax
// -c cmpto_j2k:platform=cuda[:mem_limit=<m>][:tile_limit=<t>][:rate=<r>][:lossless][:quality=<q>][:pool_size=<p>][:mct] [--cuda-device <c_index>]
/// CPU opt Syntax
// -c cmpto_j2k:platform=cpu[:thread_count=<t>][:img_limit=<i>][:rate=<r>][:lossless][:quality=<q>][:pool_size=<p>][:mct]
/**
 * @fn parse_fmt
 * @brief Parse options and configure class members accordingly
 * @param opts Configuration options
 * @throw HelpRequested if help requested
 * @throw InvalidArgument if argument provided isn't known
 */
void state_video_compress_j2k::parse_fmt(const char* opts) {
        auto split_arguments = [](std::string args, std::string delimiter) {
                auto token = std::string{};
                auto pos   = size_t{0};
                auto vec   = std::vector<std::string>{};

                if (args == "\0") {
                        return vec;
                }

                while ((pos = args.find(delimiter)) != std::string::npos) {
                        token = args.substr(0, pos);
                        vec.emplace_back(std::move(token));
                        args.erase(0, pos + delimiter.length());
                }

                vec.emplace_back(std::move(args));
                return vec;
        };

        auto args = split_arguments(opts, ":");

        // No Arguments provided, return and use defaults
        if (args.empty()) {
                return;
        }

        const auto *version = cmpto_j2k_enc_get_version();
        log_msg(LOG_LEVEL_INFO, "%s Using Codec version: %s\n",
                MOD_NAME,
                (version == nullptr ? "(unknown)" : version->name));

        const char* item = "";

        /**
         * Check if :pool_size= set manually during argument parsing.
         *  Since max_in_frames is default initialized to match compile time platform default (CUDA or CPU)
         *  Changing from :platform=cuda default to :platform=cpu default will not automatically
         *  set :pool_size= during argument parsing because opts can passed be out of order.
         *  
         * To prevent potential for overwriting user's defined default, set is_pool_size_manually_configured=true
         *  during argument parsing and check before final function return
         *  
         * If pool size is manually configured, do not set to default. 
         *  Otherwise, set max_in_frames = platform default
         */
        auto is_pool_size_manually_configured = false;

        for (const auto& arg : args) {
                item = arg.c_str();
                if (strcasecmp("help", item) == 0)              {       // :help
                        usage();
                        throw HelpRequested();

                } else if (IS_KEY_PREFIX(item, "platform"))     {       // :platform=
                        const char *const platform_name = strchr(item, '=') + 1;
                        platform = get_platform_from_name(platform_name);
                        if (j2k_compress_platform::NONE == platform) {
                                log_msg(LOG_LEVEL_ERROR,
                                        "%s Unable to find requested encoding platform: \"%s\"\n",
                                        MOD_NAME,
                                        platform_name);
                                throw InvalidArgument();
                        }

                } else if (strcasecmp("lossless", item) == 0)   {       // :lossless
                        lossless = true;

                } else if (IS_KEY_PREFIX(item, "mem_limit"))    {       // :mem_limit=
                        ASSIGN_CHECK_VAL(cuda_mem_limit, strchr(item, '=') + 1, 1);

                } else if (IS_KEY_PREFIX(item, "thread_count")) {       // :thread_count=
                        cpu_thread_count = atoi(strchr(item, '=') + 1);
                        ASSIGN_CHECK_VAL(cpu_thread_count, strchr(item, '=') + 1, MIN_CPU_THREAD_COUNT);

                } else if (IS_KEY_PREFIX(item, "tile_limit"))   {       // :tile_limit=
                        ASSIGN_CHECK_VAL(cuda_tile_limit, strchr(item, '=') + 1, 0);

                } else if (IS_KEY_PREFIX(item, "img_limit"))    {       // :img_limit=
                        ASSIGN_CHECK_VAL(cpu_img_limit, strchr(item, '=') + 1, MIN_CPU_IMG_LIMIT);

                } else if (IS_KEY_PREFIX(item, "rate"))         {       // :rate=
                        ASSIGN_CHECK_VAL(rate, strchr(item, '=') + 1, 1);

                } else if (IS_KEY_PREFIX(item, "quality"))      {       // :quality=
                        quality = std::stod(strchr(item, '=') + 1);
                        if (quality < 0.0 || quality > 1.0) {
                                log_msg(LOG_LEVEL_ERROR,
                                        "%s Quality should be in interval [0-1]\n",
                                        MOD_NAME);
                                throw InvalidArgument();
                        }

                } else if (IS_KEY_PREFIX(item, "pool_size"))    {       // :pool_size=
                        ASSIGN_CHECK_VAL(max_in_frames, strchr(item, '=') + 1, 1);
                        is_pool_size_manually_configured = true;

                } else if (strcasecmp("mct", item) == 0) {             // :mct
                        mct = strcasecmp("mct", item) ? 1 : 0;

                } else {
                        log_msg(LOG_LEVEL_ERROR,
                                "%s Unable to find option: \"%s\"\n",
                                MOD_NAME, item);
                        throw InvalidArgument();
                }
        }

        // If CPU selected
        if (j2k_compress_platform::CPU == platform) {
                /**
                 * Confirm thread_count != CMPTO_J2K_ENC_CPU_DEFAULT (0)
                 *  If it does, img_limit can be > thread_count since all threads used
                 * 
                 * If thread_count is not 0, confirm img_limit doesn't exceed thread_count
                 *  Set img_limit = thread_count if exeeded
                 */
                if (cpu_thread_count != CMPTO_J2K_ENC_CPU_DEFAULT && cpu_thread_count < static_cast<int>(cpu_img_limit)) {
                        log_msg(LOG_LEVEL_INFO,
                                "%s img_limit (%i) exceeds thread_count. Lowering to img_limit to %i to match thread_count.\n",
                                MOD_NAME,
                                cpu_img_limit,
                                cpu_thread_count);
                        cpu_img_limit = cpu_thread_count;
                }

                // If pool_size was manually set, ignore this check.
                // Otherwise, if it was not set, confirm that max_in_frames matches DEFAULT_CPU_POOL_SIZE
                if (!is_pool_size_manually_configured && max_in_frames != DEFAULT_CPU_POOL_SIZE) {
                        log_msg(LOG_LEVEL_DEBUG,
                                "%s max_in_frames set to CPU default: %i",
                                MOD_NAME,
                                DEFAULT_CPU_POOL_SIZE);
                        max_in_frames = DEFAULT_CPU_POOL_SIZE;
                }
        }
}

/**
 * @fn initialize_j2k_enc_ctx
 * @brief Initialize internal cmpto_j2k_enc_ctx_cfg for requested platform and settings
 * @return true if successsfully configured
 * @return false if unable to configure
 */
[[nodiscard]]
bool state_video_compress_j2k::initialize_j2k_enc_ctx() {
        struct cmpto_j2k_enc_ctx_cfg *ctx_cfg;

        CHECK_OK(cmpto_j2k_enc_ctx_cfg_create(&ctx_cfg),
                        "Context configuration create",
                        return false);

        if (j2k_compress_platform::CPU == platform) {
                log_msg(LOG_LEVEL_INFO, "%s Configuring for CPU\n", MOD_NAME);
                pool = std::make_unique<video_frame_pool>(max_in_frames, cpu_allocator());
                // for (unsigned int i = 0; i < cpu_count ; )
                CHECK_OK(cmpto_j2k_enc_ctx_cfg_add_cpu(
                                ctx_cfg,
                                cpu_thread_count,
                                cpu_mem_limit,
                                cpu_img_limit),
                        "Setting CPU device",
                        return false);

                log_msg(LOG_LEVEL_INFO, "%s Using %s threads on CPU. Thread Count = %i, Image Limit = %i\n",
                        MOD_NAME,
                        (cpu_thread_count == 0 ? "all available" : std::to_string(cpu_thread_count).c_str()),
                        cpu_thread_count,
                        cpu_img_limit);
        }

#ifdef HAVE_CUDA
        if (j2k_compress_platform::CUDA == platform) {
                log_msg(LOG_LEVEL_INFO, "%s Configuring for CUDA\n", MOD_NAME);
                pool = std::make_unique<video_frame_pool>(max_in_frames, cuda_allocator());
                for (unsigned int i = 0; i < cuda_devices_count; ++i) {
                        CHECK_OK(cmpto_j2k_enc_ctx_cfg_add_cuda_device(
                                        ctx_cfg,
                                        cuda_devices[i],
                                        cuda_mem_limit,
                                        cuda_tile_limit),
                                "Setting CUDA device",
                                return false);
                }
        }
#endif // HAVE_CUDA

        CHECK_OK(cmpto_j2k_enc_ctx_create(ctx_cfg, &context),
                        "Context create",
                        return false);

        CHECK_OK(cmpto_j2k_enc_ctx_cfg_destroy(ctx_cfg),
                        "Context configuration destroy",
                        NOOP);

        CHECK_OK(cmpto_j2k_enc_cfg_create(
                        context,
                        &enc_settings),
                        "Creating context configuration:",
                        return false);
        if (lossless) {
                CHECK_OK(cmpto_j2k_enc_cfg_set_lossless(
                                enc_settings,
                                lossless ? 1 : 0),
                                "Enabling lossless",
                                return false);
        } else {
                CHECK_OK(cmpto_j2k_enc_cfg_set_quantization(
                                enc_settings,
                                quality /* 0.0 = poor quality, 1.0 = full quality */),
                                "Setting quantization",
                                NOOP);
        }

        CHECK_OK(cmpto_j2k_enc_cfg_set_resolutions(enc_settings, 6),
                        "Setting DWT levels",
                        NOOP);

        return true;
}

static void R12L_to_RG48(video_frame *dst, video_frame *src) {
        int src_pitch = vc_get_linesize(src->tiles[0].width, src->color_spec);
        int dst_pitch = vc_get_linesize(dst->tiles[0].width, dst->color_spec);

        unsigned char *s = (unsigned char *) src->tiles[0].data;
        unsigned char *d = (unsigned char *) dst->tiles[0].data;
        decoder_t vc_copylineR12LtoRG48 = get_decoder_from_to(R12L, RG48);

        for (unsigned i = 0; i < src->tiles[0].height; i++) {
                vc_copylineR12LtoRG48(d, s, dst_pitch, 0, 0, 0);
                s += src_pitch;
                d += dst_pitch;
        }
}

static bool configure_with(struct state_video_compress_j2k *s, struct video_desc desc) {
        enum cmpto_sample_format_type sample_format;
        bool found = false;
        auto matches = [&](const Codec& codec) { return codec.ug_codec == desc.color_spec; };

        if (const auto& codec = std::find_if(codecs.begin(), codecs.end(), matches) ; codec != codecs.end()) {
                sample_format = codec->cmpto_sf;
                s->convertFunc = codec->convertFunc;
                s->precompress_desc = desc;
                if (codec->convert_codec != VIDEO_CODEC_NONE) {
                        s->precompress_desc.color_spec = codec->convert_codec;
                }
                found = true;
        }

        if (!found) {
                log_msg(LOG_LEVEL_ERROR, "%s Failed to find suitable pixel format\n", MOD_NAME);
                return false;
        }

        CHECK_OK(cmpto_j2k_enc_cfg_set_samples_format_type(s->enc_settings, sample_format),
                        "Setting sample format", return false);
        CHECK_OK(cmpto_j2k_enc_cfg_set_size(s->enc_settings, desc.width, desc.height),
                        "Setting image size", return false);
        if (s->rate) {
                CHECK_OK(cmpto_j2k_enc_cfg_set_rate_limit(s->enc_settings,
                                        CMPTO_J2K_ENC_COMP_MASK_ALL,
                                        CMPTO_J2K_ENC_RES_MASK_ALL, s->rate / 8 / desc.fps),
                                "Setting rate limit",
                                NOOP);
        }

        int mct = s->mct;
        if (mct == -1) {
                mct = codec_is_a_rgb(desc.color_spec) ? 1 : 0;
        }
        CHECK_OK(cmpto_j2k_enc_cfg_set_mct(s->enc_settings, mct),
                        "Setting MCT",
                        NOOP);

        s->compressed_desc = desc;
        s->compressed_desc.color_spec = codec_is_a_rgb(desc.color_spec) ? J2KR : J2K;
        s->compressed_desc.tile_count = 1;

        s->saved_desc = desc;

        return true;
}

static shared_ptr<video_frame> get_copy(struct state_video_compress_j2k *s, video_frame *frame) {
        std::shared_ptr<video_frame> ret = s->pool->get_frame();

        if (s->convertFunc) {
                s->convertFunc(ret.get(), frame);
        } else {
                memcpy(ret->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
        }

        return ret;
}

/// auxilliary data structure passed with encoded frame
struct custom_data {
        custom_data()                               = delete;
        custom_data(custom_data &b)                 = delete;
        custom_data &operator=(const custom_data &) = delete;
        ~custom_data()                              = delete;
        shared_ptr<video_frame> frame;
        video_desc              desc;
        // metadata stored separately, frame may have already been deallocated
        // by our release_cstream callback
        char metadata[VF_METADATA_SIZE];
};

/**
 * @fn j2k_compress_pop
 * @note
 * Do not return empty frame in case of error - that would be interpreted
 * as a poison pill (see below) and would stop the further processing
 * pipeline. Because of that goto + start label is used.
 */
#define HANDLE_ERROR_COMPRESS_POP do { cmpto_j2k_enc_img_destroy(img); goto start; } while (0)
static std::shared_ptr<video_frame> j2k_compress_pop(struct module *state) {
start:
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) state;

        struct cmpto_j2k_enc_img *img;
        int status;
        CHECK_OK(cmpto_j2k_enc_ctx_get_encoded_img(
                                s->context,
                                1,
                                &img /* Set to NULL if encoder stopped */,
                                &status), "Encode image", HANDLE_ERROR_COMPRESS_POP);
        {
                std::unique_lock<mutex> lk(s->lock);
                s->in_frames--;
                s->frame_popped.notify_one();
        }
        if (!img) {
                // this happens cmpto_j2k_enc_ctx_stop() is called
                // pass poison pill further
                return {};
        }
        if (status != CMPTO_J2K_ENC_IMG_OK) {
                const char * encoding_error = "";
                CHECK_OK(cmpto_j2k_enc_img_get_error(img, &encoding_error), "get error status",
                                encoding_error = "(failed)");
                log_msg(LOG_LEVEL_ERROR, "Image encoding failed: %s\n", encoding_error);
                goto start;
        }
        struct custom_data *udata = nullptr;
        size_t len;
        CHECK_OK(cmpto_j2k_enc_img_get_custom_data(img, (void **) &udata, &len),
                        "get custom data", HANDLE_ERROR_COMPRESS_POP);
        size_t size;
        void * ptr;
        CHECK_OK(cmpto_j2k_enc_img_get_cstream(img, &ptr, &size),
                        "get cstream", HANDLE_ERROR_COMPRESS_POP);

        struct video_frame *out = vf_alloc_desc(udata->desc);
        vf_restore_metadata(out, udata->metadata);
        out->tiles[0].data_len = size;
        out->tiles[0].data = (char *) malloc(size);
        memcpy(out->tiles[0].data, ptr, size);
        CHECK_OK(cmpto_j2k_enc_img_destroy(img), "Destroy image", NOOP);
        out->callbacks.dispose = j2k_compressed_frame_dispose;
        out->compress_end = get_time_in_ns();
        return shared_ptr<video_frame>(out, out->callbacks.dispose);
}

static struct module * j2k_compress_init(struct module *parent, const char *opts) {
        try {
                auto *s = new state_video_compress_j2k(parent, opts);
                return &s->module_data;
        } catch (HelpRequested const& e) {
                return static_cast<module*>(INIT_NOERR);
        } catch (InvalidArgument const& e) {
                return NULL;
        } catch (UnableToCreateJ2KEncoderCTX const& e) {
                return NULL;
        } catch (...) {
                return NULL;
        }
}

static void j2k_compressed_frame_dispose(struct video_frame *frame) {
        free(frame->tiles[0].data);
        vf_free(frame);
}

static void release_cstream(void * custom_data, size_t custom_data_size, const void * codestream, size_t codestream_size) {
        (void) codestream; (void) custom_data_size; (void) codestream_size;
        auto *udata = static_cast<struct custom_data *>(custom_data);
        udata->frame.~shared_ptr<video_frame>();
}

#define HANDLE_ERROR_COMPRESS_PUSH if (img) cmpto_j2k_enc_img_destroy(img); return
static void j2k_compress_push(struct module *state, std::shared_ptr<video_frame> tx) {
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) state;
        struct cmpto_j2k_enc_img *img = NULL;
        struct custom_data *udata = nullptr;

        if (tx == NULL) { // pass poison pill through encoder
                CHECK_OK(cmpto_j2k_enc_ctx_stop(s->context), "stop", NOOP);
                return;
        }

        const struct video_desc desc = video_desc_from_frame(tx.get());
        if (!video_desc_eq(s->saved_desc, desc)) {
                int ret = configure_with(s, desc);
                if (!ret) {
                        return;
                }
                s->pool->reconfigure(s->precompress_desc, vc_get_linesize(s->precompress_desc.width, s->precompress_desc.color_spec)
                                * s->precompress_desc.height);
        }

        assert(tx->tile_count == 1); // TODO

        CHECK_OK(cmpto_j2k_enc_img_create(s->context, &img),
                        "Image create", return);

        /**
         * Copy video desc to udata (to be able to reconstruct in j2k_compress_pop().
         * Further make a place for a shared pointer of allocated data, deleter
         * returns frame to pool in call of release_cstream() callback (called when
         * encoder no longer needs the input data).
         */
        CHECK_OK(cmpto_j2k_enc_img_allocate_custom_data(
                                img,
                                sizeof *udata,
                                (void **) &udata),
                        "Allocate custom image data",
                        HANDLE_ERROR_COMPRESS_PUSH);
        memcpy(&udata->desc, &s->compressed_desc, sizeof(s->compressed_desc));
        new (&udata->frame) shared_ptr<video_frame>(get_copy(s, tx.get()));
        vf_store_metadata(tx.get(), udata->metadata);

        CHECK_OK(cmpto_j2k_enc_img_set_samples(img, udata->frame->tiles[0].data,
                                               udata->frame->tiles[0].data_len,
                                               release_cstream),
                 "Setting image samples", HANDLE_ERROR_COMPRESS_PUSH);

        std::unique_lock<mutex> lk(s->lock);
        s->frame_popped.wait(lk, [s]{return s->in_frames < s->max_in_frames;});
        lk.unlock();
        CHECK_OK(cmpto_j2k_enc_img_encode(img, s->enc_settings),
                        "Encode image", return);
        lk.lock();
        s->in_frames++;
        lk.unlock();

}

static void j2k_compress_done(struct module *mod) {
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) mod->priv_data;

        cmpto_j2k_enc_cfg_destroy(s->enc_settings);
        cmpto_j2k_enc_ctx_destroy(s->context);

        delete s;
}

static compress_module_info get_cmpto_j2k_module_info() {
        compress_module_info module_info;
        module_info.name = "cmpto_j2k";

        auto add_module_options = [&](const auto& options) {
                for (const auto& opt : options) {
                        module_info.opts.emplace_back(module_option{opt.label,
                                        opt.description, opt.key, opt.opt_str, opt.is_boolean});
                }
        };

#ifdef HAVE_CUDA
        add_module_options(cuda_opts);
#endif // HAVE_CUDA
        add_module_options(cpu_opts);
        add_module_options(general_opts);

        codec codec_info;
        codec_info.name     = "Comprimato jpeg2000";
        codec_info.priority = 400;
        codec_info.encoders.emplace_back(encoder{"default", ""});

        module_info.codecs.emplace_back(std::move(codec_info));

        return module_info;
}

static struct video_compress_info j2k_compress_info = {
        "cmpto_j2k",
        j2k_compress_init,
        NULL,
        NULL,
        NULL,
        NULL,
        j2k_compress_push,
        j2k_compress_pop,
        get_cmpto_j2k_module_info
};

REGISTER_MODULE(cmpto_j2k, &j2k_compress_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
