/**
 * @file   video_compress/j2k.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2024 CESNET
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
 *
 * @todo
 * - check support for multiple CUDA devices with CUDA buffers
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif // HAVE_CONFIG_H

#include <cassert>
#include <cmath>
#include <climits>
#include <condition_variable>
#include <mutex>
#include <utility>

#include <cmpto_j2k_enc.h>

#ifdef HAVE_CUDA
#include "cuda_wrapper.h"
#include "cuda_wrapper/kernels.hpp"
#endif
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "module.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/macros.h"            // for IS_KEY_PREFIX, TOSTRING
#include "utils/misc.h"
#include "utils/parallel_conv.h"
#include "utils/video_frame_pool.h"
#include "video_codec.h"             // for vc_get_linesize, codec_is_a_rgb
#include "video_compress.h"
#include "video_frame.h"             // for vf_alloc_desc, vf_free, vf_resto...

#define MOD_NAME "[Cmpto J2K enc.] "

#define CHECK_OK(cmd, err_msg, action_fail) do { \
        int j2k_error = cmd; \
        if (j2k_error != CMPTO_OK) {\
                log_msg(LOG_LEVEL_ERROR, "[J2K enc.] %s: %s\n", \
                                err_msg, cmpto_j2k_enc_get_last_error()); \
                action_fail;\
        } \
} while(0)

#define NOOP ((void) 0)
#define DEFAULT_QUALITY 0.7
/// default max size of state_video_compress_j2k::pool and also value
/// for state_video_compress_j2k::max_in_frames
#define DEFAULT_POOL_SIZE 4

/// number of frames that encoder encodes at moment
#define DEFAULT_TILE_LIMIT 1
#define DEFAULT_MEM_LIMIT 1000000000LLU

using std::condition_variable;
using std::mutex;
using std::stod;
using std::shared_ptr;
using std::unique_lock;

#ifdef HAVE_CUDA
template <decltype(cuda_wrapper_malloc_host) alloc,
          decltype(cuda_wrapper_free_host)   free>
struct cmpto_j2k_enc_cuda_buffer_data_allocator
    : public video_frame_pool_allocator {
        void *allocate(size_t size) override
        {
                if (alloc == cuda_wrapper_malloc) {
                        cuda_wrapper_set_device((int) cuda_devices[0]);
                }
                void *ptr = nullptr;
                if (CUDA_WRAPPER_SUCCESS !=
                    alloc(&ptr, size)) {
                        MSG(ERROR, "Cannot allocate host buffer: %s\n",
                            cuda_wrapper_last_error_string());
                        return nullptr;
                }
                return ptr;
        }
        void deallocate(void *ptr) override { free(ptr); }
        [[nodiscard]] video_frame_pool_allocator *clone() const override
        {
                return new cmpto_j2k_enc_cuda_buffer_data_allocator(*this);
        }
};
#endif

struct state_video_compress_j2k {
        struct module module_data{};
        struct cmpto_j2k_enc_ctx *context{};
        struct cmpto_j2k_enc_cfg *enc_settings{};
        long long int rate = 0; ///< bitrate in bits per second
        int mct = -1; // force use of mct - -1 means default
        bool pool_in_device_memory = false; ///< frames in pool are on GPU
        video_frame_pool pool; ///< pool for frames allocated by us but not yet consumed by encoder

        // settings
        unsigned int max_in_frames =
            DEFAULT_POOL_SIZE; ///< max number of frames between push and pop
        double        quality    = DEFAULT_QUALITY;
        long long int mem_limit  = DEFAULT_MEM_LIMIT;
        unsigned int  tile_limit = DEFAULT_TILE_LIMIT;

        unsigned int in_frames{};   ///< number of currently encoding frames
        mutex lock;
        condition_variable frame_popped;
        video_desc saved_desc{};
        codec_t precompress_codec = VC_NONE;
        video_desc compressed_desc{};

        condition_variable configure_cv;
        bool               configured  = false;
        bool               should_exit = false;
};

// prototypes
static void j2k_compressed_frame_dispose(struct video_frame *frame);
static void j2k_compress_done(struct module *mod);
static void cleanup_common(struct state_video_compress_j2k *s);

static void parallel_conv(video_frame *dst, video_frame *src){
        int src_pitch = vc_get_linesize(src->tiles[0].width, src->color_spec);
        int dst_pitch = vc_get_linesize(dst->tiles[0].width, dst->color_spec);

        decoder_t decoder =
            get_decoder_from_to(src->color_spec, dst->color_spec);
        assert(decoder != nullptr);
        time_ns_t t0 = get_time_in_ns();
        parallel_pix_conv((int) src->tiles[0].height, dst->tiles[0].data,
                          dst_pitch, src->tiles[0].data, src_pitch,
                          decoder, 0);
        if (log_level >= LOG_LEVEL_DEBUG) {
                MSG(DEBUG, "pixfmt conversion duration: %f ms\n",
                    NS_TO_MS((double) (get_time_in_ns() - t0)));
        }
}

const cmpto_j2k_enc_preprocessor_run_callback_cuda r12l_to_rg48_cuda =
#ifdef HAVE_CUDA
    preprocess_r12l_to_rg48;
#else
    nullptr;
#endif

static struct {
        codec_t ug_codec;
        enum cmpto_sample_format_type cmpto_sf;
        codec_t convert_codec;
        cmpto_j2k_enc_preprocessor_run_callback_cuda cuda_convert_func;
} codecs[] = {
        {UYVY, CMPTO_422_U8_P1020, VIDEO_CODEC_NONE, nullptr},
        {v210, CMPTO_422_U10_V210, VIDEO_CODEC_NONE, nullptr},
        {RGB, CMPTO_444_U8_P012, VIDEO_CODEC_NONE, nullptr},
        {RGBA, CMPTO_444_U8_P012Z, VIDEO_CODEC_NONE, nullptr},
        {R10k, CMPTO_444_U10U10U10_MSB32BE_P210, VIDEO_CODEC_NONE, nullptr},
        {RG48, CMPTO_444_U12_MSB16LE_P012, VC_NONE, nullptr},
        {R12L, CMPTO_444_U12_MSB16LE_P012, RG48, r12l_to_rg48_cuda},
};

#define CPU_CONV_PARAM "j2k-enc-cpu-conv"
ADD_TO_PARAM(
    CPU_CONV_PARAM,
    "* " CPU_CONV_PARAM "\n"
    "  Enforce CPU conversion instead of CUDA (applicable to R12L now)\n");
static void
set_pool(struct state_video_compress_j2k *s, bool have_gpu_preprocess)
{
#ifdef HAVE_CUDA
        s->pool_in_device_memory = false;
        if (cuda_devices_count > 1) {
                MSG(WARNING, "More than 1 CUDA device will use CPU buffers and "
                             "conversion...\n");
        } else if (s->precompress_codec == VC_NONE || have_gpu_preprocess) {
                s->pool_in_device_memory = true;
                s->pool                  = video_frame_pool(
                    s->max_in_frames,
                    cmpto_j2k_enc_cuda_buffer_data_allocator<
                                         cuda_wrapper_malloc, cuda_wrapper_free>());
                return;
        }
        s->pool = video_frame_pool(
            s->max_in_frames,
            cmpto_j2k_enc_cuda_buffer_data_allocator<cuda_wrapper_malloc_host,
                                                     cuda_wrapper_free_host>());
#else
        assert(!have_gpu_preprocess); // if CUDA not found, we shouldn't have
        s->pool = video_frame_pool(s->max_in_frames, default_data_allocator());
#endif
}

static bool configure_with(struct state_video_compress_j2k *s, struct video_desc desc){
        enum cmpto_sample_format_type sample_format;
        cmpto_j2k_enc_preprocessor_run_callback_cuda cuda_convert_func =
            nullptr;
        bool found = false;

        for(const auto &codec : codecs){
                if(codec.ug_codec == desc.color_spec){
                        sample_format = codec.cmpto_sf;
                        s->precompress_codec = codec.convert_codec;
                        cuda_convert_func = codec.cuda_convert_func;
                        found = true;
                        break;
                }
        }

        if(!found){
                log_msg(LOG_LEVEL_ERROR, "[J2K] Failed to find suitable pixel format\n");
                return false;
        }

        if (s->configured) {
                unique_lock<mutex> lk(s->lock);
                CHECK_OK(cmpto_j2k_enc_ctx_stop(s->context), "stop", abort());
                s->frame_popped.wait(lk, [s] { return s->in_frames == 0; });
                cleanup_common(s);
                s->configured = false;
        }

        if (get_commandline_param(CPU_CONV_PARAM) != nullptr) {
                cuda_convert_func = nullptr;
        }

        struct cmpto_j2k_enc_ctx_cfg *ctx_cfg = nullptr;
        CHECK_OK(cmpto_j2k_enc_ctx_cfg_create(&ctx_cfg),
                 "Context configuration create", return false);
        for (unsigned int i = 0; i < cuda_devices_count; ++i) {
                CHECK_OK(
                    cmpto_j2k_enc_ctx_cfg_add_cuda_device(
                        ctx_cfg, cuda_devices[i], s->mem_limit, s->tile_limit),
                    "Setting CUDA device", return false);
        }
        if (cuda_convert_func != nullptr) {
                CHECK_OK(cmpto_j2k_enc_ctx_cfg_set_preprocessor_cuda(
                             ctx_cfg, nullptr, nullptr, cuda_convert_func),
                         "Setting CUDA preprocess", return false);
        }

        CHECK_OK(cmpto_j2k_enc_ctx_create(ctx_cfg, &s->context),
                 "Context create", return false);
        CHECK_OK(cmpto_j2k_enc_ctx_cfg_destroy(ctx_cfg),
                 "Context configuration destroy", NOOP);

        CHECK_OK(cmpto_j2k_enc_cfg_create(s->context, &s->enc_settings),
                 "Creating context configuration:", return false);
        CHECK_OK(cmpto_j2k_enc_cfg_set_quantization(
                     s->enc_settings,
                     s->quality /* 0.0 = poor quality, 1.0 = full quality */
                     ),
                 "Setting quantization", NOOP);

        CHECK_OK(cmpto_j2k_enc_cfg_set_resolutions(s->enc_settings, 6),
                 "Setting DWT levels", NOOP);

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

        char rate[100];
        snprintf_ch(rate, "%s", s->rate == 0 ? "unset" : format_in_si_units(s->rate));
        MSG(INFO,
            "Using parameters: quality=%.2f, bitrate=%sbps, mem_limit=%sB, "
            "tile_limit=%u, pool_size=%u, mct=%d\n",
            s->quality, rate, format_in_si_units(s->mem_limit), s->tile_limit,
            s->max_in_frames, mct);

        set_pool(s, cuda_convert_func != nullptr);

        s->compressed_desc = desc;
        s->compressed_desc.color_spec = codec_is_a_rgb(desc.color_spec) ? J2KR : J2K;
        s->compressed_desc.tile_count = 1;

        s->saved_desc = desc;

        s->configured = true;
        s->configure_cv.notify_one();

        return true;
}

/**
 * @brief copies frame from RAM to GPU
 *
 * Does the pixel format conversion as well if specified.
 */
static void
do_gpu_copy(std::shared_ptr<video_frame> &ret, video_frame *in_frame)
{
#ifdef HAVE_CUDA
        cuda_wrapper_set_device((int) cuda_devices[0]);
        cuda_wrapper_memcpy(ret->tiles[0].data, in_frame->tiles[0].data,
                            in_frame->tiles[0].data_len,
                            CUDA_WRAPPER_MEMCPY_HOST_TO_DEVICE);
#else
        (void) ret, (void) in_frame;
        abort(); // must not reach here
#endif
}

static shared_ptr<video_frame> get_copy(struct state_video_compress_j2k *s, video_frame *frame){
        std::shared_ptr<video_frame> ret = s->pool.get_frame();

        if (s->pool_in_device_memory) {
                do_gpu_copy(ret, frame);
        } else if (s->precompress_codec != VC_NONE) {
                parallel_conv(ret.get(), frame);
        } else {
                memcpy(ret->tiles[0].data, frame->tiles[0].data,
                       frame->tiles[0].data_len);
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
static std::shared_ptr<video_frame> j2k_compress_pop(struct module *state)
{
        auto *s = (struct state_video_compress_j2k *) state;
start:
        {
                unique_lock<mutex> lk(s->lock);
                s->configure_cv.wait(lk, [s] { return s->configured ||
                                                      s->should_exit; });
                if (s->should_exit) {
                        return {}; // pass poison pill further
                }
        }

        struct cmpto_j2k_enc_img *img;
        int status;
        CHECK_OK(cmpto_j2k_enc_ctx_get_encoded_img(
                     s->context, 1, &img /* Set to NULL if encoder stopped */,
                     &status),
                 "Encode image pop", HANDLE_ERROR_COMPRESS_POP);
        if (img == nullptr) {
                // this happens when cmpto_j2k_enc_ctx_stop() is called
                goto start; // reconfiguration or exit
        } else {
                unique_lock<mutex> lk(s->lock);
                s->in_frames--;
                s->frame_popped.notify_one();
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

struct {
        const char *label;
        const char *key;
        const char *description;
        const char *opt_str;
        const bool is_boolean;
        const char *placeholder;
} usage_opts[] = {
        {"Bitrate", "quality", "Target bitrate", ":rate=", false, "70M"},
        {"Quality", "quant_coeff", "Quality in range [0-1], default: " TOSTRING(DEFAULT_QUALITY), ":quality=", false, TOSTRING(DEFAULT_QUALITY)},
        {"Mem limit", "mem_limit", "CUDA device memory limit (in bytes), default: " TOSTRING(DEFAULT_MEM_LIMIT), ":mem_limit=", false, TOSTRING(DEFAULT_MEM_LIMIT)},
        {"Tile limit", "tile_limit", "Number of tiles encoded at moment (less to reduce latency, more to increase performance, 0 means infinity), default: " TOSTRING(DEFAULT_TILE_LIMIT), ":tile_limit=", false, TOSTRING(DEFAULT_TILE_LIMIT)},
        {"Pool size", "pool_size", "Total number of tiles encoder can hold at moment (same meaning as above), default: " TOSTRING(DEFAULT_POOL_SIZE) ", should be greater than <t>", ":pool_size=", false, TOSTRING(DEFAULT_POOL_SIZE)},
        {"Use MCT", "mct", "use MCT", ":mct", true, ""},
};

static void usage() {
        col() << "J2K compress usage:\n";
        col() << TERM_BOLD << TRED("\t-c cmpto_j2k");
        for(const auto& opt : usage_opts){
                assert(strlen(opt.opt_str) >= 2);
                col() << "[" << opt.opt_str;
                if (!opt.is_boolean) {
                        col() << "<" << opt.opt_str[1] << ">"; // :quality -> <q> (first letter used as ":quality=<q>")
                }
                col() << "]";
        }
        col() << "\n\t\t[--cuda-device <c_index>] [--param " CPU_CONV_PARAM "]\n" << TERM_RESET;

        col() << "where:\n";
        for(const auto& opt : usage_opts){
                if (opt.is_boolean) {
                        col() << TBOLD("\t" << opt.opt_str + 1 <<);
                } else {
                        col() << TBOLD("\t<" << opt.opt_str[1] << ">");
                }
                col() << " - " << opt.description << "\n";
        }
        col() << TBOLD("\t<c_index>") << " - CUDA device(s) to use (comma separated)\n";
        col() << TBOLD("\t--param " CPU_CONV_PARAM)
              << " - use CPU for pixfmt conversion (useful if GPU\n\t\tis fully "
                 "occupied by the encoder; an option for decoder exists as "
                 "well)\n";
        color_printf("\nOption prefixes (eg. 'q=' for quality) can be used. SI "
                     "suffixes are recognized (eg. 'r=7.5M').\n");
#ifdef HAVE_CUDA
        constexpr char cuda_supported[] = "YES";
#else
        constexpr char cuda_supported[] = TRED("NO");
#endif
        color_printf(
            "\nUltraGrid compiled with " TBOLD("CUDA") " support: %s\n",
            cuda_supported);
}

#define ASSIGN_CHECK_VAL(var, str, minval) \
        do { \
                const double val = unit_evaluate_dbl(str, false, nullptr); \
                if (std::isnan(val) || val < (minval) || val > UINT_MAX) { \
                        LOG(LOG_LEVEL_ERROR) \
                            << "[J2K] Wrong value " << (str) \
                            << " for " #var "! Value must be >= " << (minval) \
                            << ".\n"; \
                        return NULL; \
                } \
                (var) = val; \
        } while (0)

static struct module * j2k_compress_init(struct module *parent, const char *c_cfg)
{
        const auto *version = cmpto_j2k_enc_get_version();
        LOG(LOG_LEVEL_INFO) << MOD_NAME << "Using codec version: " << (version == nullptr ? "(unknown)" : version->name) << "\n";

        auto *s = new state_video_compress_j2k();

        char *tmp = (char *) alloca(strlen(c_cfg) + 1);
        strcpy(tmp, c_cfg);
        char *save_ptr, *item;
        while ((item = strtok_r(tmp, ":", &save_ptr))) {
                tmp = NULL;
                if (IS_KEY_PREFIX(item, "rate")) {
                        ASSIGN_CHECK_VAL(s->rate, strchr(item, '=') + 1, 1);
                } else if (IS_KEY_PREFIX(item, "quality")) {
                        s->quality = stod(strchr(item, '=') + 1);
                } else if (strcasecmp("mct", item) == 0 || strcasecmp("nomct", item) == 0) {
                        s->mct = strcasecmp("mct", item) == 0 ? 1 : 0;
                } else if (IS_KEY_PREFIX(item, "mem_limit")) {
                        ASSIGN_CHECK_VAL(s->mem_limit, strchr(item, '=') + 1, 1);
                } else if (IS_KEY_PREFIX(item, "tile_limit")) {
                        ASSIGN_CHECK_VAL(s->tile_limit, strchr(item, '=') + 1, 0);
                } else if (IS_KEY_PREFIX(item, "pool_size")) {
                        ASSIGN_CHECK_VAL(s->max_in_frames, strchr(item, '=') + 1, 1);
                } else if (strcasecmp("help", item) == 0) {
                        usage();
                        return static_cast<module*>(INIT_NOERR);
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[J2K] Wrong option: %s\n", item);
                        goto error;
                }
        }

        if (s->quality < 0.0 || s->quality > 1.0) {
                LOG(LOG_LEVEL_ERROR) << "[J2K] Quality should be in interval [0-1]!\n";
                goto error;
        }

        module_init_default(&s->module_data);
        s->module_data.cls = MODULE_CLASS_DATA;
        s->module_data.priv_data = s;
        s->module_data.deleter = j2k_compress_done;
        module_register(&s->module_data, parent);

        return &s->module_data;

error:
        delete s;
        return NULL;
}

static void j2k_compressed_frame_dispose(struct video_frame *frame)
{
        free(frame->tiles[0].data);
        vf_free(frame);
}

static void release_cstream(void * custom_data, size_t custom_data_size, const void * codestream, size_t codestream_size)
{
        (void) codestream; (void) custom_data_size; (void) codestream_size;
        auto *udata = static_cast<struct custom_data *>(custom_data);
        udata->frame.~shared_ptr<video_frame>();
}

static void
release_cstream_cuda(void *img_custom_data, size_t img_custom_data_size,
                      int /* device_id */, const void *samples, size_t samples_size)
{
        release_cstream(img_custom_data, img_custom_data_size, samples,
                        samples_size);
}

#define HANDLE_ERROR_COMPRESS_PUSH \
        if (udata != nullptr) { \
                udata->frame.~shared_ptr<video_frame>(); \
        } \
        if (img != nullptr) { \
                cmpto_j2k_enc_img_destroy(img); \
        } \
        return

static void j2k_compress_push(struct module *state, std::shared_ptr<video_frame> tx)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) state;
        struct cmpto_j2k_enc_img *img = NULL;
        struct custom_data *udata = nullptr;

        if (tx == NULL) { // pass poison pill through encoder
                unique_lock<mutex> lk(s->lock);
                s->should_exit = true;
                if (s->configured) {
                        CHECK_OK(cmpto_j2k_enc_ctx_stop(s->context), "stop",
                                 NOOP);
                } else {
                        s->configure_cv.notify_one();
                }
                return;
        }

        const struct video_desc desc = video_desc_from_frame(tx.get());
        if (!video_desc_eq(s->saved_desc, desc)) {
                int ret = configure_with(s, desc);
                if (!ret) {
                        return;
                }
                struct video_desc pool_desc = desc;
                if (s->precompress_codec != VC_NONE &&
                    !s->pool_in_device_memory) {
                        pool_desc.color_spec = s->precompress_codec;
                }
                s->pool.reconfigure(
                    pool_desc, (size_t) vc_get_linesize(pool_desc.width,
                                                        pool_desc.color_spec) *
                                   pool_desc.height);
        }

        assert(tx->tile_count == 1); // TODO

        CHECK_OK(cmpto_j2k_enc_img_create(s->context, &img),
                        "Image create", return);

        /*
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

        if (s->pool_in_device_memory) {
                // cmpto_j2k_enc requires the size after postprocess, which
                // doesn't equeal the IN frame data_len for R12L
                const codec_t device_codec = s->precompress_codec == VC_NONE
                                           ? udata->frame->color_spec
                                           : s->precompress_codec;
                const size_t  data_len =
                    vc_get_datalen(udata->frame->tiles[0].width,
                                   udata->frame->tiles[0].height, device_codec);
                CHECK_OK(cmpto_j2k_enc_img_set_samples_cuda(
                             img, cuda_devices[0], udata->frame->tiles[0].data,
                             data_len, release_cstream_cuda),
                         "Setting image samples", HANDLE_ERROR_COMPRESS_PUSH);
        } else {
                CHECK_OK(cmpto_j2k_enc_img_set_samples(
                             img, udata->frame->tiles[0].data,
                             udata->frame->tiles[0].data_len, release_cstream),
                         "Setting image samples", HANDLE_ERROR_COMPRESS_PUSH);
        }

        unique_lock<mutex> lk(s->lock);
        s->frame_popped.wait(lk, [s]{return s->in_frames < s->max_in_frames;});
        lk.unlock();
        bool failed = false;
        CHECK_OK(cmpto_j2k_enc_img_encode(img, s->enc_settings),
                        "Encode image push", failed = true);
        if (failed) {
                udata->frame.~shared_ptr<video_frame>();
                cmpto_j2k_enc_img_destroy(img);
                return;
        }
        lk.lock();
        s->in_frames++;
        lk.unlock();

}

static void j2k_compress_done(struct module *mod)
{
        struct state_video_compress_j2k *s =
                (struct state_video_compress_j2k *) mod->priv_data;
        cleanup_common(s);
        delete s;
}

static void
cleanup_common(struct state_video_compress_j2k *s)
{

        if (s->enc_settings != nullptr) {
                cmpto_j2k_enc_cfg_destroy(s->enc_settings);
        }
        s->enc_settings = nullptr;
        if (s->context != nullptr) {
                cmpto_j2k_enc_ctx_destroy(s->context);
        }
        s->context = nullptr;
}

static compress_module_info get_cmpto_j2k_module_info(){
        compress_module_info module_info;
        module_info.name = "cmpto_j2k";

        for(const auto& opt : usage_opts){
                module_info.opts.emplace_back(module_option{opt.label,
                                opt.description, opt.placeholder,  opt.key, opt.opt_str, opt.is_boolean});
        }

        codec codec_info;
        codec_info.name = "Comprimato jpeg2000";
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

