/**
 * @file   video_compress/jpegxs.cpp
 * @author Jan Frejlach     <536577@mail.muni.cz>
 */
/*
 * Copyright (c) 2026 CESNET
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

#include <cassert>                                 // for assert
#include <cinttypes>                               // for PRIu32
#include <climits>                                 // for LLONG_MIN
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <svt-jpegxs/SvtJpegxs.h>                  // for SvtJxsErrorType
#include <svt-jpegxs/SvtJpegxsEnc.h>
#include <svt-jpegxs/SvtJpegxsImageBufferTools.h>

#include "debug.h"
#include "lib_common.h"
#include "types.h"                                 // for video_desc, tile
#include "utils/misc.h"                            // for get_cpu_core_count
#include "video.h"
#include "video_compress.h"
#include "utils/video_frame_pool.h"
#include "utils/synchronized_queue.h"
#include "jpegxs/jpegxs_conv.h"
#include "video_frame.h"                           // for video_desc_from_frame

#define DEFAULT_POOL_SIZE 5
#define MOD_NAME "[JPEG XS enc.] "

using std::shared_ptr;
using std::condition_variable;
using std::mutex;
using std::thread;
using std::unique_lock;
using std::atomic;

namespace {
struct state_video_compress_jpegxs;

int jxs_poison_pill_obj = 0;
#define JXS_POISON_PILL ((void *) &jxs_poison_pill_obj)

struct state_video_compress_jpegxs {
        state_video_compress_jpegxs(struct module *parent, const char *opts);

        ~state_video_compress_jpegxs() {
                if (worker_send.joinable()) {
                        worker_send.join();
                }
                cleanup();
        }

        svt_jpeg_xs_encoder_api_t encoder{};
        svt_jpeg_xs_image_config_t image_config{};
        svt_jpeg_xs_frame_pool_t *frame_pool{};
        int pool_size = DEFAULT_POOL_SIZE;

        long long req_bitrate = -1;

        bool configured = 0;
        bool reconfiguring = 0;
        bool stop = false;

        video_desc saved_desc;

        video_frame_pool pool;

        void (*convert_to_planar)(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer *dst) = nullptr;
        
        void cleanup();
        bool parse_fmt(char *fmt);

        synchronized_queue<shared_ptr<struct video_frame>, DEFAULT_POOL_SIZE> in_queue;

        thread worker_send;

        mutex mtx;
        condition_variable cv_configured;
        condition_variable cv_drained;
        condition_variable cv_reconfiguring;

        atomic<uint64_t> frames_sent{0};
        atomic<uint64_t> frames_received{0};
};

static bool configure_with(struct state_video_compress_jpegxs *s, struct video_desc desc);
static void jpegxs_worker_send(state_video_compress_jpegxs *s);

state_video_compress_jpegxs::state_video_compress_jpegxs(struct module *parent, const char *opts) {
        (void) parent;

        SvtJxsErrorType_t err = svt_jpeg_xs_encoder_load_default_parameters(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &encoder);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to load JPEG XS default parameters, error code: %x\n", err);
                throw 1;
        }

        encoder.bpp_numerator = 3;
        encoder.verbose = VERBOSE_NONE;
        encoder.threads_num = get_cpu_core_count();

        if(opts && opts[0] != '\0') {
                char *fmt = strdup(opts);
                if (!parse_fmt(fmt)) {
                        free(fmt);
                        throw 1;
                }
                free(fmt);
        }

        worker_send = thread(jpegxs_worker_send, this);
}

static void jpegxs_worker_send(state_video_compress_jpegxs *s) {
while (true) {
        auto frame = s->in_queue.pop();

        if (!frame) {
                if (s->configured) {
                        svt_jpeg_xs_frame_t enc_input;
                        svt_jpeg_xs_frame_pool_get(s->frame_pool, &enc_input, /*blocking*/ 1);
                        enc_input.user_prv_ctx_ptr = JXS_POISON_PILL;
                        svt_jpeg_xs_encoder_send_picture(&s->encoder, &enc_input, /*blocking*/ 1);
                        s->frames_sent++;
                } else {
                        unique_lock<mutex> lock(s->mtx);
                        s->stop = true;
                        lock.unlock();
                        s->cv_configured.notify_one();
                }
                break;
        }

        if (!s->configured) {
                struct video_desc desc = video_desc_from_frame(frame.get());
                if (!configure_with(s, desc)) {
                        break;
                }
        }

        if(!video_desc_eq_excl_param(video_desc_from_frame(frame.get()), s->saved_desc, PARAM_INTERLACING)){
        {
                unique_lock<mutex> lock(s->mtx);
                s->reconfiguring = true;
                s->cv_drained.wait(lock, [&]{
                        return s->frames_received == s->frames_sent;
                });
        }
                s->cleanup();
                if (!configure_with(s, video_desc_from_frame(frame.get()))) {
                        break;
                }
        {
                unique_lock<mutex> lock(s->mtx);
                s->reconfiguring = false;
                s->frames_sent = 0;
                s->frames_received = 0;
                s->cv_reconfiguring.notify_one();
        }
        }

        svt_jpeg_xs_frame_t enc_input;
        SvtJxsErrorType_t err = svt_jpeg_xs_frame_pool_get(s->frame_pool, &enc_input, /*blocking*/ 1);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Failed to get frame from JPEG XS pool, error code: %x\n", err);
                continue;
        }

        enc_input.user_prv_ctx_ptr = malloc(VF_METADATA_SIZE);
        vf_store_metadata(frame.get(), enc_input.user_prv_ctx_ptr);
        
        struct tile *in_tile = vf_get_tile(frame.get(), 0);
        s->convert_to_planar((const uint8_t *) in_tile->data, in_tile->width, in_tile->height, &enc_input.image);

        err = svt_jpeg_xs_encoder_send_picture(&s->encoder, &enc_input, /*blocking*/ 1);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to send frame to encoder, error code: %x\n", err);
                continue;
        }

        s->frames_sent++;
}
}

ColourFormat subsampling_to_jpegxs(int ug_subs) {
        switch (ug_subs) {
        case 444:
                return COLOUR_FORMAT_PLANAR_YUV444_OR_RGB;
        case 422:
                return COLOUR_FORMAT_PLANAR_YUV422;
        case 420:
                return COLOUR_FORMAT_PLANAR_YUV420;
        default:
                abort();
        }
}

static void
set_bitrate(svt_jpeg_xs_encoder_api_t *encoder, long long req_bitrate,
            const struct video_desc *desc)
{
        long long numerator   = req_bitrate;
        long long denominator = desc->width * desc->height * desc->fps;
        // reduce numbers to fit uint32_t if num or den is larger
        while (numerator > UINT32_MAX || denominator > UINT32_MAX) {
                numerator /= 1024;
                denominator /= 1024;
        }
        assert(numerator > 0);
        assert(denominator > 0);
        encoder->bpp_numerator   = numerator;
        encoder->bpp_denominator = denominator;
}

static bool configure_with(struct state_video_compress_jpegxs *s, struct video_desc desc)
{
        const struct uv_to_jpegxs_conversion *conv = get_uv_to_jpegxs_conversion(desc.color_spec);
        if (!conv || !conv->convert) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported codec: %s\n", get_codec_name(desc.color_spec));
                return false;
        }
        s->convert_to_planar = conv->convert;

        s->encoder.source_width = desc.width;
        s->encoder.source_height = desc.height;
        s->encoder.input_bit_depth = get_bits_per_component(desc.color_spec);
        s->encoder.colour_format = subsampling_to_jpegxs(get_subsampling(desc.color_spec) / 10);

        SvtJxsErrorType_t err = SvtJxsErrorNone;
        uint32_t bitstream_size = 0;
        
        err = svt_jpeg_xs_encoder_get_image_config(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &s->encoder, &s->image_config, &bitstream_size);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to get image config from JPEG XS encoder parameters: %x\n", err);
                return false;
        }

        s->frame_pool = svt_jpeg_xs_frame_pool_alloc(&s->image_config, bitstream_size, s->pool_size);
        if (!s->frame_pool) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to allocate JPEG XS frame pool\n");
                return false;
        }

        if (s->req_bitrate != -1) {
                set_bitrate(&s->encoder, s->req_bitrate, &desc);
        }

        err = svt_jpeg_xs_encoder_init(SVT_JPEGXS_API_VER_MAJOR, SVT_JPEGXS_API_VER_MINOR, &s->encoder);
        if (err != SvtJxsErrorNone) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to initialize JPEG XS encoder: %x\n", err);
                return false;
        }

        s->saved_desc = desc;
        struct video_desc compressed_desc;
        compressed_desc = desc;
        compressed_desc.color_spec = JPEG_XS;
        s->pool.reconfigure(compressed_desc, bitstream_size); 
{
        unique_lock<mutex> lock(s->mtx);
        s->configured = true;
}
        s->cv_configured.notify_one();

        return true;
}

void state_video_compress_jpegxs::cleanup() {
        if (frame_pool) {
                svt_jpeg_xs_frame_pool_free(frame_pool);
        }
        if (configured) {
                svt_jpeg_xs_encoder_close(&encoder);
        }
}

bool state_video_compress_jpegxs::parse_fmt(char *fmt) {
        char *tok, *save_ptr = NULL;

        while ((tok = strtok_r(fmt, ":", &save_ptr)) != nullptr) {
                const char *val = strchr(tok, '=');
                int num = -1;
                if (val != nullptr) {
                        val += 1;
                        num = atoi(val);
                }
                if (IS_KEY_PREFIX(tok, "bitrate")) {
                        req_bitrate = unit_evaluate(val, nullptr);
                        if (req_bitrate == LLONG_MIN) {
                                MSG(ERROR, "Invalid value for bitrate: %s\n", val);
                                 return false;
                        }
                } else if (IS_KEY_PREFIX(tok, "bpp")) {
                        int num = 0, den = 1;
                        if (sscanf(val, "%d/%d", &num, &den) < 1 || num <= 0 || den <= 0) {
                                MSG(ERROR, "Invalid bpp value '%s' (must be a positive integer or fraction, e.g., 2 or 3/4).\n", val);
                                return false;
                        }
                        encoder.bpp_numerator   = num;
                        encoder.bpp_denominator = den;
                } else if (IS_KEY_PREFIX(tok, "pool_size")) {
                        if (num <= 0) {
                                MSG(ERROR, "Invalid pool size value '%s' (must be a positive integer).\n", val);
                                return false;
                        }
                        pool_size = num;
                } else if (IS_KEY_PREFIX(tok, "decomp_v")) {
                        if (num <= 0 ||num > 2) {
                                MSG(ERROR, "Invalid decomp_v value '%s' (must be 0, 1 or 2).\n", val);
                                return false;
                        }
                        encoder.ndecomp_v = num;
                } else if (IS_KEY_PREFIX(tok, "decomp_h")) {
                        if (num < 1 || num > 5) {
                                MSG(ERROR, "Invalid decomp_h value '%s' (must be 1, 2, 3, 4 or 5).\n", val);
                                return false;
                        }
                        encoder.ndecomp_h = num;
                } else if (IS_KEY_PREFIX(tok, "quantization")) {
                        if (num != 0 && num != 1) {
                                MSG(ERROR, "Invalid quantization method '%s' (must be 0 - deadzone, or 1 - uniform).\n", val);
                                return false;
                        }
                        encoder.quantization = num;
                } else if (IS_KEY_PREFIX(tok, "slice_height")) {
                        if (num <= 0) {
                                MSG(ERROR, "Invalid slice_height value '%s' (must be positive).\n", val);
                                return false;
                        }
                        encoder.slice_height = num;
                } else if (IS_KEY_PREFIX(tok, "rc")) {
                        if (num < 0 || num > 3) {
                                MSG(ERROR, "Invalid rc mode '%s' (must be 0 - CBR budget per precinct, 1 - CBR budget per precinct with padding movement, 2 - CBR budget per slice, or 3 - CBR budget per slice with max rate size).\n", val);
                                return false;
                        }
                        encoder.rate_control_mode = num;
                } else if (IS_KEY_PREFIX(tok, "threads")) {
                        const int threads = atoi(val);
                        if (threads < 0) {
                                MSG(ERROR, "Invalid number of threads '%s' (must be a positive value or 0 which means lowest possible number of threads is created).\n", tok);
                                return false;
                        }
                        encoder.threads_num = threads;
                } else if (IS_KEY_PREFIX(tok, "verbose")) {
                        if (num < VERBOSE_NONE || num > VERBOSE_INFO_FULL) {
                                MSG(ERROR, "Invalid verbose messages mode '%s' (must be between %d and %d).\n", tok, VERBOSE_NONE, VERBOSE_INFO_FULL);
                                return false;
                        }
                        encoder.verbose = num;
                } else {
                        MSG(ERROR, "Unknown configuration parameter or a missing value: %s\n", tok);
                        return false;
                }
                fmt = nullptr;
        }

        if ((encoder.slice_height & ( (1 << encoder.ndecomp_v) - 1 )) != 0) {
                MSG(ERROR, "Invalid slice_height value '%" PRIu32 "' (must be a multiple of 2^decomp_v).\n", encoder.slice_height);
                return false;
        }

        return true;
}

static const struct {
        const char *label;
        const char *key;
        const char *help_name;
        const char *description;
        const char *opt_str;
        bool is_boolean;
        const char *placeholder;
} usage_opts[] = {
        {"Bit rate", "bitrate", "bitrate", "\t\tbitrate to be used "
                "(eg. 50.5M)\n", ":bitrate=", false, "50.5M"
        },
        {"Bits per pixel", "bpp", "bpp",
                "\t\tTarget bits-per-pixel ratio for the encoder. May be given as an\n"
                "\t\tinteger (e.g., 2) or as a fraction (e.g., 3/4). Controls the\n"
                "\t\toutput bitrate indirectly. Must be a positive value.\n"
                "\t\tThe default is 3.\n",
                ":bpp=", false, "3"
        },
        {"Vertical decomposition", "decomp_v", "decomp_v",
                "\t\tNumber of vertical wavelet decompositions. Allowed values are\n"
                "\t\t0, 1, or 2. The default is 2.\n",
                ":decomp_v=", false, "2"
        },
        {"Horizontal decomposition", "decomp_h", "decomp_h",
                "\t\tNumber of horizontal wavelet decompositions. Allowed values\n"
                "\t\tare between 1 and 5. The default is 5.\n",
                ":decomp_h=", false, "5"
        },
        {"Quantization algorithm", "quantization", "quantization",
                "\t\tSelects the quantization algorithm: 0 = deadzone, 1 = uniform.\n"
                "\t\tThe default is dead-zone quantization.\n",
                ":quantization=", false, "0"
        },
        {"Slice height", "slice_height", "slice_height",
                "\t\tHeight of a slice in lines. Must be a positive integer and a\n"
                "\t\tmultiple of 2^decomp_v. The default is 16.\n",
                ":slice_height=", false, "16"
        },
        {"Rate control mode", "rc", "rc",
                "\t\tRate control mode:\n"
                "\t\t 0 = CBR budget per precinct (default option)\n"
                "\t\t 1 = CBR budget per precinct with padding movement\n"
                "\t\t 2 = CBR budget per slice\n"
                "\t\t 3 = CBR budget per slice with max rate size\n",
                ":rc=", false, "0"
        },
        {"Threads scaling parameter", "threads", "threads",
                "\t\tNumber of encoder threads. Must be between 0 and the number of\n"
                "\t\tavailable CPU cores. Value 0 means the lowest possible number\n"
                "\t\tof threads is created by the encoder. The default is 0.\n",
                ":threads=", false, "0"
        },
        {"JPEG XS pool size", "pool_size", "pool_size",
                "\t\tThe size of the SVT-JPEG-XS frame pool. Increasing the pool size\n"
                "\t\tenables more frames to be sent to the encoder's internal queue.\n"
                "\t\tThe default is 5.\n",
                ":pool_size=", false, "5"
        },
        {"Encoder verbose", "verbose", "verbose",
                "\t\tSets the verbosity level of the SVT-JPEG-XS encoder.\n"
                "\t\t0 = none, 1 = errors, 2 = system info, 3 = extended system info,\n"
                "\t\t4 = warnings, 5 = multithreading info, 6 = full verbose output.\n"
                "\t\tThe default is 0.\n",
                ":verbose=", false, "0"
        },
};

static void *jpegxs_compress_init(struct module *parent, const char *opts) {
        struct state_video_compress_jpegxs *s;
        
        if (opts && strcmp(opts, "help") == 0) {
                color_printf(TBOLD("JPEG XS") " compression usage:\n");
                color_printf("\t" TBOLD(
                        TRED("-c jpegxs") "[:bitrate=<br>|:bpp=<ratio>][:decomp_v=<0-2>][:decomp_h=<1-5>]"
                                          "[:quantization=<0-1>][:slice_height=<n>][:rc=<mode>]"
                                          "[:threads=<num_threads>][:pool_size=<n>][:verbose=<n>]") "\n");
                color_printf("\t" TBOLD(TRED("-c jpegxs") ":help") "\n");

                color_printf("\nwhere:\n");
                for (const auto &opt : usage_opts) {
                        color_printf("\t" TBOLD("<%s>") "\n%s\n",
                                opt.key, opt.description);
                }
                printf("\n");
                return INIT_NOERR;
        }

        s = new state_video_compress_jpegxs(parent, opts);

        return s;
}

static void jpegxs_compress_push(void *state, shared_ptr<video_frame> frame) {
        static_cast<struct state_video_compress_jpegxs *>(state)->in_queue.push(std::move(frame));
}

static shared_ptr<video_frame>
jpegxs_compress_pop(void *state)
{
        auto *s = static_cast<struct state_video_compress_jpegxs *>(state);
        {
                unique_lock<mutex> lock(s->mtx);
                s->cv_configured.wait(lock,
                                      [&] { return s->configured || s->stop; });

                if (s->stop) {
                        return {};
                }
                if (s->reconfiguring && s->frames_received == s->frames_sent) {
                        s->cv_drained.notify_one();
                        s->cv_reconfiguring.wait(
                            lock, [&] { return !s->reconfiguring; });
                }
        }
        svt_jpeg_xs_frame_t enc_output;
        SvtJxsErrorType_t   err = svt_jpeg_xs_encoder_get_packet(
            &s->encoder, &enc_output, /*blocking*/ 1);
        if (err != SvtJxsErrorNone) {
                MSG(ERROR, "Failed to get encoded packet, error code: %x\n",
                    err);
                free(enc_output.user_prv_ctx_ptr);
                svt_jpeg_xs_frame_pool_release(s->frame_pool, &enc_output);
                return vcomp_pop_retry;
        }
        if (enc_output.user_prv_ctx_ptr == JXS_POISON_PILL) {
                svt_jpeg_xs_frame_pool_release(s->frame_pool, &enc_output);
                return {};
        }
        s->frames_received++;

        shared_ptr<video_frame> out_frame = s->pool.get_frame();

        vf_restore_metadata(out_frame.get(), enc_output.user_prv_ctx_ptr);
        free(enc_output.user_prv_ctx_ptr);

        struct tile *out_tile = vf_get_tile(out_frame.get(), 0);
        size_t       enc_size = enc_output.bitstream.used_size;
        if (enc_size > out_tile->data_len) {
                MSG(WARNING, "Encoded frame too big (%zu > %u)\n", enc_size,
                    out_tile->data_len);
                svt_jpeg_xs_frame_pool_release(s->frame_pool, &enc_output);
                return vcomp_pop_retry;
        }

        out_tile->data_len = enc_size;
        memcpy(out_tile->data, enc_output.bitstream.buffer, enc_size);

        svt_jpeg_xs_frame_pool_release(s->frame_pool, &enc_output);
        return out_frame;
}

static void jpegxs_compress_done(void *state) {
        auto s = static_cast<state_video_compress_jpegxs *>(state);
        delete s;
}

static compress_module_info get_jpegxs_module_info() {
        compress_module_info module_info;
        module_info.name = "jpegxs";

        for(const auto& opt : usage_opts){
                module_info.opts.emplace_back(module_option{opt.label,
                                opt.description, opt.placeholder, opt.key, opt.opt_str, false});
        }

        codec codec_info;
        codec_info.name = "JPEG XS";
        codec_info.priority = 400;
        codec_info.encoders.emplace_back(encoder{"default", ""});

        module_info.codecs.emplace_back(std::move(codec_info));

        return module_info;
}

const struct video_compress_info jpegxs_info = {
        jpegxs_compress_init,
        jpegxs_compress_done,
        NULL,
        NULL,
        NULL,
        NULL,
        jpegxs_compress_push,
        jpegxs_compress_pop,
        get_jpegxs_module_info,
};

REGISTER_MODULE(jpegxs, &jpegxs_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
}
