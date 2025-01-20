/**
 * @file   video_compress/gpujpeg.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2024 CESNET
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

#include <algorithm>
#include <cassert>
#include <condition_variable>
#include <initializer_list>
#include <libgpujpeg/gpujpeg_common.h>
#include <libgpujpeg/gpujpeg_encoder.h>
#include <libgpujpeg/gpujpeg_version.h>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <vector>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "module.h"
#include "tv.h"
#include "utils/color_out.h"
#include "utils/synchronized_queue.h"
#include "utils/video_frame_pool.h"
#include "video.h"
#include "video_compress.h"

#ifndef GPUJPEG_VERSION_INT
#error "Old GPUJPEG API detected!"
#endif
#if GPUJPEG_VERSION_INT >= GPUJPEG_MK_VERSION_INT(0, 25, 0)
#define NEW_PARAM_IMG_NO_COMP_COUNT
#else
typedef int gpujpeg_sampling_factor_t;
#endif

#define MOD_NAME "[GPUJPEG enc.] "

using std::condition_variable;
using std::map;
using std::max;
using std::mutex;
using std::set;
using std::shared_ptr;
using std::thread;
using std::unique_lock;
using std::unique_ptr;
using std::vector;

namespace {
struct state_video_compress_gpujpeg;

/**
 * @brief state for single instance of encoder running on one GPU
 */
struct encoder_state {
private:
        void cleanup_state();
        shared_ptr<video_frame> compress_step(shared_ptr<video_frame> frame);
        bool configure_with(struct video_desc desc);

        struct state_video_compress_gpujpeg        *m_parent_state;
        int                                      m_device_id;
        struct gpujpeg_encoder                  *m_encoder;
        struct video_desc                        m_saved_desc;
        video_frame_pool                         m_pool;
        decoder_t                                m_decoder;
        codec_t                                  m_enc_input_codec{};
        unique_ptr<char []>                      m_decoded;

        struct gpujpeg_parameters                m_encoder_param{};
        struct gpujpeg_image_parameters          m_param_image{};
public:
        encoder_state(struct state_video_compress_gpujpeg *s, int device_id) :
                m_parent_state(s), m_device_id(device_id), m_encoder{}, m_saved_desc{},
                m_decoder{}, m_occupied{}
        {
        }
        ~encoder_state() {
                cleanup_state();
        }
        void worker();
        void compress(shared_ptr<video_frame> frame);

        synchronized_queue<shared_ptr<struct video_frame>, 1> m_in_queue; ///< queue for uncompressed frames
        thread                                   m_thread_id;
        bool                                     m_occupied; ///< protected by state_video_compress_gpujpeg::m_occupancy_lock
};

struct state_video_compress_gpujpeg {
private:
        state_video_compress_gpujpeg(struct module *parent, const char *opts);

        vector<struct encoder_state *> m_workers;
        bool                           m_uses_worker_threads; ///< true if cuda_devices_count > 1

        map<uint32_t, shared_ptr<struct video_frame>> m_out_frames; ///< frames decoded out of order
        uint32_t m_in_seq;  ///< seq of next frame to be encoded
        uint32_t m_out_seq; ///< seq of next frame to be decoded

        size_t m_ended_count; ///< number of workers ended

public:
        ~state_video_compress_gpujpeg() {
                if (m_uses_worker_threads) {
                        for (auto worker : m_workers) {
                                worker->m_thread_id.join();
                        }
                }

                for (auto worker : m_workers) {
                        delete worker;
                }
        }
        static state_video_compress_gpujpeg *create(struct module *parent, const char *opts);
        bool parse_fmt(char *fmt);
        void push(std::shared_ptr<video_frame> in_frame);
        std::shared_ptr<video_frame> pop();

        struct module           m_module_data;
        int                     m_restart_interval;
        int                     m_quality;
        bool                    m_force_interleaved = false;
        bool                    m_compress_alpha = false;
        gpujpeg_sampling_factor_t m_subsampling = 0; // -> autoselect
        enum gpujpeg_color_space m_use_internal_codec = GPUJPEG_NONE; // requested internal codec

        synchronized_queue<shared_ptr<struct video_frame>, 1> m_out_queue; ///< queue for compressed frames
        mutex                                                 m_occupancy_lock;
        condition_variable                                    m_worker_finished;
};

/**
 * @brief Compresses single frame
 *
 * This function is called either from within gpujpeg_compress_push() if only one
 * CUDA device is used to avoid context switches that introduce some overhead
 * (measured ~4% performance drop).
 *
 * When there are multiple CUDA devices to be used, it is called from encoder_state::worker().
 */
void encoder_state::compress(shared_ptr<video_frame> frame)
{
        if (frame) {
                char vf_metadata[VF_METADATA_SIZE];
                vf_store_metadata(frame.get(), vf_metadata); // seq and compress_start
                auto out = compress_step(std::move(frame));
                if (out) {
                        vf_restore_metadata(out.get(), vf_metadata);
                        out->compress_end = get_time_in_ns();
                } else {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Failed to encode frame!\n");
                        out = shared_ptr<video_frame>(vf_alloc(1), vf_free);
                        vf_restore_metadata(out.get(), vf_metadata);
                }
                m_parent_state->m_out_queue.push(out);
        } else { // pass poison pill
                m_parent_state->m_out_queue.push({});
        }
}

/**
 * Worker thread that is used if multiple CUDA devices are used - every device
 * has its own thread.
 */
void encoder_state::worker() {
        while (true) {
                auto frame = m_in_queue.pop();

                if (!frame) { // poison pill - pass and exit
                        m_parent_state->m_out_queue.push(frame);
                        break;
                }

                compress(std::move(frame));

                unique_lock<mutex> lk(m_parent_state->m_occupancy_lock);
                m_occupied = false;
                lk.unlock();
                m_parent_state->m_worker_finished.notify_one();
        }
}

static decoder_t get_decoder(codec_t in_codec, codec_t *out_codec)
{
        codec_t candidate_codecs[] = { UYVY, RGB,
#if GJ_RGBA_SUPP == 1
                RGBA,
#endif
        };

        return get_best_decoder_from(in_codec, candidate_codecs, out_codec);
}

gpujpeg_sampling_factor_t
subsampling_to_gj(int ug_subs)
{
        switch (ug_subs) {
        case 444:
                return GPUJPEG_SUBSAMPLING_444;
        case 422:
                return GPUJPEG_SUBSAMPLING_422;
        case 420:
                return GPUJPEG_SUBSAMPLING_420;
        default:
                abort();
        }
}

/**
 * Configures GPUJPEG encoder with provided parameters.
 */
bool encoder_state::configure_with(struct video_desc desc)
{
        struct video_desc compressed_desc;
        compressed_desc = desc;
        compressed_desc.color_spec = JPEG;

        if (desc.color_spec == I420) {
#if GPUJPEG_VERSION_INT < GPUJPEG_MK_VERSION_INT(0, 14, 0)
                if ((m_parent_state->m_use_internal_codec != GPUJPEG_NONE && m_parent_state->m_use_internal_codec != GPUJPEG_YCBCR_BT709) ||
                                (m_parent_state->m_subsampling != 0 && m_parent_state->m_subsampling != 420)) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Converting from planar pixel formats is "
                                        "possible only without subsampling/color space change.\n");
                        return false;
                }
#endif
                m_decoder = nullptr;
                m_enc_input_codec = desc.color_spec;
        } else {
                m_decoder = get_decoder(desc.color_spec, &m_enc_input_codec);
                if (!m_decoder) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unsupported codec: %s\n",
                                        get_codec_name(desc.color_spec));
                        return false;
                }
        }

        if (get_bits_per_component(desc.color_spec) > 8) {
                LOG(LOG_LEVEL_NOTICE) << MOD_NAME << "Converting from " << get_bits_per_component(desc.color_spec) <<
                        " to 8 bits. You may directly capture 8-bit signal to improve performance.\n";
        }

        gpujpeg_set_default_parameters(&m_encoder_param);
        if (m_parent_state->m_quality != -1) {
                m_encoder_param.quality = m_parent_state->m_quality;
        } else {
                log_msg(LOG_LEVEL_INFO, MOD_NAME "setting default encode parameters (quality: %d)\n",
                                m_encoder_param.quality);
        }

#if GPUJPEG_VERSION_INT >= GPUJPEG_MK_VERSION_INT(0, 26, 0)
        m_encoder_param.verbose =
            log_level >= LOG_LEVEL_DEBUG ? GPUJPEG_LL_VERBOSE : GPUJPEG_LL_INFO;
#else
	m_encoder_param.verbose = max<int>(0, log_level - LOG_LEVEL_INFO);
#endif
	m_encoder_param.segment_info = 1;
        gpujpeg_sampling_factor_t subsampling = m_parent_state->m_subsampling;
#if !defined NEW_PARAM_IMG_NO_COMP_COUNT
        if (subsampling == 0) {
                subsampling =
                    subsampling_to_gj(get_subsampling(m_enc_input_codec) / 10);
        }
#endif
        gpujpeg_parameters_chroma_subsampling(&m_encoder_param, subsampling);
        m_encoder_param.interleaved = (codec_is_a_rgb(m_enc_input_codec) && !m_parent_state->m_force_interleaved) ? 0 : 1;
        m_encoder_param.color_space_internal = IF_NOT_NULL_ELSE(m_parent_state->m_use_internal_codec, codec_is_a_rgb(m_enc_input_codec)
                        ? GPUJPEG_RGB : GPUJPEG_YCBCR_BT709);

        gpujpeg_image_set_default_parameters(&m_param_image);

        m_param_image.width = desc.width;
        m_param_image.height = desc.height;


#if !defined NEW_PARAM_IMG_NO_COMP_COUNT
        m_param_image.comp_count = 3;
#endif
        if (m_parent_state->m_compress_alpha) {
                if (desc.color_spec == RGBA) {
#ifdef NEW_PARAM_IMG_NO_COMP_COUNT
                        gpujpeg_parameters_chroma_subsampling(
                            &m_encoder_param, GPUJPEG_SUBSAMPLING_4444);
#else
                        m_param_image.comp_count = 4;
#endif
                } else {
                        LOG(LOG_LEVEL_WARNING) << MOD_NAME "Requested alpha encode but input codec is unsupported pixel format: "
                                << get_codec_name(desc.color_spec) << "\n";
                }
        }
        m_param_image.color_space = codec_is_a_rgb(m_enc_input_codec) ? GPUJPEG_RGB : GPUJPEG_YCBCR_BT709;

        switch (m_enc_input_codec) {
        case I420: m_param_image.pixel_format = GPUJPEG_420_U8_P0P1P2; break;
        case RGB: m_param_image.pixel_format = GPUJPEG_444_U8_P012; break;
#ifdef NEW_PARAM_IMG_NO_COMP_COUNT
        case RGBA: m_param_image.pixel_format = GPUJPEG_4444_U8_P0123; break;
#else
        case RGBA: m_param_image.pixel_format = GPUJPEG_444_U8_P012Z; break;
#endif
        case UYVY: m_param_image.pixel_format = GPUJPEG_422_U8_P1020; break;
        default:
                log_msg(LOG_LEVEL_FATAL, MOD_NAME "Unexpected codec: %s\n",
                                get_codec_name(m_enc_input_codec));
                abort();
        }
        m_encoder_param.restart_interval = IF_NOT_UNDEF_ELSE(m_parent_state->m_restart_interval,
#if GPUJPEG_VERSION_INT >= GPUJPEG_MK_VERSION_INT(0, 25, 3)
                RESTART_AUTO);
#elif GPUJPEG_VERSION_INT >= GPUJPEG_MK_VERSION_INT(0, 20, 4)
                gpujpeg_encoder_suggest_restart_interval(&m_param_image, subsampling, m_encoder_param.interleaved, m_encoder_param.verbose));
#else
                codec_is_a_rgb(m_enc_input_codec) ? 8 : 4);
#endif
        m_encoder = gpujpeg_encoder_create(NULL);

        int data_len = desc.width * desc.height * 3;
        m_pool.reconfigure(compressed_desc, data_len);

        if(!m_encoder) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create GPUJPEG encoder.\n");
                exit_uv(EXIT_FAILURE);
                return false;
        }

        m_decoded = unique_ptr<char []>(new char[4 * desc.width * desc.height]);

        m_saved_desc = desc;

        return true;
}

bool state_video_compress_gpujpeg::parse_fmt(char *fmt)
{
        if (!fmt || fmt[0] == '\0') {
                return true;
        }
        char *tok, *save_ptr = NULL;
        int pos = 0;
        while ((tok = strtok_r(fmt, ":", &save_ptr)) != nullptr) {
                if (isdigit(tok[0]) && pos == 0) {
                        m_quality = atoi(tok);
                        if (m_quality <= 0 || m_quality > 100) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error: Quality should be in interval [1-100]!\n");
                                return false;
                        }
                } else if (isdigit(tok[0]) && pos == 1) {
                        m_restart_interval = atoi(tok);
                        if (m_restart_interval < 0) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error: Restart interval should be non-negative!\n");
                                return false;
                        }
                } else {
                        if (strstr(tok, "q=") == tok) {
                                m_quality = atoi(tok + strlen("q="));
                        } else if (strstr(tok, "restart=") == tok) {
                                m_quality = atoi(tok + strlen("restart="));
                        } else if (strcasecmp(tok, "interleaved") == 0) {
                                m_force_interleaved = true;
                        } else if (strcasecmp(tok, "Y601") == 0) {
                                m_use_internal_codec = GPUJPEG_YCBCR_BT601;
                        } else if (strcasecmp(tok, "Y601full") == 0) {
                                m_use_internal_codec = GPUJPEG_YCBCR_BT601_256LVLS;
                        } else if (strcasecmp(tok, "Y709") == 0) {
                                m_use_internal_codec = GPUJPEG_YCBCR_BT709;
                        } else if (strcasecmp(tok, "RGB") == 0) {
                                m_use_internal_codec = GPUJPEG_RGB;
                        } else if (strstr(tok, "subsampling=") == tok) {
                                m_subsampling = subsampling_to_gj(
                                    atoi(strchr(tok, '=') + 1));
                        } else if (strcmp(tok, "alpha") == 0) {
#if GPUJPEG_VERSION_INT < GPUJPEG_MK_VERSION_INT(0, 20, 2)
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "GPUJPEG v0.20.2 is required for alpha support, %s found.\n",
                                                gpujpeg_version_to_string(gpujpeg_version()));
#endif
                                m_compress_alpha = true;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "WARNING: Trailing configuration parameter: %s\n", tok);
                        }
                }
                fmt = nullptr;
                pos += 1;
        }

        return true;
}

state_video_compress_gpujpeg::state_video_compress_gpujpeg(struct module *parent, const char *opts) :
        m_uses_worker_threads{}, m_in_seq{},
        m_out_seq{}, m_ended_count{},
        m_module_data{}, m_restart_interval(UNDEF), m_quality(-1)
{
        if(opts && opts[0] != '\0') {
                char *fmt = strdup(opts);
                if (!parse_fmt(fmt)) {
                        free(fmt);
                        throw 1;
                }
                free(fmt);
        }

        module_init_default(&m_module_data);
        m_module_data.cls = MODULE_CLASS_DATA;
        m_module_data.priv_data = this;
        static auto deleter = [](struct module *mod) {
                struct state_video_compress_gpujpeg *s = (struct state_video_compress_gpujpeg *) mod->priv_data;
                delete s;
        };
        m_module_data.deleter = deleter;

        module_register(&m_module_data, parent);
}

/**
 * Creates GPUJPEG encoding state and creates GPUJPEG workers for every GPU that
 * will be used for compression (if cuda_devices_count > 1).
 */
state_video_compress_gpujpeg *state_video_compress_gpujpeg::create(struct module *parent, const char *opts) {
        assert(cuda_devices_count > 0);

        auto ret = new state_video_compress_gpujpeg(parent, opts);

        for (unsigned int i = 0; i < cuda_devices_count; ++i) {
                ret->m_workers.push_back(new encoder_state(ret, cuda_devices[i]));
        }

        if (cuda_devices_count > 1) {
                ret->m_uses_worker_threads = true;
        }

        if (ret->m_uses_worker_threads) {
                for (auto worker : ret->m_workers) {
                        worker->m_thread_id = thread(&encoder_state::worker, worker);
                }
        }

        return ret;
}

static const struct {
        const char *label;
        const char *key;
        const char *help_name;
        const char *description;
        const char *opt_str;
        bool is_boolean;
} usage_opts[] = {
        {"Quality", "quality", "quality",
                "\t\tJPEG quality coefficient [0..100] - more is better\n", ":q=", false},
        {"Restart interval", "restart_interval", "restart_interval",
                "\t\tInterval between independently entropy encoded block of MCUs,\n"
                        "\t\t0 to disable. Using large intervals or disable (0) slightly\n"
                        "\t\treduces bandwidth at the expense of worse parallelization (if\n"
                        "\t\treset intervals disabled, Huffman encoding is run on CPU). Leave\n"
                        "\t\tuntouched if unsure.\n",
                ":restart=", false},
        {"Interleaved", "interleaved", "interleaved",
                "\t\tForce interleaved encoding (default for YCbCr input formats).\n"
                        "\t\tNon-interleaved has slightly better performance for RGB at the\n"
                        "\t\texpense of worse compatibility. Therefore this option may be\n"
                        "\t\tenabled safely.\n",
                ":interleaved", true},
        {"Color space", "color_space", "RGB|Y601|Y601full|Y709",
                "\t\tforce internal JPEG color space (otherwise source color space is kept).\n",
                ":", false},
        {"Subsampling", "subsampling", "sub",
                "\t\tUse specified JPEG subsampling (444, 422 or 420).\n",
                ":sub=", false},
        {"Alpha", "alpha", "alpha",
                "\t\tCompress (keep) alpha channel of RGBA.\n",
                ":alpha", true},
};

struct module * gpujpeg_compress_init(struct module *parent, const char *opts)
{
        if (gpujpeg_version() >> 8 != GPUJPEG_VERSION_INT >> 8) {
                LOG(LOG_LEVEL_WARNING) << "GPUJPEG API version mismatch! (compiled: " <<
                                gpujpeg_version_to_string(GPUJPEG_VERSION_INT) << ", library present: " <<
                                gpujpeg_version_to_string(gpujpeg_version()) << ", required same minor version)\n";
        }
        struct state_video_compress_gpujpeg *s;

        if(opts && strcmp(opts, "help") == 0) {
                col() << "GPUJPEG comperssion usage:\n";
                col() << "\t" << TBOLD(TRED("-c GPUJPEG") << "[:<quality>[:<restart_interval>]][:interleaved][:RGB|Y601|Y601full|Y709]][:subsampling=<sub>][:alpha]\n");
                col() << "where\n";

                for(const auto& i : usage_opts){
                    col() << "\t" << TBOLD(<< i.help_name <<) << "\n" << i.description;
                }

                col() << "\n";
                col() << TBOLD("Note:") << " instead of positional parameters for "
                        "quality and restart intervals " << TBOLD("\"q=\"") << " and " << TBOLD("\"restart=\"") << " can be used.\n";
                col() << "\n";
                return static_cast<module*>(INIT_NOERR);
        }
        if (opts && strcmp(opts, "check") == 0) {
                auto device_info = gpujpeg_get_devices_info();
                return device_info.device_count == 0 ? nullptr : static_cast<module*>(INIT_NOERR);
        }
        if (opts && strcmp(opts, "list_devices") == 0) {
                printf("CUDA devices:\n");
#if GPUJPEG_VERSION_INT >= GPUJPEG_MK_VERSION_INT(0, 16, 0)
                return gpujpeg_print_devices_info() == 0 ? static_cast<module*>(INIT_NOERR) : nullptr;
#else
                gpujpeg_print_devices_info();
                return static_cast<module*>(INIT_NOERR);
#endif
        }

        try {
                s = state_video_compress_gpujpeg::create(parent, opts);
        } catch (...) {
                return NULL;
        }

        return &s->m_module_data;
}

/**
 * Performs actual compression with GPUJPEG. Reconfigures encoder if needed.
 * @return compressed frame, {} if failed
 */
shared_ptr<video_frame> encoder_state::compress_step(shared_ptr<video_frame> tx)
{
        gpujpeg_set_device(m_device_id);

        // first run - initialize device
        if (!m_encoder) {
                log_msg(LOG_LEVEL_INFO, "Initializing CUDA device %d...\n", m_device_id);
                const int ret =
                    gpujpeg_init_device(m_device_id, GPUJPEG_VERBOSE);
                if(ret != 0) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "initializing CUDA device %d failed.\n", m_device_id);
                        exit_uv(EXIT_FAILURE);
                        return {};
                }
        }

        struct video_desc desc = video_desc_from_frame(tx.get());

        // if format has changed, reconfigure
        if(!video_desc_eq_excl_param(m_saved_desc, desc, PARAM_INTERLACING)) {
                cleanup_state();
                int ret = configure_with(desc);
                if(!ret) {
                        exit_uv(EXIT_FAILURE);
                        return NULL;
                }
        }

        shared_ptr<video_frame> out = m_pool.get_frame();

        for (unsigned int x = 0; x < out->tile_count;  ++x) {
                struct tile *in_tile = vf_get_tile(tx.get(), x);
                struct tile *out_tile = vf_get_tile(out.get(), x);
                uint8_t *jpeg_enc_input_data;

                if (m_decoder && m_decoder != vc_memcpy) {
                        assert(tx.get()->mem_location == CPU_MEM);
                        unsigned char *line1 = (unsigned char *) in_tile->data;
                        unsigned char *line2 = (unsigned char *) m_decoded.get();

                        for (int i = 0; i < (int) in_tile->height; ++i) {
                                m_decoder(line2, line1,
                                                vc_get_linesize(desc.width,
                                                        m_enc_input_codec),
                                                0, 8, 16);
                                line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                                line2 += vc_get_linesize(desc.width, m_enc_input_codec);
                        }
                        jpeg_enc_input_data = (uint8_t *) m_decoded.get();
                } else {
                        jpeg_enc_input_data = (uint8_t *) in_tile->data;
                }

                uint8_t *compressed;
#if GPUJPEG_VERSION_INT < GPUJPEG_MK_VERSION_INT(0, 21, 0)
                int size;
#else
                size_t size = 0;
#endif

                struct gpujpeg_encoder_input encoder_input;
                if(tx.get()->mem_location == CUDA_MEM){
                        gpujpeg_encoder_input_set_gpu_image(&encoder_input, jpeg_enc_input_data);
                } else {
                        gpujpeg_encoder_input_set_image(&encoder_input, jpeg_enc_input_data);
                }

                const int ret = gpujpeg_encoder_encode(m_encoder, &m_encoder_param, &m_param_image, &encoder_input, &compressed, &size);
                if(ret != 0) {
                        return {};
                }

                out_tile->data_len = size;
                memcpy(out_tile->data, compressed, size);
        }

        return out;
}

void encoder_state::cleanup_state()
{
        if (m_encoder)
                gpujpeg_encoder_destroy(m_encoder);
        m_encoder = NULL;
}

void state_video_compress_gpujpeg::push(std::shared_ptr<video_frame> in_frame)
{

        if (in_frame) {
                in_frame->seq = m_in_seq++;
        }

        if (!m_uses_worker_threads) {
                m_workers[0]->compress(std::move(in_frame));
                return;
        }
        if (!in_frame) { // pass poison pill to all workers
                for (auto *worker : m_workers) {
                        worker->m_in_queue.push({});
                }
                return;
        }

        int                index = 0;
        unique_lock<mutex> lk(m_occupancy_lock);
        // wait for/select not occupied worker
        m_worker_finished.wait(lk, [this, &index] {
                for (auto *worker : m_workers) {
                        if (!worker->m_occupied) {
                                return true;
                        }
                        index++;
                }
                return false;
        });
        m_workers[index]->m_occupied = true;
        lk.unlock();
        m_workers[index]->m_in_queue.push(in_frame);
}

/**
 * @brief returns compressed frame
 *
 * This function takes frames from state_video_compress_gpujpeg::m_out_queue. It checks
 * sequential number of frame from queue - if it is in the same order that
 * was sent to encoder, it is returned (according to state_video_compress_gpujpeg::m_out_seq).
 * If not, it is stored in state_video_compress_gpujpeg::m_out_frames and this function
 * further waits for frame with appropriate seq. Frames that was not successfully encoded
 * have data_len member set to 0 and are skipped here.
 */
std::shared_ptr<video_frame> state_video_compress_gpujpeg::pop()
{
start:
        if (m_out_frames.find(m_out_seq) != m_out_frames.end()) {
                auto frame = m_out_frames[m_out_seq];
                m_out_frames.erase(m_out_seq);
                m_out_seq += 1;
                if (frame->tiles[0].data_len == 0) { // was error processing that frame, skip
                        goto start;
                } else {
                        return frame;
                }
        } else {
                while (true) {
                        auto frame = m_out_queue.pop();
                        if (!frame) {
                                if (++m_ended_count == m_workers.size()) {
                                        return {};
                                } else {
                                        continue;
                                }
                        }
                        if (frame->seq == m_out_seq) {
                                m_out_seq += 1;
                                if (frame->tiles[0].data_len == 0) { // error - skip this frame
                                        goto start;
                                } else {
                                        return frame;
                                }
                        } else {
                                m_out_frames[frame->seq] = frame;
                        }
                }
        }
}

static compress_module_info get_gpujpeg_module_info(){
        compress_module_info module_info;
        module_info.name = "gpujpeg";

        for(const auto& opt : usage_opts){
                std::string desc = opt.description;
                desc.erase(std::remove(desc.begin(), desc.end(), '\t'), desc.end());
                std::replace(desc.begin(), desc.end(), '\n', ' ');
                module_info.opts.emplace_back(
                    module_option{ opt.label, std::move(desc), opt.key,
                                   opt.opt_str, opt.is_boolean });
        }

        codec codec_info;
        codec_info.name = "Jpeg";
        codec_info.priority = 300;
        codec_info.encoders.emplace_back(encoder{"default", ""});

        module_info.codecs.emplace_back(std::move(codec_info));

        return module_info;
}

static auto gpujpeg_compress_push(struct module *mod, std::shared_ptr<video_frame> in_frame) {
        static_cast<struct state_video_compress_gpujpeg *>(mod->priv_data)->push(std::move(in_frame));
}

static auto gpujpeg_compress_pull (struct module *mod) {
        return static_cast<struct state_video_compress_gpujpeg *>(mod->priv_data)->pop();
}

const struct video_compress_info gpujpeg_info = {
        "GPUJPEG",
        gpujpeg_compress_init,
        NULL,
        NULL,
        gpujpeg_compress_push,
        gpujpeg_compress_pull,
        NULL,
        NULL,
        get_gpujpeg_module_info
};

static auto gpujpeg_compress_init_deprecated(struct module *parent, const char *opts) {
        log_msg(LOG_LEVEL_WARNING, "Name \"-c JPEG\" deprecated, use \"-c GPUJPEG\" instead.\n");
        return gpujpeg_compress_init(parent, opts);
}

const struct video_compress_info deprecated_jpeg_info = {
        "JPEG",
        gpujpeg_compress_init_deprecated,
        NULL,
        NULL,
        gpujpeg_compress_push,
        gpujpeg_compress_pull,
        NULL,
        NULL,
        NULL
};


REGISTER_MODULE(gpujpeg, &gpujpeg_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);
REGISTER_HIDDEN_MODULE(jpeg, &deprecated_jpeg_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace

