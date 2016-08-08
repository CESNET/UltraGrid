/**
 * @file   src/video_compress/jpeg.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2016 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "compat/platform_time.h"
#include "debug.h"
#include "host.h"
#include "video_compress.h"
#include "module.h"
#include "lib_common.h"
#include "libgpujpeg/gpujpeg_encoder.h"
#include "utils/synchronized_queue.h"
#include "utils/video_frame_pool.h"
#include "video.h"
#include <memory>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

using namespace std;

namespace {
struct state_video_compress_jpeg;

/**
 * @brief state for single instance of encoder running on one GPU
 */
struct encoder_state {
private:
        void cleanup_state();
        shared_ptr<video_frame> compress_step(shared_ptr<video_frame> frame);
        bool configure_with(struct video_desc desc);

        struct state_video_compress_jpeg        *m_parent_state;
        int                                      m_device_id;
        struct gpujpeg_encoder                  *m_encoder;
        struct video_desc                        m_saved_desc;
        video_frame_pool<default_data_allocator> m_pool;
        decoder_t                                m_decoder;
        bool                                     m_rgb;
        int                                      m_encoder_input_linesize;
        unique_ptr<char []>                      m_decoded;
public:
        encoder_state(struct state_video_compress_jpeg *s, int device_id) :
                m_parent_state(s), m_device_id(device_id), m_encoder{}, m_saved_desc{},
                m_decoder{}, m_rgb{}, m_encoder_input_linesize{},
                m_occupied{}
        {
        }
        ~encoder_state() {
                cleanup_state();
        }
        void worker();
        void compress(shared_ptr<video_frame> frame);

        synchronized_queue<shared_ptr<struct video_frame>, 1> m_in_queue; ///< queue for uncompressed frames
        thread                                   m_thread_id;
        bool                                     m_occupied; ///< protected by state_video_compress_jpeg::m_occupancy_lock
};

struct state_video_compress_jpeg {
private:
        state_video_compress_jpeg(struct module *parent, const char *opts);

        vector<struct encoder_state *> m_workers;
        bool                           m_uses_worker_threads; ///< true if cuda_devices_count > 1

        map<uint32_t, shared_ptr<struct video_frame>> m_out_frames; ///< frames decoded out of order
        uint32_t m_in_seq;  ///< seq of next frame to be encoded
        uint32_t m_out_seq; ///< seq of next frame to be decoded

        size_t m_ended_count; ///< number of workers ended
public:
        ~state_video_compress_jpeg() {
                if (m_uses_worker_threads) {
                        for (auto worker : m_workers) {
                                worker->m_thread_id.join();
                        }
                }

                for (auto worker : m_workers) {
                        delete worker;
                }
        }
        static state_video_compress_jpeg *create(struct module *parent, const char *opts);
        bool parse_fmt(char *fmt);
        void push(std::shared_ptr<video_frame> in_frame);
        std::shared_ptr<video_frame> pop();

        struct module           m_module_data;
        int                     m_restart_interval;
        int                     m_quality;

        synchronized_queue<shared_ptr<struct video_frame>, 1> m_out_queue; ///< queue for compressed frames
        mutex                                                 m_occupancy_lock;
        condition_variable                                    m_worker_finished;
};

/**
 * @brief Compresses single frame
 *
 * This function is called either from within jpeg_compress_push() if only one
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
                auto out = compress_step(move(frame));
                if (out) {
                        vf_restore_metadata(out.get(), vf_metadata);
                        out->compress_end = time_since_epoch_in_ms();
                } else {
                        log_msg(LOG_LEVEL_WARNING, "[JPEG] Failed to encode frame!\n");
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

                compress(move(frame));

                unique_lock<mutex> lk(m_parent_state->m_occupancy_lock);
                m_occupied = false;
                lk.unlock();
                m_parent_state->m_worker_finished.notify_one();
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

        bool try_slow = false;

        m_decoder = get_decoder_from_to(desc.color_spec, UYVY, try_slow);
        if (m_decoder) {
                m_rgb = false;
        } else {
                m_decoder = get_decoder_from_to(desc.color_spec, RGB, try_slow);
                if (m_decoder) {
                        m_rgb = true;
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[JPEG] Unsupported codec: %s\n",
                                        get_codec_name(desc.color_spec));
                        if (!try_slow) {
                                log_msg(LOG_LEVEL_WARNING, "[JPEG] Slow decoders not tried!\n");
                        }
                        return false;
                }
        }

        struct gpujpeg_parameters encoder_param;
        gpujpeg_set_default_parameters(&encoder_param);
        if (m_parent_state->m_quality != -1) {
                encoder_param.quality = m_parent_state->m_quality;
        } else {
                log_msg(LOG_LEVEL_INFO, "[JPEG] setting default encode parameters (quality: %d)\n",
                                encoder_param.quality);
        }

        if (m_parent_state->m_restart_interval != -1) {
                encoder_param.restart_interval = m_parent_state->m_restart_interval;
        } else {
                encoder_param.restart_interval = m_rgb ? 8 : 4;
        }

	encoder_param.verbose = 0;
	encoder_param.segment_info = 1;

        /* LUMA */
        encoder_param.sampling_factor[0].vertical = 1;
        encoder_param.sampling_factor[0].horizontal = m_rgb ? 1 : 2;
        /* Cb and Cr */
        encoder_param.sampling_factor[1].horizontal = 1;
        encoder_param.sampling_factor[1].vertical = 1;
        encoder_param.sampling_factor[2].horizontal = 1;
        encoder_param.sampling_factor[2].vertical = 1;

        encoder_param.interleaved = m_rgb ? 0 : 1;

        struct gpujpeg_image_parameters param_image;
        gpujpeg_image_set_default_parameters(&param_image);

        param_image.width = desc.width;
        param_image.height = desc.height;

        param_image.comp_count = 3;
        param_image.color_space = m_rgb ? GPUJPEG_RGB : GPUJPEG_YCBCR_BT709;
        param_image.sampling_factor = m_rgb ? GPUJPEG_4_4_4 : GPUJPEG_4_2_2;

        m_encoder = gpujpeg_encoder_create(&encoder_param, &param_image);

        int data_len = desc.width * desc.height * 3;
        m_pool.reconfigure(compressed_desc, data_len);

        m_encoder_input_linesize = desc.width *
                (param_image.color_space == GPUJPEG_RGB ? 3 : 2);

        if(!m_encoder) {
                log_msg(LOG_LEVEL_ERROR, "[JPEG] Failed to create GPUJPEG encoder.\n");
                exit_uv(EXIT_FAILURE);
                return false;
        }

        m_decoded = unique_ptr<char []>(new char[4 * desc.width * desc.height]);

        m_saved_desc = desc;

        return true;
}

bool state_video_compress_jpeg::parse_fmt(char *fmt)
{
        if(fmt && fmt[0] != '\0') {
                char *tok, *save_ptr = NULL;
                tok = strtok_r(fmt, ":", &save_ptr);
                m_quality = atoi(tok);
                if (m_quality <= 0 || m_quality > 100) {
                        log_msg(LOG_LEVEL_ERROR, "[JPEG] Error: Quality should be in interval [1-100]!\n");
                        return false;
                }

                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        m_restart_interval = atoi(tok);
                        if (m_restart_interval < 0) {
                                log_msg(LOG_LEVEL_ERROR, "[JPEG] Error: Restart interval should be non-negative!\n");
                                return false;
                        }
                }
                tok = strtok_r(NULL, ":", &save_ptr);
                if(tok) {
                        log_msg(LOG_LEVEL_WARNING, "[JPEG] WARNING: Trailing configuration parameters.\n");
                }
        }

        return true;
}

state_video_compress_jpeg::state_video_compress_jpeg(struct module *parent, const char *opts) :
        m_uses_worker_threads{}, m_in_seq{},
        m_out_seq{}, m_ended_count{},
        m_module_data{}, m_restart_interval(-1), m_quality(-1)
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
        m_module_data.deleter = [](struct module *mod) {
                struct state_video_compress_jpeg *s = (struct state_video_compress_jpeg *) mod->priv_data;
                delete s;
        };

        module_register(&m_module_data, parent);
}

/**
 * Creates JPEG encoding state and creates JPEG workers for every GPU that
 * will be used for compression (if cuda_devices_count > 1).
 */
state_video_compress_jpeg *state_video_compress_jpeg::create(struct module *parent, const char *opts) {
        assert(cuda_devices_count > 0);

        auto ret = new state_video_compress_jpeg(parent, opts);

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

struct module * jpeg_compress_init(struct module *parent, const char *opts)
{
        struct state_video_compress_jpeg *s;

        if(opts && strcmp(opts, "help") == 0) {
                printf("JPEG comperssion usage:\n");
                printf("\t-c JPEG[:<quality>[:<restart_interval>]]\n");
                return &compress_init_noerr;
        } else if(opts && strcmp(opts, "list_devices") == 0) {
                printf("CUDA devices:\n");
                gpujpeg_print_devices_info();
                return &compress_init_noerr;
        }

        try {
                s = state_video_compress_jpeg::create(parent, opts);
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
                int ret = gpujpeg_init_device(m_device_id, TRUE);

                if(ret != 0) {
                        log_msg(LOG_LEVEL_ERROR, "[JPEG] initializing CUDA device %d failed.\n", m_device_id);
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

                if ((void *) m_decoder != (void *) memcpy) {
                        unsigned char *line1 = (unsigned char *) in_tile->data;
                        unsigned char *line2 = (unsigned char *) m_decoded.get();

                        for (int i = 0; i < (int) in_tile->height; ++i) {
                                m_decoder(line2, line1, m_encoder_input_linesize,
                                                0, 8, 16);
                                line1 += vc_get_linesize(in_tile->width, tx->color_spec);
                                line2 += m_encoder_input_linesize;
                        }
                        jpeg_enc_input_data = (uint8_t *) m_decoded.get();
                } else {
                        jpeg_enc_input_data = (uint8_t *) in_tile->data;
                }

                uint8_t *compressed;
                int size;
                int ret;

                struct gpujpeg_encoder_input encoder_input;
                ret = gpujpeg_encoder_input_copy_image(&encoder_input, m_encoder, jpeg_enc_input_data);
                if (ret != 0) {
                        return {};
                }
                if (x == out->tile_count - 1) { // optimalization - dispose frame as soon as
                                                // it is not needed
                        tx = {};
                }
                ret = gpujpeg_encoder_encode(m_encoder, &encoder_input, &compressed, &size);

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

void state_video_compress_jpeg::push(std::shared_ptr<video_frame> in_frame)
{

        if (in_frame) {
                in_frame->seq = m_in_seq++;
        }

        if (!m_uses_worker_threads) {
                m_workers[0]->compress(in_frame);
        } else {
                if (!in_frame) {
                        for (auto worker : m_workers) { // pass poison pill to all workers
                                worker->m_in_queue.push({});
                        }
                } else {
                        int index;
                        unique_lock<mutex> lk(m_occupancy_lock);
                        // wait for/select not occupied worker
                        m_worker_finished.wait(lk, [this, &index]{
                                        index = 0;
                                        for (auto worker : m_workers) {
                                        if (!worker->m_occupied) return true;
                                        index++;
                                        }
                                        return false;
                                        });
                        m_workers[index]->m_occupied = true;
                        lk.unlock();
                        m_workers[index]->m_in_queue.push(in_frame);
                }
        }
}

/**
 * @brief returns compressed frame
 *
 * This function takes frames from state_video_compress_jpeg::m_out_queue. It checks
 * sequential number of frame from queue - if it is in the same order that
 * was sent to encoder, it is returned (according to state_video_compress_jpeg::m_out_seq).
 * If not, it is stored in state_video_compress_jpeg::m_out_frames and this function
 * further waits for frame with appropriate seq. Frames that was not successfully encoded
 * have data_len member set to 0 and are skipped here.
 */
std::shared_ptr<video_frame> state_video_compress_jpeg::pop()
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

const struct video_compress_info jpeg_info = {
        "JPEG",
        jpeg_compress_init,
        NULL,
        NULL,
        [](struct module *mod, std::shared_ptr<video_frame> in_frame) {
                static_cast<struct state_video_compress_jpeg *>(mod->priv_data)->push(in_frame);
        },
        [](struct module *mod) {
                return static_cast<struct state_video_compress_jpeg *>(mod->priv_data)->pop();
        },
        [] {
                return gpujpeg_init_device(cuda_devices[0], TRUE) == 0 ? list<compress_preset>{
                        { "60", 60, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 0.68);},
                                {10, 0.6, 75}, {10, 0.6, 75} },
                        { "80", 70, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 0.87);},
                                {12, 0.6, 90}, {15, 0.6, 100} },
                        { "90", 80, [](const struct video_desc *d){return (long)(d->width * d->height * d->fps * 1.54);},
                                {15, 0.6, 100}, {20, 0.6, 150} },
                } : list<compress_preset>{};
        }
};

REGISTER_MODULE(jpeg, &jpeg_info, LIBRARY_CLASS_VIDEO_COMPRESS, VIDEO_COMPRESS_ABI_VERSION);

} // end of anonymous namespace

