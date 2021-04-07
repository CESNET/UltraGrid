/**
 * @file   video_display/vrg.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020-2021 CESNET, z. s. p. o.
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
#endif
#include "config_win32.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <rtp/rtp.h>
#include <queue>
#include <string>

// VrgInputFormat::RGBA conflicts with codec_t::RGBA
#define RGBA VR_RGBA
#ifdef HAVE_VRG_H
#include <vrgstream.h>
#else
#include "vrgstream-fallback.h"
#endif
#undef RGBA

#include "debug.h"
#include "lib_common.h"
#include "utils/misc.h"
#include "utils/video_frame_pool.h"
#include "video.h"
#include "video_display.h"

#define MAX_QUEUE_SIZE 1
#define MOD_NAME "[VRG] "
#define MAGIC_VRG to_fourcc('V', 'R', 'G', ' ')

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#endif

using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
using std::chrono::microseconds;
using std::condition_variable;
using std::cout;
using std::mutex;
using std::queue;
using std::unique_lock;
using namespace std::string_literals;

#ifdef HAVE_CUDA
struct cuda_malloc_host_allocate {
        cudaError_t operator()(void **ptr, size_t size) {
                return cudaMallocHost(ptr, size);
        }
};

struct cuda_malloc_managed_allocate {
        cudaError_t operator()(void **ptr, size_t size) {
                return cudaMallocManaged(ptr, size);
        }
};
#endif

template<typename cuda_allocator>
struct vrg_cuda_allocator : public video_frame_pool_allocator {
        void *allocate(size_t size) override {
                void *ptr = nullptr;
#ifdef HAVE_CUDA
                if (cuda_allocator()(&ptr, size) != cudaSuccess) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Cannot alloc CUDA buffer!\n";
                        return nullptr;
                }
#else
                LOG(LOG_LEVEL_WARNING) << MOD_NAME << "CUDA not compiled in, falling back to plain malloc!\n";
                return malloc(size);
#endif
                return ptr;
        }
        void deallocate(void *ptr) override {
#ifdef HAVE_CUDA
                cudaFree(ptr);
#else
                free(ptr);
#endif
        }
        video_frame_pool_allocator *clone() const override {
                return new vrg_cuda_allocator(*this);
        }
};

struct state_vrg {
        uint32_t magic;
        struct video_desc saved_desc;
#ifdef HAVE_CUDA
        video_frame_pool pool{0, vrg_cuda_allocator<cuda_malloc_host_allocate>()};
#else
        video_frame_pool pool{0, vrg_cuda_allocator<default_data_allocator>()};
#endif

        high_resolution_clock::time_point t0 = high_resolution_clock::now();
        long long int frames;
        long long int frames_last;

        struct rtp *rtp;

        queue<struct video_frame *> queue;
        mutex lock;
        condition_variable cv;
};

static void display_vrg_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *available_cards = NULL;
        *count = 0;
}

static void *display_vrg_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        UNUSED(parent);
        if ("help"s == fmt) {
                cout << "Usage:\n\t-d vrg[:managed]\n";
                cout << "where\n\tmanaged - use managed memory\n";
                return NULL;
        }

        struct state_vrg *s = new state_vrg();
        if (s == NULL) {
                return NULL;
        }
        s->magic = MAGIC_VRG;

        if ("managed"s == fmt) {
#ifdef HAVE_CUDA
                s->pool.replace_allocator(vrg_cuda_allocator<cuda_malloc_managed_allocate>());
#endif
        }

        return s;
}

static void display_vrg_run(void *state)
{
        struct state_vrg *s = (struct state_vrg *) state;
        codec_t configured_codec = VIDEO_CODEC_NONE;

        while (1) {
                unique_lock<mutex> lk(s->lock);
                s->cv.wait(lk, [s]{ return s->queue.size() > 0; });
                struct video_frame *f = s->queue.front();
                s->queue.pop();
                lk.unlock();

                if (f == nullptr) { // poison pill
                        break;
                }

                if (configured_codec != f->color_spec) {
                        enum VrgInputFormat vrg_format{};
                        switch (f->color_spec) {
                                case CUDA_I420:
                                case I420:
                                        vrg_format = YUV420;
                                        break;
                                case CUDA_RGBA:
                                case RGBA:
                                        vrg_format = VR_RGBA;
                                        break;
                                default:
                                        abort();

                        }
                        enum VrgStreamApiError ret = vrgStreamInit(vrg_format);
                        if (ret != Ok) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME "Initialization failed: " << ret << "\n";
                                vf_free(f);
                                continue;
                        }
                        configured_codec = f->color_spec;
                }

                enum VrgStreamApiError ret;
                high_resolution_clock::time_point t_start = high_resolution_clock::now();
                ret = vrgStreamSubmitFrame(&f->render_packet, f->tiles[0].data, CPU);
                if (ret != Ok) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Submit Frame failed: " << ret << "\n";
                }
                high_resolution_clock::time_point t_end = high_resolution_clock::now();
                LOG(LOG_LEVEL_DEBUG) << "[VRG] Frame submit took " <<
                        duration_cast<microseconds>(t_end - t_start).count() / 1000000.0
                        << " seconds\n";

                struct RenderPacket render_packet{};
                ret = vrgStreamRenderFrame(&render_packet);
                if (ret != Ok) {
                        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Render Frame failed: " << ret << "\n";
                } else {
                        LOG(LOG_LEVEL_DEBUG) << MOD_NAME "Received RenderPacket for frame " << render_packet.frame << ".\n";
                        rtp_send_rtcp_app(s->rtp, "VIEW", sizeof render_packet, (char *) &render_packet);
                }

                high_resolution_clock::time_point now = high_resolution_clock::now();
                double seconds = duration_cast<microseconds>(now - s->t0).count() / 1000000.0;
                if (seconds >= 5) {
                        long long frames = s->frames - s->frames_last;
                        LOG(LOG_LEVEL_INFO) << "[VRG] " << frames << " frames in "
                                << seconds << " seconds = " <<  frames / seconds << " FPS\n";
                        s->t0 = now;
                        s->frames_last = s->frames;
                }

                s->frames += 1;
                vf_free(f);
        }
}

static void display_vrg_done(void *state)
{
        struct state_vrg *s = (struct state_vrg *) state;
        assert(s->magic == MAGIC_VRG);
        delete s;
}

static struct video_frame *display_vrg_getf(void *state)
{
        struct state_vrg *s = (struct state_vrg *) state;
        assert(s->magic == MAGIC_VRG);

        return s->pool.get_pod_frame();
}

static int display_vrg_putf(void *state, struct video_frame *frame, int flags)
{
        struct state_vrg *s = (struct state_vrg *) state;
        assert(s->magic == MAGIC_VRG);

        if (flags == PUTF_DISCARD) {
                vf_free(frame);
                return 0;
        }

        unique_lock<mutex> lk(s->lock);
        if ((flags & PUTF_NONBLOCK) != 0u && s->queue.size() >= MAX_QUEUE_SIZE) {
                vf_free(frame);
                return 1;
        }
        s->queue.push(frame);
        lk.unlock();
        s->cv.notify_one();

        return 0;
}

static int display_vrg_ctl_property(void *state, int property, void *val, size_t *len)
{
        struct state_vrg *s = (struct state_vrg *) state;
        codec_t codecs[] = {
#ifdef HAVE_CUDA
                CUDA_I420,
                CUDA_RGBA,
#endif
                I420,
                RGBA,
        };
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) > *len) {
                                return FALSE;
                        }
                        memcpy(val, codecs, sizeof(codecs));
                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_S_RTP:
                        s->rtp = *(struct rtp **) val;
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_vrg_reconfigure(void *state, struct video_desc desc)
{
        struct state_vrg *s = (struct state_vrg *) state;
        assert(s->magic == MAGIC_VRG);
        assert(desc.color_spec == CUDA_I420 || desc.color_spec == CUDA_RGBA || desc.color_spec == RGBA || desc.color_spec == I420);

        s->saved_desc = desc;
#if 0
        if (desc.color_spec == CUDA_I420 || desc.color_spec == CUDA_RGBA) {
                s->pool.replace_allocator(vrg_cuda_allocator());
        } else {
                s->pool.replace_allocator(default_data_allocator());
        }
#endif
        s->pool.reconfigure(desc);

        return TRUE;
}

static void display_vrg_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_vrg_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_vrg_info = {
        display_vrg_probe,
        display_vrg_init,
        display_vrg_run,
        display_vrg_done,
        display_vrg_getf,
        display_vrg_putf,
        display_vrg_reconfigure,
        display_vrg_ctl_property,
        display_vrg_put_audio_frame,
        display_vrg_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_MODULE(vrg, &display_vrg_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

