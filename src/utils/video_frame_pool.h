/**
 * @file   utils/video_frame_pool.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2020 CESNET, z. s. p. o.
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

#ifndef VIDEO_FRAME_POOL_H_
#define VIDEO_FRAME_POOL_H_

#include "debug.h"
#include "host.h"
#include "video.h"

#ifdef __cplusplus

#include <cassert>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <iostream>
#include <mutex>
#include <memory>
#include <queue>
#include <stdexcept>

struct default_data_allocator {
        void *allocate(size_t size) {
                return malloc(size);
        }
        void deallocate(void *ptr) {
                free(ptr);
        }
};

template <typename allocator>
struct video_frame_pool {
        public:
                /**
                 * @param max_used_frames maximal frames that are allocated.
                 *                        0 means unlimited. If get_frames()
                 *                        is called and that number of frames
                 *                        is unreturned, get_frames() will block.
                 */
                video_frame_pool(unsigned int max_used_frames = 0) : m_generation(0), m_desc(), m_max_data_len(0), m_unreturned_frames(0), m_max_used_frames(max_used_frames) {
                }

                virtual ~video_frame_pool() {
                        std::unique_lock<std::mutex> lk(m_lock);
                        remove_free_frames();
                        // wait also for all frames we gave out to return us
                        m_frame_returned.wait(lk, [this] {return m_unreturned_frames == 0;});
                }

                /**
                 * @param new_size  if omitted, deduce from video desc (only for pixel formats)
                 */
                void reconfigure(struct video_desc new_desc, size_t new_size = SIZE_MAX) {
                        std::unique_lock<std::mutex> lk(m_lock);
                        m_desc = new_desc;
                        m_max_data_len = new_size != SIZE_MAX ? new_size : new_desc.height * vc_get_linesize(new_desc.width, new_desc.color_spec);
                        remove_free_frames();
                        m_generation++;
                }

                /**
                 * Returns free frame
                 *
                 * If the pool size is exhausted (see constructor param),
                 * the call will block until there is some available.
                 */
                std::shared_ptr<video_frame> get_frame() {
                        assert(m_generation != 0);
                        struct video_frame *ret = NULL;
                        std::unique_lock<std::mutex> lk(m_lock);
                        if (!m_free_frames.empty()) {
                                ret = m_free_frames.front();
                                m_free_frames.pop();
                        } else if (m_max_used_frames > 0 && m_max_used_frames == m_unreturned_frames) {
                                m_frame_returned.wait(lk, [this] {return m_unreturned_frames < m_max_used_frames;});
                                assert(!m_free_frames.empty());
                                ret = m_free_frames.front();
                                m_free_frames.pop();
                        } else {
                                try {
                                        ret = vf_alloc_desc(m_desc);
                                        for (unsigned int i = 0; i < m_desc.tile_count; ++i) {
                                                ret->tiles[i].data = (char *)
                                                        m_allocator.allocate(m_max_data_len);
                                                if (ret->tiles[i].data == NULL) {
                                                        throw std::runtime_error("Cannot allocate data");
                                                }
                                                ret->tiles[i].data_len = m_max_data_len;
                                        }
                                } catch (std::exception &e) {
                                        std::cerr << e.what() << std::endl;
                                        deallocate_frame(ret);
                                        throw e;
                                }
                        }
                        m_unreturned_frames += 1;
                        return std::shared_ptr<video_frame>(ret, std::bind([this](struct video_frame *frame, int generation) {
                                        std::unique_lock<std::mutex> lk(m_lock);

                                        assert(m_unreturned_frames > 0);
                                        m_unreturned_frames -= 1;
                                        m_frame_returned.notify_one();

                                        if (this->m_generation != generation) {
                                                this->deallocate_frame(frame);
                                        } else {
                                                m_free_frames.push(frame);
                                        }
                                }, std::placeholders::_1, m_generation));
                }

                /** @returns legacy struct pointer with dispose callback properly set */
                struct video_frame *get_disposable_frame() {
                        auto && frame = get_frame();
                        struct video_frame *out = frame.get();
                        out->callbacks.dispose_udata = new std::shared_ptr<video_frame>(frame);
                        out->callbacks.dispose = [](video_frame *f) { delete static_cast<std::shared_ptr<video_frame> *>(f->callbacks.dispose_udata); };
                        return out;
                }

                allocator & get_allocator() {
                        return m_allocator;
                }

        private:
                void remove_free_frames() {
                        while (!m_free_frames.empty()) {
                                struct video_frame *frame = m_free_frames.front();
                                m_free_frames.pop();
                                deallocate_frame(frame);
                        }
                }

                void deallocate_frame(struct video_frame *frame) {
                        if (frame == NULL)
                                return;
                        for (unsigned int i = 0; i < frame->tile_count; ++i) {
                                m_allocator.deallocate(frame->tiles[i].data);
                        }
                        vf_free(frame);
                }

                std::queue<struct video_frame *> m_free_frames;
                std::mutex        m_lock;
                std::condition_variable m_frame_returned;
                int               m_generation;
                struct video_desc m_desc;
                size_t            m_max_data_len;
                unsigned int      m_unreturned_frames;
                allocator         m_allocator;
                unsigned int      m_max_used_frames;
};
#endif //  __cplusplus

EXTERN_C void *video_frame_pool_init(struct video_desc desc, int len);
EXTERN_C struct video_frame *video_frame_pool_get_disposable_frame(void *);
EXTERN_C void video_frame_pool_destroy(void *);

#endif // VIDEO_FRAME_POOL_H_

