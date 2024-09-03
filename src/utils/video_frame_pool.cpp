/**
 * @file   utils/video_frame_pool.cpp
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

#include "config_msvc.h"

#include <cassert>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "video_frame_pool.h"

void *default_data_allocator::allocate(size_t size) {
        return malloc(size);
}
void default_data_allocator::deallocate(void *ptr) {
        free(ptr);
}
struct video_frame_pool_allocator *default_data_allocator::clone() const {
        return new default_data_allocator(*this);
}

video_frame_pool::video_frame_pool(unsigned int max_used_frames, video_frame_pool_allocator const &alloc) : m_allocator(alloc.clone()), m_generation(0), m_desc(), m_max_data_len(0), m_unreturned_frames(0), m_max_used_frames(max_used_frames) {
}

video_frame_pool::~video_frame_pool() {
        std::unique_lock<std::mutex> lk(m_lock);
        remove_free_frames();
        // wait also for all frames we gave out to return us
        m_frame_returned.wait(lk, [this] {return m_unreturned_frames == 0;});
}

void video_frame_pool::reconfigure(struct video_desc new_desc, size_t new_size) {
        std::unique_lock<std::mutex> lk(m_lock);
        m_desc = new_desc;
        m_max_data_len = new_size != SIZE_MAX ? new_size : new_desc.height * vc_get_linesize(new_desc.width, new_desc.color_spec);
        remove_free_frames();
        m_generation++;
}

std::shared_ptr<video_frame> video_frame_pool::get_frame() {
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
                                ret->tiles[i].data =
                                    (char *) m_allocator->allocate(
                                        m_max_data_len + MAX_PADDING);
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

struct video_frame *video_frame_pool::get_disposable_frame() {
        auto && frame = get_frame();
        struct video_frame *out = frame.get();
        out->callbacks.dispose_udata =
            new std::shared_ptr<video_frame>(std::move(frame));
        static auto dispose = [](video_frame *f) { delete static_cast<std::shared_ptr<video_frame> *>(f->callbacks.dispose_udata); };
        out->callbacks.dispose = dispose;
        return out;
}

struct video_frame *video_frame_pool::get_pod_frame() {
        auto && frame = get_frame();
        struct video_frame *out = vf_alloc_desc(video_desc_from_frame(frame.get()));
        for (unsigned int i = 0; i < frame->tile_count; ++i) {
                out->tiles[i].data = frame->tiles[i].data;
        }
        out->callbacks.dispose_udata =
            new std::shared_ptr<video_frame>(std::move(frame));
        static auto deleter = [](video_frame *f) { delete static_cast<std::shared_ptr<video_frame> *>(f->callbacks.dispose_udata); };
        out->callbacks.data_deleter = deleter;
        return out;
}

video_frame_pool_allocator const & video_frame_pool::get_allocator() {
        return *m_allocator;
}

void video_frame_pool::remove_free_frames() {
        while (!m_free_frames.empty()) {
                struct video_frame *frame = m_free_frames.front();
                m_free_frames.pop();
                deallocate_frame(frame);
        }
}

void video_frame_pool::deallocate_frame(struct video_frame *frame) {
        if (frame == NULL)
                return;
        for (unsigned int i = 0; i < frame->tile_count; ++i) {
                m_allocator->deallocate(frame->tiles[i].data);
        }
        vf_free(frame);
}

void *video_frame_pool_init(struct video_desc desc, int len) {
        auto *out = new video_frame_pool(len, default_data_allocator());
        out->reconfigure(desc);
        return (void *) out;
}

struct video_frame *video_frame_pool_get_disposable_frame(void *state) {
        auto *s = static_cast<video_frame_pool* >(state);
        return s->get_disposable_frame();
}

void video_frame_pool_destroy(void *state) {
        auto *s = static_cast<video_frame_pool* >(state);
        delete s;
}

