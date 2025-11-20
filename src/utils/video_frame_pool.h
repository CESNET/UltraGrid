/**
 * @file   utils/video_frame_pool.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020-2025 CESNET, zájmoveé sdružení právnických osob
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

#include "types.h"         // for video_frame
#include "utils/macros.h"

#ifdef __cplusplus

#include <cstddef>         // for size_t
#include <memory>

struct video_frame_pool_allocator {
        virtual void *allocate(size_t size) = 0;
        virtual void deallocate(void *ptr) = 0;
        virtual struct video_frame_pool_allocator *clone() const = 0;
        virtual ~video_frame_pool_allocator() {}
};

struct default_data_allocator : public video_frame_pool_allocator {
        void *allocate(size_t size) override;
        void deallocate(void *ptr) override;
        struct video_frame_pool_allocator *clone() const override;
};

struct video_frame_pool {
        public:
                /**
                 * @param max_used_frames maximal frames that are allocated.
                 *                        0 means unlimited. If get_frames()
                 *                        is called and that number of frames
                 *                        is unreturned, get_frames() will block.
                 */
                video_frame_pool(unsigned int max_used_frames = 0, video_frame_pool_allocator const &alloc = default_data_allocator());
                video_frame_pool &operator=(video_frame_pool &&);
                ~video_frame_pool();

                /**
                 * @param new_size  if omitted, deduce from video desc (only for pixel formats)
                 */
                void reconfigure(struct video_desc new_desc, size_t new_size = SIZE_MAX);

                /**
                 * Returns free frame
                 *
                 * If the pool size is exhausted (see constructor param),
                 * the call will block until there is some available.
                 */
                std::shared_ptr<video_frame> get_frame() noexcept(false);

                /** @returns legacy struct pointer with dispose callback properly set */
                struct video_frame *get_disposable_frame();

                /** @returns frame eligible to be freed by vf_free() */
                struct video_frame *get_pod_frame();

                video_frame_pool_allocator const & get_allocator();

        private:
                struct impl;
                std::unique_ptr<impl> m_impl;
};
#endif //  __cplusplus

EXTERN_C void *video_frame_pool_init(struct video_desc desc, int len);
EXTERN_C struct video_frame *video_frame_pool_get_disposable_frame(void *);
EXTERN_C void video_frame_pool_destroy(void *);

#endif // VIDEO_FRAME_POOL_H_

