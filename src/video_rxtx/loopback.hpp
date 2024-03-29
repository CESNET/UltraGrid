/**
 * @file   video_rxtx/loopback.hpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2023 CESNET, z. s. p. o.
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

#ifndef VIDEO_RXTX_LOOPBACK_HPP
#define VIDEO_RXTX_LOOPBACK_HPP

#include <condition_variable>
#include <memory>
#include <mutex>
#include <queue>

#include "types.h"
#include "video_rxtx.hpp"

struct display;

class loopback_video_rxtx : public video_rxtx {
public:
        loopback_video_rxtx(std::map<std::string, param_u> const &);
        virtual ~loopback_video_rxtx();

private:
        static void *receiver_thread(void *arg);
        virtual void send_frame(std::shared_ptr<video_frame>) noexcept override;
        void *receiver_loop();
        virtual void *(*get_receiver_thread() noexcept)(void *arg) override;

        struct display *m_display_device;
        struct video_desc m_configure_desc{};
        std::queue<std::shared_ptr<video_frame>> m_frames;
        std::condition_variable m_frame_ready;
        std::mutex m_lock;
};

#endif // !defined VIDEO_RXTX_LOOPBACK_HPP

