/**
 * @file   video_rxtx/loopback.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2026 CESNET, zájmové sdružení právnických osob
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
 * @todo
 * * works only when device can directly display the codec natively
 * * add also audio
 */

#include <chrono>              // for milliseconds
#include <condition_variable>  // for condition_variable
#include <cstring>             // for memcpy
#include <memory>              // for shared_ptr
#include <mutex>               // for mutex, unique_lock
#include <ostream>             // for char_traits, basic_ostream, operator<<
#include <queue>               // for queue
#include <utility>             // for move

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/thread.h"
#include "video_display.h"
#include "video_frame.h"

using std::chrono::milliseconds;
using std::mutex;
using std::unique_lock;

static const int BUFF_MAX_LEN = 2;
static const char *MODULE_NAME = "[loopback] ";


#include "types.h"
#include "video_rxtx.h"

class loopback_video_rxtx {
public:
        loopback_video_rxtx(const struct vrxtx_params *params);
        void send_frame(std::shared_ptr<video_frame>) noexcept;
        static void *receiver_thread(void *arg);

private:
        struct module *m_parent;
        void *receiver_loop();

        struct display *m_display_device;
        struct video_desc m_configure_desc{};
        std::queue<std::shared_ptr<video_frame>> m_frames;
        std::condition_variable m_frame_ready;
        std::mutex m_lock;
};

loopback_video_rxtx::loopback_video_rxtx(const struct vrxtx_params *params)
    : m_parent(params->parent), m_display_device(params->display_device)
{
}

void *loopback_video_rxtx::receiver_thread(void *arg)
{
        auto s = static_cast<loopback_video_rxtx *>(arg);
        return s->receiver_loop();
}

void
loopback_video_rxtx::send_frame(std::shared_ptr<video_frame> f) noexcept
{
        unique_lock<mutex> lk(m_lock);
        if (m_frames.size() >= BUFF_MAX_LEN) {
                LOG(LOG_LEVEL_WARNING) << MODULE_NAME << "Max buffer len " <<
                        BUFF_MAX_LEN << " exceeded.\n";
        }
        m_frames.push(f);
        lk.unlock();
        m_frame_ready.notify_one();
}

static void
should_exit_callback(void *should_exit)
{
        *(bool *) should_exit = true;
}

void *loopback_video_rxtx::receiver_loop()
{
        set_thread_name(__func__);
        bool should_exit = false;
        register_should_exit_callback(m_parent, should_exit_callback, &should_exit);

        while (!should_exit) {
                unique_lock<mutex> lk(m_lock);
                m_frame_ready.wait_for(lk, milliseconds(100), [this]{return m_frames.size() > 0;});
                if (m_frames.size() == 0) {
			continue;
		}

                auto frame = m_frames.front();
                m_frames.pop();
                lk.unlock();
                auto new_desc = video_desc_from_frame(frame.get());
                if (m_configure_desc != new_desc) {
                        if (display_reconfigure(m_display_device, new_desc, VIDEO_NORMAL) == false) {
                                LOG(LOG_LEVEL_ERROR) << "Unable to reconfigure display!\n";
                                continue;
                        }
                        m_configure_desc = new_desc;
                }
                auto display_f = display_get_frame(m_display_device);
                memcpy(display_f->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
                display_put_frame(m_display_device, display_f, PUTF_BLOCKING);
        }
        display_put_frame(m_display_device, nullptr, PUTF_BLOCKING);
        unregister_should_exit_callback(m_parent, should_exit_callback,
                                        &should_exit);
        return nullptr;
}

static void*
create_video_rxtx_loopback(const struct vrxtx_params *params)
{
        return new loopback_video_rxtx(params);
}

static void done(void *state) {
        auto *s = static_cast<loopback_video_rxtx *>(state);
        delete s;
}

static void
send_frame(void *state, std::shared_ptr<video_frame> f)
{
        auto *s = static_cast<loopback_video_rxtx *>(state);
        s->send_frame(std::move(f));
}

static const struct video_rxtx_info loopback_video_rxtx_info = {
        .long_name    = "loopback dummy transport",
        .create       = create_video_rxtx_loopback,
        .done         = done,
        .ctl_property = nullptr,

        .send_audio_frame = nullptr,
        .recv_audio_frame = nullptr,

        .send_video_frame   = send_frame,
        .send_video_frame_c = nullptr,
        .video_recv_routine = loopback_video_rxtx::receiver_thread,
        .join_video_sender  = nullptr,
};

REGISTER_MODULE(loopback, &loopback_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

