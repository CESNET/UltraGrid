/**
 * @file   video_rxtx/loopback.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
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
 *
 * @todo
 * * works only when device can directly display the codec natively
 * * add also audio
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "video_rxtx/loopback.h"

#include <chrono>

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video_display.h"
#include "video_frame.h"

using std::chrono::milliseconds;
using std::condition_variable;
using std::mutex;
using std::unique_lock;

static const int BUFF_MAX_LEN = 2;
static const char *MODULE_NAME = "[loopback] ";

loopback_video_rxtx::loopback_video_rxtx(std::map<std::string, param_u> const &params)
        : video_rxtx(params)
{
        m_display_device = static_cast<struct display *>(params.at("display_device").ptr);
}

loopback_video_rxtx::~loopback_video_rxtx()
{
}

void *loopback_video_rxtx::receiver_thread(void *arg)
{
        auto s = static_cast<loopback_video_rxtx *>(arg);
        return s->receiver_loop();
}

void loopback_video_rxtx::send_frame(std::shared_ptr<video_frame> f)
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

void *loopback_video_rxtx::receiver_loop()
{
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
                        if (display_reconfigure(m_display_device, new_desc, VIDEO_NORMAL) == FALSE) {
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
        return nullptr;
}

void *(*loopback_video_rxtx::get_receiver_thread())(void *arg)
{
        return receiver_thread;
}

static video_rxtx *create_video_rxtx_loopback(std::map<std::string, param_u> const &params)
{
        return new loopback_video_rxtx(params);
}

static const struct video_rxtx_info loopback_video_rxtx_info = {
        "loopback dummy transport",
        create_video_rxtx_loopback
};

REGISTER_MODULE(loopback, &loopback_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

