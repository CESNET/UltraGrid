/**
 * @file   video_rxtx/sage.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2026 CESNET, zájmové sdružení právnických osob
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

#include <cstring>          // for NULL, memcpy, memset, strlen
#include <memory>           // for shared_ptr
#include <sstream>          // for basic_ostringstream, operator<<, basic_os...
#include <string>           // for allocator, char_traits, basic_string, string
#include <utility>          // for move

#include "lib_common.h"
#include "types.h"
#include "video_display.h"
#include "video_frame.h"    // for video_desc_from_frame, video_desc_eq
#include "video_rxtx.h"

using namespace std;

class sage_video_rxtx {
public:
        sage_video_rxtx(const struct vrxtx_params *params,
                        const struct common_opts  *common);
        ~sage_video_rxtx();
        void send_frame(std::shared_ptr<video_frame>) noexcept;

private:
        struct video_desc     m_saved_video_desc;
        struct display       *m_sage_tx_device;
};

sage_video_rxtx::sage_video_rxtx(
    const struct vrxtx_params                 *params,
    [[maybe_unused]] const struct common_opts *common)
{
        ostringstream oss;

        auto *m_sender_mod = params->sender_mod;

        if (strlen(params->protocol_opts) > 0) {
                oss << params->protocol_opts << ":";
        }

        oss << "fs=" << params->receiver;
        oss << ":tx"; // indicates that we are in tx mode
        int ret = initialize_video_display(m_sender_mod, "sage",
                        oss.str().c_str(), 0, NULL, &m_sage_tx_device);
        if(ret != 0) {
                throw string("Unable to initialize SAGE TX.");
        }
        display_run_new_thread(m_sage_tx_device);
        memset(&m_saved_video_desc, 0, sizeof(m_saved_video_desc));
}

void
sage_video_rxtx::send_frame(shared_ptr<video_frame> tx_frame) noexcept
{
        if (!video_desc_eq(m_saved_video_desc,
                           video_desc_from_frame(tx_frame.get()))) {
                display_reconfigure(m_sage_tx_device,
                                video_desc_from_frame(tx_frame.get()), VIDEO_NORMAL);
                m_saved_video_desc = video_desc_from_frame(tx_frame.get());
        }
        struct video_frame *frame =
                display_get_frame(m_sage_tx_device);
        memcpy(frame->tiles[0].data, tx_frame->tiles[0].data,
                        tx_frame->tiles[0].data_len);
        display_put_frame(m_sage_tx_device, frame, PUTF_NONBLOCK);
}

sage_video_rxtx::~sage_video_rxtx()
{
        // poisoned pill to exit thread
        display_put_frame(m_sage_tx_device, NULL, PUTF_NONBLOCK);

        display_join(m_sage_tx_device);
        display_done(m_sage_tx_device);
}

static void *
create_video_rxtx_sage(const struct vrxtx_params *params,
                       const struct common_opts  *common)
{
        return new sage_video_rxtx(params, common);
}

static void done(void *state) {
        auto *s = static_cast<sage_video_rxtx *>(state);
        delete s;
}

static void
send_frame(void *state, std::shared_ptr<video_frame> f)
{
        auto *s = static_cast<sage_video_rxtx *>(state);
        s->send_frame(std::move(f));
}

static const struct video_rxtx_info sage_video_rxtx_info = {
        .long_name          = "SAGE",
        .create             = create_video_rxtx_sage,
        .done               = done,
        .send_audio_frame   = nullptr,
        .send_video_frame   = send_frame,
        .video_recv_routine = nullptr,
        .ctl_property       = nullptr,
        .join_sender        = nullptr,
};

REGISTER_MODULE(sage, &sage_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

