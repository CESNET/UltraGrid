/**
 * @file   hd-rum-translator/hd-rum-decompress.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 * @brief  decompressing part of transcoding reflector
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

#include <cassert>
#include <condition_variable>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

#include "hd-rum-translator/hd-rum-decompress.h"
#include "hd-rum-translator/hd-rum-recompress.h"

#include "audio/types.h"
#include "capture_filter.h"
#include "debug.h"
#include "host.h"
#include "rtp/rtp.h"
#include "tv.h"
#include "video.h"
#include "video_display.h"
#include "video_display/pipe.h"                   // for pipe_frame_recv_del...
#include "video_rxtx.h"                           // for video_rxtx, vrxtx_pa...
#include "video_rxtx/rtp.hpp"                     // for rtp_rxtx_medium

#include "utils/profile_timer.hpp"

static constexpr int MAX_QUEUE_SIZE = 2;
#define MOD_NAME "[hd-rum-decompress] "

using std::condition_variable;
using std::condition_variable;
using std::map;
using std::mutex;
using std::string;
using std::thread;
using std::unique_lock;

namespace hd_rum_decompress {
struct state_transcoder_decompress final {
        struct video_rxtx *video_rxtx = nullptr;
        struct rtp_rxtx_common *rtp_common_state = nullptr;

        struct state_recompress *recompress = nullptr;

        std::queue<std::shared_ptr<video_frame>> received_frame;

        mutex              lock;
        condition_variable have_frame_cv;
        condition_variable frame_consumed_cv;
        thread             worker_thread;

        struct display *display = nullptr;
        struct control_state *control = nullptr;

        static void frame_arrived(void *s, struct video_frame *f, struct audio_frame *a);

        void worker();

        struct capture_filter *capture_filter_state = nullptr;

        struct common_opts common = COMMON_OPTS_INIT;
};

void state_transcoder_decompress::frame_arrived(void *state, struct video_frame *f, struct audio_frame *a)
{
        PROFILE_FUNC;
        auto *s = (struct state_transcoder_decompress *) state;
        if (f == nullptr) { // skip poison pill from vdisp/pipe
                return;
        }
        if (a) {
                LOG(LOG_LEVEL_WARNING) << "Unexpectedly receiving audio!\n";
                AUDIO_FRAME_DISPOSE(a);
        }
        auto deleter = vf_free;
        // apply capture filter
        f = capture_filter(s->capture_filter_state, f);
        if (f == nullptr) {
                return;
        }
        if (f->callbacks.dispose != nullptr) {
                deleter = f->callbacks.dispose;
        }

        unique_lock<mutex> l(s->lock);
        if (s->received_frame.size() >= MAX_QUEUE_SIZE) {
                fprintf(stderr, "Hd-rum-decompress max queue size (%d) reached!\n", MAX_QUEUE_SIZE);
        }
        s->frame_consumed_cv.wait(l, [s]{ return s->received_frame.size() < MAX_QUEUE_SIZE; });
        s->received_frame.emplace(f, deleter);
        l.unlock();
        s->have_frame_cv.notify_one();
}
} // namespace hd_rum_decompress

using namespace hd_rum_decompress;

ssize_t hd_rum_decompress_write(void *state, void *buf, size_t count)
{
        auto *s = static_cast<state_transcoder_decompress *>(state);

        struct rtp_rxtx_medium *video = &s->rtp_common_state->medium[TX_MEDIA_VIDEO];
        return rtp_send_raw_rtp_data(video->network_device, (char *) buf,
                                     (int) count);
}

void state_transcoder_decompress::worker()
{
        PROFILE_FUNC;
        bool should_exit = false;
        while (!should_exit) {
                unique_lock<mutex> l(lock);
                PROFILE_DETAIL("wait for received frame");
                have_frame_cv.wait(l, [this]{return !received_frame.empty();});

                auto frame = std::move(received_frame.front());
                l.unlock();
                PROFILE_DETAIL("");

                if(!frame){
                        should_exit = true;
                } else {
                        recompress_process_async(recompress, frame);
                }

                // we are removing from queue now because special messages are "accepted" when queue is empty
                l.lock();
                received_frame.pop();
                l.unlock();
                frame_consumed_cv.notify_one();
        }
}

void *hd_rum_decompress_init(struct module *parent, struct hd_rum_output_conf conf, const char *capture_filter, struct state_recompress *recompress)
{
        if (conf.mode == CONFERENCE && strcmp(conf.arg, "help") == 0) {
                struct display *display = nullptr;
                const int       ret     = initialize_video_display(
                    parent, "conference", "reflhelp", 0, nullptr, &display);
                assert(ret == 1);
                return nullptr;
        }

        auto *s = new state_transcoder_decompress();

        s->recompress = recompress;
        s->common.force_ip_version = 0;
        s->common.start_time = get_time_in_ns();

        const struct pipe_frame_recv_delegate dlg = {
                .state = s, .frame_arrived = state_transcoder_decompress::frame_arrived
        };
        char cfg[128] = "";
        int ret = -1;

        switch(conf.mode){
        case NORMAL:
                snprintf(cfg, sizeof cfg, "%p", &dlg);
                ret = initialize_video_display(parent, "pipe", cfg, 0, nullptr, &s->display);
                break;
        case BLEND:
                snprintf(cfg, sizeof cfg, "pipe:%p", &dlg);
                ret = initialize_video_display(parent, "blend", cfg, 0, nullptr, &s->display);
                break;
        case CONFERENCE:
                snprintf(cfg, sizeof cfg, "pipe:%p#%s", &dlg, conf.arg);
                ret = initialize_video_display(parent, "conference", cfg, 0, nullptr, &s->display);
                break;
        }

        assert(ret == 0 && MOD_NAME "Unable to initialize auxiliary display");

        struct vrxtx_params params = VRXTX_INIT;

        // common
        s->common.parent = parent;
        params.medium[TX_MEDIA_VIDEO].rxtx_mode = MODE_RECEIVER;

        //RTP
        // should be localhost and RX TX ports the same (here dynamic) in order to work like a pipe
        s->common.receiver = "localhost";
        params.medium[TX_MEDIA_VIDEO].rx_port = 0;
        params.medium[TX_MEDIA_VIDEO].tx_port = 0;
        // params["video_delay"].vptr = nullptr;

        // UltraGrid RTP
        params.decoder_mode = VIDEO_NORMAL;
        params.display_device = s->display;

        ret =
            vrxtx_init("ultragrid_rtp", &params, &s->common, &s->video_rxtx);
        assert(ret == 0 && MOD_NAME "Unable to initialize RXTX");
        size_t len = sizeof s->rtp_common_state; // NOLINT(bugprone-sizeof-expression)
        bool ctl_rc = rxtx_ctl_property(s->video_rxtx, GET_RTP_COMMON_STATE,
                                        (void *) &s->rtp_common_state, &len);
        assert(ctl_rc && MOD_NAME "Cannot get RTP state from RXTX!");

        s->worker_thread = thread(&state_transcoder_decompress::worker, s);
        display_run_new_thread(s->display);

        if (capture_filter_init(parent, capture_filter,
                                &s->capture_filter_state) != 0) {
                MSG(FATAL, "Unable to initialize capture filter!\n");
                abort();
        }

        return s;
}

void hd_rum_decompress_done(void *state) {
        auto *s = static_cast<state_transcoder_decompress *>(state);

        {
                unique_lock<mutex> l(s->lock);
                s->received_frame.push({});
                l.unlock();
                s->have_frame_cv.notify_one();
        }

        s->worker_thread.join();

        display_put_frame(s->display, nullptr, 0);
        vrxtx_join(s->video_rxtx);
        vrxtx_destroy(s->video_rxtx);

        display_join(s->display);
        display_done(s->display);
        capture_filter_destroy(s->capture_filter_state);

        delete s;
}
