/**
 * @file   video_capture/ug_input.cpp
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include "audio/audio.h"
#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_display/pipe.hpp" // frame_recv_delegate
#include "video_rxtx.h"
#include "video_rxtx/ultragrid_rtp.h"

#include <chrono>
#include <iostream>
#include <mutex>
#include <memory>
#include <queue>
#include <thread>

static constexpr int MAX_QUEUE_SIZE = 2;

using namespace std;
using namespace std::chrono;

struct ug_input_state  : public frame_recv_delegate {
        mutex lock;
        queue<pair<struct video_frame *, struct audio_frame *>> frame_queue;
        struct display *display;

        void frame_arrived(struct video_frame *f, struct audio_frame *a);
        thread         receiver_thread;
        thread             display_thread;
        unique_ptr<ultragrid_rtp_video_rxtx> video_rxtx;
        struct state_audio *audio;

        virtual ~ug_input_state() {}

        std::chrono::steady_clock::time_point t0;
        int frames;
};

void ug_input_state::frame_arrived(struct video_frame *f, struct audio_frame *a)
{
        lock_guard<mutex> lk(lock);
        if (frame_queue.size() < MAX_QUEUE_SIZE) {
                frame_queue.push({f, a});
        } else {
                cerr << "[ug_input] Dropping frame!" << endl;
                AUDIO_FRAME_DISPOSE(a);
                VIDEO_FRAME_DISPOSE(f);
        }
}

static int vidcap_ug_input_init(struct vidcap_params *cap_params, void **state)
{
        uint16_t port = 5004;

        if (strcmp("help", vidcap_params_get_fmt(cap_params)) == 0) {
                printf("Usage:\n");
                printf("\t-t ug_input[:<port>] [-s embedded]\n");
                return VIDCAP_INIT_NOERR;
        }
        ug_input_state *s = new ug_input_state();

        if (isdigit(vidcap_params_get_fmt(cap_params)[0])) {
                port = atoi(vidcap_params_get_fmt(cap_params));
        }

        char cfg[128] = "";
        snprintf(cfg, sizeof cfg, "%p", s);
        int ret = initialize_video_display(vidcap_params_get_parent(cap_params), "pipe", cfg, 0, NULL, &s->display);
        assert(ret == 0 && "Unable to initialize proxy display");

        auto start_time = std::chrono::steady_clock::now();
        map<string, param_u> params;

        // common
        params["parent"].ptr = vidcap_params_get_parent(cap_params);
        params["exporter"].ptr = NULL;
        params["compression"].str = "none";
        params["rxtx_mode"].i = MODE_RECEIVER;
        params["paused"].i = false;

        //RTP
        params["mtu"].i = 9000; // doesn't matter anyway...
        // should be localhost and RX TX ports the same (here dynamic) in order to work like a pipe
        params["receiver"].str = "localhost";
        params["rx_port"].i = port;
        params["tx_port"].i = 0;
        params["force_ip_version"].i = 0;
        params["mcast_if"].str = NULL;
        params["fec"].str = "none";
        params["encryption"].str = NULL;
        params["bitrate"].ll = 0;
        params["start_time"].cptr = (const void *) &start_time;
        params["video_delay"].vptr = 0;

        // UltraGrid RTP
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = s->display;

        s->video_rxtx = unique_ptr<ultragrid_rtp_video_rxtx>(dynamic_cast<ultragrid_rtp_video_rxtx *>(video_rxtx::create("ultragrid_rtp", params)));
        assert (s->video_rxtx);

        if (vidcap_params_get_flags(cap_params) & VIDCAP_FLAG_AUDIO_ANY) {
                const char *audio_scale = "none";
                s->audio = audio_cfg_init(vidcap_params_get_parent(cap_params), "localhost", port + 2, 0 /* send_port */,
                                "none", "embedded",
                                "ultragrid_rtp", "",
                                "none", NULL, NULL, audio_scale, false, 0, NULL, "PCM", RATE_UNLIMITED, NULL,
                                &start_time, 1500, NULL);
                if (s->audio == nullptr) {
                        delete s;
                        return VIDCAP_INIT_FAIL;
                }

                audio_register_display_callbacks(s->audio,
                                s->display,
                                (void (*)(void *, struct audio_frame *)) display_put_audio_frame,
                                (int (*)(void *, int, int, int)) display_reconfigure_audio,
                                (int (*)(void *, int, void *, size_t *)) display_ctl_property);

                audio_start(s->audio);
        }
        s->t0 = steady_clock::now();

        s->receiver_thread = thread(&video_rxtx::receiver_thread, s->video_rxtx.get());
        s->display_thread = thread(display_run, s->display);

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_ug_input_done(void *state)
{
        auto s = (ug_input_state *) state;

        audio_join(s->audio);
        s->receiver_thread.join();

        display_put_frame(s->display, NULL, 0);
        s->display_thread.join();
        display_done(s->display);

        s->video_rxtx->join();

        while (!s->frame_queue.empty()) {
                auto item = s->frame_queue.front();
                s->frame_queue.pop();
                VIDEO_FRAME_DISPOSE(item.first);
                AUDIO_FRAME_DISPOSE(item.second);
        }
        audio_done(s->audio);

        delete s;
}

static struct video_frame *vidcap_ug_input_grab(void *state, struct audio_frame **audio)
{
        auto s = (ug_input_state *) state;
        *audio = NULL;
        lock_guard<mutex> lk(s->lock);
        if (s->frame_queue.empty()) {
                return NULL;
        } else {
                auto item = s->frame_queue.front();
                struct video_frame *frame = item.first;
                *audio = item.second;
                s->frame_queue.pop();
                frame->callbacks.dispose = vf_free;

                s->frames++;
                auto curr_time = steady_clock::now();
                double seconds = duration_cast<duration<double>>(curr_time - s->t0).count();
                if (seconds >= 5.0) {
                        float fps = s->frames / seconds;
                        log_msg(LOG_LEVEL_INFO, "[ug_input] %d frames in %g seconds = %g FPS\n",
                                        s->frames, seconds, fps);
                        s->t0 = curr_time;
                        s->frames = 0;
                }

                return frame;
        }
}

static struct vidcap_type *vidcap_ug_input_probe(bool /* verbose */, void (**deleter)(void *))
{
        struct vidcap_type *vt;
        *deleter = free;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name = "ug_input";
                vt->description = "Dummy capture from UG received network";
        }
        return vt;
}

static const struct video_capture_info vidcap_ug_input_info = {
        vidcap_ug_input_probe,
        vidcap_ug_input_init,
        vidcap_ug_input_done,
        vidcap_ug_input_grab,
        false
};

REGISTER_MODULE(ug_input, &vidcap_ug_input_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

