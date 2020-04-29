/**
 * @file   hd-rum-translator/hd-rum-decompress.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @brief  decompressing part of transcoding reflector
 */
/*
 * Copyright (c) 2013-2019 CESNET, z. s. p. o.
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
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "hd-rum-translator/hd-rum-decompress.h"
#include "hd-rum-translator/hd-rum-recompress.h"

#include <chrono>
#include <condition_variable>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "capture_filter.h"
#include "debug.h"
#include "host.h"
#include "rtp/rtp.h"

#include "video.h"
#include "video_display.h"
#include "video_rxtx/ultragrid_rtp.h"

static constexpr int MAX_QUEUE_SIZE = 2;

using namespace std;

namespace hd_rum_decompress {
struct state_transcoder_decompress : public frame_recv_delegate {
        struct output_port_info {
                inline output_port_info(void *s, bool a) : state(s), active(a) {}
                void *state;
                bool active;
        };

        struct message {
                inline message(shared_ptr<video_frame> && f) : type(FRAME), frame(std::move(f)) {}
                inline message() : type(QUIT) {}
                inline message(int ri) : type(REMOVE_INDEX), remove_index(ri) {}
                inline message(void *ns) : type(NEW_RECOMPRESS), new_recompress_state(ns) {}
                inline message(message && original);
                inline ~message();
                enum { FRAME, REMOVE_INDEX, NEW_RECOMPRESS, QUIT } type;
                union {
                        shared_ptr<video_frame> frame;
                        int remove_index;
                        void *new_recompress_state;
                };
        };

        vector<output_port_info> output_ports;

        ultragrid_rtp_video_rxtx* video_rxtx;

        queue<message> received_frame;

        mutex              lock;
        condition_variable have_frame_cv;
        condition_variable frame_consumed_cv;
        thread             worker_thread;

        struct display *display;
        struct control_state *control;
        thread         receiver_thread;

        void frame_arrived(struct video_frame *f);

        virtual ~state_transcoder_decompress() {}
        void worker();

        thread             display_thread;

        struct capture_filter *capture_filter_state;
};

void state_transcoder_decompress::frame_arrived(struct video_frame *f)
{
        auto deleter = vf_free;
        // apply capture filter
        if (f) {
                f = capture_filter(capture_filter_state, f);
        }
        if (f && f->callbacks.dispose) {
                deleter = f->callbacks.dispose;
        }
        if (!f) {
                return;
        }

        unique_lock<mutex> l(lock);
        if (received_frame.size() >= MAX_QUEUE_SIZE) {
                fprintf(stderr, "Hd-rum-decompress max queue size (%d) reached!\n", MAX_QUEUE_SIZE);
        }
        frame_consumed_cv.wait(l, [this]{ return received_frame.size() < MAX_QUEUE_SIZE; });
        received_frame.push(shared_ptr<video_frame>(f, deleter));
        l.unlock();
        have_frame_cv.notify_one();
}

inline state_transcoder_decompress::message::message(message && original)
        : type(original.type)
{
        switch (original.type) {
                case FRAME:
                        new (&frame) shared_ptr<video_frame>(std::move(original.frame));
                        break;
                case REMOVE_INDEX:
                        remove_index = original.remove_index;
                        break;
                case NEW_RECOMPRESS:
                        new_recompress_state = original.new_recompress_state;
                        break;
                case QUIT:
                        break;
        }
}

inline state_transcoder_decompress::message::~message() {
        // shared_ptr has non-trivial destructor
        if (type == FRAME) {
                frame.~shared_ptr<video_frame>();
        }
}
} // end of hd-rum-decompress namespace

using namespace hd_rum_decompress;

void hd_rum_decompress_set_active(void *state, void *recompress_port, bool active)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        for (auto && port : s->output_ports) {
                if (port.state == recompress_port) {
                        port.active = active;
                }
        }
}

ssize_t hd_rum_decompress_write(void *state, void *buf, size_t count)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        return rtp_send_raw_rtp_data(s->video_rxtx->m_network_devices[0],
                        (char *) buf, count);
}

void state_transcoder_decompress::worker()
{
        bool should_exit = false;
        while (!should_exit) {
                unique_lock<mutex> l(lock);
                have_frame_cv.wait(l, [this]{return !received_frame.empty();});

                message msg(std::move(received_frame.front()));
                l.unlock();

                switch (msg.type) {
                case message::QUIT:
                        should_exit = true;
                        break;
                case message::REMOVE_INDEX:
                        recompress_done(output_ports[msg.remove_index].state);
                        output_ports.erase(output_ports.begin() + msg.remove_index);
                        break;
                case message::NEW_RECOMPRESS:
                        output_ports.emplace_back(msg.new_recompress_state, true);
                        break;
                case message::FRAME:
                        for (unsigned int i = 0; i < output_ports.size(); ++i) {
                                if (output_ports[i].active)
                                        recompress_process_async(output_ports[i].state, msg.frame);
                        }
                        break;
                }

                // we are removing from queue now because special messages are "accepted" when queue is empty
                l.lock();
                received_frame.pop();
                l.unlock();
                frame_consumed_cv.notify_one();
        }
}

void *hd_rum_decompress_init(struct module *parent, struct hd_rum_output_conf conf, const char *capture_filter)
{
        struct state_transcoder_decompress *s;
        int force_ip_version = 0;

        s = new state_transcoder_decompress();
        chrono::steady_clock::time_point start_time(chrono::steady_clock::now());

        char cfg[128] = "";
        int ret;

        switch(conf.mode){
        case NORMAL:
                snprintf(cfg, sizeof cfg, "%p", s);
                ret = initialize_video_display(parent, "pipe", cfg, 0, NULL, &s->display);
                break;
        case BLEND:
                snprintf(cfg, sizeof cfg, "pipe:%p", s);
                ret = initialize_video_display(parent, "proxy", cfg, 0, NULL, &s->display);
                break;
        case CONFERENCE:
                snprintf(cfg, sizeof cfg, "pipe:%p#%s", s, conf.arg);
                ret = initialize_video_display(parent, "conference", cfg, 0, NULL, &s->display);
                break;
        }

        assert(ret == 0 && "Unable to initialize auxiliary display");

        map<string, param_u> params;

        // common
        params["parent"].ptr = parent;
        params["exporter"].ptr = NULL;
        params["compression"].str = "none";
        params["rxtx_mode"].i = MODE_RECEIVER;
        params["paused"].b = true;

        //RTP
        params["mtu"].i = 9000; // doesn't matter anyway...
        // should be localhost and RX TX ports the same (here dynamic) in order to work like a pipe
        params["receiver"].str = "localhost";
        params["rx_port"].i = 0;
        params["tx_port"].i = 0;
        params["force_ip_version"].b = force_ip_version;
        params["mcast_if"].str = NULL;
        params["fec"].str = "none";
        params["encryption"].str = NULL;
        params["bitrate"].ll = 0;
        params["start_time"].ptr = (void *) &start_time;
        params["video_delay"].vptr = 0;

        // UltraGrid RTP
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = s->display;

        try {
                s->video_rxtx = dynamic_cast<ultragrid_rtp_video_rxtx *>(video_rxtx::create("ultragrid_rtp", params));
                assert (s->video_rxtx);

                s->worker_thread = thread(&state_transcoder_decompress::worker, s);
                s->receiver_thread = thread(&video_rxtx::receiver_thread, s->video_rxtx);
                s->display_thread = thread(display_run, s->display);

                if (capture_filter_init(parent, capture_filter, &s->capture_filter_state) != 0) {
                        log_msg(LOG_LEVEL_ERROR, "Unable to initialize capture filter!\n");
                        return nullptr;
                }
        } catch (string const &s) {
                cerr << s << endl;
                return nullptr;
        }

        return (void *) s;
}

void hd_rum_decompress_done(void *state) {
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        should_exit = true;
        s->receiver_thread.join();

        {
                unique_lock<mutex> l(s->lock);
                s->received_frame.push({});
                l.unlock();
                s->have_frame_cv.notify_one();
        }

        s->worker_thread.join();

        // cleanup
        for (unsigned int i = 0; i < s->output_ports.size(); ++i) {
                recompress_done(s->output_ports[i].state);
        }

        display_put_frame(s->display, NULL, 0);
        s->display_thread.join();
        s->video_rxtx->join();

        delete s->video_rxtx;

        display_done(s->display);
        capture_filter_destroy(s->capture_filter_state);

        delete s;
}

void hd_rum_decompress_remove_port(void *state, int index) {
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        unique_lock<mutex> l(s->lock);
        s->received_frame.push(index);
        s->have_frame_cv.notify_one();
        s->frame_consumed_cv.wait(l, [s]{ return s->received_frame.size() == 0; });
}


void hd_rum_decompress_append_port(void *state, void *recompress_state)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        unique_lock<mutex> l(s->lock);
        s->received_frame.push(recompress_state);
        s->have_frame_cv.notify_one();
        s->frame_consumed_cv.wait(l, [s]{ return s->received_frame.size() == 0; });
}

int hd_rum_decompress_get_num_active_ports(void *state)
{
        struct state_transcoder_decompress *s = (struct state_transcoder_decompress *) state;

        int ret = 0;
        for (auto && port : s->output_ports) {
                if (port.active) {
                        ret += 1;
                }
        }

        return ret;
}

