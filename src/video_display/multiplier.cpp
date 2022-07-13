/**
 * @file   video_display/multiplier.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2021 CESNET, z. s. p. o.
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
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "utils/misc.h"

#include <condition_variable>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>

using namespace std;

static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;
static constexpr const char *MOD_NAME = "[multiplier] ";
static constexpr int SKIP_FIRST_N_FRAMES_IN_STREAM = 5;

namespace{
struct disp_deleter{ void operator()(display *d){ display_done(d); } };
using unique_disp = std::unique_ptr<struct display, disp_deleter>;
}

struct state_multiplier_common {
        std::vector<unique_disp> displays;

        struct video_desc display_desc;

        queue<struct video_frame *> incoming_queue;
        condition_variable in_queue_decremented_cv;

        mutex lock;
        condition_variable cv;

        struct module *parent;
        thread worker_thread;
};

struct state_multiplier {
        shared_ptr<struct state_multiplier_common> common;
        struct video_desc desc;
};

static void show_help(){
        printf("Multiplier display\n");
        printf("Usage:\n");
        printf("\t-d multiplier:<display1>[:<display_config1>][#<display2>[:<display_config2>]]...\n");
}

static void *display_multiplier_init(struct module *parent, const char *fmt, unsigned int flags)
{
        auto s = std::make_unique<state_multiplier>();

        if(!fmt || strlen(fmt) == 0){
                show_help();
                return &display_init_noerr;
        }

        if (isdigit(fmt[0])) { // fork
                struct state_multiplier *orig;
                sscanf(fmt, "%p", &orig);
                s->common = orig->common;
                return s.release();
        }

        std::string_view fmt_sv = fmt;

        if (fmt_sv == "help") { 
                show_help();
                return &display_init_noerr;
        }

        s->common = std::make_shared<state_multiplier_common>();
        s->common->parent = parent;

        for(auto tok = tokenize(fmt_sv, '#'); !tok.empty(); tok = tokenize(fmt_sv, '#')){
                LOG(LOG_LEVEL_VERBOSE) << MOD_NAME << "Initializing display " << tok << "\n";
                auto display = std::string(tokenize(tok, ':'));
                auto cfg = std::string(tok);

                struct display *d_ptr;
                if (initialize_video_display(parent, display.c_str(), cfg.c_str(), flags, NULL, &d_ptr) != 0) {
                        LOG(LOG_LEVEL_FATAL) << "[multiplier] Unable to initialize a display " << display << "!\n";
                        abort();
                }
                unique_disp disp(d_ptr);
                if (display_needs_mainloop(disp.get()) && !s->common->displays.empty()) {
                        LOG(LOG_LEVEL_ERROR) << "[multiplier] Display " << display << " needs mainloop and should be given first!\n";
                }

                s->common->displays.push_back(std::move(disp));
        }

        return s.release();
}

static void check_reconf(struct state_multiplier_common *s, struct video_desc desc)
{
        if (!video_desc_eq(desc, s->display_desc)) {
                s->display_desc = desc;
                fprintf(stderr, "RECONFIGURED\n");
                for(auto& disp : s->displays){
                        display_reconfigure(disp.get(), s->display_desc, VIDEO_NORMAL);
                }
        }
}

static void display_multiplier_worker(void *state)
{
        shared_ptr<struct state_multiplier_common> s = ((struct state_multiplier *)state)->common;
        int skipped = 0;

        while (1) {
                struct video_frame *frame;
                {
                        unique_lock<mutex> lg(s->lock);
                        s->cv.wait(lg, [s]{return s->incoming_queue.size() > 0;});
                        frame = s->incoming_queue.front();
                        s->incoming_queue.pop();
                        s->in_queue_decremented_cv.notify_one();
                }

                if (!frame) {
                        for (auto& disp : s->displays) {
                                display_put_frame(disp.get(), NULL, PUTF_BLOCKING);
                        }
                        break;
                }

                if (skipped < SKIP_FIRST_N_FRAMES_IN_STREAM){
                        skipped++;
                        vf_free(frame);
                        continue;
                }

                check_reconf(s.get(), video_desc_from_frame(frame));

                for (auto& disp : s->displays) {
                        struct video_frame *real_display_frame = display_get_frame(disp.get());
                        memcpy(real_display_frame->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
                        display_put_frame(disp.get(), real_display_frame, PUTF_BLOCKING);
                }

                vf_free(frame);
        }
}

/**
 * Multiplier main loop
 *
 * Runs threads for all slave displays except the first one and an additional
 * one for this display. For the first display given on command-line it then
 * switches to its run-loop. This allows a flawless run on macOS where a GUI
 * worker (GL/SDL) needs to be run in the main thread to work properly.
 */
static void display_multiplier_run(void *state)
{
        shared_ptr<struct state_multiplier_common> s = ((struct state_multiplier *)state)->common;

        assert(!s->displays.empty());

        for (size_t i = 1; i < s->displays.size(); i++) {
                display_run_new_thread(s->displays[i].get());
        }

        s->worker_thread = thread(display_multiplier_worker, state);

        display_run_this_thread(s->displays[0].get());

        s->worker_thread.join();
        for (size_t i = 1; i < s->displays.size(); i++) {
                display_join(s->displays[i].get());
        }
}

static void display_multiplier_done(void *state)
{
        struct state_multiplier *s = (struct state_multiplier *)state;
        delete s;
}

static struct video_frame *display_multiplier_getf(void *state)
{
        struct state_multiplier *s = (struct state_multiplier *)state;

        return vf_alloc_desc_data(s->desc);
}

static int display_multiplier_putf(void *state, struct video_frame *frame, int flags)
{
        shared_ptr<struct state_multiplier_common> s = ((struct state_multiplier *)state)->common;

        if (flags == PUTF_DISCARD) {
                vf_free(frame);
        } else {
                unique_lock<mutex> lg(s->lock);
                if (s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        fprintf(stderr, "Multiplier: queue full!\n");
                }
                if (flags == PUTF_NONBLOCK && s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        vf_free(frame);
                        return 1;
                }
                s->in_queue_decremented_cv.wait(lg, [s]{return s->incoming_queue.size() < IN_QUEUE_MAX_BUFFER_LEN;});
                s->incoming_queue.push(frame);
                lg.unlock();
                s->cv.notify_one();
        }

        return 0;
}

static int display_multiplier_get_property(void *state, int property, void *val, size_t *len)
{
        //TODO Figure out forking, for now just disable multi. sources
        shared_ptr<struct state_multiplier_common> s = ((struct state_multiplier *)state)->common;
        if (property == DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES) {
#if 0
                ((struct multi_sources_supp_info *) val)->val = true;
                ((struct multi_sources_supp_info *) val)->fork_display = display_multiplier_fork;
                ((struct multi_sources_supp_info *) val)->state = state;
                *len = sizeof(struct multi_sources_supp_info);
                return TRUE;
#endif
                return FALSE;

        }
        //TODO Find common properties, for now just return properties of the first display
        return display_ctl_property(s->displays[0].get(), property, val, len);
}

static int display_multiplier_reconfigure(void *state, struct video_desc desc)
{
        struct state_multiplier *s = (struct state_multiplier *) state;

        s->desc = desc;

        return 1;
}

static void display_multiplier_put_audio_frame(void *state, const struct audio_frame *frame)
{
        auto *s = static_cast<struct state_multiplier *>(state);

        display_put_audio_frame(s->common->displays.at(0).get(), frame);
}

static int display_multiplier_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        auto *s = static_cast<struct state_multiplier *>(state);

        return display_reconfigure_audio(s->common->displays.at(0).get(), quant_samples, channels, sample_rate);
}

static auto display_multiplier_needs_mainloop(void *state)
{
        auto s = static_cast<struct state_multiplier *>(state)->common;
        return !s->displays.empty() && display_needs_mainloop(s->displays[0].get());
}

static void display_multiplier_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = nullptr;
        *count = 0;
}

static const struct video_display_info display_multiplier_info = {
        display_multiplier_probe,
        display_multiplier_init,
        display_multiplier_run,
        display_multiplier_done,
        display_multiplier_getf,
        display_multiplier_putf,
        display_multiplier_reconfigure,
        display_multiplier_get_property,
        display_multiplier_put_audio_frame,
        display_multiplier_reconfigure_audio,
        display_multiplier_needs_mainloop,
        false,
};

REGISTER_HIDDEN_MODULE(multiplier, &display_multiplier_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

