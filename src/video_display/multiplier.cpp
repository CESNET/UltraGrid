/**
 * @file   video_display/multiplier.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2015 CESNET, z. s. p. o.
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

#include <condition_variable>
#include <chrono>
#include <list>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>

using namespace std;

static constexpr int BUFFER_LEN = 5;
static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;
static constexpr int SKIP_FIRST_N_FRAMES_IN_STREAM = 5;

struct sub_display {
        struct display *real_display;
        thread disp_thread;
};

struct state_multiplier_common {
        ~state_multiplier_common() {

                for(auto& disp : displays){
                        display_done(disp.real_display);
                }
        }

        std::vector<struct sub_display> displays;

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
        struct state_multiplier *s;
        char *fmt_copy = NULL;
        const char *requested_display = NULL;
        const char *cfg = NULL;

        s = new state_multiplier();

        if (fmt && strlen(fmt) > 0) {
                if (strcmp(fmt, "help") == 0) { 
                    show_help();
                    delete s;
                    return &display_init_noerr;
                }
            
                if (isdigit(fmt[0])) { // fork
                        struct state_multiplier *orig;
                        sscanf(fmt, "%p", &orig);
                        s->common = orig->common;
                        return s;
                } else {
                        fmt_copy = strdup(fmt);
                }
        } else {
                show_help();
                return &display_init_noerr;
        }
        s->common = shared_ptr<state_multiplier_common>(new state_multiplier_common());

        struct sub_display disp;

        char *saveptr;
        for(char *token = strtok_r(fmt_copy, "#", &saveptr); token; token = strtok_r(NULL, "#", &saveptr)){
                requested_display = token;
                printf("%s\n", token);
                cfg = NULL;
                char *delim = strchr(token, ':');
                if (delim) {
                        *delim = '\0';
                        cfg = delim + 1;
                }
                if (initialize_video_display(parent, requested_display, cfg, flags, NULL, &disp.real_display) != 0) {
                        LOG(LOG_LEVEL_FATAL) << "[multiplier] Unable to initialize a display " << requested_display << "!\n";
                        abort();
                }

                s->common->displays.push_back(std::move(disp));
        }
        free(fmt_copy);

        s->common->parent = parent;

        return s;
}

static void check_reconf(struct state_multiplier_common *s, struct video_desc desc)
{
        if (!video_desc_eq(desc, s->display_desc)) {
                s->display_desc = desc;
                fprintf(stderr, "RECONFIGURED\n");
                for(auto& disp : s->displays){
                        display_reconfigure(disp.real_display, s->display_desc, VIDEO_NORMAL);
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
                                display_put_frame(disp.real_display, NULL, PUTF_BLOCKING);
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
                        struct video_frame *real_display_frame = display_get_frame(disp.real_display);
                        memcpy(real_display_frame->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
                        display_put_frame(disp.real_display, real_display_frame, PUTF_BLOCKING);
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

        for (int i = 1; i < (int) s->displays.size(); i++) {
                s->displays[i].disp_thread = thread(display_run, s->displays[i].real_display);
        }

        s->worker_thread = thread(display_multiplier_worker, state);

        // run the displays[0] worker
        if (s->displays.size() > 0) {
                display_run(s->displays[0].real_display);
        }

        s->worker_thread.join();
        for (int i = 1; i < (int) s->displays.size(); i++) {
                s->displays[i].disp_thread.join();
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
        return display_get_property(s->displays[0].real_display, property, val, len);
}

static int display_multiplier_reconfigure(void *state, struct video_desc desc)
{
        struct state_multiplier *s = (struct state_multiplier *) state;

        s->desc = desc;

        return 1;
}

static void display_multiplier_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_multiplier_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_multiplier_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        display_multiplier_init,
        display_multiplier_run,
        display_multiplier_done,
        display_multiplier_getf,
        display_multiplier_putf,
        display_multiplier_reconfigure,
        display_multiplier_get_property,
        display_multiplier_put_audio_frame,
        display_multiplier_reconfigure_audio,
};

REGISTER_MODULE(multiplier, &display_multiplier_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

