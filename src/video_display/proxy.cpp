/**
 * @file   video_display/proxy.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @brief  This is an umbrella header for video functions.
 */
/*
 * Copyright (c) 2014 CESNET, z. s. p. o.
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
#include "video.h"
#include "video_display.h"
#include "video_display/proxy.h"

#include <condition_variable>
#include <chrono>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <unordered_map>

using namespace std;

static constexpr int TRANSITION_COUNT = 10;
static constexpr int BUFFER_LEN = 5;
static constexpr chrono::milliseconds SOURCE_TIMEOUT(500);
static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;

struct state_proxy {
        struct display *real_display;
        struct video_desc desc;
        struct video_desc display_desc;

        uint32_t current_ssrc;
        uint32_t old_ssrc;

        int transition;

        queue<struct video_frame *> incoming_queue;
        condition_variable in_queue_decremented_cv;
        map<uint32_t, list<struct video_frame *> > frames;
        unordered_map<uint32_t, chrono::system_clock::time_point> disabled_ssrc;

        pthread_t thread_id;

        mutex lock;
        condition_variable cv;
};

void *display_proxy_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(parent);
        struct state_proxy *s;
        char *fmt_copy = NULL;
        const char *requested_display = "gl";
        const char *cfg = NULL;

        s = new state_proxy();

        if (fmt && strlen(fmt) > 0) {
                fmt_copy = strdup(fmt);
                requested_display = fmt_copy;
                char *delim = strchr(fmt_copy, ':');
                if (delim) {
                        *delim = '\0';
                        cfg = delim + 1;
                }
        }
        assert (initialize_video_display(parent, requested_display, cfg, flags, &s->real_display) == 0);
        free(fmt_copy);

        pthread_create(&s->thread_id, NULL, (void *(*)(void *)) display_run,
                        s->real_display);

        return s;
}

void display_proxy_run(void *state)
{
        struct state_proxy *s = (struct state_proxy *)state;
        bool prefill = false;

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
                        display_put_frame(s->real_display, NULL, PUTF_BLOCKING);
                        break;
                }

                chrono::system_clock::time_point now = chrono::system_clock::now();
                auto it = s->disabled_ssrc.find(frame->ssrc);
                if (it != s->disabled_ssrc.end()) {
                        it->second = now;
                        vf_free(frame);
                        continue;
                }

                it = s->disabled_ssrc.begin();
                while (it != s->disabled_ssrc.end()) {
                        if (chrono::duration_cast<chrono::milliseconds>(now - it->second) > SOURCE_TIMEOUT) {
                                verbose_msg("Source 0x%08lx timeout. Deleting from proxy display.\n", it->first);
                                s->disabled_ssrc.erase(it++);
                        } else {
                                ++it;
                        }
                }

                if (frame->ssrc != s->current_ssrc && frame->ssrc != s->old_ssrc) {
                        s->old_ssrc = s->current_ssrc; // if != 0, we will be in transition state
                        s->current_ssrc = frame->ssrc;
                        prefill = true;
                }

                /// @todo....
                if (frame->tiles[0].data == NULL) {
                        if (!video_desc_eq(video_desc_from_frame(frame), s->display_desc)) {
                                s->display_desc = video_desc_from_frame(frame);
                                fprintf(stderr, "RECONFIGURED\n");
                                display_reconfigure(s->real_display, s->display_desc);
                        }
                        continue;
                }

                s->frames[frame->ssrc].push_back(frame);

                if (s->frames[s->current_ssrc].size() >= BUFFER_LEN) {
                        prefill = false;
                }

                // we may receive two streams concurrently, therefore we use the timing according to
                // the later one
                if (frame->ssrc != s->current_ssrc) {
                        continue;
                }

                if (s->old_ssrc != 0) {
                        if (prefill) {
                                auto & ssrc_list = s->frames[s->old_ssrc];
                                if (ssrc_list.empty()) {
                                        fprintf(stderr, "SMOLIK1!\n");
                                } else {
                                        frame = ssrc_list.front();
                                        ssrc_list.pop_front();
                                        struct video_frame *real_display_frame = display_get_frame(s->real_display);
                                        memcpy(real_display_frame->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
                                        vf_free(frame);
                                        display_put_frame(s->real_display, real_display_frame, PUTF_BLOCKING);
                                }
                        } else {
                                auto & old_list = s->frames[s->old_ssrc];
                                auto & new_list = s->frames[s->current_ssrc];

                                s->transition += 1;

                                if (old_list.empty() || new_list.empty()) {
                                        fprintf(stderr, "SMOLIK2!\n");
                                        // ok here, we do not have nothing to mix, therefore cancel smooth transition
                                        s->transition = TRANSITION_COUNT;
                                } else {
                                        struct video_frame *old_frame, *new_frame;

                                        old_frame = old_list.front();
                                        old_list.pop_front();
                                        new_frame = new_list.front();
                                        new_list.pop_front();

                                        struct video_frame *real_display_frame = display_get_frame(s->real_display);
                                        for (unsigned int i = 0; i < new_frame->tiles[0].data_len; ++i) {
                                                int old_val = ((unsigned char *) old_frame->tiles[0].data)[i];
                                                int new_val = ((unsigned char *) new_frame->tiles[0].data)[i];
                                                ((unsigned char *) real_display_frame->tiles[0].data)[i] =
                                                        ((new_val * s->transition) + (old_val * (TRANSITION_COUNT - s->transition))) / TRANSITION_COUNT;
                                        }
                                        vf_free(old_frame);
                                        vf_free(new_frame);
                                        display_put_frame(s->real_display, real_display_frame, PUTF_BLOCKING);
                                }
                        }
                } else {
                        if (!prefill) {
                                if (s->frames[s->current_ssrc].empty()) {
                                        // this should not happen, heh ?
                                        fprintf(stderr, "SMOLIK3!\n");
                                } else {
                                        frame = s->frames[s->current_ssrc].front();
                                        s->frames[s->current_ssrc].pop_front();

                                        struct video_frame *real_display_frame = display_get_frame(s->real_display);
                                        memcpy(real_display_frame->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
                                        vf_free(frame);
                                        display_put_frame(s->real_display, real_display_frame, PUTF_BLOCKING);
                                }
                        }
                }

                if (s->old_ssrc != 0 && s->transition >= TRANSITION_COUNT) {
                        for (auto && frame : s->frames[s->old_ssrc]) {
                                vf_free(frame);
                        }

                        s->frames.erase(s->old_ssrc);

                        s->disabled_ssrc[s->old_ssrc] = chrono::system_clock::now();
                        s->old_ssrc = 0u;
                        s->transition = 0;
                }
        }

        pthread_join(s->thread_id, NULL);
}

void display_proxy_done(void *state)
{
        struct state_proxy *s = (struct state_proxy *)state;
        display_done(s->real_display);

        for (auto && ssrc_map : s->frames) {
                for (auto && frame : ssrc_map.second) {
                        vf_free(frame);
                }
        }

        delete s;
}

struct video_frame *display_proxy_getf(void *state)
{
        struct state_proxy *s = (struct state_proxy *)state;

        unique_lock<mutex> lg(s->lock);
        return vf_alloc_desc_data(s->desc);
}

int display_proxy_putf(void *state, struct video_frame *frame, int flags)
{
        struct state_proxy *s = (struct state_proxy *) state;

        if (flags == PUTF_DISCARD) {
                vf_free(frame);
        } else {
                unique_lock<mutex> lg(s->lock);
                if (s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        fprintf(stderr, "Proxy: queue full!\n");
                }
                if (flags == PUTF_NONBLOCK && s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN) {
                        return 1;
                }
                s->in_queue_decremented_cv.wait(lg, [s]{return s->incoming_queue.size() < IN_QUEUE_MAX_BUFFER_LEN;});
                s->incoming_queue.push(frame);
                lg.unlock();
                s->cv.notify_one();
        }

        return 0;
}

display_type_t *display_proxy_probe(void)
{
        display_type_t *dt;

        dt = (display_type_t *) calloc(1, sizeof(display_type_t));
        if (dt != NULL) {
                dt->id = DISPLAY_PROXY_ID;
                dt->name = "proxy";
                dt->description = "No display device";
        }
        return dt;
}

int display_proxy_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_proxy *s = (struct state_proxy *)state;
        if (property == DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES) {
                *(int *) val = TRUE;
                *len = sizeof(int);
                return TRUE;

        } else {
                return display_get_property(s->real_display, property, val, len);
        }
}

int display_proxy_reconfigure(void *state, struct video_desc desc)
{
        /**
         * @todo this is wrong, because reconfigure will be used from multiple threads...
         */
        struct state_proxy *s = (struct state_proxy *) state;

        unique_lock<mutex> lg(s->lock);
        s->desc = desc;
        s->in_queue_decremented_cv.wait(lg, [s]{return s->incoming_queue.size() < IN_QUEUE_MAX_BUFFER_LEN;});
        s->incoming_queue.push(vf_alloc_desc(desc));
        lg.unlock();
        s->cv.notify_one();

        return 1;
}

void display_proxy_put_audio_frame(void *state, struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

int display_proxy_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

