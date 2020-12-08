/**
 * @file   video_display/dummy.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015-2020 CESNET, z. s. p. o.
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
#include "rang.hpp"
#include "video.h"
#include "video_display.h"

#include <chrono>
#include <iostream>
#include <vector>

const constexpr char *MOD_NAME = "[dummy] ";

using namespace std;
using namespace std::chrono;
using rang::fg;
using rang::style;

struct dummy_display_state {
        dummy_display_state() : f(nullptr), t0(steady_clock::now()), frames(0) {}
        struct video_frame *f;
        steady_clock::time_point t0;
        vector<codec_t> codecs = {I420, UYVY, YUYV, v210, R12L, RGBA, RGB, BGR};
        vector<int> rgb_shift = {0, 8, 16};
        int frames;
};

static auto display_dummy_init(struct module * /* parent */, const char *cfg, unsigned int /* flags */) -> void *
{
        if ("help"s == cfg) {
                cout << "Usage:\n";
                cout << "\t" << style::bold << fg::red << "-d dummy" << fg::reset << "[:codec=<codec>][:rgb_shift=<r>,<g>,<b>]\n" << style::reset;
                cout << "where\n";
                cout << "\t" << style::bold << "<codec>" << style::reset << "   - force the use of a codec instead of default set\n";
                cout << "\t" << style::bold << "rgb_shift" << style::reset << " - if using output codec RGBA, use specified shifts instead of default (0, 8, 16)\n";
                return static_cast<void *>(&display_init_noerr);
        }
        auto s = make_unique<dummy_display_state>();
        auto *ccpy = static_cast<char *>(alloca(strlen(cfg) + 1));
        strcpy(ccpy, cfg);
        char *item = nullptr;
        char *save_ptr = nullptr;
        while ((item = strtok_r(ccpy, ":", &save_ptr)) != nullptr) {
                if (strstr(item, "codec=") != nullptr) {
                        s->codecs.clear();
                        s->codecs.push_back(get_codec_from_name(item + "codec="s.length()));
                        if (s->codecs[0] == VIDEO_CODEC_NONE) {
                                LOG(LOG_LEVEL_ERROR) << MOD_NAME << "Wrong codec spec!\n";
                                return nullptr;
                        }
                } else if (strstr(item, "rgb_shift=") != nullptr) {
                        item += strlen("rgb_shift=");
                        size_t len;
                        s->rgb_shift[0] = stoi(item, &len);
                        item += len + 1;
                        s->rgb_shift[1] = stoi(item, &len);
                        item += len + 1;
                        s->rgb_shift[2] = stoi(item, &len);
                }
                ccpy = nullptr;
        }

        return static_cast<void *>(s.release());
}

static void display_dummy_run(void *)
{
}

static void display_dummy_done(void *state)
{
        auto s = (dummy_display_state *) state;

        vf_free(s->f);
        delete s;
}

static struct video_frame *display_dummy_getf(void *state)
{
        return ((dummy_display_state *) state)->f;
}

static int display_dummy_putf(void *state, struct video_frame * /* frame */, int flags)
{
        if (flags == PUTF_DISCARD) {
                return 0;
        }
        auto s = (dummy_display_state *) state;
        auto curr_time = steady_clock::now();
        s->frames += 1;
        double seconds = duration_cast<duration<double>>(curr_time - s->t0).count();
        if (seconds >= 5.0) {
                double fps = s->frames / seconds;
                LOG(LOG_LEVEL_INFO) << MOD_NAME << s->frames << " frames in " << seconds << " seconds = " << fps << " FPS\n",
                s->t0 = curr_time;
                s->frames = 0;
        }

        return 0;
}

static auto display_dummy_get_property(void *state, int property, void *val, size_t *len)
{
        auto *s = static_cast<dummy_display_state *>(state);

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (s->codecs.size() * sizeof s->codecs[0] > *len) {
                                return FALSE;
                        }
                        *len = s->codecs.size() * sizeof(codec_t);
                        memcpy(val, s->codecs.data(), *len);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if (s->rgb_shift.size() * sizeof s->rgb_shift[0] > *len) {
                                return FALSE;
                        }
                        *len = s->rgb_shift.size() * sizeof(s->rgb_shift[0]);
                        memcpy(val, s->rgb_shift.data(), *len);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_dummy_reconfigure(void *state, struct video_desc desc)
{
        dummy_display_state *s = (dummy_display_state *) state;
        vf_free(s->f);
        s->f = vf_alloc_desc_data(desc);

        return TRUE;
}

static void display_dummy_put_audio_frame(void *, struct audio_frame *)
{
}

static int display_dummy_reconfigure_audio(void *, int, int, int)
{
        return TRUE;
}

static const struct video_display_info display_dummy_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        display_dummy_init,
        display_dummy_run,
        display_dummy_done,
        display_dummy_getf,
        display_dummy_putf,
        display_dummy_reconfigure,
        display_dummy_get_property,
        display_dummy_put_audio_frame,
        display_dummy_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_MODULE(dummy, &display_dummy_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

