/**
 * @file   video_display/dummy.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2015 CESNET, z. s. p. o.
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

#include <chrono>

using namespace std;
using namespace std::chrono;

struct dummy_display_state {
        dummy_display_state() : f(nullptr), t0(steady_clock::now()), frames(0) {}
        struct video_frame *f;
        steady_clock::time_point t0;
        int frames;
};

static void *display_dummy_init(struct module *, const char *, unsigned int)
{
        return new dummy_display_state();
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
                log_msg(LOG_LEVEL_INFO, "[dummy] %d frames in %g seconds = %g FPS\n",
                                s->frames, seconds, fps);
                s->t0 = curr_time;
                s->frames = 0;
        }

        return 0;
}

static int display_dummy_get_property(void *, int property, void *val, size_t *len)
{
        codec_t codecs[] = {UYVY, YUYV, v210, R12L, RGBA, RGB, BGR};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }
                        
                        *len = sizeof(codecs);
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
        return FALSE;
}

static const struct video_display_info display_dummy_info = {
        [](struct device_info **available_cards, int *count) {
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
};

REGISTER_MODULE(dummy, &display_dummy_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

