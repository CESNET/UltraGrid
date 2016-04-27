/**
 * @file   video_display/dump.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016 CESNET, z. s. p. o.
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
/**
 * @todo Add audio support
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "video_export.h"

#include <chrono>

using namespace std;
using namespace std::chrono;

struct dump_display_state {
        dump_display_state()
        {
                string dirname;
                time_t now = time(NULL);
                dirname = "dump." + to_string(now);
                if (platform_mkdir(dirname.c_str()) == -1) {
                        perror("mkdir");
                        throw string("Unable to create directory!");
                }
                e = video_export_init(dirname.c_str());
        }
        ~dump_display_state() {
                vf_free(f);
                video_export_destroy(e);
        }
        struct video_frame *f = nullptr;
        steady_clock::time_point t0 = steady_clock::now();
        int frames = 0;
        struct video_export *e;
        size_t max_tile_data_len = 0;
};

static void *display_dump_init(struct module *, const char *, unsigned int)
{
        return new dump_display_state();
}

static void display_dump_run(void *)
{
}

static void display_dump_done(void *state)
{
        auto s = (dump_display_state *) state;

        delete s;
}

static struct video_frame *display_dump_getf(void *state)
{
        auto s = (dump_display_state *) state;
        for (unsigned int i = 0; i < s->f->tile_count; ++i) {
                s->f->tiles[i].data_len = s->max_tile_data_len;
        }
        return s->f;
}

static int display_dump_putf(void *state, struct video_frame *frame, int)
{
        auto s = (dump_display_state *) state;
        auto curr_time = steady_clock::now();
        s->frames += 1;
        double seconds = duration_cast<duration<double>>(curr_time - s->t0).count();
        if (seconds >= 5.0) {
                double fps = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[dump] %d frames in %g seconds = %g FPS\n",
                                s->frames, seconds, fps);
                s->t0 = curr_time;
                s->frames = 0;
        }

        if (frame) {
                video_export(s->e, frame);
        }

        return 0;
}

static int display_dump_get_property(void *, int property, void *val, size_t *len)
{
        codec_t codecs[VIDEO_CODEC_COUNT - 1];

        for (int i = 0; i < VIDEO_CODEC_COUNT - 1; ++i) {
                codecs[i] = (codec_t) (i + 1); // all codecs (exclude VIDEO_CODEC_NONE)
        }

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                                *len = sizeof(codecs);
                        } else {
                                return FALSE;
                        }
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        *len = sizeof(int);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_dump_reconfigure(void *state, struct video_desc desc)
{
        dump_display_state *s = (dump_display_state *) state;
        vf_free(s->f);
        s->f = vf_alloc_desc(desc);
        s->max_tile_data_len = 4 * desc.width * desc.height;
        for (unsigned int i = 0; i < s->f->tile_count; ++i) {
                s->f->tiles[i].data_len = s->max_tile_data_len;
                s->f->tiles[i].data = (char *) malloc(s->f->tiles[i].data_len);
        }
        s->f->data_deleter = vf_data_deleter;

        return TRUE;
}

static void display_dump_put_audio_frame(void *, struct audio_frame *)
{
}

static int display_dump_reconfigure_audio(void *, int, int, int)
{
        return FALSE;
}

static const struct video_display_info display_dump_info = {
        [](struct device_info **available_cards, int *count) {
                *available_cards = nullptr;
                *count = 0;
        },
        display_dump_init,
        display_dump_run,
        display_dump_done,
        display_dump_getf,
        display_dump_putf,
        display_dump_reconfigure,
        display_dump_get_property,
        display_dump_put_audio_frame,
        display_dump_reconfigure_audio,
};

REGISTER_MODULE(dump, &display_dump_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

