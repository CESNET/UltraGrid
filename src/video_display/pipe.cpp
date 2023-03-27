/**
 * @file   video_display/pipe.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2023 CESNET, z. s. p. o.
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

#include <iostream>
#include <list>
#include <mutex>

#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "video_display/pipe.hpp"

using std::cout;
using std::list;
using std::mutex;
using std::lock_guard;

struct state_pipe {
        struct module *parent;
        frame_recv_delegate *delegate;
        codec_t decode_to;
        struct video_desc desc{};
        list<struct audio_frame *> audio_frames{};
        mutex audio_lock{};
};

static struct display *display_pipe_fork(void *state)
{
        struct state_pipe *s = (struct state_pipe *) state;
        char fmt[2 + sizeof(void *) * 2 + 1] = "";
        struct display *out;

        snprintf(fmt, sizeof fmt, "%p", s->delegate);
        int rc = initialize_video_display(s->parent,
                "pipe", fmt, 0, NULL, &out);
        if (rc == 0) return out; else return NULL;
}

static void display_pipe_usage() {
        cout << "Usage:\n"
                "\t-d pipe:<ptr>[:codec=<c>]\n";
}

/**
 * @note
 * Audio is always received regardless if enabled in flags.
 */
static void *display_pipe_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        codec_t decode_to = UYVY;
        frame_recv_delegate *delegate;

        if (!fmt || strlen(fmt) == 0 || strcmp(fmt, "help") == 0) {
                fprintf(stderr, "Pipe dummy video driver. For internal usage - please do not use.\n");
                if (fmt != nullptr && strcmp(fmt, "help") == 0) {
                        display_pipe_usage();
                }
                return nullptr;
        }

        sscanf(fmt, "%p", &delegate);
        if (strchr(fmt, ':') != nullptr) {
                fmt = strchr(fmt, ':') + 1;
                if (strstr(fmt, "codec=") == fmt) {
                        const char *codec_name = fmt + strlen("codec=");
                        decode_to = get_codec_from_name(codec_name);
                        if (decode_to == VIDEO_CODEC_NONE) {
                                LOG(LOG_LEVEL_ERROR) << "Wrong codec name: " << codec_name << "\n";
                                return nullptr;
                        }
                } else {
                        display_pipe_usage();
                        return nullptr;
                }
        }

        auto *s = new state_pipe{parent, delegate, decode_to};

        return s;
}

static void display_pipe_done(void *state)
{
        struct state_pipe *s = (struct state_pipe *)state;

        for (auto & a : s->audio_frames) {
                free(a->data);
                free(a);
        }
        delete s;
}

static struct video_frame *display_pipe_getf(void *state)
{
        struct state_pipe *s = (struct state_pipe *)state;

        struct video_frame *out = vf_alloc_desc_data(s->desc);
        // explicit dispose is needed because we do not process the frame
        // by ourselves but it is passed to further processing
        out->callbacks.dispose = vf_free;
        return out;
}

static void display_pipe_dispose_audio(struct audio_frame *f) {
        free(f->data);
        free(f);
}

static struct audio_frame * display_pipe_get_audio(struct state_pipe *s)
{
        lock_guard<mutex> lk(s->audio_lock);
        size_t len = 0;
        if (s->audio_frames.empty()) {
                return nullptr;
        }
        struct audio_desc desc = audio_desc_from_frame(s->audio_frames.front());
        for (auto it = s->audio_frames.begin(); it != s->audio_frames.end(); ) {
                if (!audio_desc_eq(desc, audio_desc_from_frame(*it))) {
                        LOG(LOG_LEVEL_WARNING) << "[pipe] Discarding audio - incompatible format!\n";
                        free((*it)->data);
                        free(*it);
                        it = s->audio_frames.erase(it);
                        continue;
                }
                len += (*it)->data_len;
                ++it;
        }
        if (len == 0) {
                return nullptr;
        }
        auto out = (struct audio_frame *) calloc(1, sizeof(struct audio_frame));
        audio_frame_write_desc(out, desc);
        out->max_size = len;
        out->data = (char *) malloc(len);
        for (auto it = s->audio_frames.begin(); it != s->audio_frames.end(); ) {
                append_audio_frame(out, (*it)->data, (*it)->data_len);
                free((*it)->data);
                free(*it);
                it = s->audio_frames.erase(it);
        }
        out->dispose = display_pipe_dispose_audio;
        return out;
}

static int display_pipe_putf(void *state, struct video_frame *frame, long long flags)
{
        struct state_pipe *s = (struct state_pipe *) state;

        if (flags == PUTF_DISCARD) {
                VIDEO_FRAME_DISPOSE(frame);
                return TRUE;
        }

        struct audio_frame *af = display_pipe_get_audio(s);
        s->delegate->frame_arrived(frame, af);

        return TRUE;
}

static int display_pipe_get_property(void *state, int property, void *val, size_t *len)
{
        auto *s = static_cast<struct state_pipe *>(state);
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codec_t) > *len) {
                                return FALSE;
                        }
                        memcpy(val, &s->decode_to, sizeof(s->decode_to));
                        *len = sizeof s->decode_to;
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                case DISPLAY_PROPERTY_VIDEO_MODE:
                        *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTS_MULTI_SOURCES:
                        ((struct multi_sources_supp_info *) val)->val = false;
                        ((struct multi_sources_supp_info *) val)->fork_display = display_pipe_fork;
                        ((struct multi_sources_supp_info *) val)->state = state;
                        *len = sizeof(struct multi_sources_supp_info);
                        break;
                case DISPLAY_PROPERTY_AUDIO_FORMAT:
                        {
                                assert (*len >= sizeof(struct audio_desc));
                                struct audio_desc *desc = (struct audio_desc *) val;
                                desc->codec = AC_PCM; // decompress
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_pipe_reconfigure(void *state, struct video_desc desc)
{
        struct state_pipe *s = (struct state_pipe *) state;

        s->desc = desc;

        return 1;
}

static void display_pipe_put_audio_frame(void *state, const struct audio_frame *frame)
{
        auto s = (struct state_pipe *) state;
        lock_guard<mutex> lk(s->audio_lock);
        s->audio_frames.push_back(audio_frame_copy(frame, false));
}

static int display_pipe_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return TRUE;
}

static void display_pipe_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *available_cards = nullptr;
        *count = 0;
}

static const struct video_display_info display_pipe_info = {
        display_pipe_probe,
        display_pipe_init,
        NULL, // _run
        display_pipe_done,
        display_pipe_getf,
        display_pipe_putf,
        display_pipe_reconfigure,
        display_pipe_get_property,
        display_pipe_put_audio_frame,
        display_pipe_reconfigure_audio,
        DISPLAY_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_HIDDEN_MODULE(pipe, &display_pipe_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

