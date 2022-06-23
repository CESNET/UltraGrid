/**
 * @file   video_display/unix_sock.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET, z. s. p. o.
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

#include <condition_variable>
#include <chrono>
#include <list>
#include <map>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <cmath>
#include <chrono>
#include <iostream>

#include "video.h"
#include "video_display.h"
#include "video_codec.h"
#include "utils/misc.h"
#include "utils/macros.h"
#include "utils/color_out.h"
#include "utils/sv_parse_num.hpp"
#include "tools/ipc_frame.h"
#include "tools/ipc_frame_ug.h"
#include "tools/ipc_frame_unix.h"

#define MOD_NAME "[unix sock disp] "

#define DEFAULT_PREVIEW_FILENAME "ug_preview_disp_unix"
#define DEFAULT_DISP_FILENAME "ug_unix"

#define DEFAULT_SCALE_W 960
#define DEFAULT_SCALE_H 540

static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;
static constexpr int SKIP_FIRST_N_FRAMES_IN_STREAM = 5;

namespace{
struct frame_deleter { void operator()(video_frame *f){ vf_free(f); } };
using unique_frame = std::unique_ptr<video_frame, frame_deleter>;
}

struct state_unix_sock {
        std::queue<unique_frame> incoming_queue;
        std::condition_variable frame_consumed_cv;
        std::condition_variable frame_available_cv;
        std::mutex lock;

        struct video_desc desc;
        struct video_desc display_desc;

        Ipc_frame_uniq ipc_frame;
        Ipc_frame_writer_uniq frame_writer;

        int target_width = -1;
        int target_height = -1;

        bool ignore_putf_blocking = false;

        struct module *parent;
};

static void show_help(){
        std::cout << "unix_socket/preview display. The two display are identical apart from their defaults and the fact that preview never blocks on putf().\n";
        std::cout << "usage:\n";
        std::cout << rang::style::bold << rang::fg::red << "\t-d (unix_socket|preview)" << rang::fg::reset << "[:path=<path>][:target_size=<w>x<h>]"
                << "\n\n" << rang::style::reset;
        std::cout << "options:\n";
        std::cout << BOLD("\tpath=<path>")           << "\tpath to unix socket to connect to. Defaults are \""
                << PLATFORM_TMP_DIR DEFAULT_PREVIEW_FILENAME "\" for preview and \""
                << PLATFORM_TMP_DIR DEFAULT_DISP_FILENAME "\" for unix_sock\n";
        std::cout << BOLD("\ttarget_size=<w>x<h>")<< "\tScales the video frame so that the total number of pixel is around <w>x<h>. If -1x-1 is passed, no scaling takes place."
                << "Defaults are -1x-1 for unix_sock and " TOSTRING(DEFAULT_SCALE_W) "x" TOSTRING(DEFAULT_SCALE_H) " for preview.\n";
}

static void *display_unix_sock_init(struct module *parent,
                const char *fmt,
                unsigned int flags,
                bool is_preview)
{
        UNUSED(flags);
        UNUSED(fmt);

        auto s = std::make_unique<state_unix_sock>();

        std::string_view fmt_sv = fmt ? fmt : "";

        std::string socket_path = PLATFORM_TMP_DIR DEFAULT_DISP_FILENAME;

        if(is_preview){
                socket_path = PLATFORM_TMP_DIR DEFAULT_PREVIEW_FILENAME;
                s->target_width = DEFAULT_SCALE_W;
                s->target_height = DEFAULT_SCALE_H;
                s->ignore_putf_blocking = true;
        }

        while(!fmt_sv.empty()){
                auto tok = tokenize(fmt_sv, ':');
                auto key = tokenize(tok, '=');

                if(key == "help"){
                        show_help();
                        return &display_init_noerr;
                } else if(key == "path"){
                        socket_path = tokenize(tok, '=');
                } else if(key == "target_size"){
                        auto val = tokenize(tok, '=');
                        if(!parse_num(tokenize(val, 'x'), s->target_width)
                            || !parse_num(tokenize(val, 'x'), s->target_height))
                        {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to parse resolution\n");
                                return nullptr;
                        }
                }
        }

        s->parent = parent;
        s->ipc_frame.reset(ipc_frame_new());
        s->frame_writer.reset(ipc_frame_writer_new(socket_path.c_str()));
        if(!s->frame_writer){
                return nullptr;
        }

        return s.release();
}

static void display_unix_sock_run(void *state)
{
        auto s = static_cast<state_unix_sock *>(state);
        int skipped = 0;

        using clk = std::chrono::steady_clock;

        clk::duration report_period = std::chrono::seconds(5);
        auto next_report = clk::now() + report_period;
        int frames_sent = 0;

        while (1) {
                auto frame = [&]{
                        std::unique_lock<std::mutex> l(s->lock);
                        s->frame_available_cv.wait(l,
                                        [s]{return s->incoming_queue.size() > 0;});
                        auto frame = std::move(s->incoming_queue.front());
                        s->incoming_queue.pop();
                        s->frame_consumed_cv.notify_one();
                        return frame;
                }();

                if (!frame) {
                        break;
                }

                if (skipped < SKIP_FIRST_N_FRAMES_IN_STREAM){
                        skipped++;
                        continue;
                }

                assert(frame->tile_count == 1);
                const tile *tile = &frame->tiles[0];

                int scale = ipc_frame_get_scale_factor(tile->width, tile->height,
                                s->target_width, s->target_height);

                if(!ipc_frame_from_ug_frame(s->ipc_frame.get(), frame.get(),
                                        RGB, scale))
                {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Unable to convert\n");
                        continue;
                }

                errno = 0;
                if(!ipc_frame_writer_write(s->frame_writer.get(), s->ipc_frame.get())){
                        perror(MOD_NAME "Unable to send frame");
                        continue;
                }
                frames_sent++;

                if(clk::now() > next_report){
                        float seconds = std::chrono::duration_cast<std::chrono::seconds>(report_period).count();
                        log_msg(LOG_LEVEL_NOTICE, MOD_NAME "%d frames in %f seconds (%f fps)\n", frames_sent, seconds, (float) frames_sent / seconds); 
                        frames_sent = 0;
                        next_report = clk::now() + report_period;
                }
        }
}

static void display_unix_sock_done(void *state)
{
        auto s = static_cast<state_unix_sock *>(state);

        delete s;
}

static struct video_frame *display_unix_sock_getf(void *state)
{
        auto s = static_cast<state_unix_sock *>(state);

        return vf_alloc_desc_data(s->desc);
}

static int display_unix_sock_putf(void *state, struct video_frame *frame, int flags)
{
        auto s = static_cast<state_unix_sock *>(state);
        auto f = unique_frame(frame);

        if (flags == PUTF_DISCARD)
                return 0;

        std::unique_lock<std::mutex> lg(s->lock);
        if (s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN){
                if(flags == PUTF_NONBLOCK){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "queue full!\n");
                        return 1;
                }
                if(s->ignore_putf_blocking){
                        return 1;
                }
        }

        s->frame_consumed_cv.wait(lg, [s]{return s->incoming_queue.size() < IN_QUEUE_MAX_BUFFER_LEN;});
        s->incoming_queue.push(std::move(f));
        lg.unlock();
        s->frame_available_cv.notify_one();

        return 0;
}

static int display_unix_sock_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[] = {UYVY, RGBA, RGB};
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }

                        *len = sizeof(codecs);
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
                default:
                        return FALSE;
        }
        return TRUE;
}

static int display_unix_sock_reconfigure(void *state, struct video_desc desc)
{
        auto s = static_cast<state_unix_sock *>(state);

        s->desc = desc;

        return 1;
}

static void display_unix_sock_put_audio_frame(void *state, const struct audio_frame *frame)
{
        UNUSED(state);
        UNUSED(frame);
}

static int display_unix_sock_reconfigure_audio(void *state, int quant_samples, int channels,
                int sample_rate)
{
        UNUSED(state);
        UNUSED(quant_samples);
        UNUSED(channels);
        UNUSED(sample_rate);

        return FALSE;
}

static const struct video_display_info display_unix_sock_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        [](struct module *parent, const char *fmt, unsigned int flags){
                return display_unix_sock_init(parent, fmt, flags, false);
        },
        display_unix_sock_run,
        display_unix_sock_done,
        display_unix_sock_getf,
        display_unix_sock_putf,
        display_unix_sock_reconfigure,
        display_unix_sock_get_property,
        display_unix_sock_put_audio_frame,
        display_unix_sock_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

static const struct video_display_info display_preview_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *available_cards = nullptr;
                *count = 0;
        },
        [](struct module *parent, const char *fmt, unsigned int flags){
                return display_unix_sock_init(parent, fmt, flags, true);
        },
        display_unix_sock_run,
        display_unix_sock_done,
        display_unix_sock_getf,
        display_unix_sock_putf,
        display_unix_sock_reconfigure,
        display_unix_sock_get_property,
        display_unix_sock_put_audio_frame,
        display_unix_sock_reconfigure_audio,
        DISPLAY_DOESNT_NEED_MAINLOOP,
};

REGISTER_HIDDEN_MODULE(unix_sock, &display_unix_sock_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

REGISTER_HIDDEN_MODULE(preview, &display_preview_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

