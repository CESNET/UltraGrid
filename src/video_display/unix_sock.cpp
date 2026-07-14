/**
 * @file   video_display/unix_sock.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2022-2026 CESNET, zájmové sdružení právických osob
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

#include "debug.h"
#include "lib_common.h"

#include <condition_variable>
#include <vector>
#include <memory>
#include <mutex>
#include <queue>
#include <cmath>
#include <thread>

#include "host.h"
#include "video_display.h"
#include "video_frame.h"
#include "types.h"
#include "utils/fs.h"
#include "utils/misc.h"
#include "utils/macros.h"
#include "utils/color_out.h"
#include "utils/string_view_utils.hpp"
#include "../tools/ipc_frame.h"
#include "../tools/ipc_frame_ug.h"
#include "../tools/ipc_frame_unix.h"

#define MOD_NAME "[unix sock disp] "

#define DEFAULT_PREVIEW_FILENAME "ug_preview_disp_unix"
#define DEFAULT_DISP_FILENAME "ug_unix"

#define DEFAULT_SCALE_W 960
#define DEFAULT_SCALE_H 540

using ipc_frame_conv_func_t = bool (*)(Ipc_frame *dst,
                const video_frame *src,
                codec_t codec,
                unsigned scale_factor);

static constexpr unsigned int IN_QUEUE_MAX_BUFFER_LEN = 5;
static constexpr int SKIP_FIRST_N_FRAMES_IN_STREAM = 5;

namespace{
using unique_frame = std::unique_ptr<video_frame, deleter_from_fcn<vf_free>>;

struct state_unix_sock{
        std::queue<unique_frame> incoming_queue;
        std::condition_variable frame_consumed_cv;
        std::condition_variable frame_available_cv;
        std::mutex lock;

        video_desc desc = {};

        Ipc_frame_uniq ipc_frame;
        Ipc_frame_writer_uniq frame_writer;

        int target_width = -1;
        int target_height = -1;

        bool ignore_putf_blocking = false;

        ipc_frame_conv_func_t ipc_conv = ipc_frame_from_ug_frame;

        module *parent = nullptr;
        std::thread worker_thread;
};

void show_help(){
        col() << "unix_socket/preview display. The two display are identical apart from their defaults and the fact that preview never blocks on putf().\n";
        col() << "usage:\n";
        col() << TBOLD(TRED("\t-d (unix_socket|preview)") << "[:path=<path>][:target_size=<w>x<h>]")
                << "\n\n";
        col() << "options:\n";
        col() << TBOLD("\tpath=<path>")           << "\tpath to unix socket to connect to. Defaults are \""
                << get_temp_dir() << DEFAULT_PREVIEW_FILENAME "\" for preview and \""
                << get_temp_dir() << DEFAULT_DISP_FILENAME "\" for unix_sock\n";
        col() << TBOLD("\ttarget_size=<w>x<h>")<< "\tScales the video frame so that the total number of pixel is around <w>x<h>. If -1x-1 is passed, no scaling takes place."
                << "Defaults are -1x-1 for unix_sock and " TOSTRING(DEFAULT_SCALE_W) "x" TOSTRING(DEFAULT_SCALE_H) " for preview.\n";
        col() << TBOLD("\thq")           << "\tUse higher quality downscale\n";
}

void display_unix_sock_run(void *state);

void *display_unix_sock_init(module *parent,
                const char *fmt,
                unsigned int flags,
                bool is_preview)
{
        UNUSED(flags);
        UNUSED(fmt);

        auto s = std::make_unique<state_unix_sock>();
        std::string_view fmt_sv = fmt ? fmt : "";

        std::string socket_path = get_temp_dir();
        socket_path += DEFAULT_DISP_FILENAME;

        if(is_preview){
                socket_path = get_temp_dir();
                socket_path += DEFAULT_PREVIEW_FILENAME;
                s->target_width = DEFAULT_SCALE_W;
                s->target_height = DEFAULT_SCALE_H;
                s->ignore_putf_blocking = true;
        }

        while(!fmt_sv.empty()){
                auto tok = tokenize(fmt_sv, ':', '"');
                auto key = tokenize(tok, '=');

                if(key == "help"){
                        show_help();
                        return INIT_NOERR;
                } else if(key == "path"){
                        socket_path = tokenize(tok, '=');
                } else if(key == "key"){
                        socket_path += tokenize(tok, '=');
                } else if(key == "hq"){
                        s->ipc_conv = ipc_frame_from_ug_frame_hq;
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
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to init ipc writer for path %s\n", socket_path.c_str());
                return nullptr;
        }

        s->worker_thread = std::thread(display_unix_sock_run, s.get());
        return s.release();
}

void display_unix_sock_run(void *state){
        auto s = static_cast<state_unix_sock *>(state);
        int skipped = 0;

        while(true){
                std::unique_lock<std::mutex> l(s->lock);
                s->frame_available_cv.wait(l, [s]{ return !s->incoming_queue.empty(); });
                auto frame = std::move(s->incoming_queue.front());
                s->incoming_queue.pop();
                l.unlock();
                s->frame_consumed_cv.notify_one();

                if(!frame){
                        break;
                }

                if(skipped < SKIP_FIRST_N_FRAMES_IN_STREAM){
                        skipped++;
                        continue;
                }

                const tile *tile = &frame->tiles[0];

                int scale = ipc_frame_get_scale_factor(tile->width, tile->height,
                        s->target_width, s->target_height);

                if(!s->ipc_conv(s->ipc_frame.get(), frame.get(),
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
        }
}

void display_unix_sock_done(void *state){
        auto s = static_cast<state_unix_sock *>(state);

        if(s->worker_thread.joinable()){
                s->worker_thread.join();
        }

        delete s;
}

video_frame *display_unix_sock_getf(void *state){
        auto s = static_cast<state_unix_sock *>(state);

        return vf_alloc_desc_data(s->desc);
}

bool display_unix_sock_putf(void *state, video_frame *frame, long long flags){
        auto s = static_cast<state_unix_sock *>(state);
        auto f = unique_frame(frame);

        if(flags == PUTF_DISCARD)
                return true;

        std::unique_lock<std::mutex> lg(s->lock);
        if(s->incoming_queue.size() >= IN_QUEUE_MAX_BUFFER_LEN){
                if(flags != PUTF_BLOCKING){
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "queue full!\n");
                        return false;
                }
                if(s->ignore_putf_blocking){
                        return false;
                }
        }

        s->frame_consumed_cv.wait(lg, [s]{ return s->incoming_queue.size() < IN_QUEUE_MAX_BUFFER_LEN; });
        s->incoming_queue.push(std::move(f));
        lg.unlock();
        s->frame_available_cv.notify_one();

        return true;
}

bool display_unix_sock_get_property(void *state, int property, void *val, size_t *len){
        UNUSED(state);
        codec_t codecs[] = {UYVY, RGBA, RGB};
        interlacing_t supported_il_modes[] = {PROGRESSIVE, INTERLACED_MERGED, SEGMENTED_FRAME};
        int rgb_shift[] = {0, 8, 16};

        switch(property){
        case DISPLAY_PROPERTY_CODECS:
                if(sizeof(codecs) <= *len){
                        memcpy(val, codecs, sizeof(codecs));
                } else{
                        return false;
                }

                *len = sizeof(codecs);
                break;
        case DISPLAY_PROPERTY_RGB_SHIFT:
                if(sizeof(rgb_shift) > *len){
                        return false;
                }
                memcpy(val, rgb_shift, sizeof(rgb_shift));
                *len = sizeof(rgb_shift);
                break;
        case DISPLAY_PROPERTY_BUF_PITCH:
                *static_cast<int *>(val) = PITCH_DEFAULT;
                *len = sizeof(int);
                break;
        case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                if(sizeof(supported_il_modes) <= *len){
                        memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                } else{
                        return false;
                }
                *len = sizeof(supported_il_modes);
                break;
        default:
                return false;
        }
        return true;
}

bool display_unix_sock_reconfigure(void *state, video_desc desc){
        auto s = static_cast<state_unix_sock *>(state);

        s->desc = desc;

        return true;
}

void display_unix_sock_probe(device_info **available_cards, int *count, void (**deleter)(void *)){
        UNUSED(deleter);
        *available_cards = nullptr;
        *count = 0;
}

void *display_unix_sock_init_preview(module *parent, const char *fmt, unsigned int flags){
        return display_unix_sock_init(parent, fmt, flags, true);
}

void *display_unix_sock_init_no_preview(module *parent, const char *fmt, unsigned int flags){
        return display_unix_sock_init(parent, fmt, flags, false);
}
} //anon namespace

constexpr video_display_info display_unix_sock_info = {
        display_unix_sock_probe,
        display_unix_sock_init_no_preview,
        nullptr, // _run
        display_unix_sock_done,
        display_unix_sock_getf,
        display_unix_sock_putf,
        display_unix_sock_reconfigure,
        display_unix_sock_get_property,
        nullptr,
        nullptr,
        MOD_NAME,
};
REGISTER_HIDDEN_MODULE(unix_sock, &display_unix_sock_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

constexpr video_display_info display_preview_info = []() constexpr  {
        auto info = display_unix_sock_info;
        info.init = display_unix_sock_init_preview;
        info.generic_fps_indicator_prefix = "[unix sock preview] ";
        return info;
}();
REGISTER_HIDDEN_MODULE(preview, &display_preview_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

