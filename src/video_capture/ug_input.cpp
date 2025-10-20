/**
 * @file   video_capture/ug_input.cpp
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

#include <cassert>
#include <chrono>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

#include "audio/audio.h"
#include "audio/types.h"
#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "tv.h"
#include "video.h"
#include "utils/color_out.h"
#include "video_capture.h"
#include "video_display.h"
#include "video_display/pipe.hpp" // frame_recv_delegate
#include "video_rxtx.hpp"
#include "video_rxtx/ultragrid_rtp.hpp"

#define MOD_NAME "[ug_input] "
static constexpr int MAX_QUEUE_SIZE = 2;

using std::lock_guard;
using std::map;
using std::mutex;
using std::pair;
using std::queue;
using std::stoi;
using std::string;
using std::thread;
using std::unique_ptr;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::steady_clock;

struct ug_input_state final : public frame_recv_delegate {
        mutex lock;
        queue<pair<struct video_frame *, struct audio_frame *>> frame_queue;
        struct display *display = nullptr;

        void frame_arrived(struct video_frame *f, struct audio_frame *a) override;
        thread         receiver_thread;
        unique_ptr<ultragrid_rtp_video_rxtx> video_rxtx;
        struct state_audio *audio = nullptr;

        ~ug_input_state() override = default;

        std::chrono::steady_clock::time_point t0 = {};
        int frames = 0;

        struct common_opts common = { COMMON_OPTS_INIT };
};

void ug_input_state::frame_arrived(struct video_frame *f, struct audio_frame *a)
{
        lock_guard<mutex> lk(lock);
        if (frame_queue.size() < MAX_QUEUE_SIZE) {
                frame_queue.push({f, a});
        } else {
                MSG(WARNING, "Dropping frame!\n");
                AUDIO_FRAME_DISPOSE(a);
                VIDEO_FRAME_DISPOSE(f);
        }
}

static void
usage()
{
        printf("Usage:\n");
        color_printf("\t" TBOLD(
            TRED("-t ug_input") "[:<port>[:codec=<c>]] [-s embedded]") "\n");
        printf("where:\n");
        color_printf("\t" TBOLD("<port>") " - UG port to listen to\n");
        color_printf("\t" TBOLD("<c>") " - enforce pixfmt to decode to\n");
}

static bool
parse_fmt(char *fmt, uint16_t *port, codec_t *decode_to)
{
        char *tok     = nullptr;
        char *saveptr = nullptr;
        while ((tok = strtok_r(fmt, ":", &saveptr)) != nullptr) {
                fmt             = nullptr;
                const char *val = strchr(tok, '=') + 1;
                if (isdigit(tok[0])) {
                        *port = stoi(tok);
                } else if (IS_KEY_PREFIX(tok, "codec")) {
                        *decode_to = get_codec_from_name(val);
                        if (*decode_to == VIDEO_CODEC_NONE) {
                                MSG(ERROR, "Invalid codec: %s\n", val);
                                return false;
                        }
                } else {
                        MSG(ERROR, "Invalid option: %s\n", tok);
                        return false;
                }
        }
        return true;
}

static int vidcap_ug_input_init(struct vidcap_params *cap_params, void **state)
{
        uint16_t port = 5004;
        codec_t  decode_to = VIDEO_CODEC_NONE;

        if (strcmp("help", vidcap_params_get_fmt(cap_params)) == 0) {
                usage();
                return VIDCAP_INIT_NOERR;
        }
        char      *fmt_cpy   = strdup(vidcap_params_get_fmt(cap_params));
        const bool parse_ret = parse_fmt(fmt_cpy, &port, &decode_to);
        free(fmt_cpy);
        if (!parse_ret) {
                return VIDCAP_INIT_FAIL;
        }

        auto *s = new ug_input_state();

        char cfg[128] = "";
        snprintf(cfg, sizeof cfg, "%p", s);
        if (decode_to != VIDEO_CODEC_NONE) {
                snprintf(cfg + strlen(cfg), sizeof cfg - strlen(cfg),
                         ":codec=%s", get_codec_name(decode_to));
        }
        int ret = initialize_video_display(vidcap_params_get_parent(cap_params), "pipe", cfg, 0, NULL, &s->display);
        assert(ret == 0 && "Unable to initialize proxy display");

        map<string, param_u> params;

        // common
        s->common.parent = vidcap_params_get_parent(cap_params);
        params["exporter"].ptr = NULL;
        params["compression"].str = "none";
        params["rxtx_mode"].i = MODE_RECEIVER;

        //RTP
        params["common"].ptr = &s->common;
        // should be localhost and RX TX ports the same (here dynamic) in order to work like a pipe
        params["receiver"].str = "localhost";
        params["rx_port"].i = port;
        params["tx_port"].i = 0;
        params["fec"].str = "none";
        params["bitrate"].ll = 0;

        // UltraGrid RTP
        params["decoder_mode"].l = VIDEO_NORMAL;
        params["display_device"].ptr = s->display;

        s->video_rxtx = unique_ptr<ultragrid_rtp_video_rxtx>(dynamic_cast<ultragrid_rtp_video_rxtx *>(video_rxtx::create("ultragrid_rtp", params)));
        assert (s->video_rxtx);

        if (vidcap_params_get_flags(cap_params) & VIDCAP_FLAG_AUDIO_ANY) {
                struct audio_options opt = {
                        .host = "localhost",
                        .recv_port = port + 2,
                        .send_port = 0,
                        .recv_cfg = "embedded",
                        .send_cfg = "none",
                        .proto = "ultragrid_rtp",
                        .proto_cfg = "",
                        .fec_cfg = "none",
                        .channel_map = NULL,
                        .scale = "none",
                        .echo_cancellation = false,
                        .codec_cfg = "PCM"
                };
                if (audio_init(&s->audio, &opt, &s->common) != 0) {
                        delete s;
                        return VIDCAP_INIT_FAIL;
                }

                struct additional_audio_data aux_aud_data = {
                        { s->display, display_put_audio_frame,
                         display_reconfigure_audio, display_ctl_property

                        },
                        s->video_rxtx.get()
                };
                audio_register_aux_data(s->audio, aux_aud_data);

                audio_start(s->audio);
        }
        s->t0 = steady_clock::now();

        s->receiver_thread = thread(&video_rxtx::receiver_thread, s->video_rxtx.get());
        display_run_new_thread(s->display);

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_ug_input_done(void *state)
{
        auto s = (ug_input_state *) state;

        audio_join(s->audio);
        s->receiver_thread.join();

        display_put_frame(s->display, NULL, 0);
        display_join(s->display);
        display_done(s->display);

        s->video_rxtx->join();

        while (!s->frame_queue.empty()) {
                auto item = s->frame_queue.front();
                s->frame_queue.pop();
                VIDEO_FRAME_DISPOSE(item.first);
                AUDIO_FRAME_DISPOSE(item.second);
        }
        audio_done(s->audio);

        delete s;
}

static struct video_frame *vidcap_ug_input_grab(void *state, struct audio_frame **audio)
{
        auto s = (ug_input_state *) state;
        *audio = NULL;
        lock_guard<mutex> lk(s->lock);
        if (s->frame_queue.empty()) {
                return NULL;
        } else {
                auto item = s->frame_queue.front();
                struct video_frame *frame = item.first;
                *audio = item.second;
                s->frame_queue.pop();
                frame->callbacks.dispose = vf_free;

                s->frames++;
                auto curr_time = steady_clock::now();
                double seconds = duration_cast<duration<double>>(curr_time - s->t0).count();
                if (seconds >= 5.0) {
                        float fps = s->frames / seconds;
                        log_msg(LOG_LEVEL_INFO, "[ug_input] %d frames in %g seconds = %g FPS\n",
                                        s->frames, seconds, fps);
                        s->t0 = curr_time;
                        s->frames = 0;
                }

                return frame;
        }
}

static void vidcap_ug_input_probe(device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 0;
        *available_cards = nullptr;
}

static const struct video_capture_info vidcap_ug_input_info = {
        vidcap_ug_input_probe,
        vidcap_ug_input_init,
        vidcap_ug_input_done,
        vidcap_ug_input_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(ug_input, &vidcap_ug_input_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

