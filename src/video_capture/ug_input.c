/**
 * @file   video_capture/ug_input.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2026 CESNET, zájmové sdružení právických osob
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

#include <assert.h>                // for assert
#include <ctype.h>                 // for isdigit
#include <pthread.h>               // for pthread_mutex_unlock, pthread_mute...
#include <stdint.h>                // for uint16_t
#include <stdio.h>                 // for printf, snprintf
#include <stdlib.h>                // for free
#include <string.h>                // for strlen, strchr, strcmp, strdup

#include "audio/audio.h"           // for audio_options, AUDIO_OPTIONS_INIT
#include "audio/types.h"           // for AUDIO_FRAME_DISPOSE
#include "compat/c23.h"            // IWYU pragma: keep
#include "debug.h"                 // for LOG_LEVEL_ERROR, LOG_LEVEL_WARNING
#include "host.h"                  // for common_opts, COMMON_OPTS_INIT
#include "lib_common.h"            // for REGISTER_MODULE, library_class
#include "types.h"                 // for VIDEO_CODEC_NONE, codec_t, device_...
#include "utils/color_out.h"       // for TBOLD, color_printf, TRED
#include "utils/list.h"            // for simple_linked_list
#include "utils/macros.h"          // for to_fourcc, IS_KEY_PREFIX, snprintf_ch
#include "utils/pthread.h"         // for CHK_PTHR
#include "video_capture.h"         // for VIDCAP_INIT_FAIL, VIDCAP_INIT_NOERR
#include "video_capture_params.h"  // for vidcap_params_get_fmt, vidcap_para...
#include "video_codec.h"           // for get_codec_from_name, get_codec_name
#include "video_display.h"         // for display_done, display_join, displa...
#include "video_display/pipe.h"    // for pipe_frame_recv_delegate
#include "video_frame.h"           // for VIDEO_FRAME_DISPOSE, vf_free
#include "video_rxtx.h"            // for video_rxtx, vrxtx_params, VRXTX_INIT

struct vidcap_params;

#define MAGIC to_fourcc('V', 'C', 'u', 'i')
#define MOD_NAME "[ug_input] "
static const int MAX_QUEUE_SIZE = 2;

struct av_frame {
        struct video_frame *vframe;
        struct audio_frame *aframe;
};

struct ug_input_state {
        uint32_t magic;
        pthread_mutex_t lock;
        struct simple_linked_list *frame_queue;
        struct display *display;

        struct video_rxtx *video_rxtx;
        struct state_audio *audio;
};

static void vidcap_ug_input_done(void *state);

static void
ug_input_frame_arrived(void *state, struct video_frame *v,
                       struct audio_frame *a)
{
        struct ug_input_state *s = state;
        assert(s->magic == MAGIC);
        pthread_mutex_lock(&s->lock);
        if (simple_linked_list_size(s->frame_queue) < MAX_QUEUE_SIZE) {
                struct av_frame *avf = (struct av_frame *) malloc(sizeof *avf);
                avf->aframe = a;
                avf->vframe = v;
                simple_linked_list_append(s->frame_queue, avf);
        } else {
                MSG(WARNING, "Dropping frame!\n");
                AUDIO_FRAME_DISPOSE(a);
                VIDEO_FRAME_DISPOSE(v);
        }
        pthread_mutex_unlock(&s->lock);
}

static void
usage()
{
        printf("Usage:\n");
        color_printf("\t" TBOLD(
            TRED("-t ug_input") "[:port=<port>][:codec=<c>]] [-s embedded]") "\n");
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
                        MSG(WARNING, "port specification without the keyword "
                                     "port= is deprecated\n");
                        *port = atoi(tok);
                } else if (IS_KEY_PREFIX(tok, "port")) {
                        *port = atoi(val);
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

static int vidcap_ug_input_init(const struct vidcap_params *cap_params, void **state)
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

        struct ug_input_state *s = calloc(1, sizeof *s);
        s->magic = MAGIC;
        pthread_mutex_init(&s->lock, nullptr);
        s->frame_queue = simple_linked_list_init();

        char cfg[128] = "";

        const struct pipe_frame_recv_delegate dlg = {
                .state = s, .frame_arrived = ug_input_frame_arrived
        };
        snprintf_ch(cfg, "%p", &dlg);
        if (decode_to != VIDEO_CODEC_NONE) {
                snprintf(cfg + strlen(cfg), sizeof cfg - strlen(cfg),
                         ":codec=%s", get_codec_name(decode_to));
        }
        int ret =
            initialize_video_display(vidcap_params_get_parent(cap_params),
                                     "pipe", cfg, 0, nullptr, &s->display);
        assert(ret == 0 && "Unable to initialize proxy display");

        struct vrxtx_params params = VRXTX_INIT;

        // common
        struct common_opts common = COMMON_OPTS_INIT;
        common.parent = vidcap_params_get_parent(cap_params);
        params.rxtx_mode = MODE_RECEIVER;

        //RTP
        // should be localhost and RX TX ports the same (here dynamic) in order to work like a pipe
        common.receiver = "localhost";
        params.rx_port = port;
        params.tx_port = 0;
        // following 3 already set by VRTX_INIT
        // params["fec"].str = "none";
        // params["bitrate"].ll = 0;
        // UltraGrid RTP
        // params["decoder_mode"].l = VIDEO_NORMAL;
        params.display_device = s->display;

        int rc = vrxtx_init("ultragrid_rtp", &params, &common, &s->video_rxtx);
        assert(rc == 0);

        if (vidcap_params_get_flags(cap_params) & VIDCAP_FLAG_AUDIO_ANY) {
                struct audio_options opt = AUDIO_OPTIONS_INIT;
                opt.recv_port            = port + 2;
                opt.send_port            = 0;
                opt.recv_cfg             = "embedded";
                opt.proto                = "ultragrid_rtp";
                opt.display              = s->display;
                opt.vrxtx                = s->video_rxtx;

                if (audio_init(&s->audio, &opt, &common) != 0) {
                        vidcap_ug_input_done(s);
                        return VIDCAP_INIT_FAIL;
                }

                audio_start(s->audio);
        }

        display_run_new_thread(s->display);

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_ug_input_done(void *state)
{
        struct ug_input_state *s = state;
        assert(s->magic == MAGIC);

        audio_join(s->audio);
        vrxtx_join(s->video_rxtx);

        // display_put_frame(s->display, nullptr, 0); // already done by ultragrid_rtp_video_rxtx::receiver_loop
        display_join(s->display);
        display_done(s->display);

        while (simple_linked_list_size(s->frame_queue) == 0) {
                struct av_frame *item = simple_linked_list_pop(s->frame_queue);
                VIDEO_FRAME_DISPOSE(item->vframe);
                AUDIO_FRAME_DISPOSE(item->aframe);
                free(item);
        }

        audio_done(s->audio);
        vrxtx_destroy(s->video_rxtx);

        CHK_PTHR(pthread_mutex_destroy(&s->lock));
        simple_linked_list_destroy(s->frame_queue);

        free(s);
}

static struct video_frame *vidcap_ug_input_grab(void *state, struct audio_frame **audio)
{
        struct ug_input_state *s = state;
        *audio = nullptr;
        pthread_mutex_lock(&s->lock);
        while (simple_linked_list_size(s->frame_queue) == 0) {
                pthread_mutex_unlock(&s->lock);
                return nullptr;
        }

        struct av_frame *item = simple_linked_list_pop(s->frame_queue);
        struct video_frame *frame = item->vframe;
        *audio                    = item->aframe;
        free(item);
        frame->callbacks.dispose = vf_free;
        pthread_mutex_unlock(&s->lock);

        return frame;
}

static void
vidcap_ug_input_probe(struct device_info **available_cards, int *count,
                      void (**deleter)(void *))
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
        MOD_NAME,
};

REGISTER_MODULE(ug_input, &vidcap_ug_input_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

