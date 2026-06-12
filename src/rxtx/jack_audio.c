// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

/**
 * @file
 * audio transport over JACK with video delegated to ultragrid_rtp
 * @note
 * not tested - seems like not being used much
 */

#include <stdlib.h>       // for calloc, free

#include "audio/jack.h"   // for jack_done, jack_receive, jack_send, jack_start
#include "audio/types.h"  // for audio_frame
#include "audio/utils.h"  // for audio_frame2_to_audio_frame
#include "debug.h"        // for LOG_LEVEL_ERROR, MSG
#include "lib_common.h"   // for REGISTER_MODULE, library_class
#include "rxtx.h"         // for rxtx_params, rx_audio_frames, rxtx_medium_...
#include "types.h"        // for tx_media_type, video_frame (ptr only)

struct audio_frame2;

#define MOD_NAME "[rxtx/jack] "

struct jack_audio_rxtx {
        void              *jack_connection;
        struct rxtx *video_rxtx;
};

static void
jack_trans_done(void *state)
{
        struct jack_audio_rxtx *s = state;
        jack_done(s->jack_connection);
        free(s);
}

static void *
jack_trans_init(const struct rxtx_params *params)
{
        struct jack_audio_rxtx *s = calloc(
            1, sizeof(struct jack_audio_rxtx));
        s->jack_connection = jack_start(params->protocol_opts);
        if (s->jack_connection == nullptr) {
                MSG(ERROR, "initialization failed!\n");
                jack_trans_done(s);
                return nullptr;
        }
        if (params->medium[TX_MEDIA_VIDEO].rxtx_mode != 0) {
                struct rxtx_params ug_rtp_params = *params;
                int                 rc =
                    rxtx_init("ultragrid_rtp", &ug_rtp_params, &s->video_rxtx);
                if (rc != 0) {
                        jack_trans_done(s);
                        return nullptr;
                }
        }
        return s;
}

/**
 * this must be defined because non-null video_rxtx_info.video_recv_routine
 * indicates that we are capable of receiving video
 */
static void *
dummy_jack_video_receiver_thread(void * /* arg */)
{
        return nullptr;
}

static void
jack_video_join(void *arg)
{
        struct jack_audio_rxtx *s = arg;
        rxtx_join(s->video_rxtx);
}

static void
jack_video_send_frame(void *state, struct video_frame *f)
{
        struct jack_audio_rxtx *s = state;
        rxtx_send_video(s->video_rxtx, f);
}

static void
jack_send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        struct jack_audio_rxtx *s = state;
        char                    data[1024 * 1024];
        struct audio_frame      f = { 0 };
        f.data                    = data;
        f.max_size                = sizeof data;
        audio_frame2_to_audio_frame(&f, frame);
        jack_send(s->jack_connection, &f);
}

static struct rx_audio_frames *
jack_recv_audio_frame(void *state)
{
        struct jack_audio_rxtx *s = state;
        struct audio_frame2 *decoded = jack_receive(s->jack_connection);
        if (!decoded) {
                return nullptr;
        }
        struct rx_audio_frames *ret = calloc(1, sizeof(struct rx_audio_frames));
        ret->frame                  = decoded;
        return ret;
}

static const struct rxtx_info jack_audio_rxtx_info = {
        .long_name    = "JACK audio transport (UG RTP for video)",
        .create       = jack_trans_init,
        .done         = jack_trans_done,
        .ctl_property = nullptr,

        .send_audio_frame = jack_send_audio_frame,
        .recv_audio_frame = jack_recv_audio_frame,

        .send_video_frame   = nullptr,
        .send_video_frame_c = jack_video_send_frame,
        .video_recv_routine = dummy_jack_video_receiver_thread,
        .join_video_sender  = jack_video_join,
};

REGISTER_MODULE(jack, &jack_audio_rxtx_info, LIBRARY_CLASS_RXTX,
                RXTX_ABI_VERSION);
