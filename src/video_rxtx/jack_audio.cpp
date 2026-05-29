// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

/**
 * @file
 * audio transport over JACK with video delegated to ultragrid_rtp
 */

#include "audio/jack.h"
#include "audio/types.h"
#include "audio/utils.h"
#include "debug.h"
#include "host.h"
#include "rtp/pbuf.h"
#include "lib_common.h"
#include "utils/thread.h"
#include "video_display.h"
#include "video_frame.h"

#include "types.h"
#include "video_rxtx.h"

#define MOD_NAME "[rxtx/jack] "

struct jack_audio_rxtx {
        void              *jack_connection;
        struct video_rxtx *video_rxtx;
};

static void
jack_trans_done(void *state)
{
        auto *s = (struct jack_audio_rxtx *) state;
        free(s);
}

static void *
jack_trans_init(const struct vrxtx_params *params,
                const struct common_opts  *common)
{
        auto *s = (struct jack_audio_rxtx *) calloc(
            1, sizeof(struct jack_audio_rxtx));
        s->jack_connection = jack_start(params->protocol_opts);
        if (s->jack_connection == nullptr) {
                MSG(ERROR, "initialization failed!\n");
                jack_trans_done(s);
                return nullptr;
        }
        if (params->medium[TX_MEDIA_VIDEO].rxtx_mode != 0) {
                struct vrxtx_params ug_rtp_params = *params;
                ug_rtp_params.protocol_opts       = "";
                int rc = vrxtx_init("ultragrid_rtp", &ug_rtp_params, common,
                                    &s->video_rxtx);
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
        auto *s = (struct jack_audio_rxtx *) arg;
        vrxtx_join(s->video_rxtx);
}

static void
jack_video_send_frame(void *state, std::shared_ptr<video_frame> f)
{
        auto *s = (struct jack_audio_rxtx *) state;
        vrxtx_send(s->video_rxtx, std::move(f));
}

static void
jack_send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        auto              *s = (struct jack_audio_rxtx *) state;
        char               data[1024 * 1024];
        struct audio_frame f = {};
        f.data               = data;
        f.max_size           = sizeof data;
        audio_frame2_to_audio_frame(&f, frame);
        jack_send(s->jack_connection, &f);
}

static struct rx_audio_frames *
jack_recv_audio_frame(void *state)
{
        auto *s = (struct jack_audio_rxtx *) state;
        struct acodec_state jack_pbuf{};
        bool decoded = jack_receive(s->jack_connection, &jack_pbuf);
        if (!decoded) {
                return nullptr;
        }
        auto *ret = (struct rx_audio_frames *) calloc(
            1, sizeof(struct rx_audio_frames));
        ret->frame = jack_pbuf.decoded;
        return ret;
}


static const struct video_rxtx_info jack_audio_rxtx_info = {
        .long_name          = "JACK audio transport (UG RTP for video)",
        .create             = jack_trans_init,
        .done               = jack_trans_done,
        .send_audio_frame   = jack_send_audio_frame,
        .recv_audio_frame   = jack_recv_audio_frame,
        .send_video_frame   = jack_video_send_frame,
        .video_recv_routine = dummy_jack_video_receiver_thread,
        .ctl_property       = nullptr,
        .join_sender        = jack_video_join,
};

REGISTER_MODULE(jack, &jack_audio_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX,
                VIDEO_RXTX_ABI_VERSION);
