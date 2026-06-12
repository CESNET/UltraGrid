/*
 * Copyright (c) 2018-2026 CESNET, zájmové sdružení právnických osob
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
 * @file
 * @author Martin Pulec     <pulec@cesnet.cz>
 * @todo
 * * works only when device can directly display the codec natively
 */

#include <pthread.h> // for pthread_mutex_lock, pthread_mutex_unlock
#include <stdlib.h>  // for calloc, free
#include <string.h>  // for memcpy

#include "audio/types.h"   // for audio_frame2_copy, audio_frame2_get_all_d...
#include "compat/c23.h"    // IWYU pragma: keep
#include "debug.h"         // for LOG_LEVEL_ERROR, LOG_LEVEL_WARNING, MSG
#include "host.h"          // for register_should_exit_callback, unregister...
#include "lib_common.h"    // for REGISTER_MODULE, library_class
#include "rxtx.h"          // for rxtx_medium_params, rxtx_params, rx_audio...
#include "utils/list.h"    // for simple_linked_list_size, simple_linked_li...
#include "utils/pthread.h" // for CHK_PTHR, ug_pthread_cond_init, ug_pthrea...
#include "utils/thread.h"  // for set_thread_name
#include "video_display.h" // for PUTF_BLOCKING, display_put_frame, display...
#include "video_frame.h"   // for video_desc_eq, video_desc_from_frame

#define MOD_NAME "[rxtx/loopback] "

static const int BUFF_MAX_LEN = 2;

#include "types.h"

struct loopback_rxtx {
        struct module             *parent;

        struct loopback_rxtx_audio {
                bool                       active;
                bool                       callback_reg;
                struct simple_linked_list *frames;
                pthread_cond_t             frame_ready;
                pthread_mutex_t            lock;
        } audio;

        struct loopback_rxtx_video {
                bool                       discard_in_frames;
                struct display            *display_device;
                struct video_desc          configured_desc;
                struct simple_linked_list *frames;
                pthread_cond_t             frame_ready;
                pthread_mutex_t            lock;
        } video;
};

// prototypes
static void should_exit_audio(void *arg);

static void*
init(const struct rxtx_params *params)
{
        struct loopback_rxtx *s = calloc(1, sizeof *s);
        s->parent               = params->parent;
        s->audio.frames         = simple_linked_list_init();
        s->video.frames         = simple_linked_list_init();
        s->video.display_device = params->display_device;
        ug_pthread_mutex_init(&s->audio.lock);
        ug_pthread_mutex_init(&s->video.lock);
        pthread_cond_init(&s->audio.frame_ready, nullptr);
        pthread_cond_init(&s->video.frame_ready, nullptr);

        if (params->medium[TX_MEDIA_VIDEO].rxtx_mode == MODE_SENDER) {
                MSG(WARNING, "Running a video sender only - discarding all "
                             "frames...\n");
                s->video.discard_in_frames = true;
        }
        if (params->medium[TX_MEDIA_VIDEO].rxtx_mode == MODE_RECEIVER) {
                MSG(WARNING,
                    "Running a video receiver only - will not receive "
                    "anything...\n");
        }

        if (params->medium[TX_MEDIA_AUDIO].rxtx_mode != 0) {
                if (params->medium[TX_MEDIA_AUDIO].rxtx_mode == MODE_SENDER) {
                        MSG(WARNING,
                            "Running an audio sender only - discarding "
                            "all frames...\n");
                } else if (params->medium[TX_MEDIA_AUDIO].rxtx_mode ==
                           MODE_RECEIVER) {
                        MSG(WARNING,
                            "Running an audio receiver only - will not receive "
                            "anything...\n");
                } else {
                        s->audio.active = true;
                        register_should_exit_callback(s->parent,
                                                      should_exit_audio, s);
                        s->audio.callback_reg = true;
                }
        }
        return s;
}

static void
send_video_frame(void *state, struct video_frame *f)
{
        struct loopback_rxtx       *s     = state;
        struct loopback_rxtx_video *video = &s->video;

        bool discard_frame = false;

        CHK_PTHR(pthread_mutex_lock(&video->lock));
        {
                if (video->discard_in_frames) {
                        discard_frame = true;
                        goto unlock;
                }
                if (simple_linked_list_size(video->frames) >= BUFF_MAX_LEN) {
                        MSG(WARNING, "Max video buffer len %d exceeded.\n",
                            BUFF_MAX_LEN);
                        discard_frame = true;
                        goto unlock;
                }
                simple_linked_list_append(video->frames, f);
        }
unlock:
        CHK_PTHR(pthread_mutex_unlock(&video->lock));

        if (discard_frame) {
                f->callbacks.dispose(f);
                return;
        }

        CHK_PTHR(pthread_cond_signal(&video->frame_ready));
}

static void
should_exit_audio(void *arg)
{
        struct loopback_rxtx       *s     = arg;
        struct loopback_rxtx_audio *audio = &s->audio;

        CHK_PTHR(pthread_mutex_lock(&audio->lock));
        {
                s->audio.active = false;
                simple_linked_list_append(s->audio.frames,
                                          nullptr); // poison pill
        }
        CHK_PTHR(pthread_mutex_unlock(&audio->lock));
        CHK_PTHR(pthread_cond_signal(&audio->frame_ready));
}

static void
should_exit_video_recv_thr(void *arg)
{
        struct loopback_rxtx       *s     = arg;
        struct loopback_rxtx_video *video = &s->video;

        CHK_PTHR(pthread_mutex_lock(&video->lock));
        {
                video->discard_in_frames = true;
                simple_linked_list_append(video->frames, nullptr); // poison pill
        }
        CHK_PTHR(pthread_mutex_unlock(&video->lock));
        CHK_PTHR(pthread_cond_signal(&video->frame_ready));
}

static void *
video_receiver_thread(void *arg)
{
        set_thread_name(__func__);

        struct loopback_rxtx       *s     = arg;
        struct loopback_rxtx_video *video = &s->video;

        register_should_exit_callback(s->parent, should_exit_video_recv_thr, s);

        while (true) {
                struct video_frame *frame = nullptr;
                CHK_PTHR(pthread_mutex_lock(&video->lock));
                {
                        while (simple_linked_list_size(video->frames) == 0) {
                                pthread_cond_wait(&video->frame_ready,
                                                  &video->lock);
                        }
                        frame = simple_linked_list_pop(video->frames);
                }
                CHK_PTHR(pthread_mutex_unlock(&video->lock));

                if (frame == nullptr) { // poison pill
			break;
                }

                struct video_desc new_desc = video_desc_from_frame(frame);
                if (!video_desc_eq(video->configured_desc, new_desc)) {
                        bool reconfigured = display_reconfigure(
                            video->display_device, new_desc, VIDEO_NORMAL);
                        if (!reconfigured) {
                                MSG(ERROR, "Unable to reconfigure display!\n");
                                frame->callbacks.dispose(frame);
                                continue;
                        }
                        s->video.configured_desc = new_desc;
                }
                struct video_frame *display_f =
                    display_get_frame(video->display_device);
                memcpy(display_f->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
                display_put_frame(video->display_device, display_f, PUTF_BLOCKING);
                frame->callbacks.dispose(frame);
        }
        display_put_frame(video->display_device, nullptr, PUTF_BLOCKING);
        unregister_should_exit_callback(s->parent, should_exit_video_recv_thr, s);

        return nullptr;
}

static void
send_audio_frame(void *state, const struct audio_frame2 *frame)
{
        struct loopback_rxtx       *s     = state;
        struct loopback_rxtx_audio *audio = &s->audio;

        bool ignore_frame = false;

        CHK_PTHR(pthread_mutex_lock(&audio->lock));
        {
                if (!audio->active) {
                        ignore_frame = true;
                        goto unlock;
                }
                if (simple_linked_list_size(audio->frames) >= BUFF_MAX_LEN) {
                        MSG(WARNING, "Max audio buffer len %d exceeded.\n",
                            BUFF_MAX_LEN);
                        ignore_frame = true;
                        goto unlock;
                }
                struct audio_frame2 *copy = audio_frame2_copy(frame);
                simple_linked_list_append(audio->frames, copy);
        }
unlock:
        CHK_PTHR(pthread_mutex_unlock(&audio->lock));

        if (!ignore_frame) {
                CHK_PTHR(pthread_cond_signal(&audio->frame_ready));
        }
}

static struct rx_audio_frames *
recv_audio_frame(void *state)
{
        struct loopback_rxtx       *s     = state;
        struct loopback_rxtx_audio *audio = &s->audio;

        struct audio_frame2 *frame = nullptr;
        CHK_PTHR(pthread_mutex_lock(&audio->lock));
        {
                if (!audio->active) {
                        CHK_PTHR(pthread_mutex_unlock(&audio->lock));
                        return nullptr;
                }
                while (simple_linked_list_size(audio->frames) == 0) {
                        pthread_cond_wait(&audio->frame_ready, &audio->lock);
                }
                frame = simple_linked_list_pop(audio->frames);
        }
        CHK_PTHR(pthread_mutex_unlock(&audio->lock));

        if (!frame) { // poison pill passed
                return nullptr;
        }

        struct rx_audio_frames *frm = calloc(1, sizeof *frm);
        frm->frame = frame;
        frm->expected_bytes         = frm->received_bytes =
            (long long) audio_frame2_get_all_data_len(frame);

        return frm;
}

static void
done(void *state)
{
        struct loopback_rxtx *s = state;

        if (s->audio.callback_reg) {
                unregister_should_exit_callback(s->parent, should_exit_audio,
                                                s);
        }

        simple_linked_list_destroy(s->audio.frames);
        simple_linked_list_destroy(s->video.frames);
        CHK_PTHR(pthread_cond_destroy(&s->audio.frame_ready));
        CHK_PTHR(pthread_cond_destroy(&s->video.frame_ready));
        CHK_PTHR(pthread_mutex_destroy(&s->audio.lock));
        CHK_PTHR(pthread_mutex_destroy(&s->video.lock));
        free(s);
}

static const struct rxtx_info loopback_video_rxtx_info = {
        .long_name    = "loopback dummy transport",
        .create       = init,
        .done         = done,
        .ctl_property = nullptr,

        .send_audio_frame = send_audio_frame,
        .recv_audio_frame = recv_audio_frame,

        .send_video_frame   = nullptr,
        .send_video_frame_c = send_video_frame,
        .video_recv_routine = video_receiver_thread,
        .join_video_sender  = nullptr,
};

REGISTER_MODULE(loopback, &loopback_video_rxtx_info, LIBRARY_CLASS_RXTX, RXTX_ABI_VERSION);

