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
 * * add also audio
 */

#include <pthread.h> // for pthread_mutex_lock, pthread_mutex_unlock
#include <stdlib.h>  // for calloc, free
#include <string.h>  // for memcpy

#include "compat/c23.h"    // IWYU pragma: keep
#include "debug.h"         // for LOG_LEVEL_ERROR, LOG_LEVEL_WARNING, MSG
#include "host.h"          // for register_should_exit_callback, unregister...
#include "lib_common.h"    // for REGISTER_MODULE, library_class
#include "utils/list.h"    // for simple_linked_list_size, simple_linked_li...
#include "utils/pthread.h" // for CHK_PTHR, ug_pthread_cond_init, ug_pthrea...
#include "utils/thread.h"  // for set_thread_name
#include "video_display.h" // for PUTF_BLOCKING, display_put_frame, display...
#include "video_frame.h"   // for video_desc_eq, video_desc_from_frame

#define MOD_NAME "[rxtx/loopback] "

static const int BUFF_MAX_LEN = 2;

#include "types.h"
#include "video_rxtx.h"

struct loopback_video_rxtx {
        struct module             *parent;
        struct display            *display_device;
        struct video_desc          configured_desc;
        struct simple_linked_list *frames;
        pthread_cond_t             frame_ready;
        pthread_mutex_t            lock;
        bool                       discard_in_frames;
};

static void*
init(const struct vrxtx_params *params)
{
        struct loopback_video_rxtx *s = calloc(1, sizeof *s);
        s->parent = params->parent;
        s->display_device = params->display_device;
        s->frames = simple_linked_list_init();
        ug_pthread_mutex_init(&s->lock);
        pthread_cond_init(&s->frame_ready, nullptr);
        if (params->medium[TX_MEDIA_VIDEO].rxtx_mode == MODE_SENDER) {
                MSG(WARNING,
                    "Running as a sender only - discarding all frames...\n");
                s->discard_in_frames = true;
        }
        if (params->medium[TX_MEDIA_VIDEO].rxtx_mode == MODE_RECEIVER) {
                MSG(WARNING, "Running as a receiver only - will not receive "
                             "anything...\n");
        }
        return s;
}

static void
send_frame(void *state, struct video_frame *f)
{
        struct loopback_video_rxtx *s = state;
        bool discard_frame = false;

        CHK_PTHR(pthread_mutex_lock(&s->lock));
        {
                if (s->discard_in_frames) {
                        discard_frame = true;
                        goto unlock;
                }
                if (simple_linked_list_size(s->frames) >= BUFF_MAX_LEN) {
                        MSG(WARNING, "Max buffer len %d exceeded.\n",
                            BUFF_MAX_LEN);
                        discard_frame = true;
                        goto unlock;
                }
                simple_linked_list_append(s->frames, f);
        }
unlock:
        CHK_PTHR(pthread_mutex_unlock(&s->lock));

        if (discard_frame) {
                f->callbacks.dispose(f);
                return;
        }

        CHK_PTHR(pthread_cond_signal(&s->frame_ready));
}

static void
should_exit_callback(void *arg)
{
        struct loopback_video_rxtx *s = arg;
        CHK_PTHR(pthread_mutex_lock(&s->lock));
        {
                s->discard_in_frames = true;
                simple_linked_list_append(s->frames, nullptr); // poison pill
        }
        CHK_PTHR(pthread_mutex_unlock(&s->lock));
        CHK_PTHR(pthread_cond_signal(&s->frame_ready));
}

static void *
receiver_thread(void *arg)
{
        set_thread_name(__func__);

        struct loopback_video_rxtx *s = arg;

        register_should_exit_callback(s->parent, should_exit_callback, s);

        while (true) {
                struct video_frame *frame = nullptr;
                CHK_PTHR(pthread_mutex_lock(&s->lock));
                {
                        while (simple_linked_list_size(s->frames) == 0) {
                                pthread_cond_wait(&s->frame_ready, &s->lock);
                        }
                        frame = simple_linked_list_pop(s->frames);
                }
                CHK_PTHR(pthread_mutex_unlock(&s->lock));

                if (frame == nullptr) { // poison pill
			break;
                }

                struct video_desc new_desc = video_desc_from_frame(frame);
                if (!video_desc_eq(s->configured_desc, new_desc)) {
                        bool reconfigured = display_reconfigure(s->display_device, new_desc, VIDEO_NORMAL);
                        if (!reconfigured) {
                                MSG(ERROR, "Unable to reconfigure display!\n");
                                frame->callbacks.dispose(frame);
                                continue;
                        }
                        s->configured_desc = new_desc;
                }
                struct video_frame *display_f = display_get_frame(s->display_device);
                memcpy(display_f->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
                display_put_frame(s->display_device, display_f, PUTF_BLOCKING);
                frame->callbacks.dispose(frame);
        }
        display_put_frame(s->display_device, nullptr, PUTF_BLOCKING);
        unregister_should_exit_callback(s->parent, should_exit_callback, s);

        return nullptr;
}

static void done(void *state) {
        struct loopback_video_rxtx *s = state;

        simple_linked_list_destroy(s->frames);
        CHK_PTHR(pthread_cond_destroy(&s->frame_ready));
        CHK_PTHR(pthread_mutex_destroy(&s->lock));
        free(s);
}

static const struct video_rxtx_info loopback_video_rxtx_info = {
        .long_name    = "loopback dummy transport",
        .create       = init,
        .done         = done,
        .ctl_property = nullptr,

        .send_audio_frame = nullptr,
        .recv_audio_frame = nullptr,

        .send_video_frame   = nullptr,
        .send_video_frame_c = send_frame,
        .video_recv_routine = receiver_thread,
        .join_video_sender  = nullptr,
};

REGISTER_MODULE(loopback, &loopback_video_rxtx_info, LIBRARY_CLASS_VIDEO_RXTX, VIDEO_RXTX_ABI_VERSION);

