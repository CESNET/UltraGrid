/**
 * @file   capture_filter/display.c
 * @author Martin Pulec <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET z.s.p.o.
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
 * @todo
 * * the data is 2x copied
 * * use decoder
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/list.h"
#include "video.h"
#include "video_codec.h"
#include "video_display.h"
#include "video_frame.h"

#define MAX_QUEUE_LEN 3
#define MOD_NAME "[display cap. f.] "

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct capture_filter_display {
        struct display *d;

        struct simple_linked_list *frame_queue;

        pthread_mutex_t lock;
        pthread_cond_t cv;
        pthread_t thread_id;
};

static void *worker(void *arg) {
        struct capture_filter_display *s = arg;
        struct video_desc display_desc = { 0 };

        while (1) {
                pthread_mutex_lock(&s->lock);
                while (simple_linked_list_size(s->frame_queue) == 0) {
                        pthread_cond_wait(&s->cv, &s->lock);
                }
                struct video_frame *f = simple_linked_list_pop(s->frame_queue);
                pthread_mutex_unlock(&s->lock);
                if (!f) {
                        display_put_frame(s->d, NULL, PUTF_BLOCKING); // pass poison pill
                        break;
                }

                struct video_desc new_desc = video_desc_from_frame(f);
                if (!video_desc_eq(display_desc, new_desc)) {
                        if (!display_reconfigure(s->d, new_desc, VIDEO_NORMAL)) {
                                log_msg(LOG_LEVEL_ERROR, "Unable to reconfigure!");
                                continue;
                        }
                        display_desc = new_desc;
                }
                struct video_frame *df = display_get_frame(s->d);
                memcpy(df->tiles[0].data, f->tiles[0].data, f->tiles[0].data_len);
                display_put_frame(s->d, df, PUTF_BLOCKING);
                vf_free(f);
        }

        return NULL;
}

static int init(struct module *parent, const char *cfg, void **state)
{
        if (strcmp(cfg, "help") == 0) {
                printf("Previews captured frame with specified dispay.\n"
                                "Usage:\n"
                                "\t--capture-filter display:<display_cfg>\n");
                return 1;
        }

        char *requested_display = strdupa(cfg);
        char *delim = strchr(requested_display, ':');
        const char *fmt = "";
        if (delim) {
                *delim = '\0';
                fmt = delim + 1;
        }
        struct display *d;
        int ret = initialize_video_display(parent, requested_display,
                fmt, 0, NULL, &d);
        if (ret != 0) {
                return ret;
        }

        struct capture_filter_display *s = calloc(1, sizeof *s);
        s->d = d;
        s->frame_queue = simple_linked_list_init();
        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->cv, NULL);

        display_run_new_thread(s->d);
        pthread_create(&s->thread_id, NULL, worker, s);

        *state = s;
        return 0;
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct capture_filter_display *s = state;
        struct video_frame *f = in ? vf_get_copy(in) : NULL;
        pthread_mutex_lock(&s->lock);
        if (simple_linked_list_size(s->frame_queue) == MAX_QUEUE_LEN) {
                pthread_mutex_unlock(&s->lock);
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Queue full, frame dropped.\n");
                return in;
        }
        simple_linked_list_append(s->frame_queue, f);
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->cv);

        return in;
}

static void done(void *state)
{
        struct capture_filter_display *s = state;
        filter(s, NULL);
        pthread_join(s->thread_id, NULL);
        display_join(s->d);
        display_done(s->d);
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->cv);
        simple_linked_list_destroy(s->frame_queue);
        free(s);
}

static const struct capture_filter_info capture_filter_display = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_HIDDEN_MODULE(display, &capture_filter_display, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

