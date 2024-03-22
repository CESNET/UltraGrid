/**
 * @file   video_display/caca.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023-2024 CESNET, z. s. p. o.
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

#include <caca.h>
#include <pthread.h>          // for pthread_mutex_unlock, pthread_cond_destroy
#include <stdbool.h>          // for bool, true, false
#include <stdio.h>            // for snprintf
#include <stdlib.h>           // for NULL, calloc, free, getenv, size_t
#include <string.h>           // for strcmp, strlen, memcpy, strchr, strstr
#include <time.h>             // for timespec_get, TIME_UTC, timespec

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "tv.h"
#include "video.h"
#include "video_display.h"

#define MOD_NAME "[caca] "

struct state_caca {
        caca_canvas_t  *canvas;
        caca_display_t *display;
        caca_dither_t  *dither;
        int screen_w;
        int screen_h;

        struct video_desc desc;
        struct video_frame *f;

        _Bool started;
        _Bool should_exit;
        pthread_t thread_id;
        pthread_mutex_t lock;
        pthread_cond_t frame_ready_cv;
        pthread_cond_t frame_consumed_cv;
};

static bool  display_caca_putf(void *state, struct video_frame *frame,
                               long long timeout_ns);
static void *worker(void *arg);

static void display_caca_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 1;
        *available_cards = calloc(*count, sizeof **available_cards);
        snprintf((*available_cards)[0].name, sizeof (*available_cards)[0].name, "caca");
}

static void display_caca_done(void *state)
{
        struct state_caca *s = state;
        if (s->started) {
                if (!s->should_exit) {
                        display_caca_putf(state, NULL, PUTF_BLOCKING);
                }
                pthread_join(s->thread_id, NULL);
        }
        if (s->dither) {
                caca_free_dither(s->dither);
        }
        if (s->display) {
                caca_free_display(s->display);
        }
        if (s->canvas) {
                caca_free_canvas(s->canvas);
        }
        vf_free(s->f);
        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->frame_ready_cv);
        pthread_cond_destroy(&s->frame_consumed_cv);
        free(s);
}

static void usage() {
        color_printf("Display " TBOLD("caca") " syntax:\n");
        color_printf("\t" TBOLD(TRED("caca") "[:driver=<drv>]") "\n");
        color_printf("\nAvailable drivers:\n");
        const char * const * drivers = caca_get_display_driver_list();
        while (*drivers) {
                const char *driver = *drivers++;
                const char *desc = *drivers++;
                color_printf("\t- " TBOLD("%s") " - %s\n", driver, desc ? desc : "(no description");
        }
}

static void *display_caca_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        UNUSED(parent);
        const char *driver = NULL;
        if (strlen(fmt) > 0) {
                if (strstr(fmt, "driver=") == fmt) {
                        driver = strchr(fmt, '=') + 1;
                } else {
                        usage();
                        return strcmp(fmt, "help") == 0 ? INIT_NOERR : NULL;
                }
        }
        struct state_caca *s = calloc(1, sizeof *s);

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->frame_ready_cv, NULL);
        pthread_cond_init(&s->frame_consumed_cv, NULL);

        s->canvas = caca_create_canvas(0, 0);
        if (!s->canvas) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create canvas\n");
                display_caca_done(s);
                return NULL;
        }
        s->display = caca_create_display_with_driver(s->canvas, driver);
        if (!s->display) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create display\n");
                display_caca_done(s);
                return NULL;
        }
        driver = caca_get_display_driver(s->display);
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Using display driver: %s\n", driver);

        if (strcmp(driver, "x11") != 0 && strcmp(driver, "null") != 0) {
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Disabling keyboard control.\n");
                set_commandline_param("disable-keyboard-control", "");
        }

        s->f = get_splashscreen();

        pthread_create(&s->thread_id, NULL, worker, s);
        s->started = 1;

        return s;
}

static struct video_frame *display_caca_getf(void *state)
{
        struct state_caca *s = state;
        return vf_alloc_desc_data(s->desc);
}

static void handle_events(struct state_caca *s, struct video_frame *last_frame)
{
        caca_event_t e;
        int ret = 0;
        while ((ret = caca_get_event(s->display, CACA_EVENT_ANY, &e, 0)) > 0) {
                switch (e.type) {
                        case CACA_EVENT_KEY_PRESS:
                                if (e.data.key.ch == 'q' || e.data.key.ch == CACA_KEY_CTRL_C) {
                                        exit_uv(0);
                                }
                                break;
                        case CACA_EVENT_RESIZE:
                                s->screen_w = e.data.resize.w;
                                s->screen_h = e.data.resize.h;
                                verbose_msg(MOD_NAME "Resized to %dx%d\n", s->screen_w, s->screen_h);
                                if (last_frame) {
                                        caca_dither_bitmap(s->canvas, 0, 0, s->screen_w, s->screen_h, s->dither, last_frame->tiles[0].data);
                                        caca_refresh_display(s->display);
                                }
                                break;
                        case CACA_EVENT_QUIT:
                                exit_uv(0);
                                break;
                        default:
                                break;
                }
        }
}

static _Bool reconfigure(struct state_caca *s, struct video_desc desc) {
        enum {
                RMASK = 0x0000ff,
                GMASK = 0x00ff00,
                BMASK = 0xff0000,
                AMASK = 0x000000,
        };
        caca_free_dither(s->dither);
        s->dither = caca_create_dither(8 * get_bpp(desc.color_spec),
                        desc.width, desc.height,
                        get_bpp(desc.color_spec) * desc.width,
                        RMASK, GMASK, BMASK, AMASK);
        if (!s->dither) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create dither\n");
                exit_uv(1);
                return 0;
        }
        s->screen_w = caca_get_canvas_width(s->canvas);
        s->screen_h = caca_get_canvas_height(s->canvas);

        return 1;
}

static void *worker(void *arg)
{
        struct video_desc display_desc = { 0 };
        struct state_caca *s = arg;
        struct video_frame *last_frame = NULL;
        while (1) {
                struct video_frame *f = NULL;
                struct timespec ts;
                timespec_get(&ts, TIME_UTC);
                ts_add_nsec(&ts, 200 * NS_IN_MS); // 200 ms
                pthread_mutex_lock(&s->lock);
                while (!s->f && !s->should_exit) {
                        pthread_cond_timedwait(&s->frame_ready_cv, &s->lock, &ts);
                        handle_events(s, last_frame);
                }
                if (s->should_exit) {
                        pthread_mutex_unlock(&s->lock);
                        break;
                }
                f = s->f;
                s->f = NULL;
                pthread_mutex_unlock(&s->lock);
                pthread_cond_signal(&s->frame_consumed_cv);
                if (!video_desc_eq(display_desc, video_desc_from_frame(f))) {
                        if (!reconfigure(s, video_desc_from_frame(f))) {
                                vf_free(f);
                                continue;
                        }
                        display_desc = video_desc_from_frame(f);
                }

                handle_events(s, last_frame);
                caca_dither_bitmap(s->canvas, 0, 0, s->screen_w, s->screen_h, s->dither, f->tiles[0].data);
                caca_refresh_display(s->display);
                handle_events(s, last_frame);
                vf_free(last_frame);
                last_frame = f;
        }
        vf_free(last_frame);
        return NULL;
}

static bool display_caca_putf(void *state, struct video_frame *frame, long long timeout_ns)
{
        if (timeout_ns == PUTF_DISCARD) {
                vf_free(frame);
                return true;
        }

        struct state_caca *s = state;

        pthread_mutex_lock(&s->lock);
        while (s->f) {
                pthread_cond_wait(&s->frame_consumed_cv, &s->lock);
        }
        if (frame) {
                s->f = frame;
        } else {
                s->should_exit = 1;
        }
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->frame_ready_cv);

        return true;
}

static bool display_caca_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);

        codec_t c[2] = { RGB, RGBA };
        if (property == DISPLAY_PROPERTY_CODECS && sizeof c <= *len) {
                memcpy(val, c, sizeof c);
                *len = sizeof c;
                return true;
        }
        return false;
}

static bool display_caca_reconfigure(void *state, struct video_desc desc)
{
        struct state_caca *s = state;
        s->desc = desc;
        return true;
}

static const void *display_caca_info_get() {
        static _Thread_local struct video_display_info display_caca_info = {
                display_caca_probe,
                display_caca_init,
                NULL, // _run
                display_caca_done,
                display_caca_getf,
                display_caca_putf,
                display_caca_reconfigure,
                display_caca_get_property,
                NULL,
                NULL,
                DISPLAY_NO_GENERIC_FPS_INDICATOR,
        };
        const char *display = getenv("DISPLAY");
        if (display != NULL && strlen(display) > 0) { // print FPS only if caca has own window (not the terminal)
                display_caca_info.generic_fps_indicator_prefix = MOD_NAME;
        }
        return &display_caca_info;
};

REGISTER_MODULE_WITH_FUNC(caca, display_caca_info_get, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
