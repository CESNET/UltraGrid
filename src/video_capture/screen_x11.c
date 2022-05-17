/**
 * @file   video_capture/screen_x11.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2021 CESNET, z.s.p.o.
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
 * The XGetImage() is a bit slow, consider using XShm as OBS does.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "utils/misc.h"
#include "utils/parallel_conv.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/types.h"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include <X11/Xlib.h>
#ifdef HAVE_XFIXES
#include <X11/extensions/Xfixes.h>
#endif // HAVE_XFIXES
#include <X11/Xutil.h>

#define MOD_NAME "[screen capture] "
#define QUEUE_SIZE_MAX 3

/* prototypes of functions defined in this module */
static void show_help(void);
static void *grab_thread(void *args);

static void show_help()
{
        printf("Screen capture\n");
        printf("Usage\n");
        printf("\t-t screen[:fps=<fps>][:display=<d>]\n");
        printf("\t\t<fps> - preferred grabbing fps (otherwise unlimited)\n");
        printf("\t\t<display> - display to capture (including the colon!)\n");
}

struct grabbed_data;

struct grabbed_data {
        XImage *data;
        struct grabbed_data *next;
};

struct vidcap_screen_x11_state {
        struct video_frame       *frame; 
        struct tile       *tile; 
        int frames;
        struct       timeval t, t0;
        Display *dpy;
        Window root;

        struct grabbed_data * volatile head, * volatile tail;
        volatile int queue_len;

        pthread_mutex_t lock;
        pthread_cond_t worker_cv;
        volatile bool worker_waiting;
        pthread_cond_t boss_cv;
        volatile bool boss_waiting;

        volatile bool should_exit_worker;

        pthread_t worker_id;

        struct timeval prev_time;

        double fps;

        bool initialized;
        int cpu_count;
        char *req_display;
};

static bool initialize(struct vidcap_screen_x11_state *s) {
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);

        XWindowAttributes wa;

        s->dpy = XOpenDisplay(s->req_display);
        if (!s->dpy) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot open display %s\n", IF_NOT_NULL_ELSE(s->req_display, IF_NOT_NULL_ELSE(getenv("DISPLAY"), "($DISPLAY undefined!)")));
                return false;
        }

        s->root = DefaultRootWindow(s->dpy);
        if (!s->dpy) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Cannot get root window!\n");
                return false;
        }

        XGetWindowAttributes(s->dpy, DefaultRootWindow(s->dpy), &wa);
        s->tile->width = wa.width;
        s->tile->height = wa.height;

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->boss_cv, NULL);
        pthread_cond_init(&s->worker_cv, NULL);

        s->boss_waiting = false;
        s->worker_waiting = false;

        s->head = s->tail = NULL;
        s->queue_len = 0;

        s->should_exit_worker = false;

        s->frame->color_spec = RGB;
        if(s->fps > 0.0) {
                s->frame->fps = s->fps;
        } else {
                s->frame->fps = 30;
        }
        s->frame->interlacing = PROGRESSIVE;
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;

        s->tile->data = (char *) malloc(s->tile->data_len);

        pthread_create(&s->worker_id, NULL, grab_thread, s);

        return true;
}


static void *grab_thread(void *args)
{
        struct vidcap_screen_x11_state *s = args;

        while(!s->should_exit_worker) {
                struct grabbed_data *new_item = malloc(sizeof(struct grabbed_data));

#ifdef HAVE_XFIXES
                XFixesCursorImage *cursor =
                        XFixesGetCursorImage (s->dpy);
#endif // HAVE_XFIXES
                new_item->data = XGetImage(s->dpy,s->root, 0,0, s->tile->width, s->tile->height, AllPlanes, ZPixmap);
                assert(new_item->data != NULL);

#ifdef HAVE_XFIXES
                if (cursor) {
                        uint32_t *image_data = (uint32_t *)(void *) new_item->data->data;
                        for(int x = 0; x < cursor->width; ++x) {
                                for(int y = 0; y < cursor->height; ++y) {
                                        if(cursor->x + x >= (int) s->tile->width ||
                                                        cursor->y + y >= (int) s->tile->height)
                                                continue;
                                        //image_data[x + y * s->tile->width] = cursor->pixels[x + y * cursor->width];
                                        uint_fast32_t cursor_pix = cursor->pixels[x + y * cursor->width];
                                        ///fprintf(stderr, "%d %d\n", cursor->x + x, cursor->y + y);
                                        int alpha = cursor_pix >> 24 & 0xff;
                                        int r1 = cursor_pix >> 16 & 0xff,
                                            g1 = cursor_pix >> 8 & 0xff,
                                            b1 = cursor_pix >> 0 & 0xff;
                                        uint_fast32_t image_pix = image_data[cursor->x + x + (cursor->y + y) * s->tile->width];
                                        int r2 = image_pix >> 16 & 0xff,
                                            g2 = image_pix >> 8 & 0xff,
                                            b2 = image_pix >> 0 & 0xff;
                                        float scale_image = (float) (255 - alpha)/ 255;
                                        float scale_cursor = (float) alpha / 255;

                                        image_data[cursor->x + x + (cursor->y + y) * s->tile->width] =
                                                ((int) (r1 * scale_cursor + r2 * scale_image) & 0xff) << 16 |
                                                ((int) (g1 * scale_cursor + g2 * scale_image) & 0xff) << 8 |
                                                ((int) (b1 * scale_cursor + b2 * scale_image) & 0xff) << 0;
                                }
                        }

                        XFree(cursor);
                }
#endif // HAVE_XFIXES

                new_item->next = NULL;

                pthread_mutex_lock(&s->lock);
                {
                        while(s->queue_len > QUEUE_SIZE_MAX && !s->should_exit_worker) {
                                s->worker_waiting = true;
                                pthread_cond_wait(&s->worker_cv, &s->lock);
                                s->worker_waiting = false;
                        }

                        if(s->head) {
                                s->tail->next = new_item;
                                s->tail = new_item;
                        } else {
                                s->head = s->tail = new_item;
                        }
                        s->queue_len += 1;

                        if(s->boss_waiting)
                                pthread_cond_signal(&s->boss_cv);

                }
                pthread_mutex_unlock(&s->lock);
        }

        return NULL;
}

static struct vidcap_type * vidcap_screen_x11_probe(bool verbose, void (**deleter)(void *))
{
        struct vidcap_type*		vt;
        *deleter = free;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt == NULL) {
                return NULL;
        }

        vt->name        = "screen";
        vt->description = "Grabbing screen";

        if (!verbose) {
                return vt;
        }

        vt->card_count = 1;
        vt->cards = calloc(vt->card_count, sizeof(struct device_info));
        // vt->cards[0].dev can be "" since screen cap. doesn't require parameters
        snprintf(vt->cards[0].name, sizeof vt->cards[0].name, "Screen capture");

        int framerates[] = {24, 30, 60};

        snprintf(vt->cards[0].modes[0].name,
                        sizeof vt->cards[0].modes[0].name,
                        "Unlimited fps");
        snprintf(vt->cards[0].modes[0].id,
                        sizeof vt->cards[0].modes[0].id,
                        "{\"fps\":\"\"}");

        for(unsigned i = 0; i < sizeof(framerates) / sizeof(framerates[0]); i++){
                snprintf(vt->cards[0].modes[i + 1].name,
                                sizeof vt->cards[0].modes[0].name,
                                "%d fps",
                                framerates[i]);
                snprintf(vt->cards[0].modes[i + 1].id,
                                sizeof vt->cards[0].modes[0].id,
                                "{\"fps\":\"%d\"}",
                                framerates[i]);
        }
        return vt;
}

static _Bool parse_fmt(struct vidcap_screen_x11_state *s, char *fmt) {
        char *tok, *save_ptr;
        while ((tok = strtok_r(fmt, ":", &save_ptr)) != NULL) {
                fmt = NULL;
                if (strstr(tok, "fps=") == tok) {
                        s->fps = atoi(strchr(tok, '=') + 1);
                } else if (strstr(tok, "display=") == tok) {
                        char *val = strchr(tok, '=') + 1;
                        s->req_display = strdup(val);
                        tok = strtok_r(NULL, ":", &save_ptr);
                        if (!tok) {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Display string doesn't contain the colon\n");
                                return 0;
                        }
                        s->req_display = realloc(s->req_display, strlen(s->req_display) + 1 + strlen(tok) + 1);
                        strcat(s->req_display, ":");
                        strcat(s->req_display, tok);
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unknown option \"%s\"!\n", tok);
                        return 0;
                }
        }
        return 1;
}

static int vidcap_screen_x11_init(struct vidcap_params *params, void **state)
{
        printf("vidcap_screen_init\n");

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }

        const char *fmt = vidcap_params_get_fmt(params);
        if (strcmp(fmt, "help") == 0) {
                show_help();
                return VIDCAP_INIT_NOERR;
        }

        struct vidcap_screen_x11_state *s = calloc(1, sizeof(struct vidcap_screen_x11_state));
        if(s == NULL) {
                printf("Unable to allocate screen capture state\n");
                return VIDCAP_INIT_FAIL;
        }
        s->cpu_count = get_cpu_core_count();
        gettimeofday(&s->t0, NULL);

#ifndef HAVE_XFIXES
        fprintf(stderr, "[Screen capture] Compiled without XFixes library, cursor won't be shown!\n");
#endif // ! HAVE_XFIXES

        char *fmt_cpy = strdup(fmt);
        _Bool ret = parse_fmt(s, fmt_cpy);
        free(fmt_cpy);
        if (!ret) {
                free(s);
                return VIDCAP_INIT_FAIL;
        }

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_screen_x11_finish(void *state)
{
        struct vidcap_screen_x11_state *s = (struct vidcap_screen_x11_state *) state;

        assert(s != NULL);
        pthread_mutex_lock(&s->lock);
        if(s->boss_waiting) {
                pthread_cond_signal(&s->boss_cv);
        }

        s->should_exit_worker = true;
        if(s->worker_waiting) {
                pthread_cond_signal(&s->worker_cv);
        }

        pthread_mutex_unlock(&s->lock);

        if(s->worker_id) {
                pthread_join(s->worker_id, NULL);
        }
}

static void vidcap_screen_x11_done(void *state)
{
        struct vidcap_screen_x11_state *s = (struct vidcap_screen_x11_state *) state;

        assert(s != NULL);

        vidcap_screen_x11_finish(state);

        pthread_mutex_lock(&s->lock);
        {
                while(s->queue_len > 0) {
                        struct grabbed_data *item = s->head;
                        s->head = s->head->next;
                        XDestroyImage(item->data);
                        free(item);
                        s->queue_len -= 1;
                }
        }
        pthread_mutex_unlock(&s->lock);

        if(s->tile)
                free(s->tile->data);

        vf_free(s->frame);
        free(s->req_display);
        free(s);
}

static struct video_frame * vidcap_screen_x11_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_screen_x11_state *s = (struct vidcap_screen_x11_state *) state;

        if (!s->initialized) {
                s->initialized = initialize(s);
                if (!s->initialized) {
                        fprintf(stderr, "Cannot capture screen - unable to initialize!\n");
                        exit_uv(1);
                        return NULL;
                }
        }

        *audio = NULL;

        struct grabbed_data *item = NULL;

        pthread_mutex_lock(&s->lock);
        {
                while(s->queue_len == 0) {
                        s->boss_waiting = true;
                        pthread_cond_wait(&s->boss_cv, &s->lock);
                        s->boss_waiting = false;
                }

                item = s->head;
                s->head = s->head->next;
                s->queue_len -= 1;

                if(s->worker_waiting) {
                        pthread_cond_signal(&s->worker_cv);
                }
        }
        pthread_mutex_unlock(&s->lock);

        /*
         * The more correct way is to use X pixel accessor (XGetPixel) as in previous version
         * Unfortunatelly, this approach is damn slow. Current approach might be incorrect in
         * some configurations, but seems to work currently. To be corrected if there is an
         * opposite case.
         */
        parallel_pix_conv(s->tile->height, s->tile->data, vc_get_linesize(s->tile->width, RGB), &item->data->data[0], vc_get_linesize(s->tile->width, RGBA), vc_copylineABGRtoRGB, s->cpu_count);

        XDestroyImage(item->data);
        free(item);

        if(s->fps > 0.0) {
                struct timeval cur_time;

                gettimeofday(&cur_time, NULL);
                while(tv_diff_usec(cur_time, s->prev_time) < 1000000.0 / s->frame->fps) {
                        gettimeofday(&cur_time, NULL);
                }
                s->prev_time = cur_time;
        }

        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);        
        if (seconds >= 5) {
                float fps  = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[screen capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = s->t;
                s->frames = 0;
        }

        s->frames++;

        return s->frame;
}

static const struct video_capture_info vidcap_screen_x11_info = {
        vidcap_screen_x11_probe,
        vidcap_screen_x11_init,
        vidcap_screen_x11_done,
        vidcap_screen_x11_grab,
        false
};

REGISTER_MODULE(screen, &vidcap_screen_x11_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

