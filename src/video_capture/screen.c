/*
 * FILE:    screen.c
 * AUTHORS: Martin Benes     <martinbenesh@gmail.com>
 *          Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *          Petr Holub       <hopet@ics.muni.cz>
 *          Milos Liska      <xliska@fi.muni.cz>
 *          Jiri Matela      <matela@ics.muni.cz>
 *          Dalibor Matura   <255899@mail.muni.cz>
 *          Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the CESNET nor the names of its contributors may be
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
 *
 */


#include "host.h"
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "debug.h"
#include "video_codec.h"
#include "video_capture.h"

#include "tv.h"

#include "video_capture/screen.h"
#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include "video_display.h"
#include "video.h"

#include <pthread.h>

#ifdef HAVE_MACOSX
#include <OpenGL/OpenGL.h>
#include <OpenGL/gl.h>
#include <Carbon/Carbon.h>
#else
#include <X11/Xlib.h>
#ifdef HAVE_XFIXES
#include <X11/extensions/Xfixes.h>
#endif // HAVE_XFIXES
#include <X11/Xutil.h>
#include "x11_common.h"
#endif

#define QUEUE_SIZE_MAX 3

/* prototypes of functions defined in this module */
static void show_help(void);
#ifdef HAVE_LINUX
static void *grab_thread(void *args);
#endif // HAVE_LINUX

static volatile bool should_exit = false;

static void show_help()
{
        printf("Screen capture\n");
        printf("Usage\n");
        printf("\t-t screen[:fps=<fps>]\n");
        printf("\t\t<fps> - preferred grabbing fps (otherwise unlimited)\n");
}

/* defined in main.c */
extern int uv_argc;
extern char **uv_argv;

static struct vidcap_screen_state *state;

#ifdef HAVE_LINUX
struct grabbed_data;

struct grabbed_data {
        XImage *data;
        struct grabbed_data *next;
};
#endif

struct vidcap_screen_state {
        struct video_frame       *frame; 
        struct tile       *tile; 
        int frames;
        struct       timeval t, t0;
#ifdef HAVE_MACOSX
        CGDirectDisplayID display;
#else
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
#endif

        struct timeval prev_time;

        double fps;

};

pthread_once_t initialized = PTHREAD_ONCE_INIT;

static void initialize() {
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);


#ifndef HAVE_MACOSX
        XWindowAttributes wa;

        x11_lock();

        s->dpy = x11_acquire_display();

        x11_unlock();

        s->root = DefaultRootWindow(s->dpy);

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

#else
        s->display = CGMainDisplayID();
        CGImageRef image = CGDisplayCreateImage(s->display);

        s->tile->width = CGImageGetWidth(image);
        s->tile->height = CGImageGetHeight(image);
        CFRelease(image);
#endif

        s->frame->color_spec = RGB;
        if(s->fps > 0.0) {
                s->frame->fps = s->fps;
        } else {
                s->frame->fps = 30;
        }
        s->frame->interlacing = PROGRESSIVE;
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;

#ifndef HAVE_MACOSX
        s->tile->data = (char *) malloc(s->tile->data_len);

        pthread_create(&s->worker_id, NULL, grab_thread, s);
#else
        s->tile->data = (char *) malloc(s->tile->data_len);
#endif

        return;

        goto error; // dummy use (otherwise compiler would complain about unreachable code (Mac)
error:
        fprintf(stderr, "[Screen cap.] Initialization failed!\n");
        exit_uv(128);
}


#ifdef HAVE_LINUX
static void *grab_thread(void *args)
{
        struct vidcap_screen_state *s = args;

        while(!s->should_exit_worker) {
                struct grabbed_data *new_item = malloc(sizeof(struct grabbed_data));

#ifdef HAVE_XFIXES
                XFixesCursorImage *cursor =
                        XFixesGetCursorImage (s->dpy);
#endif // HAVE_XFIXES
                new_item->data = XGetImage(s->dpy,s->root, 0,0, s->tile->width, s->tile->height, AllPlanes, ZPixmap);

#ifdef HAVE_XFIXES
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
#endif // HAVE_LINUX

struct vidcap_type * vidcap_screen_probe(void)
{
        struct vidcap_type*		vt;

        vt = (struct vidcap_type *) malloc(sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->id          = VIDCAP_SCREEN_ID;
                vt->name        = "screen";
                vt->description = "Grabbing screen";
        }
        return vt;
}

void * vidcap_screen_init(char *init_fmt, unsigned int flags)
{
        struct vidcap_screen_state *s;

        printf("vidcap_screen_init\n");

        UNUSED(flags);

        state = s = (struct vidcap_screen_state *) malloc(sizeof(struct vidcap_screen_state));
        if(s == NULL) {
                printf("Unable to allocate screen capture state\n");
                return NULL;
        }

        gettimeofday(&s->t0, NULL);

        s->fps = 0.0;

        s->frame = NULL;
        s->tile = NULL;

#ifdef HAVE_LINUX
        s->worker_id = 0;
#ifndef HAVE_XFIXES
        fprintf(stderr, "[Screen capture] Compiled without XFixes library, cursor won't be shown!\n");
#endif // ! HAVE_XFIXES
#endif // HAVE_LINUX

        s->prev_time.tv_sec = 
                s->prev_time.tv_usec = 0;


        s->frames = 0;

        if(init_fmt) {
                if (strcmp(init_fmt, "help") == 0) {
                        show_help();
                        return NULL;
                } else if (strncasecmp(init_fmt, "fps=", strlen("fps=")) == 0) {
                        s->fps = atoi(init_fmt + strlen("fps="));
                }
        }

        return s;
}

void vidcap_screen_finish(void *state)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        assert(s != NULL);
        should_exit = true;
#ifdef HAVE_LINUX
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
#endif // HAVE_LINUX
}

void vidcap_screen_done(void *state)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        assert(s != NULL);
#ifdef HAVE_LINUX
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
#endif

        if(s->tile) {
#ifdef HAVE_MACOS_X
                free(s->tile->data);
#endif
        }
        vf_free(s->frame);
        free(s);
}

struct video_frame * vidcap_screen_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_screen_state *s = (struct vidcap_screen_state *) state;

        pthread_once(&initialized, initialize);

        *audio = NULL;

#ifndef HAVE_MACOSX

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
        vc_copylineABGRtoRGB((unsigned char *) s->tile->data,
                        (unsigned char *) &item->data->data[0], s->tile->data_len);

        XDestroyImage(item->data);
        free(item);
#else
        CGImageRef image = CGDisplayCreateImage(s->display);
        CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(image));
        const unsigned char *pixels = CFDataGetBytePtr(data);

        int linesize = s->tile->width * 4;
        int y;
        unsigned char *dst = (unsigned char *) s->tile->data;
        const unsigned char *src = (const unsigned char *) pixels;
        for(y = 0; y < (int) s->tile->height; ++y) {
                vc_copylineRGBA (dst, src, linesize, 16, 8, 0);
                src += linesize;
                dst += linesize;
        }

        CFRelease(data);
        CFRelease(image);

#endif

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
                fprintf(stderr, "[screen capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = s->t;
                s->frames = 0;
        }

        s->frames++;

        return s->frame;
}

