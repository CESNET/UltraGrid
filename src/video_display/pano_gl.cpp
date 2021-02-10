/**
 * @file   video_display/pano_gl.cpp
 * @author Martin Piatka    <piatka@cesnet.cz>
 */
/*
 * Copyright (c) 2010-2021 CESNET, z. s. p. o.
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
#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <chrono>

#include <assert.h>
#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "video.h"
#include "video_display.h"
#include "video_display/splashscreen.h"

#include "opengl_utils.hpp"
#include "utils/profile_timer.hpp"

#define MAX_BUFFER_SIZE   1

struct state_vr{
        Sdl_window window = Sdl_window(true);

        bool running = false;
        bool fs = false;

        unsigned sdl_frame_event;
        unsigned sdl_redraw_event;

        video_desc current_desc;
        int buffered_frames_count;

        Scene scene;

        int threshold_fps = 60; //Target fps for redraws caused by interaction

        SDL_TimerID redraw_timer = 0;
        bool redraw_needed = false;

        std::chrono::steady_clock::time_point last_frame;

        std::mutex lock;
        std::condition_variable frame_consumed_cv;
        std::queue<video_frame *> free_frame_queue;
};

static void * display_panogl_init(struct module *parent, const char *fmt, unsigned int flags) {
        state_vr *s = new state_vr();
        s->sdl_frame_event = SDL_RegisterEvents(1);
        s->sdl_redraw_event = SDL_RegisterEvents(1);

        return s;
}

static void draw(state_vr *s){
        PROFILE_FUNC;
        auto now = std::chrono::steady_clock::now();
        s->last_frame = now;

        glClear(GL_COLOR_BUFFER_BIT);
        s->scene.render(s->window.width, s->window.height);

        SDL_GL_SwapWindow(s->window.sdl_window);
}

static Uint32 redraw_callback(Uint32 interval, void *param){
        int event_id = *static_cast<int *>(param);
        SDL_Event event;
        event.type = event_id;
        SDL_PushEvent(&event);

        return interval;
}

static void redraw(state_vr *s,
                bool video_framerate_over_threshold = false,
                bool triggered_by_frame = false)
{
        if(video_framerate_over_threshold && triggered_by_frame){
                /* If video framerate is over the threshold we draw the
                 * incoming frames immediately. Since this draw already
                 * contains changes caused by interaction (e.g. moving the
                 * camera) we cancel the redraw timer
                 */
                if(s->redraw_timer){
                        SDL_RemoveTimer(s->redraw_timer);
                        s->redraw_timer = 0;
                        s->redraw_needed = false;
                }
                draw(s);
        } else {
                if(!s->redraw_timer){
                        /* Redrawing due to interaction is delayed, so that if
                         * a fresh video frame arrives very soon after the
                         * interaction it can be drawn together with the
                         * changes caused by interaction
                         */
                        s->redraw_timer = SDL_AddTimer(1000 / s->threshold_fps,
                                        redraw_callback, &s->sdl_redraw_event);

                        if(triggered_by_frame)
                                draw(s);
                }
                s->redraw_needed = !triggered_by_frame;
        }
}

static void handle_window_event(state_vr *s, SDL_Event *event){
        if(event->window.event == SDL_WINDOWEVENT_RESIZED){
                glViewport(0, 0, event->window.data1, event->window.data2);
                s->window.width = event->window.data1;
                s->window.height = event->window.data2;
                redraw(s);
        }
}

static void handle_keyboard_event(state_vr *s, SDL_Event *event){
        if(event->key.type == SDL_KEYDOWN){
                switch(event->key.keysym.sym){
                        case SDLK_f:
                                s->fs = !s->fs;
                                SDL_SetWindowFullscreen(s->window.sdl_window, s->fs ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                                redraw(s);
                                break;
                        case SDLK_q:
                                exit_uv(0);
                                break;
                        default:
                                break;
                }
        }
}

static void handle_user_event(state_vr *s, SDL_Event *event){
        if(event->type == s->sdl_frame_event){
                std::unique_lock<std::mutex> lk(s->lock);
                s->buffered_frames_count -= 1;
                lk.unlock();
                s->frame_consumed_cv.notify_one();

                video_frame *frame = static_cast<video_frame *>(event->user.data1);

                if(frame){
                        s->scene.put_frame(frame);

                        lk.lock();
                        s->free_frame_queue.push(frame);
                        lk.unlock();
                } else {
                        //poison
                        s->running = false;
                        return;
                }
                redraw(s, frame->fps >= s->threshold_fps, true);
        } else if(event->type == s->sdl_redraw_event){
                if(s->redraw_needed){
                        draw(s);
                        s->redraw_needed = false;
                } else {
                        SDL_RemoveTimer(s->redraw_timer);
                        s->redraw_timer = 0;
                }
        }
}

static void display_panogl_run(void *state) {
        state_vr *s = static_cast<state_vr *>(state);

        draw(s);

        s->running = true;
        while(s->running){
                SDL_Event event;
                if (!SDL_WaitEvent(&event)) {
                        continue;
                }

                switch(event.type){
                        case SDL_MOUSEMOTION:
                                if(event.motion.state & SDL_BUTTON_LMASK){
                                        s->scene.rotate(event.motion.xrel / 8.f,
                                                        event.motion.yrel / 8.f);

                                        redraw(s);
                                }
                                break;
                        case SDL_MOUSEWHEEL:
                                s->scene.fov -= event.wheel.y;
                                redraw(s);
                                break;
                        case SDL_WINDOWEVENT:
                                handle_window_event(s, &event);
                                break;
                        case SDL_KEYDOWN:
                                handle_keyboard_event(s, &event);
                                break;
                        case SDL_QUIT:
                                exit_uv(0);
                                break;
                        default: 
                                if(event.type >= SDL_USEREVENT)
                                        handle_user_event(s, &event);
                                break;
                }
        }
}

static void display_panogl_done(void *state) {
        state_vr *s = static_cast<state_vr *>(state);

        delete s;
}

static struct video_frame * display_panogl_getf(void *state) {
        struct state_vr *s = static_cast<state_vr *>(state);

        std::lock_guard<std::mutex> lock(s->lock);

        while (s->free_frame_queue.size() > 0) {
                struct video_frame *buffer = s->free_frame_queue.front();
                s->free_frame_queue.pop();
                if (video_desc_eq(video_desc_from_frame(buffer), s->current_desc)) {
                        return buffer;
                } else {
                        vf_free(buffer);
                }
        }

        return vf_alloc_desc_data(s->current_desc);
}

static int display_panogl_putf(void *state, struct video_frame *frame, int nonblock) {
        PROFILE_FUNC;
        struct state_vr *s = static_cast<state_vr *>(state);

        if (nonblock == PUTF_DISCARD) {
                vf_free(frame);
                return 0;
        }

        std::unique_lock<std::mutex> lk(s->lock);
        if (s->buffered_frames_count >= MAX_BUFFER_SIZE && nonblock == PUTF_NONBLOCK
                        && frame != NULL) {
                vf_free(frame);
                printf("1 frame(s) dropped!\n");
                return 1;
        }
        s->frame_consumed_cv.wait(lk, [s]{return s->buffered_frames_count < MAX_BUFFER_SIZE;});
        s->buffered_frames_count += 1;
        lk.unlock();
        SDL_Event event;
        event.type = s->sdl_frame_event;
        event.user.data1 = frame;
        PROFILE_DETAIL("push frame event");
        SDL_PushEvent(&event);

        return 0;
}

static int display_panogl_reconfigure(void *state, struct video_desc desc) {
        state_vr *s = static_cast<state_vr *>(state);

        s->current_desc = desc;
        return 1;
}

static int display_panogl_get_property(void *state, int property, void *val, size_t *len) {
        UNUSED(state);
        codec_t codecs[] = {
                RGBA,
                RGB,
                UYVY
        };
        enum interlacing_t supported_il_modes[] = {PROGRESSIVE};
        int rgb_shift[] = {0, 8, 16};

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if(sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                        } else {
                                return FALSE;
                        }

                        *len = sizeof(codecs);
                        break;
                case DISPLAY_PROPERTY_RGB_SHIFT:
                        if(sizeof(rgb_shift) > *len) {
                                return FALSE;
                        }
                        memcpy(val, rgb_shift, sizeof(rgb_shift));
                        *len = sizeof(rgb_shift);
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = PITCH_DEFAULT;
                        *len = sizeof(int);
                        break;
                case DISPLAY_PROPERTY_SUPPORTED_IL_MODES:
                        if(sizeof(supported_il_modes) <= *len) {
                                memcpy(val, supported_il_modes, sizeof(supported_il_modes));
                        } else {
                                return FALSE;
                        }
                        *len = sizeof(supported_il_modes);
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}


static const struct video_display_info display_panogl_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
                strcpy((*available_cards)[0].id, "pano_gl");
                strcpy((*available_cards)[0].name, "Panorama Gl SW display");
                (*available_cards)[0].repeatable = true;
        },
        display_panogl_init,
        display_panogl_run,
        display_panogl_done,
        display_panogl_getf,
        display_panogl_putf,
        display_panogl_reconfigure,
        display_panogl_get_property,
        NULL,
        NULL,
        DISPLAY_NEEDS_MAINLOOP,
};

REGISTER_MODULE(pano_gl, &display_panogl_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
