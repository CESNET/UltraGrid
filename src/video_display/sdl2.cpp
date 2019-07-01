/**
 * @file   video_display/sdl2.cpp
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2019 CESNET, z. s. p. o.
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
 * Missing from SDL1:
 * * audio (would be perhaps better as an audio playback device)
 * * CoUniverse related things - change title (via message), fixed size,
 *   predefined size etc. Perhaps no longer needed.
 * * autorelease_pool (macOS) - perhaps not needed
 * @todo
 * * frames are copied, better would be to preallocate textures and set
 *   video_frame::tiles::data to SDL_LockTexture() pixels. This, however,
 *   needs decoder to use either pitch (toggling fullscreen or resize) or
 *   forcing decoder to reconfigure pitch.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "rang.hpp"
#include "video_display.h"
#include "video_display/splashscreen.h"
#include "video.h"

#include <SDL2/SDL.h>

#include <cstdint>
#include <condition_variable>
#include <list>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <utility> // pair

#define MAGIC_SDL2   0x3cc234a1
#define MAX_BUFFER_SIZE   1

using rang::fg;
using rang::style;
using namespace std;
using namespace std::chrono;

#define SDL_USER_NEWFRAME SDL_USEREVENT

struct state_sdl2 {
        uint32_t                magic{MAGIC_SDL2};

        chrono::steady_clock::time_point tv{chrono::steady_clock::now()};
        unsigned long long int  frames{0};

        int                     display_idx{0};
        int                     x{SDL_WINDOWPOS_UNDEFINED},
                                y{SDL_WINDOWPOS_UNDEFINED};
        int                     renderer_idx{-1};
        SDL_Window             *window{nullptr};
        SDL_Renderer           *renderer{nullptr};
        SDL_Texture            *texture{nullptr};

        bool                    fs{false};
        bool                    deinterlace{false};
        bool                    vsync{true};
        bool                    fixed_size{false};
        int                     fixed_w{0}, fixed_h{0};
        uint32_t                window_flags{0}; ///< user requested flags

        mutex                   lock;
        condition_variable      frame_consumed_cv;
        int                     buffered_frames_count{0};

        struct video_desc       current_desc{};
        struct video_desc       current_display_desc{};
        struct video_frame     *last_frame{nullptr};

        queue<struct video_frame *> free_frame_queue;
};

static void show_help(void);
static int display_sdl_putf(void *state, struct video_frame *frame, int nonblock);
static int display_sdl_reconfigure(void *state, struct video_desc desc);
static int display_sdl_reconfigure_real(void *state, struct video_desc desc);

static void display_frame(struct state_sdl2 *s, struct video_frame *frame)
{
        if (!frame) {
                return;
        }
        if (!video_desc_eq(video_desc_from_frame(frame), s->current_display_desc)) {
                if (!display_sdl_reconfigure_real(s, video_desc_from_frame(frame))) {
                        goto free_frame;
                }
        }

        if (!s->deinterlace) {
                SDL_UpdateTexture(s->texture, NULL, frame->tiles[0].data, vc_get_linesize(frame->tiles[0].width, frame->color_spec));
        } else {
		unsigned char *pixels;
		int pitch;
		SDL_LockTexture(s->texture, NULL, (void **) &pixels, &pitch);
		vc_deinterlace_ex((unsigned char *) frame->tiles[0].data, vc_get_linesize(frame->tiles[0].width, frame->color_spec), pixels, pitch, frame->tiles[0].height);
		SDL_UnlockTexture(s->texture);
	}

        SDL_RenderCopy(s->renderer, s->texture, NULL, NULL);
        SDL_RenderPresent(s->renderer);

free_frame:
        if (frame == s->last_frame) {
                return; // we are only redrawing on window resize
        }

        if (s->last_frame) {
                s->lock.lock();
                s->free_frame_queue.push(s->last_frame);
                s->lock.unlock();
        }
        s->last_frame = frame;

        s->frames++;
        auto tv = chrono::steady_clock::now();
        double seconds = duration_cast<duration<double>>(tv - s->tv).count();
        if (seconds > 5) {
                double fps = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[SDL] %d frames in %g seconds = %g FPS\n",
                                s->frames, seconds, fps);
                s->tv = tv;
                s->frames = 0;
        }
}

static void display_sdl_run(void *arg)
{
        struct state_sdl2 *s = (struct state_sdl2 *) arg;
        bool should_exit_sdl = false;

        while (!should_exit_sdl) {
                SDL_Event sdl_event;
                if (SDL_WaitEvent(&sdl_event)) {
                        switch (sdl_event.type) {
                        case SDL_USER_NEWFRAME:
                        {
                                std::unique_lock<std::mutex> lk(s->lock);
                                s->buffered_frames_count -= 1;
                                lk.unlock();
                                s->frame_consumed_cv.notify_one();
                                if (sdl_event.user.data1 != NULL) {
                                        display_frame(s, (struct video_frame *) sdl_event.user.data1);
                                } else { // poison pill received
                                        should_exit_sdl = true;
                                }
                                break;
                        }
                        case SDL_KEYDOWN:
                                switch (sdl_event.key.keysym.sym) {
                                case SDLK_d:
                                        s->deinterlace = !s->deinterlace;
                                        log_msg(LOG_LEVEL_INFO, "Deinterlacing: %s\n",
                                                        s->deinterlace ? "ON" : "OFF");
                                        break;
                                case SDLK_f:
                                        s->fs = !s->fs;
                                        SDL_SetWindowFullscreen(s->window, s->fs ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                                        break;
                                case SDLK_q:
                                        exit_uv(0);
                                        break;
                                default:
                                        log_msg(LOG_LEVEL_DEBUG, "[SDL] Unhandled key: %s\n",
                                                        SDL_GetKeyName(sdl_event.key.keysym.sym));
                                        break;
                                }
                                break;
                        case SDL_WINDOWEVENT:
                                if (sdl_event.window.event == SDL_WINDOWEVENT_RESIZED
                                                || sdl_event.window.event == SDL_WINDOWEVENT_EXPOSED) {
                                        // clear both buffers
                                        SDL_RenderClear(s->renderer);
                                        display_frame(s, s->last_frame);
                                        SDL_RenderClear(s->renderer);
                                        display_frame(s, s->last_frame);
                                }
                                break;
                        case SDL_QUIT:
                                exit_uv(0);
                                break;
                        }
                }
        }
}

static void show_help(void)
{
        SDL_Init(0);
        printf("SDL options:\n");
        cout << style::bold << fg::red << "\t-d sdl" << fg::reset << "[[:fs|:d|:display=<didx>|:driver=<drv>|:novsync|:renderer=<ridx>|:nodecorate|:fixed_size[=WxH]|:window_flags=<f>|:pos=<x>,<y>]*|:help]\n" << style::reset;
        printf("\twhere:\n");
        cout << style::bold <<"\t\t       d" << style::reset << " - deinterlace\n";
        cout << style::bold <<"\t\t      fs" << style::reset << " - fullscreen\n";
        cout << style::bold <<"\t\t  <didx>" << style::reset << " - display index\n";
        cout << style::bold <<"\t\t   <drv>" << style::reset << " - one of following: ";
        for (int i = 0; i < SDL_GetNumVideoDrivers(); ++i) {
                cout << (i == 0 ? "" : ", ") << style::bold << SDL_GetVideoDriver(i) << style::reset;
        }
        cout << style::bold <<"\n";
        cout << style::bold <<"\t         novsync" << style::reset << " - disable sync on VBlank\n";
        cout << style::bold <<"\t      nodecorate" << style::reset << " - disable window border\n";
        cout << style::bold <<"\tfixed_size[=WxH]" << style::reset << " - use fixed sized window\n";
        cout << style::bold <<"\t    window_flags" << style::reset << " - flags to be passed to SDL_CreateWindow (use prefix 0x for hex)\n";
        cout << style::bold <<"\t\t  <ridx>" << style::reset << " - renderer index: ";
        for (int i = 0; i < SDL_GetNumRenderDrivers(); ++i) {
                SDL_RendererInfo renderer_info;
                if (SDL_GetRenderDriverInfo(i, &renderer_info) == 0) {
                        cout << (i == 0 ? "" : ", ") << style::bold << i << style::reset << " - " << style::bold << renderer_info.name << style::reset;
                }
        }
        printf("\n");
        cout << "\n\tKeyboard shortcuts:\n";
        list<pair<string, string>> sh{{"'d'", "toggle deinterlace"},
                {"'f'", "toggle fullscreen"}, {"'q'", "quit"}};
        for (auto i : sh) {
                cout << style::bold << "\t\t" << i.first << style::reset << "\t - " << i.second << "\n";
        }
        SDL_Quit();
}

static int display_sdl_reconfigure(void *state, struct video_desc desc)
{
        struct state_sdl2 *s = (struct state_sdl2 *) state;

        s->current_desc = desc;
        return 1;
}

static const unordered_map<codec_t, uint32_t, hash<int>> pf_mapping = {
        { UYVY, SDL_PIXELFORMAT_UYVY },
        { YUYV, SDL_PIXELFORMAT_YUY2 },
        { RGB, SDL_PIXELFORMAT_RGB24 },
        { BGR, SDL_PIXELFORMAT_BGR24 },
#if SDL_COMPILEDVERSION >= SDL_VERSIONNUM(2, 0, 5)
        { RGBA, SDL_PIXELFORMAT_RGBA32 },
#else
        { RGBA, SDL_PIXELFORMAT_ABGR8888 },
#endif
};

static bool create_texture(struct state_sdl2 *s, struct video_desc desc) {
        uint32_t format;
        auto it = pf_mapping.find(desc.color_spec);
        if (it == pf_mapping.end()) {
                abort();
        }
        format = it->second;

        if (s->texture) {
                SDL_DestroyTexture(s->texture);
        }

        s->texture = SDL_CreateTexture(s->renderer, format, SDL_TEXTUREACCESS_STREAMING, desc.width, desc.height);
        if (!s->texture) {
                log_msg(LOG_LEVEL_ERROR, "[SDL] Unable to create texture: %s\n", SDL_GetError);
                return false;
        }

        return true;
}

static int display_sdl_reconfigure_real(void *state, struct video_desc desc)
{
        struct state_sdl2 *s = (struct state_sdl2 *)state;

        log_msg(LOG_LEVEL_NOTICE, "[SDL] Reconfigure to size %dx%d\n", desc.width,
                        desc.height);

        s->current_display_desc = desc;

        if (s->fixed_size && s->window) {
                SDL_RenderSetLogicalSize(s->renderer, desc.width, desc.height);
                return create_texture(s, desc);
        }

        if (s->window) {
                SDL_DestroyWindow(s->window);
        }
        int flags = s->window_flags | SDL_WINDOW_RESIZABLE;
        if (s->fs) {
                flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
        }
        const char *window_title = "UltraGrid - SDL2 Display";
        if (get_commandline_param("window-title")) {
                window_title = get_commandline_param("window-title");
        }
        int width = s->fixed_w ? s->fixed_w : desc.width;
        int height = s->fixed_h ? s->fixed_h : desc.height;
        int x = s->x == SDL_WINDOWPOS_UNDEFINED ? SDL_WINDOWPOS_CENTERED_DISPLAY(s->display_idx) : s->x;
        int y = s->y == SDL_WINDOWPOS_UNDEFINED ? SDL_WINDOWPOS_CENTERED_DISPLAY(s->display_idx) : s->y;
        s->window = SDL_CreateWindow(window_title, x, y, width, height, flags);
        if (!s->window) {
                log_msg(LOG_LEVEL_ERROR, "[SDL] Unable to create window: %s\n", SDL_GetError());
                return FALSE;
        }

        if (s->renderer) {
                SDL_DestroyRenderer(s->renderer);
        }
        s->renderer = SDL_CreateRenderer(s->window, s->renderer_idx, SDL_RENDERER_ACCELERATED | (s->vsync ? SDL_RENDERER_PRESENTVSYNC : 0));
        if (!s->renderer) {
                log_msg(LOG_LEVEL_ERROR, "[SDL] Unable to create renderer: %s\n", SDL_GetError());
                return FALSE;
        }
        SDL_RendererInfo renderer_info;
        if (SDL_GetRendererInfo(s->renderer, &renderer_info) == 0) {
                log_msg(LOG_LEVEL_NOTICE, "[SDL] Using renderer: %s\n", renderer_info.name);
        }

        SDL_SetHint(SDL_HINT_RENDER_SCALE_QUALITY, "linear");
        SDL_RenderSetLogicalSize(s->renderer, desc.width, desc.height);

        if (!create_texture(s, desc)) {
                return FALSE;
        }

        return TRUE;
}

/**
 * Load splashscreen
 * Function loads graphic data from header file "splashscreen.h", where are
 * stored splashscreen data in RGB format.
 */
static void loadSplashscreen(struct state_sdl2 *s) {
        struct video_desc desc;

        desc.width = 512;
        desc.height = 512;
        desc.color_spec = RGBA;
        desc.interlacing = PROGRESSIVE;
        desc.fps = 1;
        desc.tile_count = 1;

        display_sdl_reconfigure(s, desc);

        struct video_frame *frame = vf_alloc_desc_data(desc);

        const char *data = splash_data;
        memset(frame->tiles[0].data, 0, frame->tiles[0].data_len);
        for (unsigned int y = 0; y < splash_height; ++y) {
                char *line = frame->tiles[0].data;
                line += vc_get_linesize(frame->tiles[0].width,
                                frame->color_spec) *
                        (((frame->tiles[0].height - splash_height) / 2) + y);
                line += vc_get_linesize(
                                (frame->tiles[0].width - splash_width)/2,
                                frame->color_spec);
                for (unsigned int x = 0; x < splash_width; ++x) {
                        HEADER_PIXEL(data,line);
                        line += 4;
                }
        }

        display_sdl_putf(s, frame, PUTF_BLOCKING);
}

static void *display_sdl_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(parent);
        if (flags & DISPLAY_FLAG_AUDIO_ANY) {
                log_msg(LOG_LEVEL_ERROR, "UltraGrid SDL2 module currently doesn't support audio!\n");
                return NULL;
        }
        const char *driver = NULL;
        struct state_sdl2 *s = new state_sdl2{};

        if (fmt == NULL) {
                fmt = "";
        }
        char *tmp = (char *) alloca(strlen(fmt) + 1);
        strcpy(tmp, fmt);
        char *tok, *save_ptr;
        while((tok = strtok_r(tmp, ":", &save_ptr)))
        {
                if (strcmp(tok, "d") == 0) {
                        s->deinterlace = true;
                } else if (strncmp(tok, "display=", strlen("display=")) == 0) {
                        s->display_idx = atoi(tok + strlen("display="));
                } else if (strncmp(tok, "driver=", strlen("driver=")) == 0) {
                        driver = tok + strlen("driver=");
                } else if (strcmp(tok, "fs") == 0) {
                        s->fs = true;
                } else if (strcmp(tok, "help") == 0) {
                        show_help();
                        delete s;
                        return &display_init_noerr;
                } else if (strcmp(tok, "novsync") == 0) {
                        s->vsync = false;
                } else if (strcmp(tok, "nodecorate") == 0) {
                        s->window_flags |= SDL_WINDOW_BORDERLESS;
		} else if (strncmp(tok, "fixed_size", strlen("fixed_size")) == 0) {
			s->fixed_size = true;
			if (strncmp(tok, "fixed_size=", strlen("fixed_size=")) == 0) {
				char *size = tok + strlen("fixed_size=");
				if (strchr(size, 'x')) {
					s->fixed_w = atoi(size);
					s->fixed_h = atoi(strchr(size, 'x') + 1);
				}
			}
                } else if (strstr(tok, "window_flags=") == tok) {
                        int f;
                        if (sscanf(tok + strlen("window_flags="), "%i", &f) != 1) {
                                log_msg(LOG_LEVEL_ERROR, "Wrong window_flags: %s\n", tok);
                                delete s;
                                return NULL;
                        }
                        s->window_flags |= f;
		} else if (strstr(tok, "pos=") == tok) {
                        tok += strlen("pos=");
                        if (strchr(tok, ',') == nullptr) {
                                log_msg(LOG_LEVEL_ERROR, "[SDL] position: %s\n", tok);
                                delete s;
                                return NULL;
                        }
                        s->x = atoi(tok);
                        s->y = atoi(strchr(tok, ',') + 1);
                } else if (strncmp(tok, "renderer=", strlen("renderer=")) == 0) {
                        s->renderer_idx = atoi(tok + strlen("renderer="));
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[SDL] Wrong option: %s\n", tok);
                        delete s;
                        return NULL;
                }
                tmp = NULL;
        }

        int ret = SDL_Init(SDL_INIT_EVENTS);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize SDL2: %s\n", SDL_GetError());
                delete s;
                return NULL;
        }
        ret = SDL_VideoInit(driver);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize SDL2 video: %s\n", SDL_GetError());
                delete s;
                return NULL;
        }
        log_msg(LOG_LEVEL_NOTICE, "[SDL] Using driver: %s\n", SDL_GetCurrentVideoDriver());

        SDL_ShowCursor(SDL_DISABLE);
        SDL_DisableScreenSaver();

        loadSplashscreen(s);

        log_msg(LOG_LEVEL_NOTICE, "SDL2 initialized successfully.\n");

        return (void *) s;
}

static void display_sdl_done(void *state)
{
        struct state_sdl2 *s = (struct state_sdl2 *)state;

        assert(s->magic == MAGIC_SDL2);

        vf_free(s->last_frame);

        while (s->free_frame_queue.size() > 0) {
                struct video_frame *buffer = s->free_frame_queue.front();
                s->free_frame_queue.pop();
                vf_free(buffer);
        }

        if (s->texture) {
                SDL_DestroyTexture(s->texture);
        }

        if (s->renderer) {
                SDL_DestroyRenderer(s->renderer);
        }

        if (s->window) {
                SDL_DestroyWindow(s->window);
        }

        SDL_ShowCursor(SDL_ENABLE);

        SDL_VideoQuit();
        SDL_QuitSubSystem(SDL_INIT_EVENTS);
        SDL_Quit();

        delete s;
}

static struct video_frame *display_sdl_getf(void *state)
{
        struct state_sdl2 *s = (struct state_sdl2 *)state;
        assert(s->magic == MAGIC_SDL2);

        lock_guard<mutex> lock(s->lock);

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

static int display_sdl_putf(void *state, struct video_frame *frame, int nonblock)
{
        struct state_sdl2 *s = (struct state_sdl2 *)state;

        assert(s->magic == MAGIC_SDL2);

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
        event.type = SDL_USER_NEWFRAME;
        event.user.data1 = frame;
        SDL_PushEvent(&event);

        return 0;
}

static int display_sdl_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        codec_t codecs[pf_mapping.size()];

        int i = 0;
        for (auto item : pf_mapping) {
                codecs[i++] = item.first;
        }

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (sizeof(codecs) <= *len) {
                                memcpy(val, codecs, sizeof(codecs));
                                *len = sizeof(codecs);
                        } else {
                                return FALSE;
                        }
                        break;
                default:
                        return FALSE;
        }
        return TRUE;
}

static const struct video_display_info display_sdl2_info = {
        [](struct device_info **available_cards, int *count, void (**deleter)(void *)) {
                UNUSED(deleter);
                *count = 1;
                *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
                strcpy((*available_cards)[0].id, "sdl");
                strcpy((*available_cards)[0].name, "SDL2 SW display");
                (*available_cards)[0].repeatable = true;
        },
        display_sdl_init,
        display_sdl_run,
        display_sdl_done,
        display_sdl_getf,
        display_sdl_putf,
        display_sdl_reconfigure,
        display_sdl_get_property,
        NULL,
        NULL
};

REGISTER_MODULE(sdl, &display_sdl2_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

