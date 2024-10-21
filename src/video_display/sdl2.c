/**
 * @file   video_display/sdl2.c
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2023 CESNET, z. s. p. o.
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
 * @todo
 * Missing from SDL1:
 * * audio (would be perhaps better as an audio playback device)
 * * autorelease_pool (macOS) - perhaps not needed
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H

#include "debug.h"
#include "host.h"
#include "keyboard_control.h" // K_*
#include "lib_common.h"
#include "messaging.h"
#include "module.h"
#include "utils/color_out.h"
#include "utils/list.h"
#include "video_display.h"
#include "video.h"

/// @todo remove the defines when no longer needed
#ifdef __arm64__
#define SDL_DISABLE_MMINTRIN_H 1
#define SDL_DISABLE_IMMINTRIN_H 1
#endif // defined __arm64__
#if __has_include(<SDL2/SDL.h>)
#include <SDL2/SDL.h>
#else
#include <SDL.h>
#endif

#define SDL2_DEINTERLACE_IMPOSSIBLE_MSG_ID 0x327058e5
#define MAGIC_SDL2   0x3cc234a1
#define BUFFER_COUNT   2
#define MOD_NAME "[SDL] "

struct state_sdl2;

static void show_help(void);
static void display_frame(struct state_sdl2 *s, struct video_frame *frame);
static struct video_frame *display_sdl2_getf(void *state);
static void display_sdl2_new_message(struct module *mod);
static int display_sdl2_reconfigure_real(void *state, struct video_desc desc);
static void loadSplashscreen(struct state_sdl2 *s);

enum deint { DEINT_OFF, DEINT_ON, DEINT_FORCE };

struct state_sdl2 {
        struct module           mod;

        int                     texture_pitch;

        Uint32                  sdl_user_new_frame_event;
        Uint32                  sdl_user_new_message_event;
        Uint32                  sdl_user_reconfigure_event;

        int                     display_idx;
        int                     x;
        int                     y;
        int                     renderer_idx;
        SDL_Window             *window;
        SDL_Renderer           *renderer;

        bool                    fs;
        enum deint              deinterlace;
        bool                    keep_aspect;
        bool                    vsync;
        bool                    fixed_size;
        unsigned                fixed_w, fixed_h;
        uint32_t                window_flags; ///< user requested flags

        pthread_mutex_t         lock;
        pthread_cond_t          frame_consumed_cv;

        pthread_cond_t          reconfigured_cv;
        int                     reconfiguration_status;

        struct video_desc       current_display_desc;
        struct video_frame     *last_frame;

        struct simple_linked_list *free_frame_queue;

};

static const char *deint_to_string(enum deint val) {
        switch (val) {
                case DEINT_OFF: return "OFF";
                case DEINT_ON: return "ON";
                case DEINT_FORCE: return "FORCE";
        }
        return NULL;
}

static const struct {
        char key;
        const char *description;
} keybindings[] = {
        {'d', "toggle deinterlace"},
        {'f', "toggle fullscreen"},
        {'q', "quit"},
};

#define SDL_CHECK(cmd) do { int ret = cmd; if (ret < 0) { log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error (%s): %s\n", #cmd, SDL_GetError());} } while(0)

static void display_frame(struct state_sdl2 *s, struct video_frame *frame)
{
        if (!frame) {
                return;
        }

        SDL_Texture *texture = (SDL_Texture *) frame->callbacks.dispose_udata;
        if (s->deinterlace == DEINT_FORCE || (s->deinterlace == DEINT_ON && frame->interlacing == INTERLACED_MERGED)) {
                size_t pitch = vc_get_linesize(frame->tiles[0].width, frame->color_spec);
                if (!vc_deinterlace_ex(frame->color_spec, (unsigned char *) frame->tiles[0].data, pitch, (unsigned char *) frame->tiles[0].data, pitch, frame->tiles[0].height)) {
                         log_msg_once(LOG_LEVEL_ERROR, SDL2_DEINTERLACE_IMPOSSIBLE_MSG_ID, MOD_NAME "Cannot deinterlace, unsupported pixel format '%s'!\n", get_codec_name(frame->color_spec));
                }
        }

        SDL_RenderClear(s->renderer);
        SDL_UnlockTexture(texture);
        SDL_CHECK(SDL_RenderCopy(s->renderer, texture, NULL, NULL));
        SDL_RenderPresent(s->renderer);

        int pitch = 0;
        SDL_CHECK(SDL_LockTexture(texture, NULL, (void **) &frame->tiles[0].data, &pitch));
        assert(pitch == s->texture_pitch);

        if (frame == s->last_frame) {
                return; // we are only redrawing on window resize
        }

        pthread_mutex_lock(&s->lock);
        simple_linked_list_append(s->free_frame_queue, frame);
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->frame_consumed_cv);
        s->last_frame = frame;
}

static int64_t translate_sdl_key_to_ug(SDL_Keysym sym) {
        sym.mod &= ~(KMOD_NUM | KMOD_CAPS); // remove num+caps lock modifiers

        // ctrl alone -> do not interpret
        if (sym.sym == SDLK_LCTRL || sym.sym == SDLK_RCTRL) {
                return 0;
        }

        bool ctrl = false;
        bool shift = false;
        if (sym.mod & KMOD_CTRL) {
                ctrl = true;
        }
        sym.mod &= ~KMOD_CTRL;

        if (sym.mod & KMOD_SHIFT) {
                shift = true;
        }
        sym.mod &= ~KMOD_SHIFT;

        if (sym.mod != 0) {
                return -1;
        }

        if ((sym.sym & SDLK_SCANCODE_MASK) == 0) {
                if (shift) {
                        sym.sym = toupper(sym.sym);
                }
                return ctrl ? K_CTRL(sym.sym) : sym.sym;
        }
        switch (sym.sym) {
        case SDLK_RIGHT: return K_RIGHT;
        case SDLK_LEFT:  return K_LEFT;
        case SDLK_DOWN:  return K_DOWN;
        case SDLK_UP:    return K_UP;
        case SDLK_PAGEDOWN:    return K_PGDOWN;
        case SDLK_PAGEUP:    return K_PGUP;
        }
        return -1;
}

static bool display_sdl2_process_key(struct state_sdl2 *s, int64_t key)
{
        switch (key) {
        case 'd':
                s->deinterlace = s->deinterlace == DEINT_OFF ? DEINT_ON : DEINT_OFF;
                log_msg(LOG_LEVEL_INFO, "Deinterlacing: %s\n", deint_to_string(s->deinterlace));
                return true;
        case 'f':
                s->fs = !s->fs;
                SDL_SetWindowFullscreen(s->window, s->fs ? SDL_WINDOW_FULLSCREEN_DESKTOP : 0);
                return true;
        case 'q':
                exit_uv(0);
                return true;
        default:
                return false;
        }
}

static void display_sdl2_run(void *arg)
{
        struct state_sdl2 *s = (struct state_sdl2 *) arg;

        loadSplashscreen(s);

        while (1) {
                SDL_Event sdl_event;
                if (!SDL_WaitEvent(&sdl_event)) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "SDL_WaitEvent error: %s\n", SDL_GetError());
                        continue;
                }
                if (sdl_event.type == s->sdl_user_reconfigure_event) {
                        pthread_mutex_lock(&s->lock);
                        struct video_desc desc = *(struct video_desc *) sdl_event.user.data1;
                        s->reconfiguration_status = display_sdl2_reconfigure_real(s, desc);
                        pthread_mutex_unlock(&s->lock);
                        pthread_cond_signal(&s->reconfigured_cv);

                } else if (sdl_event.type == s->sdl_user_new_frame_event) {
                        if (sdl_event.user.data1 == NULL) { // poison pill received
                                break;
                        }
                        display_frame(s, (struct video_frame *) sdl_event.user.data1);
                } else if (sdl_event.type == s->sdl_user_new_message_event) {
                        struct msg_universal *msg;
                        while ((msg = (struct msg_universal *) check_message(&s->mod))) {
                                log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Received message: %s\n", msg->text);
                                struct response *r;
                                int key;
                                if (strstr(msg->text, "win-title ") == msg->text) {
                                        SDL_SetWindowTitle(s->window, msg->text + strlen("win-title "));
                                        r = new_response(RESPONSE_OK, NULL);
                                } else if (sscanf(msg->text, "%d", &key) == 1) {
                                        if (!display_sdl2_process_key(s, key)) {
                                                r = new_response(RESPONSE_BAD_REQUEST, "Unsupported key for SDL");
                                        } else {
                                                r = new_response(RESPONSE_OK, NULL);
                                        }
                                } else {
                                        r = new_response(RESPONSE_BAD_REQUEST, "Wrong command");
                                }

                                free_message((struct message*) msg, r);
                        }
                } else if (sdl_event.type == SDL_KEYDOWN) {
                        log_msg(LOG_LEVEL_VERBOSE, MOD_NAME "Pressed key %s (scancode: %d, sym: %d, mod: %d)!\n", SDL_GetKeyName(sdl_event.key.keysym.sym), sdl_event.key.keysym.scancode, sdl_event.key.keysym.sym, sdl_event.key.keysym.mod);
                        int64_t sym = translate_sdl_key_to_ug(sdl_event.key.keysym);
                        if (sym > 0) {
                                if (!display_sdl2_process_key(s, sym)) { // unknown key -> pass to control
                                        keycontrol_send_key(get_root_module(&s->mod), sym);
                                }
                        } else if (sym == -1) {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Cannot translate key %s (scancode: %d, sym: %d, mod: %d)!\n", SDL_GetKeyName(sdl_event.key.keysym.sym), sdl_event.key.keysym.scancode, sdl_event.key.keysym.sym, sdl_event.key.keysym.mod);
                        }
                } else if (sdl_event.type == SDL_WINDOWEVENT) {
                        // https://forums.libsdl.org/viewtopic.php?p=38342
                        if (s->keep_aspect && sdl_event.window.event == SDL_WINDOWEVENT_RESIZED) {
                                double area = sdl_event.window.data1 * sdl_event.window.data2;
                                int width = sqrt(area / ((double) s->current_display_desc.height / s->current_display_desc.width));
                                int height = sqrt(area / ((double) s->current_display_desc.width / s->current_display_desc.height));
                                SDL_SetWindowSize(s->window, width, height);
                                debug_msg("[SDL] resizing to %d x %d\n", width, height);
                        }
                        if (sdl_event.window.event == SDL_WINDOWEVENT_EXPOSED
                                        || sdl_event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                                // clear both buffers
                                SDL_RenderClear(s->renderer);
                                display_frame(s, s->last_frame);
                                SDL_RenderClear(s->renderer);
                                display_frame(s, s->last_frame);
                        }
                } else if (sdl_event.type == SDL_QUIT) {
                        exit_uv(0);
                }
        }
}

static void sdl2_print_displays() {
        for (int i = 0; i < SDL_GetNumVideoDisplays(); ++i) {
                if (i > 0) {
                        printf(", ");
                }
                const char *dname = SDL_GetDisplayName(i);
                if (dname == NULL) {
                        dname = SDL_GetError();
                }
                color_printf(TBOLD("%d") " - %s", i, dname);
        }
        printf("\n");
}

static void show_help(void)
{
        SDL_CHECK(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS));
        printf("SDL options:\n");
        color_printf(TBOLD(
            TRED("\t-d sdl") "[[:fs|:d|:display=<didx>|:driver=<drv>|:novsync|:"
                             "renderer=<ridx>|:nodecorate|:size[=WxH]|:window_"
                             "flags=<f>|:keep-aspect]*|:help]") "\n");
        printf("where:\n");
        color_printf(TBOLD("\td[force]") " - deinterlace (force even for progresive video)\n");
        color_printf(TBOLD("\t      fs") " - fullscreen\n");
        color_printf(TBOLD("\t  <didx>") " - display index, available indices: ");
        sdl2_print_displays();
        color_printf(TBOLD("\t   <drv>") " - one of following: ");
        for (int i = 0; i < SDL_GetNumVideoDrivers(); ++i) {
                color_printf("%s" TBOLD("%s"), (i == 0 ? "" : ", "), SDL_GetVideoDriver(i));
        }
        color_printf("\n");
        color_printf(TBOLD("     keep-aspect") " - keep window aspect ratio respecive to the video\n");
        color_printf(TBOLD("         novsync") " - disable sync on VBlank\n");
        color_printf(TBOLD("      nodecorate") " - disable window border\n");
        color_printf(
            TBOLD("            size") " - window size in pixels with optional "
                                      "position\n"
                                      "                   "
                                      "(syntax: " TBOLD(
                                          "[<W>x<H>][{+-}<X>[{+-}<Y>]]") ")\n");
        color_printf(TBOLD("\t  <ridx>") " - renderer index: ");
        for (int i = 0; i < SDL_GetNumRenderDrivers(); ++i) {
                SDL_RendererInfo renderer_info;
                if (SDL_GetRenderDriverInfo(i, &renderer_info) == 0) {
                        color_printf("%s" TBOLD("%d") " - " TBOLD("%s"), (i == 0 ? "" : ", "), i, renderer_info.name);
                }
        }
        printf("\n");
        printf("\nKeyboard shortcuts:\n");
        for (unsigned int i = 0; i < sizeof keybindings / sizeof keybindings[0]; ++i) {
                color_printf("\t" TBOLD("'%c'") "\t - %s\n", keybindings[i].key, keybindings[i].description);
        }
        SDL_Quit();
}

static bool display_sdl2_reconfigure(void *state, struct video_desc desc)
{
        struct state_sdl2 *s = (struct state_sdl2 *) state;

        if (desc.interlacing == INTERLACED_MERGED && s->deinterlace == DEINT_OFF) {
                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Receiving interlaced video but deinterlacing is off - suggesting toggling it on (press 'd' or pass cmdline option)\n");
        }

        pthread_mutex_lock(&s->lock);

        SDL_Event event;
        event.type = s->sdl_user_reconfigure_event;
        event.user.data1 = &desc;
        SDL_CHECK(SDL_PushEvent(&event));

        s->reconfiguration_status = -1;
        while (s->reconfiguration_status == -1) {
                pthread_cond_wait(&s->reconfigured_cv, &s->lock);
        }
        pthread_mutex_unlock(&s->lock);

        return s->reconfiguration_status;
}

static const struct {
        codec_t first;
        uint32_t second;
} pf_mapping[] = {
        { I420, SDL_PIXELFORMAT_IYUV },
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

static uint32_t get_ug_to_sdl_format(codec_t ug_codec) {
        if (ug_codec == R10k) {
                return SDL_PIXELFORMAT_ARGB2101010;
        }

        for (unsigned int i = 0; i < sizeof pf_mapping / sizeof pf_mapping[0]; ++i) {
                if (pf_mapping[i].first == ug_codec) {
                        return pf_mapping[i].second;
                }
        }
        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong codec: %s\n", get_codec_name(ug_codec));
        return SDL_PIXELFORMAT_UNKNOWN;
}

ADD_TO_PARAM("sdl2-r10k",
         "* sdl2-r10k\n"
         "  Enable 10-bit RGB support for SDL2 (EXPERIMENTAL)\n");

static int get_supported_pfs(codec_t *codecs) {
        int count = 0;

        for (unsigned int i = 0; i < sizeof pf_mapping / sizeof pf_mapping[0]; ++i) {
                codecs[count++] = pf_mapping[i].first;
        }
        if (get_commandline_param("sdl2-r10k") != NULL) {
                codecs[count++] = R10k;
        }
        return count;
}

static void cleanup_frames(struct state_sdl2 *s) {
        s->last_frame = NULL;
        struct video_frame *buffer = NULL;
        while ((buffer = simple_linked_list_pop(s->free_frame_queue)) != NULL) {
                vf_free(buffer);
        }
}

static void vf_sdl_texture_data_deleter(struct video_frame *buf) {
        SDL_Texture *texture = (SDL_Texture *) buf->callbacks.dispose_udata;
        SDL_DestroyTexture(texture);
}

static bool recreate_textures(struct state_sdl2 *s, struct video_desc desc) {
        cleanup_frames(s);

        for (int i = 0; i < BUFFER_COUNT; ++i) {
                SDL_Texture *texture = SDL_CreateTexture(s->renderer, get_ug_to_sdl_format(desc.color_spec), SDL_TEXTUREACCESS_STREAMING, desc.width, desc.height);
                if (!texture) {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable to create texture: %s\n", SDL_GetError());
                        return false;
                }
                struct video_frame *f = vf_alloc_desc(desc);
                f->callbacks.dispose_udata = (void *) texture;
                SDL_CHECK(SDL_LockTexture(texture, NULL, (void **) &f->tiles[0].data, &s->texture_pitch));
                f->tiles[0].data_len = desc.height * s->texture_pitch;
                f->callbacks.data_deleter = vf_sdl_texture_data_deleter;
                simple_linked_list_append(s->free_frame_queue, f);
        }

        return true;
}

static int display_sdl2_reconfigure_real(void *state, struct video_desc desc)
{
        struct state_sdl2 *s = (struct state_sdl2 *)state;

        log_msg(LOG_LEVEL_NOTICE, "[SDL] Reconfigure to size %dx%d\n", desc.width,
                        desc.height);

        if (s->fixed_size && s->window) {
                SDL_RenderSetLogicalSize(s->renderer, desc.width, desc.height);
                return recreate_textures(s, desc);
        }

        if (s->window) {
                SDL_DestroyWindow(s->window);
        }
        int flags = s->window_flags | SDL_WINDOW_RESIZABLE | SDL_WINDOW_ALLOW_HIGHDPI;
        if (s->fs) {
                flags |= SDL_WINDOW_FULLSCREEN_DESKTOP;
        }
        const char *window_title = "UltraGrid - SDL2 Display";
        if (get_commandline_param("window-title")) {
                window_title = get_commandline_param("window-title");
        }
        int width = s->fixed_w ? s->fixed_w : desc.width;
        int height = s->fixed_h ? s->fixed_h : desc.height;
        int x = s->x == SDL_WINDOWPOS_UNDEFINED ? (int) SDL_WINDOWPOS_CENTERED_DISPLAY(s->display_idx) : s->x;
        int y = s->y == SDL_WINDOWPOS_UNDEFINED ? (int) SDL_WINDOWPOS_CENTERED_DISPLAY(s->display_idx) : s->y;
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

        if (!recreate_textures(s, desc)) {
                return FALSE;
        }

        s->current_display_desc = desc;

        return TRUE;
}

static void loadSplashscreen(struct state_sdl2 *s) {
        struct video_frame *frame = get_splashscreen();
        display_sdl2_reconfigure_real(s, video_desc_from_frame(frame));
        struct video_frame *splash = display_sdl2_getf(s);
        memcpy(splash->tiles[0].data, frame->tiles[0].data, frame->tiles[0].data_len);
        vf_free(frame);
        display_frame(s, splash); // don't be tempted to use _putf() - it will use event queue and there may arise a race-condition with recv thread
}

static bool set_size(struct state_sdl2 *s, const char *tok)
{
        if (strstr(tok, "fixed_size=") == tok) {
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "fixed_size with dimensions is "
                                 " deprecated, use size"
                                 " instead\n");
        }
        tok = strchr(tok, '=') + 1;
        if (strpbrk(tok, "x+-") == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong size spec: %s\n", tok);
                return false;
        }
        if (strchr(tok, 'x') != NULL) {
                s->fixed_size = true;
                s->fixed_w = atoi(tok);
                s->fixed_h = atoi(strchr(tok, 'x') + 1);
        }
        tok = strpbrk(tok, "+-");
        if (tok != NULL) {
                s->x = atoi(tok);
                tok = strpbrk(tok + 1, "+-");
        }
        if (tok != NULL) {
                s->y = atoi(tok);
        }
        return true;
}

static void *display_sdl2_init(struct module *parent, const char *fmt, unsigned int flags)
{
        if (flags & DISPLAY_FLAG_AUDIO_ANY) {
                log_msg(LOG_LEVEL_ERROR, "UltraGrid SDL2 module currently doesn't support audio!\n");
                return NULL;
        }
        const char *driver = NULL;
        struct state_sdl2 *s = calloc(1, sizeof *s);

        s->x = s->y = SDL_WINDOWPOS_UNDEFINED;
        s->renderer_idx = -1;
        s->vsync = true;

        if (fmt == NULL) {
                fmt = "";
        }
        char *tmp = (char *) alloca(strlen(fmt) + 1);
        strcpy(tmp, fmt);
        char *tok, *save_ptr;
        while((tok = strtok_r(tmp, ":", &save_ptr)))
        {
                if (strcmp(tok, "d") == 0 || strcmp(tok, "dforce") == 0) {
                        s->deinterlace = strcmp(tok, "d") == 0 ? DEINT_ON : DEINT_OFF;
                } else if (strncmp(tok, "display=", strlen("display=")) == 0) {
                        s->display_idx = atoi(tok + strlen("display="));
                } else if (strncmp(tok, "driver=", strlen("driver=")) == 0) {
                        driver = tok + strlen("driver=");
                } else if (strcmp(tok, "fs") == 0) {
                        s->fs = true;
                } else if (strcmp(tok, "help") == 0) {
                        show_help();
                        free(s);
                        return INIT_NOERR;
                } else if (strcmp(tok, "novsync") == 0) {
                        s->vsync = false;
                } else if (strcmp(tok, "nodecorate") == 0) {
                        s->window_flags |= SDL_WINDOW_BORDERLESS;
                } else if (strcmp(tok, "keep-aspect") == 0) {
                        s->keep_aspect = true;
                } else if (strstr(tok, "fixed_size=") == tok ||
                           strstr(tok, "size=") == tok) {
                        if (!set_size(s, tok)) {
                                free(s);
                                return NULL;
                        }
                } else if (strcmp(tok, "fixed_size") == 0) {
                        log_msg(LOG_LEVEL_WARNING,
                                MOD_NAME "fixed_size deprecated, use size with "
                                         "dimensions\n");
                        s->fixed_size = true;
                } else if (strstr(tok, "window_flags=") == tok) {
                        int f;
                        if (sscanf(tok + strlen("window_flags="), "%i", &f) != 1) {
                                log_msg(LOG_LEVEL_ERROR, "Wrong window_flags: %s\n", tok);
                                free(s);
                                return NULL;
                        }
                        s->window_flags |= f;
		} else if (strstr(tok, "pos=") == tok) {
                        tok += strlen("pos=");
                        if (strchr(tok, ',') == NULL) {
                                log_msg(LOG_LEVEL_ERROR, "[SDL] position: %s\n", tok);
                                free(s);
                                return NULL;
                        }
                        s->x = atoi(tok);
                        s->y = atoi(strchr(tok, ',') + 1);
                        log_msg(LOG_LEVEL_WARNING,
                                MOD_NAME "pos is deprecated, use "
                                         "\"size=%+d%+d\" instead.\n",
                                s->x, s->y);
                } else if (strncmp(tok, "renderer=", strlen("renderer=")) == 0) {
                        s->renderer_idx = atoi(tok + strlen("renderer="));
                } else {
                        log_msg(LOG_LEVEL_ERROR, "[SDL] Wrong option: %s\n", tok);
                        free(s);
                        return NULL;
                }
                tmp = NULL;
        }

        int ret = SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize SDL2: %s\n", SDL_GetError());
                free(s);
                return NULL;
        }
        ret = SDL_VideoInit(driver);
        if (ret < 0) {
                log_msg(LOG_LEVEL_ERROR, "Unable to initialize SDL2 video: %s\n", SDL_GetError());
                free(s);
                return NULL;
        }
        log_msg(LOG_LEVEL_NOTICE, "[SDL] Using driver: %s\n", SDL_GetCurrentVideoDriver());

        SDL_ShowCursor(SDL_DISABLE);
        SDL_DisableScreenSaver();

        module_init_default(&s->mod);
        s->mod.priv_magic = MAGIC_SDL2;
        s->mod.new_message = display_sdl2_new_message;
        s->mod.cls = MODULE_CLASS_DATA;
        module_register(&s->mod, parent);

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->frame_consumed_cv, NULL);
        pthread_cond_init(&s->reconfigured_cv , NULL);

        s->sdl_user_new_frame_event = SDL_RegisterEvents(3);
        assert(s->sdl_user_new_frame_event != (Uint32) -1);
        s->sdl_user_new_message_event = s->sdl_user_new_frame_event + 1;
        s->sdl_user_reconfigure_event = s->sdl_user_new_frame_event + 2;

        s->free_frame_queue = simple_linked_list_init();

        for (unsigned int i = 0; i < sizeof keybindings / sizeof keybindings[0]; ++i) {
                if (keybindings[i].key == 'q') { // don't report 'q' to avoid accidental close - user can use Ctrl-c there
                        continue;
                }
                char key_str[128];
                snprintf(key_str, sizeof key_str, "%d", keybindings[i].key);
                keycontrol_register_key(&s->mod, keybindings[i].key, key_str, keybindings[i].description);
        }

        log_msg(LOG_LEVEL_NOTICE, "SDL2 initialized successfully.\n");

        return (void *) s;
}

static void display_sdl2_done(void *state)
{
        struct state_sdl2 *s = (struct state_sdl2 *)state;

        assert(s->mod.priv_magic == MAGIC_SDL2);

        cleanup_frames(s);

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

        simple_linked_list_destroy(s->free_frame_queue);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->frame_consumed_cv);
        pthread_cond_destroy(&s->reconfigured_cv);

        module_done(&s->mod);

        free(s);
}

static struct video_frame *display_sdl2_getf(void *state)
{
        struct state_sdl2 *s = (struct state_sdl2 *)state;
        assert(s->mod.priv_magic == MAGIC_SDL2);

        pthread_mutex_lock(&s->lock);
        while (simple_linked_list_size(s->free_frame_queue) == 0) {
                pthread_cond_wait(&s->frame_consumed_cv, &s->lock);
        }
        struct video_frame *buffer = simple_linked_list_pop(s->free_frame_queue);
        pthread_mutex_unlock(&s->lock);

        return buffer;
}

static bool display_sdl2_putf(void *state, struct video_frame *frame, long long timeout_ns)
{
        struct state_sdl2 *s = (struct state_sdl2 *)state;

        assert(s->mod.priv_magic == MAGIC_SDL2);

        pthread_mutex_lock(&s->lock);
        if (timeout_ns == PUTF_DISCARD) {
                assert(frame != NULL);
                simple_linked_list_append(s->free_frame_queue, frame);
                pthread_mutex_unlock(&s->lock);
                return true;
        }

        if (frame != NULL && timeout_ns > 0) {
                int rc = 0;
                while (rc == 0 && simple_linked_list_size(s->free_frame_queue) == 0) {
                        if (timeout_ns == PUTF_BLOCKING) {
                                rc = pthread_cond_wait(&s->frame_consumed_cv, &s->lock);
                        } else {
                                struct timespec ts;
                                timespec_get(&ts, TIME_UTC);
                                ts_add_nsec(&ts, timeout_ns);
                                rc = pthread_cond_timedwait(&s->frame_consumed_cv, &s->lock, &ts);
                        }
                }
        }
        if (frame != NULL && simple_linked_list_size(s->free_frame_queue) == 0) {
                simple_linked_list_append(s->free_frame_queue, frame);
                log_msg(LOG_LEVEL_INFO, MOD_NAME "1 frame(s) dropped!\n");
                pthread_mutex_unlock(&s->lock);
                return false;
        }
        pthread_mutex_unlock(&s->lock);
        SDL_Event event;
        event.type = s->sdl_user_new_frame_event;
        event.user.data1 = frame;
        SDL_CHECK(SDL_PushEvent(&event));

        return true;
}

static bool display_sdl2_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_sdl2 *s = (struct state_sdl2 *) state;
        codec_t codecs[VIDEO_CODEC_COUNT];
        size_t codecs_len = get_supported_pfs(codecs) * sizeof(codec_t);

        switch (property) {
                case DISPLAY_PROPERTY_CODECS:
                        if (codecs_len <= *len) {
                                memcpy(val, codecs, codecs_len);
                                *len = codecs_len;
                        } else {
                                return false;
                        }
                        break;
                case DISPLAY_PROPERTY_BUF_PITCH:
                        *(int *) val = codec_is_planar(s->current_display_desc.color_spec) ? PITCH_DEFAULT : s->texture_pitch;
                        *len = sizeof(int);
                        break;
                default:
                        return false;
        }
        return true;
}

static void display_sdl2_new_message(struct module *mod)
{
        struct state_sdl2 *s = (struct state_sdl2 *) mod;

        SDL_Event event;
        event.type = s->sdl_user_new_message_event;
        SDL_CHECK(SDL_PushEvent(&event));
}

static void display_sdl2_probe(struct device_info **available_cards, int *count, void (**deleter)(void *)) {
        UNUSED(deleter);
        *count = 1;
        *available_cards = (struct device_info *) calloc(1, sizeof(struct device_info));
        strcpy((*available_cards)[0].dev, "");
        strcpy((*available_cards)[0].name, "SDL2 SW display");
        dev_add_option(&(*available_cards)[0], "Deinterlace", "Deinterlace", "deinterlace", ":d", true);
        dev_add_option(&(*available_cards)[0], "Fullscreen", "Launch as fullscreen", "fullscreen", ":fs", true);
        dev_add_option(&(*available_cards)[0], "No decorate", "Disable window decorations", "nodecorate", ":nodecorate", true);
        dev_add_option(&(*available_cards)[0], "Disable vsync", "Disable vsync", "novsync", ":novsync", true);

        (*available_cards)[0].repeatable = true;
}

static const struct video_display_info display_sdl2_info = {
        display_sdl2_probe,
        display_sdl2_init,
        display_sdl2_run,
        display_sdl2_done,
        display_sdl2_getf,
        display_sdl2_putf,
        display_sdl2_reconfigure,
        display_sdl2_get_property,
        NULL,
        NULL,
        MOD_NAME,
};

REGISTER_MODULE(sdl, &display_sdl2_info, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);

