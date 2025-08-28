/**
 * @file   video_display/sdl3.c
 * @author Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 * @author Milos Liska      <xliska@fi.muni.cz>
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2025 CESNET
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
 * @todo errata (SDL3 vs SDL2)
 * 1. [macOS] Vulkan renderer doesn't work (no matter if linked with MoltenVK or
 * loader)
 * 2. [all platforms] with `renderer=vulkan` - none of YCbCr textures work
 * (segfaults - wrong pitch/texture?)
 * 3. p010 works just on macOS/Metal, crashes on Vulkan (see previous point)
 * 4. p010 corrupted on d3d[12] - pixfmts skipped in query*() as a workaround
 */

#include <SDL3/SDL.h>
#include <assert.h>   // for assert
#include <ctype.h>    // for toupper
#include <inttypes.h> // for PRIu8
#include <math.h>     // for sqrt
#include <pthread.h>  // for pthread_mutex_unlock, pthread_mutex_lock
#include <stdbool.h>  // for true, bool, false
#include <stdint.h>   // for int64_t, uint32_t
#include <stdio.h>    // for printf, sscanf, snprintf
#include <stdlib.h>   // for atoi, free, calloc
#include <string.h>   // for NULL, strlen, strcmp, strstr, strchr
#include <time.h>     // for timespec_get, TIME_UTC, timespec

#include "compat/endian.h"    // for be32toh, htole32
#include "debug.h"            // for log_msg, LOG_LEVEL_ERROR, LOG_LEVEL_W...
#include "host.h"             // for get_commandline_param, exit_uv, ADD_T...
#include "keyboard_control.h" // for keycontrol_register_key, keycontrol_s...
#include "lib_common.h"       // for REGISTER_MODULE, library_class
#include "messaging.h"        // for new_response, msg_universal, RESPONSE...
#include "module.h"           // for module, get_root_module, module_done
#include "pixfmt_conv.h"      // for v210_to_p010le
#include "tv.h"               // for ts_add_nsec
#include "types.h"            // for video_desc, tile, video_frame, device...
#include "utils/color_out.h"  // for color_printf, TBOLD, TRED
#include "utils/list.h"       // for simple_linked_list_append, simple_lin...
#include "utils/macros.h"     // for STR_LEN
#include "video.h"            // for get_video_desc_from_string
#include "video_codec.h"      // for get_codec_name, codec_is_planar, vc_d...
#include "video_display.h"    // for display_property, get_splashscreen
#include "video_frame.h"      // for vf_free, vf_alloc_desc, video_desc_fr...

#define MAGIC_SDL3   0x60540F2D
#define BUFFER_COUNT 2
#define MOD_NAME     "[SDL3] "

struct state_sdl3;

static void show_help(const char *driver, bool full);
static void display_frame(struct state_sdl3 *s, struct video_frame *frame);
static struct video_frame *display_sdl3_getf(void *state);
static void                display_sdl3_new_message(struct module *mod);
static bool display_sdl3_reconfigure_real(void *state, struct video_desc desc);
static void loadSplashscreen(struct state_sdl3 *s);

enum deint { DEINT_OFF, DEINT_ON, DEINT_FORCE };

struct video_frame_sdl3_data {
        SDL_Texture *texture; ///< associated texture
        char        *preconv_data;
};

static void convert_UYVY_IYUV(const struct video_frame *uv_frame,
                              unsigned char *tex_data, size_t y_pitch);
static void convert_UYVY_NV12(const struct video_frame *uv_frame,
                              unsigned char *tex_data, size_t y_pitch);
static void convert_R10k_ARGB2101010(const struct video_frame *uv_frame,
                                     unsigned char *tex_data, size_t y_pitch);
static void convert_R10k_ABGR2101010(const struct video_frame *uv_frame,
                                     unsigned char *tex_data, size_t y_pitch);
static void convert_RGBA_BGRA(const struct video_frame *uv_frame,
                              unsigned char *tex_data, size_t y_pitch);
static void convert_Y216_P010(const struct video_frame *uv_frame,
                              unsigned char *tex_data, size_t y_pitch);
static void convert_v210_P010(const struct video_frame *uv_frame,
                              unsigned char *tex_data, size_t y_pitch);
struct fmt_data {
        codec_t              ug_codec;
        enum SDL_PixelFormat sdl_tex_fmt;
        void (*convert)(const struct video_frame *uv_frame,
                        unsigned char *tex_data, size_t tex_pitch);
};
// order matters relative to fixed ug codec - first usable SDL fmt is used
static const struct fmt_data pf_mapping_template[] = {
        { I420, SDL_PIXELFORMAT_IYUV,        NULL                     }, // gles2,ogl,vk,d3d,d3d1[12],metal
        { RGBA, SDL_PIXELFORMAT_RGBA32,      NULL                     }, // gles2,ogl,gpu,metal
        { RGBA, SDL_PIXELFORMAT_BGRA32,      convert_RGBA_BGRA        }, // gles2,ogl,gpu,sw,vk,d3d,d3d1[12],metal
        { RGBA, SDL_PIXELFORMAT_RGBX32,      NULL                     }, // gles2,ogl,gpu
        { RGBA, SDL_PIXELFORMAT_BGRX32,      convert_RGBA_BGRA        }, // gles2,ogl,gpu,sw,vk,d3d12
        { UYVY, SDL_PIXELFORMAT_UYVY,        NULL                     }, // mac ogl
        { UYVY, SDL_PIXELFORMAT_IYUV,        convert_UYVY_IYUV        }, // fallback
        { UYVY, SDL_PIXELFORMAT_NV12,        convert_UYVY_NV12        }, // ditto
        { YUYV, SDL_PIXELFORMAT_YUY2,        NULL                     },
        { RGB,  SDL_PIXELFORMAT_RGB24,       NULL                     },
        { BGR,  SDL_PIXELFORMAT_BGR24,       NULL                     },
        { Y216, SDL_PIXELFORMAT_P010,        convert_Y216_P010        }, // vk.metal,d3d1[12]
        { v210, SDL_PIXELFORMAT_P010,        convert_v210_P010        }, // ditto
        { R10k, SDL_PIXELFORMAT_ARGB2101010, convert_R10k_ARGB2101010 },
        { R10k, SDL_PIXELFORMAT_ABGR2101010, convert_R10k_ABGR2101010 }, // vk,metal,d3d1[12]
        { R10k, SDL_PIXELFORMAT_XBGR2101010, convert_R10k_ABGR2101010 },
};
struct state_sdl3 {
        uint32_t magic;
        struct module   mod;
        struct fmt_data supp_fmts[(sizeof pf_mapping_template /
                                   sizeof pf_mapping_template[0]) +
                                  1]; // nul terminated
        const struct fmt_data *cs_data;
        int                   texture_pitch;

        Uint32 sdl_user_new_frame_event;
        Uint32 sdl_user_new_message_event;
        Uint32 sdl_user_reconfigure_event;

        int           display_idx;
        int           x;
        int           y;
        char          req_renderers_name[STR_LEN];
        SDL_BlendMode req_blend_mode; // SDL_BLENDMODE_NONE == 0
        SDL_Window   *window;
        SDL_Renderer *renderer;

        bool       fs;
        enum deint deinterlace;
        bool       keep_aspect;
        bool       vsync;
        bool       fixed_size;
        unsigned   fixed_w, fixed_h;
        uint32_t   window_flags; ///< user requested flags

        pthread_mutex_t lock;
        pthread_cond_t  frame_consumed_cv;

        pthread_cond_t reconfigured_cv;
        int            reconfiguration_status;

        struct video_desc   current_display_desc;
        struct video_frame *last_frame;

        struct simple_linked_list *free_frame_queue;
};

static const char *
deint_to_string(enum deint val)
{
        switch (val) {
        case DEINT_OFF:
                return "OFF";
        case DEINT_ON:
                return "ON";
        case DEINT_FORCE:
                return "FORCE";
        }
        return NULL;
}

static const struct {
        char        key;
        const char *description;
} keybindings[] = {
        { 'd', "toggle deinterlace" },
        { 'f', "toggle fullscreen"  },
        { 'q', "quit"               },
};

#define SDL_CHECK(cmd, ...) \
        do { \
                if (!(cmd)) { \
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Error (%s): %s\n", \
                                #cmd, SDL_GetError()); \
                        __VA_ARGS__; \
                } \
        } while (0)

static void
display_frame(struct state_sdl3 *s, struct video_frame *frame)
{
        if (!frame) {
                return;
        }

        struct video_frame_sdl3_data *frame_data =
            frame->callbacks.dispose_udata;
        if (s->deinterlace == DEINT_FORCE ||
            (s->deinterlace == DEINT_ON &&
             frame->interlacing == INTERLACED_MERGED)) {
                size_t pitch =
                    vc_get_linesize(frame->tiles[0].width, frame->color_spec);
                if (!vc_deinterlace_ex(frame->color_spec,
                                       (unsigned char *) frame->tiles[0].data,
                                       pitch,
                                       (unsigned char *) frame->tiles[0].data,
                                       pitch, frame->tiles[0].height)) {
                        MSG_ONCE(ERROR,
                                 "Cannot deinterlace, unsupported "
                                 "pixel format '%s'!\n",
                                 get_codec_name(frame->color_spec));
                }
        }

        int pitch = 0;
        if (s->cs_data->convert != NULL) {
                unsigned char *tex_data = NULL;
                SDL_CHECK(SDL_LockTexture(frame_data->texture, NULL,
                                          (void **) &tex_data, &pitch));
                s->cs_data->convert(frame, tex_data, pitch);
        }

        SDL_RenderClear(s->renderer);
        SDL_UnlockTexture(frame_data->texture);
        SDL_CHECK(
            SDL_RenderTexture(s->renderer, frame_data->texture, NULL, NULL));
        SDL_RenderPresent(s->renderer);

        if (s->cs_data->convert == NULL) {
                SDL_CHECK(SDL_LockTexture(frame_data->texture, NULL,
                                          (void **) &frame->tiles[0].data,
                                          &pitch));
                assert(pitch == s->texture_pitch);
        }

        if (frame == s->last_frame) {
                return; // we are only redrawing on window resize
        }

        pthread_mutex_lock(&s->lock);
        simple_linked_list_append(s->free_frame_queue, frame);
        pthread_mutex_unlock(&s->lock);
        pthread_cond_signal(&s->frame_consumed_cv);
        s->last_frame = frame;
}

static int64_t
translate_sdl_key_to_ug(SDL_KeyboardEvent keyev)
{
        keyev.mod &=
            ~(SDL_KMOD_NUM | SDL_KMOD_CAPS); // remove num+caps lock modifiers

        // ctrl alone -> do not interpret
        if (keyev.key == SDLK_LCTRL || keyev.key == SDLK_RCTRL) {
                return 0;
        }

        bool ctrl  = false;
        bool shift = false;
        if (keyev.mod & SDL_KMOD_CTRL) {
                ctrl = true;
        }
        keyev.mod &= ~SDL_KMOD_CTRL;

        if (keyev.mod & SDL_KMOD_SHIFT) {
                shift = true;
        }
        keyev.mod &= ~SDL_KMOD_SHIFT;

        if (keyev.mod != 0) {
                return -1;
        }

        if ((keyev.key & SDLK_SCANCODE_MASK) == 0) {
                if (shift) {
                        keyev.key = toupper(keyev.key);
                }
                return ctrl ? K_CTRL(keyev.key) : keyev.key;
        }
        switch (keyev.key) {
        case SDLK_RIGHT:
                return K_RIGHT;
        case SDLK_LEFT:
                return K_LEFT;
        case SDLK_DOWN:
                return K_DOWN;
        case SDLK_UP:
                return K_UP;
        case SDLK_PAGEDOWN:
                return K_PGDOWN;
        case SDLK_PAGEUP:
                return K_PGUP;
        }
        return -1;
}

static bool
display_sdl3_process_key(struct state_sdl3 *s, int64_t key)
{
        switch (key) {
        case 'd':
                s->deinterlace =
                    s->deinterlace == DEINT_OFF ? DEINT_ON : DEINT_OFF;
                log_msg(LOG_LEVEL_INFO, "Deinterlacing: %s\n",
                        deint_to_string(s->deinterlace));
                return true;
        case 'f':
                s->fs = !s->fs;
                SDL_SetWindowFullscreen(s->window, s->fs);
                return true;
        case 'q':
                exit_uv(0);
                return true;
        default:
                return false;
        }
}

static void
display_sdl3_run(void *arg)
{
        struct state_sdl3 *s = arg;

        while (1) {
                SDL_Event sdl_event;
                if (!SDL_WaitEvent(&sdl_event)) {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "SDL_WaitEvent error: %s\n",
                                SDL_GetError());
                        continue;
                }
                if (sdl_event.type == s->sdl_user_reconfigure_event) {
                        pthread_mutex_lock(&s->lock);
                        struct video_desc desc =
                            *(struct video_desc *) sdl_event.user.data1;
                        s->reconfiguration_status =
                            display_sdl3_reconfigure_real(s, desc);
                        pthread_mutex_unlock(&s->lock);
                        pthread_cond_signal(&s->reconfigured_cv);

                } else if (sdl_event.type == s->sdl_user_new_frame_event) {
                        if (sdl_event.user.data1 ==
                            NULL) { // poison pill received
                                break;
                        }
                        display_frame(
                            s, (struct video_frame *) sdl_event.user.data1);
                } else if (sdl_event.type == s->sdl_user_new_message_event) {
                        struct msg_universal *msg;
                        while ((msg = (struct msg_universal *) check_message(
                                    &s->mod))) {
                                log_msg(LOG_LEVEL_VERBOSE,
                                        MOD_NAME "Received message: %s\n",
                                        msg->text);
                                struct response *r;
                                int              key;
                                if (strstr(msg->text, "win-title ") ==
                                    msg->text) {
                                        SDL_SetWindowTitle(
                                            s->window,
                                            msg->text + strlen("win-title "));
                                        r = new_response(RESPONSE_OK, NULL);
                                } else if (sscanf(msg->text, "%d", &key) == 1) {
                                        if (!display_sdl3_process_key(s, key)) {
                                                r = new_response(
                                                    RESPONSE_BAD_REQUEST,
                                                    "Unsupported key for SDL");
                                        } else {
                                                r = new_response(RESPONSE_OK,
                                                                 NULL);
                                        }
                                } else {
                                        r = new_response(RESPONSE_BAD_REQUEST,
                                                         "Wrong command");
                                }

                                free_message((struct message *) msg, r);
                        }
                } else if (sdl_event.type == SDL_EVENT_KEY_DOWN) {
                        MSG(VERBOSE,
                            "Pressed key %s (scancode: %d, sym: %d, mod: "
                            "%d)!\n",
                            SDL_GetKeyName(sdl_event.key.key),
                            sdl_event.key.scancode, sdl_event.key.key,
                            sdl_event.key.mod);
                        const int64_t sym =
                            translate_sdl_key_to_ug(sdl_event.key);
                        if (sym > 0) {
                                if (!display_sdl3_process_key(
                                        s, sym)) { // unknown key -> pass to
                                                   // control
                                        keycontrol_send_key(
                                            get_root_module(&s->mod), sym);
                                }
                        } else if (sym == -1) {
                                MSG(WARNING,
                                    "Cannot translate key %s (scancode: "
                                    "%d, sym: %d, mod: %d)!\n",
                                    SDL_GetKeyName(sdl_event.key.key),
                                    sdl_event.key.scancode, sdl_event.key.key,
                                    sdl_event.key.mod);
                        }
                        // https://forums.libsdl.org/viewtopic.php?p=38342
                } else if (s->keep_aspect &&
                           sdl_event.type == SDL_EVENT_WINDOW_RESIZED) {
                        double area =
                            sdl_event.window.data1 * sdl_event.window.data2;
                        int width = sqrt(
                            area / ((double) s->current_display_desc.height /
                                    s->current_display_desc.width));
                        int height = sqrt(
                            area / ((double) s->current_display_desc.width /
                                    s->current_display_desc.height));
                        SDL_SetWindowSize(s->window, width, height);
                        MSG(DEBUG, "resizing to %d x %d\n", width, height);
                } else if (sdl_event.type == SDL_EVENT_WINDOW_RESIZED) {
                        // clear both buffers
                        SDL_RenderClear(s->renderer);
                        display_frame(s, s->last_frame);
                        SDL_RenderClear(s->renderer);
                        display_frame(s, s->last_frame);
                } else if (sdl_event.type == SDL_EVENT_QUIT) {
                        exit_uv(0);
                }
        }
}

static void
sdl3_print_displays()
{
        int            count    = 0;
        SDL_DisplayID *displays = SDL_GetDisplays(&count);
        for (int i = 0; i < count; ++i) {
                if (i > 0) {
                        printf(", ");
                }
                const char *dname = SDL_GetDisplayName(displays[i]);
                if (dname == NULL) {
                        dname = SDL_GetError();
                }
                color_printf(TBOLD("%d") " - %s", i, dname);
        }
}

static SDL_DisplayID get_display_id_to_idx(int idx)
{
        int            count    = 0;
        SDL_DisplayID *displays = SDL_GetDisplays(&count);
        if (idx < count) {
                return displays[idx];
        }
        MSG(ERROR, "Display index %d out of range!\n", idx);
        return 0;
}

static void
show_help(const char *driver, bool full)
{
        if (driver != NULL) {
                SDL_SetHint(SDL_HINT_VIDEO_DRIVER, driver);
        }
        SDL_CHECK(SDL_InitSubSystem(SDL_INIT_VIDEO));
        printf("SDL options:\n");
        color_printf(TBOLD(TRED(
            "\t-d sdl") "[[:fs|:d|:display=<didx>|:driver=<drv>|:novsync|:"
                        "renderer=<name[s]>|:nodecorate|:size[=WxH]|:window_"
                        "flags=<f>|:keep-aspect]*]") "\n");
        color_printf(TBOLD(
            "\t-d sdl[:driver=<drv>]:[full]help") "\n");
        printf("where:\n");
        color_printf(TBOLD(
            "\td[force]") " - deinterlace (force even for progressive video)\n");
        color_printf(TBOLD("\t      fs") " - fullscreen\n");
        color_printf(
            TBOLD("\t  <didx>") " - display index, available indices: ");
        sdl3_print_displays();
        color_printf("%s\n", (driver == NULL ? TBOLD(" *")  : ""));
        color_printf(TBOLD("\t   <drv>") " - one of following: ");
        for (int i = 0; i < SDL_GetNumVideoDrivers(); ++i) {
                color_printf("%s" TBOLD("%s"), (i == 0 ? "" : ", "),
                             SDL_GetVideoDriver(i));
        }
        color_printf("\n");
        color_printf(TBOLD("     keep-aspect") " - keep window aspect ratio "
                                               "respective to the video\n");
        color_printf(TBOLD("         novsync") " - disable sync on VBlank\n");
        color_printf(TBOLD("      nodecorate") " - disable window border\n");
        color_printf(
            TBOLD("            size") " - window size in pixels with optional "
                                      "position\n"
                                      "                   "
                                      "(syntax: " TBOLD(
                                          "[<W>x<H>][{+-}<X>[{+-}<Y>]]")
                                      " or mode name)\n");
        color_printf(TBOLD("      <renderer>") " - renderer, one or more of:");
        for (int i = 0; i < SDL_GetNumRenderDrivers(); ++i) {
                const char *renderer_name = SDL_GetRenderDriver(i);
                if (renderer_name != NULL) {
                        color_printf("%s" TBOLD("%s"), (i == 0 ? " " : ", "),
                                     renderer_name);
                }
        }
        printf("\n");
        if (full) {
                color_printf(TBOLD("   blend[=<val>]") " - set alpha blending "
                                                       "(default is opaque)\n");
        }
        if (driver == NULL) {
                color_printf(
                    TBOLD("*") " available values depend on the driver "
                                 "selection. These are for the default driver - "
                                 "specify a driver for help with another "
                                 "one.\n");
        }
        printf("\nKeyboard shortcuts:\n");
        for (unsigned int i = 0; i < sizeof keybindings / sizeof keybindings[0];
             ++i) {
                color_printf("\t" TBOLD("'%c'") "\t - %s\n", keybindings[i].key,
                             keybindings[i].description);
        }
        int ver = SDL_GetVersion();
        printf("\nSDL version (linked): %" PRIu8 ".%" PRIu8 ".%" PRIu8 "\n",
               ver / 1000000, (ver / 1000) % 1000, ver % 1000);
        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        SDL_Quit();
}

static bool
display_sdl3_reconfigure(void *state, struct video_desc desc)
{
        struct state_sdl3 *s = state;

        if (desc.interlacing == INTERLACED_MERGED &&
            s->deinterlace == DEINT_OFF) {
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "Receiving interlaced video but deinterlacing "
                                 "is off - suggesting toggling it on (press "
                                 "'d' or pass cmdline option)\n");
        }
        if (desc.color_spec == R10k) {
                MSG(WARNING,
                    "Displaying 10-bit RGB, which is experimental. In case of "
                    "problems use '--param decoder-use-codec='!R10k'` and "
                    "please report.\n");
        }

        pthread_mutex_lock(&s->lock);

        SDL_Event event;
        event.type       = s->sdl_user_reconfigure_event;
        event.user.data1 = &desc;
        SDL_CHECK(SDL_PushEvent(&event));

        s->reconfiguration_status = -1;
        while (s->reconfiguration_status == -1) {
                pthread_cond_wait(&s->reconfigured_cv, &s->lock);
        }
        pthread_mutex_unlock(&s->lock);

        return s->reconfiguration_status;
}

static const struct fmt_data *
get_ug_to_sdl_format(const struct fmt_data *supp_fmts, codec_t ug_codec)
{
        for (unsigned int i = 0; supp_fmts[i].ug_codec != VC_NONE; ++i) {
                if (supp_fmts[i].ug_codec == ug_codec) {
                        return &supp_fmts[i];
                }
        }
        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong codec: %s\n",
                get_codec_name(ug_codec));
        return NULL;
}

static int
get_supported_pfs(const struct fmt_data *supp_fmts, codec_t *codecs)
{
        bool codec_set[VC_COUNT] = {};
        int count = 0;
        for (int i = 0; supp_fmts[i].ug_codec != VC_NONE; ++i) {
                if (codec_set[supp_fmts[i].ug_codec]) {
                        continue;
                }
                codecs[count++] = supp_fmts[i].ug_codec;
                codec_set[supp_fmts[i].ug_codec] = true;
        }
        return count;
}

static void
query_renderer_supported_fmts(SDL_Renderer    *renderer,
                              struct fmt_data *supp_fmts, bool blacklist_p010)
{
        assert(renderer != NULL);
        assert(supp_fmts != NULL);
        SDL_PropertiesID renderer_props =
            SDL_GetRendererProperties(renderer);
        const SDL_PixelFormat *const fmts = SDL_GetPointerProperty(
            renderer_props, SDL_PROP_RENDERER_TEXTURE_FORMATS_POINTER, NULL);
        if (fmts == NULL) {
                SDL_DestroyProperties(renderer_props);
                MSG(ERROR, "No supported pixel format!\n");
                return;
        }

        if (log_level >= LOG_LEVEL_VERBOSE) {
                const SDL_PixelFormat *it = fmts;
                MSG(VERBOSE, "Supported pixel formats:\n");
                while (*it != SDL_PIXELFORMAT_UNKNOWN) {
                        MSG(VERBOSE, " - %s\n", SDL_GetPixelFormatName(*it++));
                }
        }
        int count = 0;
        for (unsigned i = 0; i < ARR_COUNT(pf_mapping_template); ++i) {
                const SDL_PixelFormat *it = fmts;
                while (*it != SDL_PIXELFORMAT_UNKNOWN) {
                        if (*it == SDL_PIXELFORMAT_P010 && blacklist_p010) {
                                MSG(VERBOSE, "Skipping P010 for D3D1[12] renderers.\n");
                                it++;
                                continue;
                        }
                        if (*it == pf_mapping_template[i].sdl_tex_fmt) {
                                memcpy(&supp_fmts[count++],
                                       &pf_mapping_template[i],
                                       sizeof pf_mapping_template[i]);
                                break;
                        }
                        it++;
                }
                if (*it == SDL_PIXELFORMAT_UNKNOWN) {
                        MSG(DEBUG,
                            "Pixel format %s (%s) not supported by the "
                            "renderer!\n",
                            get_codec_name(pf_mapping_template[i].ug_codec),
                            SDL_GetPixelFormatName(
                                pf_mapping_template[i].sdl_tex_fmt));
                }
        }
        memset(&supp_fmts[count], 0, sizeof supp_fmts[count]); // terminate
        SDL_DestroyProperties(renderer_props);
}

static void
cleanup_frames(struct state_sdl3 *s)
{
        s->last_frame              = NULL;
        struct video_frame *buffer = NULL;
        while ((buffer = simple_linked_list_pop(s->free_frame_queue)) != NULL) {
                vf_free(buffer);
        }
        s->texture_pitch = PITCH_DEFAULT;
}

static void
vf_sdl_texture_data_deleter(struct video_frame *buf)
{
        struct video_frame_sdl3_data *frame_data =
            buf->callbacks.dispose_udata;
        SDL_DestroyTexture(frame_data->texture);
        free(frame_data->preconv_data);
        free(frame_data);
}

static bool
recreate_textures(struct state_sdl3 *s, struct video_desc desc)
{
        cleanup_frames(s);

        for (int i = 0; i < BUFFER_COUNT; ++i) {
                SDL_PropertiesID prop = SDL_CreateProperties();
                SDL_SetNumberProperty(prop,
                                      SDL_PROP_TEXTURE_CREATE_FORMAT_NUMBER,
                                      s->cs_data->sdl_tex_fmt);
                SDL_SetNumberProperty(prop,
                                      SDL_PROP_TEXTURE_CREATE_ACCESS_NUMBER,
                                      SDL_TEXTUREACCESS_STREAMING);
                SDL_SetNumberProperty(prop,
                                      SDL_PROP_TEXTURE_CREATE_WIDTH_NUMBER,
                                      desc.width);
                SDL_SetNumberProperty(prop,
                                      SDL_PROP_TEXTURE_CREATE_HEIGHT_NUMBER,
                                      desc.height);
                if (!codec_is_a_rgb(desc.color_spec)) {
                        const enum SDL_Colorspace cs =
                            get_commandline_param("color-601") != NULL
                                ? SDL_COLORSPACE_BT601_LIMITED
                                : SDL_COLORSPACE_BT709_LIMITED;
                        SDL_SetNumberProperty(
                            prop, SDL_PROP_TEXTURE_CREATE_COLORSPACE_NUMBER,
                            cs);
                }
                if (desc.color_spec == R10k) {
                        SDL_SetNumberProperty(
                            prop, SDL_PROP_TEXTURE_CREATE_COLORSPACE_NUMBER,
                            SDL_COLORSPACE_SRGB);
                }
                SDL_Texture *texture =
                    SDL_CreateTextureWithProperties(s->renderer, prop);
                SDL_SetTextureBlendMode(texture, s->req_blend_mode);
                SDL_DestroyProperties(prop);
                if (!texture) {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME "Unable to create texture: %s\n",
                                SDL_GetError());
                        return false;
                }
                struct video_frame           *f = vf_alloc_desc(desc);
                struct video_frame_sdl3_data *frame_data =
                    calloc(1, sizeof *frame_data);
                frame_data->texture = texture;
                if (s->cs_data->convert != NULL) {
                        frame_data->preconv_data = f->tiles[0].data =
                            malloc(f->tiles[0].data_len);
                } else {
                        SDL_CHECK(SDL_LockTexture(texture, NULL,
                                                  (void **) &f->tiles[0].data,
                                                  &s->texture_pitch),
                                  return false);
                        if (!codec_is_planar(desc.color_spec)) {
                                f->tiles[0].data_len =
                                    desc.height * s->texture_pitch;
                        }
                }
                f->callbacks.dispose_udata = frame_data;
                f->callbacks.data_deleter = vf_sdl_texture_data_deleter;
                simple_linked_list_append(s->free_frame_queue, f);
        }

        return true;
}

static void
vulkan_warn(const char *req_renderers_name, const char *actual_renderer_name)
{
        if (strcmp(actual_renderer_name, "vulkan") != 0) {
                return;
        }
        bool explicit = req_renderers_name[0] != '\0';
        log_msg(explicit ? LOG_LEVEL_WARNING : LOG_LEVEL_ERROR,
                "Selected renderer vulkan is known for having "
                "issues!%s\n",
                explicit ? "" : " Please report!");
}

static bool
display_sdl3_reconfigure_real(void *state, struct video_desc desc)
{
        struct state_sdl3 *s = state;

        MSG(NOTICE, "Reconfigure to size %dx%d\n", desc.width, desc.height);

        if (s->fixed_size && s->window) {
                goto skip_window_creation;
        }

        if (s->window) {
                SDL_DestroyWindow(s->window);
        }
        int flags = s->window_flags | SDL_WINDOW_RESIZABLE |
                    SDL_WINDOW_HIGH_PIXEL_DENSITY;
        if (s->fs) {
                flags |= SDL_WINDOW_FULLSCREEN;
        }
        const char *window_title = "UltraGrid - SDL3 Display";
        if (get_commandline_param("window-title")) {
                window_title = get_commandline_param("window-title");
        }
        int width  = s->fixed_w ? s->fixed_w : desc.width;
        int height = s->fixed_h ? s->fixed_h : desc.height;
        const SDL_DisplayID display_id = get_display_id_to_idx(s->display_idx);
        int x      = s->x == SDL_WINDOWPOS_UNDEFINED
                         ? (int) SDL_WINDOWPOS_CENTERED_DISPLAY(display_id)
                         : s->x;
        int y      = s->y == SDL_WINDOWPOS_UNDEFINED
                         ? (int) SDL_WINDOWPOS_CENTERED_DISPLAY(display_id)
                         : s->y;
        s->window  = SDL_CreateWindow(window_title, width, height, flags);
        if (!s->window) {
                MSG(ERROR, "Unable to create window: %s\n", SDL_GetError());
                return false;
        }
        SDL_SetWindowPosition(s->window, x, y);

        if (s->renderer) {
                SDL_DestroyRenderer(s->renderer);
        }
        SDL_PropertiesID renderer_prop = SDL_CreateProperties();
        SDL_SetPointerProperty(
            renderer_prop, SDL_PROP_RENDERER_CREATE_WINDOW_POINTER, s->window);
        if (strlen(s->req_renderers_name) > 0) {
                SDL_SetStringProperty(renderer_prop,
                                      SDL_PROP_RENDERER_CREATE_NAME_STRING,
                                      s->req_renderers_name);
        }
        s->renderer = SDL_CreateRendererWithProperties(renderer_prop);
        SDL_DestroyProperties(renderer_prop);
        if (!s->renderer) {
                MSG(ERROR, "Unable to create renderer: %s\n", SDL_GetError());
                return false;
        }
        if (s->vsync) {
                // try adaptive first, if it doesn't succeed try 1
                if (!SDL_SetRenderVSync(s->renderer,
                                        SDL_RENDERER_VSYNC_ADAPTIVE)) {
                        SDL_CHECK(SDL_SetRenderVSync(s->renderer, 1));
                }
        }
        const char *renderer_name = SDL_GetRendererName(s->renderer);
        bool is_d3d = false;
        if (renderer_name != NULL) {
                is_d3d = strstr(renderer_name, "direct3d") == renderer_name;
                MSG(NOTICE, "Using renderer: %s\n", renderer_name);
                vulkan_warn(s->req_renderers_name, renderer_name);
        }
        query_renderer_supported_fmts(s->renderer, s->supp_fmts, is_d3d);
skip_window_creation:
        s->cs_data = get_ug_to_sdl_format(s->supp_fmts, desc.color_spec);
        if (s->cs_data == NULL) {
                return false;
        }
        MSG(VERBOSE, "Setting SDL3 pix fmt: %s\n",
            SDL_GetPixelFormatName(s->cs_data->sdl_tex_fmt));

        SDL_SetRenderLogicalPresentation(s->renderer, desc.width, desc.height,
                                         SDL_LOGICAL_PRESENTATION_LETTERBOX);

        if (!recreate_textures(s, desc)) {
                return false;
        }

        s->current_display_desc = desc;

        return true;
}

static void
loadSplashscreen(struct state_sdl3 *s)
{
        struct video_frame *frame = get_splashscreen();
        if (!display_sdl3_reconfigure_real(s, video_desc_from_frame(frame))) {
                MSG(WARNING, "Cannot render splashscreeen!\n");
                vf_free(frame);
                return;
        }
        struct video_frame *splash = display_sdl3_getf(s);
        memcpy(splash->tiles[0].data, frame->tiles[0].data,
               frame->tiles[0].data_len);
        vf_free(frame);
        display_frame(s, splash); // don't be tempted to use _putf() - it will
                                  // use event queue and there may arise a
                                  // race-condition with recv thread
}

static bool
set_size(struct state_sdl3 *s, const char *tok)
{
        if (strstr(tok, "fixed_size=") == tok) {
                log_msg(LOG_LEVEL_WARNING,
                        MOD_NAME "fixed_size with dimensions is "
                                 " deprecated, use size"
                                 " instead\n");
        }
        tok = strchr(tok, '=') + 1;
        if (strpbrk(tok, "x+-") == NULL) {
                struct video_desc desc = get_video_desc_from_string(tok);
                if (desc.width != 0) {
                        s->fixed_size = true;
                        s->fixed_w = desc.width;
                        s->fixed_h = desc.height;
                        return true;
                }
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Wrong size spec: %s\n", tok);
                return false;
        }
        if (strchr(tok, 'x') != NULL) {
                s->fixed_size = true;
                s->fixed_w    = atoi(tok);
                s->fixed_h    = atoi(strchr(tok, 'x') + 1);
        }
        tok = strpbrk(tok, "+-");
        if (tok != NULL) {
                s->x = atoi(tok);
                tok  = strpbrk(tok + 1, "+-");
        }
        if (tok != NULL) {
                s->y = atoi(tok);
        }
        return true;
}

static void
sdl_set_log_level()
{
        if (log_level <= LOG_LEVEL_INFO) {
                return;
        }
        // SDL_INFO corresponds rather to UG verbose and SDL_VERBOSE is actually
        // more detailed than SDL_DEBUG so map this to UG debug
        SDL_SetLogPriorities(log_level == LOG_LEVEL_VERBOSE
                                 ? SDL_LOG_PRIORITY_INFO
                                 : SDL_LOG_PRIORITY_VERBOSE);
}

static void *
display_sdl3_init(struct module *parent, const char *fmt, unsigned int flags)
{
        sdl_set_log_level();

        if (flags & DISPLAY_FLAG_AUDIO_ANY) {
                MSG(ERROR,
                    "UltraGrid SDL3 module currently doesn't support audio!\n");
                return NULL;
        }
        const char        *driver = NULL;
        struct state_sdl3 *s      = calloc(1, sizeof *s);

        s->magic  = MAGIC_SDL3;
        s->x = s->y = SDL_WINDOWPOS_UNDEFINED;
        s->vsync    = true;

        if (fmt == NULL) {
                fmt = "";
        }
        char buf[STR_LEN];
        snprintf(buf, sizeof buf, "%s", fmt);
        char *tmp = buf;
        char *tok, *save_ptr;
        while ((tok = strtok_r(tmp, ":", &save_ptr))) {
                if (strcmp(tok, "d") == 0 || strcmp(tok, "dforce") == 0) {
                        s->deinterlace =
                            strcmp(tok, "d") == 0 ? DEINT_ON : DEINT_OFF;
                } else if (IS_KEY_PREFIX(tok, "display")) {
                        s->display_idx = atoi(strchr(tok, '=') + 1);
                } else if (IS_KEY_PREFIX(tok, "driver")) {
                        driver = strchr(tok, '=') + 1;
                        ;
                } else if (IS_PREFIX(tok, "fs")) {
                        s->fs = true;
                } else if (IS_PREFIX(tok, "help") || strcmp(tok, "fullhelp") == 0) {
                        show_help(driver, strcmp(tok, "fullhelp") == 0);
                        free(s);
                        return INIT_NOERR;
                } else if (IS_PREFIX(tok, "novsync")) {
                        s->vsync = false;
                } else if (IS_PREFIX(tok, "nodecorate")) {
                        s->window_flags |= SDL_WINDOW_BORDERLESS;
                } else if (IS_PREFIX(tok, "keep-aspect")) {
                        s->keep_aspect = true;
                } else if (IS_KEY_PREFIX(tok, "fixed_size") ||
                           IS_KEY_PREFIX(tok, "size")) {
                        if (!set_size(s, tok)) {
                                free(s);
                                return NULL;
                        }
                } else if (strcmp(tok, "fixed_size") == 0) {
                        log_msg(LOG_LEVEL_WARNING,
                                MOD_NAME "fixed_size deprecated, use size with "
                                         "dimensions\n");
                        s->fixed_size = true;
                } else if (IS_KEY_PREFIX(tok, "window_flags")) {
                        int f;
                        if (sscanf(strchr(tok, '=') + 1, "%i", &f) != 1) {
                                log_msg(LOG_LEVEL_ERROR,
                                        "Wrong window_flags: %s\n", tok);
                                free(s);
                                return NULL;
                        }
                        s->window_flags |= f;
                } else if (IS_KEY_PREFIX(tok, "position")) {
                        tok = strchr(tok, '=') + 1;
                        if (strchr(tok, ',') == NULL) {
                                MSG(ERROR, "position: %s\n", tok);
                                free(s);
                                return NULL;
                        }
                        s->x = atoi(tok);
                        s->y = atoi(strchr(tok, ',') + 1);
                        log_msg(LOG_LEVEL_WARNING,
                                MOD_NAME "pos is deprecated, use "
                                         "\"size=%+d%+d\" instead.\n",
                                s->x, s->y);
                } else if (IS_KEY_PREFIX(tok, "renderer")) {
                        snprintf_ch(s->req_renderers_name, "%s",
                                    strchr(tok, '=') + 1);
                } else if (IS_KEY_PREFIX(tok, "blend")) {
                        s->req_blend_mode = atoi(strchr(tok, '=') + 1);
                } else if (strcmp(tok, "blend") == 0) {
                        s->req_blend_mode = SDL_BLENDMODE_BLEND;
                } else {
                        MSG(ERROR, "Wrong option: %s\n", tok);
                        free(s);
                        return NULL;
                }
                tmp = NULL;
        }

#ifdef __linux__
        if (driver == NULL && getenv("DISPLAY") == NULL &&
            getenv("WAYLAND_DISPLAY") == NULL) {
                MSG(NOTICE, "X11/Wayland doesn't seem to be running (according "
                            "to env vars). Setting driver=KMSDRM.\n");
                driver = "KMSDRM";
        }
#endif // defined __linux__

        if (driver != NULL) {
                SDL_SetHint(SDL_HINT_VIDEO_DRIVER, driver);
        }
        if (!SDL_InitSubSystem(SDL_INIT_VIDEO)) {
                MSG(ERROR, "Unable to initialize SDL3 video: %s\n",
                    SDL_GetError());
                free(s);
                return NULL;
        }
        if (!SDL_InitSubSystem(SDL_INIT_EVENTS)) {
                MSG(ERROR, "Unable to initialize SDL3 events: %s\n",
                    SDL_GetError());
                free(s);
                return NULL;
        }
        MSG(NOTICE, "Using driver: %s\n", SDL_GetCurrentVideoDriver());

        SDL_HideCursor();
        SDL_DisableScreenSaver();

        module_init_default(&s->mod);
        s->mod.new_message = display_sdl3_new_message;
        s->mod.cls         = MODULE_CLASS_DATA;
        module_register(&s->mod, parent);

        pthread_mutex_init(&s->lock, NULL);
        pthread_cond_init(&s->frame_consumed_cv, NULL);
        pthread_cond_init(&s->reconfigured_cv, NULL);

        s->sdl_user_new_frame_event = SDL_RegisterEvents(3);
        assert(s->sdl_user_new_frame_event != (Uint32) -1);
        s->sdl_user_new_message_event = s->sdl_user_new_frame_event + 1;
        s->sdl_user_reconfigure_event = s->sdl_user_new_frame_event + 2;

        s->free_frame_queue = simple_linked_list_init();

        for (unsigned int i = 0; i < sizeof keybindings / sizeof keybindings[0];
             ++i) {
                if (keybindings[i].key ==
                    'q') { // don't report 'q' to avoid accidental close - user
                           // can use Ctrl-c there
                        continue;
                }
                char key_str[128];
                snprintf(key_str, sizeof key_str, "%d", keybindings[i].key);
                keycontrol_register_key(&s->mod, keybindings[i].key, key_str,
                                        keybindings[i].description);
        }

        loadSplashscreen(s);

        log_msg(LOG_LEVEL_NOTICE, "SDL3 initialized successfully.\n");

        return (void *) s;
}

static void
display_sdl3_done(void *state)
{
        struct state_sdl3 *s = state;

        assert(s->magic == MAGIC_SDL3);

        cleanup_frames(s);

        if (s->renderer) {
                SDL_DestroyRenderer(s->renderer);
        }

        if (s->window) {
                SDL_DestroyWindow(s->window);
        }

        SDL_ShowCursor();

        SDL_QuitSubSystem(SDL_INIT_VIDEO);
        SDL_QuitSubSystem(SDL_INIT_EVENTS);
        SDL_Quit();

        simple_linked_list_destroy(s->free_frame_queue);

        pthread_mutex_destroy(&s->lock);
        pthread_cond_destroy(&s->frame_consumed_cv);
        pthread_cond_destroy(&s->reconfigured_cv);

        module_done(&s->mod);

        free(s);
}

static struct video_frame *
display_sdl3_getf(void *state)
{
        struct state_sdl3 *s = state;
        assert(s->magic == MAGIC_SDL3);

        pthread_mutex_lock(&s->lock);
        while (simple_linked_list_size(s->free_frame_queue) == 0) {
                pthread_cond_wait(&s->frame_consumed_cv, &s->lock);
        }
        struct video_frame *buffer =
            simple_linked_list_pop(s->free_frame_queue);
        pthread_mutex_unlock(&s->lock);

        return buffer;
}

static void
convert_R10k_ARGB2101010(const struct video_frame *uv_frame,
                         unsigned char *tex_data, size_t pitch)
{
        assert(pitch == (size_t) uv_frame->tiles[0].width * 4);
        assert((uintptr_t) uv_frame->tiles[0].data % 4 == 0);
        assert((uintptr_t) tex_data % 4 == 0);
        const uint32_t *in  = (const void *) uv_frame->tiles[0].data;
        uint32_t       *out = (void *) tex_data;
        const size_t    count =
            (size_t) uv_frame->tiles[0].width * uv_frame->tiles[0].height;
        enum {
                LOOP_ITEMS = 16,
        };
        unsigned int i = 0;
        for (; i < count / LOOP_ITEMS; ++i) {
                for (int j = 0; j < LOOP_ITEMS; ++j) {
                        uint32_t val = be32toh(*in++);
                        *out++       = htole32(0x3 << 30 | val >> 2);
                }
        }
        i *= LOOP_ITEMS;
        for (; i < count; ++i) {
                uint32_t val = be32toh(*in++);
                *out++       = htole32(0x3 << 30 | val >> 2);
        }
}

static void
convert_R10k_ABGR2101010(const struct video_frame *uv_frame,
                         unsigned char *tex_data, size_t pitch)
{
        const size_t src_linesize = vc_get_linesize(uv_frame->tiles[0].width, R10k);
        for (unsigned i = 0; i < uv_frame->tiles[0].height; ++i) {
                const uint8_t *in =
                    (uint8_t *) uv_frame->tiles[0].data + (i * src_linesize);
                uint32_t *out = (void *) (tex_data + (i * pitch));
                assert((uintptr_t) out% 4 == 0);
                for (unsigned i = 0; i < uv_frame->tiles[0].width; ++i) {
                        uint32_t val  = 0x3 << 30 |                           // A
                                 (in[3] & 0xfc) << 18 | (in[2] & 0xf) << 26 | // B
                                 (in[2] & 0xf0) << 6 | (in[1] & 0x3f) << 14 | // G
                                 (in[1] & 0xc0) >> 6 | in[0] << 2;            // R
                        *out++ = htole32(val);
                        in += 4;
                }
        }
}

static void
convert_RGBA_BGRA(const struct video_frame *uv_frame, unsigned char *tex_data,
                  size_t pitch)
{
        unsigned char *out_data[2]   = { tex_data, 0 };
        int out_linesize[2] = { (int) pitch, 0 };
        rgba_to_bgra(
            out_data, out_linesize, (unsigned char *) uv_frame->tiles[0].data,
            (int) uv_frame->tiles[0].width, (int) uv_frame->tiles[0].height);
}

static void
convert_UYVY_IYUV(const struct video_frame *uv_frame, unsigned char *tex_data,
                  size_t y_pitch)
{
        const size_t y_h = uv_frame->tiles[0].height;
        const size_t chr_h = (y_h + 1) / 2;
        int          out_linesize[3] = { (int) y_pitch,
                                         (int) (y_pitch + 1) / 2,
                                         (int) (y_pitch + 1) / 2 };
        unsigned char *out_data[3] = { tex_data,
                                       tex_data + (y_h * out_linesize[0]),
                                       tex_data + (y_h * out_linesize[0]) +
                                           (chr_h * out_linesize[1]) };
        uyvy_to_i420(
            out_data, out_linesize, (unsigned char *) uv_frame->tiles[0].data,
            (int) uv_frame->tiles[0].width, (int) uv_frame->tiles[0].height);
}

static void
convert_UYVY_NV12(const struct video_frame *uv_frame, unsigned char *tex_data,
                  size_t y_pitch)
{
        unsigned char *out_data[2] = {
                tex_data, tex_data + (y_pitch * uv_frame->tiles[0].height)
        };
        int out_linesize[2] = { (int) y_pitch, (int) ((y_pitch + 1) / 2) * 2 };
        uyvy_to_nv12(
            out_data, out_linesize, (unsigned char *) uv_frame->tiles[0].data,
            (int) uv_frame->tiles[0].width, (int) uv_frame->tiles[0].height);
}

/**
 * @todo
 * currently seem to work only on Metal
 */
static void
convert_Y216_P010(const struct video_frame *uv_frame, unsigned char *tex_data,
                  size_t y_pitch)
{
        unsigned char *out_data[2] = {
                tex_data, tex_data + (y_pitch * uv_frame->tiles[0].height)
        };
        int out_linesize[2] = { (int) y_pitch, (int) ((y_pitch + 1) / 2) * 2 };
        y216_to_p010le(
            out_data, out_linesize, (unsigned char *) uv_frame->tiles[0].data,
            (int) uv_frame->tiles[0].width, (int) uv_frame->tiles[0].height);
}

/// @copydoc convert_Y216_P010
static void
convert_v210_P010(const struct video_frame *uv_frame, unsigned char *tex_data,
                  size_t y_pitch)
{
        unsigned char *out_data[2] = {
                tex_data, tex_data + (y_pitch * uv_frame->tiles[0].height)
        };
        int out_linesize[2] = { (int) y_pitch, (int) ((y_pitch + 1) / 2) * 2 };
        v210_to_p010le(
            out_data, out_linesize, (unsigned char *) uv_frame->tiles[0].data,
            (int) uv_frame->tiles[0].width, (int) uv_frame->tiles[0].height);
}

static bool
display_sdl3_putf(void *state, struct video_frame *frame, long long timeout_ns)
{
        struct state_sdl3 *s = state;

        assert(s->magic == MAGIC_SDL3);

        if (frame == NULL) { // poison pill
                SDL_Event event;
                event.type       = s->sdl_user_new_frame_event;
                event.user.data1 = NULL;
                SDL_CHECK(SDL_PushEvent(&event));
                return true;
        }

        pthread_mutex_lock(&s->lock);
        if (timeout_ns == PUTF_DISCARD) {
                assert(frame != NULL);
                simple_linked_list_append(s->free_frame_queue, frame);
                pthread_mutex_unlock(&s->lock);
                return true;
        }

        if (timeout_ns > 0) {
                int rc = 0;
                while (rc == 0 &&
                       simple_linked_list_size(s->free_frame_queue) == 0) {
                        if (timeout_ns == PUTF_BLOCKING) {
                                rc = pthread_cond_wait(&s->frame_consumed_cv,
                                                       &s->lock);
                        } else {
                                struct timespec ts;
                                timespec_get(&ts, TIME_UTC);
                                ts_add_nsec(&ts, timeout_ns);
                                rc = pthread_cond_timedwait(
                                    &s->frame_consumed_cv, &s->lock, &ts);
                        }
                }
        }
        if (simple_linked_list_size(s->free_frame_queue) == 0) {
                simple_linked_list_append(s->free_frame_queue, frame);
                log_msg(LOG_LEVEL_INFO, MOD_NAME "1 frame(s) dropped!\n");
                pthread_mutex_unlock(&s->lock);
                return false;
        }
        pthread_mutex_unlock(&s->lock);
        SDL_Event event;
        event.type       = s->sdl_user_new_frame_event;
        event.user.data1 = frame;
        SDL_CHECK(SDL_PushEvent(&event));

        return true;
}

static bool
display_sdl3_get_property(void *state, int property, void *val, size_t *len)
{
        struct state_sdl3 *s = state;

        switch (property) {
        case DISPLAY_PROPERTY_CODECS: {
                codec_t codecs[VIDEO_CODEC_COUNT];
                const size_t codecs_len =
                    get_supported_pfs(s->supp_fmts, codecs) * sizeof(codec_t);
                if (codecs_len <= *len) {
                        memcpy(val, codecs, codecs_len);
                        *len = codecs_len;
                } else {
                        return false;
                }
                break;
        }
        case DISPLAY_PROPERTY_BUF_PITCH:
                *(int *) val =
                    codec_is_planar(s->current_display_desc.color_spec)
                        ? PITCH_DEFAULT
                        : s->texture_pitch;
                *len = sizeof(int);
                break;
        default:
                return false;
        }
        return true;
}

static void
display_sdl3_new_message(struct module *mod)
{
        struct state_sdl3 *s = (struct state_sdl3 *) mod;

        SDL_Event event;
        event.type = s->sdl_user_new_message_event;
        SDL_CHECK(SDL_PushEvent(&event));
}

static void
display_sdl3_probe(struct device_info **available_cards, int *count,
                   void (**deleter)(void *))
{
        UNUSED(deleter);
        *count = 1;
        *available_cards =
            (struct device_info *) calloc(1, sizeof(struct device_info));
        strcpy((*available_cards)[0].dev, "");
        strcpy((*available_cards)[0].name, "SDL3 SW display");
        dev_add_option(&(*available_cards)[0], "Deinterlace", "Deinterlace",
                       "deinterlace", ":d", true);
        dev_add_option(&(*available_cards)[0], "Fullscreen",
                       "Launch as fullscreen", "fullscreen", ":fs", true);
        dev_add_option(&(*available_cards)[0], "No decorate",
                       "Disable window decorations", "nodecorate",
                       ":nodecorate", true);
        dev_add_option(&(*available_cards)[0], "Disable vsync", "Disable vsync",
                       "novsync", ":novsync", true);

        (*available_cards)[0].repeatable = true;
}

static const struct video_display_info display_sdl3_info = {
        display_sdl3_probe,
        display_sdl3_init,
        display_sdl3_run,
        display_sdl3_done,
        display_sdl3_getf,
        display_sdl3_putf,
        display_sdl3_reconfigure,
        display_sdl3_get_property,
        NULL,
        NULL,
        MOD_NAME,
};

REGISTER_MODULE(sdl, &display_sdl3_info, LIBRARY_CLASS_VIDEO_DISPLAY,
                VIDEO_DISPLAY_ABI_VERSION);
REGISTER_MODULE_WITH_FLAG(sdl3, &display_sdl3_info, LIBRARY_CLASS_VIDEO_DISPLAY,
                          VIDEO_DISPLAY_ABI_VERSION, MODULE_FLAG_ALIAS);
