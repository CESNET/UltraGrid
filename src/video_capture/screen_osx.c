/**
 * @file   video_capture/screen_osx.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2025 CESNET
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

#include <Carbon/Carbon.h>
#include <alloca.h>
#include <assert.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <strings.h>
#include <string.h>
#include <sys/time.h>

#include "debug.h"
#include "lib_common.h"
#include "pixfmt_conv.h"
#include "tv.h"
#include "types.h"
#include "utils/video_frame_pool.h"
#include "video_capture.h"
#include "video_capture_params.h"
#include "video_codec.h"
#include "video_frame.h"
struct audio_frame;
struct vidcap_params;

#define MAX_DISPLAY_COUNT 10
#define MOD_NAME "[screen cap mac] "


/* prototypes of functions defined in this module */
static void show_help(void);
static void vidcap_screen_osx_done(void *state);

static void show_help()
{
        printf("Screen capture\n");
        printf("Usage\n");
        printf("\t-t screen[:fps=<fps>][:codec=<c>][:display=<d>]\n");
        printf("\t\t<fps> - preferred grabbing fps (otherwise unlimited)\n");
        printf("\t\t <c>  - requested codec to capture (RGB /default/ or RGBA)\n");
        printf("\t\t <d>  - display ID or \"primary\" or \"secondary\"\n");
        printf("\n\nAvailable displays:\n");

        CGDirectDisplayID screens[MAX_DISPLAY_COUNT];
        uint32_t count = 0;
        CGGetOnlineDisplayList(sizeof screens / sizeof screens[0], screens, &count);

        for (unsigned int i = 0; i < count; ++i) {
                char flags[128];
                strcpy(flags, CGDisplayIsMain(screens[i]) ? "primary" : "secondary");
                if (CGDisplayIsBuiltin(screens[i])) {
                        strncat(flags, ", builtin", sizeof flags - strlen(flags) - 1);
                }
                printf("\tID %u) %s\n", screens[i], flags);
        }
}

struct vidcap_screen_osx_state {
        struct video_desc desc;
        void *video_frame_pool;
        int frames;
        struct       timeval t, t0;
        CGDirectDisplayID display;
        decoder_t       decode; ///< decoder, must accept BGRA (different shift)

        struct timeval prev_time;

        bool initialized;
};

static bool initialize(struct vidcap_screen_osx_state *s) {
        CGImageRef image = CGDisplayCreateImage(s->display);
        if (image == NULL) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unable create image (wrong display ID?)\n");
                return false;
        }

        s->desc.width = CGImageGetWidth(image);
        s->desc.height = CGImageGetHeight(image);
        CFRelease(image);
        s->video_frame_pool = video_frame_pool_init(s->desc, 2);

        return true;
}

static void vidcap_screen_osx_probe(struct device_info **available_cards, int *count, void (**deleter)(void *))
{
        *deleter = free;
        *count = 1;
        *available_cards = calloc(*count, sizeof(struct device_info));

        snprintf((*available_cards)[0].name, sizeof (*available_cards)[0].name, "Screen capture");

        int framerates[] = {24, 30, 60};

        snprintf((*available_cards)[0].modes[0].name, sizeof (*available_cards)[0].name,
                        "Unlimited fps");
        snprintf((*available_cards)[0].modes[0].id, sizeof (*available_cards)[0].modes[0].id,
                        "{\"fps\":\"\"}");

        for(unsigned i = 0; i < sizeof(framerates) / sizeof(framerates[0]); i++){
                snprintf((*available_cards)[0].modes[i + 1].name, sizeof (*available_cards)[0].name,
                                "%d fps", framerates[i]);
                snprintf((*available_cards)[0].modes[i + 1].id, sizeof (*available_cards)[0].modes[0].id,
                                "{\"fps\":\"%d\"}", framerates[i]);
        }
}

static int vidcap_screen_osx_init(struct vidcap_params *params, void **state)
{
        struct vidcap_screen_osx_state *s;

        printf("vidcap_screen_init\n");

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPORTED;
        }

        s = (struct vidcap_screen_osx_state *) calloc(1, sizeof(struct vidcap_screen_osx_state));
        if(s == NULL) {
                printf("Unable to allocate screen capture state\n");
                return VIDCAP_INIT_FAIL;
        }

        s->initialized = false;

        gettimeofday(&s->t0, NULL);

        s->desc.tile_count = 1;
        s->desc.color_spec = RGB;
        s->desc.fps = 30;
        s->desc.interlacing = PROGRESSIVE;

        s->display = CGMainDisplayID();

        if (vidcap_params_get_fmt(params) && strlen(vidcap_params_get_fmt(params)) > 0) {
                if (strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                        show_help();
                        return VIDCAP_INIT_NOERR;
                }
                char *fmt = alloca(strlen(vidcap_params_get_fmt(params) + 1));
                strcpy(fmt, vidcap_params_get_fmt(params));
                char *save_ptr = NULL;
                char *item = NULL;
                while ((item = strtok_r(fmt, ":", &save_ptr)) != NULL) {
                        if (strncasecmp(item, "fps=", strlen("fps=")) == 0) {
                                s->desc.fps = atof(item + strlen("fps="));
                        } else if (strncasecmp(item, "codec=", strlen("codec=")) == 0) {
                                s->desc.color_spec = get_codec_from_name(item + strlen("codec="));
                        } else if (strncasecmp(item, "display=", strlen("display=")) == 0) {
                                char *display = item + strlen("display=");

                                if (strcasecmp(display, "secondary") == 0) {
                                        CGDirectDisplayID screens[MAX_DISPLAY_COUNT];
                                        uint32_t count = 0;
                                        CGGetOnlineDisplayList(sizeof screens / sizeof screens[0], screens, &count);
                                        uint32_t i = 0;
                                        for (; i < count; ++i) {
                                                if (!CGDisplayIsMain(screens[i])) {
                                                        s->display = screens[i];
                                                        break;
                                                }
                                        }
                                        if (i == count) {
                                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "No secondary screen found!\n");
                                                vidcap_screen_osx_done(s);
                                                return VIDCAP_INIT_FAIL;
                                        }
                                } if (strcasecmp(display, "primary") != 0) { // primary was already set
                                        s->display = atol(display);
                                }
                        } else {
                                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Unrecognized option \"%s\"\n", item);
                                vidcap_screen_osx_done(s);
                                return VIDCAP_INIT_FAIL;
                        }
                        fmt = NULL;
                }
        }

        switch (s->desc.color_spec) {
        case RGB:
                s->decode = vc_copylineBGRAtoRGB;
                break;
        case RGBA:
                s->decode = vc_copylineRGBA;
                break;
        default:
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Only RGB and RGBA are currently supported!\n");
                vidcap_screen_osx_done(s);
                return VIDCAP_INIT_FAIL;
        }

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_screen_osx_done(void *state)
{
        struct vidcap_screen_osx_state *s = (struct vidcap_screen_osx_state *) state;

        assert(s != NULL);

        video_frame_pool_destroy(s->video_frame_pool);

        free(s);
}

static struct video_frame * vidcap_screen_osx_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_screen_osx_state *s = (struct vidcap_screen_osx_state *) state;

        if (!s->initialized) {
                s->initialized = initialize(s);
                if (!s->initialized) {
                        return NULL;
                }
        }

        struct video_frame *frame = video_frame_pool_get_disposable_frame(s->video_frame_pool);
        struct tile *tile = vf_get_tile(frame, 0);

        *audio = NULL;

        CGImageRef image = CGDisplayCreateImage(s->display);
        CFDataRef data = CGDataProviderCopyData(CGImageGetDataProvider(image));
        const unsigned char *pixels = CFDataGetBytePtr(data);

        int src_linesize = tile->width * 4;
        int dst_linesize = vc_get_linesize(tile->width, frame->color_spec);
        unsigned char *dst = (unsigned char *) tile->data;
        const unsigned char *src = (const unsigned char *) pixels;
        for (unsigned int y = 0; y < tile->height; ++y) {
                s->decode(dst, src, dst_linesize, 16, 8, 0);
                src += src_linesize;
                dst += dst_linesize;
        }

        CFRelease(data);
        CFRelease(image);

        struct timeval cur_time;
        gettimeofday(&cur_time, NULL);
        while(tv_diff_usec(cur_time, s->prev_time) < 1000000.0 / frame->fps) {
                gettimeofday(&cur_time, NULL);
        }
        s->prev_time = cur_time;

        gettimeofday(&s->t, NULL);
        double seconds = tv_diff(s->t, s->t0);        
        if (seconds >= 5) {
                float fps  = s->frames / seconds;
                log_msg(LOG_LEVEL_INFO, "[screen capture] %d frames in %g seconds = %g FPS\n", s->frames, seconds, fps);
                s->t0 = s->t;
                s->frames = 0;
        }

        s->frames++;

        return frame;
}

static const struct video_capture_info vidcap_screen_osx_info = {
        vidcap_screen_osx_probe,
        vidcap_screen_osx_init,
        vidcap_screen_osx_done,
        vidcap_screen_osx_grab,
        VIDCAP_NO_GENERIC_FPS_INDICATOR,
};

REGISTER_MODULE(screen, &vidcap_screen_osx_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

