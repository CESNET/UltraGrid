/**
 * @file   screen_osx.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2012-2013 CESNET, z.s.p.o.
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
#endif /* HAVE_CONFIG_H */

#include "debug.h"
#include "host.h"
#include "lib_common.h"
#include "video.h"
#include "video_capture.h"

#include "tv.h"

#include "audio/audio.h"

#include <stdio.h>
#include <stdlib.h>
#include <strings.h>

#include <pthread.h>

#include <Carbon/Carbon.h>

/* prototypes of functions defined in this module */
static void show_help(void);

static void show_help()
{
        printf("Screen capture\n");
        printf("Usage\n");
        printf("\t-t screen[:fps=<fps>]\n");
        printf("\t\t<fps> - preferred grabbing fps (otherwise unlimited)\n");
}

struct vidcap_screen_osx_state {
        struct video_frame       *frame; 
        struct tile       *tile; 
        int frames;
        struct       timeval t, t0;
        CGDirectDisplayID display;

        struct timeval prev_time;

        double fps;

        bool initialized;
};

static void initialize(struct vidcap_screen_osx_state *s) {
        s->frame = vf_alloc(1);
        s->tile = vf_get_tile(s->frame, 0);

        s->display = CGMainDisplayID();
        CGImageRef image = CGDisplayCreateImage(s->display);

        s->tile->width = CGImageGetWidth(image);
        s->tile->height = CGImageGetHeight(image);
        CFRelease(image);

        s->frame->color_spec = RGBA;
        if(s->fps > 0.0) {
                s->frame->fps = s->fps;
        } else {
                s->frame->fps = 30;
        }
        s->frame->interlacing = PROGRESSIVE;
        s->tile->data_len = vc_get_linesize(s->tile->width, s->frame->color_spec) * s->tile->height;

        s->tile->data = (char *) malloc(s->tile->data_len);

        return;

        goto error; // dummy use (otherwise compiler would complain about unreachable code (Mac)
error:
        fprintf(stderr, "[Screen cap.] Initialization failed!\n");
        exit_uv(EXIT_FAILURE);
}

static struct vidcap_type * vidcap_screen_osx_probe(bool verbose)
{
        struct vidcap_type*		vt;

        vt = (struct vidcap_type *) calloc(1, sizeof(struct vidcap_type));
        if (vt != NULL) {
                vt->name        = "screen";
                vt->description = "Grabbing screen";

                if (verbose) {
                        vt->card_count = 1;
                        vt->cards = calloc(vt->card_count, sizeof(struct device_info));
                        // vt->cards[0].id can be "" since screen cap. doesn't require parameters
                        snprintf(vt->cards[0].name, sizeof vt->cards[0].name, "Screen capture");
                }
        }
        return vt;
}

static int vidcap_screen_osx_init(const struct vidcap_params *params, void **state)
{
        struct vidcap_screen_osx_state *s;

        printf("vidcap_screen_init\n");

        if (vidcap_params_get_flags(params) & VIDCAP_FLAG_AUDIO_ANY) {
                return VIDCAP_INIT_AUDIO_NOT_SUPPOTED;
        }

        s = (struct vidcap_screen_osx_state *) malloc(sizeof(struct vidcap_screen_osx_state));
        if(s == NULL) {
                printf("Unable to allocate screen capture state\n");
                return VIDCAP_INIT_FAIL;
        }

        s->initialized = false;

        gettimeofday(&s->t0, NULL);

        s->fps = 0.0;

        s->frame = NULL;
        s->tile = NULL;

        s->prev_time.tv_sec = 
                s->prev_time.tv_usec = 0;

        s->frames = 0;

        if(vidcap_params_get_fmt(params)) {
                if (strcmp(vidcap_params_get_fmt(params), "help") == 0) {
                        show_help();
                        return VIDCAP_INIT_NOERR;
                } else if (strncasecmp(vidcap_params_get_fmt(params), "fps=", strlen("fps=")) == 0) {
                        s->fps = atoi(vidcap_params_get_fmt(params) + strlen("fps="));
                }
        }

        *state = s;
        return VIDCAP_INIT_OK;
}

static void vidcap_screen_osx_done(void *state)
{
        struct vidcap_screen_osx_state *s = (struct vidcap_screen_osx_state *) state;

        assert(s != NULL);

        if(s->tile) {
                free(s->tile->data);
        }
        vf_free(s->frame);
        free(s);
}

static struct video_frame * vidcap_screen_osx_grab(void *state, struct audio_frame **audio)
{
        struct vidcap_screen_osx_state *s = (struct vidcap_screen_osx_state *) state;

        if (!s->initialized) {
                initialize(s);
                s->initialized = true;
        }

        *audio = NULL;

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

static const struct video_capture_info vidcap_screen_osx_info = {
        vidcap_screen_osx_probe,
        vidcap_screen_osx_init,
        vidcap_screen_osx_done,
        vidcap_screen_osx_grab,
};

REGISTER_MODULE(screen, &vidcap_screen_osx_info, LIBRARY_CLASS_VIDEO_CAPTURE, VIDEO_CAPTURE_ABI_VERSION);

