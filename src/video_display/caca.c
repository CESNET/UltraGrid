/**
 * @file   video_display/caca.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET, z. s. p. o.
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

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"

#include <caca.h>

#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_display.h"

#define MOD_NAME "[caca] "

struct state_caca {
        caca_canvas_t  *canvas;
        caca_display_t *display;
        caca_dither_t  *dither;
        int screen_w;
        int screen_h;

        struct video_frame *f;
};

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
        caca_free_dither(s->dither);
        caca_free_display(s->display);
        caca_free_canvas(s->canvas);
        vf_free(s->f);
        free(s);
}

static void *display_caca_init(struct module *parent, const char *fmt, unsigned int flags)
{
        UNUSED(flags);
        UNUSED(parent);
        if (strlen(fmt) > 0) {
                color_printf("Display " TBOLD("caca") " expects no parameters.\n");
                return strcmp(fmt, "help") == 0 ? INIT_NOERR : NULL;
        }
        struct state_caca *s = calloc(1, sizeof *s);
        s->canvas = caca_create_canvas(0, 0);
        if (!s->canvas) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create canvas\n");
                display_caca_done(s);
                return NULL;
        }
        s->display = caca_create_display(s->canvas);
        if (!s->display) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create display\n");
                display_caca_done(s);
                return NULL;
        }
        return s;
}

static struct video_frame *display_caca_getf(void *state)
{
        struct state_caca *s = state;
        return s->f;
}

static int display_caca_putf(void *state, struct video_frame *frame, long long timeout_ns)
{
        if (frame == NULL || timeout_ns == PUTF_DISCARD) {
                return 0;
        }

        struct state_caca *s = state;

        caca_dither_bitmap(s->canvas, 0, 0, s->screen_w, s->screen_h, s->dither, frame->tiles[0].data);
        caca_refresh_display(s->display);
        return 0;
}

static int display_caca_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);

        codec_t c[2] = { RGB, RGBA };
        if (property == DISPLAY_PROPERTY_CODECS && sizeof c <= *len) {
                memcpy(val, c, sizeof c);
                *len = sizeof c;
                return TRUE;
        }
        return FALSE;
}

static int display_caca_reconfigure(void *state, struct video_desc desc)
{
        struct state_caca *s = state;
        vf_free(s->f);
        s->f = vf_alloc_desc_data(desc);
        enum {
                RMASK = 0xff0000,
                GMASK = 0x00ff00,
                BMASK = 0x0000ff,
                AMASK = 0x000000,
        };
        caca_free_dither(s->dither);
        s->dither = caca_create_dither(8 * get_bpp(desc.color_spec),
                        desc.width, desc.height,
                        get_bpp(desc.color_spec) * desc.width,
                        RMASK, GMASK, BMASK, AMASK);
        if (!s->dither) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Failed to create dither\n");
                return FALSE;
        }
        s->screen_w = caca_get_canvas_width(s->canvas);
        s->screen_h = caca_get_canvas_height(s->canvas);

        return TRUE;
}

static const void *display_caca_info_get() {
        static _Thread_local struct video_display_info display_caca_info = {
                display_caca_probe,
                display_caca_init,
                NULL,
                display_caca_done,
                display_caca_getf,
                display_caca_putf,
                display_caca_reconfigure,
                display_caca_get_property,
                NULL,
                NULL,
                DISPLAY_DOESNT_NEED_MAINLOOP,
                DISPLAY_NO_GENERIC_FPS_INDICATOR,
        };
        const char *display = getenv("DISPLAY");
        if (display != NULL && strlen(display) > 0) { // print FPS only if caca has own window (not the terminal)
                display_caca_info.generic_fps_indicator_prefix = MOD_NAME;
        }
        return &display_caca_info;
};

REGISTER_MODULE_WITH_FUNC(caca, display_caca_info_get, LIBRARY_CLASS_VIDEO_DISPLAY, VIDEO_DISPLAY_ABI_VERSION);
