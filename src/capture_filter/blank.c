/**
 * @file   capture_filter/blank.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2023 CESNET, z. s. p. o.
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

#include "capture_filter.h"
#include "libavcodec/lavc_common.h"
#include "lib_common.h"
#include "messaging.h"
#include "module.h"

#include "video.h"
#include "video_codec.h"

#include <libswscale/swscale.h>

#define FACTOR 6

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_blank {
        struct module mod;

        int x, y, width, height;
        double x_relative, y_relative, width_relative, height_relative;

        struct video_desc saved_desc;

        bool in_relative_units;
        bool black;

        struct SwsContext *ctx_downscale,
                          *ctx_upscale;
};

static bool parse(struct state_blank *s, char *cfg)
{
        int vals[4];
        double vals_relative[4];
        unsigned int counter = 0;
        bool black = false;

        memset(&s->saved_desc, 0, sizeof(s->saved_desc));

        if (strchr(cfg, '%')) {
                s->in_relative_units = true;
        } else {
                s->in_relative_units = false;
        }

        char *item, *save_ptr = NULL;
        while ((item = strtok_r(cfg, ":", &save_ptr))) {
                if (s->in_relative_units) {
                        vals_relative[counter] = atof(item) / 100.0;
                        if (vals_relative[counter] < 0.0)
                                vals_relative[counter] = 0.0;
                } else {
                        vals[counter] = atoi(item);
                        if (vals[counter] < 0)
                                vals[counter] = 0;
                }

                cfg = NULL;
                counter += 1;
                if (counter == sizeof(vals) / sizeof(int))
                        break;
        }
        while ((item = strtok_r(cfg, ":", &save_ptr))) {
                if (strcmp(item, "black") == 0) {
                        black = true;
                } else {
                        fprintf(stderr, "[Blank] Unknown config value: %s\n",
                                        item);
                        return false;
                }
        }

        if(counter != sizeof(vals) / sizeof(int)) {
                fprintf(stderr, "[Blank] Few config values.\n");
                return false;
        }

        if (s->in_relative_units) {
                s->x_relative = vals_relative[0];
                s->y_relative = vals_relative[1];
                s->width_relative = vals_relative[2];
                s->height_relative = vals_relative[3];
        } else {
                s->x = vals[0];
                s->y = vals[1];
                s->width = (vals[2] + FACTOR - 1) / FACTOR * FACTOR;
                s->height = vals[3];
        }
        s->black = black;

        return true;
}

static int init(struct module *parent, const char *cfg, void **state)
{
        if (cfg && strcasecmp(cfg, "help") == 0) {
                printf("Blanks specified rectangular area:\n\n");
                printf("blank usage:\n");
                printf("\tblank:x:y:widht:height[:black]\n");
                printf("\t\tor\n");
                printf("\tblank:x%%:y%%:widht%%:height%%[:black]\n");
                printf("\t(all values in pixels)\n");
                return 1;
        }

        struct state_blank *s = calloc(1, sizeof *s);
        assert(s);

        if (cfg) {
                char *tmp = strdup(cfg);
                bool ret = parse(s, tmp);
                free(tmp);
                if (!ret) {
                        free(s);
                        return -1;
                }
        }

        module_init_default(&s->mod);
        s->mod.cls = MODULE_CLASS_DATA;
        module_register(&s->mod, parent);

        *state = s;
        return 0;
}

static void done(void *state)
{
        struct state_blank *s = state;
        module_done(&s->mod);

        sws_freeContext(s->ctx_downscale);
        sws_freeContext(s->ctx_upscale);
        free(s);
}

static struct response * process_message(struct state_blank *s, struct msg_universal *msg)
{
        if (parse(s, msg->text)) {
                return new_response(RESPONSE_OK, NULL);
        } else {
                return new_response(RESPONSE_BAD_REQUEST, NULL);
        }
}

/**
 * @note v210 etc. will be green
 */
static struct video_frame *filter(void *state, struct video_frame *in)
{
        assert(in->tile_count == 1);

        struct state_blank *s = state;
        codec_t codec = in->color_spec;
        int bpp = get_bpp(codec);
        enum AVPixelFormat av_pixfmt = AV_PIX_FMT_NONE;

        if (get_ug_to_av_pixfmt(codec) != AV_PIX_FMT_NONE) {
                av_pixfmt = get_ug_to_av_pixfmt(codec);
        }

        if (s->in_relative_units) {
                s->x = s->x_relative * in->tiles[0].width;
                s->y = s->y_relative * in->tiles[0].height;
                s->width = s->width_relative * in->tiles[0].width;
                s->width = (s->width + FACTOR - 1) / FACTOR * FACTOR;
                s->height = s->height_relative * in->tiles[0].height;
        }

        int width = s->width;
        int height = s->height;
        int x = s->x;
        int y = s->y;

        x = (x + bpp - 1) / bpp * bpp;

        if (y + height > (int) in->tiles[0].height) {
                height = in->tiles[0].height - y;
        }

        if (x + width > (int) in->tiles[0].width) {
                width = in->tiles[0].width - x;
                width = width / FACTOR * FACTOR;
        }

        // don't know why but when not done, it leaves black or grey border on the right side
        // for BGR
        if (codec == BGR) {
                width = width / (FACTOR * 3) * (FACTOR * 3);
        }

        if (!video_desc_eq(s->saved_desc,
                                video_desc_from_frame(in))) {

                if (av_pixfmt == AV_PIX_FMT_NONE || !sws_isSupportedInput(av_pixfmt) ||
                                !sws_isSupportedOutput(av_pixfmt)) {
                        fprintf(stderr, "Unable to find suitable pixfmt!\n");
                        return in;
                }

                sws_freeContext(s->ctx_downscale);
                sws_freeContext(s->ctx_upscale);

                s->ctx_downscale = sws_getContext(width, height, av_pixfmt,
                                width / FACTOR, height / FACTOR, av_pixfmt, SWS_FAST_BILINEAR,0,0,0);
                s->ctx_upscale = sws_getContext(width / FACTOR, height / FACTOR, av_pixfmt,
                                width, height, av_pixfmt, SWS_FAST_BILINEAR,0,0,0);

                if (s->ctx_downscale == NULL || s->ctx_upscale == NULL) {
                        fprintf(stderr, "Unable to initialize scaling context!");
                        return in;
                }

                s->saved_desc = video_desc_from_frame(in);
        }

        struct message *msg;
        while ((msg = check_message(&s->mod))) {
                struct response *r = process_message(s, (struct msg_universal *) msg);
                free_message(msg, r);
        }

        if (width <= 0 || height <= 0)
                return in;

        int orig_stride = vc_get_linesize(in->tiles[0].width, in->color_spec);
        char *orig = in->tiles[0].data + x * bpp + y * orig_stride;
        int tmp_stride = vc_get_linesize(width / FACTOR, in->color_spec);
        size_t tmp_len = tmp_stride * (height / FACTOR);
        uint8_t *tmp = (uint8_t *) malloc(tmp_len);
        if (s->black) {
                if (codec == UYVY) {
                        unsigned char pattern[] = { 127, 0 };

                        for (size_t i = 0; i < tmp_len; i += get_bpp(codec)) {
                                memcpy(tmp + i, pattern, get_bpp(codec));
                        }
                } else {
                        memset(tmp, 0, tmp_len);
                }
        } else {
                sws_scale(s->ctx_downscale, (const uint8_t * const *) &orig, &orig_stride, 0, height, &tmp, &tmp_stride);
        }
        sws_scale(s->ctx_upscale, (const uint8_t * const *) &tmp, &tmp_stride, 0, height / FACTOR, (uint8_t **) &orig, &orig_stride);

        free(tmp);

        return in;
}

static const struct capture_filter_info capture_filter_blank = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(blank, &capture_filter_blank, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

