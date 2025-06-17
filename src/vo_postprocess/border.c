/**
 * @file   vo_postprocess/border.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2016-2025 CESNET
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

#include <assert.h>          // for assert
#include <stdbool.h>         // for bool, false, true
#include <stdint.h>          // for uint32_t, uint8_t
#include <stdio.h>           // for printf
#include <stdlib.h>          // for free, NULL, atoi, calloc, strtol, size_t
#include <string.h>          // for memcpy, strlen, strcmp, strdup, strtok_r

#include "compat/strings.h"  // for strncasecmp
#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "vo_postprocess.h"

struct state_border {
        struct video_desc saved_desc;
        uint8_t color[4];             ///< border color in RGBA
        unsigned int width;           ///< border width in pixels (must be even)
        unsigned int height;          ///< border height in pixels (must be even)
        struct video_frame *in;
};

static bool border_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state), UNUSED(property), UNUSED(val), UNUSED(len);
        return false;
}

static void * border_init(const char *config) {
        struct state_border *s = calloc(1, sizeof *s);
        memcpy(s->color, ((uint8_t []){ 0xff, 0xff, 0x00, 0xff }), sizeof s->color);
        s->width = 10;
        s->height = 10;

        if (strcmp(config, "help") == 0) {
                printf("border video postprocess takes optional parameters: color to be the border drawn with and border width. Example:\n");
                printf("\t-p border[:color=rrggbb][:width=<x>][:height=<y>]\n");
                printf("\n");
                free(s);
                return NULL;
        }
        char *tmp = strdup(config);
        char *config_copy = tmp;
        char *item, *save_ptr;
        while ((item = strtok_r(config_copy, ":", &save_ptr))) {
                if (strncasecmp(item, "color=", strlen("color=")) == 0) {
                        const char *color = item + strlen("color=");
                        if (color[0] == '#') {
                                color += 1; //skip #
                        }
                        if (strlen(color) == 6) {
                                char color_str[3] = "";
                                color += 1; //skip #

                                for (int i = 0; i < 3; ++i) {
                                        color_str[0] = color[0];
                                        color_str[1] = color[1];

                                        s->color[i] = strtol(color_str, NULL, 16);

                                        color += 2;
                                }
                        } else {
                                log_msg(LOG_LEVEL_ERROR, "Wrong color format!\n");
                                free(tmp);
                                free(s);
                                return NULL;
                        }
                } else if (strncasecmp(item, "width=", strlen("width=")) == 0) {
                        s->width = atoi(item + strlen("width="));
                        s->width = (s->width + 1) / 2 * 2;
                } else if (strncasecmp(item, "height=", strlen("height=")) == 0) {
                        s->height = atoi(item + strlen("height="));
                        s->height = (s->height + 1) / 2 * 2;
                } else {
                        log_msg(LOG_LEVEL_ERROR, "Wrong config!\n");
                        free(tmp);
                        free(s);
                        return NULL;
                }

                config_copy = NULL;
        }

        free(tmp);

        return s;
}

static bool
border_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_border *s = (struct state_border *) state;
        s->saved_desc = desc;
        vf_free(s->in);
        s->in = vf_alloc_desc_data(s->saved_desc);
        return true;
}

static struct video_frame * border_getf(void *state)
{
        struct state_border *s = (struct state_border *) state;
        return s->in;
}

static bool border_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        assert(req_pitch == vc_get_linesize(in->tiles[0].width, in->color_spec));
        assert(in->tile_count == 1);

        struct state_border *s = (struct state_border *) state;

        memcpy(out->tiles[0].data + s->height * req_pitch, in->tiles[0].data + s->height * req_pitch, in->tiles[0].data_len - 2 * s->height * req_pitch);

        if (in->color_spec == UYVY) {
                uint32_t rgba[2] = {0, 0};
                uint32_t uyvy = 0;
                memcpy(&rgba[0], s->color, 4);
                memcpy(&rgba[1], s->color, 4);
                decoder_t vc_copylineRGBAtoUYVY = get_decoder_from_to(RGBA, UYVY);
                vc_copylineRGBAtoUYVY((unsigned char *) &uyvy, (unsigned char *) rgba, 4, 0, 0, 0);
                // up and down
                for (unsigned int i = 0; i < s->height; ++i) {
                        char *line1 = out->tiles[0].data + i * req_pitch;
                        char *line2 = out->tiles[0].data + (out->tiles[0].height - 1 - i) * req_pitch;
                        for (unsigned int x = 0; x < out->tiles[0].width; x += 2) {
                                memcpy(line1 + x * 2, &uyvy, 4);
                                memcpy(line2 + x * 2, &uyvy, 4);
                        }
                }
                // sides
                for (unsigned int i = 0; i < s->width; ++i) {
                        if (i % 2 == 0) {
                                for (unsigned int y = 0; y < out->tiles[0].height; y += 1) {
                                        char *line = out->tiles[0].data + y * req_pitch;
                                        char *line_end = out->tiles[0].data + y * req_pitch +
                                                vc_get_linesize(out->tiles[0].width, UYVY) - 4;
                                        memcpy(line + i / 2 * 4, &uyvy, 4);
                                        memcpy(line_end - i / 2 * 4, &uyvy, 4);
                                }
                        }
                }
        } else if (in->color_spec == RGB || in->color_spec == RGBA) {
                int bpp = get_bpp(in->color_spec);
                for (unsigned int i = 0; i < s->height; ++i) {
                        char *line1 = out->tiles[0].data + i * req_pitch;
                        char *line2 = out->tiles[0].data + (out->tiles[0].height - 1 - i) * req_pitch;
                        for (unsigned int x = 0; x < out->tiles[0].width; x += 1) {
                                memcpy(line1 + x * bpp, s->color, bpp);
                                memcpy(line2 + x * bpp, s->color, bpp);
                        }
                }

                // sides
                for (unsigned int i = 0; i < s->width; ++i) {
                        for (unsigned int y = 0; y < out->tiles[0].height; y += 1) {
                                char *line = out->tiles[0].data + y * req_pitch;
                                char *line_end = out->tiles[0].data + y * req_pitch +
                                        (out->tiles[0].width - 1) * bpp;
                                memcpy(line + i * bpp, s->color, bpp);
                                memcpy(line_end - i * bpp, s->color, bpp);
                        }
                }
        } else {
                log_msg(LOG_LEVEL_WARNING, "Unsupported pixel format!\n");
                return false;
        }

        return true;
}

static void border_done(void *state)
{
        struct state_border *s = (struct state_border *) state;

        vf_free(s->in);

        free(s);
}

static void border_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_border *s = (struct state_border *) state;

        *out = s->saved_desc;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}

static const struct vo_postprocess_info vo_pp_border_info = {
        border_init,
        border_postprocess_reconfigure,
        border_getf,
        border_get_out_desc,
        border_get_property,
        border_postprocess,
        border_done,
};

REGISTER_MODULE(border, &vo_pp_border_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);

