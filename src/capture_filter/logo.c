/**
 * @file   capture_filter/logo.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2025 CESNET
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
#include <stdbool.h>         // for false, bool, true
#include <stdio.h>           // for printf, fprintf, stderr
#include <stdlib.h>          // for free, NULL, malloc, atoi, calloc
#include <string.h>          // for strlen, strtok_r, strdup

#include "capture_filter.h"
#include "compat/strings.h"  // for strcasecmp
#include "debug.h"
#include "lib_common.h"
#include "pixfmt_conv.h"     // for get_decoder_from_to, decoder_t, vc_copyl...
#include "types.h"           // for video_frame, tile, RGB
#include "utils/macros.h"
#include "utils/pam.h"
#include "video_codec.h"
#include "video_frame.h"     // for VIDEO_FRAME_DISPOSE

struct module;

#define MOD_NAME "[logo] "

struct state_capture_filter_logo {
        unsigned char *logo;
        unsigned int width;
        unsigned int height;
        int x;
        int y;
};

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

static bool load_logo_data_from_file(struct state_capture_filter_logo *s, const char *filename) {
        if (strcasecmp(filename + (MAX(strlen(filename), 4) - 4), ".pam") == 0) {
                bool rgb;
                unsigned char *data;
                struct pam_metadata info;
                if (!pam_read(filename, &info, &data, malloc)) {
                        return false;
                }
                s->width = info.width;
                s->height = info.height;
                if (info.ch_count != 3 && info.ch_count != 4) {
                        log_msg(LOG_LEVEL_ERROR,
                                MOD_NAME
                                "Unsupported channel count %d in PAM file.\n",
                                info.ch_count);
                        free(data);
                        return false;
                }
                rgb = info.ch_count == 3;
                int datalen = info.ch_count * s->width * s->height;
                if (rgb) {
                        datalen = 4 * s->width * s->height;
                        unsigned char * tmp =  malloc(datalen);
                        vc_copylineRGBtoRGBA(tmp, data, datalen, 0, 8, 16);
                        s->logo = tmp;
                        free(data);
                } else {
                        s->logo = data;
                }
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Only logo in PAM format is currently supported.\n");
                return false;
        }

        return true;
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        struct state_capture_filter_logo *s = calloc(1, sizeof *s);

        s->x = s->y = -1;

        if (strlen(cfg) == 0 || strcasecmp(cfg, "help") == 0) {
                printf("Draws overlay logo over video:\n\n");
                printf("'logo' usage:\n");
                printf("\tlogo:<file>[:<x>[:<y>]]\n");
                printf("\t\t<file> - is path to logo to be added in PAM format with alpha\n");
                free(s);
                return 1;
        }
        char *tmp = strdup(cfg);
        char *save_ptr = NULL;
        char *item;
        if ((item = strtok_r(tmp, ":", &save_ptr)) == NULL) {
                fprintf(stderr, "File name with logo required!\n");
                goto error;
        }

        if (!load_logo_data_from_file(s, item)) {
                goto error;
        }

        if ((item = strtok_r(NULL, ":", &save_ptr))) {
                s->x = atoi(item);
                if ((item = strtok_r(NULL, ":", &save_ptr))) {
                        s->y = atoi(item);
                }
        }
        free(tmp);
        tmp = NULL;

        *state = s;
        return 0;
error:
        free(tmp);
        free(s->logo);
        free(s);
        return -1;
}

static void done(void *state)
{
        struct state_capture_filter_logo *s = (struct state_capture_filter_logo *)
                state;
        free(s->logo);
        free(s);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_capture_filter_logo *s = (struct state_capture_filter_logo *)
                state;
        decoder_t decoder, coder;
        decoder = get_decoder_from_to(in->color_spec, RGB);
        if (decoder == NULL) {
                MSG(ERROR, "Cannot find decoder from %s to RGB!\n",
                    get_codec_name(in->color_spec));
        }
        coder = get_decoder_from_to(RGB, in->color_spec);
        if (coder == NULL) {
                MSG(ERROR, "Cannot find encoder from %s to RGB!\n",
                    get_codec_name(in->color_spec));
        }
        if (decoder == NULL || coder == NULL)
                return in;

        int rect_x = s->x;
        int rect_y = s->y;

        if (rect_x < 0 || rect_x + s->width > in->tiles[0].width) {
                rect_x = in->tiles[0].width - s->width;
        }
        assert(get_pf_block_bytes(in->color_spec) > 0);
        rect_x = (rect_x / get_pf_block_bytes(in->color_spec)) * get_pf_block_bytes(in->color_spec);

        if (rect_y < 0 || rect_y + s->height > in->tiles[0].height) {
                rect_y = in->tiles[0].height - s->height;
        }

        if (rect_x < 0 || rect_y < 0)
                return in;

        int dec_width = s->width;
        dec_width = (dec_width  + 1) / get_pf_block_bytes(in->color_spec) * get_pf_block_bytes(in->color_spec);
        int linesize = dec_width * 3;

        unsigned char *segment = (unsigned char *) malloc(linesize * s->height);

        for (unsigned int y = 0; y < s->height; ++y) {
                decoder(segment + y * linesize, (unsigned char *) in->tiles[0].data + (y + rect_y) *
                                vc_get_linesize(in->tiles[0].width, in->color_spec) +
                                vc_get_linesize(rect_x, in->color_spec), linesize,
                                0, 8, 16);
        }

        const unsigned char *overlay_data = s->logo;
        for (unsigned int y = 0; y < s->height; ++y) {
                unsigned char *image_data = segment + y * linesize;
                for (unsigned int x = 0; x < s->width; ++x) {
                        int alpha = overlay_data[3];
                        for (int i = 0; i < 3; ++i) {
                                *image_data = (*image_data * (255 - alpha) + *overlay_data++ * alpha) / 255;
                                image_data++;
                        }
                        overlay_data++; // skip alpha
                }
        }

        for (unsigned int y = 0; y < s->height; ++y) {
                coder((unsigned char *) in->tiles[0].data + (rect_y + y) *
                                vc_get_linesize(in->tiles[0].width, in->color_spec) +
                                vc_get_linesize(rect_x, in->color_spec),
                                segment + y * linesize,
                                vc_get_linesize(s->width, in->color_spec), 0, 8, 16);
        }

        free(segment);

        return in;
}

static const struct capture_filter_info capture_filter_logo = {
        init,
        done,
        filter,
};

REGISTER_MODULE(logo, &capture_filter_logo, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

