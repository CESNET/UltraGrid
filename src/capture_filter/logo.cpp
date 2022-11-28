/**
 * @file   capture_filter/logo.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2013-2021 CESNET z.s.p.o.
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

#include <fstream>
#include <iostream>

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/macros.h"
#include "utils/pam.h"
#include "video.h"
#include "video_codec.h"

using namespace std;

struct state_capture_filter_logo {
        unsigned char *logo = NULL;
        unsigned int width{}, height{};
        int x{}, y{};
        ~state_capture_filter_logo() {
                free(logo);
        }
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
                if (info.depth != 3 && info.depth != 4) {
                        cerr << "Unsupported depth passed.";
                        free(data);
                        return false;
                }
                rgb = info.depth == 3;
                int datalen = info.depth * s->width * s->height;
                if (rgb) {
                        datalen = 4 * s->width * s->height;
                        auto tmp = (unsigned char *) malloc(datalen);
                        vc_copylineRGBtoRGBA(tmp, data, datalen, 0, 8, 16);
                        s->logo = tmp;
                        free(data);
                } else {
                        s->logo = data;
                }
        } else {
                cerr << "Only logo in PAM format is currently supported.";
                return false;
        }

        return true;
}

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        struct state_capture_filter_logo *s = new state_capture_filter_logo();

        s->x = s->y = -1;

        if (strlen(cfg) == 0 || strcasecmp(cfg, "help") == 0) {
                printf("Draws overlay logo over video:\n\n");
                printf("'logo' usage:\n");
                printf("\tlogo:<file>[:<x>[:<y>]]\n");
                printf("\t\t<file> - is path to logo to be added in PAM format with alpha\n");
                delete s;
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
        tmp = nullptr;

        *state = s;
        return 0;
error:
        free(tmp);
        delete s;
        return -1;
}

static void done(void *state)
{
        struct state_capture_filter_logo *s = (struct state_capture_filter_logo *)
                state;
        delete s;
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_capture_filter_logo *s = (struct state_capture_filter_logo *)
                state;
        decoder_t decoder, coder;
        decoder = get_decoder_from_to(in->color_spec, RGB);
        coder = get_decoder_from_to(RGB, in->color_spec);
        int rect_x = s->x;
        int rect_y = s->y;
        assert(coder != NULL && decoder != NULL);

        if (decoder == NULL || coder == NULL)
                return in;

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

