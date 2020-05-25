/**
 * @file   capture_filter/matrix.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020 CESNET, z. s. p. o.
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
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_codec.h"

#define MOD_NAME "[matrix cap. f.] "

struct state_capture_filter_matrix {
        double transform_matrix[9];
};

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (!cfg || strcmp(cfg, "help") == 0) {
                printf("Performs matrix transformation on input pixels.\n\n"
                       "usage:\n");
                color_out(COLOR_OUT_BOLD, "\t--capture-filter matrix:a:b:c:d:e:f:g:h:i\n");
                printf("where numbers a-i are members of 3x3 transformation matrix [a b c; d e f; g h i], decimals.\n"
                       "Coefficients are applied at unpacked pixels (eg. on Y Cb and Cr channels of UYVY). Result is marked as RGB.\n"
                       "Currently only RGB and UYVY is supported on input. No additional color transformation is performed.\n");
                return 1;
        }
        struct state_capture_filter_matrix *s = calloc(1, sizeof(struct state_capture_filter_matrix));
        char *cfg_c = strdup(cfg);
        char *item, *save_ptr, *tmp = cfg_c;
        int i = 0;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                s->transform_matrix[i++] = atof(item);
                tmp = NULL;
        }
        free(cfg_c);

        if (i != 9) {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Not enough numbers for transformation matrix - expected: 9, got: %d\n", i);
                free(s);
                return -1;
        }

        *state = s;
        return 0;
}

static void done(void *state)
{
        free(state);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
        struct state_capture_filter_matrix *s = state;
        struct video_desc desc = video_desc_from_frame(in);
        desc.color_spec = RGB;
        struct video_frame *out = vf_alloc_desc_data(desc);
        out->callbacks.dispose = vf_free;

        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

        if (in->color_spec == UYVY) {
                for (unsigned int i = 0; i < in->tiles[0].data_len; i += 4) {
                        unsigned char a[3], b[3];
                        a[1] = b[1] = *in_data++;
                        a[0] = *in_data++;
                        a[2] = b[2] = *in_data++;
                        b[0] = *in_data++;
                        *out_data++ = s->transform_matrix[0] * (a[0] - 16) +
                                s->transform_matrix[1] * (a[1] - 128) +
                                s->transform_matrix[2] * (a[2] - 128);
                        *out_data++ = s->transform_matrix[3] * (a[0] - 16) +
                                s->transform_matrix[4] * (a[1] - 128) +
                                s->transform_matrix[5] * (a[2] - 128);
                        *out_data++ = s->transform_matrix[6] * (a[0] - 16) +
                                s->transform_matrix[7] * (a[1] - 128) +
                                s->transform_matrix[8] * (a[2] - 128);
                        *out_data++ = s->transform_matrix[0] * (b[0] - 16) +
                                s->transform_matrix[1] * (b[1] - 128) +
                                s->transform_matrix[2] * (b[2] - 128);
                        *out_data++ = s->transform_matrix[3] * (b[0] - 16) +
                                s->transform_matrix[4] * (b[1] - 128) +
                                s->transform_matrix[5] * (b[2] - 128);
                        *out_data++ = s->transform_matrix[6] * (b[0] - 16) +
                                s->transform_matrix[7] * (b[1] - 128) +
                                s->transform_matrix[8] * (b[2] - 128);
                }
        } else if (in->color_spec == RGB) {
                for (unsigned int i = 0; i < in->tiles[0].data_len; i += 3) {
                        unsigned char a[3];
                        a[0] = *in_data++;
                        a[1] = *in_data++;
                        a[2] = *in_data++;
                        *out_data++ = s->transform_matrix[0] * a[0] +
                                s->transform_matrix[1] * a[1] +
                                s->transform_matrix[2] * a[2];
                        *out_data++ = s->transform_matrix[3] * a[0] +
                                s->transform_matrix[4] * a[1] +
                                s->transform_matrix[5] * a[2];
                        *out_data++ = s->transform_matrix[6] * a[0] +
                                s->transform_matrix[7] * a[1] +
                                s->transform_matrix[8] * a[2];
                }
        } else {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Only UYVY or RGB is currently supported!\n");
                VIDEO_FRAME_DISPOSE(in);
                vf_free(out);
                return NULL;
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static const struct capture_filter_info capture_filter_matrix = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(matrix, &capture_filter_matrix, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

/* vim: set expandtab sw=8: */
