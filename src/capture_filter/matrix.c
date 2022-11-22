/**
 * @file   capture_filter/matrix.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020-2021 CESNET, z. s. p. o.
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
#include "utils/macros.h"
#include "video.h"
#include "video_codec.h"
#include "vo_postprocess/capture_filter_wrapper.h"

#define MOD_NAME "[matrix cap. f.] "

struct state_capture_filter_matrix {
        double transform_matrix[9];
        bool check_bounds;
        void *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper (otherwise unused)
};

static int init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);

        if (strlen(cfg) == 0 || strcmp(cfg, "help") == 0) {
                printf("Performs matrix transformation on input pixels.\n\n"
                       "usage:\n");
                color_printf(TERM_BOLD "\t--capture-filter matrix:a:b:c:d:e:f:g:h:i[:no-bounds-check]\n" TERM_RESET);
                printf("where numbers a-i are members of 3x3 transformation matrix [a b c; d e f; g h i], decimals.\n"
                       "Coefficients are applied at unpacked pixels (eg. on Y Cb and Cr channels of UYVY). Result is marked as RGB.\n"
                       "Currently only RGB and UYVY is supported on input. No additional color transformation is performed.\n");
                printf("\nOptional \"no-bounds-check\" options disables check for overflows/underflows which improves performance\n"
                                "but may give incorrect results if operation oveflows or underflows.\n");
                return 1;
        }
        struct state_capture_filter_matrix *s = calloc(1, sizeof(struct state_capture_filter_matrix));
        s->check_bounds = true;
        char *cfg_c = strdup(cfg);
        char *item = NULL;
        char *save_ptr = NULL;
        char *tmp = cfg_c;
        int i = 0;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                if (i == 9) {
                        if (strcmp(item, "no-bound-check") == 0) {
                                s->check_bounds = false;
                        } else {
                                log_msg(LOG_LEVEL_WARNING, MOD_NAME "Excess initializer given: %s\n", item);
                        }
                        break;
                }
                char *endptr = NULL;
                errno = 0;
                s->transform_matrix[i++] = strtod(item, &endptr);
                if (errno != 0 || *endptr != '\0') {
                        log_msg(LOG_LEVEL_WARNING, MOD_NAME "Problem converting number %s\n", item);
                }
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
        if (in->color_spec == UYVY) {
                desc.color_spec = RGB;
        }
        struct video_frame *out = vf_alloc_desc(desc);
        if (s->vo_pp_out_buffer) {
                out->tiles[0].data = s->vo_pp_out_buffer;
        } else {
                out->tiles[0].data = malloc(out->tiles[0].data_len);
                out->callbacks.data_deleter = vf_data_deleter;
        }
        out->callbacks.dispose = vf_free;

        if (s->check_bounds) {
                if (in->color_spec == UYVY) {
                        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
                        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

                        for (unsigned int i = 0; i < in->tiles[0].data_len; i += 4) {
                                double a[3];
                                double b[3];
                                a[1] = b[1] = *in_data++;
                                a[0] = *in_data++;
                                a[2] = b[2] = *in_data++;
                                b[0] = *in_data++;
                                int val = s->transform_matrix[0] * (a[0] - 16) +
                                                        s->transform_matrix[1] * (a[1] - 128) +
                                                        s->transform_matrix[2] * (a[2] - 128);
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[3] * (a[0] - 16) +
                                                        s->transform_matrix[4] * (a[1] - 128) +
                                                        s->transform_matrix[5] * (a[2] - 128);
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[6] * (a[0] - 16) +
                                                        s->transform_matrix[7] * (a[1] - 128) +
                                                        s->transform_matrix[8] * (a[2] - 128);
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[0] * (b[0] - 16) +
                                                        s->transform_matrix[1] * (b[1] - 128) +
                                                        s->transform_matrix[2] * (b[2] - 128);
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[3] * (b[0] - 16) +
                                                        s->transform_matrix[4] * (b[1] - 128) +
                                                        s->transform_matrix[5] * (b[2] - 128);
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[6] * (b[0] - 16) +
                                                        s->transform_matrix[7] * (b[1] - 128) +
                                                        s->transform_matrix[8] * (b[2] - 128),
                                *out_data++ = CLAMP(val, 0, 255);
                        }
                } else if (in->color_spec == RGB) {
                        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
                        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

                        for (unsigned int i = 0; i < in->tiles[0].data_len; i += 3) {
                                double a[3];
                                a[0] = *in_data++;
                                a[1] = *in_data++;
                                a[2] = *in_data++;
                                int val = s->transform_matrix[0] * a[0] +
                                                        s->transform_matrix[1] * a[1] +
                                                        s->transform_matrix[2] * a[2];
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[3] * a[0] +
                                                        s->transform_matrix[4] * a[1] +
                                                        s->transform_matrix[5] * a[2];
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[6] * a[0] +
                                                        s->transform_matrix[7] * a[1] +
                                                        s->transform_matrix[8] * a[2];
                                *out_data++ = CLAMP(val, 0, 255);
                        }
                } else if (in->color_spec == RG48) {
                        uint16_t *in_data = (uint16_t *)(void *) in->tiles[0].data;
                        uint16_t *out_data = (uint16_t *)(void *) out->tiles[0].data;

                        for (unsigned int i = 0; i < in->tiles[0].data_len; i += 6) {
                                double a[3];
                                a[0] = *in_data++;
                                a[1] = *in_data++;
                                a[2] = *in_data++;
                                int val = s->transform_matrix[0] * a[0] +
                                                        s->transform_matrix[1] * a[1] +
                                                        s->transform_matrix[2] * a[2];
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[3] * a[0] +
                                                        s->transform_matrix[4] * a[1] +
                                                        s->transform_matrix[5] * a[2];
                                *out_data++ = CLAMP(val, 0, 255);
                                val = s->transform_matrix[6] * a[0] +
                                                        s->transform_matrix[7] * a[1] +
                                                        s->transform_matrix[8] * a[2];
                                *out_data++ = CLAMP(val, 0, 255);
                        }
                } else {
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Only UYVY, RGB or RG48 is currently supported!\n");
                        VIDEO_FRAME_DISPOSE(in);
                        vf_free(out);
                        return NULL;
                }
        } else {
                if (in->color_spec == UYVY) {
                        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
                        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

                        for (unsigned int i = 0; i < in->tiles[0].data_len; i += 4) {
                                double a[3];
                                double b[3];
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
                        unsigned char *in_data = (unsigned char *) in->tiles[0].data;
                        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

                        for (unsigned int i = 0; i < in->tiles[0].data_len; i += 3) {
                                double a[3];
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
                } else if (in->color_spec == RG48) {
                        uint16_t *in_data = (uint16_t *)(void *) in->tiles[0].data;
                        uint16_t *out_data = (uint16_t *)(void *) out->tiles[0].data;

                        for (unsigned int i = 0; i < in->tiles[0].data_len; i += 6) {
                                double a[3];
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
                        log_msg(LOG_LEVEL_ERROR, MOD_NAME "Only UYVY, RGB or RG48 is currently supported!\n");
                        VIDEO_FRAME_DISPOSE(in);
                        vf_free(out);
                        return NULL;
                }
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static void vo_pp_set_out_buffer(void *state, char *buffer)
{
        struct state_capture_filter_matrix *s = state;
        s->vo_pp_out_buffer = buffer;
}

static const struct capture_filter_info capture_filter_matrix = {
        .init = init,
        .done = done,
        .filter = filter,
};

REGISTER_MODULE(matrix, &capture_filter_matrix, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(matrix, init, filter, done, vo_pp_set_out_buffer)


/* vim: set expandtab sw=8: */
