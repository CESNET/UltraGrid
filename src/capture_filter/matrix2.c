/**
 * @file   capture_filter/matrix.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2024 CESNET
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

#include <errno.h>                                  // for errno
#include <stdlib.h>                                 // for NULL, free, calloc
#include <string.h>                                 // for strcmp, memcpy

#include "capture_filter.h"                         // for CAPTURE_FILTER_AB...
#include "debug.h"                                  // for LOG_LEVEL_ERROR, MSG
#include "lib_common.h"                             // for REGISTER_MODULE
#include "types.h"                                  // for tile, video_frame
#include "utils/color_out.h"                        // for color_printf, TBOLD
#include "utils/macros.h"                           // for STR_LEN, snprintf_ch
#include "video_codec.h"                            // for vc_get_linesize
#include "video_frame.h"                            // for vf_alloc_desc
#include "vo_postprocess/capture_filter_wrapper.h"  // for ADD_VO_PP_CAPTURE...

struct module;

#define MOD_NAME "[matrix2 cap. f.] "

enum {
        MATRIX_VOL = 9,
};

static const double y601_y709_matrix[9] = {
        1, -0.11555, -0.207938,
        0,  1.01864,  0.114618,
        0,  0.075049, 1.025327,
};

struct state_capture_filter_matrix2 {
        double transform_matrix[MATRIX_VOL];
        void  *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper
                                 ///< (otherwise unused)
};

static void
usage(void)
{
        color_printf("Capture filter/postprocessor " TBOLD(
            "matrix2") " performs matrix transformation on input "
                       "pixels.\n\n"
                       "usage:\n");
        color_printf(TBOLD("\t-F/-p matrix2:a:b:c:d:e:f:g:h:i\n"));
        color_printf("or\n\t");
        color_printf(TBOLD("-F/-p matrix2:y601_to_y709") "\n");
        color_printf("\nwhere numbers a-i are members of 3x3 transformation "
                     "matrix [a b c; d e f; g h i], decimals.\n"
                     "Coefficients are applied to unpacked pixels (eg. on Y Cb "
                     "and Cr channels of UYVY).\n");
        color_printf("\nCurrently only " TBOLD("UYVY") " is supported.\n");
        color_printf(
            "\nSee also " TBOLD("matrix") " capture filter/postprocessor.\n");
        color_printf("\n");
}

static bool
parse_fmt(struct state_capture_filter_matrix2 *s, char *cfg)
{
        char *item     = NULL;
        char *save_ptr = NULL;
        char *tmp      = cfg;
        int   i        = 0;
        while ((item = strtok_r(tmp, ":", &save_ptr)) != NULL) {
                if (i == 0 && strcmp(item, "y601_to_y709") == 0) {
                        memcpy(s->transform_matrix, y601_y709_matrix,
                               sizeof y601_y709_matrix);
                        i = MATRIX_VOL;
                        break;
                }
                char *endptr             = NULL;
                errno                    = 0;
                s->transform_matrix[i++] = strtod(item, &endptr);
                if (errno != 0 || *endptr != '\0') {
                        MSG(WARNING, "Problem converting number %s\n", item);
                }
                tmp = NULL;
        }

        if (i != MATRIX_VOL) {
                MSG(ERROR,
                    "Not enough numbers for transformation matrix - expected: "
                    "9, got: %d\n",
                    i);
                return false;
        }
        return true;
}

static int
init(struct module *parent, const char *cfg, void **state)
{
        (void) parent;

        if (strlen(cfg) == 0 || strcmp(cfg, "help") == 0) {
                usage();
                return 1;
        }
        struct state_capture_filter_matrix2 *s = calloc(1, sizeof *s);
        char                                 cfg_c[STR_LEN];
        snprintf_ch(cfg_c, "%s", cfg);
        if (!parse_fmt(s, cfg_c)) {
                free(s);
                return -1;
        }

        *state = s;
        return 0;
}

static void
done(void *state)
{
        free(state);
}

static struct video_frame *
filter(void *state, struct video_frame *in)
{
        struct state_capture_filter_matrix2 *s = state;
        struct video_desc desc = video_desc_from_frame(in);
        struct video_frame *out = vf_alloc_desc(desc);
        if (s->vo_pp_out_buffer) {
                out->tiles[0].data = s->vo_pp_out_buffer;
        } else {
                out->tiles[0].data = malloc(out->tiles[0].data_len);
                out->callbacks.data_deleter = vf_data_deleter;
        }
        out->callbacks.dispose = vf_free;

        if (in->color_spec != UYVY) {
                MSG(ERROR, "Sorry, only UYVY supported by now.\n");
        }
        unsigned char *in_data  = (unsigned char *) in->tiles[0].data;
        unsigned char *out_data = (unsigned char *) out->tiles[0].data;

        for (unsigned int i = 0; i < in->tiles[0].data_len; i += 4) {
                double u  = *in_data++ - 128;
                double y1 = *in_data++ - 16;
                double v  = *in_data++ - 128;
                double y2 = *in_data++ - 16;
                double y  = (y1 + y2) / 2;
                // U
                *out_data++ = 128 + s->transform_matrix[3] * y +
                              s->transform_matrix[4] * u +
                              s->transform_matrix[5] * v;
                // Y
                *out_data++ = 16 + s->transform_matrix[0] * y1 +
                              s->transform_matrix[1] * u +
                              s->transform_matrix[2] * v;
                // V
                *out_data++ = 128 + s->transform_matrix[6] * y +
                              s->transform_matrix[7] * u +
                              s->transform_matrix[8] * v;
                // Y
                *out_data++ = 16 + s->transform_matrix[0] * y2 +
                              s->transform_matrix[1] * u +
                              s->transform_matrix[2] * v;
        }

        VIDEO_FRAME_DISPOSE(in);

        return out;
}

static void
vo_pp_set_out_buffer(void *state, char *buffer)
{
        struct state_capture_filter_matrix2 *s = state;
        s->vo_pp_out_buffer                    = buffer;
}

static const struct capture_filter_info capture_filter_matrix2 = {
        .init   = init,
        .done   = done,
        .filter = filter,
};

REGISTER_MODULE(matrix2, &capture_filter_matrix2, LIBRARY_CLASS_CAPTURE_FILTER,
                CAPTURE_FILTER_ABI_VERSION);
// coverity[leaked_storage:SUPPRESS]
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(matrix2, init, filter, done,
                                 vo_pp_set_out_buffer, NULL)
