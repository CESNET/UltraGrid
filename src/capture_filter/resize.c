/**
 * @file    capture_filter/resize.c
 * @author  Gerard Castillo     <gerard.castillo@i2cat.net>
 * @author  Marc Palau          <marc.palau@i2cat.net>
 * @author  Martin Pulec        <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014      Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2014-2023 CESNET, z. s. p. o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by the Fundació i2CAT,
 *      Internet I Innovació Digital a Catalunya. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
 *
 */

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include "capture_filter.h"
#include "capture_filter/resize_utils.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_codec.h"
#include "vo_postprocess/capture_filter_wrapper.h"

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct resize_param {
    enum resize_mode {
        NONE,
        USE_FRACTION,
        USE_DIMENSIONS,
    } mode;
    union {
        struct {
            int num;
            int denom;
        };
        struct {
            int target_width;
            int target_height;
        };
    };
    bool force_interlaced, force_progressive;
};

struct state_resize {
    struct resize_param param;
    struct video_desc saved_desc;
    struct video_desc out_desc;
    char *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper (otherwise unused)
};

static void usage() {
    printf("Scaling by scale factor:\n\n");
    printf("resize usage:\n");
    color_printf("\t" TBOLD(TRED("resize") ":numerator[/denominator]") "\n");
    printf("or\n");
    printf("\t" TBOLD(TRED("resize") ":<width>x<height>") "\n\n");
    color_printf("Scaling examples:\n"
                 "\t" TBOLD("resize:1/2")
                 " - downscale input frame size by scale factor of 2\n"
                 "\t" TBOLD("resize:1280x720")
                 " - scales input to 1280x720\n"
                 "\t" TBOLD("resize:720x576i")
                 " - scales input to PAL (overrides interlacing setting)\n");
}

static int init(struct module * parent, const char *cfg, void **state)
{
    UNUSED(parent);
    struct resize_param param = { 0 };

    if (strlen(cfg) == 0) {
        log_msg(LOG_LEVEL_ERROR, "[RESIZE ERROR] No configuration!\n");
        usage();
        return -1;
    }

    if(strcasecmp(cfg, "help") == 0) {
        usage();
        return 1;
    }
    char *endptr;
    if (strchr(cfg, 'x')) {
        param.mode = USE_DIMENSIONS;
        param.target_width = strtol(cfg, &endptr, 10);
        errno = 0;
        param.target_height = strtol(strchr(cfg, 'x') + 1, &endptr, 10);
        if (errno != 0) {
            perror("strtol");
            usage();
            return -1;
        }
    } else {
        param.mode = USE_FRACTION;
        param.num = strtol(cfg, &endptr, 10);
        if(strchr(cfg, '/')) {
            param.denom = strtol(strchr(cfg, '/') + 1, &endptr, 10);
        } else {
            param.denom = 1;
        }
    }

    if (*endptr == 'i' || *endptr == 'p') {
        if (*endptr == 'i') {
            param.force_interlaced = true;
        } else {
            param.force_progressive = true;
        }
        endptr += 1;
    }

    if (*endptr != '\0') {
        log_msg(LOG_LEVEL_ERROR, "[RESIZE ERROR] Unrecognized part of config string: %s\n", endptr);
        usage();
        return -1;
    }

    // check validity of options
    switch (param.mode) {
    case USE_FRACTION:
        if (param.num <= 0 || param.denom <= 0) {
            log_msg(LOG_LEVEL_ERROR, "\n[RESIZE ERROR] resize factors must be greater than zero!\n");
            usage();
            return -1;
        }
        break;
    case USE_DIMENSIONS:
        if (param.target_width <= 0 || param.target_height <= 0) {
            log_msg(LOG_LEVEL_ERROR, "\n[RESIZE ERROR] Targed widht and height must be greater than zero!\n");
            usage();
            return -1;
        }
        break;
    default:
        usage();
        return -1;
    }

    struct state_resize *s = calloc(1, sizeof(struct state_resize));
    s->param = param;

    *state = s;
    return 0;
}

static void done(void *state)
{
    free(state);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
    struct state_resize *s = state;

    if (!video_desc_eq(video_desc_from_frame(in), s->saved_desc)) {
    	struct video_desc desc = video_desc_from_frame(in);
        s->saved_desc = desc;
        if (s->param.mode == USE_DIMENSIONS) {
            desc.width = s->param.target_width;
            desc.height = s->param.target_height;
        } else {
            desc.width = in->tiles[0].width * s->param.num / s->param.denom;
            desc.height = in->tiles[0].height * s->param.num / s->param.denom;
        }
        desc.color_spec = RGB;
        if (s->param.force_interlaced) {
                desc.interlacing = INTERLACED_MERGED;
        } else if (s->param.force_progressive) {
                desc.interlacing = PROGRESSIVE;
        }
        s->out_desc = desc;
        printf("[resize filter] resizing from %dx%d to %dx%d\n", s->saved_desc.width, s->saved_desc.height, s->out_desc.width, s->out_desc.height);
    }

    struct video_frame *frame = vf_alloc_desc(s->out_desc);
    if (s->vo_pp_out_buffer) {
        frame->tiles[0].data = s->vo_pp_out_buffer;
    } else {
        frame->tiles[0].data = (char *) malloc(frame->tiles[0].data_len);
        frame->callbacks.data_deleter = vf_data_deleter;
    }

    for (unsigned int i = 0; i < frame->tile_count; i++) {
        if (s->param.mode == USE_DIMENSIONS) {
                resize_frame(in->tiles[i].data, in->color_spec,
                             frame->tiles[i].data, in->tiles[i].width,
                             in->tiles[i].height, s->param.target_width,
                             s->param.target_height);
        } else {
                resize_frame_factor(in->tiles[i].data, in->color_spec,
                                    frame->tiles[i].data, in->tiles[i].width,
                                    in->tiles[i].height,
                                    (double) s->param.num / s->param.denom);
        }
    }

    VIDEO_FRAME_DISPOSE(in);

    frame->callbacks.dispose = vf_free;

    return frame;
}

static void vo_pp_set_out_buffer(void *state, char *buffer)
{
        struct state_resize *s = state;
        s->vo_pp_out_buffer = buffer;
}

static struct capture_filter_info capture_filter_resize = {
    init,
    done,
    filter,
};

REGISTER_MODULE(resize, &capture_filter_resize, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
// coverity[leaked_storage:SUPPRESS]
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(resize, init, filter, done, vo_pp_set_out_buffer)

/* vim: set expandtab sw=4: */
