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
#include "utils/macros.h"
#include "utils/parallel_conv.h"
#include "video.h"
#include "video_codec.h"
#include "vo_postprocess/capture_filter_wrapper.h"

#define MOD_NAME "[resize filter] "

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_resize {
    struct resize_param param;
    struct video_desc saved_desc;
    struct video_desc out_desc;
    char *vo_pp_out_buffer; ///< buffer to write to if we use vo_pp wrapper (otherwise unused)
    decoder_t decoder;
    struct video_frame *dec_frame;
};

static void usage() {
    printf("Scaling by scale factor:\n\n");
    printf("resize usage:\n");
    color_printf("\t" TBOLD(TRED("resize") ":numerator[/denominator][algo=<a>]") "\n");
    printf("or\n");
    printf("\t" TBOLD(TRED("resize") ":<width>x<height>[algo=<a>]") "\n\n");
    color_printf("Scaling examples:\n"
                 "\t" TBOLD("resize:1/2")
                 " - downscale input frame size by scale factor of 2\n"
                 "\t" TBOLD("resize:1280x720")
                 " - scales input to 1280x720\n"
                 "\t" TBOLD("resize:720x576")
                 " - scales input to PAL\n");
    color_printf(
        "\nOptions:\n"
        "\t" TBOLD(
            "algo") " - scaling algorithm to use (list with `algo:help`)\n");
    color_printf("\n");
}

static int
parse_fmt(char *cfg, struct resize_param *param)
{
    char *save_ptr = NULL;
    char *item     = NULL;
    while ((item = strtok_r(cfg, ":", &save_ptr))) {
        cfg = NULL;
        if (IS_KEY_PREFIX(item, "algo")) {
            param->algo = resize_algo_from_string(strchr(item, '=') + 1);
            if (param->algo < 0) {
                    return param->algo == RESIZE_ALGO_HELP_SHOWN ? 1 : -1;
            }
            continue;
        }
        if (!isdigit(item[0])) {
            log_msg(LOG_LEVEL_ERROR,
                    "[RESIZE ERROR] Unrecognized part of config "
                    "string: %s\n",
                    item);
            return -1;
        }
        if (strchr(item, 'x')) {
            param->mode          = USE_DIMENSIONS;
            param->target_width  = strtol(item, NULL, 10);
            errno                = 0;
            param->target_height = strtol(strchr(item, 'x') + 1, NULL, 10);
        } else {
            param->mode   = USE_FRACTION;
            param->factor = strtol(item, NULL, 10);
            if (strchr(item, '/')) {
                    param->factor /= strtol(strchr(item, '/') + 1, NULL, 10);
            }
        }
    }

    if (param->mode == USE_DIMENSIONS && param->target_width > 0 &&
        param->target_height > 0) {
        return 0;
    }
    if (param->mode == USE_FRACTION && param->factor > 0) {
        return 0;
    }

    MSG(ERROR, "No or incorrect resize size!\n");
    return -1;
}

static int init(struct module * parent, const char *cfg, void **state)
{
    UNUSED(parent);
    struct resize_param param = { .algo = RESIZE_ALGO_DFL };

    if(strcasecmp(cfg, "help") == 0) {
        usage();
        return 1;
    }

    char *fmt = strdup(cfg);
    const int rc = parse_fmt(fmt, &param);
    free(fmt);
    if (rc != 0) {
        return rc;
    }

    struct state_resize *s = calloc(1, sizeof(struct state_resize));
    s->param = param;

    *state = s;
    return 0;
}

static void
cleanup_common(struct state_resize *s)
{
    vf_free(s->dec_frame);
    s->dec_frame = NULL;
}

static void
done(void *state)
{
    cleanup_common((struct state_resize *) state);
    free(state);
}

static bool
reconfigure_if_needed(struct state_resize *s, const struct video_frame *in)
{
    if (video_desc_eq(video_desc_from_frame(in), s->saved_desc)) {
        return true;
    }
    struct video_desc dec_desc         = video_desc_from_frame(in);
    s->out_desc                        = video_desc_from_frame(in);
    const codec_t supp_in_codecs[] = { RESIZE_SUPPORTED_PIXFMT_INIT,
                                           VIDEO_CODEC_NONE };
    if (codec_is_in_set(in->color_spec, supp_in_codecs)) {
        dec_desc.color_spec = in->color_spec;
        s->decoder = vc_memcpy;
    } else {
        s->decoder = get_best_decoder_from(in->color_spec, supp_in_codecs,
                                           &dec_desc.color_spec);
        if (s->decoder == NULL) {
            MSG(ERROR,
                "Cannot decode %s to neither of supported input formats!\n",
                get_codec_name(in->color_spec));
            return false;
        }
    }
    s->out_desc.color_spec =
        get_bits_per_component(dec_desc.color_spec) == DEPTH8 ? RGB : RG48;
    MSG(INFO, "Decoding through %s to output pixfmt %s.\n",
        get_codec_name(dec_desc.color_spec),
        get_codec_name(s->out_desc.color_spec));

    if (s->param.mode == USE_DIMENSIONS) {
        s->out_desc.width  = s->param.target_width;
        s->out_desc.height = s->param.target_height;
    } else {
        s->out_desc.width = in->tiles[0].width * s->param.factor;
        s->out_desc.height = in->tiles[0].height * s->param.factor;
    }
    s->saved_desc = video_desc_from_frame(in);
    cleanup_common(s);
    if (s->decoder != vc_memcpy) {
        s->dec_frame               = vf_alloc_desc_data(dec_desc);
    }
    MSG(NOTICE, "resizing from %dx%d to %dx%d\n", s->saved_desc.width,
        s->saved_desc.height, s->out_desc.width, s->out_desc.height);
    return true;
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
    struct state_resize *s = state;

    if (!reconfigure_if_needed(s, in)) {
        VIDEO_FRAME_DISPOSE(in);
        return NULL;
    }

    struct video_frame *out_frame = vf_alloc_desc(s->out_desc);
    if (s->vo_pp_out_buffer) {
        out_frame->tiles[0].data = s->vo_pp_out_buffer;
    } else {
        out_frame->tiles[0].data = (char *) malloc(out_frame->tiles[0].data_len);
        out_frame->callbacks.data_deleter = vf_data_deleter;
    }

    for (unsigned int i = 0; i < out_frame->tile_count; i++) {
        if (s->decoder != vc_memcpy) {
            parallel_pix_conv(
                (int) in->tiles[i].height, s->dec_frame->tiles[i].data,
                vc_get_linesize(in->tiles[i].width, s->dec_frame->color_spec),
                in->tiles[i].data,
                vc_get_linesize(in->tiles[i].width, in->color_spec), s->decoder,
                0);
        }
        struct video_frame *const in_frame =
            s->decoder == vc_memcpy ? in : s->dec_frame;

        resize_frame(in_frame->tiles[i].data, in_frame->color_spec,
                     out_frame->tiles[i].data, in_frame->tiles[i].width,
                     in_frame->tiles[i].height, &s->param);
    }

    VIDEO_FRAME_DISPOSE(in);

    out_frame->callbacks.dispose = vf_free;

    return out_frame;
}

static void vo_pp_set_out_buffer(void *state, char *buffer)
{
        struct state_resize *s = state;
        s->vo_pp_out_buffer = buffer;
}

static const struct capture_filter_info capture_filter_resize = {
    init,
    done,
    filter,
};

REGISTER_MODULE(resize, &capture_filter_resize, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
// coverity[leaked_storage:SUPPRESS]
ADD_VO_PP_CAPTURE_FILTER_WRAPPER(resize, init, filter, done, vo_pp_set_out_buffer)

/* vim: set expandtab sw=4: */
