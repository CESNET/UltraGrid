/**
 * @file   vo_postprocess/delay.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2024 CESNET z.s.p.o.
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

#include <assert.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "debug.h"
#include "lib_common.h"
#include "types.h"
#include "utils/color_out.h"
#include "utils/list.h"
#include "utils/macros.h"
#include "video_codec.h"
#include "video_display.h"
#include "video_frame.h"
#include "vo_postprocess.h"

#define MOD_NAME "[vpp/delay] "

struct state_delay {
        double                     delay_sec;
        int                        delay_frames;
        struct video_desc          desc;
        struct simple_linked_list *cached_frames;
};

static bool
delay_get_property(void *state, int property, void *val, size_t *len)
{
        (void) state;
        if (property != VO_PP_PROPERTY_CODECS) {
                return false;
        }
        const size_t codecs_len = (VC_END - VC_FIRST) * sizeof(codec_t);
        assert(codecs_len <= *len);
        *len            = codecs_len;
        codec_t *codecs = val;
        for (int i = VC_FIRST; i < VC_END; ++i) {
                codecs[i] = i;
        }
        return true;
}

static void
usage()
{
        color_printf(TBOLD("delay") " postprocessor settings:\n");
        color_printf(
            TBOLD(TRED("\t-p delay:[seconds=<s>|frames=<f>]")) " | " TBOLD(
                "-p delay:help") "\n");
        color_printf("\nSeconds can be given as a decimal number.\n");
        color_printf("\n");
}

static void *
delay_init(const char *config)
{
        enum {
                MAX_VAL_FRM = 2000,
                MAX_VAL_SEC = 60,
        };
        if (!IS_KEY_PREFIX(config, "seconds") &&
            !IS_KEY_PREFIX(config, "frames")) {
                usage();
                return NULL;
        }

        const char *val_s = strchr(config, '=') + 1;
        double      val   = strtod(val_s, NULL);
        if (val <= 0 || val > MAX_VAL_FRM ||
            (IS_KEY_PREFIX(config, "seconds") && val > MAX_VAL_SEC)) {
                MSG(ERROR, "Wrong delay value: %s\n", val_s);
                return NULL;
        }
        if (IS_KEY_PREFIX(config, "frames") && fabs(round(val) - val) > 0.0) {
                MSG(ERROR, "Number of frames should be an integer, given: %s\n",
                    val_s);
                return NULL;
        }

        struct state_delay *s = calloc(1, sizeof(struct state_delay));
        assert(s != NULL);
        if (IS_KEY_PREFIX(config, "seconds")) {
                s->delay_sec = val;
        } else {
                s->delay_frames = (int) val;
        }
        MSG(INFO, "Delay set to %g %s.\n", val,
            s->delay_sec ? "seconds" : "frames");
        s->cached_frames = simple_linked_list_init();

        return s;
}

static bool
delay_reconfigure(void *state, struct video_desc desc)
{
        struct state_delay *s = (struct state_delay *) state;
        s->desc               = desc;

        struct video_frame *f = NULL;
        while ((f = simple_linked_list_pop(s->cached_frames)) != NULL) {
                vf_free(f);
        }

        return true;
}

static struct video_frame *
delay_getf(void *state)
{
        struct state_delay *s = state;

        return vf_alloc_desc_data(s->desc);
}

static bool
delay_postprocess(void *state, struct video_frame *in, struct video_frame *out,
                  int req_pitch)
{
        struct state_delay *s = state;

        simple_linked_list_append(s->cached_frames, in);

        const int list_size = simple_linked_list_size(s->cached_frames);
        if (s->delay_frames >= 0 && list_size < s->delay_frames) {
                return false;
        }

        if (list_size / s->desc.fps < s->delay_sec) {
                return false;
        }

        struct video_frame *f = simple_linked_list_pop(s->cached_frames);
        const size_t        linesize =
            vc_get_linesize(f->tiles[0].width, f->color_spec);
        for (size_t i = 0; i < s->desc.height; ++i) {
                memcpy(out->tiles[0].data + i * req_pitch,
                       f->tiles[0].data + i * linesize, linesize);
        }
        vf_free(f);

        return true;
}

static void
delay_done(void *state)
{
        struct state_delay *s = state;

        struct video_frame *f = NULL;
        while ((f = simple_linked_list_pop(s->cached_frames)) != NULL) {
                vf_free(f);
        }
        simple_linked_list_destroy(s->cached_frames);
        free(s);
}

static void
delay_get_out_desc(void *state, struct video_desc *out, int *in_display_mode,
                   int *out_frames)
{
        struct state_delay *s = state;

        *out             = s->desc;
        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames      = 1;
}

static const struct vo_postprocess_info vo_pp_delay_info = {
        delay_init,         delay_reconfigure, delay_getf, delay_get_out_desc,
        delay_get_property, delay_postprocess, delay_done,
};

REGISTER_MODULE(delay, &vo_pp_delay_info, LIBRARY_CLASS_VIDEO_POSTPROCESS,
                VO_PP_ABI_VERSION);
