// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob
/**
 * @file
 * similar to @ref src/capture_filter/add_frame.c but different implementation
 * (here, the API differences are substantial for the either vo_pp->vcf or
 * vcf->vo_pp wrapper to be used)
 */

#include <assert.h>           // for assert, static_assert
#include <stdint.h>           // for uint32_t
#include <stdlib.h>           // for free, atoi, calloc
#include <string.h>           // for memcpy, strchr, strcmp, strdup, strtok_r
#include <time.h>             // for nanosleep, timespec

#include "compat/c23.h"       // IWYU pragma: keep
#include "debug.h"            // for LOG_LEVEL_ERROR, MSG
#include "host.h"             // for INIT_NOERR
#include "lib_common.h"       // for REGISTER_MODULE, library_class
#include "tv.h"               // for time_ns_t, SEC_TO_NS, get_time_in_ns
#include "types.h"            // for tile, video_frame, video_desc
#include "utils/color_out.h"  // for color_printf, TBOLD
#include "utils/macros.h"     // for to_fourcc, IS_KEY_PREFIX, IS_PREFIX
#include "utils/text.h"       // for color_printf_wrapped
#include "video_display.h"    // for display_prop_vid_mode
#include "video_frame.h"      // for vf_free, vf_alloc_desc_data, video_desc...
#include "vo_postprocess.h"   // for VO_PP_ABI_VERSION, VO_PP_ABI_POSTPROCES...

#define MAGIC    to_fourcc('V', 'P', 'a', 'f')
#define MOD_NAME "[vpp/add_frame] "

static_assert(VO_PP_ABI_VERSION == VO_PP_ABI_POSTPROCESS_NULLPTR);

struct state_vo_pp_add_frame {
        uint32_t            magic;
        bool                nodelay;
        int                 add_frame_cnt; ///< +1 added frame
        int                 curr_idx;
        struct video_frame *f;
        time_ns_t           t0;
};

static void add_frame_done(void *state);

static bool
add_frame_get_property(void * /* state */, int /* property */, void * /* val */,
                       size_t * /* len */)
{
        return false;
}

static void
usage()
{
        color_printf("video postprocessor " TBOLD("add_frame")
                     " adds 1 extra frame per every amount of "
                     "frames.\n\n");

        color_printf("Typical use case is to convert 50p to 60p.\n\n");

        color_printf("Usage:\n"
                     "\t" TBOLD("-p add_frame:every=<num>[:nodelay]")
                     "\n\n");

        color_printf(
            "(in the proposed 50p->60p case, the <num> will be 5)\n\n");

        color_printf(
            "Example converting 50i->60p (notice `nodelay` for DF):\n");
        color_printf("\t " TBOLD("uv -p double_framerate:nodelay,add_frame:e=5")
                     " -t testcard:fps=50i -d gl\n");
        color_printf("or simply for 50p->60p:\n");
        color_printf("\t " TBOLD("uv -p add_frame:e=5")
                     " -d gl\n");
        color_printf("\n");

        color_printf_wrapped(
            "See also the capture analogous filter " TBOLD("add_frame")
            " (same use case but on the sencer) and postprocessor " TBOLD(
                    "every")
            " for frame dropping (eg. to achieve 60p->50p conversion).\n\n");
}

static bool
parse_fmt(struct state_vo_pp_add_frame *s, char *fmt)
{
        char *tok   = nullptr;
        char *state = nullptr;
        while ((tok = strtok_r(fmt, ":", &state)) != nullptr) {
                fmt = nullptr;
                if (IS_KEY_PREFIX(tok, "every")) {
                        s->add_frame_cnt = atoi(strchr(tok, '=') + 1);
                } else if (IS_PREFIX(tok, "nodelay")) {
                        s->nodelay = true;
                } else {
                        MSG(ERROR, "Unknown option: %s!\n", tok);
                        return false;
                }
        }
        if (s->add_frame_cnt < 1) {
                MSG(ERROR,
                    "Number of frames missing or invalid (%d)! Must be >= "
                    "1...\n",
                    s->add_frame_cnt);
                return false;
        }
        return true;
}

static void *
add_frame_init(const char *config)
{
        if (strcmp(config, "help") == 0) {
                usage();
                return INIT_NOERR;
        }
        struct state_vo_pp_add_frame *s = calloc(1, sizeof *s);
        s->magic                        = MAGIC;
        char *ccfg                      = strdup(config);
        bool  ret                       = parse_fmt(s, ccfg);
        free(ccfg);
        if (!ret) {
                add_frame_done(s);
                return nullptr;
        }

        return s;
}

static bool
add_frame_reconfigure(void *state, struct video_desc desc)
{
        struct state_vo_pp_add_frame *s = state;
        vf_free(s->f);
        s->f      = vf_alloc_desc_data(desc);
        s->f->fps = desc.fps / s->add_frame_cnt * (s->add_frame_cnt + 1);

        return true;
}

static struct video_frame *
add_frame_getf(void *state)
{
        struct state_vo_pp_add_frame *s = state;

        return s->f;
}

static bool
add_frame_postprocess(void *state, struct video_frame *in,
                      struct video_frame *out, int req_pitch)
{
        struct state_vo_pp_add_frame *s = state;
        assert(in == nullptr || in == s->f);
        if (in == nullptr && s->curr_idx != 1) {
                return false;
        }

        vf_copy_data_pitch(out, req_pitch, s->f);

        if (!s->nodelay) {
                time_ns_t now              = get_time_in_ns();
                time_ns_t new_frame_budget = SEC_TO_NS(1) / s->f->fps;
                time_ns_t sleep_ns =
                    (s->t0 + (s->curr_idx * new_frame_budget)) - now;
                if (sleep_ns > 0) {
                        nanosleep(&(struct timespec){ .tv_nsec = sleep_ns },
                                  nullptr);
                }
        }

        s->curr_idx = (s->curr_idx + 1) % (s->add_frame_cnt + 1);

        return true;
}

static void
add_frame_done(void *state)
{
        struct state_vo_pp_add_frame *s = state;

        vf_free(s->f);
        free(s);
}

static void
add_frame_get_out_desc(void *state, struct video_desc *out,
                       int *in_display_mode)
{
        struct state_vo_pp_add_frame *s = state;

        *out             = video_desc_from_frame(s->f);
        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
}

static const struct vo_postprocess_info vo_pp_add_frameinfo = {
        .init           = add_frame_init,
        .reconfigure    = add_frame_reconfigure,
        .getf           = add_frame_getf,
        .get_out_desc   = add_frame_get_out_desc,
        .get_property   = add_frame_get_property,
        .vo_postprocess = add_frame_postprocess,
        .done           = add_frame_done,
};

REGISTER_MODULE(add_frame, &vo_pp_add_frameinfo,
                LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
