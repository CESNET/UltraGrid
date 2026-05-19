// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob
/**
 * @file
 * this CF can be used eg. for 50p->60p adjustments
 */

#include <assert.h> // for assert
#include <stdint.h> // for uint32_t
#include <stdlib.h> // for free, atoi, calloc
#include <string.h> // for memcpy, strchr, strcmp, strdup, strtok_r
#include <time.h>   // for nanosleep, timespec

#include "capture_filter.h"  // for CAPTURE_FILTER_ABI_VERSION, capture_fil...
#include "compat/c23.h"      // IWYU pragma: keep
#include "debug.h"           // for LOG_LEVEL_ERROR, MSG
#include "lib_common.h"      // for REGISTER_MODULE, library_class
#include "tv.h"              // for time_ns_t, get_time_in_ns, SEC_TO_NS
#include "types.h"           // for video_frame, video_frame_callbacks, tile
#include "utils/color_out.h" // for color_printf, TBOLD
#include "utils/macros.h"    // for to_fourcc, IS_KEY_PREFIX, IS_PREFIX
#include "utils/text.h"      // for wrap_paragraph
#include "video_frame.h"     // for vf_free, vf_alloc_desc, video_desc_from...

struct module;

#define MAGIC    to_fourcc('C', 'F', 'a', 'f')
#define MOD_NAME "[cf/add_frame] "

struct state_add_frame {
        uint32_t            magic;
        bool                nodelay;
        int                 add_frame_cnt; ///< +1 added frame
        int                 curr_idx;
        struct video_frame *cached;
        time_ns_t           t0;
};

static void
usage()
{
        color_printf("capture filter " TBOLD("add_frame")
                     " adds 1 extra frame per every amount of "
                     "frames.\n\n");

        color_printf("Typical use case is to convert 50p to 60p.\n\n");

        color_printf("Usage:\n"
                     "\t" TBOLD("-F add_frame:every=<num>[:nodelay]")
                     "\n\n");

        color_printf(
            "(in the proposed 50p->60p case, the <num> will be 5)\n\n");

        color_printf("Example converting 50i->60p (notice `nodelay` for DF):\n");
        color_printf("\t " TBOLD(
            "uv -F double_framerate:nodelay,add_frame:e=5")
                     " -t testcard:fps=50i\n");
        color_printf("or simply for 50p->60p:\n");
        color_printf("\t " TBOLD("uv -F add_frame:e=5")
                     " -t testcard:fps=50p\n");
        color_printf("\n");

        color_printf_wrapped("See also the capture filter " TBOLD("every")
                             " for frame dropping (eg. to achieve 60p->50p "
                             "conversion) and video postprocessor " TBOLD(
                                     "add_frame")
                             ".\n\n");
}

static void
done(void *state)
{
        struct state_add_frame *s = state;
        if (s->cached != nullptr) {
                vf_free(s->cached);
        }
        free(s);
}

static bool
parse_fmt(struct state_add_frame *s, char *fmt)
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

static int
init(struct module * /* parent */, const char *cfg, void **state)
{
        assert(cfg != nullptr);
        if (strcmp(cfg, "help") == 0) {
                usage();
                return 1;
        }
        struct state_add_frame *s = calloc(1, sizeof *s);
        s->magic                  = MAGIC;
        char *ccfg                = strdup(cfg);
        bool  ret                 = parse_fmt(s, ccfg);
        free(ccfg);
        if (!ret) {
                done(s);
                return -1;
        }
        *state = s;
        return 0;
}

// from ./override_prop.c
static void
dispose_frame(struct video_frame *f)
{
        VIDEO_FRAME_DISPOSE((struct video_frame *) f->callbacks.dispose_udata);
        vf_free(f);
}

/**
 * use new frane "envelope" with overriden fps
 */
static struct video_frame *
wrap_with_fps_adj(struct state_add_frame *s, struct video_frame *in)
{
        struct video_frame *out = vf_alloc_desc(video_desc_from_frame(in));
        out->fps = out->fps / s->add_frame_cnt * (s->add_frame_cnt + 1);
        memcpy(out->tiles, in->tiles, in->tile_count * sizeof(struct tile));
        out->callbacks.dispose       = dispose_frame;
        out->callbacks.dispose_udata = in;

        return out;
}

static struct video_frame *
filter(void *state, struct video_frame *in)
{
        struct state_add_frame *s = state;
        if (in == nullptr) {
                if (s->cached == nullptr) {
                        return nullptr;
                }
                in        = s->cached;
                s->cached = nullptr;
        }

        // first frame will be dupped
        if (s->curr_idx == 0) {
                if (s->cached != nullptr) {
                        vf_free(s->cached);
                }
                s->cached                    = vf_get_copy(in);
                s->cached->callbacks.dispose = vf_free;

                s->curr_idx = 1;
                s->t0       = get_time_in_ns();

                return wrap_with_fps_adj(s, in);
        }

        // wrap with new header (fps adj)
        double new_fps = in->fps / s->add_frame_cnt * (s->add_frame_cnt + 1);
        struct video_frame *out = wrap_with_fps_adj(s, in);

        if (!s->nodelay) {
                time_ns_t now              = get_time_in_ns();
                time_ns_t new_frame_budget = SEC_TO_NS(1) / new_fps;
                time_ns_t sleep_ns =
                    (s->t0 + (s->curr_idx * new_frame_budget)) - now;
                if (sleep_ns > 0) {
                        nanosleep(&(struct timespec){ .tv_nsec = sleep_ns },
                                  nullptr);
                }
        }

        s->curr_idx = (s->curr_idx + 1) % (s->add_frame_cnt + 1);
        return out;
}

static const struct capture_filter_info capture_filter_blank = {
        .init   = init,
        .done   = done,
        .filter = filter,
};

REGISTER_MODULE(add_frame, &capture_filter_blank, LIBRARY_CLASS_CAPTURE_FILTER,
                CAPTURE_FILTER_ABI_VERSION);
