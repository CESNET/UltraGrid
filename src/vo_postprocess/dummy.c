// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob
/**
 * @file
 * dummy (noop) vo_postprocess
 */

#include <assert.h> // for assert
#include <stdint.h> // for uint32_t
#include <stdlib.h> // for calloc, free
#include <string.h> // for memcpy, strcmp, strlen

#include "capture_filter/vo_pp_wrapper.h" // for ADD_CAPTURE_FILTER_VO_PP_W...
#include "compat/c23.h"                   // IWYU pragma: keep
#include "host.h"                         // for INIT_NOERR
#include "lib_common.h"                   // for REGISTER_MODULE, library_c...
#include "types.h"                        // for tile, video_frame, video_desc
#include "utils/color_out.h"              // for color_printf, TBOLD
#include "utils/macros.h"                 // for to_fourcc
#include "video_codec.h"                  // for vc_get_linesize
#include "video_display.h"                // for display_prop_vid_mode
#include "video_frame.h"                  // for vf_alloc_desc_data, vf_free
#include "vo_postprocess.h"               // for VO_PP_ABI_VERSION, vo_post...

#define MAGIC    to_fourcc('V', 'P', 'd', 'u')
#define MOD_NAME "[vpp/dummy] "

static_assert(VO_PP_ABI_VERSION  == VO_PP_ABI_POSTPROCESS_NULLPTR);

struct state_dummy {
        uint32_t            magic;
        struct video_frame *f;
};

static bool
dummy_get_property(void * /* state */, int /* property */, void * /* val */,
                   size_t * /* len */)
{
        return false;
}

static void
usage()
{
        color_printf(TBOLD("dummy")
                     " postprocessor/capture filter\n");
        color_printf("\n");
}

static void *
dummy_init(const char *config)
{
        if (strlen(config) != 0) {
                usage();
                return strcmp(config, "help") == 0 ? nullptr : INIT_NOERR;
        }

        struct state_dummy *s = calloc(1, sizeof *s);
        s->magic              = MAGIC;

        return s;
}

static bool
dummy_reconfigure(void *state, struct video_desc desc)
{
        struct state_dummy *s = state;
        s->f                  = vf_alloc_desc_data(desc);
        return true;
}

static struct video_frame *
dummy_getf(void *state)
{
        struct state_dummy *s = state;
        return s->f;
}

static bool
dummy_postprocess(void * /* state */, struct video_frame *in,
                  struct video_frame *out, int req_pitch)
{
        if (in == nullptr) {
                return false;
        }

        for (unsigned i = 0; i < out->tile_count; ++i) {
                const size_t src_linesize =
                    vc_get_linesize(out->tiles[i].width, out->color_spec);
                for (size_t ln = 0; ln < out->tiles[i].height; ++ln) {
                        memcpy(out->tiles[i].data + (ln * req_pitch),
                               in->tiles[i].data + (ln * src_linesize),
                               src_linesize);
                }
        }

        return true;
}

static void
dummy_done(void *state)
{
        struct state_dummy *s = state;
        assert(s->magic == MAGIC);
        vf_free(s->f);
        free(s);
}

static void
dummy_get_out_desc(void *state, struct video_desc *out, int *in_display_mode)
{
        struct state_dummy *s = state;

        *out             = video_desc_from_frame(s->f);
        *in_display_mode = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
}

static const struct vo_postprocess_info vo_pp_dummy_info = {
        .init           = dummy_init,
        .reconfigure    = dummy_reconfigure,
        .getf           = dummy_getf,
        .get_out_desc   = dummy_get_out_desc,
        .get_property   = dummy_get_property,
        .vo_postprocess = dummy_postprocess,
        .done           = dummy_done,
};

REGISTER_MODULE(dummy, &vo_pp_dummy_info, LIBRARY_CLASS_VIDEO_POSTPROCESS,
                VO_PP_ABI_VERSION);
ADD_CAPTURE_FILTER_VO_PP_WRAPPER(vo_pp_dummy, dummy_init, dummy_reconfigure,
                                 dummy_get_out_desc, dummy_postprocess,
                                 dummy_done);
