/**
 * @file   vo_postprocess/temporal_3d.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2025 CESNET
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

#include <assert.h>  // for assert
#include <stdbool.h> // for bool, true, false
#include <stdint.h>  // for uint32_t
#include <stdio.h>   // for printf, NULL, size_t
#include <stdlib.h>  // for free, malloc, calloc
#include <string.h>  // for strcmp, strlen

#include "compat/usleep.h"   // for usleep
#include "debug.h"           // for MSG
#include "lib_common.h"      // for REGISTER_MODULE, library_class
#include "tv.h"              // for time_ns_t
#include "types.h"           // for tile, video_desc, video_frame
#include "utils/color_out.h" // for color_printf, TBOLD
#include "utils/macros.h"    // for to_fourcc
#include "video_codec.h"     // for vc_get_linesize
#include "video_display.h"   // for display_prop_vid_mode
#include "video_frame.h"     // for vf_get_tile, vf_free
#include "vo_postprocess.h"  // for VO_PP_ABI_VERSION, vo_postprocess_info

// #include "vo_postprocess.h"

#define MAGIC    to_fourcc('v', 'p', 't', '3')
#define MOD_NAME "[temporal_3d] "
#define TIMEOUT  "20ms"

struct state_temporal_3d {
        uint32_t            magic;
        struct video_frame *in;
        time_ns_t           first_tile_time;
        bool disable_timing; ///< issue the second eye right after the first
};

static void temporal_3d_done(void *state);

static bool
temporal_3d_get_property(void *state, int property, void *val, size_t *len)
{
        (void) state;
        if (property == VO_PP_VIDEO_MODE) {
                assert(*len >= sizeof(int));
                *len         = sizeof(int);
                *(int *) val = DISPLAY_PROPERTY_VIDEO_SEPARATE_3D;
                return true;
        }

        return false;
}

static void
usage()
{
        color_printf(
            TBOLD("temporal_3d") " postprocessor interleaves left and "
                                 "right eye temporarily into a single stream "
                                 "with double FPS.\n\n");
        color_printf(
            "Usage:\n\t" TBOLD(TRED("-p temporal_3d") "[:nodelay]") "\n\n");
        printf("Parameters:\n");
        printf(
            "\tnodelay - disable timing, pass right eye right after first\n");
        printf("\t          (may help performance)\n");
}

static void *
temporal_3d_init(const char *config)
{
        if (strcmp(config, "help") == 0) {
                usage();
                return NULL;
        }
        struct state_temporal_3d *s = calloc(1, sizeof *s);
        s->magic                    = MAGIC;
        if (strcmp(config, "nodelay") == 0) {
                s->disable_timing = true;
                if (get_commandline_param("decoder-drop-policy") == NULL) {
                        MSG(NOTICE,
                            "nodelay option used, setting drop policy to %s "
                            "timeout.\n",
                            TIMEOUT);
                        set_commandline_param("decoder-drop-policy", TIMEOUT);
                }
        } else {
                MSG(ERROR, "Unknown option: %s!\n", config);
                temporal_3d_done(s);
                return NULL;
        }

        return s;
}

static bool
temporal_3d_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_temporal_3d *s = state;
        assert(s->magic == MAGIC);
        assert(desc.tile_count == 2);
        s->in = vf_alloc_desc_data(desc);

        return true;
}

static struct video_frame *
temporal_3d_getf(void *state)
{
        struct state_temporal_3d *s = state;

        return s->in;
}

/**
 * Creates from 2 tiles (left and right eye) one in interlaced format.
 *
 * @param[in]  state     postprocessor state
 * @param[in]  in        input frame. Must contain exactly 2 tiles
 * @param[out] out       output frame to be written to. Should have only one
 * tile
 * @param[in]  req_pitch requested pitch in buffer
 */
static bool
temporal_3d_postprocess(void *state, struct video_frame *in,
                        struct video_frame *out, int req_pitch)
{
        struct state_temporal_3d *s = state;
        assert(in == NULL || in == s->in);
        assert(out->tile_count == 1);

        if (in != NULL) {
                s->first_tile_time = get_time_in_ns();
        }

        struct tile *in_tile = &s->in->tiles[in == NULL ? 1 : 0];
        const int linesize = vc_get_linesize(in_tile->width, s->in->color_spec);
        for (size_t y = 0; y < out->tiles[0].height; ++y) {
                memcpy(out->tiles[0].data + (y * req_pitch),
                       in_tile->data + (y * linesize), linesize);
        }

        // delay the other tile for correct timing
        if (!s->disable_timing && in == NULL) {
                time_ns_t t1 = get_time_in_ns();
                long long since_first_tile_us =
                    NS_TO_US(t1 - s->first_tile_time);
                long long sleep_us =
                    (US_IN_SEC / s->in->fps) - (double) since_first_tile_us;
                if (sleep_us > 0) {
                        usleep(sleep_us);
                }
        }

        return true;
}

static void
temporal_3d_done(void *state)
{
        struct state_temporal_3d *s = state;
        vf_free(s->in);
        free(state);
}

static void
temporal_3d_get_out_desc(void *state, struct video_desc *out,
                         int *in_display_mode, int *out_frames)
{
        struct state_temporal_3d *s = (struct state_temporal_3d *) state;

        *out = video_desc_from_frame(s->in);
        out->fps *= 2;
        out->tile_count = 1;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
        *out_frames      = 2;
}

static const struct vo_postprocess_info vo_pp_temporal_3d_info = {
        temporal_3d_init,         temporal_3d_postprocess_reconfigure,
        temporal_3d_getf,         temporal_3d_get_out_desc,
        temporal_3d_get_property, temporal_3d_postprocess,
        temporal_3d_done,
};

REGISTER_MODULE(temporal_3d, &vo_pp_temporal_3d_info,
                LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
