// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#include <assert.h> // for assert
#include <stdint.h> // for uint16_t, uint8_t, uint32_t
#include <stdlib.h> // for size_t, NULL, free, malloc
#include <string.h> // for memcpy, strcmp, strlen

#include "compat/c23.h"      // IWYU pragma: keep
#include "debug.h"           // for LOG_LEVEL_ERROR, MSG
#include "host.h"            // for INIT_NOERR
#include "lib_common.h"      // for REGISTER_MODULE, library_class
#include "types.h"           // for tile, video_frame, Y416, video_desc, R12L
#include "utils/color_out.h" // for TBOLD
#include "utils/macros.h"    // for MAX, MIN, to_fourcc
#include "utils/misc.h"      // for get_cpu_core_count
#include "utils/text.h"      // for color_printf_wrapped
#include "utils/worker.h"    // for task_run_parallel
#include "video_codec.h"     // for vc_get_linesize
#include "video_display.h"   // for display_prop_vid_mode
#include "video_frame.h"     // for vf_free, vf_alloc_desc_data, video_desc...
#include "vo_postprocess.h"  // for VO_PP_ABI_VERSION, VO_PP_PROPERTY_CODECS

#define MAGIC    to_fourcc('V', 'P', 'Y', 'R')
#define MOD_NAME "[vopp/y416 to r12l fake] "

struct state_vopp_y416_to_r12l_fake {
        uint32_t            magic;
        struct video_frame *f;
};

static bool
get_property(void *state, int property, void *val, size_t *len)
{
        codec_t supported[] = { Y416 };

        (void) state;

        switch (property) {
        case VO_PP_PROPERTY_CODECS:
                assert(*len > (int) sizeof(supported));
                memcpy(val, &supported, sizeof(supported));
                *len = sizeof(supported);
                return true;
        }

        return false;
}

static void *
init(const char *config)
{
        if (strcmp(config, "help") == 0) {
                color_printf_wrapped(
                    TBOLD("r12l_to_y416_fake")
                    " fake-converts R12L to Y416 not doing any conversion, "
                    "just pretending R is Y, G is Cb and R Cr\n");
                return INIT_NOERR;
        }
        if (strlen(config) > 0) {
                MSG(ERROR, "y416_to_r12l_fake doesn't take any arguments.\n");
                return NULL;
        }
        struct state_vopp_y416_to_r12l_fake *s = malloc(sizeof *s);
        s->magic                               = MAGIC;
        return s;
}

static bool
reconfigure(void *state, struct video_desc desc)
{
        struct state_vopp_y416_to_r12l_fake *s = state;
        assert(desc.color_spec == Y416);
        assert(desc.tile_count == 1);

        vf_free(s->f);
        desc.color_spec = Y416;
        s->f            = vf_alloc_desc_data(desc);

        return true;
}

static struct video_frame *
getf(void *state)
{
        struct state_vopp_y416_to_r12l_fake *s = state;
        return s->f;
}

struct task_data {
        int width;
        int height;
        const uint16_t *restrict src;
        uint8_t *restrict dst;
        int dst_pitch;
};

// adapted yuv444pXXle_to_r12l from from_lavc_vid_conv.c
static inline void *
y416_to_r12l(void *arg)
{
        struct task_data *d            = arg;
        const int         width        = d->width;
        const int         height       = d->height;
        const uint16_t   *src          = d->src;
        unsigned char    *dst          = d->dst;
        int               dst_linesize = vc_get_linesize(width, R12L);
        assert(dst_linesize <= d->dst_pitch);
        assert(width % 8 == 0);

        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; x += 8) {
                        uint16_t r[8];
                        uint16_t g[8];
                        uint16_t b[8];
                        for (int j = 0; j < 8; ++j) {
                                uint16_t gg = *src++;
                                uint16_t rr = *src++;
                                uint16_t bb = *src++;
                                src++; // drop alpha
                                rr   = rr >> 4;
                                gg   = gg >> 4;
                                bb   = bb >> 4;
                                rr   = MIN(4095, rr);
                                gg   = MIN(4095, gg);
                                bb   = MIN(4095, bb);
                                r[j] = rr;
                                g[j] = gg;
                                b[j] = bb;
                        }

                        dst[0]      = r[0] & 0xff;
                        dst[1]      = (g[0] & 0xf) << 4 | r[0] >> 8;
                        dst[2]      = g[0] >> 4;
                        dst[3]      = b[0] & 0xff;
                        dst[4 + 0]  = (r[1] & 0xf) << 4 | b[0] >> 8;
                        dst[4 + 1]  = r[1] >> 4;
                        dst[4 + 2]  = g[1] & 0xff;
                        dst[4 + 3]  = (b[1] & 0xf) << 4 | g[1] >> 8;
                        dst[8 + 0]  = b[1] >> 4;
                        dst[8 + 1]  = r[2] & 0xff;
                        dst[8 + 2]  = (g[2] & 0xf) << 4 | r[2] >> 8;
                        dst[8 + 3]  = g[2] >> 4;
                        dst[12 + 0] = b[2] & 0xff;
                        dst[12 + 1] = (r[3] & 0xf) << 4 | b[2] >> 8;
                        dst[12 + 2] = r[3] >> 4;
                        dst[12 + 3] = g[3] & 0xff;
                        dst[16 + 0] = (b[3] & 0xf) << 4 | g[3] >> 8;
                        dst[16 + 1] = b[3] >> 4;
                        dst[16 + 2] = r[4] & 0xff;
                        dst[16 + 3] = (g[4] & 0xf) << 4 | r[4] >> 8;
                        dst[20 + 0] = g[4] >> 4;
                        dst[20 + 1] = b[4] & 0xff;
                        dst[20 + 2] = (r[5] & 0xf) << 4 | b[4] >> 8;
                        dst[20 + 3] = r[5] >> 4;
                        dst[24 + 0] = g[5] & 0xff;
                        dst[24 + 1] = (b[5] & 0xf) << 4 | g[5] >> 8;
                        dst[24 + 2] = b[5] >> 4;
                        dst[24 + 3] = r[6] & 0xff;
                        dst[28 + 0] = (g[6] & 0xf) << 4 | r[6] >> 8;
                        dst[28 + 1] = g[6] >> 4;
                        dst[28 + 2] = b[6] & 0xff;
                        dst[28 + 3] = (r[7] & 0xf) << 4 | b[6] >> 8;
                        dst[32 + 0] = r[7] >> 4;
                        dst[32 + 1] = g[7] & 0xff;
                        dst[32 + 2] = (b[7] & 0xf) << 4 | g[7] >> 8;
                        dst[32 + 3] = b[7] >> 4;
                        dst += 36;
                }
                dst += d->dst_pitch - dst_linesize;
        }
        return nullptr;
}

static bool
postprocess(void *state, struct video_frame *in, struct video_frame *out,
            int req_pitch)
{
        (void) state;

        if (in == nullptr) {
                return false;
        }

        assert(in->tile_count == 1);
        assert(in->color_spec == Y416);

        size_t    src_linesize = vc_get_linesize(in->tiles[0].width, Y416);
        size_t    dst_linesize = vc_get_linesize(in->tiles[0].width, R12L);
        const int cpu_count    = get_cpu_core_count();
        struct task_data data[cpu_count];
        for (int i = 0; i < cpu_count; ++i) {
                data[i].width = (int) in->tiles[0].width;
                size_t height = in->tiles[0].height / cpu_count;
                if (i < cpu_count - 1) {
                        data[i].height = (int) height;
                } else { // we are last so we need to do the rest
                        data[i].height = (int) (in->tiles[0].height -
                                                (height * (cpu_count - 1)));
                }
                data[i].src       = (uint16_t *) (in->tiles[0].data +
                                                  (i * height * src_linesize));
                data[i].dst       = (uint8_t *) (out->tiles[0].data +
                                                 (i * height * dst_linesize));
                data[i].dst_pitch = req_pitch;
        }
        task_run_parallel(y416_to_r12l, cpu_count, data, sizeof data[0],
                          nullptr);
        return true;
}

static void
done(void *state)
{
        assert(state);
        struct state_vopp_y416_to_r12l_fake *s = state;
        assert(s->magic == MAGIC);

        vf_free(s->f);
        free(state);
}

static void
get_out_desc(void *state, struct video_desc *out, int *in_display_mode)
{
        struct state_vopp_y416_to_r12l_fake *s = state;

        struct video_desc desc = video_desc_from_frame(s->f);
        desc.color_spec        = R12L;
        *out                   = desc;
        // this doesn't matter unless multiple tiles implemented
        *in_display_mode = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
}

static const struct vo_postprocess_info vo_pp_y416_to_r12l = {
        init, reconfigure, getf, get_out_desc, get_property, postprocess, done,
};

REGISTER_MODULE(y416_to_r12l_fake, &vo_pp_y416_to_r12l,
                LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
