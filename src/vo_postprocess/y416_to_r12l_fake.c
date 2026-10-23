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
        bool                full_range;
        struct video_frame *f;
        bool r10k;
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

static bool
parse_fmt(struct state_vopp_y416_to_r12l_fake *s, char *fmt)
{
        char *save_ptr = nullptr;
        char *item = nullptr;
        while ((item = strtok_r(fmt, ":", &save_ptr))) {
                fmt = nullptr;
                if (IS_PREFIX(item, "full-range")) {
                        s->full_range = true;
                } else if (strcasecmp(item, "r10k") == 0) {
                        s->r10k = true;
                } else {
                        MSG(ERROR, "unknown argument '%s'; see :help.\n", item);
                        return false;
                }
        }
        return true;
}

static void *
init(const char *config)
{
        if (strcmp(config, "help") == 0) {
                color_printf_wrapped(
                    TBOLD("y416_to_r1x_fake")
                    " fake-converts Y416 to R12L or R10k not doing any conversion, "
                    "just pretending Y is G, Cb is R and Cr B (scaled from "
                    "limited range).\n\n");
                color_printf_wrapped("Usage:\n\t" TBOLD(
                    TRED("-p y416_to_r1x_fake") "[:full-range][:r10k]")
                                     "\n");
                color_printf("\n");
                color_printf_wrapped("Use " TBOLD(":full-range")
                                     " to disable limited->full scaling.\n");
                color_printf_wrapped("Use " TBOLD(":r10k")
                                     " convert to R10k instead of R12L.\n");
                color_printf("\n");
                return INIT_NOERR;
        }

        struct state_vopp_y416_to_r12l_fake *s = calloc(1, sizeof *s);
        s->magic                               = MAGIC;

        char *fmt = strdup(config);
        bool ret = parse_fmt(s, fmt);
        free(fmt);
        if (!ret) {
                free(s);
                return nullptr;
        }
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
[[gnu::always_inline]]
static inline void *
y416_to_r10k(void *arg, bool full_range)
{
        enum {
                Y10_Y16_LMT_SCALE = 54, // floor((940-64)*(1<<6)/1024)
                C10_C16_LMT_SCALE = 56, // floor((960-64)*(1<<6)/1024)
                FOOTR_16B         = 4096,
                MAXVAL_10B        = 1023,
        };
        struct task_data *d            = arg;
        const int         width        = d->width;
        const int         height       = d->height;
        const uint16_t   *src          = d->src;

        for (int y = 0; y < height; ++y) {
                unsigned char *dst = d->dst + (y * d->dst_pitch);
                for (int x = 0; x < width; x += 1) {
                        uint16_t r = *src++;
                        uint16_t g = *src++;
                        uint16_t b = *src++;
                        src++; // drop alpha
                        if (full_range) {
                                r = r >> 6;
                                g = g >> 6;
                                b = b >> 6;
                        } else {
                                r = MAX(FOOTR_16B, r);
                                g = MAX(FOOTR_16B, g);
                                b = MAX(FOOTR_16B, b);
                                r = (r - FOOTR_16B) / C10_C16_LMT_SCALE;
                                g = (g - FOOTR_16B) / Y10_Y16_LMT_SCALE;
                                b = (b - FOOTR_16B) / C10_C16_LMT_SCALE;
                        }
                        r    = MIN(MAXVAL_10B, r);
                        g    = MIN(MAXVAL_10B, g);
                        b    = MIN(MAXVAL_10B, b);
                        *dst++ = r >> 2;
                        *dst++ = (r & 0x3) << 6 | g >> 4;
                        *dst++ = (g & 0xF) << 4 | b >> 6;
                        *dst++ = (b & 0x3F) << 2;
                }
        }
        return nullptr;
}

static void *
y416_to_r10k_full(void *arg)
{
        return y416_to_r10k(arg, true);
}

static void *
y416_to_r10k_limited(void *arg)
{
        return y416_to_r10k(arg, false);
}

// adapted yuv444pXXle_to_r12l from from_lavc_vid_conv.c
[[gnu::always_inline]]
static inline void *
y416_to_r12l(void *arg, bool full_range)
{
        enum {
                Y12_Y16_LMT_SCALE = 13, // (3760-256)*(1<<4)/4096
                C12_C16_LMT_SCALE = 14, // (3840-256)*(1<<4)/4096
                FOOTR_16B         = 4096,
                MAXVAL_12B        = 4095,
        };
        struct task_data *d            = arg;
        const int         width        = d->width;
        const size_t      height       = d->height;
        const uint16_t   *src          = d->src;

        for (size_t y = 0; y < height; ++y) {
                unsigned char *dst = d->dst + (y * d->dst_pitch);
                // if width is odd, last pixel is not written now
                for (int x = 0; x < width - 1; x += 2) {
                        uint16_t r[2];
                        uint16_t g[2];
                        uint16_t b[2];
                        for (int j = 0; j < 2; ++j) {
                                uint16_t rr = *src++;
                                uint16_t gg = *src++;
                                uint16_t bb = *src++;
                                src++; // drop alpha
                                if (full_range) {
                                        rr = rr >> 4;
                                        gg = gg >> 4;
                                        bb = bb >> 4;
                                } else {
                                        rr = MAX(FOOTR_16B, rr);
                                        gg = MAX(FOOTR_16B, gg);
                                        bb = MAX(FOOTR_16B, bb);
                                        rr = (rr - FOOTR_16B) / C12_C16_LMT_SCALE;
                                        gg = (gg - FOOTR_16B) / Y12_Y16_LMT_SCALE;
                                        bb = (bb - FOOTR_16B) / C12_C16_LMT_SCALE;
                                }
                                rr   = MIN(MAXVAL_12B, rr);
                                gg   = MIN(MAXVAL_12B, gg);
                                bb   = MIN(MAXVAL_12B, bb);
                                r[j] = rr;
                                g[j] = gg;
                                b[j] = bb;
                        }

                        dst[0] = r[0] & 0xff;
                        dst[1] = (g[0] & 0xf) << 4 | r[0] >> 8;
                        dst[2] = g[0] >> 4;
                        dst[3] = b[0] & 0xff;
                        dst[4] = (r[1] & 0xf) << 4 | b[0] >> 8;
                        dst[5] = r[1] >> 4;
                        dst[6] = g[1] & 0xff;
                        dst[7] = (b[1] & 0xf) << 4 | g[1] >> 8;
                        dst[8] = b[1] >> 4;
                        dst += 9;
                }
        }
        return nullptr;
}

static void *
y416_to_r12l_full(void *arg)
{
        return y416_to_r12l(arg, true);
}

static void *
y416_to_r12l_limited(void *arg)
{
        return y416_to_r12l(arg, false);
}

static bool
postprocess(void *state, struct video_frame *in, struct video_frame *out,
            int req_pitch)
{
        struct state_vopp_y416_to_r12l_fake *s = state;

        if (in == nullptr) {
                return false;
        }

        assert(in->tile_count == 1);
        assert(in->color_spec == Y416);
        assert((uintptr_t) in->tiles[0].data % sizeof(uint16_t) == 0);

        size_t    src_linesize = vc_get_linesize(in->tiles[0].width, Y416);
        size_t    dst_linesize = vc_get_linesize(in->tiles[0].width, out->color_spec);
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
                data[i].src = (const void *) (in->tiles[0].data +
                                              (i * height * src_linesize));
                data[i].dst = (uint8_t *) (out->tiles[0].data +
                                           (i * height * dst_linesize));
                data[i].dst_pitch = req_pitch;
        }
        time_ns_t t0 = get_time_in_ns();
        runnable_t runner = nullptr;
        if (s->r10k) {
                runner = s->full_range ? y416_to_r10k_full : y416_to_r10k_limited;
        } else {
                runner = s->full_range ? y416_to_r12l_full : y416_to_r12l_limited;
        }
        task_run_parallel(runner, cpu_count, data, sizeof data[0], nullptr);
        time_ns_t t1 = get_time_in_ns();
        MSG(DEBUG, "duration %f ms\n", NS_TO_MS_DBL(t1 - t0));
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
        desc.color_spec        = s->r10k ? R10k : R12L;
        *out                   = desc;
        // this doesn't matter unless multiple tiles implemented
        *in_display_mode = DISPLAY_PROPERTY_VIDEO_SEPARATE_TILES;
}

static const struct vo_postprocess_info vo_pp_y416_to_r12l = {
        init, reconfigure, getf, get_out_desc, get_property, postprocess, done,
};

REGISTER_MODULE(y416_to_r1x_fake, &vo_pp_y416_to_r12l,
                LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
// compat - remove in future
REGISTER_HIDDEN_MODULE(y416_to_r12l_fake, &vo_pp_y416_to_r12l,
                LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
