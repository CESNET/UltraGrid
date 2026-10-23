// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#include <assert.h> // for assert
#include <stdint.h> // for uint16_t, uint32_t
#include <stdlib.h> // for size_t, calloc, free
#include <string.h> // for strcmp, strlen

#include "capture_filter.h"  // for CAPTURE_FILTER_ABI_VERSION, capture_fil...
#include "compat/c23.h"      // IWYU pragma: keep
#include "debug.h"           // for LOG_LEVEL_ERROR, MSG
#include "lib_common.h"      // for REGISTER_MODULE, library_class
#include "tv.h"
#include "types.h"           // for tile, video_frame, Y416, video_desc, R12L
#include "utils/color_out.h" // for TBOLD
#include "utils/macros.h"    // for to_fourcc
#include "utils/misc.h"      // for get_cpu_core_count
#include "utils/text.h"      // for color_printf_wrapped
#include "utils/worker.h"    // for task_run_parallel
#include "video_codec.h"     // for vc_get_linesize
#include "video_frame.h"     // for VIDEO_FRAME_DISPOSE, vf_alloc_desc_data
struct module;

#define MAGIC    to_fourcc('C', 'F', 'R', 'Y')
#define MOD_NAME "[r12l to y416 fake cap. f.] "

struct state_cf_r12l_to_y416_fake {
        uint32_t magic;
        bool     full_range;
};

static int
init(struct module *parent, const char *cfg, void **state)
{
        (void) parent;

        if (strcmp(cfg, "help") == 0) {
                color_printf_wrapped(
                    TBOLD("r1x_to_y416_fake")
                    " fake-converts R10k or R12L to Y416 not doing actual conversion, "
                    "just pretending G is Y, R is Cb and B Cr (scaled to "
                    "limited range).\n\n");
                color_printf_wrapped("Usage:\n\t" TBOLD(
                    TRED("-F r1x_to_y416_fake") "[:full-range]")
                                     "\n");
                color_printf("\n");
                color_printf_wrapped("Use " TBOLD(":full-range")
                                     " to disable full->limited scaling.\n");
                color_printf("\n");
                return 1;
        }

        struct state_cf_r12l_to_y416_fake *s =
            calloc(1, sizeof(struct state_cf_r12l_to_y416_fake));
        s->magic = MAGIC;

        if (IS_PREFIX(cfg, "full-range")) {
                s->full_range = true;
        } else if (strlen(cfg) > 0) {
                MSG(ERROR, "unknown argument '%s'; see :help.\n", cfg);
                free(s);
                return -1;
        }

        *state = s;
        return 0;
}

static void
done(void *state)
{
        if (!state) {
                return;
        }
        struct state_cf_r12l_to_y416_fake *s = state;
        assert(s->magic == MAGIC);
        free(state);
}

struct task_data {
        int width;
        int height;
        const uint8_t *restrict src;
        uint16_t *restrict dst;
};

// adapted r10k_to_yuv4XXpYYle from to_lavc_vid_conv.c
[[gnu::always_inline]] static inline void *
r10k_to_y416(void *arg, bool full_range)
{
        enum {
                Y10_Y16_LMT_SCALE = 54, // floor((940-64)*(1<<6)/1024)
                C10_C16_LMT_SCALE = 56, // floor((960-64)*(1<<6)/1024)
                FOOTR_16B          = 4096,
        };
        struct task_data *d      = arg;
        int               width  = d->width;
        int               height = d->height;
        uint16_t         *out    = d->dst;
#define WRITE_RES                                                              \
        if (full_range) {                                                      \
                *out++ = r << 6;                                               \
                *out++ = g << 6;                                               \
                *out++ = b << 6;                                               \
        } else {                                                               \
                *out++ = (C10_C16_LMT_SCALE * r) + FOOTR_16B;                  \
                *out++ = (Y10_Y16_LMT_SCALE * g) + FOOTR_16B;                  \
                *out++ = (C10_C16_LMT_SCALE * b) + FOOTR_16B;                  \
        }                                                                      \
        *out++ = 0xFFFF;

        size_t src_linesize = vc_get_linesize(width, R10k);
        for (int y = 0; y < height; ++y) {
                const uint8_t *src = d->src + (y * src_linesize);
                OPTIMIZED_FOR(int x = 0; x < width; x += 1)
                {
                        unsigned int byte1 = *src++;
                        unsigned int byte2 = *src++;
                        unsigned int byte3 = *src++;
                        unsigned int byte4 = *src++;
                        uint16_t     r     = byte1 << 2U | (byte2 & 0xC0U) >> 6;
                        uint16_t g = (byte2 & 0x3FU) << 4U | (byte3 & 0xF0U)
                                                                  >> 4U;
                        uint16_t b = (byte3 & 0xFU) << 6U | (byte4 & 0xFCU)
                                                                 >> 2U;
                        WRITE_RES
                }
        }
        return nullptr;
#undef WRITE_RES
}

static void *
r10k_to_y416_full(void *arg)
{
        return r10k_to_y416(arg, true);
}

static void *
r10k_to_y416_limited(void *arg)
{
        return r10k_to_y416(arg, false);
}

// adapted r12l_to_yuv4XXpYYle from to_lavc_vid_conv.c
[[gnu::always_inline]] static inline void *
r12l_to_y416(void *arg, bool full_range)
{
        enum {
                Y12_Y16_LMT_SCALE = 13, // (3760-256)*(1<<4)/4096
                C12_C16_LMT_SCALE = 14, // (3840-256)*(1<<4)/4096
                FOOTR_16B          = 4096,
        };
        struct task_data *d      = arg;
        int               width  = d->width;
        int               height = d->height;
        uint16_t         *out    = d->dst;
#define WRITE_RES                                                              \
        if (full_range) {                                                      \
                *out++ = r << 4;                                               \
                *out++ = g << 4;                                               \
                *out++ = b << 4;                                               \
        } else {                                                               \
                *out++ = (C12_C16_LMT_SCALE * r) + FOOTR_16B;                  \
                *out++ = (Y12_Y16_LMT_SCALE * g) + FOOTR_16B;                  \
                *out++ = (C12_C16_LMT_SCALE * b) + FOOTR_16B;                  \
        }                                                                      \
        *out++ = 0xFFFF;

        size_t src_linesize = vc_get_linesize(width, R12L);
        for (int y = 0; y < height; ++y) {
                const uint8_t *src = d->src + (y * src_linesize);
                int            x   = 0;
                OPTIMIZED_FOR(; x < (width & ~1); x += 2)
                {
                        uint16_t r = src[0] | (src[1] & 0xF) << 8;
                        uint16_t g = src[1] >> 4 | src[2] << 4;
                        uint16_t b = src[3] | (src[4] & 0xF) << 8;
                        WRITE_RES
                        r = src[4] >> 4 | src[5] << 4;
                        g = src[6] | (src[7] & 0xF) << 8;
                        b = src[7] >> 4 | src[8] << 4;
                        WRITE_RES
                        src += 9;
                }
                if (x < width) {
                        uint16_t r = src[0] | (src[1] & 0xF) << 8;
                        uint16_t g = src[1] >> 4 | src[2] << 4;
                        uint16_t b = src[3] | (src[4] & 0xF) << 8;
                        WRITE_RES
                }
        }
        return nullptr;
#undef WRITE_RES
}

static void *
r12l_to_y416_full(void *arg)
{
        return r12l_to_y416(arg, true);
}

static void *
r12l_to_y416_limited(void *arg)
{
        return r12l_to_y416(arg, false);
}

static struct video_frame *
filter(void *state, struct video_frame *in)
{
        struct state_cf_r12l_to_y416_fake *s = state;
        if (in == nullptr) {
                return nullptr;
        }
        assert(in->tile_count == 1);
        assert(in->color_spec == R12L || in->color_spec == R10k);
        size_t    src_linesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
        size_t    dst_linesize = vc_get_linesize(in->tiles[0].width, Y416);
        const int cpu_count    = get_cpu_core_count();
        struct task_data  data[cpu_count];
        struct video_desc desc  = video_desc_from_frame(in);
        desc.color_spec         = Y416;
        struct video_frame *out = vf_alloc_desc_data(desc);
        out->callbacks.dispose  = vf_free;
        for (int i = 0; i < cpu_count; ++i) {
                data[i].width  = (int) in->tiles[0].width;
                data[i].height = (int) in->tiles[0].height;
                size_t height  = in->tiles[0].height / cpu_count;
                if (i < cpu_count - 1) {
                        data[i].height = (int) height;
                } else { // we are last so we need to do the rest
                        data[i].height = (int) (in->tiles[0].height -
                                                (height * (cpu_count - 1)));
                }
                data[i].src = (uint8_t *) (in->tiles[0].data +
                                           (i * height * src_linesize));
                data[i].dst =
                    (void *) (out->tiles[0].data + (i * height * dst_linesize));
        }
        time_ns_t t0 = get_time_in_ns();
        runnable_t runner = nullptr;
        if (in->color_spec == R10k) {
                runner =
                    s->full_range ? r10k_to_y416_full : r10k_to_y416_limited;
        } else {
                runner = s->full_range ? r12l_to_y416_full
                                      : r12l_to_y416_limited;
        }
        task_run_parallel(runner, cpu_count, data, sizeof data[0], nullptr);
        time_ns_t t1 = get_time_in_ns();
        MSG(DEBUG, "duration %f ms\n", NS_TO_MS_DBL(t1 - t0));
        VIDEO_FRAME_DISPOSE(in);
        return out;
}

static const struct capture_filter_info r1x_to_y416_fake_info = {
        .init   = init,
        .done   = done,
        .filter = filter,
};

REGISTER_MODULE(r1x_to_y416_fake, &r1x_to_y416_fake_info,
                LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
// compat - remove in future
REGISTER_HIDDEN_MODULE(r12l_to_y416_fake, &r1x_to_y416_fake_info,
                LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
