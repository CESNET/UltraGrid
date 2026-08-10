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
};

static int
init(struct module *parent, const char *cfg, void **state)
{
        (void) parent;

        if (strcmp(cfg, "help") == 0) {
                color_printf_wrapped(
                    TBOLD("r12l_to_y416_fake")
                    " fake-converts R12L to Y416 not doing any conversion, "
                    "just pretending R is Y, G is Cb and R Cr\n");
                return 1;
        }
        if (strlen(cfg) > 0) {
                MSG(ERROR, "r12l_to_y416_fake doesn't take any arguments.\n");
                return -1;
        }

        struct state_cf_r12l_to_y416_fake *s =
            calloc(1, sizeof(struct state_cf_r12l_to_y416_fake));
        s->magic = MAGIC;
        *state   = s;
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

// adapted r12l_to_yuv4XXpYYle from to_lavc_vid_conv.c
static void *
r12l_to_y416(void *arg)
{
        struct task_data *d      = arg;
        int               width  = d->width;
        int               height = d->height;
        const uint8_t    *src    = d->src;
        uint16_t         *out    = d->dst;
#define WRITE_RES                                                              \
        *out++ = g << 4;                                                       \
        *out++ = r << 4;                                                       \
        *out++ = b << 4;                                                       \
        *out++ = 0xFFFF;

        assert(width % 8 == 0);
        for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; x += 8) {
                        uint16_t r = 0;
                        uint16_t g = 0;
                        uint16_t b = 0;

                        r = src[0];
                        r |= (src[1] & 0xF) << 8;
                        g = src[2] << 4 | src[1] >> 4; // g0
                        b = src[3];
                        src += 4;

                        b |= (src[0] & 0xF) << 8;
                        WRITE_RES
                        r = src[1] << 4 | src[0] >> 4; // r1
                        g = src[2];
                        g |= (src[3] & 0xF) << 8;
                        b = src[3] >> 4;
                        src += 4;

                        b |= src[0] << 4; // b1
                        WRITE_RES
                        r = src[1];
                        r |= (src[2] & 0xF) << 8;
                        g = src[3] << 4 | src[2] >> 4; // g2
                        src += 4;

                        b = src[0];
                        b |= (src[1] & 0xF) << 8;
                        WRITE_RES
                        r = src[2] << 4 | src[1] >> 4; // r3
                        g = src[3];
                        src += 4;

                        g |= (src[0] & 0xF) << 8;
                        b = src[1] << 4 | src[0] >> 4; // b3
                        WRITE_RES
                        r = src[2];
                        r |= (src[3] & 0xF) << 8;
                        g = src[3] >> 4;
                        src += 4;

                        g |= src[0] << 4; // g4
                        b = src[1];
                        b |= (src[2] & 0xF) << 8;
                        WRITE_RES
                        r = src[3] << 4 | src[2] >> 4; // r5
                        src += 4;

                        g = src[0];
                        g |= (src[1] & 0xF) << 8;
                        b = src[2] << 4 | src[1] >> 4; // b5
                        WRITE_RES
                        r = src[3];
                        src += 4;

                        r |= (src[0] & 0xF) << 8;
                        g = src[1] << 4 | src[0] >> 4; // g6
                        b = src[2];
                        b |= (src[3] & 0xF) << 8;
                        WRITE_RES
                        r = src[3] >> 4;
                        src += 4;

                        r |= src[0] << 4; // r7
                        g = src[1];
                        g |= (src[2] & 0xF) << 8;
                        b = src[3] << 4 | src[2] >> 4; // b7
                        WRITE_RES
                        src += 4;
                }
        }
        return nullptr;
#undef WRITE_RES
}

static struct video_frame *
filter(void *state, struct video_frame *in)
{
        if (in == nullptr) {
                return nullptr;
        }
        assert(in->tile_count == 1);
        assert(in->color_spec == R12L);
        (void) state;
        size_t    src_linesize = vc_get_linesize(in->tiles[0].width, R12L);
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
                data[i].dst = (uint16_t *) (out->tiles[0].data +
                                            (i * height * dst_linesize));
        }
        task_run_parallel(r12l_to_y416, cpu_count, data, sizeof data[0],
                          nullptr);
        VIDEO_FRAME_DISPOSE(in);
        return out;
}

static const struct capture_filter_info r12l_to_y416_fake_info = {
        .init   = init,
        .done   = done,
        .filter = filter,
};

REGISTER_MODULE(r12l_to_y416_fake, &r12l_to_y416_fake_info,
                LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);
