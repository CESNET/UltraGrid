/**
 * @file   vo_postprocess/crop.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "utils/color_out.h"
#include "video.h"
#include "video_display.h"
#include "vo_postprocess.h"

struct state_crop {
        int width;
        int height;
        int xoff;
        int yoff;
        struct video_desc in_desc;
        struct video_desc out_desc;
        struct video_frame *in;
};

static bool crop_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state), UNUSED(property), UNUSED(val), UNUSED(len);
        return false;
}

static void usage(_Bool capture_filter) {
        color_printf("This filter spatially crops video to given dimensions. If the video is smaller than the crop, no cropping is performed.\n\n");
        color_printf("crop filter takes optional parameters: " TBOLD("width") ", "
                        TBOLD("height") ", "
                        TBOLD("xoff") " and "
                        TBOLD("yoff") ". Example:\n"
                        "\t" TBOLD(TRED("%s crop") "[:width=<w>][:height=<h>][:xoff=<x>][:yoff=<y>]") "\n\n",
                        capture_filter ? "--capture-filter" : "-p");
}

static void * crop_init(const char *config) {

        if (strcmp(config, "help") == 0) {
                usage(0);
                return NULL;
        }

        struct state_crop *s = calloc(1, sizeof *s);

        char *tmp = strdup(config);
        char *config_copy = tmp;
        char *item, *save_ptr;
        while ((item = strtok_r(config_copy, ":", &save_ptr))) {
                if (strncasecmp(item, "width=", strlen("width=")) == 0) {
                        s->width = atoi(item + strlen("width="));
                } else if (strncasecmp(item, "height=", strlen("height=")) == 0) {
                        s->height = atoi(item + strlen("height="));
                } else if (strncasecmp(item, "xoff=", strlen("xoff=")) == 0) {
                        s->xoff = atoi(item + strlen("xoff="));
                } else if (strncasecmp(item, "yoff=", strlen("yoff=")) == 0) {
                        s->yoff = atoi(item + strlen("yoff="));
                } else {
                        log_msg(LOG_LEVEL_ERROR, "Wrong config: %s!\n", item);
                        free(tmp);
                        free(s);
                        return NULL;
                }

                config_copy = NULL;
        }

        free(tmp);

        return s;
}

static int crop_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_crop *s = state;
        vf_free(s->in);

        s->in_desc = desc;
        s->in = vf_alloc_desc_data(desc);

        s->out_desc = desc;
        s->out_desc.width = s->width ? MIN(s->width, desc.width) : desc.width;
        s->out_desc.height = s->height ? MIN(s->height, desc.height) : desc.height;

        // make sure that width is divisible by pixel block size
        assert(get_pf_block_bytes(desc.color_spec) != 0);
        int linesize = (int) (s->out_desc.width * get_bpp(desc.color_spec)) / get_pf_block_bytes(desc.color_spec)
                * get_pf_block_bytes(desc.color_spec);
        s->out_desc.width = linesize / get_bpp(desc.color_spec);

        return TRUE;
}

static struct video_frame * crop_getf(void *state)
{
        struct state_crop *s = state;
        return s->in;
}

static bool crop_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        assert(in->tile_count == 1);
        assert(get_pf_block_bytes(in->color_spec) != 0);

        struct state_crop *s = state;
        int src_linesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
        int xoff = s->xoff + out->tiles[0].width > in->tiles[0].width ? in->tiles[0].width - out->tiles[0].width : (unsigned) s->xoff;
        int xoff_bytes = (int) (xoff * get_bpp(in->color_spec)) / get_pf_block_bytes(in->color_spec)
                * get_pf_block_bytes(in->color_spec);
        int yoff = s->yoff + out->tiles[0].height > in->tiles[0].height ? in->tiles[0].height - out->tiles[0].height : (unsigned) s->yoff;

        for (int y = 0 ; y < (int) out->tiles[0].height; y++) {
                memcpy(out->tiles[0].data + y * req_pitch,
                                in->tiles[0].data + (yoff + y) * src_linesize + xoff_bytes,
                                req_pitch);
        }

        return true;
}

static void crop_done(void *state)
{
        struct state_crop *s = state;

        vf_free(s->in);
        free(s);
}

static void crop_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        *out = ((struct state_crop *) state)->out_desc;

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}

static int cf_crop_init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        if (strcmp(cfg, "help") == 0) {
                usage(1);
                return 1;
        }
        void *s = crop_init(cfg);
        if (!s) {
                return -1;
        }
        *state = s;
        return 0;
}

static struct video_frame *cf_crop_filter(void *state, struct video_frame *f)
{
        struct state_crop *s = state;

        if (!video_desc_eq(s->in_desc, video_desc_from_frame(f))) {
                if (!crop_postprocess_reconfigure(s, video_desc_from_frame(f))) {
                        abort(); // cannot fail now
                }
        }

        struct video_frame *out = vf_alloc_desc_data(s->out_desc);
        out->callbacks.dispose = vf_free;
        if (!crop_postprocess(state, f, out, vc_get_linesize(s->out_desc.width, f->color_spec))) {
                VIDEO_FRAME_DISPOSE(f);
                vf_free(out);
                return NULL;
        }
        VIDEO_FRAME_DISPOSE(f);
        return out;
}

static const struct vo_postprocess_info vo_pp_crop_info = {
        crop_init,
        crop_postprocess_reconfigure,
        crop_getf,
        crop_get_out_desc,
        crop_get_property,
        crop_postprocess,
        crop_done,
};

static const struct capture_filter_info capture_filter_crop_info = {
        cf_crop_init,
        crop_done,
        cf_crop_filter
};

REGISTER_MODULE(crop, &vo_pp_crop_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
REGISTER_MODULE(crop, &capture_filter_crop_info, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

