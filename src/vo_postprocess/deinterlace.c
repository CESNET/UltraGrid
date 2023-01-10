/**
 * @file   vo_postprocess/deinterlace.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET, z. s. p. o.
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
#endif

#include <pthread.h>
#include <stdlib.h>

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "vo_postprocess.h"

struct state_deinterlace {
        struct video_frame *out; ///< for postprocess only
};

static void usage()
{
        printf("Deinterlaces output video frames.\nUsage:\n");
        printf("\t-p deinterlace\n");
}

static void * deinterlace_init(const char *config) {
        if (strcmp(config, "help") == 0) {
                usage();
                return NULL;
        }

        struct state_deinterlace *s = calloc(1, sizeof(struct state_deinterlace));
        assert(s != NULL);

        return s;
}

static int cf_deinterlace_init(struct module *parent, const char *cfg, void **state)
{
        UNUSED(parent);
        void *s = deinterlace_init(cfg);
        if (!s) {
                return 1;
        }
        *state = s;
        return 0;
}

static bool deinterlace_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return false;
}

static int deinterlace_reconfigure(void *state, struct video_desc desc)
{
        struct state_deinterlace *s = (struct state_deinterlace *) state;

        vf_free(s->out);
        assert(desc.tile_count == 1);
        s->out = vf_alloc_desc_data(desc);

        return TRUE;
}

static struct video_frame * deinterlace_getf(void *state)
{
        struct state_deinterlace *s = (struct state_deinterlace *) state;

        return s->out;
}

static bool deinterlace_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        UNUSED(state);
        assert (req_pitch == vc_get_linesize(in->tiles[0].width, in->color_spec));
        assert (video_desc_eq(video_desc_from_frame(out), video_desc_from_frame(in)));
        assert (in->tiles[0].data_len <= vc_get_linesize(in->tiles[0].width, in->color_spec) * in->tiles[0].height);
        assert (out->tiles[0].data_len <= vc_get_linesize(in->tiles[0].width, in->color_spec) * in->tiles[0].height);

        vc_deinterlace_ex(in->color_spec, (unsigned char *) in->tiles[0].data, vc_get_linesize(in->tiles[0].width, in->color_spec),
                                (unsigned char *) out->tiles[0].data, vc_get_linesize(out->tiles[0].width, in->color_spec),
                                in->tiles[0].height);
        memcpy(out->tiles[0].data, in->tiles[0].data, in->tiles[0].data_len);

        return true;
}

static struct video_frame *cf_deinterlace_filter(void *state, struct video_frame *f)
{
        UNUSED(state);

        struct video_frame *out = vf_alloc_desc_data(video_desc_from_frame(f));
        out->callbacks.dispose = vf_free;
        if (!deinterlace_postprocess(state, f, out, vc_get_linesize(f->tiles[0].width, f->color_spec))) {
                VIDEO_FRAME_DISPOSE(f);
                vf_free(out);
                return NULL;
        }
        VIDEO_FRAME_DISPOSE(f);
        return out;
}

static void deinterlace_done(void *state)
{
        struct state_deinterlace *s = (struct state_deinterlace *) state;
        
        vf_free(s->out);
        free(s);
}

static void deinterlace_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_deinterlace *s = (struct state_deinterlace *) state;

        *out = video_desc_from_frame(s->out);

        UNUSED(in_display_mode);
        //*in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}

static const struct vo_postprocess_info vo_pp_deinterlace_info = {
        deinterlace_init,
        deinterlace_reconfigure,
        deinterlace_getf,
        deinterlace_get_out_desc,
        deinterlace_get_property,
        deinterlace_postprocess,
        deinterlace_done,
};

static const struct capture_filter_info capture_filter_deinterlace_info = {
        cf_deinterlace_init,
        deinterlace_done,
        cf_deinterlace_filter
};

REGISTER_MODULE(deinterlace, &vo_pp_deinterlace_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
REGISTER_MODULE(deinterlace, &capture_filter_deinterlace_info, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

