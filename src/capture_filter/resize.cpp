/*
 * FILE:    capture_filter/resize.cpp
 * AUTHORS: Gerard Castillo     <gerard.castillo@i2cat.net>
 *          Marc Palau          <marc.palau@i2cat.net>
 *
 * Copyright (c) 2005-2010 Fundació i2CAT, Internet I Innovació Digital a Catalunya
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *
 *      This product includes software developed by the Fundació i2CAT,
 *      Internet I Innovació Digital a Catalunya. This product also includes
 *      software developed by CESNET z.s.p.o.
 *
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
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
 *
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include "capture_filter/resize_utils.h"

#include "lib_common.h"

#include "capture_filter.h"

#include "debug.h"

#include "video.h"
#include "video_codec.h"

#ifdef __cplusplus
extern "C" {
#endif


struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_resize {
    int num;
    int denom;
    double scale_factor;
    int target_width, target_height;
    struct video_frame *frame;
    struct video_desc saved_desc;
    bool force_interlaced;
};

static void usage() {
    printf("\nScaling by scale factor:\n\n");
    printf("resize usage:\n");
    printf("\tresize:numerator[/denominator]\n");
    printf("\tor\n");
    printf("\tresize:<width>x<height>\n\n");
    printf("Scaling examples:\n"
                    "\tresize:1/2 - downscale input frame size by scale factor of 2\n"
                    "\tresize:720x576i - scales input to PAL (overrides interlacing setting)\n");
}

static int init(struct module * /* parent */, const char *cfg, void **state)
{
    int n = 0, w = 0, h = 0;
    int denom = 1;
    bool force_interlaced = false;
    if(cfg) {
        char *endptr;
        if(strcasecmp(cfg, "help") == 0) {
            usage();
            return 1;
        }
        if (strchr(cfg, 'x')) {
            w = strtol(cfg, &endptr, 10);
            errno = 0;
            h = strtol(strchr(cfg, 'x') + 1, &endptr, 10);
            if (errno != 0) {
                perror("strtol");
                usage();
                return -1;
            }
            if (*endptr == 'i') {
                    force_interlaced = true;
            }
        } else {
            n = strtol(cfg, &endptr, 10);
            if(strchr(cfg, '/')) {
                denom = strtol(strchr(cfg, '/') + 1, &endptr, 10);
            }
        }

        if (*endptr != '\0') {
            usage();
            return -1;
        }
    } else {
        usage();
        return -1;
    }

    if((n <= 0 || denom <= 0) && ((w <= 0) || (h <= 0))){
        printf("\n[RESIZE ERROR] resize factors must be greater than zero!\n");
        usage();
        return -1;
    }

    struct state_resize *s = (state_resize*) calloc(1, sizeof(struct state_resize));
    s->num = n;
    s->denom = denom;
    s->scale_factor = (double)s->num/s->denom;
    s->force_interlaced = force_interlaced;

    s->target_width = w;
    s->target_height = h;

    *state = s;
    return 0;
}

static void done(void *state)
{
    struct state_resize *s = (state_resize*) state;

    vf_free(s->frame);
    free(state);
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
    struct state_resize *s = (state_resize*) state;
    unsigned int i;
    int res = 0;

    if (!video_desc_eq(video_desc_from_frame(in), s->saved_desc)) {
    	struct video_desc desc = video_desc_from_frame(in);
        if (s->target_width != 0) {
            desc.width = s->target_width;
            desc.height = s->target_height;
        } else {
            desc.width = in->tiles[0].width * s->num / s->denom;
            desc.height = in->tiles[0].height * s->num / s->denom;
        }
        desc.color_spec = RGB;
    	s->frame = vf_alloc_desc_data(desc);
        if (s->force_interlaced) {
                s->frame->interlacing = INTERLACED_MERGED;
        }
        s->saved_desc = video_desc_from_frame(in);
        printf("[resize filter] resizing from %dx%d to %dx%d\n", in->tiles[0].width, in->tiles[0].height, s->frame->tiles[0].width, s->frame->tiles[0].height);
    }

    for(i=0; i<s->frame->tile_count;i++){
        if (s->target_width != 0) {
            res = resize_frame(in->tiles[i].data, in->color_spec, s->frame->tiles[i].data, in->tiles[i].width, in->tiles[i].height, s->target_width, s->target_height);
        } else {
            res = resize_frame(in->tiles[i].data, in->color_spec, s->frame->tiles[i].data, in->tiles[i].width, in->tiles[i].height, s->scale_factor);
        }

        if(res!=0){
            error_msg("\n[RESIZE ERROR] Unable to resize with scale factor configured [%d/%d] in tile number %d\n", s->num, s->denom, i);
            error_msg("\t\t No scale factor applied at all. No frame returns...\n");
            return NULL;
        }
    }

    VIDEO_FRAME_DISPOSE(in);

    return s->frame;
}

static struct capture_filter_info capture_filter_resize = {
    init,
    done,
    filter,
};

#ifdef __cplusplus
}
#endif

REGISTER_MODULE(resize, &capture_filter_resize, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

/* vim: set expandtab: sw=4 */
