/*
 * FILE:    capture_filter/resize.c
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

#ifdef __cplusplus
extern "C" {
#endif

#include "capture_filter.h"

#include "debug.h"

#include "video.h"
#include "video_codec.h"

#define MAX_TILES 16

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);

struct state_resize {
    int num;
    int orig_width;
    int orig_height;
    int denom;
    int reinit;
    struct video_frame *frame;
};

static void usage() {
    printf("\nDownscaling by scale factor:\n\n");
    printf("resize usage:\n");
    printf("\tresize:numerator/denominator\n\n");
    printf("Downscaling example: resize:1/2 - downscale input frame size by scale factor of 2\n");
}

static int init(struct module *parent, const char *cfg, void **state)
{
    UNUSED(parent);

    int n;
    int denom = 1;;
    if(cfg) {
        if(strcasecmp(cfg, "help") == 0) {
            usage();
            return 1;
        }
        n = atoi(cfg);
        if(strchr(cfg, '/')) {
            denom = atoi(strchr(cfg, '/') + 1);
        }
    } else {
        usage();
        return -1;
    }

    if(n <= 0 || denom <= 0){
        printf("\n[RESIZE ERROR] numerator and denominator resize factors must be greater than zero!\n");
        usage();
        return -1;
    }
    if(n/denom > 1.0){
        printf("\n[RESIZE ERROR] numerator and denominator resize factors must be lower than 1 (only downscaling is supported)\n");
        usage();
        return -1;
    }

    struct state_resize *s = (state_resize*) calloc(1, sizeof(struct state_resize));
    s->reinit = 1;
    s->num = n;
    s->denom = denom;

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
    assert(in->tile_count <= MAX_TILES);

    if(s->reinit==1){
    	struct video_desc desc = video_desc_from_frame(in);
    	s->frame = vf_alloc_desc_data(desc);
    }

    for(i=0; i<s->frame->tile_count;i++){
        if(s->reinit==1){
        	//TODO: all tiles could have different sizes and other color specs different than UYVY and RGB
            s->orig_width = s->frame->tiles[i].width;
            s->orig_height = s->frame->tiles[i].height;
            s->frame->tiles[i].width *= s->num;
            s->frame->tiles[i].width /= s->denom;
            s->frame->tiles[i].height *= s->num;
            s->frame->tiles[i].height /= s->denom;
            s->frame->color_spec = RGB;
            s->reinit = 0;
        	if(i==0) printf("[resize filter] resizing from %dx%d to %dx%d\n", in->tiles[i].width, in->tiles[i].height, s->frame->tiles[i].width, s->frame->tiles[i].height);
        }

        res = resize_frame(in->tiles[i].data, in->color_spec, s->frame->tiles[i].data, &s->frame->tiles[i].data_len, s->orig_width, s->orig_height, (double)s->num/s->denom);

        if(res!=0){
            error_msg("\n[RESIZE ERROR] Unable to resize with scale factor configured [%d/%d] in tile number %d\n", s->num, s->denom, i);
            error_msg("\t\t No scale factor applied at all. No frame returns...\n");
            return NULL;
        }else{
            s->frame->color_spec = RGB;
            s->frame->codec = RGB;
        }
    }

    return s->frame;
}

struct capture_filter_info capture_filter_resize = {
    "resize",
    init,
    done,
    filter,
};

#ifdef __cplusplus
}
#endif
