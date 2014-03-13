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

#include "capture_filter.h"

#include "debug.h"

#include "video.h"
#include "video_codec.h"

#define MAX_TILES 16

struct module;

static int init(struct module *parent, const char *cfg, void **state);
static void done(void *state);
static struct video_frame *filter(void *state, struct video_frame *in);
//int resize(char *data, unsigned int data_len, struct opencv_tile_struct *opencv);
//int reconfigure_opencv_tile_struct(struct opencv_tile_struct *opencv, unsigned int width, unsigned int height, int num, int denom);

struct opencv_tile_struct {
    int width;
    int height;


};

struct state_resize {
    int num;
    int denom;
    int reinit;
    struct video_frame *frame;
    struct opencv_tile_struct **opencv;
};

static void usage() {
    printf("\nUp/Downscaling by scale factor:\n\n");
    printf("resize usage:\n");
    printf("\tresize:numerator[/denominator]\n\n");
    printf("Downscaling example: resize:2 - downscale input frame size by scale factor of 2\n");
    printf("Upscaling example: resize:1/2 - upscale input frame size by scale factor of 2\n");
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

    struct state_resize *s = calloc(1, sizeof(struct state_resize));
    s->reinit = 0;
    s->num = n;
    s->denom = denom;
    s->frame = vf_alloc(MAX_TILES);
    s->opencv = calloc(MAX_TILES, sizeof(struct opencv_tile_struct*));

    *state = s;
    return 0;
}

static void done(void *state)
{
    struct state_resize *s = state;

    vf_free(s->frame);
    free(state);
}

int resize(char *indata, char *outdata, unsigned int data_len, struct opencv_tile_struct *opencv){
    int res = 0;

    //TODO RESIZE METHOD HERE!!!

    return res;
}

int reconfigure_opencv_tile_struct(struct opencv_tile_struct *opencv,unsigned int width, unsigned int height, int num, int denom){
    int res = 0;

    //TODO DALE VUEY

    return 0;
}

static struct video_frame *filter(void *state, struct video_frame *in)
{
    struct state_resize *s = state;
    int i, j;
    int res = 0;
    assert(in->tile_count <= MAX_TILES);
    memcpy(s->frame, in, sizeof(struct video_frame));
    memcpy(s->frame->tiles, in->tiles, in->tile_count * sizeof(struct tile));

    if(s->reinit == 0){
        for(i=0; i<s->frame->tile_count;i++){
            res = reconfigure_opencv_tile_struct(s->opencv[i], s->frame->tiles[i].width, s->frame->tiles[i].height, s->num, s->denom);
            if(res!=0){
                //TODO ERROR!!!
            }
        }
        s->reinit = 1;
    }

    for(i=0; i<s->frame->tile_count;i++){

        res = resize(in->tiles[i].data, s->frame->tiles[i].data, s->frame->tiles[i].data_len, s->opencv[i]);

        if(res!=0){
            printf("\n[RESIZE ERROR] Unable to resize with scale factor configured [%d/%d] in tile number %d\n", s->num, s->denom, i);
            printf("\t\t No scale factor applied at all. Bypassing original frame.\n");
            return in;
        }

        s->frame->tiles[i].width *= s->num;
        s->frame->tiles[i].width /= s->denom;
        s->frame->tiles[i].height *= s->num;
        s->frame->tiles[i].height /= s->denom;
    }


    return s->frame;
}

struct capture_filter_info capture_filter_resize = {
    .name = "resize",
    .init = init,
    .done = done,
    .filter = filter,
};

