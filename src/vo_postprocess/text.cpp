/**
 * @file   vo_postprocess/text.cpp
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
#endif /* HAVE_CONFIG_H */

#include <memory>
#include <wand/magick_wand.h>

#include "debug.h"
#include "video.h"
#include "video_display.h" /* DISPLAY_PROPERTY_VIDEO_SEPARATE_FILES */
#include "vo_postprocess/text.h"

#include "tv.h"

using namespace std;

struct state_text {
        struct video_frame *in;
        unique_ptr<char []> data;
        string text;

        DrawingWand *dw;
        MagickWand *wand;
};

bool text_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return false;
}

#define TEXT_H 36
#define MARGIN_X 10
#define MARGIN_Y 10

#define W 200
#define H (TEXT_H + 2*MARGIN_Y)

void * text_init(char *config) {
        struct state_text *s;

        if(!config || strcmp(config, "help") == 0) {
                printf("3d-interlaced takes no parameters.\n");
                return NULL;
        }
        s = new state_text();
        s->text = config;

        MagickWandGenesis();

        return s;
}

int text_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_text *s = (struct state_text *) state;

        vf_free(s->in);
        if (s->wand)
                DestroyMagickWand(s->wand);
        if (s->dw)
                DestroyDrawingWand(s->dw);
        s->in = 0;
        s->wand = 0;
        s->dw = 0;

        s->in = vf_alloc_desc_data(desc);

        const char *color;
        const char *color_outline;
        const char *colorspace;
        if (desc.color_spec == RGBA) {
                color = "#333333FF";
                color_outline = "#FFFFFFFF";
                colorspace = "rgba";
        } else if (desc.color_spec == RGB) {
                color = "#333333FF";
                color_outline = "#FFFFFFFF";
                colorspace = "rgb";
        } else {
                color = "#228080FF";
                color_outline = "#EB8080FF";
                colorspace = "UYVY";
        }

        s->dw = NewDrawingWand();
        DrawSetFontSize(s->dw, TEXT_H);
        auto status = DrawSetFont(s->dw, "helvetica");
        //assert (status != MagickTrue);
        {
               PixelWand *pw = NewPixelWand();
               PixelSetColor(pw, color);
               DrawSetFillColor(s->dw, pw);
               PixelSetColor(pw, color_outline);
               DrawSetStrokeColor(s->dw, pw);
               DestroyPixelWand(pw);
        }

        s->wand = NewMagickWand();
        status = MagickSetFormat(s->wand, colorspace);
        if(status != MagickTrue) {
                return FALSE;
        }

        status = MagickSetSize(s->wand, W, H);
        if(status != MagickTrue) {
                return FALSE;
        }

        status = MagickSetDepth(s->wand, 8);
        if(status != MagickTrue) {
                return FALSE;
        }

        return TRUE;
}

struct video_frame * text_getf(void *state)
{
        struct state_text *s = (struct state_text *) state;

        return s->in;
}

/**
 * Creates from 2 tiles (left and right eye) one in interlaced format.
 *
 * @param[in]  state     postprocessor state
 * @param[in]  in        input frame. Must contain exactly 2 tiles
 * @param[out] out       output frame to be written to. Should have only ony tile
 * @param[in]  req_pitch requested pitch in buffer
 */
bool text_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_text *s = (struct state_text *) state;

        int dstlinesize = vc_get_linesize(W, in->color_spec);
        int srclinesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
        auto tmp = unique_ptr<char []>(new char[H * dstlinesize]);
        for (int y = 0; y < H; y++) {
                memcpy(tmp.get() + y * dstlinesize, in->tiles[0].data + y * srclinesize, dstlinesize);
        }

        MagickRemoveImage(s->wand);
        auto status = MagickReadImageBlob(s->wand, tmp.get(), H * dstlinesize);
        assert (status == MagickTrue);
        //double *ret = MagickQueryFontMetrics(wand, dw, s->text.c_str());
        //fprintf(stderr, "%f %f %f %f %f\n", ret[0], ret[1], ret[2], ret[3], ret[4]);
        unsigned char *data;
        size_t data_len;
        status = MagickAnnotateImage(s->wand, s->dw, MARGIN_X, MARGIN_Y + TEXT_H, 0, s->text.c_str());
        assert (status == MagickTrue);
        status = MagickDrawImage(s->wand, s->dw);
        assert (status == MagickTrue);
        data = MagickGetImageBlob(s->wand, &data_len);
        assert(data);

        memcpy(out->tiles[0].data, in->tiles[0].data, in->tiles[0].data_len);

        if (data_len == H * dstlinesize) {
                for (int y = 0; y < H; y++) {
                        memcpy(out->tiles[0].data + y * srclinesize, data + y * dstlinesize, dstlinesize);
                }
        }

        free(data);

        return true;
}

void text_done(void *state)
{
        struct state_text *s = (struct state_text *) state;

        vf_free(s->in);

        DestroyMagickWand(s->wand);
        DestroyDrawingWand(s->dw);
        MagickWandTerminus();

        delete s;
}

void text_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_text *s = (struct state_text *) state;

        *out = video_desc_from_frame(s->in);

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}

