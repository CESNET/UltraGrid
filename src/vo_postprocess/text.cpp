/**
 * @file   vo_postprocess/text.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2015 CESNET, z. s. p. o.
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
/**
 * @todo
 * Add more options - eg. text position and size.
 * Add support for more pixel formats.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#include <memory>

#ifdef WAND7
#include <MagickWand/MagickWand.h>
#else
#include <wand/magick_wand.h>
#endif

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "vo_postprocess.h"

#include "tv.h"

using namespace std;

struct state_text {
        struct video_frame *in;
        unique_ptr<char []> data;
        string text;
        int width, height; // width in pixels
        struct video_desc saved_desc;

        DrawingWand *dw;
        MagickWand *wand;
};

static bool text_get_property(void *state, int property, void *val, size_t *len)
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

static pthread_once_t vo_postprocess_text_initialized = PTHREAD_ONCE_INIT;

static void init_text() {
        MagickWandGenesis();
        atexit(MagickWandTerminus);
}


static void * text_init(const char *config) {
        pthread_once(&vo_postprocess_text_initialized, init_text);

        struct state_text *s;

        if(!config || strcmp(config, "help") == 0) {
                printf("text video postprocess takes as a parameter text to be drawed. Examples:\n");
                printf("\t-p text:stream1\n");
                printf("\t-p \"text:Video stream from location XY\"\n");
                printf("\n");
                return NULL;
        }
        s = new state_text();
        s->text = config;

        return s;
}

static int cf_text_init(struct module * /* parent */, const char *cfg, void **state)
{
        void *s = text_init(cfg);
        if (!s) {
                return 1;
        } else {
                *state = s;
                return 0;
        }
}

static int text_postprocess_reconfigure(void *state, struct video_desc desc)
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

        s->width = min<unsigned long>(MARGIN_X + s->text.length() * TEXT_H, desc.width);
        s->height = min<unsigned long>(2*MARGIN_Y + TEXT_H, desc.height);

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
        } else if (desc.color_spec == UYVY) {
                color = "#228080FF";
                color_outline = "#EB8080FF";
                colorspace = "UYVY";
        } else {
                log_msg(LOG_LEVEL_ERROR, "[text vo_pp.] Codec not supported! Please report to "
                                PACKAGE_BUGREPORT ".\n");
                return FALSE;
        }

        s->dw = NewDrawingWand();
        DrawSetFontSize(s->dw, TEXT_H);
        auto status = DrawSetFont(s->dw, "helvetica");
        if(status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] DraweSetFont failed!\n");
                return FALSE;
        }
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
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickSetFormat failed!\n");
                return FALSE;
        }

        status = MagickSetSize(s->wand, s->width, s->height);
        if(status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickSetSize failed!\n");
                return FALSE;
        }

        status = MagickSetDepth(s->wand, 8);
        if(status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickSetDepth failed!\n");
                return FALSE;
        }

        return TRUE;
}

static struct video_frame * text_getf(void *state)
{
        struct state_text *s = (struct state_text *) state;

        return s->in;
}

/**
 * @todo
 * Rendering of the text is a bit slow. Since the text doesn't change at all, it should be
 * prerendered and then only alpha blended.
 */
static bool text_postprocess(void *state, struct video_frame *in, struct video_frame *out, int req_pitch)
{
        struct state_text *s = (struct state_text *) state;

        int dstlinesize = vc_get_linesize(s->width, in->color_spec);
        int srclinesize = vc_get_linesize(in->tiles[0].width, in->color_spec);
        auto tmp = unique_ptr<char []>(new char[s->height * dstlinesize]);
        for (int y = 0; y < s->height; y++) {
                memcpy(tmp.get() + y * dstlinesize, in->tiles[0].data + y * srclinesize, dstlinesize);
        }

        MagickRemoveImage(s->wand);
        auto status = MagickReadImageBlob(s->wand, tmp.get(), s->height * dstlinesize);
        if (status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickReadImageBlob failed!\n");
                return false;
        }
        //double *ret = MagickQueryFontMetrics(wand, dw, s->text.c_str());
        //fprintf(stderr, "%f %f %f %f %f\n", ret[0], ret[1], ret[2], ret[3], ret[4]);
        unsigned char *data;
        size_t data_len;
        status = MagickAnnotateImage(s->wand, s->dw, MARGIN_X, MARGIN_Y + TEXT_H, 0, s->text.c_str());
        if (status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickAnnotateImage failed!\n");
                return false;
        }
        status = MagickDrawImage(s->wand, s->dw);
        if (status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickDrawImage failed!\n");
                return false;
        }
        data = MagickGetImageBlob(s->wand, &data_len);
        if (!data) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickGetImageBlob failed!\n");
                return false;
        }

        memcpy(out->tiles[0].data, in->tiles[0].data, in->tiles[0].data_len);

        if ((int) data_len == s->height * dstlinesize) {
                for (int y = 0; y < s->height; y++) {
                        memcpy(out->tiles[0].data + y * req_pitch, data + y * dstlinesize, dstlinesize);
                }
        }

        free(data);

        return true;
}

struct video_frame *cf_text_filter(void *state, struct video_frame *f)
{
        struct state_text *s = (struct state_text *) state;

        if (s->saved_desc != video_desc_from_frame(f)) {
                if (text_postprocess_reconfigure(state, video_desc_from_frame(f))) {
                        s->saved_desc = video_desc_from_frame(f);
                } else {
                        log_msg(LOG_LEVEL_WARNING, "[text] Cannot reinitialize!\n");
                        return NULL;
                }
        }

        struct video_frame *out = vf_alloc_desc_data(s->saved_desc);
        out->callbacks.dispose = vf_free;
        if (text_postprocess(state, f, out, vc_get_linesize(f->tiles[0].width, f->color_spec))) {
                VIDEO_FRAME_DISPOSE(f);
                return out;
        } else {
                VIDEO_FRAME_DISPOSE(f);
                vf_free(out);
                return NULL;
        }
}

static void text_done(void *state)
{
        struct state_text *s = (struct state_text *) state;

        vf_free(s->in);

        DestroyMagickWand(s->wand);
        DestroyDrawingWand(s->dw);

        delete s;
}

static void text_get_out_desc(void *state, struct video_desc *out, int *in_display_mode, int *out_frames)
{
        struct state_text *s = (struct state_text *) state;

        *out = video_desc_from_frame(s->in);

        *in_display_mode = DISPLAY_PROPERTY_VIDEO_MERGED;
        *out_frames = 1;
}

static const struct vo_postprocess_info vo_pp_text_info = {
        text_init,
        text_postprocess_reconfigure,
        text_getf,
        text_get_out_desc,
        text_get_property,
        text_postprocess,
        text_done,
};

static const struct capture_filter_info capture_filter_text_info = {
        cf_text_init,
        text_done,
        cf_text_filter
};


REGISTER_MODULE(text, &vo_pp_text_info, LIBRARY_CLASS_VIDEO_POSTPROCESS, VO_PP_ABI_VERSION);
REGISTER_MODULE(text, &capture_filter_text_info, LIBRARY_CLASS_CAPTURE_FILTER, CAPTURE_FILTER_ABI_VERSION);

