/**
 * @file   vo_postprocess/text.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2023 CESNET, z. s. p. o.
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
 * @file
 * @todo
 * Add more options - eg. text position and size.
 * Add support for more pixel formats.
 */


#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif /* HAVE_CONFIG_H */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#ifdef WAND7
#include <MagickWand/MagickWand.h>
#else
#pragma GCC diagnostic warning "-Wignored-qualifiers"
#include <wand/magick_wand.h>
#endif
#pragma GCC diagnostic pop

#include "capture_filter.h"
#include "debug.h"
#include "lib_common.h"
#include "video.h"
#include "video_display.h"
#include "vo_postprocess.h"
#include "utils/color_out.h"
#include "utils/macros.h"
#include "utils/string.h" // replace_all
#include "utils/text.h"

struct state_text {
        struct video_frame *in;
        char *data;
        size_t data_size;
        char *text;
        int req_x, req_y, req_h;
        int width;                 // pixels
        int text_base_y;
        int margin_x, margin_y, text_h;

        // from the imagemagick font metrics
        int text_width_px;
        int text_height_px;
        struct video_desc saved_desc;

        DrawingWand *dw;
        MagickWand *wand_bg;
        MagickWand *wand_text;
};

static bool text_get_property(void *state, int property, void *val, size_t *len)
{
        UNUSED(state);
        UNUSED(property);
        UNUSED(val);
        UNUSED(len);

        return false;
}

// all three macros are divisors of the actual video pixel dimensions
#define TEXT_H_DIV 30
#define MARGIN_X_DIV 60
#define MARGIN_Y_DIV 60

static pthread_once_t vo_postprocess_text_initialized = PTHREAD_ONCE_INIT;

static void init_text() {
        MagickWandGenesis();
        atexit(MagickWandTerminus);
}


static void * text_init(const char *config) {
        pthread_once(&vo_postprocess_text_initialized, init_text);

        struct state_text *s;

        if (strlen(config) == 0 || strcmp(config, "help") == 0) {
                char desc[] = TBOLD("text") " video postprocess takes as a parameter text to be drawed. "
                        "Colons in text must be escaped with a backslash (see Examples). Spaces may be escaped or the whole argument should be enclosed by quotation marks.\n";
                color_printf("%s", wrap_paragraph(desc));
                color_printf("\nUsage:\n");
                color_printf("\t" TBOLD("-p text:<text>") "\n");
                color_printf("\t" TBOLD("-p text:x=<x>:y=<y>:h=<text_height>:t=<text>") "\n");
                color_printf("\nExamples:\n");
                color_printf(TBOLD("\t-p text:stream1\n"
                                        "\t-p text:x=100:y=100:h=20:t=text\n"
                                        "\t-p \"text:Video stream from location XY\"\n"
                                        "\t-p \"text:Text can also contains escaped colons - \\:\"\n")
                                "\n");
                return NULL;
        }

        s = calloc(1, sizeof *s);
        s->req_x = s->req_y = s->req_h = -1;

        char *config_copy = strdup(config);
        replace_all(config_copy, ESCAPED_COLON, DELDEL);
        char *tmp = config_copy, *item, *save_ptr;
        while ((item = strtok_r(tmp, ":", &save_ptr))) {
                if (strstr(item, "x=") != NULL) {
                        s->req_x = atoi(item + 2);
                } else if (strstr(item, "y=") != NULL) {
                        s->req_y = atoi(item + 2);
                } else if (strstr(item, "h=") != NULL) {
                        s->req_h = atoi(item + 2);
                } else if (strstr(item, "t=") != NULL) {
                        replace_all(item + 2, DELDEL, ":");
                        s->text = strdup(item + 2);
                } else {
                        replace_all(item, DELDEL, ":");
                        s->text = strdup(item);
                }
                tmp = NULL;
        }
        free(config_copy);

        return s;
}

static int cf_text_init(struct module *parent, const char *cfg, void **state)
{
        (void) parent;
        void *s = text_init(cfg);
        if (!s) {
                return 1;
        } else {
                *state = s;
                return 0;
        }
}

static bool
text_postprocess_reconfigure(void *state, struct video_desc desc)
{
        struct state_text *s = (struct state_text *) state;

        vf_free(s->in);
        if (s->wand_text)
                DestroyMagickWand(s->wand_text);
        if (s->wand_bg)
                DestroyMagickWand(s->wand_bg);
        if (s->dw)
                DestroyDrawingWand(s->dw);
        s->in = 0;
        s->wand_bg = 0;
        s->wand_text = 0;
        s->dw = 0;

        s->in = vf_alloc_desc_data(desc);

        s->margin_x = s->req_x == -1 ? (int) desc.width / MARGIN_X_DIV : s->req_x;
        s->margin_y = s->req_y == -1 ? (int) desc.height / MARGIN_Y_DIV : s->req_y;
        s->text_h = s->req_h == -1 ? (int) desc.height / TEXT_H_DIV : s->req_h;
        s->width = MIN(s->margin_x + strlen(s->text) * s->text_h, desc.width);
        s->text_base_y = MIN(s->margin_y + s->text_h, (int) desc.height);


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
                return false;
        }

        s->dw = NewDrawingWand();
        DrawSetFontSize(s->dw, s->text_h);
        MagickBooleanType status = DrawSetFont(s->dw, "helvetica");
        if(status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] DraweSetFont failed!\n");
                return false;
        }
        {
               PixelWand *pw = NewPixelWand();
               PixelSetColor(pw, color);
               DrawSetFillColor(s->dw, pw);
               PixelSetColor(pw, color_outline);
               DrawSetStrokeColor(s->dw, pw);
               DestroyPixelWand(pw);
        }

        s->wand_bg = NewMagickWand();
        status = MagickSetFormat(s->wand_bg, colorspace);
        s->wand_text = NewMagickWand();
        status &= MagickSetFormat(s->wand_text, colorspace);
        if(status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickSetFormat failed!\n");
                assert(false);
                return false;
        }

        // TODO this is wrong not sure what to set
        // status = MagickSetSize(s->wand_bg, s->, s->text_height_pt);

        if(status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickSetSize failed!\n");
                assert(false);
                return false;
        }

        status = MagickSetDepth(s->wand_bg, 8);
        status &= MagickSetDepth(s->wand_text, 8);
        if(status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickSetDepth failed!\n");
                assert(false);
                return false;
        }
        
        
        // MagickLabelImage doesn't create the canvas for you as CLI would
        PixelWand *transparent_bg = NewPixelWand();
        PixelSetColor(transparent_bg, "#cccccc80");
        // PixelSetColor(transparent, "#00000000");

        // still need a dummy canvas here unlike in CLI
        MagickNewImage(s->wand_text, 1, 1, transparent_bg);

        // char width, char height, ascender, descender, txt w, text h
        double *ret = MagickQueryFontMetrics(s->wand_text, s->dw, s->text);
        assert(ret != NULL);
        s->text_width_px = ret[4];  // whole height, including the descender
        s->text_height_px = ret[5];
        double descender = ret[3];  // negative
        double ascender = ret[2];

        printf("font metrics: %d %d %f\n", s->text_width_px, s->text_height_px, descender);
        // printf("width px, height pt??: %d %d\n", s->width, s->height);


        MagickRemoveImage(s->wand_text);
        assert(MagickNewImage(s->wand_text, s->text_width_px, s->text_height_px, transparent_bg));

        double x,y;
        x = 0;
        y = s->text_height_px + descender;
        double rot = 0;
        assert(MagickAnnotateImage(s->wand_text, s->dw, x, y, rot, s->text));
        assert(MagickDrawImage(s->wand_text, s->dw));  // does nothing...?

        assert(MagickWriteImage(s->wand_text, "text.png"));

        DestroyPixelWand(transparent_bg);

        

        printf("%d %d %d\n", s->width, s->text_base_y, s->text_height_px);

        return true;
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
        MagickBooleanType status;
        struct state_text *s = (struct state_text *) state;

        // --- load wand_bg from video frame somehow ---

        int dstlinesize = vc_get_linesize(s->width, in->color_spec);
        int srclinesize = vc_get_linesize(in->tiles[0].width, in->color_spec);

        printf("sizes: %d %d %d\n", in->tiles[0].width, srclinesize, dstlinesize);
        // assume it's 2, 3 or 4
        int bytes_per_pixel = srclinesize / in->tiles[0].width;


        // copy over just the relevant "stripe"
        // cannot copy just that rectangle because pixel formats
        // are not necessarily byte aligned (obviously)

        char *tmp = malloc((size_t) s->text_height_px * dstlinesize);

        // printf("%d %d\n", srclinesize, dstlinesize);

        for (ptrdiff_t y = 0; y < s->text_height_px; y++) {
                // oob skip
                if (y + s->margin_y < 0) 
                        continue;
                memcpy(tmp + (y) * dstlinesize,
                       in->tiles[0].data + (y + s->margin_y) * srclinesize, 
                       dstlinesize);
        }

        // printf("dstlsz: %d\n", dstlinesize);

        MagickRemoveImage(s->wand_bg);
        assert(MagickSetSize(s->wand_bg, dstlinesize / bytes_per_pixel, 
                             s->text_height_px));
        status = MagickReadImageBlob(s->wand_bg, tmp, s->text_height_px * dstlinesize);
        if (status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickReadImageBlob failed!\n");
                return false;
        }

        unsigned char *data, *datatx;
        size_t data_len, datatx_len;

        status = MagickCompositeImage(s->wand_bg, s->wand_text, OverCompositeOp, 0, s->margin_x, 0);
        if (status != MagickTrue) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickCompositeImage failed!\n");
                return false;
        }

        data = MagickGetImageBlob(s->wand_bg, &data_len);
        datatx = MagickGetImageBlob(s->wand_text, &datatx_len);
        if (!data) {
                log_msg(LOG_LEVEL_WARNING, "[text vo_pp.] MagickGetImageBlob failed!\n");
                return false;
        }

        // send the input stream...
        memcpy(out->tiles[0].data, in->tiles[0].data, in->tiles[0].data_len);

        printf("%ld %ld\n", data_len, datatx_len);

        // ...and the text stripe on top of it
        // if ((int) data_len == s->text_height_pt * dstlinesize) {
                for (int y = 0; y < s->text_height_px; y++) {
                        // oob skip
                        if (y + s->margin_y < 0) 
                                continue;
                        memcpy(out->tiles[0].data + (y + s->margin_y) * req_pitch,
                               data + y * dstlinesize,
                               dstlinesize);
                }
        // }

        return true;
}

static struct video_frame *cf_text_filter(void *state, struct video_frame *f)
{
        struct state_text *s = (struct state_text *) state;

        if (!video_desc_eq(s->saved_desc, video_desc_from_frame(f))) {
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

        DestroyMagickWand(s->wand_bg);
        DestroyMagickWand(s->wand_text);
        DestroyDrawingWand(s->dw);

        free(s->data);
        free(s->text);
        free(s);
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

