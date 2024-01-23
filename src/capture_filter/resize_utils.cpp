/**
 * @file    capture_filter/resize_utils.cpp
 * @author  Gerard Castillo     <gerard.castillo@i2cat.net>
 *          Marc Palau          <marc.palau@i2cat.net>
 *          Martin Pulec        <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014      Fundació i2CAT, Internet I Innovació Digital a Catalunya
 * Copyright (c) 2015-2023 CESNET, z. s. p. o.
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

#include <cstdlib>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#pragma GCC diagnostic ignored "-Wcast-qual"
#ifdef HAVE_OPENCV2_OPENCV_HPP
#include <opencv2/opencv.hpp>
#else
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#endif
#pragma GCC diagnostic pop

#include "capture_filter/resize_utils.h"
#include "debug.h"
#include "utils/color_out.h"
#include "video.h"

#define DEFAULT_ALGO INTER_LINEAR
#define MOD_NAME "[resize] "

using cv::INTER_AREA;
using cv::INTER_CUBIC;
using cv::INTER_LANCZOS4;
using cv::INTER_LINEAR;
// using cv::INTER_LINEAR_EXACT;
using cv::INTER_NEAREST;
// using cv::INTER_NEAREST_EXACT;
using cv::Mat;
using cv::Rect;
using cv::Size;

static const char *resize_algo_to_string(int algo);

static Mat ug_to_rgb_mat(codec_t codec, int width, int height, char *indata) {
    Mat yuv;
    Mat rgb;
    int pix_fmt = CV_8UC2;
    int cv_color = 0;
    int num = 1;
    int den = 1;

    switch (codec) {
    case RG48:
        rgb.create(height, width, CV_16UC3);
        rgb.data = (uchar*)indata;
        return rgb;
    case RGB:
        rgb.create(height, width, CV_8UC3);
        rgb.data = (uchar*)indata;
        return rgb;
    case RGBA:
        cv_color = CV_RGBA2RGB;
        pix_fmt = CV_8UC4;
        break;
    case I420:
        pix_fmt = CV_8U;
        num = 3;
        den = 2;
        cv_color = CV_YUV2RGB_I420;
        break;
    case UYVY:
        cv_color = CV_YUV2RGB_UYVY;
        break;
    case YUYV:
        cv_color = CV_YUV2RGB_YUYV;
        break;
    default:
        LOG(LOG_LEVEL_ERROR) << MOD_NAME "Unsupported codec: " << codec << "\n";
        abort();
    }
    yuv.create(height * num / den, width, pix_fmt);
    yuv.data = (uchar*)indata;
    cvtColor(yuv, rgb, cv_color);

    return rgb;
}

static int
get_out_cv_data_type(codec_t pixfmt)
{
    return get_bits_per_component(pixfmt) == DEPTH16 ? CV_16UC3 : CV_8UC3;
}

static void
resize_frame_dimensions(char *indata, codec_t in_color, char *outdata, int width,
             int height, int target_width,
             int target_height, int algo)
{
    const codec_t out_color =
        get_bits_per_component(in_color) == 16 ? RG48 : RGB;
    Mat rgb = ug_to_rgb_mat(in_color, (int) width, (int) height, indata);

    double in_aspect = (double) width / height;
    double out_aspect = (double) target_width / target_height;
    Rect r;
    if (in_aspect == out_aspect) {
        r.x = 0;
        r.y = 0;
        r.width = target_width;
        r.height = target_height;
    } else if (in_aspect > out_aspect) {
        r.x = 0;
        r.width = target_width;
        r.height = (int) (target_width / in_aspect);
        r.y = (target_height - r.height) / 2;
        // clear top and bottom margin
        size_t linesize = vc_get_linesize(target_width, out_color);
        size_t top_margin_size = r.y * linesize;
        size_t bottom_margin_size = (target_height - r.y - r.height) * linesize;
        memset(outdata, 0, top_margin_size);
        memset(outdata + linesize * target_height - bottom_margin_size, 0, bottom_margin_size);
    } else {
        r.y = 0;
        r.height = target_height;
        r.width = (int) (target_height * in_aspect);
        r.x = (target_width - r.width) / 2;
        // clear left and right margins
        size_t linesize = vc_get_linesize(target_width, out_color);
        size_t left_margin_size = vc_get_linesize(r.x, out_color);
        size_t right_margin_size = vc_get_linesize(target_width - r.x - r.width, out_color);
        for (int i = 0; i < target_height; ++i) {
            memset(outdata + i * linesize, 0, left_margin_size);
            memset(outdata + (i + 1) * linesize - right_margin_size, 0, right_margin_size);
        }
    }

    Mat out((int) target_height, (int) target_width,
            get_out_cv_data_type(in_color), outdata);
    resize(rgb, out(r), r.size(), 0, 0, algo);
}

void
resize_frame(char *indata, codec_t in_color, char *outdata, int width,
             int height, struct resize_param *resize_spec)
{
    if (resize_spec->algo == RESIZE_ALGO_DFL) {
        resize_spec->algo = DEFAULT_ALGO;
        MSG(NOTICE, "using resize algorithm: %s\n",
          resize_algo_to_string(DEFAULT_ALGO));
    }

    DEBUG_TIMER_START(resize);
    if (resize_spec->mode == resize_param::USE_FRACTION) {
        const double factor = resize_spec->factor;
        Mat rgb = ug_to_rgb_mat(in_color, (int) width, (int) height, indata);
        Mat out((int) (height * factor), (int) (width * factor),
                get_out_cv_data_type(in_color), outdata);
        resize(rgb, out, Size(0, 0), factor, factor, resize_spec->algo);
    } else if (resize_spec->mode == resize_param::USE_DIMENSIONS) {
        resize_frame_dimensions(indata, in_color, outdata, width, height,
                                resize_spec->target_width,
                                resize_spec->target_height, resize_spec->algo);
    } else {
        abort();
    }
    DEBUG_TIMER_STOP(resize);
}

static const struct {
    int         val;
    const char *name;
} interp_map[] = {
        {INTER_NEAREST,        "nearest"      },
        { INTER_LINEAR,        "linear"       },
        { INTER_CUBIC,         "cubic"        },
        { INTER_AREA,          "area"         },
        { INTER_LANCZOS4,      "lanczos4"     },
        // { INTER_LINEAR_EXACT,  "linear_exact" },
        // { INTER_NEAREST_EXACT, "nearest_exact"},
};

int
resize_algo_from_string(const char *str)
{
    if (strcmp(str, "help") == 0) {
        color_printf("Available resize algorithms:\n");
        for (auto const &i : interp_map) {
            color_printf("\t" TBOLD("%s") "%s\n", i.name,
                         i.val == DEFAULT_ALGO ? " (default)" : "");
        }
        return RESIZE_ALGO_HELP_SHOWN;
    }
    for (auto const &i : interp_map) {
        if (strcmp(i.name, str) == 0) {
            return i.val;
        }
    }
    return RESIZE_ALGO_UNKN;
}

static const char *
resize_algo_to_string(int algo)
{
    for (auto const &i : interp_map) {
        if (i.val == algo) {
            return i.name;
        }
    }
    return "(unknown algo!)";
}

/* vim: set expandtab sw=4: */
