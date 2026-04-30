#include "openapv_conversions.h"

static void rgba_to_yuv4444p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static void uyvy_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static void yuyv_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static void vuya_to_yuv4444p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static void r10k_to_yuv4444p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static void r12l_to_yuv4444p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static void v210_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static void rgb_to_yuv444p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static void bgr_to_yuv444p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static const struct uv_to_openapv_conversion uv_to_openapv_conversions[] = {
        { RGBA, OAPV_CS_YCBCR4444_10LE, rgba_to_yuv4444p },
        { UYVY, OAPV_CS_YCBCR422_10LE, uyvy_to_yuv422p },
        { YUYV, OAPV_CS_YCBCR422_10LE, yuyv_to_yuv422p },
        { VUYA, OAPV_CS_YCBCR4444_10LE, vuya_to_yuv4444p },
        { R10k, OAPV_CS_YCBCR4444_10LE, r10k_to_yuv4444p },
        { R12L, OAPV_CS_YCBCR4444_12LE, r12l_to_yuv4444p },
        { v210, OAPV_CS_YCBCR422_10LE, v210_to_yuv422p },
        { RGB, OAPV_CS_YCBCR444_10LE, rgb_to_yuv444p },
        { BGR, OAPV_CS_YCBCR444_10LE, bgr_to_yuv444p },
        { VIDEO_CODEC_NONE, OAPV_CS_UNKNOWN, NULL }
};

const struct uv_to_openapv_conversion *get_uv_to_openapv_conversion(codec_t codec) {

        const struct uv_to_openapv_conversion *conv = uv_to_openapv_conversions;
        while (conv->src_color_format != VIDEO_CODEC_NONE) {
                if (conv->src_color_format == codec)
                        return conv;
                conv++;
        }

        return NULL;
}