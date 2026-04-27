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

static void i420_to_yuv420p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        return;
}

static const struct uv_to_openapv_conversion uv_to_openapv_conversions[] = {
        { RGBA, OAPV_CS_YCBCR4444, rgba_to_yuv4444p },
        { UYVY, OAPV_CS_YCBCR422, uyvy_to_yuv422p },
        { YUYV, OAPV_CS_YCBCR422, yuyv_to_yuv422p },
        { VUYA, OAPV_CS_YCBCR4444, vuya_to_yuv4444p },
        { R10k, OAPV_CS_YCBCR4444_10LE, r10k_to_yuv4444p },
        { R12L, OAPV_CS_YCBCR4444_12LE, r12l_to_yuv4444p },
        { v210, OAPV_CS_YCBCR422_10LE, v210_to_yuv422p },
        { RGB, OAPV_CS_YCBCR444, rgb_to_yuv444p },
        { BGR, OAPV_CS_YCBCR444, bgr_to_yuv444p },
        { I420, OAPV_CS_YCBCR420, i420_to_yuv420p },
        { VIDEO_CODEC_NONE, OAPV_CS_UNKNOWN, NULL }
};