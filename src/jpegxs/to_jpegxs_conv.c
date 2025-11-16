#include <svt-jpegxs/SvtJpegxs.h>
#include <string.h>
#include <stdio.h>

#include "jpegxs_conv.h"
#include "types.h"
#include "video_codec.h"

static void uyvy_to_yuv422p(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint8_t *dst_y = (uint8_t *) dst->data_yuv[0] + y * dst->stride[0];
                uint8_t *dst_u = (uint8_t *) dst->data_yuv[1] + y * dst->stride[1];
                uint8_t *dst_v = (uint8_t *) dst->data_yuv[2] + y * dst->stride[2];

                for (int x = 0; x < width; x += 2) {
                        *dst_u++ = *src++;
                        *dst_y++ = *src++;
                        *dst_v++ = *src++;
                        *dst_y++ = *src++;
                }
        }
}

static void yuyv_to_yuv422p(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint8_t *dst_y = (uint8_t *) dst->data_yuv[0] + y * dst->stride[0];
                uint8_t *dst_u = (uint8_t *) dst->data_yuv[1] + y * dst->stride[1];
                uint8_t *dst_v = (uint8_t *) dst->data_yuv[2] + y * dst->stride[2];

                for (int x = 0; x < width; x += 2) {
                        *dst_y++ = *src++;
                        *dst_u++ = *src++;
                        *dst_y++ = *src++;
                        *dst_v++ = *src++;
                }
        }
}

static void i420_to_yuv420p(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer_t *dst) {

        const int y_size = width * height;
        const int uv_size = (width / 2) * (height / 2);

        const uint8_t *src_y = src;
        const uint8_t *src_u = src_y + y_size;
        const uint8_t *src_v = src_u + uv_size;

        memcpy(dst->data_yuv[0], src_y, y_size);
        memcpy(dst->data_yuv[1], src_u, uv_size);
        memcpy(dst->data_yuv[2], src_v, uv_size);
}

static void rgb_to_rgbp(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint8_t *dst_r = (uint8_t *) dst->data_yuv[0] + y * dst->stride[0];
                uint8_t *dst_g = (uint8_t *) dst->data_yuv[1] + y * dst->stride[1];
                uint8_t *dst_b = (uint8_t *) dst->data_yuv[2] + y * dst->stride[2];

                for (int x = 0; x < width; ++x) {
                        *dst_r++ = *src++;
                        *dst_g++ = *src++;
                        *dst_b++ = *src++;
                }
        }
}

static void v210_to_yuv422p10le(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer_t *dst) {

        for (int y = 0; y < height; ++y) {
                const uint32_t *src_row = (const uint32_t *)(src + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) dst->data_yuv[0] + y * dst->stride[0];
                uint16_t *dst_u = (uint16_t *) dst->data_yuv[1] + y * dst->stride[1];
                uint16_t *dst_v = (uint16_t *) dst->data_yuv[2] + y * dst->stride[2];

                for (int x = 0; x < width; x += 6) {
                        uint32_t w0 = *src_row++;
                        uint32_t w1 = *src_row++;
                        uint32_t w2 = *src_row++;
                        uint32_t w3 = *src_row++;

                        *dst_y++ = (w0 >> 10) & 0x3ff;
                        *dst_y++ = w1 & 0x3ff;
                        *dst_y++ = (w1 >> 20) & 0x3ff;
                        *dst_y++ = (w2 >> 10) & 0x3ff;
                        *dst_y++ = w3 & 0x3ff;
                        *dst_y++ = (w3 >> 20) & 0x3ff;

                        *dst_u++ = w0 & 0x3ff;
                        *dst_u++ = (w1 >> 10) & 0x3ff;
                        *dst_u++ = (w2 >> 20) & 0x3ff;

                        *dst_v++ = (w0 >> 20) & 0x3ff;
                        *dst_v++ = w2 & 0x3ff;
                        *dst_v++ = (w3 >> 10) & 0x3ff;
                }
        }       
}

static void r10k_to_rgbp10le(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer_t *dst) {

        for (int y = 0; y < height; ++y) {
                const uint32_t *src_row = (const uint32_t *)(src + y * vc_get_linesize(width, R10k));
                uint16_t *dst_r = (uint16_t *) dst->data_yuv[0] + y * dst->stride[0];
                uint16_t *dst_g = (uint16_t *) dst->data_yuv[1] + y * dst->stride[1];
                uint16_t *dst_b = (uint16_t *) dst->data_yuv[2] + y * dst->stride[2];

                for (int x = 0; x < width; ++x) {
                        uint32_t w = __builtin_bswap32(*src_row++);
                        *dst_r++ = (w >> 22) & 0x3ff;
                        *dst_g++ = (w >> 12) & 0x3ff;
                        *dst_b++ = (w >> 2) & 0x3ff;
                }
        }       
}

static const struct uv_to_jpegxs_conversion uv_to_jpegxs_conversions[] = {
        { UYVY, COLOUR_FORMAT_PLANAR_YUV422, uyvy_to_yuv422p },
        { YUYV, COLOUR_FORMAT_PLANAR_YUV422, yuyv_to_yuv422p },
        { I420, COLOUR_FORMAT_PLANAR_YUV420, i420_to_yuv420p },
        { RGB, COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, rgb_to_rgbp },
        { v210, COLOUR_FORMAT_PLANAR_YUV422, v210_to_yuv422p10le},
        { R10k, COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, r10k_to_rgbp10le },
        { VIDEO_CODEC_NONE, COLOUR_FORMAT_INVALID, NULL }
};

const struct uv_to_jpegxs_conversion *get_uv_to_jpegxs_conversion(codec_t codec) {

        const struct uv_to_jpegxs_conversion *conv = uv_to_jpegxs_conversions;
        while (conv->src != VIDEO_CODEC_NONE) {
                if (conv->src == codec)
                        return conv;
                conv++;
        }

        return NULL;
}