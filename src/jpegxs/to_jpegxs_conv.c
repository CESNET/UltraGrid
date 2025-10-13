#include <svt-jpegxs/SvtJpegxs.h>
#include <string.h>

#include "jpegxs_conv.h"
#include "types.h"
#include "color_space.h"

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

static void rgb_to_yuv444p(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer_t *dst) {

        int bit_depth = 8;
        
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, bit_depth);
        for (int y = 0; y < height; ++y) {
                uint8_t *dst_y = (uint8_t *) dst->data_yuv[0] + y * dst->stride[0];
                uint8_t *dst_u = (uint8_t *) dst->data_yuv[1] + y * dst->stride[1];
                uint8_t *dst_v = (uint8_t *) dst->data_yuv[2] + y * dst->stride[2];

                for (int x = 0; x < width; ++x) {
                        const comp_type_t r = *src++;
                        const comp_type_t g = *src++;
                        const comp_type_t b = *src++;

                        const comp_type_t res_y =
                            (RGB_TO_Y(cfs, r, g, b) >> COMP_BASE) +
                            (1 << (bit_depth - 4));
                        const comp_type_t res_cb =
                            (RGB_TO_CB(cfs, r, g, b) >> COMP_BASE) +
                            (1 << (bit_depth - 1));
                        const comp_type_t res_cr =
                            (RGB_TO_CR(cfs, r, g, b) >> COMP_BASE) +
                            (1 << (bit_depth - 1));

                        *dst_y++ = CLAMP_LIMITED_Y(res_y, bit_depth);
                        *dst_u++ = CLAMP_LIMITED_Y(res_cb, bit_depth);
                        *dst_v++ = CLAMP_LIMITED_Y(res_cr, bit_depth);
                }
        }
}

static const struct uv_to_jpegxs_conversion uv_to_jpegxs_conversions[] = {
        { UYVY, COLOUR_FORMAT_PLANAR_YUV422, uyvy_to_yuv422p },
        { YUYV, COLOUR_FORMAT_PLANAR_YUV422, yuyv_to_yuv422p },
        { I420, COLOUR_FORMAT_PLANAR_YUV420, i420_to_yuv420p },
        { RGB, COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, rgb_to_yuv444p },
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