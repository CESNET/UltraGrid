#include <svt-jpegxs/SvtJpegxs.h>
#include <string.h>

#include "jpegxs_conv.h"
#include "types.h"
#include "color_space.h"

static void yuv422p_to_uyvy(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; y++) {
                uint8_t *src_y = (uint8_t *) src->data_yuv[0] + y * src->stride[0];
                uint8_t *src_u = (uint8_t *) src->data_yuv[1] + y * src->stride[1];
                uint8_t *src_v = (uint8_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; x += 2) {
                        *dst++ = *src_u++;
                        *dst++ = *src_y++;
                        *dst++ = *src_v++;
                        *dst++ = *src_y++;
                }
        }
}

static void yuv422p_to_yuyv(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; y++) {
                uint8_t *src_y = (uint8_t *) src->data_yuv[0] + y * src->stride[0];
                uint8_t *src_u = (uint8_t *) src->data_yuv[1] + y * src->stride[1];
                uint8_t *src_v = (uint8_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; x += 2) {
                        *dst++ = *src_y++;
                        *dst++ = *src_u++;
                        *dst++ = *src_y++;
                        *dst++ = *src_v++;
                }
        }
}

static void yuv420p_to_i420(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        const int y_size = width * height;
        const int uv_size = (width / 2) * (height / 2);

        uint8_t *dst_y = dst;
        uint8_t *dst_u = dst_y + y_size;
        uint8_t *dst_v = dst_u + uv_size;

        memcpy(dst_y, src->data_yuv[0], y_size);
        memcpy(dst_u, src->data_yuv[1], uv_size);
        memcpy(dst_v, src->data_yuv[2], uv_size);
}

static void yuv444p_to_rgb(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        int bit_depth = 8;
        const struct color_coeffs cfs = *get_color_coeffs(CS_DFL, bit_depth);
        for (int y = 0; y < height; ++y) {
                uint8_t *src_y = (uint8_t *) src->data_yuv[0] + y * src->stride[0];
                uint8_t *src_u = (uint8_t *) src->data_yuv[1] + y * src->stride[1];
                uint8_t *src_v = (uint8_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; ++x) {
                        int cb = *src_u++ - 128;
                        int cr = *src_v++ - 128;
                        int y = *src_y++ * cfs.y_scale;
                        comp_type_t r = YCBCR_TO_R(cfs, y, cb, cr) >> COMP_BASE;
                        comp_type_t g = YCBCR_TO_G(cfs, y, cb, cr) >> COMP_BASE;
                        comp_type_t b = YCBCR_TO_B(cfs, y, cb, cr) >> COMP_BASE;
                        *dst++ = CLAMP(r, 1, 254);
                        *dst++ = CLAMP(g, 1, 254);
                        *dst++ = CLAMP(b, 1, 254);
                }
        }
}

static const struct jpegxs_to_uv_conversion jpegxs_to_uv_conversions[] = {
        { COLOUR_FORMAT_PLANAR_YUV422, UYVY, yuv422p_to_uyvy },
        { COLOUR_FORMAT_PLANAR_YUV422, YUYV, yuv422p_to_yuyv },
        { COLOUR_FORMAT_PLANAR_YUV420, I420, yuv420p_to_i420 },
        { COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, RGB, yuv444p_to_rgb },
        { COLOUR_FORMAT_INVALID, VIDEO_CODEC_NONE, NULL }
};

const struct jpegxs_to_uv_conversion *get_jpegxs_to_uv_conversion(codec_t codec) {
    
        const struct jpegxs_to_uv_conversion *conv = jpegxs_to_uv_conversions;
        while (conv->dst != VIDEO_CODEC_NONE) {
                if (conv->dst == codec)
                        return conv;
                conv++;
        }

        return NULL;
}

const struct jpegxs_to_uv_conversion *get_default_jpegxs_to_uv_conversion(ColourFormat_t fmt) {

        const struct jpegxs_to_uv_conversion *conv = jpegxs_to_uv_conversions;
        while (conv->dst != VIDEO_CODEC_NONE) {
                if (conv->src == fmt)
                        return conv;
                conv++;
        }

        return NULL;
}