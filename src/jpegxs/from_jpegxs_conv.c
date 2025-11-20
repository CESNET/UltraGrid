#include <svt-jpegxs/SvtJpegxs.h>
#include <string.h>
#include <stdio.h>

#include "jpegxs_conv.h"
#include "types.h"
#include "video_codec.h"

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

static void rgbp_to_rgb(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint8_t *src_r = (uint8_t *) src->data_yuv[0] + y * src->stride[0];
                uint8_t *src_g = (uint8_t *) src->data_yuv[1] + y * src->stride[1];
                uint8_t *src_b = (uint8_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; ++x) {
                        *dst++ = *src_r++;
                        *dst++ = *src_g++;
                        *dst++ = *src_b++;
                }
        }
}

static void yuv422p10le_to_v210(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint32_t *dst_row = (uint32_t *)(dst + y * vc_get_linesize(width, v210));
                uint16_t *src_y = (uint16_t *) src->data_yuv[0] + y * src->stride[0];
                uint16_t *src_u = (uint16_t *) src->data_yuv[1] + y * src->stride[1];
                uint16_t *src_v = (uint16_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; x += 6) {
                        uint16_t y0 = *src_y++;
                        uint16_t y1 = *src_y++;
                        uint16_t y2 = *src_y++;
                        uint16_t y3 = *src_y++;
                        uint16_t y4 = *src_y++;
                        uint16_t y5 = *src_y++;

                        uint16_t u0 = *src_u++;
                        uint16_t u2 = *src_u++;
                        uint16_t u4 = *src_u++;

                        uint16_t v0 = *src_v++;
                        uint16_t v2 = *src_v++;
                        uint16_t v4 = *src_v++;

                        uint32_t w0 = ((v0 & 0x3FF) << 20) | ((y0 & 0x3FF) << 10) | (u0 & 0x3FF);
                        uint32_t w1 = ((y2 & 0x3FF) << 20) | ((u2 & 0x3FF) << 10) | (y1 & 0x3FF);
                        uint32_t w2 = ((u4 & 0x3FF) << 20) | ((y3 & 0x3FF) << 10) | (v2 & 0x3FF);
                        uint32_t w3 = ((y5 & 0x3FF) << 20) | ((v4 & 0x3FF) << 10) | (y4 & 0x3FF);
                        
                        *dst_row++ = w0;
                        *dst_row++ = w1;
                        *dst_row++ = w2;
                        *dst_row++ = w3;
                }
        }
}

static void rgbp10le_to_r10k(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint32_t *dst_row = (uint32_t *)(dst + y * vc_get_linesize(width, R10k));
                uint16_t *src_r = (uint16_t *) src->data_yuv[0] + y * src->stride[0];
                uint16_t *src_g = (uint16_t *) src->data_yuv[1] + y * src->stride[1];
                uint16_t *src_b = (uint16_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; ++x) {
                        uint16_t r = *src_r++;
                        uint16_t g = *src_g++;
                        uint16_t b = *src_b++;
                        uint32_t w = ((r & 0x3FF) << 22) | ((g & 0x3FF) << 12) | ((b & 0x3FF) << 2);
                        *dst_row++ = __builtin_bswap32(w);
                }
        }
}

static void rgbp12le_to_r12l(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst) {

        for (int y = 0; y < height; ++y) {
                uint16_t *dst_row = (uint16_t *)(dst + y * vc_get_linesize(width, R12L));
                uint16_t *src_r = (uint16_t *) src->data_yuv[0] + y * src->stride[0];
                uint16_t *src_g = (uint16_t *) src->data_yuv[1] + y * src->stride[1];
                uint16_t *src_b = (uint16_t *) src->data_yuv[2] + y * src->stride[2];

                for (int x = 0; x < width; ++x) {
                        *dst_row++ = *src_b++ & 0x0FFF;
                        *dst_row++ = *src_g++ & 0x0FFF;
                        *dst_row++ = *src_r++ & 0x0FFF;
                }
        }
}

static const struct jpegxs_to_uv_conversion jpegxs_to_uv_conversions[] = {
        { COLOUR_FORMAT_PLANAR_YUV422, UYVY, 8, yuv422p_to_uyvy },
        { COLOUR_FORMAT_PLANAR_YUV422, YUYV, 8, yuv422p_to_yuyv },
        { COLOUR_FORMAT_PLANAR_YUV420, I420, 8, yuv420p_to_i420 },
        { COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, RGB, 8, rgbp_to_rgb },
        { COLOUR_FORMAT_PLANAR_YUV422, v210, 10, yuv422p10le_to_v210 },
        { COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, R10k, 10, rgbp10le_to_r10k },
        { COLOUR_FORMAT_PLANAR_YUV444_OR_RGB, R12L, 12,  rgbp12le_to_r12l},
        { COLOUR_FORMAT_INVALID, VIDEO_CODEC_NONE, 0, NULL }
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

const struct jpegxs_to_uv_conversion *get_default_jpegxs_to_uv_conversion(ColourFormat_t fmt, int depth) {

        const struct jpegxs_to_uv_conversion *conv = jpegxs_to_uv_conversions;
        while (conv->dst != VIDEO_CODEC_NONE) {
                if (conv->src == fmt && conv->depth == depth)
                        return conv;
                conv++;
        }

        return NULL;
}