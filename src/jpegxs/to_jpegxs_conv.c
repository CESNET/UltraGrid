#include <svt-jpegxs/SvtJpegxs.h>
#include <string.h>

#include "jpegxs_conv.h"
#include "types.h"

static void uyvy_to_yuv422p(const uint8_t *uyvy, int width, int height, svt_jpeg_xs_image_buffer_t *in_buf) {
        uint8_t *dst_y = (uint8_t *) in_buf->data_yuv[0];
        uint8_t *dst_u = (uint8_t *) in_buf->data_yuv[1];
        uint8_t *dst_v = (uint8_t *) in_buf->data_yuv[2];

        for (int y = 0; y < height; ++y) {
                const uint8_t *src_line = uyvy + y * width * 2;
                uint8_t *dst_y_line = dst_y + y * width;
                uint8_t *dst_u_line = dst_u + y * (width / 2);
                uint8_t *dst_v_line = dst_v + y * (width / 2);

                for (int x = 0; x < width; x += 2) {
                        int i = x * 2;
                        uint8_t u = src_line[i + 0];
                        uint8_t y0 = src_line[i + 1];
                        uint8_t v = src_line[i + 2];
                        uint8_t y1 = src_line[i + 3];

                        dst_y_line[x + 0] = y0;
                        dst_y_line[x + 1] = y1;

                        int chroma_index = x / 2;
                        dst_u_line[chroma_index] = u;
                        dst_v_line[chroma_index] = v;
                }
        }
}

static void yuyv_to_yuv422p(const uint8_t *yuyv, int width, int height, svt_jpeg_xs_image_buffer_t *in_buf) {
        uint8_t *dst_y = (uint8_t *) in_buf->data_yuv[0];
        uint8_t *dst_u = (uint8_t *) in_buf->data_yuv[1];
        uint8_t *dst_v = (uint8_t *) in_buf->data_yuv[2];

        for (int y = 0; y < height; ++y) {
                const uint8_t *src_line = yuyv + y * width * 2;
                uint8_t *dst_y_line = dst_y + y * width;
                uint8_t *dst_u_line = dst_u + y * (width / 2);
                uint8_t *dst_v_line = dst_v + y * (width / 2);

                for (int x = 0; x < width; x += 2) {
                        int i = x * 2;
                        uint8_t y0 = src_line[i + 0];
                        uint8_t u = src_line[i + 1];
                        uint8_t y1 = src_line[i + 2];
                        uint8_t v = src_line[i + 3];

                        dst_y_line[x + 0] = y0;
                        dst_y_line[x + 1] = y1;

                        int chroma_index = x / 2;
                        dst_u_line[chroma_index] = u;
                        dst_v_line[chroma_index] = v;
                }
        }
}

static void i420_to_yuv420p(const uint8_t *i420, int width, int height, svt_jpeg_xs_image_buffer_t *in_buf) {
        const int y_size = width * height;
        const int uv_width = width / 2;
        const int uv_height = height / 2;
        const int u_size = uv_width * uv_height;

        const uint8_t *src_y = i420;
        const uint8_t *src_u = i420 + y_size;
        const uint8_t *src_v = i420 + y_size + u_size;

        uint8_t *dst_y = (uint8_t *) in_buf->data_yuv[0];
        uint8_t *dst_u = (uint8_t *) in_buf->data_yuv[1];
        uint8_t *dst_v = (uint8_t *) in_buf->data_yuv[2];

        for (int y = 0; y < height; ++y) {
                memcpy(dst_y + y * width, src_y + y * width, width);
        }

        for (int y = 0; y < uv_height; ++y) {
                memcpy(dst_u + y * uv_width, src_u + y * uv_width, uv_width);
        }

        for (int y = 0; y < uv_height; ++y) {
                memcpy(dst_v + y * uv_width, src_v + y * uv_width, uv_width);
        }
}

static const struct uv_to_jpegxs_conversion uv_to_jpegxs_conversions[] = {
    { UYVY, COLOUR_FORMAT_PLANAR_YUV422, uyvy_to_yuv422p },
    { YUYV, COLOUR_FORMAT_PLANAR_YUV422, yuyv_to_yuv422p },
    { I420, COLOUR_FORMAT_PLANAR_YUV420, i420_to_yuv420p },
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