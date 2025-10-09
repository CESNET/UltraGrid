#include <svt-jpegxs/SvtJpegxs.h>
#include <string.h>

#include "jpegxs_conv.h"
#include "types.h"

static void yuv422p_to_uyvy(const svt_jpeg_xs_image_buffer_t *image, int width, int height, uint8_t *uyvy) {

        uint8_t *src_y = (uint8_t*) image->data_yuv[0];
        uint8_t *src_u = (uint8_t*) image->data_yuv[1];
        uint8_t *src_v = (uint8_t*) image->data_yuv[2];

        for (int y = 0; y < height; y++) {
                uint8_t *dst_line = uyvy + y * width * 2;
                const uint8_t *src_y_line = src_y + y * image->stride[0];
                const uint8_t *src_u_line = src_u + y * image->stride[1];
                const uint8_t *src_v_line = src_v + y * image->stride[2];

                for (int x = 0; x < width; x += 2) {
                        int chroma_index = x / 2;
                        uint8_t u = src_u_line[chroma_index];
                        uint8_t y0 = src_y_line[x];
                        uint8_t v = src_v_line[chroma_index];
                        uint8_t y1 = src_y_line[x + 1];

                        int i = x * 2;
                        dst_line[i + 0] = u;
                        dst_line[i + 1] = y0;
                        dst_line[i + 2] = v;
                        dst_line[i + 3] = y1;
                }
        }
}

static const struct jpegxs_to_uv_conversion uv_to_jpegxs_conversions[] = {
    { COLOUR_FORMAT_PLANAR_YUV422, UYVY, yuv422p_to_uyvy },
    { COLOUR_FORMAT_INVALID, VIDEO_CODEC_NONE, NULL }
};

const struct jpegxs_to_uv_conversion *get_jpegxs_to_uv_conversion(codec_t codec) {
    
    const struct jpegxs_to_uv_conversion *conv = uv_to_jpegxs_conversions;
    while (conv->dst != VIDEO_CODEC_NONE) {
        if (conv->dst == codec)
            return conv;
        conv++;
    }

    return NULL;
}

const struct jpegxs_to_uv_conversion *get_default_jpegxs_to_uv_conversion(ColourFormat_t fmt) {

    const struct jpegxs_to_uv_conversion *conv = uv_to_jpegxs_conversions;
    while (conv->dst != VIDEO_CODEC_NONE) {
        if (conv->src == fmt)
            return conv;
        conv++;
    }
    
    return NULL;
}