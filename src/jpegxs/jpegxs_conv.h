#ifndef JPEGXS_CONV_H
#define JPEGXS_CONV_H

#include <svt-jpegxs/SvtJpegxs.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif

struct uv_to_jpegxs_conversion {
    codec_t src;
    ColourFormat_t dst;
    void (*convert)(const uint8_t *src, int width, int height, svt_jpeg_xs_image_buffer_t *dst);
};

struct jpegxs_to_uv_conversion {
    ColourFormat_t src;
    codec_t dst;
    void (*convert)(const svt_jpeg_xs_image_buffer_t *src, int width, int height, uint8_t *dst);
};

const struct uv_to_jpegxs_conversion *get_uv_to_jpegxs_conversion(codec_t codec);

const struct jpegxs_to_uv_conversion *get_jpegxs_to_uv_conversion(codec_t codec);
const struct jpegxs_to_uv_conversion *get_default_jpegxs_to_uv_conversion(ColourFormat_t fmt);

#ifdef __cplusplus
}
#endif

#endif // JPEGXS_CONV_H