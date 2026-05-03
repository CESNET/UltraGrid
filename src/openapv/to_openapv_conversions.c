#include "openapv_conversions.h"
#include "../color_space.h"
#include "../video_codec.h"

static void uyvy_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const int y_stride_px = dst->s[0] / (int) sizeof(uint16_t);
        const int u_stride_px = dst->s[1] / (int) sizeof(uint16_t);
        const int v_stride_px = dst->s[2] / (int) sizeof(uint16_t);

        for (int y = 0; y < height; ++y) {
                uint16_t *dst_y = (uint16_t *) dst->a[0] + y * y_stride_px;
                uint16_t *dst_u = (uint16_t *) dst->a[1] + y * u_stride_px;
                uint16_t *dst_v = (uint16_t *) dst->a[2] + y * v_stride_px;

                for (int x = 0; x < width; x += 2) {
                        *dst_u++ = *src++ << 2;
                        *dst_y++ = *src++ << 2;
                        *dst_v++ = *src++ << 2;
                        *dst_y++ = *src++ << 2;
                }
        }
}

static void yuyv_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const int y_stride_px = dst->s[0] / (int) sizeof(uint16_t);
        const int u_stride_px = dst->s[1] / (int) sizeof(uint16_t);
        const int v_stride_px = dst->s[2] / (int) sizeof(uint16_t);
        
        for (int y = 0; y < height; ++y) {
                uint16_t *dst_y = (uint16_t *) dst->a[0] + y * y_stride_px;
                uint16_t *dst_u = (uint16_t *) dst->a[1] + y * u_stride_px;
                uint16_t *dst_v = (uint16_t *) dst->a[2] + y * v_stride_px;

                for (int x = 0; x < width; x += 2) {
                        *dst_y++ = *src++ << 2;
                        *dst_u++ = *src++ << 2;
                        *dst_y++ = *src++ << 2;
                        *dst_v++ = *src++ << 2;
                }
        }
}

static void v210_to_yuv422p(const uint8_t *src, int width, int height, oapv_imgb_t *dst) {
        const int y_stride_px = dst->s[0] / (int) sizeof(uint16_t);
        const int u_stride_px = dst->s[1] / (int) sizeof(uint16_t);
        const int v_stride_px = dst->s[2] / (int) sizeof(uint16_t);

        for (int y = 0; y < height; ++y) {
                const uint32_t *src_row = (const uint32_t *)(src + y * vc_get_linesize(width, v210));
                uint16_t *dst_y = (uint16_t *) dst->a[0] + y * y_stride_px;
                uint16_t *dst_u = (uint16_t *) dst->a[1] + y * u_stride_px;
                uint16_t *dst_v = (uint16_t *) dst->a[2] + y * v_stride_px;

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

static const struct uv_to_openapv_conversion uv_to_openapv_conversions[] = {
        { UYVY, OAPV_CS_YCBCR422_10LE, uyvy_to_yuv422p },
        { YUYV, OAPV_CS_YCBCR422_10LE, yuyv_to_yuv422p },
        { v210, OAPV_CS_YCBCR422_10LE, v210_to_yuv422p },
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