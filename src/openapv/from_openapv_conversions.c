#include "from_openapv_conversions.h"

#include "../from_planar.h"
#include "../types.h"

#include <oapv/oapv.h>
#include <stdint.h>

static void yuv4444p10le_to_vuya(const struct from_planar_data d)
{
        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_y = (const void *) (d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_u = (const void *) (d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_v = (const void *) (d.in_data[2] + d.in_linesize[2] * y);
                const uint16_t *src_a = (const void *) (d.in_data[3] + d.in_linesize[3] * y);
                uint8_t *dst = d.out_data + (size_t) y * d.out_pitch;

                const int width = d.width;
                for (int x = 0; x < width; ++x) {
                        *dst++ = (uint8_t) (*src_v++ >> 2); // V
                        *dst++ = (uint8_t) (*src_u++ >> 2); // U
                        *dst++ = (uint8_t) (*src_y++ >> 2); // Y
                        *dst++ = (uint8_t) (*src_a++ >> 2); // A
                }
        }
}

static void
yuv444p10le_to_y416(const struct from_planar_data d)
{
        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_y = (const void *) (d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_u = (const void *) (d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_v = (const void *) (d.in_data[2] + d.in_linesize[2] * y);
                uint16_t *dst = (void *) (d.out_data + (size_t) y * d.out_pitch);

                const int width = d.width;
                for (int x = 0; x < width; ++x) {
                        *dst++ = (uint16_t) (*src_u++ << 6); // Cb
                        *dst++ = (uint16_t) (*src_y++ << 6); // Y
                        *dst++ = (uint16_t) (*src_v++ << 6); // Cr
                        *dst++ = 0xFFFFU;                     // A
                }
        }
}

static void
yuv4444p10le_to_y416(const struct from_planar_data d)
{
        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_y = (const void *) (d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_u = (const void *) (d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_v = (const void *) (d.in_data[2] + d.in_linesize[2] * y);
                const uint16_t *src_a = (const void *) (d.in_data[3] + d.in_linesize[3] * y);
                uint16_t *dst = (void *) (d.out_data + (size_t) y * d.out_pitch);

                const int width = d.width;
                for (int x = 0; x < width; ++x) {
                        *dst++ = (uint16_t) (*src_u++ << 6); // Cb
                        *dst++ = (uint16_t) (*src_y++ << 6); // Y
                        *dst++ = (uint16_t) (*src_v++ << 6); // Cr
                        *dst++ = (uint16_t) (*src_a++ << 6); // A
                }
        }
}

static void
yuv444p12le_to_y416(const struct from_planar_data d)
{
        for (int y = 0; y < d.height; ++y) {
                const uint16_t *src_y = (const void *) (d.in_data[0] + d.in_linesize[0] * y);
                const uint16_t *src_u = (const void *) (d.in_data[1] + d.in_linesize[1] * y);
                const uint16_t *src_v = (const void *) (d.in_data[2] + d.in_linesize[2] * y);
                uint16_t *dst = (void *) (d.out_data + (size_t) y * d.out_pitch);

                const int width = d.width;
                for (int x = 0; x < width; ++x) {
                        *dst++ = (uint16_t) (*src_u++ << 4); // Cb
                        *dst++ = (uint16_t) (*src_y++ << 4); // Y
                        *dst++ = (uint16_t) (*src_v++ << 4); // Cr
                        *dst++ = 0xFFFFU;                     // A
                }
        }
}

static const struct from_openapv_conversion from_openapv_conversions[] = {
        { UYVY, OAPV_CS_YCBCR422_10LE,  yuv422pXX_to_uyvy     },
        { YUYV, OAPV_CS_YCBCR422_10LE,  yuv422p_to_yuyv       },
        { v210, OAPV_CS_YCBCR422_10LE,  yuv422p10le_to_v210   },
        { Y416, OAPV_CS_YCBCR444_10LE,  yuv444p10le_to_y416   },
        { VUYA, OAPV_CS_YCBCR4444_10LE, yuv4444p10le_to_vuya  },
        { Y416, OAPV_CS_YCBCR4444_10LE, yuv4444p10le_to_y416  },
        { Y416, OAPV_CS_YCBCR444_12LE,  yuv444p12le_to_y416   },
        { VIDEO_CODEC_NONE, OAPV_CS_UNKNOWN, NULL              },
};

const struct from_openapv_conversion *
get_from_openapv_conversion(codec_t dst_codec, int src_cs)
{
        for (const struct from_openapv_conversion *c = from_openapv_conversions;
             c->dst_codec != VIDEO_CODEC_NONE; ++c) {
                if (c->dst_codec == dst_codec && c->required_src_cs == src_cs) {
                        return c;
                }
        }
        return NULL;
}
