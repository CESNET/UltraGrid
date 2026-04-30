#include <oapv/oapv.h>

#include "types.h"

#ifdef __cplusplus
extern "C" {
#endif


struct uv_to_openapv_conversion {
        codec_t src_color_format;
        int dst_color_format;
        void (*convert)(const uint8_t *src, int width, int height, oapv_imgb_t *dst);
};
const struct uv_to_openapv_conversion *get_uv_to_openapv_conversion(codec_t codec);

#ifdef __cplusplus
}
#endif