#ifndef FROM_OPENAPV_CONVERSIONS_H
#define FROM_OPENAPV_CONVERSIONS_H

#include "types.h"
#include "../from_planar.h"

#ifdef __cplusplus
extern "C" {
#endif

struct from_openapv_conversion {
        codec_t               dst_codec;
        int                   required_src_cs;
        decode_planar_func_t *convert;
};

const struct from_openapv_conversion *
get_from_openapv_conversion(codec_t dst_codec, int src_cs);

#ifdef __cplusplus
}
#endif

#endif // FROM_OPENAPV_CONVERSIONS_H
