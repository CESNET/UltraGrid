#ifndef R12L_DECOMPRESS_OPENCL_H
#define R12L_DECOMPRESS_OPENCL_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *r12l_decompress_opencl_create(void);
void r12l_decompress_opencl_destroy(void *state);
bool r12l_decompress_opencl_convert(void *state, const unsigned char *source,
                                    size_t source_stride,
                                    unsigned char *destination,
                                    size_t destination_stride, int width,
                                    int height);
bool r10k_decompress_opencl_convert(void *state, const unsigned char *source,
                                    size_t source_stride,
                                    unsigned char *destination,
                                    size_t destination_stride, int width,
                                    int height);
const char *r12l_decompress_opencl_error(void *state);

#ifdef __cplusplus
}
#endif

#endif
