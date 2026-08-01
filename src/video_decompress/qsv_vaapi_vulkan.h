#ifndef VIDEO_DECOMPRESS_QSV_VAAPI_VULKAN_H
#define VIDEO_DECOMPRESS_QSV_VAAPI_VULKAN_H

#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

void *qsv_vaapi_vulkan_create(void *va_display, int width, int height);
void qsv_vaapi_vulkan_destroy(void *state);
bool qsv_vaapi_vulkan_convert(void *state, unsigned int va_surface,
                              unsigned char *destination,
                              size_t destination_stride);
const char *qsv_vaapi_vulkan_error(void *state);

#ifdef __cplusplus
}
#endif

#endif
