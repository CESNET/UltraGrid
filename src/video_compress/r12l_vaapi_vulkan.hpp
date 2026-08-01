#ifndef VIDEO_COMPRESS_R12L_VAAPI_VULKAN_HPP
#define VIDEO_COMPRESS_R12L_VAAPI_VULKAN_HPP

#include <cstddef>
#include <memory>
#include <string>

class r12l_vaapi_vulkan {
public:
        r12l_vaapi_vulkan();
        ~r12l_vaapi_vulkan();
        r12l_vaapi_vulkan(const r12l_vaapi_vulkan &) = delete;
        r12l_vaapi_vulkan &operator=(const r12l_vaapi_vulkan &) = delete;

        bool init(int width, int height, void *va_display);
        bool convert(const unsigned char *source, std::size_t source_stride,
                     unsigned int va_surface);
        const std::string &error() const;

private:
        struct impl;
        std::unique_ptr<impl> m;
};

#endif
