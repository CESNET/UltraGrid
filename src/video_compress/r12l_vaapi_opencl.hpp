#ifndef R12L_VAAPI_OPENCL_HPP
#define R12L_VAAPI_OPENCL_HPP

#include <cstddef>
#include <memory>
#include <string>

struct AVBufferRef;
struct AVFrame;

class r12l_vaapi_opencl {
public:
        r12l_vaapi_opencl();
        ~r12l_vaapi_opencl();
        r12l_vaapi_opencl(const r12l_vaapi_opencl &) = delete;
        r12l_vaapi_opencl &operator=(const r12l_vaapi_opencl &) = delete;

        bool init(int width, int height);
        bool convert(const unsigned char *input, std::size_t input_stride,
                     unsigned char *output, std::size_t output_stride);
        const std::string &error() const;

private:
        struct impl;
        std::unique_ptr<impl> m;
};

#endif
