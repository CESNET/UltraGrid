#define CL_TARGET_OPENCL_VERSION 120

#include "r12l_vaapi_opencl.hpp"

#include <CL/cl.h>
#include <CL/cl_va_api_media_sharing_intel.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
constexpr cl_channel_type cl_unorm_int_101010_2 = 0x10E0;
constexpr const char *kernel_source = R"CLC(
inline uint lo12(__global const uchar *p) {
        return (uint) p[0] | (((uint) p[1] & 15u) << 8);
}
inline uint hi12(__global const uchar *p) {
        return ((uint) p[1] >> 4) | ((uint) p[2] << 4);
}
inline float s12(uint v) {
        uint ten = (v * 1023u + 2047u) / 4095u;
        return convert_float(ten) * (1.0f / 1023.0f);
}
__kernel void r12l_to_va_rgb(__global const uchar *src,
                             __global uint *dst, ulong pairs) {
        ulong i = get_global_id(0);
        if (i >= pairs) return;
        __global const uchar *p = src + i * 9;
        uint r0 = lo12(p), g0 = hi12(p), b0 = lo12(p + 3);
        uint r1 = hi12(p + 3), g1 = lo12(p + 6), b1 = hi12(p + 6);
        uint ir0 = (r0 * 1023u + 2047u) / 4095u;
        uint ig0 = (g0 * 1023u + 2047u) / 4095u;
        uint ib0 = (b0 * 1023u + 2047u) / 4095u;
        uint ir1 = (r1 * 1023u + 2047u) / 4095u;
        uint ig1 = (g1 * 1023u + 2047u) / 4095u;
        uint ib1 = (b1 * 1023u + 2047u) / 4095u;
        dst[i * 2] = ib0 | (ig0 << 10) | (ir0 << 20) | (3u << 30);
        dst[i * 2 + 1] = ib1 | (ig1 << 10) | (ir1 << 20) | (3u << 30);
}
__kernel void r12l_to_va_image(__global const uchar *src,
                               write_only image2d_t dst, int width) {
        int x = get_global_id(0), y = get_global_id(1);
        if (x >= width) return;
        ulong pair = ((ulong)y * (ulong)width + (ulong)x) >> 1;
        __global const uchar *p = src + pair * 9;
        uint r, g, b;
        if ((x & 1) == 0) {
                r = lo12(p); g = hi12(p); b = lo12(p + 3);
        } else {
                r = hi12(p + 3); g = lo12(p + 6); b = hi12(p + 6);
        }
        float scale = 1.0f / 4095.0f;
        // The imported packed surface is RGBA 10:10:10:2. Its low component
        // carries U/B, followed by Y/G and V/R.
        write_imagef(dst, (int2)(x, y),
                     (float4)(b * scale, g * scale, r * scale, 1.0f));
}
)CLC";
}

struct r12l_vaapi_opencl::impl {
        cl_context context{};
        cl_command_queue queue{};
        cl_program program{};
        cl_kernel kernel{};
        cl_kernel va_kernel{};
        cl_mem input{};
        cl_mem output{};
        cl_platform_id platform{};
        cl_device_id device{};
        void *va_display{};
        clCreateFromVA_APIMediaSurfaceINTEL_fn create_va_surface{};
        clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn acquire_va_surfaces{};
        clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn release_va_surfaces{};
        bool va_sharing{};
        int width{};
        int height{};
        std::size_t row{};
        std::size_t pairs{};
        std::vector<unsigned char> staging;
        std::vector<unsigned char> staging_output;
        std::string error;

        ~impl() {
                if (output) clReleaseMemObject(output);
                if (input) clReleaseMemObject(input);
                if (va_kernel) clReleaseKernel(va_kernel);
                if (kernel) clReleaseKernel(kernel);
                if (program) clReleaseProgram(program);
                if (queue) clReleaseCommandQueue(queue);
                if (context) clReleaseContext(context);
        }

        bool fail(const char *what, cl_int status) {
                error = std::string(what) + " failed (" + std::to_string(status) + ")";
                return false;
        }
};

r12l_vaapi_opencl::r12l_vaapi_opencl() : m(std::make_unique<impl>()) {}
r12l_vaapi_opencl::~r12l_vaapi_opencl() = default;
const std::string &r12l_vaapi_opencl::error() const { return m->error; }

bool
r12l_vaapi_opencl::init(int width, int height, void *va_display)
{
        cl_uint platform_count = 0;
        cl_int status = clGetPlatformIDs(0, nullptr, &platform_count);
        if (status != CL_SUCCESS || platform_count == 0) return m->fail("clGetPlatformIDs", status);
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);

        m->platform = platforms[0];
        m->va_display = va_display;
        status = clGetDeviceIDs(m->platform, CL_DEVICE_TYPE_GPU, 1, &m->device,
                                nullptr);
        if (status != CL_SUCCESS) return m->fail("clGetDeviceIDs", status);

        const cl_context_properties props[] = {
                CL_CONTEXT_PLATFORM,
                reinterpret_cast<cl_context_properties>(m->platform),
                CL_CONTEXT_VA_API_DISPLAY_INTEL,
                reinterpret_cast<cl_context_properties>(va_display), 0};
        const cl_context_properties plain_props[] = {
                CL_CONTEXT_PLATFORM,
                reinterpret_cast<cl_context_properties>(m->platform), 0};
        m->context = clCreateContext(va_display ? props : plain_props, 1,
                                     &m->device, nullptr, nullptr, &status);
        if (status != CL_SUCCESS && va_display) {
                m->context = clCreateContext(plain_props, 1, &m->device,
                                             nullptr, nullptr, &status);
        }
        if (status != CL_SUCCESS) return m->fail("clCreateContext", status);
        m->queue = clCreateCommandQueue(m->context, m->device, 0, &status);
        if (status != CL_SUCCESS) return m->fail("clCreateCommandQueue", status);

        const std::size_t source_len = std::strlen(kernel_source);
        const char *source = kernel_source;
        m->program = clCreateProgramWithSource(m->context, 1, &source,
                                               &source_len, &status);
        if (status != CL_SUCCESS) return m->fail("clCreateProgramWithSource", status);
        status = clBuildProgram(m->program, 1, &m->device, "-cl-std=CL1.2",
                                nullptr, nullptr);
        if (status != CL_SUCCESS) return m->fail("clBuildProgram", status);
        m->kernel = clCreateKernel(m->program, "r12l_to_va_rgb", &status);
        if (status != CL_SUCCESS) return m->fail("clCreateKernel", status);
        m->va_kernel =
                clCreateKernel(m->program, "r12l_to_va_image", &status);
        if (status != CL_SUCCESS) return m->fail("clCreateKernel VA", status);

        if (va_display) {
                m->create_va_surface =
                    reinterpret_cast<clCreateFromVA_APIMediaSurfaceINTEL_fn>(
                        clGetExtensionFunctionAddressForPlatform(
                            m->platform,
                            "clCreateFromVA_APIMediaSurfaceINTEL"));
                m->acquire_va_surfaces =
                    reinterpret_cast<clEnqueueAcquireVA_APIMediaSurfacesINTEL_fn>(
                        clGetExtensionFunctionAddressForPlatform(
                            m->platform,
                            "clEnqueueAcquireVA_APIMediaSurfacesINTEL"));
                m->release_va_surfaces =
                    reinterpret_cast<clEnqueueReleaseVA_APIMediaSurfacesINTEL_fn>(
                        clGetExtensionFunctionAddressForPlatform(
                            m->platform,
                            "clEnqueueReleaseVA_APIMediaSurfacesINTEL"));
                m->va_sharing = m->create_va_surface &&
                                m->acquire_va_surfaces &&
                                m->release_va_surfaces;
        }

        m->width = width;
        m->height = height;
        m->row = static_cast<std::size_t>(width) * 9U / 2U;
        m->pairs = static_cast<std::size_t>(width) * height / 2U;
        m->input = clCreateBuffer(m->context, CL_MEM_READ_ONLY,
                                  m->row * height, nullptr, &status);
        if (status != CL_SUCCESS) return m->fail("clCreateBuffer", status);
        m->output = clCreateBuffer(m->context, CL_MEM_WRITE_ONLY,
                                   static_cast<std::size_t>(width) * 4U * height,
                                   nullptr, &status);
        if (status != CL_SUCCESS) return m->fail("clCreateBuffer output", status);
        const cl_ulong pairs = m->pairs;
        status = clSetKernelArg(m->kernel, 0, sizeof(m->input), &m->input);
        status |= clSetKernelArg(m->kernel, 1, sizeof(m->output), &m->output);
        status |= clSetKernelArg(m->kernel, 2, sizeof(pairs), &pairs);
        return status == CL_SUCCESS || m->fail("clSetKernelArg", status);
}

bool
r12l_vaapi_opencl::convert(const unsigned char *source,
                           std::size_t source_stride, unsigned char *destination,
                           std::size_t destination_stride)
{
        if (source_stride != m->row) {
                m->staging.resize(m->row * m->height);
                for (int y = 0; y < m->height; ++y)
                        std::memcpy(m->staging.data() + y * m->row,
                                    source + y * source_stride, m->row);
                source = m->staging.data();
        }
        cl_int status = clEnqueueWriteBuffer(m->queue, m->input, CL_FALSE, 0,
                                             m->row * m->height, source, 0,
                                             nullptr, nullptr);
        if (status != CL_SUCCESS) return m->fail("R12L GPU upload", status);
        const std::size_t local = 256;
        const std::size_t global = (m->pairs + local - 1) / local * local;
        status = clEnqueueNDRangeKernel(m->queue, m->kernel, 1, nullptr, &global,
                                        &local, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) return m->fail("R12L GPU kernel", status);
        const std::size_t output_row = static_cast<std::size_t>(m->width) * 4U;
        unsigned char *download = destination;
        if (destination_stride != output_row) {
                m->staging_output.resize(output_row * m->height);
                download = m->staging_output.data();
        }
        status = clEnqueueReadBuffer(m->queue, m->output, CL_TRUE, 0,
                                     output_row * m->height, download, 0,
                                     nullptr, nullptr);
        if (status != CL_SUCCESS) return m->fail("RGB10 GPU download", status);
        if (download != destination)
                for (int y = 0; y < m->height; ++y)
                        std::memcpy(destination + y * destination_stride,
                                    download + y * output_row, output_row);
        return true;
}

bool
r12l_vaapi_opencl::va_surface_sharing_available() const
{
        return m->va_sharing;
}

bool
r12l_vaapi_opencl::convert_to_va_surface(const unsigned char *source,
                                         std::size_t source_stride,
                                         unsigned int va_surface)
{
        if (!m->va_sharing) {
                m->error = "OpenCL/VA-API surface sharing is unavailable";
                return false;
        }
        if (source_stride != m->row) {
                m->staging.resize(m->row * m->height);
                for (int y = 0; y < m->height; ++y)
                        std::memcpy(m->staging.data() + y * m->row,
                                    source + y * source_stride, m->row);
                source = m->staging.data();
        }
        cl_int status = clEnqueueWriteBuffer(m->queue, m->input, CL_FALSE, 0,
                                             m->row * m->height, source, 0,
                                             nullptr, nullptr);
        if (status != CL_SUCCESS) return m->fail("R12L GPU upload", status);
        VASurfaceID surface = va_surface;
        cl_mem image = m->create_va_surface(m->context, CL_MEM_WRITE_ONLY,
                                            &surface, 0, &status);
        if (status != CL_SUCCESS || !image)
                return m->fail("clCreateFromVA_APIMediaSurfaceINTEL", status);
        cl_image_format format{};
        status = clGetImageInfo(image, CL_IMAGE_FORMAT, sizeof format, &format,
                                nullptr);
        if (status != CL_SUCCESS ||
            format.image_channel_order != CL_RGBA ||
            (format.image_channel_data_type != cl_unorm_int_101010_2 &&
             format.image_channel_data_type != CL_UNORM_INT_101010)) {
                clReleaseMemObject(image);
                m->va_sharing = false;
                m->error = "VA Y410 surface is not exposed as writable RGBA10";
                return false;
        }
        status = m->acquire_va_surfaces(m->queue, 1, &image, 0, nullptr,
                                        nullptr);
        const int width = m->width;
        if (status == CL_SUCCESS)
                status = clSetKernelArg(m->va_kernel, 0, sizeof(m->input),
                                        &m->input);
        if (status == CL_SUCCESS)
                status = clSetKernelArg(m->va_kernel, 1, sizeof(image), &image);
        if (status == CL_SUCCESS)
                status = clSetKernelArg(m->va_kernel, 2, sizeof(width), &width);
        const std::size_t local[] = {16, 16};
        const std::size_t global[] = {
            (static_cast<std::size_t>(m->width) + 15U) & ~15U,
            (static_cast<std::size_t>(m->height) + 15U) & ~15U};
        if (status == CL_SUCCESS)
                status = clEnqueueNDRangeKernel(m->queue, m->va_kernel, 2,
                                                nullptr, global, local, 0,
                                                nullptr, nullptr);
        if (status == CL_SUCCESS)
                status = m->release_va_surfaces(m->queue, 1, &image, 0,
                                                nullptr, nullptr);
        if (status == CL_SUCCESS) status = clFinish(m->queue);
        clReleaseMemObject(image);
        return status == CL_SUCCESS ||
               m->fail("OpenCL VA surface conversion", status);
}
