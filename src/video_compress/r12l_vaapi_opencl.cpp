#define CL_TARGET_OPENCL_VERSION 120

#include "r12l_vaapi_opencl.hpp"

#include <CL/cl.h>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
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
)CLC";
}

struct r12l_vaapi_opencl::impl {
        cl_context context{};
        cl_command_queue queue{};
        cl_program program{};
        cl_kernel kernel{};
        cl_mem input{};
        cl_mem output{};
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
r12l_vaapi_opencl::init(int width, int height)
{
        cl_uint platform_count = 0;
        cl_int status = clGetPlatformIDs(0, nullptr, &platform_count);
        if (status != CL_SUCCESS || platform_count == 0) return m->fail("clGetPlatformIDs", status);
        std::vector<cl_platform_id> platforms(platform_count);
        clGetPlatformIDs(platform_count, platforms.data(), nullptr);

        cl_platform_id platform = platforms[0];
        cl_device_id cl_device{};
        status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &cl_device,
                                nullptr);
        if (status != CL_SUCCESS) return m->fail("clGetDeviceIDs", status);

        const cl_context_properties props[] = {
                CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platform),
                0
        };
        m->context = clCreateContext(props, 1, &cl_device, nullptr, nullptr, &status);
        if (status != CL_SUCCESS) return m->fail("clCreateContext", status);
        m->queue = clCreateCommandQueue(m->context, cl_device, 0, &status);
        if (status != CL_SUCCESS) return m->fail("clCreateCommandQueue", status);

        const std::size_t source_len = std::strlen(kernel_source);
        const char *source = kernel_source;
        m->program = clCreateProgramWithSource(m->context, 1, &source,
                                               &source_len, &status);
        if (status != CL_SUCCESS) return m->fail("clCreateProgramWithSource", status);
        status = clBuildProgram(m->program, 1, &cl_device, "-cl-std=CL1.2",
                                nullptr, nullptr);
        if (status != CL_SUCCESS) return m->fail("clBuildProgram", status);
        m->kernel = clCreateKernel(m->program, "r12l_to_va_rgb", &status);
        if (status != CL_SUCCESS) return m->fail("clCreateKernel", status);

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
