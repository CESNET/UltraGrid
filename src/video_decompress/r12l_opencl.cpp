#define CL_TARGET_OPENCL_VERSION 120

#include "video_decompress/r12l_opencl.h"

#include <CL/cl.h>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr const char *kernel_source = R"CLC(
inline uint s10to12(uint x) { return (x * 4095u + 511u) / 1023u; }
inline void put12(__global uchar *p, uint a, uint b) {
  p[0]=(uchar)a; p[1]=(uchar)((a>>8)|(b<<4)); p[2]=(uchar)(b>>4);
}
__kernel void xv30_to_r12l(__global const uint *src,
                           __global uchar *dst, ulong pairs) {
  ulong i=get_global_id(0); if (i>=pairs) return;
  uint a=src[i*2], b=src[i*2+1]; __global uchar *p=dst+i*9;
  put12(p,s10to12((a>>20)&1023u),s10to12((a>>10)&1023u));
  put12(p+3,s10to12(a&1023u),s10to12((b>>20)&1023u));
  put12(p+6,s10to12((b>>10)&1023u),s10to12(b&1023u));
}
__kernel void xv30_to_r10k(__global const uint *src,
                           __global uchar *dst, ulong pixels) {
  ulong i=get_global_id(0); if (i>=pixels) return;
  uint v=src[i];
  uint r=(v>>20)&1023u, g=(v>>10)&1023u, b=v&1023u;
  __global uchar *p=dst+i*4;
  p[0]=(uchar)(r>>2);
  p[1]=(uchar)((r<<6)|(g>>4));
  p[2]=(uchar)((g<<4)|(b>>6));
  p[3]=(uchar)((b<<2)|3u);
}
)CLC";

struct converter {
        cl_context context{};
        cl_command_queue queue{};
        cl_program program{};
        cl_kernel kernel{};
        cl_kernel r10k_kernel{};
        cl_mem input{};
        cl_mem output{};
        int width{};
        int height{};
        bool configured_r10k{};
        size_t pairs{};
        std::vector<unsigned char> staging_input;
        std::vector<unsigned char> staging_output;
        std::string error;

        ~converter() {
                if (output) clReleaseMemObject(output);
                if (input) clReleaseMemObject(input);
                if (r10k_kernel) clReleaseKernel(r10k_kernel);
                if (kernel) clReleaseKernel(kernel);
                if (program) clReleaseProgram(program);
                if (queue) clReleaseCommandQueue(queue);
                if (context) clReleaseContext(context);
        }
        bool fail(const char *operation, cl_int status) {
                error = std::string(operation) + " failed (" +
                        std::to_string(status) + ")";
                return false;
        }
        bool initialize() {
                cl_uint count = 0;
                cl_int status = clGetPlatformIDs(0, nullptr, &count);
                if (status != CL_SUCCESS || count == 0)
                        return fail("clGetPlatformIDs", status);
                std::vector<cl_platform_id> platforms(count);
                clGetPlatformIDs(count, platforms.data(), nullptr);
                cl_device_id device{};
                status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1,
                                        &device, nullptr);
                if (status != CL_SUCCESS)
                        return fail("clGetDeviceIDs", status);
                context = clCreateContext(nullptr, 1, &device, nullptr, nullptr,
                                          &status);
                if (status != CL_SUCCESS) return fail("clCreateContext", status);
                queue = clCreateCommandQueue(context, device, 0, &status);
                if (status != CL_SUCCESS)
                        return fail("clCreateCommandQueue", status);
                const size_t length = std::strlen(kernel_source);
                const char *source = kernel_source;
                program = clCreateProgramWithSource(context, 1, &source,
                                                    &length, &status);
                if (status != CL_SUCCESS)
                        return fail("clCreateProgramWithSource", status);
                status = clBuildProgram(program, 1, &device, nullptr, nullptr,
                                        nullptr);
                if (status != CL_SUCCESS) return fail("clBuildProgram", status);
                kernel = clCreateKernel(program, "xv30_to_r12l", &status);
                if (status != CL_SUCCESS)
                        return fail("clCreateKernel R12L", status);
                r10k_kernel =
                        clCreateKernel(program, "xv30_to_r10k", &status);
                return status == CL_SUCCESS ||
                       fail("clCreateKernel R10k", status);
        }
        bool configure(int new_width, int new_height, bool r10k) {
                if (width == new_width && height == new_height && input &&
                    configured_r10k == r10k)
                        return true;
                if (output) clReleaseMemObject(output);
                if (input) clReleaseMemObject(input);
                output = input = nullptr;
                width = new_width;
                height = new_height;
                configured_r10k = r10k;
                pairs = static_cast<size_t>(width) * height / 2U;
                cl_int status = CL_SUCCESS;
                input = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                       static_cast<size_t>(width) * 4U * height,
                                       nullptr, &status);
                if (status != CL_SUCCESS) return fail("clCreateBuffer input", status);
                output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                        static_cast<size_t>(width) *
                                                (r10k ? 4U : 9U) * height /
                                                (r10k ? 1U : 2U),
                                        nullptr, &status);
                if (status != CL_SUCCESS) return fail("clCreateBuffer output", status);
                const cl_ulong pair_count = pairs;
                cl_kernel selected = r10k ? r10k_kernel : kernel;
                const cl_ulong work_items =
                        r10k ? static_cast<cl_ulong>(width) * height
                             : pair_count;
                status = clSetKernelArg(selected, 0, sizeof(input), &input);
                status |= clSetKernelArg(selected, 1, sizeof(output), &output);
                status |= clSetKernelArg(selected, 2, sizeof(work_items),
                                         &work_items);
                return status == CL_SUCCESS ||
                       fail("clSetKernelArg", status);
        }
};
}

extern "C" void *
r12l_decompress_opencl_create(void)
{
        auto *state = new converter;
        if (!state->initialize()) {
                delete state;
                return nullptr;
        }
        return state;
}

extern "C" void
r12l_decompress_opencl_destroy(void *state)
{
        delete static_cast<converter *>(state);
}

extern "C" const char *
r12l_decompress_opencl_error(void *state)
{
        return static_cast<converter *>(state)->error.c_str();
}

extern "C" bool
r12l_decompress_opencl_convert(void *opaque, const unsigned char *source,
                               size_t source_stride,
                               unsigned char *destination,
                               size_t destination_stride, int width, int height)
{
        auto *state = static_cast<converter *>(opaque);
        if (!state->configure(width, height, false)) return false;
        const size_t input_row = static_cast<size_t>(width) * 4U;
        if (source_stride != input_row) {
                state->staging_input.resize(input_row * height);
                for (int y = 0; y < height; ++y)
                        std::memcpy(state->staging_input.data() + y * input_row,
                                    source + y * source_stride, input_row);
                source = state->staging_input.data();
        }
        cl_int status = clEnqueueWriteBuffer(
                state->queue, state->input, CL_FALSE, 0, input_row * height,
                source, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) return state->fail("XV30 GPU upload", status);
        const size_t local = 256;
        const size_t global =
                (state->pairs + local - 1U) / local * local;
        status = clEnqueueNDRangeKernel(state->queue, state->kernel, 1, nullptr,
                                        &global, &local, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) return state->fail("XV30 GPU kernel", status);
        const size_t output_row = static_cast<size_t>(width) * 9U / 2U;
        unsigned char *download = destination;
        if (destination_stride != output_row) {
                state->staging_output.resize(output_row * height);
                download = state->staging_output.data();
        }
        status = clEnqueueReadBuffer(state->queue, state->output, CL_TRUE, 0,
                                     output_row * height, download, 0, nullptr,
                                     nullptr);
        if (status != CL_SUCCESS)
                return state->fail("R12L GPU download", status);
        if (download != destination)
                for (int y = 0; y < height; ++y)
                        std::memcpy(destination + y * destination_stride,
                                    download + y * output_row, output_row);
        return true;
}

extern "C" bool
r10k_decompress_opencl_convert(void *opaque, const unsigned char *source,
                               size_t source_stride,
                               unsigned char *destination,
                               size_t destination_stride, int width, int height)
{
        auto *state = static_cast<converter *>(opaque);
        if (!state->configure(width, height, true)) return false;
        const size_t row_bytes = static_cast<size_t>(width) * 4U;
        if (source_stride != row_bytes) {
                state->staging_input.resize(row_bytes * height);
                for (int y = 0; y < height; ++y)
                        std::memcpy(state->staging_input.data() + y * row_bytes,
                                    source + y * source_stride, row_bytes);
                source = state->staging_input.data();
        }
        cl_int status = clEnqueueWriteBuffer(
                state->queue, state->input, CL_FALSE, 0, row_bytes * height,
                source, 0, nullptr, nullptr);
        if (status != CL_SUCCESS) return state->fail("XV30 GPU upload", status);
        const size_t pixels = static_cast<size_t>(width) * height;
        const size_t local = 256;
        const size_t global = (pixels + local - 1U) / local * local;
        status = clEnqueueNDRangeKernel(state->queue, state->r10k_kernel, 1,
                                        nullptr, &global, &local, 0, nullptr,
                                        nullptr);
        if (status != CL_SUCCESS)
                return state->fail("XV30 to R10k GPU kernel", status);
        unsigned char *download = destination;
        if (destination_stride != row_bytes) {
                state->staging_output.resize(row_bytes * height);
                download = state->staging_output.data();
        }
        status = clEnqueueReadBuffer(state->queue, state->output, CL_TRUE, 0,
                                     row_bytes * height, download, 0, nullptr,
                                     nullptr);
        if (status != CL_SUCCESS)
                return state->fail("R10k GPU download", status);
        if (download != destination)
                for (int y = 0; y < height; ++y)
                        std::memcpy(destination + y * destination_stride,
                                    download + y * row_bytes, row_bytes);
        return true;
}
