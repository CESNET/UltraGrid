/**
 * @file   utils/opencl.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2025 CESNET
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * 3. Neither the name of CESNET nor the names of its contributors may be
 *    used to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "opencl.h"

#include "host.h"              // for ADD_TO_PARAM, get_commandline_param

#define MOD_NAME   "[OpenCL] "
#define PARAM_NAME "opencl-device"
#define PARAM_HELP "* opencl-device=<platf_id>-<dev_id> | <type> | help\n"\
                "  Use specified OpenCL device.\n"
ADD_TO_PARAM(PARAM_NAME, PARAM_HELP);

#ifdef HAVE_OPENCL
#include <CL/cl.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "compat/strings.h"    // for strcasecmp
#include "debug.h"             // for MSG
#include "utils/macros.h"      // for ARR_COUNT, SHORT_STR
#include "utils/color_out.h"   // for TBOLD

#define CHECK_OPENCL(cmd, errmsg, erraction) \
        if ((cmd) != CL_SUCCESS) { \
                MSG(ERROR, "%s\n", (errmsg)); \
                erraction; \
        }

// the order gives the preference of device
static const struct {
        cl_device_type cl_type;
        const char    *name;
} dev_type_map[] = {
        { CL_DEVICE_TYPE_CPU,         "CPU"         },
        { CL_DEVICE_TYPE_GPU,         "GPU"         },
        { CL_DEVICE_TYPE_ACCELERATOR, "Accelerator" },
#ifdef CL_VERSION_1_2
        { CL_DEVICE_TYPE_CUSTOM,      "custom"      },
#endif
};

static void
print_device_details(cl_device_id device)
{
        cl_uint        uint_val    = 0;
        cl_ulong       ulong_val   = 0;
        cl_device_type device_type = 0;
        char           buf[SHORT_STR];

        if (clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof device_type,
                            &device_type, NULL) == CL_SUCCESS) {
                const char *type = "UNKNOWN";
                for (unsigned i = 0; i < ARR_COUNT(dev_type_map); ++i) {
                        if (dev_type_map[i].cl_type == device_type) {
                                type = dev_type_map[i].name;
                        }
                }
                printf("    Type: %s\n", type);
        }
        if (clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof buf, buf, NULL) ==
            CL_SUCCESS) {
                printf("    Version: %s\n", buf);
        }
        if (clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS,
                            sizeof uint_val, &uint_val, NULL) == CL_SUCCESS) {
                printf("    Max compute units: %u\n", uint_val);
        }
        if (clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof ulong_val,
                            &ulong_val, NULL) == CL_SUCCESS) {
                printf("    Global memory size: %lu B\n", ulong_val);
        }
}

void
list_opencl_devices(bool full)
{
        const char record_sep = full ? '\n' : ' ';

        printf("Available OpenCL devices:\n");
        // Get the number of available platforms
        cl_uint num_platforms = 0;
        CHECK_OPENCL(clGetPlatformIDs(0, NULL, &num_platforms),
                     "Cannot get number of platforms!", return);
        cl_platform_id *platforms =
            (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
        CHECK_OPENCL(clGetPlatformIDs(num_platforms, platforms, NULL),
                     "Cannot get platforms", return);

        // Iterate through the available platforms and get the device IDs
        for (cl_uint i = 0; i < num_platforms; i++) {
                // Print the device IDs for the current platform
                printf("Platform %d: %c", i, record_sep);
                char platform_name[100];
                CHECK_OPENCL(clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME,
                                               sizeof(platform_name),
                                               platform_name, NULL),
                             "Cannot get platform name", continue);
                printf("%s%s\n", full ? "  Name: " : "", platform_name);

                // Get the number of available devices for the current platform
                cl_uint num_devices = 0;
                CHECK_OPENCL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                               &num_devices), "Cannot get platform num devices", continue);
                cl_device_id *devices =
                    (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
                CHECK_OPENCL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                                            num_devices, devices, NULL),
                             "Cannot get platform devices",
                             free((void *) devices);
                             continue);

                for (cl_uint j = 0; j < num_devices; j++) {
                        printf("  Device %d: %c", j, record_sep);
                        char device_name[100];
                        CHECK_OPENCL(clGetDeviceInfo(devices[j], CL_DEVICE_NAME,
                                                     sizeof(device_name),
                                                     device_name, NULL),
                                     "Cannot get device name", continue);
                        printf("%s%s\n", full ? "    Name: " : "",
                               device_name);
                        if (full) {
                                print_device_details(devices[j]);
                        }
                }

                free((void *) devices);
        }

        free((void *) platforms);
}

static bool
get_device(cl_device_type req_type, int req_platform_idx,
                  int req_device_idx, void **platform_id, void **device_id)
{
        // platform+device index is used from all devices, otherwise
        // first device of requested type returned
        assert(req_type == CL_DEVICE_TYPE_ALL ||
               (req_platform_idx == 0 || req_device_idx == 0));

        // Get the number of available platforms
        cl_uint num_platforms = 0;
        CHECK_OPENCL(clGetPlatformIDs(0, NULL, &num_platforms),
                     "Cannot get number of platforms!", return false);
        if (req_platform_idx >= (int) num_platforms) {
                MSG(ERROR, "Platform index %d out of bound (%u platforms)\n",
                    req_platform_idx, num_platforms);
                return false;
        }
        cl_platform_id platform = NULL;
        {
                cl_platform_id *platforms = (cl_platform_id *) malloc(
                    num_platforms * sizeof(cl_platform_id));
                CHECK_OPENCL(clGetPlatformIDs(num_platforms, platforms, NULL),
                             "Cannot get platforms", free((void *) platforms);
                             return false);
                platform = platforms[req_platform_idx];
                free((void *) platforms);
        }

        // Get the number of available devices for the current platform
        cl_uint num_devices = 0;
        CHECK_OPENCL(clGetDeviceIDs(platform, req_type, 0, NULL, &num_devices),
                     "Cannot get platform num devices", return false);
        if (req_device_idx >= (int) num_devices) {
                if (req_type == CL_DEVICE_TYPE_ALL) {
                        MSG(ERROR,
                            "Device index %d out of bound (%u devices for "
                            "platform "
                            "%u)\n",
                            req_device_idx, num_devices, num_platforms);
                }
                return false;
        }
        cl_device_id *devices =
            (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
        CHECK_OPENCL(
            clGetDeviceIDs(platform, req_type, num_devices, devices, NULL),
            "Cannot get platform devices", free((void *) devices);
            return false);

        *platform_id = platform;
        *device_id   = devices[req_device_idx];
        char   platform_name[SHORT_STR];
        char   device_name[SHORT_STR];
        cl_int platform_rc =
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof platform_name,
                              platform_name, NULL);
        cl_int name_rc =
            clGetDeviceInfo(devices[req_platform_idx], CL_DEVICE_NAME,
                            sizeof(device_name), device_name, NULL);
        MSG(INFO, "Using OpenCL platform %s, device %s\n",
            platform_rc == CL_SUCCESS ? platform_name : "UNKNOWN",
            name_rc == CL_SUCCESS ? device_name : "UNKNOWN");

        free((void *) devices);
        return true;
}

static void usage() {
        printf("Usage:\n");
        color_printf("\t--param " TBOLD(
            PARAM_NAME "=<platform_idx>-<device_idx> | gpu | cpu | "
                       "accelerator") "\n");
        color_printf("\t--param " TBOLD(PARAM_NAME "=help") "\n");
        printf("\n");
        printf("If device is specified by type, first found is "
               "selected.\n");
        list_opencl_devices(true);
}

bool
opencl_get_device(void **platform_id, void **device_id)
{
        cl_device_type req_type         = CL_DEVICE_TYPE_ALL;
        int            req_platform_idx = 0;
        int            req_device_idx   = 0;

        const char *req_device = get_commandline_param(PARAM_NAME);
        if (req_device == NULL) { // implicit - prefer any acc->GPU->CPU->other
                const cl_device_type try_type[] = { CL_DEVICE_TYPE_ACCELERATOR,
                                                    CL_DEVICE_TYPE_GPU,
                                                    CL_DEVICE_TYPE_CPU,
                                                    CL_DEVICE_TYPE_ALL };
                for (unsigned i = 0; i < ARR_COUNT(try_type); i++) {
                        if (get_device(try_type[i], 0, 0, platform_id,
                                       device_id)) {
                                return true;
                        }
                }
                return false;
        }

        if (strcmp(req_device, "help") == 0) {
                usage();
                return false;
        }
        if (isdigit(req_device[0])) {
                req_platform_idx = atoi(req_device);
                if (strchr(req_device, '-') != NULL) {
                        req_device_idx = atoi(strchr(req_device, '-') + 1);
                }
        } else {
                for (unsigned i = 0; i < ARR_COUNT(dev_type_map); ++i) {
                        if (strcasecmp(dev_type_map[i].name, req_device) == 0) {
                                req_type = dev_type_map[i].cl_type;
                        }
                }
                if (req_type) {
                        MSG(ERROR, "unknown device type: %s!\n", req_device);
                        return false;
                }
        }
        if (req_type == CL_DEVICE_TYPE_ALL) { // by index
                return get_device(CL_DEVICE_TYPE_ALL, req_platform_idx,
                                  req_device_idx, platform_id, device_id);
        }
        // by type
        if (get_device(req_type, 0, 0, platform_id, device_id)) {
                return true;
        }
        MSG(ERROR, "Cannot find device of requested type %s!\n", req_device);
        return false;
}

#else // !defined HAVE_OPENCL
#include "debug.h"             // for MSG
void
list_opencl_devices(bool full)
{
        (void) full;
        MSG(ERROR, "OpenCL support not compiled in!\n");
}
bool
opencl_get_device(void **platform_id, void **device_id)
{
        (void) platform_id, (void) device_id;
        MSG(ERROR, "OpenCL support not compiled in!\n");
        return false;
}
#endif // defined HAVE_OPENCL
