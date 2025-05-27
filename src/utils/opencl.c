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

#include <stdio.h>
#include <string.h>

#include "debug.h" // for MSG
#include "host.h"  // for get_commandline_param

#define MOD_NAME   "[OpenCL] "
#define PARAM_NAME "opencl-device"
#define PARAM_HELP "* opencl-device=<platform_id>-<device_id>|help\n"\
                "  Use specified OpenCL device.\n"
ADD_TO_PARAM(PARAM_NAME, PARAM_HELP);

#ifdef HAVE_OPENCL
#include <CL/cl.h>
static void
list_opencl_devices()
{
        // Get the number of available platforms
        cl_uint num_platforms = 0;
        clGetPlatformIDs(0, NULL, &num_platforms);
        cl_platform_id *platforms =
            (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
        clGetPlatformIDs(num_platforms, platforms, NULL);

        // Iterate through the available platforms and get the device IDs
        for (cl_uint i = 0; i < num_platforms; i++) {
                // Get the number of available devices for the current platform
                cl_uint num_devices = 0;
                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL,
                               &num_devices);
                cl_device_id *devices =
                    (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
                clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices,
                               devices, NULL);

                // Print the device IDs for the current platform
                printf("Platform %d:\n", i);
                for (cl_uint j = 0; j < num_devices; j++) {
                        printf("  Device %d: %p\n", j, devices[j]);
                }

                free((void *) devices);
        }

        free((void *) platforms);
}

bool
opencl_get_device(void **platform_id, void **device_id)
{
        int req_platform_idx = 0;
        int req_device_idx   = 0;

        const char *req_device = get_commandline_param(PARAM_NAME);
        if (req_device != NULL) {
                if (strcmp(req_device, "help") == 0) {
                        list_opencl_devices();
                        return false;
                }
                req_platform_idx = atoi(req_device);
                if (strchr(req_device, '-') != NULL) {
                        req_device_idx = atoi(strchr(req_device, '-') + 1);
                }
        }

        // Get the number of available platforms
        cl_uint num_platforms = 0;
        clGetPlatformIDs(0, NULL, &num_platforms);
        if (req_platform_idx >= (int) num_platforms) {
                MSG(ERROR, "Platform index %d out of bound (%u platforms)\n",
                    req_platform_idx, num_platforms);
                return false;
        }
        cl_platform_id *platforms =
            (cl_platform_id *) malloc(num_platforms * sizeof(cl_platform_id));
        clGetPlatformIDs(num_platforms, platforms, NULL);

        // Get the number of available devices for the current platform
        cl_uint num_devices = 0;
        clGetDeviceIDs(platforms[req_device_idx], CL_DEVICE_TYPE_ALL, 0, NULL,
                       &num_devices);
        if (req_device_idx >= (int) num_devices) {
                MSG(ERROR,
                    "Defice index %d out of bound (%u devices for platform "
                    "%u)\n",
                    req_device_idx, num_devices, num_platforms);
                free((void*) platforms);
                return false;
        }
        cl_device_id *devices =
            (cl_device_id *) malloc(num_devices * sizeof(cl_device_id));
        clGetDeviceIDs(platforms[req_device_idx], CL_DEVICE_TYPE_ALL,
                       num_devices, devices, NULL);

        *platform_id = platforms[req_platform_idx];
        *device_id   = devices[req_device_idx];
        MSG(VERBOSE, "Using OpenCL platform %d, device %d\n", req_platform_idx,
            req_device_idx);

        free((void *) devices);
        free((void *) platforms);
        return true;
}
#else
bool
opencl_get_device(void **platform_id, void **device_id)
{
        (void) platform_id, (void) device_id;
        MSG(ERROR, "OpenCL support not compiled in!\n");
        return false;
}
#endif
