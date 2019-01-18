/**
 * @file   audio/portaudio_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018 CESNET, z. s. p. o.
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
#include "config_unix.h"
#include "config_win32.h"
#endif

#include <iostream>
#include <portaudio.h>

#include "debug.h"
#include "portaudio_common.h"
#include "rang.hpp"

using std::cout;

#define MODULE_NAME "[PortAudio] "

static const char *portaudio_get_api_name(PaDeviceIndex device) {
        for (int i = 0; i < Pa_GetHostApiCount(); ++i) {
                const PaHostApiInfo *info = Pa_GetHostApiInfo(i);
                for (int j = 0; j < info->deviceCount; ++j) {
                        if (device == Pa_HostApiDeviceIndexToDeviceIndex(i, j)) {
                                return info->name;
                        }
                }

        }
        return "(unknown API)";
}

void portaudio_print_device_info(PaDeviceIndex device)
{
        if( (device < 0) || (device >= Pa_GetDeviceCount()) )
        {
                printf("Requested info on non-existing device");
                return;
        }

        const   PaDeviceInfo *device_info = Pa_GetDeviceInfo(device);
        printf("%s (output channels: %d; input channels: %d; %s)", device_info->name, device_info->maxOutputChannels, device_info->maxInputChannels, portaudio_get_api_name(device));
}

void portaudio_print_available_devices(enum portaudio_device_direction kind)
{
        int numDevices;
        int i;

        PaError error;

        error = Pa_Initialize();
        if(error != paNoError)
        {
                log_msg(LOG_LEVEL_ERROR, "error initializing portaudio\n");
                log_msg(LOG_LEVEL_ERROR, "\tPortAudio error: %s\n", Pa_GetErrorText( error ) );
                return;
        }

        numDevices = Pa_GetDeviceCount();
        if( numDevices < 0)
        {
                log_msg(LOG_LEVEL_ERROR, "Error getting portaudio devices number\n");
                goto error;
        }
        if( numDevices == 0)
        {
                log_msg(LOG_LEVEL_ERROR, "There are NO available audio devices!\n");
                goto error;
        }

        cout << rang::style::bold << "\tportaudio" << rang::style::reset << " - use default Portaudio device (marked with star)\n";

        for(i = 0; i < numDevices; i++)
        {
                if((i == Pa_GetDefaultInputDevice() && kind == PORTAUDIO_IN) ||
                                (i == Pa_GetDefaultOutputDevice() && kind == PORTAUDIO_OUT))
                        printf("(*) ");

                cout << rang::style::bold << "\tportaudio:" << i << rang::style::reset << " - ";
                portaudio_print_device_info(i);
                printf("\n");
        }

error:
        Pa_Terminate();
}

void portaudio_print_version() {
        int error = Pa_Initialize();
        if(error != paNoError) {
                log_msg(LOG_LEVEL_INFO, MODULE_NAME "Cannot get version.\n");
                return;
        }

        const PaVersionInfo *info = Pa_GetVersionInfo();

        if (info && info->versionText) {
                log_msg(LOG_LEVEL_INFO, MODULE_NAME "Using %s\n", info->versionText);
        }
}
