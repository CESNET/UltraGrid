/**
 * @file   audio/portaudio_common.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2018-2025 CESNET
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

#include <assert.h>           // for assert
#include <portaudio.h>
#include <stdbool.h>
#include <stdio.h>            // for snprintf, printf, NULL
#include <stdlib.h>           // for calloc
#include <string.h>           // for strncpy, strstr

#include "debug.h"
#include "portaudio_common.h"
#include "types.h"
#include "utils/color_out.h"

#define MOD_NAME "[PortAudio] "

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

const char *portaudio_get_device_name(PaDeviceIndex device) {
        if( (device < 0) || (device >= Pa_GetDeviceCount()) )
        {
                log_msg(LOG_LEVEL_ERROR, MOD_NAME "Requested info on non-existing device");
                return NULL;
        }

        return Pa_GetDeviceInfo(device)->name;
}

static const char *portaudio_get_device_details(PaDeviceIndex device) {
        assert(device >= 0 && device < Pa_GetDeviceCount());
        const PaDeviceInfo *device_info = Pa_GetDeviceInfo(device);
        _Thread_local static char buffer[1024];
        snprintf(buffer, sizeof buffer, "(max chan in: %d, out: %d; %s)",
                 device_info->maxInputChannels, device_info->maxOutputChannels,
                 portaudio_get_api_name(device));
        return buffer;
}

void
portaudio_print_help(enum portaudio_device_direction kind, bool full)
{
        printf("\nAvailable PortAudio %s devices:\n",
               kind == PORTAUDIO_OUT ? "playback" : "capture");

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

        color_printf("\t" TBOLD("portaudio") " - use default Portaudio device (marked with star)\n");

        for(i = 0; i < numDevices; i++)
        {
                const char *highlight = TERM_BOLD;
                const PaDeviceInfo *device_info = Pa_GetDeviceInfo(i);
                // filter out (or differently highlight in verbose mode) unusable devices
                if ((device_info->maxInputChannels == 0 && kind == PORTAUDIO_IN) ||
                                (device_info->maxOutputChannels == 0 && kind == PORTAUDIO_OUT)) {
                        if (log_level < LOG_LEVEL_VERBOSE) {
                                continue;
                        } else {
                                highlight = TERM_BOLD TERM_FG_BRIGHT_BLACK;
                        }
                }
                if((i == Pa_GetDefaultInputDevice() && kind == PORTAUDIO_IN) ||
                                (i == Pa_GetDefaultOutputDevice() && kind == PORTAUDIO_OUT))
                        printf("(*) ");

                color_printf("\t%sportaudio:%d" TERM_RESET " - %s%s" TERM_RESET " %s", highlight, i, highlight, portaudio_get_device_name(i), portaudio_get_device_details(i));
                printf("\n");
        }

        if (full) {
                printf ("\nSupported APIs:\n");
                for (int i = 0; i < Pa_GetHostApiCount(); ++i) {
                        const PaHostApiInfo *info = Pa_GetHostApiInfo(i);
                        printf("\t" TBOLD("%s") "\n", info->name);
                }
        }

error:
        Pa_Terminate();
}

void audio_portaudio_probe(struct device_info **available_devices, int *count, enum portaudio_device_direction dir)
{
        int numDevices = 0;
        bool initialized = false;
        const char *notice = ""; // we'll always include default device, but with a notice if something wrong

        PaError error = Pa_Initialize();
        if ( error != paNoError ) {
                log_msg(LOG_LEVEL_ERROR, "\tPortAudio error: %s\n", Pa_GetErrorText(error));
                notice = " (init error)";
        } else {
                initialized = true;
        }

        if (initialized && (numDevices = Pa_GetDeviceCount()) <= 0) {
                notice = " (no device)";
                numDevices = 0;
        }
        *available_devices = calloc(1 + numDevices, sizeof(struct device_info));
        strncpy((*available_devices)[0].dev, "", sizeof (*available_devices)[0].dev);
        snprintf((*available_devices)[0].name, sizeof (*available_devices)[0].name, "Portaudio default %s%s", dir == PORTAUDIO_IN ? "input" : "output", notice);
        *count = 1;

        for(int i = 0; i < numDevices; i++) {
                const PaDeviceInfo *device_info = Pa_GetDeviceInfo(i);
                if (((device_info->maxInputChannels == 0 && dir == PORTAUDIO_IN) ||
                                (device_info->maxOutputChannels == 0 && dir == PORTAUDIO_OUT))) {
                        continue;
                }
                snprintf((*available_devices)[*count].name, sizeof (*available_devices)[0].name, "%s", device_info->name);
                snprintf((*available_devices)[*count].dev, sizeof (*available_devices)[0].dev, ":%d", i);
                *count += 1;
        }

        if (initialized) {
                Pa_Terminate();
        }
}

void portaudio_print_version() {
        int error = Pa_Initialize();
        if(error != paNoError) {
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Cannot get version.\n");
                return;
        }

#ifdef HAVE_PA_GETVERSIONINFO
        const PaVersionInfo *info = Pa_GetVersionInfo();

        if (info && info->versionText) {
                log_msg(LOG_LEVEL_INFO, MOD_NAME "Using %s\n", info->versionText);
        }
#else // compat
        log_msg(LOG_LEVEL_INFO, MOD_NAME "Using %s\n", Pa_GetVersionText());
#endif
}

/**
 * Finds a Portaudio device whose name contains substring in argument name
 * @retval -2  device selection failed
 * @retval >=0 selected device
 */
int portaudio_select_device_by_name(const char *name) {
        for (int i = 0; i < Pa_GetDeviceCount(); i++) {
                const PaDeviceInfo *device_info = Pa_GetDeviceInfo(i);
                if (strstr(device_info->name, name)) {
                        return i;
                }
        }
        log_msg(LOG_LEVEL_ERROR, MOD_NAME "No device named \"%s\" was found!\n", name);
        return -2;
}

