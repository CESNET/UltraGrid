/**
 * @file   utils/hresult.c
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019 CESNET, z. s. p. o.
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

#ifdef WIN32
#include "utils/hresult.h"

#include <audioclient.h>

const char *hresult_to_str(HRESULT res) {
        thread_local char unknown[128];
        switch (res) {
                case S_OK: return "Operation successful";
                case S_FALSE: return "Operation completed";
                case E_NOTIMPL: return "Not implemented";
                case E_NOINTERFACE: return "No such interface supported";
                case E_POINTER: return "Pointer that is not valid";
                case E_ABORT: return "Operation aborted";
                case E_FAIL: return "Unspecified failure";
                case E_UNEXPECTED: return "Unexpected failure";
                case E_ACCESSDENIED: return "General access denied error";
                case E_HANDLE: return "Handle that is not valid";
                case E_OUTOFMEMORY: return "Failed to allocate necessary memory";
                case E_INVALIDARG: return "One or more arguments are not valid";
                case 2147746288: return "CoInitialize has not been called.";
		case AUDCLNT_E_ALREADY_INITIALIZED: return "The IAudioClient object is already initialized.";
		case AUDCLNT_E_WRONG_ENDPOINT_TYPE: return "The AUDCLNT_STREAMFLAGS_LOOPBACK flag is set but the endpoint device is a capture device, not a rendering device.";
		case AUDCLNT_E_BUFFER_SIZE_NOT_ALIGNED: return "The requested buffer size is not aligned. This code can be returned for a render or a capture device if the caller specified AUDCLNT_SHAREMODE_EXCLUSIVE and the AUDCLNT_STREAMFLAGS_EVENTCALLBACK flags. The caller must call Initialize again with the aligned buffer size.";
		case AUDCLNT_E_BUFFER_SIZE_ERROR: return "Indicates that the buffer duration value requested by an exclusive-mode client is out of range. The requested duration value for pull mode must not be greater than 500 milliseconds; for push mode the duration value must not be greater than 2 seconds.";
		case AUDCLNT_E_CPUUSAGE_EXCEEDED: return "Indicates that the process-pass duration exceeded the maximum CPU usage. The audio engine keeps track of CPU usage by maintaining the number of times the process-pass duration exceeds the maximum CPU usage. The maximum CPU usage is calculated as a percent of the engine's periodicity. The percentage value is the system's CPU throttle value (within the range of 10%% and 90%%). If this value is not found, then the default value of 40%% is used to calculate the maximum CPU usage.)";
		case AUDCLNT_E_DEVICE_INVALIDATED: return "The audio endpoint device has been unplugged, or the audio hardware or associated hardware resources have been reconfigured, disabled, removed, or otherwise made unavailable for use.";
		case AUDCLNT_E_DEVICE_IN_USE: return "The endpoint device is already in use. Either the device is being used in exclusive mode, or the device is being used in shared mode and the caller asked to use the device in exclusive mode.";
		case AUDCLNT_E_ENDPOINT_CREATE_FAILED: return "The method failed to create the audio endpoint for the render or the capture device. This can occur if the audio endpoint device has been unplugged, or the audio hardware or associated hardware resources have been reconfigured, disabled, removed, or otherwise made unavailable for use.";
		case AUDCLNT_E_INVALID_DEVICE_PERIOD: return "Indicates that the device period requested by an exclusive-mode client is greater than 500 milliseconds.";
		case AUDCLNT_E_UNSUPPORTED_FORMAT: return "The audio engine (shared mode) or audio endpoint device (exclusive mode) does not support the specified format.";
		case AUDCLNT_E_EXCLUSIVE_MODE_NOT_ALLOWED: return "The caller is requesting exclusive-mode use of the endpoint device, but the user has disabled exclusive-mode use of the device.";
		case AUDCLNT_E_BUFDURATION_PERIOD_NOT_EQUAL: return "The AUDCLNT_STREAMFLAGS_EVENTCALLBACK flag is set but parameters hnsBufferDuration and hnsPeriodicity are not equal.";
		case AUDCLNT_E_SERVICE_NOT_RUNNING: return "The Windows audio service is not running.";
                default:
                       snprintf(unknown, sizeof unknown, "(unknown: %ld S=%lu,R=%lu,C=%lu,facility=%lu,code=%lu)", res, ((unsigned long int) res) >> 31u, (res >> 30u) & 0x1, (res >> 29u) & 0x1, (res >> 16u) & 0x7ff, res & 0xff);
                       return unknown;
        }
}
#endif // defined WIN32

