/**
 * @file   utils/macos.m
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023 CESNET, z. s. p. o.
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

#include <stdint.h>

#include "utils/macos.h"
#include "utils/macros.h"

static const char *get_mapped_osstatus_str(OSStatus err) {
        static const struct osstatus_map {
                OSStatus err;
                const char *msg;
        } osstatus_map[] = {
                {      0, "noErr" },
                // AUComponent.h
                { -10879, "kAudioUnitErr_InvalidProperty" },
                { -10878, "kAudioUnitErr_InvalidParameter" },
                { -10877, "kAudioUnitErr_InvalidElement" },
                { -10876, "kAudioUnitErr_NoConnection" },
                { -10875, "kAudioUnitErr_FailedInitialization" },
                { -10874, "kAudioUnitErr_TooManyFramesToProcess" },
                { -10871, "kAudioUnitErr_InvalidFile" },
                { -10870, "kAudioUnitErr_UnknownFileType" },
                { -10869, "kAudioUnitErr_FileNotSpecified" },
                { -10868, "kAudioUnitErr_FormatNotSupported" },
                { -10867, "kAudioUnitErr_Uninitialized" },
                { -10866, "kAudioUnitErr_InvalidScope" },
                { -10865, "kAudioUnitErr_PropertyNotWritable" },
                { -10863, "kAudioUnitErr_CannotDoInCurrentContext" },
                { -10851, "kAudioUnitErr_InvalidPropertyValue" },
                { -10850, "kAudioUnitErr_PropertyNotInUse" },
                { -10849, "kAudioUnitErr_Initialized" },
                { -10848, "kAudioUnitErr_InvalidOfflineRender" },
                { -10847, "kAudioUnitErr_Unauthorized" },
                { -66753, "kAudioUnitErr_MIDIOutputBufferFull" },
                { -66754, "kAudioComponentErr_InstanceTimedOut" },
                { -66749, "kAudioComponentErr_InstanceInvalidated" },
                { -66745, "kAudioUnitErr_RenderTimeout" },
                { -66744, "kAudioUnitErr_ExtensionNotFound" },
                { -66743, "kAudioUnitErr_InvalidParameterValue" },
                { -66742, "kAudioUnitErr_InvalidFilePath" },
                { -66741, "kAudioUnitErr_MissingKey" },
                // AudioHardwareBase.h
                { 'stop', "kAudioHardwareNotRunningError" },
                { 'what', "kAudioHardwareUnspecifiedError" },
                { 'who?', "kAudioHardwareUnknownPropertyError" },
                { '!siz', "kAudioHardwareBadPropertySizeError" },
                { 'nope', "kAudioHardwareIllegalOperationError" },
                { '!obj', "kAudioHardwareBadObjectError" },
                { '!dev', "kAudioHardwareBadDeviceError" },
                { '!str', "kAudioHardwareBadStreamError" },
                { 'unop', "kAudioHardwareUnsupportedOperationError" },
                { 'nrdy', "kAudioHardwareNotReadyError" },
                { '!dat', "kAudioDeviceUnsupportedFormatError" },
                { '!hog', "kAudioDevicePermissionsError" },
        };

        for (unsigned i = 0; i < sizeof osstatus_map / sizeof osstatus_map[0]; ++i) {
                if (osstatus_map[i].err == err) {
                        return osstatus_map[i].msg;
                }
        }
        return NULL;
}

const char *get_osstatus_str(OSStatus err) {
        static _Thread_local char buf[128];
        const char *mapped_err = get_mapped_osstatus_str(err);
        if (mapped_err != NULL) {
                snprintf(buf, sizeof buf, "%s", mapped_err);
        } else if (IS_FCC(err)) {
                uint32_t fcc_be = htonl(err);
                snprintf(buf, sizeof buf, "'%.4s'", (char *) &fcc_be);
        } else {
                snprintf(buf, sizeof buf, "0x%x", err);
        }
        snprintf(buf + strlen(buf), sizeof buf - strlen(buf), " (OSStatus error %d)", err);
        return buf;
}

