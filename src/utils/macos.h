/**
 * @file   utils/macos.h
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

#ifndef UTILS_MACOS_H_7D98CC2C_24D4_4DC7_8F07_007A55E4B120
#define UTILS_MACOS_H_7D98CC2C_24D4_4DC7_8F07_007A55E4B120

#include <Availability.h>
#include <MacTypes.h>

// prefer over __MAC_<maj>_<min> because those may not be defined in older
// SDKs (https://gist.github.com/torarnv/4967079) and would be evaluated as 0
#define MACOS_VERSION(major, minor) ((major) * 10000 + (minor) * 100)

// compat
#if __MAC_OS_X_VERSION_MIN_REQUIRED < MACOS_VERSION(12, 0)
#define kAudioObjectPropertyElementMain kAudioObjectPropertyElementMaster
#endif

#ifdef __cplusplus
extern "C" {
#endif

const char *get_osstatus_str(OSStatus err);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // defined UTILS_MACOS_H_7D98CC2C_24D4_4DC7_8F07_007A55E4B120

