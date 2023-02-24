/**
 * @file   utils/windows.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2019-2023 CESNET
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

#ifndef UTILS_WINDOWS_H_DA080A19_C6F3_48E1_A570_63665ED95C1F
#define UTILS_WINDOWS_H_DA080A19_C6F3_48E1_A570_63665ED95C1F

#ifndef __cplusplus
#include <stdbool.h>
#endif

/// this macro is shared between both utils/windows.cpp, where the error codes defines
/// system and bmd_hresult_to_string where only the subset below is defined by
/// LinuxCOM.h compat, so use only values defined in that header.
#define HRESULT_GET_ERROR_COMMON(res, errptr) \
        switch (res) { \
                case S_OK: errptr = "Operation successful"; break; \
                case S_FALSE: errptr = "Operation completed"; break; \
                case E_NOTIMPL: errptr = "Not implemented"; break; \
                case E_NOINTERFACE: errptr = "No such interface supported"; break; \
                case E_POINTER: errptr = "Pointer that is not valid"; break; \
                case E_ABORT: errptr = "Operation aborted"; break; \
                case E_FAIL: errptr = "Unspecified failure"; break; \
                case E_UNEXPECTED: errptr = "Unexpected failure"; break; \
                case E_ACCESSDENIED: errptr = "General access denied error"; break; \
                case E_HANDLE: errptr = "Handle that is not valid"; break; \
                case E_OUTOFMEMORY: errptr = "Failed to allocate necessary memory"; break; \
                case E_INVALIDARG: errptr = "One or more arguments are not valid"; break; \
        }

#ifdef __cplusplus
extern "C" {
#endif

///< param pass a pointer to a bool that will be passed to com_uninintialize()
bool com_initialize(bool *com_initialized);
///< @param com_initialized - pointer passed to com_initialize (or create_com_iterator)
void com_uninitialize(bool *com_initialized);

#ifdef _WIN32
#include <winerror.h>
const char *hresult_to_str(HRESULT res);
#endif

#ifdef __cplusplus
}
#endif

#endif // define UTILS_WINDOWS_H_DA080A19_C6F3_48E1_A570_63665ED95C1F

