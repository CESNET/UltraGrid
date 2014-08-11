/**
 * @file   blackmagic_common.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014 CESNET, z. s. p. o.
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

#include "debug.h"

#include "blackmagic_common.h"
#include <unordered_map>

using namespace std;

unordered_map<HRESULT, string> bmd_hresult_to_string_map = {
        {S_OK, "success"},
        {S_FALSE, "false"},
        {E_UNEXPECTED, "unexpected value"},
        {E_NOTIMPL, "not implemented"},
        {E_OUTOFMEMORY, "out of memory"},
        {E_INVALIDARG, "invalid argument"},
        {E_NOINTERFACE, "interface was not found"},
        {E_POINTER, "invalid pointer"},
        {E_HANDLE, "invalid handle"},
        {E_ABORT, "operation aborted"},
        {E_FAIL, "failure"},
        {E_ACCESSDENIED, "access denied"},
};

string bmd_hresult_to_string(HRESULT res)
{
        auto it = bmd_hresult_to_string_map.find(res);
        if (it != bmd_hresult_to_string_map.end()) {
                return it->second;
        }
        return {};
}

/**
 * returned c-sring needs to be freed when not used
 */
const char *get_cstr_from_bmd_api_str(BMD_STR bmd_string)
{
       const  char *cstr;
#ifdef HAVE_MACOSX
        cstr = (char *) malloc(128);
        CFStringGetCString(bmd_string, (char *) cstr, 128, kCFStringEncodin
                        gMacRoman);
#elif defined WIN32
        cstr = (char *) malloc(128);
        wcstombs((char *) cstr, bmd_string, 128);
#else // Linux
        cstr = bmd_string;
#endif

        return cstr;
}

void release_bmd_api_str(BMD_STR string)
{
        /// @todo what about MSW?
#ifdef HAVE_MACOSX
        CFRelease(deviceNameString);
#else
        UNUSED(string);
#endif
}

