/**
 * @file   utils/windows.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022-2023 CESNET
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

#include "utils/windows.h"

#ifdef _WIN32
#include <objbase.h>
#include "debug.h"

bool com_initialize(bool *com_initialized)
{
        *com_initialized = false;
        // Initialize COM on this thread
        HRESULT result = CoInitializeEx(NULL, COINIT_MULTITHREADED);
        if (SUCCEEDED(result)) {
                *com_initialized = true;
                return true;
        }
        if (result == RPC_E_CHANGED_MODE) {
                LOG(LOG_LEVEL_WARNING) << "COM already intiialized with a different mode!\n";
                return true;
        }
        LOG(LOG_LEVEL_ERROR) << "Initialize of COM failed - " << bmd_hresult_to_string(result) << "\n";
        return false;
}

void com_uninitialize(bool *com_initialized)
{
        if (!*com_initialized) {
                return;
        }
        *com_initialized = false;
        CoUninitialize();
}
#else
bool com_initialize(bool *com_initialized [[maybe_unused]])
{
        return true;
}
void com_uninitialize(bool *com_initialized [[maybe_unused]])
{
}
#endif // defined _WIN32

