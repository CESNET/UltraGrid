/**
 * @file   ndi_common.h
 * @author Martin Pulec      <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022 CESNET, z. s. p. o.
 * All rights reserved.
 *
 * Using sample code from NDI.
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

#ifndef NDI_COMMON_H_1A76D048_695C_4247_A24A_583C29010FC4
#define NDI_COMMON_H_1A76D048_695C_4247_A24A_583C29010FC4

#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#define NDILIB_CPP_DEFAULT_CONSTRUCTORS 0
#include <Processing.NDI.Lib.h>

#include "debug.h"
#include "lib_common.h" // LIB_HANDLE, dlclose/dlerror abstraction

#define NDILIB_NDI_LOAD "NDIlib_v5_load"
typedef NDIlib_v5 NDIlib_t;
typedef const NDIlib_t* NDIlib_load_f(void);

static const NDIlib_t *NDIlib_load(LIB_HANDLE *lib) {
#ifdef _WIN32
        // We check whether the NDI run-time is installed
        const char* p_ndi_runtime = getenv(NDILIB_REDIST_FOLDER);
        if (!p_ndi_runtime) {       // The NDI run-time is not yet installed. Let the user know and take them to the download URL.
                //MessageBoxA(NULL, "Please install the NewTek NDI Runtimes to use this application from " NDILIB_REDIST_URL ".", "Runtime Warning.", MB_OK);
                //ShellExecuteA(NULL, "open", NDILIB_REDIST_URL, 0, 0, SW_SHOWNORMAL);
                log_msg(LOG_LEVEL_WARNING, "[NDI] " NDILIB_REDIST_FOLDER " environment variable not defined. "
                                "Please install the NewTek NDI Runtimes to use this application from " NDILIB_REDIST_URL ".\n");
                return 0;
        }

        // We now load the DLL as it is installed
        char *ndi_path = (char *) alloca(strlen(p_ndi_runtime) + 2 + strlen(NDILIB_LIBRARY_NAME) + 1);
        strcpy(ndi_path, p_ndi_runtime); // NOLINT (security.insecureAPI.strcpy)
        strcat(ndi_path, "\\"); // NOLINT (security.insecureAPI.strcpy)
        strcat(ndi_path, NDILIB_LIBRARY_NAME); // NOLINT (security.insecureAPI.strcpy)

        // Try to load the library
        HMODULE hNDILib = LoadLibraryA(ndi_path);

        // The main NDI entry point for dynamic loading if we got the library
        const NDIlib_t* (*NDIlib_load)(void) = NULL;
        if (hNDILib) {
                *((FARPROC*)&NDIlib_load) = GetProcAddress(hNDILib, NDILIB_NDI_LOAD);
        }

        // If we failed to load the library then we tell people to re-install it
        if (!NDIlib_load) {       // Unload the DLL if we loaded it
                // The NDI run-time is not installed correctly. Let the user know and take them to the download URL.
                log_msg(LOG_LEVEL_ERROR, "[NDI] Failed to load " NDILIB_NDI_LOAD " from NDI: %s.\n"
                                "Try to reinstall the NewTek NDI Runtimes to use this application from " NDILIB_REDIST_URL ".\n", dlerror());
                if (hNDILib) {
                        FreeLibrary(hNDILib);
                }

                return 0;
        }
#else
        const char* p_NDI_runtime_folder = getenv(NDILIB_REDIST_FOLDER);
        char *ndi_path = (char *) alloca((p_NDI_runtime_folder != NULL ? strlen(p_NDI_runtime_folder) + 1 : 0)+ strlen(NDILIB_LIBRARY_NAME) + 1);
        if (p_NDI_runtime_folder) {
                strcpy(ndi_path, p_NDI_runtime_folder); // NOLINT (security.insecureAPI.strcpy)
                strcat(ndi_path, "/"); // NOLINT (security.insecureAPI.strcpy)
        } else {
                ndi_path[0] = '\0';
        }
        strcat(ndi_path, NDILIB_LIBRARY_NAME); // NOLINT (security.insecureAPI.strcpy)

        // Try to load the library
        void *hNDILib = dlopen(ndi_path, RTLD_LOCAL | RTLD_LAZY);

        // The main NDI entry point for dynamic loading if we got the library
        const NDIlib_t* (*NDIlib_load)(void) = NULL;
        if (hNDILib) {
                *((void**)&NDIlib_load) = dlsym(hNDILib, NDILIB_NDI_LOAD);
        }

        // If we failed to load the library then we tell people to re-install it
        if (!NDIlib_load) {       // Unload the library if we loaded it
                log_msg(LOG_LEVEL_ERROR, "[NDI] Failed to open the library: %s\n", dlerror());
                if (strlen(NDILIB_REDIST_URL) != 0) {
                        log_msg(LOG_LEVEL_ERROR, "[NDI] Please re-install the NewTek NDI Runtimes from " NDILIB_REDIST_URL " to use this application.\n");
                } else { // NDILIB_REDIST_URL is set to "" in Linux
                        log_msg(LOG_LEVEL_ERROR, "[NDI] Please install " NDILIB_LIBRARY_NAME " from the NewTek NDI SDK either to system libraries' path or set "
                                        NDILIB_REDIST_FOLDER " environment "
                                        "variable to a path containing the libndi.so file (eg. \"export " NDILIB_REDIST_FOLDER "=<NDI_SDK_PATH>/lib/x86_64-linux-gnu\").\n");
                }

                if (hNDILib) {
                        dlclose(hNDILib);
                }
                return 0;
        }
#endif
        const NDIlib_t *ret = NDIlib_load();
        if (ret == NULL) {
                dlclose(hNDILib);
        } else {
                *lib = hNDILib;
        }
        return ret;
}

static void close_ndi_library(LIB_HANDLE hNDILib) {
        if (!hNDILib) {
                return;
        }
        dlclose(hNDILib);
}

#endif // defined NDI_COMMON_H_1A76D048_695C_4247_A24A_583C29010FC4

