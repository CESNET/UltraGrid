/**
 * @file   ndi_common.h
 * @author Martin Pulec      <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022-2024 CESNET
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

#define NDILIB_CPP_DEFAULT_CONSTRUCTORS 0
#include <Processing.NDI.Lib.h>

#include "compat/dlfunc.h"
#include "debug.h"
#include "utils/color_out.h"
#include "utils/macros.h" // MAX, MERGE, TOSTRING

#ifndef USE_NDI_VERSION
#define USE_NDI_VERSION 6
#endif

#if USE_NDI_VERSION >= 6
#define NDI_API_VERSION 5
#else
#define NDI_API_VERSION USE_NDI_VERSION
#endif

#ifdef __linux__
#define FALLBACK_NDI_PATH "/usr/lib"
#elif defined __APPLE__
// redist NDI for Apple uses /usr/local/lib, which is tried prior to this path
#define FALLBACK_NDI_PATH "/Library/NDI SDK for Apple/Lib/macOS"
#else
#define FALLBACK_NDI_PATH "C:\\Program Files\\NDI\\NDI " TOSTRING(USE_NDI_VERSION) " Runtime\\v" TOSTRING(USE_NDI_VERSION)
#endif
#define NDILIB_NDI_LOAD "NDIlib_v" TOSTRING(NDI_API_VERSION) "_load"
#define MAKE_NDI_LIB_NAME(ver) MERGE(NDIlib_v,ver)
typedef MAKE_NDI_LIB_NAME(NDI_API_VERSION) NDIlib_t;
typedef const NDIlib_t* NDIlib_load_f(void);

static const NDIlib_t *NDIlib_load(LIB_HANDLE *lib) {
#ifdef _WIN32
        // We check whether the NDI run-time is installed
        const char* p_ndi_runtime = getenv(NDILIB_REDIST_FOLDER);
        if (!p_ndi_runtime) {
                p_ndi_runtime = FALLBACK_NDI_PATH;
                log_msg(LOG_LEVEL_WARNING,
                        "[NDI] " NDILIB_REDIST_FOLDER " environment variable not defined. "
                        "Trying fallback folder: %s\n",
                        FALLBACK_NDI_PATH);
        }

        // We now load the DLL as it is installed
        const size_t path_len = strlen(p_ndi_runtime) + 2 + strlen(NDILIB_LIBRARY_NAME) + 1;
        char *ndi_path = (char *)alloca(path_len);
        strncpy(ndi_path, p_ndi_runtime, path_len - 1);
        strncat(ndi_path, "\\", path_len - strlen(ndi_path) - 1);
        strncat(ndi_path, NDILIB_LIBRARY_NAME, path_len - strlen(ndi_path) - 1);

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
                                "Please install the NewTek NDI Runtimes to use this module from " NDILIB_REDIST_URL ".\n", dlerror());
                if (hNDILib) {
                        FreeLibrary(hNDILib);
                }

                return 0;
        }
#else
        const char *lib_cand[3] = {getenv(NDILIB_REDIST_FOLDER) ? getenv(NDILIB_REDIST_FOLDER) : "",
                                   "/usr/local/lib", FALLBACK_NDI_PATH};
        void *hNDILib = NULL;
        const char *last_err = "(none)";
        for (unsigned int i = 0; i < sizeof lib_cand / sizeof lib_cand[0]; i++) {
                if (i > 0) {
                        log_msg(LOG_LEVEL_INFO, "[NDI] Trying to load from fallback location: %s\n",
                                lib_cand[i]);
                }
                size_t path_len = strlen(lib_cand[i]) + 1 + strlen(NDILIB_LIBRARY_NAME) + 1;
                char *ndi_path = (char *)alloca(path_len);
                strncpy(ndi_path, lib_cand[i], path_len - 1);
                if (strlen(ndi_path) > 0) {
                        strncat(ndi_path, "/", path_len - strlen(ndi_path) - 1);
                }
                strncat(ndi_path, NDILIB_LIBRARY_NAME, path_len - strlen(ndi_path) - 1);
                // Try to load the library
                hNDILib = dlopen(ndi_path, RTLD_LOCAL | RTLD_LAZY);
                if (hNDILib) {
                        break;
                }
                last_err = dlerror();
                log_msg(LOG_LEVEL_WARNING,
                        "[NDI] Failed to open the library: %s\n", last_err);
        }

        // The main NDI entry point for dynamic loading if we got the library
        const NDIlib_t* (*NDIlib_load)(void) = NULL;
        if (hNDILib) {
                *((void**)&NDIlib_load) = dlsym(hNDILib, NDILIB_NDI_LOAD);
        }

        // If we failed to load the library then we tell people to re-install it
        if (!NDIlib_load) {       // Unload the library if we loaded it
                log_msg(LOG_LEVEL_ERROR,
                        "[NDI] Failed to open the library: %s\n", last_err);

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

#define NDI_PRINT_COPYRIGHT \
        color_printf( \
            TERM_BOLD TERM_FG_BLUE u8"This application uses NDI® available " \
                                   u8"from https://ndi.video/\n" \
                                   u8"NDI® is a registered trademark of " \
                                   u8"Vizrt NDI AB.\n\n" TERM_RESET); \
        int not_defined_function

#endif // defined NDI_COMMON_H_1A76D048_695C_4247_A24A_583C29010FC4

