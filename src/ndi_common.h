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
#include "utils/fs.h"       // for MAX_PATH_SIZE, PATH_SEPARATOR
#include "utils/macros.h"   // for MAX, MERGE, TOSTRING, snprintf_ch

#define MOD_NAME "[NDI] "

#ifndef USE_NDI_VERSION
#define USE_NDI_VERSION 6
#endif

#if USE_NDI_VERSION >= 6
#define NDI_API_VERSION 5
#else
#define NDI_API_VERSION USE_NDI_VERSION
#endif

#ifdef __linux__
#define FALLBACK_NDI_PATHS "/usr/local/lib", "/usr/lib"
#elif defined __APPLE__
// redist NDI for Apple uses /usr/local/lib, which is tried prior to this path
#define FALLBACK_NDI_PATHS \
        "/usr/local/lib", "/Library/NDI SDK for Apple/Lib/macOS", \
            "/Applications/NDI Launcher.app/Contents/Frameworks/"
#else
#define FALLBACK_NDI_PATHS \
        "C:\\Program Files\\NDI\\NDI " TOSTRING(USE_NDI_VERSION) \
            " Runtime\\v" TOSTRING(USE_NDI_VERSION), \
        "C:\\Program Files\\NDI\\NDI " TOSTRING(USE_NDI_VERSION) \
            " Tools\\Runtime",
#endif
#define NDILIB_NDI_LOAD "NDIlib_v" TOSTRING(NDI_API_VERSION) "_load"
#define MAKE_NDI_LIB_NAME(ver) MERGE(NDIlib_v,ver)
typedef MAKE_NDI_LIB_NAME(NDI_API_VERSION) NDIlib_t;
typedef const NDIlib_t* NDIlib_load_f(void);

static const NDIlib_t *NDIlib_load(LIB_HANDLE *lib) {
        char ndi_path[MAX_PATH_SIZE];
        const char *lib_cand[] = { getenv(NDILIB_REDIST_FOLDER)
                                       ? getenv(NDILIB_REDIST_FOLDER)
                                       : "",
                                   FALLBACK_NDI_PATHS };
        if (strlen(lib_cand[0]) == 0) {
                log_msg(LOG_LEVEL_WARNING,
                        "[NDI] " NDILIB_REDIST_FOLDER " environment variable not defined. "
                        "Trying fallback folders\n");
        } else {
                debug_msg("NDILIB_REDIST_FOLDER env set to %s\n",
                          lib_cand[0]);
        }
        LIB_HANDLE hNDILib = NULL;
        for (unsigned int i = 0; i < sizeof lib_cand / sizeof lib_cand[0]; i++) {
                if (i > 0) {
                        log_msg(LOG_LEVEL_INFO, "[NDI] Trying to load from fallback location: %s\n",
                                lib_cand[i]);
                }
                snprintf_ch(ndi_path, "%s%s%s", lib_cand[i],
                            strlen(lib_cand[i]) > 0 ? PATH_SEPARATOR : "",
                            NDILIB_LIBRARY_NAME);
                // Try to load the library
                hNDILib = dlopen(ndi_path, RTLD_LOCAL | RTLD_LAZY);
                if (hNDILib) {
                        break;
                }
                MSG(WARNING, "Failed to open the library %s: %s\n", ndi_path,
                    dlerror());
        }

        // The main NDI entry point for dynamic loading if we got the library
        const NDIlib_t* (*NDIlib_load)(void) = NULL;
        if (hNDILib) {
                *((FARPROC *) &NDIlib_load) =
                    dlsym(hNDILib, NDILIB_NDI_LOAD);
        }

        // If we failed to load the library then we tell people to re-install it
        if (!NDIlib_load) {       // Unload the library if we loaded it
                MSG(ERROR, "The NDI library could not have been found!\n");
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

        verbose_msg("NDI lib loaded from %s - open: %s, load: %s\n", ndi_path,
                    hNDILib == NULL ? "NOK" : "OK",
                    NDIlib_load == NULL ? "NOK" : "OK");
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

#undef MOD_NAME

#endif // defined NDI_COMMON_H_1A76D048_695C_4247_A24A_583C29010FC4

