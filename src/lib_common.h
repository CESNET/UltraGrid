/**
 * @file   lib_common.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2025 CESNET
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

#ifndef LIB_COMMON_H
#define LIB_COMMON_H

#include "host.h" // UNIQUE_LABEL

/** @brief This macro causes that this module will be statically linked with UltraGrid. */
#define MK_STATIC(A) A, NULL
#define MK_STATIC_REF(A) &A, NULL

#ifdef BUILD_LIBRARIES
#include <dlfcn.h>
/** This macro tells that the module may be statically linked as well as
 * a standalone module. */
#define MK_NAME(A) NULL, #A
#define MK_NAME_REF(A) NULL, #A

#define NULL_IF_BUILD_LIBRARIES(x) NULL

#else /* BUILD_LIBRARIES */

#define MK_NAME(A) A, NULL
#define MK_NAME_REF(A) &A, NULL

#define NULL_IF_BUILD_LIBRARIES(x) x

#endif /* BUILD_LIBRARIES */

#ifdef __cplusplus
#include <list>
void open_all(const char *pattern, std::list<void *> &libs);
#endif

#ifdef __cplusplus
extern "C" {
#endif
enum library_class {
        LIBRARY_CLASS_UNDEFINED,
        LIBRARY_CLASS_CAPTURE_FILTER,
        LIBRARY_CLASS_AUDIO_CAPTURE,
        LIBRARY_CLASS_AUDIO_PLAYBACK,
        LIBRARY_CLASS_VIDEO_CAPTURE,
        LIBRARY_CLASS_VIDEO_DISPLAY,
        LIBRARY_CLASS_AUDIO_COMPRESS,
        LIBRARY_CLASS_VIDEO_DECOMPRESS,
        LIBRARY_CLASS_VIDEO_COMPRESS,
        LIBRARY_CLASS_VIDEO_POSTPROCESS,
        LIBRARY_CLASS_VIDEO_RXTX,
        LIBRARY_CLASS_AUDIO_FILTER,
};
const void *load_library(const char *name, enum library_class, int abi_version);
void register_library(const char *name, const void *info, enum library_class,
                      int abi_version, unsigned visibility_flag);
void list_modules(enum library_class, int abi_version, bool full);
bool list_all_modules();
#ifdef __cplusplus
}
#endif

enum module_flag {
        MODULE_SHOW_VISIBLE_ONLY = 0,      ///< display only modules w/o flag
        MODULE_FLAG_HIDDEN       = 1 << 0, ///< flag - do not show in listing
        MODULE_FLAG_ALIAS =
            1 << 1, ///< ditto + hide for GUI, for explicit init only
        MODULE_SHOW_ALL =
            MODULE_FLAG_HIDDEN | MODULE_FLAG_ALIAS, ///< display all modules
};

#ifdef __cplusplus
#include <map>
#include <string>
std::map<std::string, const void *>
get_libraries_for_class(enum library_class cls, int abi_version,
                        unsigned include_flags = MODULE_FLAG_HIDDEN);
#endif

/**
 * Placeholder that installs module via constructor for every macro
 * REGISTER_MODULE* call
 * @param name     non-quoted module name
 * @param lclass   class of the module
 * @param abi      abi version (specific for every class)
 * @param funcname unique function name that will be used to register
 *                 the module (as a constructor)
 * @param flag     optional flag to limit visibility (@ref module_flag;
 *                 for technical, deprecated modules and aliases), default 0
 */
#define REGISTER_MODULE_FUNCNAME(name, info, lclass, abi, funcname, flag) \
        static void funcname(void) __attribute__((constructor)); \
\
static void funcname(void)\
{\
        register_library(#name, info, lclass, abi, flag);\
}\
struct NOT_DEFINED_STRUCT_THAT_SWALLOWS_SEMICOLON

#define REGISTER_MODULE_FUNC_FUNCNAME(name, func, lclass, abi, funcname, flag) \
        static void funcname(void) __attribute__((constructor)); \
\
static void funcname(void)\
{\
        const void *info = func();\
        if (info) {\
                register_library(#name, info, lclass, abi, flag);\
        }\
}\
struct NOT_DEFINED_STRUCT_THAT_SWALLOWS_SEMICOLON

/**
 * @brief  Registers module to global modules' registry
 * @param name   name of the module to be used to load the module (Note that
 *               it has to be without quotation marks!)
 * @param info   pointer to structure with the (class specific) info about module
 * @param lclass member of @ref library_class
 * @param abi    ABI version of info parameter, usually defined per class
 *               in appropriate class header (eg. video_display.h)
 * @note
 * Mangling of the constructor function name is because some files may define
 * multiple modules (eg. audio playback SDI) and without that, function would
 * be defined multiple times under the same name.
 */
#define REGISTER_MODULE(name, info, lclass, abi) REGISTER_MODULE_FUNCNAME(name, info, lclass, abi, UNIQUE_LABEL, 0)

/**
 * same as REGISTER_MODULE, but a function is called to get the info structure
 * on run-time allowing to run arbitrary code to do some setup
 */
#define REGISTER_MODULE_WITH_FUNC(name, func, lclass, abi) REGISTER_MODULE_FUNC_FUNCNAME(name, func, lclass, abi, UNIQUE_LABEL, 0)

/**
 * Similar to @ref REGISTER_MODULE but allow @ref module_flag to be added to limit visibility.
 */
#define REGISTER_MODULE_WITH_FLAG(name, info, lclass, abi, flag) \
        REGISTER_MODULE_FUNCNAME(name, info, lclass, abi, UNIQUE_LABEL, flag)
#define REGISTER_HIDDEN_MODULE(name, info, lclass, abi) REGISTER_MODULE_WITH_FLAG(name, info, lclass, abi, MODULE_FLAG_HIDDEN)

#endif // defined LIB_COMMON_H

