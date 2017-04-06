/**
 * @file   lib_common.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2011-2015 CESNET, z. s. p. o.
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
};
void open_all(const char *pattern);
const void *load_library(const char *name, enum library_class, int abi_version);
void register_library(const char *name, const void *info, enum library_class, int abi_version);
void list_modules(enum library_class, int abi_version);
void list_all_modules();
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <map>
#include <string>
std::map<std::string, const void *> get_libraries_for_class(enum library_class cls, int abi_version);
#endif

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
#define REGISTER_MODULE(name, info, lclass, abi) static void mod_reg_##lclass##_##name(void)  __attribute__((constructor));\
\
static void mod_reg_##lclass##_##name(void)\
{\
        register_library(#name, info, lclass, abi);\
}\
struct NOT_DEFINED_STRUCT_THAT_SWALLOWS_SEMICOLON

#endif // defined LIB_COMMON_H

