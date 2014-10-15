/*
 * FILE:   lib_common.h
 * AUTHOR: Colin Perkins <csp@isi.edu>
 *         Martin Benes     <martinbenesh@gmail.com>
 *         Lukas Hejtmanek  <xhejtman@ics.muni.cz>
 *         Petr Holub       <hopet@ics.muni.cz>
 *         Milos Liska      <xliska@fi.muni.cz>
 *         Jiri Matela      <matela@ics.muni.cz>
 *         Dalibor Matura   <255899@mail.muni.cz>
 *         Ian Wesley-Smith <iwsmith@cct.lsu.edu>
 *
 * Copyright (c) 2001-2003 University of Southern California
 * Copyright (c) 2005-2010 CESNET z.s.p.o.
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
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute. This product also includes
 *      software developed by CESNET z.s.p.o.
 * 
 * 4. Neither the name of the University, Institute, CESNET nor the names of
 *    its contributors may be used to endorse or promote products derived from
 *    this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
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
 *
 */

/** @brief This macro causes that this module will be statically linked with UltraGrid. */
#define MK_STATIC(A) A, NULL
#define MK_STATIC_REF(A) &A, NULL
#define STRINGIFY(A) #A
#define TOSTRING(x) STRINGIFY(x)

#ifdef BUILD_LIBRARIES
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
extern "C"
#endif
void *open_library(const char *name);

#ifdef __cplusplus
enum library_class {
        LIBRARY_CLASS_UNDEFINED,
        LIBRARY_CLASS_CAPTURE_FILTER,
};
#include <map>
#include <string>
void open_all(const char *pattern);
void register_library(std::string const & name, void *info, enum library_class, int abi_version);
void *load_library(std::string const & name, enum library_class, int abi_version);
std::map<std::string, void *> get_libraries_for_class(enum library_class cls, int abi_version);
#endif

