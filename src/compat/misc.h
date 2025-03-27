/**
 * @file   compat/misc.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * Miscellaneous compat functions that are simple enough to be implemented
 * in a separate file.
 */
/*
 * Copyright (c) 2021-2024 CESNET
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

#ifndef COMPAT_MISC_H_20C709DB_F4A8_4744_A0A9_96036B277011
#define COMPAT_MISC_H_20C709DB_F4A8_4744_A0A9_96036B277011

#ifdef WANT_MKDIR
#ifdef _WIN32
#include <direct.h>
#define mkdir(path, mode) _mkdir(path)
#else
#include <sys/stat.h>
#endif
#endif // defined WANT_MKDIR

#ifdef WANT_FSEEKO64
        #include <stdio.h>
        #ifdef _WIN32
                #define ftello _ftelli64
                #define fseeko _fseeki64
        #else
                #ifndef __cplusplus
                        #include <assert.h> // static_assert macro (until C23)
                #endif
                static_assert(sizeof(off_t) >= 8, "off_t less than 64b");
        #endif // !defined _WIN32
#endif // defined WANT_FSEEKO

#endif // defined COMPAT_MISC_H_20C709DB_F4A8_4744_A0A9_96036B277011

