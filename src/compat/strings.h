/**
 * @file   compat/strings.h
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 *
 * compatibility header for strcasecmp. strdup, strerror_s
 */
/*
 * Copyright (c) 2024 CESNET
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

#ifndef COMPAT_STRINGS_H_D54CAFC8_A1A0_4FF5_80A0_91F34FB11E12
#define COMPAT_STRINGS_H_D54CAFC8_A1A0_4FF5_80A0_91F34FB11E12
  
#include <string.h>

#ifdef _WIN32
#ifndef strcasecmp
#define strcasecmp _stricmp
#endif
#else // ! defined _WIN32
#include <strings.h>
#endif // _WIN32

#ifdef __cplusplus
#define COMPAT_MISC_EXT_C extern "C"
#else
#define COMPAT_MISC_EXT_C extern
#endif

// strerror_s
#ifndef _WIN32
#ifndef __STDC_LIB_EXT1__
#if ! defined __gnu_linux__ // XSI version on non-glibc
#define strerror_s(buf, bufsz, errnum) strerror_r(errnum, buf, bufsz)
#else // use the XSI variant from glibc (strerror_r is either GNU or XSI)
COMPAT_MISC_EXT_C int __xpg_strerror_r(int errcode, char *buffer, size_t length);
#define strerror_s(buf, bufsz, errnum) __xpg_strerror_r(errnum, buf, bufsz)
#endif
#endif // ! defined __STDC_LIB_EXT1__
#endif // ! defined _WIN32

// strdupa is defined as a macro
#include <string.h>
#ifndef strdupa
#define strdupa(s) (char *) memcpy(alloca(strlen(s) + 1), s, strlen(s) + 1)
#endif // defined strdupa

#undef COMPAT_MISC_EXT_C

#endif // ! defined COMPAT_STRINGS_H_D54CAFC8_A1A0_4FF5_80A0_91F34FB11E12
