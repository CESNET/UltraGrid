/**
 * @file   compat/qsort_s.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 *
 * @note
 * Must be included before config*h because those include stdlib.h.
 */
/*
 * Copyright (c) 2022-2023 CESNET z.s.p.o.
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

#ifndef COMPAT_QSORT_S_H_81DF21DE_370E_4C70_8559_600C1A9B6059
#define COMPAT_QSORT_S_H_81DF21DE_370E_4C70_8559_600C1A9B6059

#if !defined EXIT_SUCCESS // stdlib.h not yet included
#if defined  __STDC_LIB_EXT1__
#define __STDC_WANT_LIB_EXT1__ 1 // we want qsort_s
#endif
#ifdef __cplusplus
#include <cstdlib>
#else
#include <stdlib.h>
#endif // !defined __cplusplus
#endif // !defined EXIT_SUCCESS

#if !defined __STDC_LIB_EXT1__
# if defined _WIN32
#  define QSORT_S_COMP_FIRST 1 // MS version of qsort_s with comparator as first arg
# else
#  ifdef __APPLE__
#    define QSORT_S_COMP_FIRST 1 // BSD version of qsort_r() as well
#    define qsort_s(ptr, count, size, comp, context) qsort_r(ptr, count, size, context, comp)
#  else
#    define qsort_s qsort_r // GNU version
#  endif
# endif
#endif

#ifdef QSORT_S_COMP_FIRST
#define QSORT_S_COMP_DEFINE(name, a, b, context) int name(void *context, const void *a, const void *b)
#else
#define QSORT_S_COMP_DEFINE(name, a, b, context) int name(const void *a, const void *b, void *context)
#endif

#endif // ! defined COMPAT_QSORT_S_H_81DF21DE_370E_4C70_8559_600C1A9B6059

