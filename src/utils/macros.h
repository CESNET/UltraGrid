/**
 * @file   utils/macros.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2022 CESNET z.s.p.o.
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

#ifndef UTILS_MACROS_H_1982D373_8862_4453_ADFB_33AECC853E48
#define UTILS_MACROS_H_1982D373_8862_4453_ADFB_33AECC853E48

#define MERGE(a,b)  a##b
#define STRINGIFY(A) #A
#define TOSTRING(A) STRINGIFY(A) // https://stackoverflow.com/questions/240353/convert-a-preprocessor-token-to-a-string
#define IF_NOT_NULL_ELSE(val, default_val) ((val) ? (val) : (default_val))
#define UNDEF -1
#define IF_NOT_UNDEF_ELSE(val, default_val) ((val) != UNDEF ? (val) : (default_val))

/**
 * @brief Creates FourCC word
 *
 * The main idea of FourCC is that it can be anytime read by human (by hexa editor, gdb, tcpdump).
 * Therefore, this is stored as a big endian even on little-endian architectures - first byte
 * of FourCC is in the memory on the lowest address.
 */
#ifdef WORDS_BIGENDIAN
#define to_fourcc(a,b,c,d)     (((uint32_t)(d)) | ((uint32_t)(c)<<8U) | ((uint32_t)(b)<<16U) | ((uint32_t)(a)<<24U))
#else
#define to_fourcc(a,b,c,d)     (((uint32_t)(a)) | ((uint32_t)(b)<<8U) | ((uint32_t)(c)<<16U) | ((uint32_t)(d)<<24U))
#endif

/* Use following macro only if there are no dependencies between loop
 * iterations (GCC), perhals the same holds also for clang. */
#define __NL__
#if defined __clang__ // try clang first - on macOS, clang defines both __clang__ and __GNUC__
#define OPTIMIZED_FOR _Pragma("clang loop vectorize(assume_safety) interleave(enable)") __NL__ for
#elif defined __GNUC__
#define OPTIMIZED_FOR _Pragma("GCC ivdep") __NL__ for
#else
#define OPTIMIZED_FOR for
#endif

#endif // !defined UTILS_MACROS_H_1982D373_8862_4453_ADFB_33AECC853E48

