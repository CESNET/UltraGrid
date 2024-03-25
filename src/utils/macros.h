/**
 * @file   utils/macros.h
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2014-2023 CESNET z.s.p.o.
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

#ifdef __cplusplus
#include <cctype>
#else
#include <ctype.h>
#endif

#define MERGE(a,b)  a##b
#define STRINGIFY(A) #A
#define TOSTRING(A) STRINGIFY(A) // https://stackoverflow.com/questions/240353/convert-a-preprocessor-token-to-a-string

#define IF_NOT_NULL_ELSE(val, default_val) ((val) ? (val) : (default_val))
#define UNDEF -1
#define IF_NOT_UNDEF_ELSE(val, default_val) ((val) != UNDEF ? (val) : (default_val))

#define DIV_ROUNDED_UP(value, div) ((((value) % (div)) != 0) ? ((value) / (div) + 1) : ((value) / (div)))

#define SWAP(a, b) do { b ^= a; a ^= b; b ^= a; } while (0)
#define SWAP_PTR(a, b) do { void *tmp = (a); (a) = (b); (b) = tmp; } while(0)

#undef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#undef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))
#undef CLAMP
#define CLAMP(x, low, high)  (((x) > (high)) ? (high) : (((x) < (low)) ? (low) : (x)))

#ifndef EXTERN_C
#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif
#endif // defined EXTERN_C

/// unconditional alternative to assert that is not affected by NDEBUG macro
#define UG_ASSERT(cond) \
        do { /* NOLINT(cppcoreguidelines-avoid-do-while) */ \
                if (!(cond)) { \
                        fprintf(stderr, \
                                "%s:%d: %s: Assertion `" #cond "' failed.\n", \
                                __FILE__, __LINE__, __func__); \
                        abort(); \
                } \
        } while (0)

/// shortcut for `snprintf(var, sizeof var...)`, `var` must be a char array
#define snprintf_ch(str, ...) snprintf(str, sizeof str, __VA_ARGS__)
#define starts_with(str, token) !strncmp(str, token, strlen(token))

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
#define IS_FCC(val) (isprint((val) >> 24U & 0xFFU) && isprint((val) >> 16U & 0xFFU) && isprint((val) >> 8U & 0xFFU) && isprint((val) & 0xFFU))


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

/// the following limits are used mostly for static array allocations
enum {
        MAX_CPU_CORES = 256,  ///< maximal expected CPU core count
        STR_LEN       = 2048, ///< "standard" string length placeholder
};

/// expands to true value if <k> from tok in format <k>=<v> is prefix of key
#define IS_KEY_PREFIX(tok, key) \
        (strchr((tok), '=') != 0 && \
         strncmp(key, tok, strchr((tok), '=') - (tok)) == 0)
/// similar as above, but also key without a value is accepted (value optional)
#define IS_PREFIX(tok, key) \
        (IS_KEY_PREFIX(tok, key) || strncmp(tok, key, strlen(tok)) == 0)

#endif // !defined UTILS_MACROS_H_1982D373_8862_4453_ADFB_33AECC853E48

