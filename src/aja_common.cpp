/**
 * @file   aja_common.cpp
 * @author Martin Pulec     <pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2020 CESNET, z. s. p. o.
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

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif
#include "config_msvc.h"
#include "config_unix.h"
#include "config_win32.h"

#include "aja_common.h"

#ifndef BYTE_SWAP
#ifdef WORDS_BIGENDIAN
#define BYTE_SWAP(x) (3 - x)
#else
#define BYTE_SWAP(x) x
#endif
#endif

#include "utils/macros.h" // OPTIMIZED_FOR

#ifndef UNUSED
# define UNUSED(x) ((void) x)
#endif

void
vc_copylineR12LtoR12A(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dstlen - 36; x += 36) {
                *dst++ = src[BYTE_SWAP(1)] << 4 |  src[BYTE_SWAP(0)] >> 4; // r0
                *dst++ = src[BYTE_SWAP(0)] << 4 | src[BYTE_SWAP(2)] >> 4; // r0+g0
                *dst++ = src[BYTE_SWAP(2)] << 4 | src[BYTE_SWAP(1)] >> 4; // g0
                *dst++ = src[BYTE_SWAP(4)] << 4 | src[BYTE_SWAP(3)] >> 4; // b0
                *dst++ = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(5)] >> 4; // b0+r1
                *dst++ = src[BYTE_SWAP(5)] << 4 | src[BYTE_SWAP(4)] >> 4; // r1
                *dst++ = src[BYTE_SWAP(7)] << 4 | src[BYTE_SWAP(6)] >> 4; // g1
                *dst++ = src[BYTE_SWAP(6)] << 4 | src[BYTE_SWAP(8)] >> 4; // g1+b1
                *dst++ = src[BYTE_SWAP(8)] << 4 | src[BYTE_SWAP(7)] >> 4; // b1
                *dst++ = src[BYTE_SWAP(10)] << 4 | src[BYTE_SWAP(9)] >> 4; // r2
                *dst++ = src[BYTE_SWAP(9)] << 4 | src[BYTE_SWAP(11)] >> 4; // r2+g2
                *dst++ = src[BYTE_SWAP(11)] << 4 | src[BYTE_SWAP(10)] >> 4; // g2
                *dst++ = src[BYTE_SWAP(13)] << 4 | src[BYTE_SWAP(12)] >> 4; // b2
                *dst++ = src[BYTE_SWAP(12)] << 4 | src[BYTE_SWAP(14)] >> 4; // b2+r3
                *dst++ = src[BYTE_SWAP(14)] << 4 | src[BYTE_SWAP(13)] >> 4; // r3
                *dst++ = src[BYTE_SWAP(16)] << 4 | src[BYTE_SWAP(15)] >> 4; // g3
                *dst++ = src[BYTE_SWAP(15)] << 4 | src[BYTE_SWAP(17)] >> 4; // g3+b3
                *dst++ = src[BYTE_SWAP(17)] << 4 | src[BYTE_SWAP(16)] >> 4; // b3
                *dst++ = src[BYTE_SWAP(19)] << 4 | src[BYTE_SWAP(18)] >> 4; // r4
                *dst++ = src[BYTE_SWAP(18)] << 4 | src[BYTE_SWAP(20)] >> 4; // r4+g4
                *dst++ = src[BYTE_SWAP(20)] << 4 | src[BYTE_SWAP(19)] >> 4; // g4
                *dst++ = src[BYTE_SWAP(22)] << 4 | src[BYTE_SWAP(21)] >> 4; // b4
                *dst++ = src[BYTE_SWAP(20)] >> 4 | src[BYTE_SWAP(23)] >> 4; // b4+r5
                *dst++ = src[BYTE_SWAP(23)] << 4 | src[BYTE_SWAP(22)] >> 4; // r5
                *dst++ = src[BYTE_SWAP(25)] << 4 | src[BYTE_SWAP(24)] >> 4; // g5
                *dst++ = src[BYTE_SWAP(24)] << 4 | src[BYTE_SWAP(26)] >> 4; // g5+b5
                *dst++ = src[BYTE_SWAP(26)] << 4 | src[BYTE_SWAP(25)] >> 4; // b5
                *dst++ = src[BYTE_SWAP(28)] << 4 | src[BYTE_SWAP(27)] >> 4; // r6
                *dst++ = src[BYTE_SWAP(27)] << 4 | src[BYTE_SWAP(29)] >> 4; // r6+g6
                *dst++ = src[BYTE_SWAP(29)] << 4 | src[BYTE_SWAP(28)] >> 4; // g6
                *dst++ = src[BYTE_SWAP(31)] << 4 | src[BYTE_SWAP(30)] >> 4; // b6
                *dst++ = src[BYTE_SWAP(30)] << 4 | src[BYTE_SWAP(32)] >> 4; // b6+r7
                *dst++ = src[BYTE_SWAP(32)] << 4 | src[BYTE_SWAP(31)] >> 4; // r7
                *dst++ = src[BYTE_SWAP(34)] << 4 | src[BYTE_SWAP(33)] >> 4; // g7
                *dst++ = src[BYTE_SWAP(33)] << 4 | src[BYTE_SWAP(35)] >> 4; // g7+b7
                *dst++ = src[BYTE_SWAP(35)] << 4 | src[BYTE_SWAP(34)] >> 4; // b7
                src += 36;
        }
}

/**
 * Converts AJA NTV2_FBF_12BIT_RGB_PACKED to R12L
 */
void
vc_copylineR12AtoR12L(unsigned char * __restrict dst, const unsigned char * __restrict src, int dstlen, int rshift,
                int gshift, int bshift)
{
        UNUSED(rshift);
        UNUSED(gshift);
        UNUSED(bshift);

        OPTIMIZED_FOR (int x = 0; x <= dstlen - 36; x += 36) {
                *dst++ = src[BYTE_SWAP(0)] << 4 |  src[BYTE_SWAP(1)] >> 4; // r0
                *dst++ = src[BYTE_SWAP(2)] << 4 |  src[BYTE_SWAP(0)] >> 4; // r0+g0
                *dst++ = src[BYTE_SWAP(1)] << 4 | src[BYTE_SWAP(2)] >> 4; // g0
                *dst++ = src[BYTE_SWAP(3)] << 4 | src[BYTE_SWAP(4)] >> 4; // b0
                *dst++ = src[BYTE_SWAP(5)] << 4 | src[BYTE_SWAP(3)] >> 4; // b0+r1
                *dst++ = src[BYTE_SWAP(4)] << 4 | src[BYTE_SWAP(5)] >> 4; // r1
                *dst++ = src[BYTE_SWAP(6)] << 4 | src[BYTE_SWAP(7)] >> 4; // g1
                *dst++ = src[BYTE_SWAP(8)] << 4 | src[BYTE_SWAP(6)] >> 4; // g1+b1
                *dst++ = src[BYTE_SWAP(7)] << 4 | src[BYTE_SWAP(8)] >> 4;
                *dst++ = src[BYTE_SWAP(9)] << 4 | src[BYTE_SWAP(10)] >> 4;
                *dst++ = src[BYTE_SWAP(11)] << 4 | src[BYTE_SWAP(9)] >> 4;
                *dst++ = src[BYTE_SWAP(10)] << 4 | src[BYTE_SWAP(11)] >> 4;
                *dst++ = src[BYTE_SWAP(12)] << 4 | src[BYTE_SWAP(13)] >> 4;
                *dst++ = src[BYTE_SWAP(14)] << 4 | src[BYTE_SWAP(12)] >> 4;
                *dst++ = src[BYTE_SWAP(13)] << 4 | src[BYTE_SWAP(14)] >> 4;
                *dst++ = src[BYTE_SWAP(15)] << 4 | src[BYTE_SWAP(16)] >> 4;
                *dst++ = src[BYTE_SWAP(17)] << 4 | src[BYTE_SWAP(15)] >> 4;
                *dst++ = src[BYTE_SWAP(16)] << 4 | src[BYTE_SWAP(17)] >> 4;
                *dst++ = src[BYTE_SWAP(18)] << 4 | src[BYTE_SWAP(19)] >> 4;
                *dst++ = src[BYTE_SWAP(20)] << 4 | src[BYTE_SWAP(18)] >> 4;
                *dst++ = src[BYTE_SWAP(19)] << 4 | src[BYTE_SWAP(20)] >> 4;
                *dst++ = src[BYTE_SWAP(21)] << 4 | src[BYTE_SWAP(22)] >> 4;
                *dst++ = src[BYTE_SWAP(23)] << 4 | src[BYTE_SWAP(21)] >> 4;
                *dst++ = src[BYTE_SWAP(22)] << 4 | src[BYTE_SWAP(23)] >> 4; // 23 - R5u - @todo problem here
                *dst++ = src[BYTE_SWAP(24)] << 4 | src[BYTE_SWAP(25)] >> 4;
                *dst++ = src[BYTE_SWAP(26)] << 4 | src[BYTE_SWAP(24)] >> 4;
                *dst++ = src[BYTE_SWAP(25)] << 4 | src[BYTE_SWAP(26)] >> 4;
                *dst++ = src[BYTE_SWAP(27)] << 4 | src[BYTE_SWAP(28)] >> 4;
                *dst++ = src[BYTE_SWAP(29)] << 4 | src[BYTE_SWAP(27)] >> 4;
                *dst++ = src[BYTE_SWAP(28)] << 4 | src[BYTE_SWAP(29)] >> 4;
                *dst++ = src[BYTE_SWAP(30)] << 4 | src[BYTE_SWAP(31)] >> 4;
                *dst++ = src[BYTE_SWAP(32)] << 4 | src[BYTE_SWAP(30)] >> 4;
                *dst++ = src[BYTE_SWAP(31)] << 4 | src[BYTE_SWAP(32)] >> 4;
                *dst++ = src[BYTE_SWAP(33)] << 4 | src[BYTE_SWAP(34)] >> 4;
                *dst++ = src[BYTE_SWAP(35)] << 4 | src[BYTE_SWAP(33)] >> 4;
                *dst++ = src[BYTE_SWAP(34)] << 4 | src[BYTE_SWAP(35)] >> 4;
                src += 36;
        }
}

