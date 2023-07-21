/**
 * @file   utils/math.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2022-2023 CESNET, z. s. p. o.
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

#include <limits.h>

#include "math.h"

long long gcd(long long a, long long b)
{
        // Everything divides 0
        if (a == 0) {
                return b;
        }
        if (b == 0) {
                return a;
        }

        // base case
        if (a == b) {
                return a;
        }

        // a is greater
        if (a > b) {
                return gcd(a-b, b);
        }
        return gcd(a, b-a);
}

long long lcm(long long a, long long b) {
        return a * b / gcd(a, b);
}

bool
is_power_of_two(unsigned long long x)
{
        return x != 0 && (x & (x - 1)) == 0;
}

/**
 * @returns nearest power of two greater or equal than given number x
 * @retval  0 on overflow (next power of two is ULLONG_MAX + 1)
 */
unsigned long long
next_power_of_two(unsigned long long x)
{
        if (x <= 1) {
                return 1;
        }
        const unsigned long long pos = __builtin_clzll(x - 1);
        return pos > 0 ? 1 << (sizeof(unsigned long long) * CHAR_BIT - pos) : 0;
}
