/**
 * @file   utils/random.c
 * @author Martin Pulec     <martin.pulec@cesnet.cz>
 */
/*
 * Copyright (c) 2023-2025 CESNET, zájmové sdružení právnických osob
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
/**
 * @file
 * @note
 * Remarks (possible improvements):
 * - lbl_random reads from '/dev/urandom' instead if available (so Linux only)
 * and HAVE_DEV_URANDOM defined (which isn't)
 * - init_rng() in rtp.c used also hostname for seeding (but there may not be
 * reasonable value, eg. '::1' if no host given to rcvr)
 */

#ifndef _WIN32
#include <stdio.h>
#include <unistd.h>
#endif
#define _CRT_RAND_S // request rand_s (Win)
#include <stdlib.h>

#include "random.h"
#include "tv.h"

/**
 * Initialize (seed) random engine. Called with common_preinit (host.cpp)>
 *
 * @note
 * Windows rand_s() is not supposed to be seeded
 * @remark
 * thread-safe
 */
void
ug_rand_init(void)
{
#ifndef _WIN32
        unsigned seed =
            (getpid() * 42) ^ (unsigned) get_time_in_ns() ^ (unsigned) clock();
        FILE *dev_r = fopen("/dev/random", "rb");
        if (dev_r != NULL) {
                unsigned seed_dev_r = 0;
                if (fread(&seed_dev_r, sizeof seed_dev_r, 1, dev_r) == 1) {
                        seed ^= seed_dev_r;
                }
                fclose(dev_r);
        }
        srandom(seed);
#endif
}

/**
 * @returns pseudorandom number in interval [0, 2^32-1]
 *
 * @remark
 * thread-safe
 */
uint32_t
ug_rand(void)
{
#ifdef _WIN32
        uint32_t ret = 0;
        rand_s(&ret);
#else
        uint32_t ret = (uint32_t) random() << 1;
        ret |= random() & 0x1;
#endif
        return ret;
}

/**
 * Behaves similarly to drand48()
 *
 * uses only 2^32 values from the interval (drand48 may use 48 bits)
 *
 * @returns val in interval [0.0, 1.0)
 *
 * @remark
 * thread-safe
 */
double
ug_drand()
{
        return ug_rand() / (UINT32_MAX + 1.0);
}
