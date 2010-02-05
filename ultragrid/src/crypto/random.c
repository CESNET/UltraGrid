/*
 * Copyright (c) 1993 Regents of the University of California.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 *	This product includes software developed by the Computer Systems
 *	Engineering Group at Lawrence Berkeley Laboratory.
 * 4. Neither the name of the University nor of the Laboratory may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

/*
 * LBL random number generator.
 *
 * Written by Steve McCanne & Chris Torek (mccanne@ee.lbl.gov,
 * torek@ee.lbl.gov), November, 1992.
 *
 * This implementation is based on ``Two Fast Implementations of
 * the "Minimal Standard" Random Number Generator", David G. Carta,
 * Communications of the ACM, Jan 1990, Vol 33 No 1.
 */

#include "config.h"
#include "config_win32.h"
#include "config_unix.h"
#include "crypto/random.h"

static uint32_t randseed = 1;

void lbl_srandom(uint32_t seed)
{
        randseed = seed;
}

uint32_t lbl_random(void)
{
#ifdef HAVE_DEV_URANDOM
        int fd, res, l;

        fd = open("/dev/urandom", O_RDONLY);
        if (fd == -1) {
                perror("Cannot open random sequence generator");
                abort();
        }
        l = read(fd, &res, sizeof(res));
        if (l != sizeof(res)) {
                perror("Cannot read random data");
                abort();
        }
        close(fd);
        return res;
#else
        register uint32_t x = randseed;
        register uint32_t hi, lo, t;

        hi = x / 127773;
        lo = x % 127773;
        t = 16807 * lo - 2836 * hi;
        if (t <= 0)
                t += 0x7fffffff;
        randseed = t;
        return (t);
#endif
}
