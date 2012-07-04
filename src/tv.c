/*
 * FILE:     tv.c
 * AUTHOR:   Colin Perkins <csp@csperkins.org>
 * MODIFIED: Ladan Gharai  <ladan@isi.edu>
 *
 * Copyright (c) 2001-2003 University of Southern California
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
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
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
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 *
 */

#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#include "debug.h"
#include "crypto/random.h"
#include "tv.h"

uint32_t get_local_mediatime(void)
{
        static struct timeval start_time;
        static uint32_t random_offset;
        static int first = 0;

        struct timeval curr_time;

        if (first == 0) {
                gettimeofday(&start_time, NULL);
                random_offset = lbl_random();
                first = 1;
        }

        gettimeofday(&curr_time, NULL);
        return (tv_diff(curr_time, start_time) * 90000) + random_offset;
}

double tv_diff(struct timeval curr_time, struct timeval prev_time)
{
        /* Return (curr_time - prev_time) in seconds */
        double ct, pt;

        ct = (double)curr_time.tv_sec +
            (((double)curr_time.tv_usec) / 1000000.0);
        pt = (double)prev_time.tv_sec +
            (((double)prev_time.tv_usec) / 1000000.0);
        return (ct - pt);
}

uint32_t tv_diff_usec(struct timeval curr_time, struct timeval prev_time)
{
        /* Return curr_time - prev_time in usec - i wonder if these numbers will be too big? */
        uint32_t tmp, tmp1, tmp2;

        /* We return an unsigned, so fail is prev_time is later than curr_time */
        assert(curr_time.tv_sec >= prev_time.tv_sec);
        if (curr_time.tv_sec == prev_time.tv_sec) {
                assert(curr_time.tv_usec >= prev_time.tv_usec);
        }

        tmp1 = (curr_time.tv_sec - prev_time.tv_sec) * ((uint32_t) 1000000);
        tmp2 = curr_time.tv_usec - prev_time.tv_usec;
        tmp = tmp1 + tmp2;

        return tmp;
}

void tv_add(struct timeval *ts, double offset_secs)
{
        unsigned int offset = (unsigned long)(offset_secs * 1000000.0);

        ts->tv_usec += offset;
        while (ts->tv_usec >= 1000000) {
                ts->tv_sec++;
                ts->tv_usec -= 1000000;
        }
}

void tv_add_usec(struct timeval *ts, double offset)
{
        ts->tv_usec += offset;
        while (ts->tv_usec >= 1000000) {
                ts->tv_sec++;
                ts->tv_usec -= 1000000;
        }
}

int tv_gt(struct timeval a, struct timeval b)
{
        /* Returns (a>b) */
        if (a.tv_sec > b.tv_sec) {
                return TRUE;
        }
        if (a.tv_sec < b.tv_sec) {
                return FALSE;
        }
        assert(a.tv_sec == b.tv_sec);
        return a.tv_usec > b.tv_usec;
}
