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
        if (curr_time.tv_sec < prev_time.tv_sec ||
                        (curr_time.tv_sec == prev_time.tv_sec && curr_time.tv_usec < prev_time.tv_usec)) {
                log_msg(LOG_LEVEL_WARNING, "Discontinuity in time: %lf s!\n", tv_diff(curr_time, prev_time));
                return 0;
        }

        tmp1 = (curr_time.tv_sec - prev_time.tv_sec) * ((uint32_t) 1000000);
        tmp2 = curr_time.tv_usec - prev_time.tv_usec;
        tmp = tmp1 + tmp2;

        return tmp;
}

inline void tv_add_usec(struct timeval *ts, long long offset) ATTRIBUTE(always_inline); // to allow tv_add inline this function

void tv_add(struct timeval *ts, double offset_secs)
{
        tv_add_usec(ts, offset_secs * 1000000.0);
}

void tv_add_usec(struct timeval *ts, long long offset)
{
        long long new_usec = ts->tv_usec + offset;

        if (new_usec < MS_IN_SEC) {
                ts->tv_usec = new_usec;
                return;
        }
        if (new_usec < 2 * MS_IN_SEC) {
                ts->tv_sec++;
                ts->tv_usec = new_usec - MS_IN_SEC;
                return;
        }
        lldiv_t d = lldiv(new_usec, MS_IN_SEC);
        ts->tv_sec += d.quot;
        ts->tv_usec = d.rem;
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

/*
 * STANDARD TRANSPORT - RTP STANDARD
 * Calculate initial time on first execution, add per 'sample' time otherwise.
 */
typedef struct { //shared struct for audio and video streams (sync.)
	bool init;
	uint32_t random_startime_offset;
	struct timeval vtime;
	double vfps;
	struct timeval atime;
	struct timeval start_time;
} std_time_struct;

std_time_struct standard_time = { true, 0, { 0, 0 }, 25, { 0, 0 }, { 0, 0 } };

/**
 * @param samples       number of samples in unit of seconds
 * @param rate          RTP timestamp scale (usually sample rate, but for OPUS always 48000)
 */
uint32_t get_std_audio_local_mediatime(double samples, int rate)
{
        if (standard_time.init) {
			gettimeofday(&standard_time.start_time, NULL);
			standard_time.atime = standard_time.start_time;
			standard_time.vtime = standard_time.start_time;
			standard_time.random_startime_offset = lbl_random();
            tv_add_usec(&standard_time.vtime, standard_time.random_startime_offset);
            tv_add_usec(&standard_time.atime, standard_time.random_startime_offset);

        	standard_time.init = false;
        }
        else {
            tv_add(&standard_time.atime, samples);
        }

        return ((double)standard_time.atime.tv_sec + (((double)standard_time.atime.tv_usec) / 1000000.0)) * rate;
}

uint32_t get_std_video_local_mediatime(void)
{
	    double vrate = 90000; //default and standard video sample rate (Hz)
	    double nextFraction;
        unsigned nextSecsIncrement;
        static struct timeval t0;
        struct timeval tcurr;

        if (standard_time.init) {
			gettimeofday(&standard_time.start_time, NULL);
			gettimeofday(&t0, NULL);
			standard_time.atime = standard_time.start_time;
			standard_time.vtime = standard_time.start_time;
			standard_time.random_startime_offset = lbl_random();
            tv_add_usec(&standard_time.vtime, standard_time.random_startime_offset);
            tv_add_usec(&standard_time.atime, standard_time.random_startime_offset);

        	standard_time.init = false;
        }
        else {
			gettimeofday(&tcurr, NULL);
			nextFraction = ( standard_time.vtime.tv_usec / 1000000.0 ) + ( tv_diff(tcurr,t0));
			nextSecsIncrement = (long) nextFraction;
			standard_time.vtime.tv_sec += (long) nextSecsIncrement;
			standard_time.vtime.tv_usec = (long) ((nextFraction - nextSecsIncrement) * 1000000);
			t0 = tcurr;
        }

        return ((double)standard_time.vtime.tv_sec + (((double)standard_time.vtime.tv_usec) / 1000000.0)) * vrate;
}
