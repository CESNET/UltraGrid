/*
 * FILE:     tv.h
 * AUTHOR:   Colin Perkins <csp@csperkins.org>
 * MODIFIED: Ladan Gharai <ladan@isi.edu>
 *
 * Copyright (c) 2001-2003 University of Southern California
 * Copyright (c) 2005-2022 CESNET z.s.p.o.
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

#ifndef TV_H_8332A958_38EB_4FE7_94E6_22C71BECD013
#define TV_H_8332A958_38EB_4FE7_94E6_22C71BECD013

#ifdef __cplusplus
#include <ctime>
#else
#include <time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

uint32_t get_local_mediatime(void);
double   tv_diff(struct timeval curr_time, struct timeval prev_time);
uint32_t tv_diff_usec(struct timeval curr_time, struct timeval prev_time);
void     tv_add(struct timeval *ts, double offset_secs);
void     tv_add_usec(struct timeval *ts, long long offset);
int      tv_gt(struct timeval a, struct timeval b);
uint32_t get_std_audio_local_mediatime(double samples, int rate);
uint32_t get_std_video_local_mediatime(void);

typedef long long time_ns_t;
#define NS_IN_SEC 1000000000LL
static inline time_ns_t get_time_in_ns() {
        struct timespec ts = { 0, 0 };
        timespec_get(&ts, TIME_UTC);
        return ts.tv_sec * NS_IN_SEC + ts.tv_nsec;
}

#ifdef __cplusplus
}
#endif

#endif // ! defined TV_H_8332A958_38EB_4FE7_94E6_22C71BECD013
